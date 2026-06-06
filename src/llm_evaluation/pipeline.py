"""Orquestração ponta a ponta da corrida de avaliação (recuperação, geração, verificação).

Recursos pesados (modelo de embeddings, cliente HTTP do LLM e do juiz) são instanciados
uma vez por corrida e reaproveitados em todos os itens.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

from llm_evaluation.config import AppConfig
from llm_evaluation.critic_schema import CRITIC_SCHEMA_VERSION
from llm_evaluation.datasets_rag import build_chunks_for_item
from llm_evaluation.generation import generate_answer
from llm_evaluation.lexical_metrics import attach_lexical_to_meta
from llm_evaluation.llm_client import (
    LlmClient,
    default_judge_from_env,
    default_llm_from_env,
    resolve_models_from_env,
)
from llm_evaluation.observability import TrackingLlmClient, UsageAccumulator
from llm_evaluation.pattern_detection import compute_diagnostico
from llm_evaluation.retrieval import Embedder, Retriever, make_embedder
from llm_evaluation.retrieval_hints import format_retrieval_hints
from llm_evaluation.retrieval_metrics import compute_retrieval_metrics
from llm_evaluation.types import EvalItem, RetrievedChunk, RunRecord, VerificationSignals
from llm_evaluation.veredito import veredito_e_negativo
from llm_evaluation.verification.aggregate import anomaly_from_signals
from llm_evaluation.verification.embedding_verify import max_cosine_answer_to_chunks
from llm_evaluation.verification.gold import gold_correct as gc
from llm_evaluation.verification.gold import gold_incorrect as gi
from llm_evaluation.verification.gold import is_refusal
from llm_evaluation.verification.judge import run_judge_for_retrieved

CriticHook = Callable[
    [EvalItem, list[RetrievedChunk], str, LlmClient],
    tuple[dict[str, Any], bool],
]

DEFAULT_WEAK_RETRIEVAL_MESSAGE_PT = (
    "Não consigo responder com confiança apenas com o contexto recuperado "
    "(relevância insuficiente)."
)

PROGRESS_EVERY = 25  # imprime progresso a cada N itens; útil em batches grandes
ITEM_RETRY_ATTEMPTS = 3
ITEM_RETRY_BACKOFF_SECONDS = (1.0, 3.0, 8.0)


def _item_retry_backoff_seconds(exc: Exception, attempt_index: int) -> float:
    """Backoff por item; 429 usa pausa maior (quota/rate limit)."""
    idx = attempt_index - 1
    if idx < len(ITEM_RETRY_BACKOFF_SECONDS):
        base = ITEM_RETRY_BACKOFF_SECONDS[idx]
    else:
        base = ITEM_RETRY_BACKOFF_SECONDS[-1]
    if "429" in str(exc).lower():
        floor = float(os.environ.get("LLM_EVAL_429_ITEM_BACKOFF", "30") or "30")
        return max(base, floor)
    return base


def _chunks_texts(retrieved: list[RetrievedChunk]) -> list[str]:
    return [c.text for c in retrieved]


def _retrieval_context_strong(
    retrieved: list[RetrievedChunk],
    *,
    min_score: float,
) -> bool:
    """True quando há chunk ouro no top-k com score do melhor chunk acima do limiar."""
    if not retrieved:
        return False
    top_score = retrieved[0].score
    has_gold = any(c.is_gold for c in retrieved)
    return has_gold and top_score >= min_score


def is_weak_retrieval(
    cfg: AppConfig,
    retrieved: list[RetrievedChunk],
) -> tuple[bool, float | None]:
    """Gate de qualidade pré-geração: relevância pergunta↔melhor chunk abaixo do limiar.

    Com ``rag.min_retrieval_score is None`` o gate está desligado (sempre "forte").
    Com RAG ligado mas sem chunks recuperados, considera-se fraco quando o gate está activo.
    """
    if not cfg.rag.enabled:
        return False, None
    thr = cfg.rag.min_retrieval_score
    if thr is None:
        return False, None
    if not retrieved:
        return True, None
    top = retrieved[0].score
    return top < thr, top


def weak_retrieval_answer(cfg: AppConfig) -> str:
    msg = (cfg.generation.weak_retrieval_message or "").strip()
    return msg if msg else DEFAULT_WEAK_RETRIEVAL_MESSAGE_PT


def generate_answer_for_item(
    cfg: AppConfig,
    llm: LlmClient,
    item: EvalItem,
    retrieved: list[RetrievedChunk],
) -> tuple[str, dict[str, Any]]:
    """Gera resposta ou devolve recusa curada quando o gate de recuperação dispara."""
    weak, top_score = is_weak_retrieval(cfg, retrieved)
    qualidade: dict[str, Any] = {"curada_por_recuperacao_fraca": False}
    if weak and cfg.generation.skip_llm_on_weak_retrieval:
        qualidade["curada_por_recuperacao_fraca"] = True
        qualidade["score_melhor_chunk"] = top_score
        qualidade["limiar_recuperacao"] = cfg.rag.min_retrieval_score
        return weak_retrieval_answer(cfg), qualidade
    answer, gen_meta = generate_answer(
        llm,
        item,
        retrieved,
        rag_enabled=cfg.rag.enabled,
        prompt_style=cfg.generation.prompt_style,
    )
    qualidade.update(gen_meta)
    if (
        cfg.generation.anti_refusal_repair
        and is_refusal(answer)
        and _retrieval_context_strong(
            retrieved,
            min_score=cfg.generation.anti_refusal_min_retrieval_score,
        )
    ):
        repaired = False
        attempts = max(1, cfg.generation.anti_refusal_max_attempts)
        for _ in range(attempts):
            alt, alt_meta = generate_answer(
                llm,
                item,
                retrieved,
                rag_enabled=cfg.rag.enabled,
                prompt_style=cfg.generation.prompt_style,
                force_specific_answer=True,
            )
            if not is_refusal(alt):
                answer = alt
                qualidade.update(alt_meta)
                repaired = True
                break
        qualidade["recusa_reparada_por_contexto"] = repaired
        qualidade["reparo_recusa_tentativas"] = attempts
    return answer, qualidade


def verify_item(
    *,
    cfg: AppConfig,
    item: EvalItem,
    answer: str,
    retrieved: list[RetrievedChunk],
    embedder: Embedder,
    judge_client: LlmClient,
    corpus_chunks: list[str] | None = None,
) -> tuple[VerificationSignals, dict[str, Any] | None]:
    if cfg.dataset.reference_type == "answer_lists" and cfg.verification.verify_gold:
        g_ok = gc(answer, item.correct_answers, item.incorrect_answers)
        g_bad = gi(answer, item.correct_answers, item.incorrect_answers)
    else:
        g_ok = None
        g_bad = None

    emb_max: float | None = None
    emb_max_ret: float | None = None
    emb_max_gold: float | None = None
    emb_low: bool | None = None
    had_corpus = bool(corpus_chunks)
    thr = cfg.verification.embedding_min_cosine

    if cfg.verification.verify_embedding and cfg.rag.enabled:
        scores: list[float] = []
        if retrieved:
            emb_max_ret = max_cosine_answer_to_chunks(
                answer,
                _chunks_texts(retrieved),
                embedder,
            )
            scores.append(emb_max_ret)
        gold_text = (item.rag_gold_chunk or "").strip()
        if gold_text and cfg.verification.embedding_use_gold_chunk:
            emb_max_gold = max_cosine_answer_to_chunks(answer, [gold_text], embedder)
            scores.append(emb_max_gold)
        if scores:
            emb_max = max(scores)
            emb_low = emb_max < thr
        elif had_corpus and not retrieved:
            emb_low = True
            emb_max = 0.0

    jr = None
    j_neg: bool | None = None
    judge_meta: dict[str, Any] | None = None
    should_call_judge = cfg.verification.verify_judge
    gate = cfg.verification.judge_gate_embedding_max_cosine
    gate_strong_ctx_ok = True
    if cfg.verification.judge_gate_requires_strong_context:
        gate_strong_ctx_ok = _retrieval_context_strong(
            retrieved,
            min_score=cfg.verification.judge_gate_min_retrieval_score,
        )
    if (
        should_call_judge
        and gate is not None
        and emb_max is not None
        and emb_max >= gate
        and not is_refusal(answer)
        and gate_strong_ctx_ok
    ):
        should_call_judge = False
        judge_meta = {
            "judge_skipped_by_gate": True,
            "judge_gate_embedding_max_cosine": gate,
            "embedding_max_coseno_item": emb_max,
        }
        if cfg.verification.judge_gate_requires_strong_context:
            judge_meta["judge_gate_requires_strong_context"] = True
            judge_meta["judge_gate_min_retrieval_score"] = (
                cfg.verification.judge_gate_min_retrieval_score
            )

    if should_call_judge:
        jr, j_run = run_judge_for_retrieved(
            question=item.question,
            answer=answer,
            retrieved=retrieved,
            client=judge_client,
            prompt_style=cfg.verification.judge_prompt_style,
            return_chain_of_thought=cfg.verification.judge_return_chain_of_thought,
            max_parse_retries=cfg.verification.judge_max_parse_retries,
            max_chunks=cfg.rag.top_k,
            max_context_chars=cfg.verification.judge_max_context_chars,
            retrieval_meta=format_retrieval_hints(retrieved),
        )
        from llm_evaluation.verification.judge import judge_run_meta_as_context

        judge_meta = judge_run_meta_as_context(j_run)
        if jr.raw.get("fallback_heuristico"):
            j_neg = None
        else:
            j_neg = veredito_e_negativo(
                jr.veredito,
                cfg.verification.judge_aggregation_verdicts,
            )
            if (
                jr.veredito == "incompleto"
                and cfg.verification.judge_incompleto_contexto_forte_negativo
                and _retrieval_context_strong(
                    retrieved,
                    min_score=cfg.verification.judge_incompleto_contexto_forte_min_score,
                )
            ):
                j_neg = True
                judge_meta["incompleto_contexto_forte_negativo"] = True
                judge_meta["incompleto_contexto_forte_min_score"] = (
                    cfg.verification.judge_incompleto_contexto_forte_min_score
                )
            if veredito_e_negativo(jr.veredito, cfg.verification.negative_judge_verdicts):
                judge_meta["veredito_diagnostico_negativo"] = True

    return (
        VerificationSignals(
            gold_correct=g_ok,
            gold_incorrect=g_bad,
            is_refusal=is_refusal(answer),
            embedding_max_cosine=emb_max,
            embedding_low_support=emb_low,
            embedding_max_cosine_retrieved=emb_max_ret,
            embedding_max_cosine_gold=emb_max_gold,
            judge=jr,
            judge_negative=j_neg,
        ),
        judge_meta,
    )


def _run_one_with_resources(
    cfg: AppConfig,
    item: EvalItem,
    *,
    baseline_profile: str,
    embedder: Embedder,
    llm: LlmClient,
    judge_client: LlmClient,
    usage_acc: UsageAccumulator | None = None,
    critic_hook: CriticHook | None = None,
) -> RunRecord:
    chunks = build_chunks_for_item(item, cfg.rag.chunk_max_chars) if cfg.rag.enabled else []
    retriever = Retriever(embedder, chunks)
    retrieved = (
        retriever.retrieve(
            item.question,
            cfg.rag.top_k,
            inject_remove_gold=cfg.rag.inject_retrieval_failure,
            item=item,
        )
        if cfg.rag.enabled
        else []
    )

    answer, qualidade_geracao = generate_answer_for_item(cfg, llm, item, retrieved)
    if critic_hook is not None:
        if qualidade_geracao.get("curada_por_recuperacao_fraca"):
            critic_raw: dict[str, Any] = {
                "cadeia_de_pensamento": [
                    "Geração curada por gate de recuperação; passo de crítica não aplicado.",
                ],
                "problemas": ["nenhum"],
                "nota": "sem_llm_respondedor",
                "schema_version": CRITIC_SCHEMA_VERSION,
            }
            critic_flag = False
        else:
            critic_raw, critic_flag = critic_hook(item, retrieved, answer, llm)
    signals, judge_meta = verify_item(
        cfg=cfg,
        item=item,
        answer=answer,
        retrieved=retrieved,
        embedder=embedder,
        judge_client=judge_client,
        corpus_chunks=chunks,
    )
    anomaly = anomaly_from_signals(
        signals,
        verify_gold=cfg.verification.verify_gold,
        verify_embedding=cfg.verification.verify_embedding,
        verify_judge=cfg.verification.verify_judge,
        negative_judge_verdicts=cfg.verification.negative_judge_verdicts,
        policy=cfg.aggregation.policy,
        judge_aggregation_verdicts=cfg.verification.judge_aggregation_verdicts,
    )
    meta: dict[str, Any] = {
        "orquestracao": cfg.orchestration,
        "qualidade_geracao": qualidade_geracao,
        "metricas_recuperacao": compute_retrieval_metrics(
            item,
            retrieved,
            rag_enabled=cfg.rag.enabled,
        ),
        "referencias": [str(a)[:500] for a in item.correct_answers[:20]],
    }
    if critic_hook is not None:
        meta["critica"] = critic_raw
        meta["flag_critica"] = critic_flag
    gold_pass = (item.rag_gold_chunk or "").strip()
    if gold_pass:
        if len(gold_pass) <= 12_000:
            meta["passagem_ouro_rag"] = gold_pass
        else:
            meta["passagem_ouro_rag"] = gold_pass[:12_000] + "…"
    if judge_meta is not None:
        meta["contexto_juiz"] = judge_meta
    attach_lexical_to_meta(meta, cfg, item, answer)
    if usage_acc is not None:
        meta["observabilidade"] = usage_acc.snapshot_for_item()
        usage_acc.reset()
    meta["diagnostico"] = compute_diagnostico(
        item=item,
        answer=answer,
        signals=signals,
        meta=meta,
        anomaly_flag=anomaly,
        pattern_overrides=cfg.patterns.overrides or None,
    )
    from llm_evaluation.explainability import build_explicacao

    rec_stub = RunRecord(
        item_id=item.id,
        question=item.question,
        answer=answer,
        gold_correct=signals.gold_correct,
        anomaly_flag=anomaly,
        signals=signals,
        retrieved=retrieved,
        baseline_profile=baseline_profile,
        meta=meta,
    )
    meta["explicacao"] = build_explicacao(rec_stub, cfg=cfg)

    return RunRecord(
        item_id=item.id,
        question=item.question,
        answer=answer,
        gold_correct=signals.gold_correct,
        anomaly_flag=anomaly,
        signals=signals,
        retrieved=retrieved,
        baseline_profile=baseline_profile,
        meta=meta,
    )


def _failed_record(
    item: EvalItem,
    *,
    baseline_profile: str,
    orchestration: str,
    err: Exception,
    attempt: int,
) -> RunRecord:
    meta: dict[str, Any] = {
        "orquestracao": orchestration,
        "processing_error": {
            "type": type(err).__name__,
            "message": str(err),
            "attempts": attempt,
        },
        "metricas_recuperacao": {"rag_ativo": False},
        "qualidade_geracao": {"erro_execucao_item": True},
        "referencias": [str(a)[:500] for a in item.correct_answers[:20]],
    }
    return RunRecord(
        item_id=item.id,
        question=item.question,
        answer="Falha ao processar item (erro interno).",
        gold_correct=None,
        anomaly_flag=True,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile=baseline_profile,
        meta=meta,
    )


def run_batch(
    cfg: AppConfig,
    items: list[EvalItem],
    *,
    on_record: Callable[[RunRecord], None] | None = None,
    critic_hook: CriticHook | None = None,
) -> list[RunRecord]:
    """Corre todos os itens reaproveitando embedder e clientes; imprime progresso.

    Se ``on_record`` for passado, é invocado a cada registo concluído — ideal
    para escrita incremental de ``predictions.jsonl`` (resiliência a falhas).
    """
    profile = cfg.baselines.profile
    embedder = make_embedder(cfg.embeddings.backend, cfg.embeddings.model_name)
    gen_model, judge_model = resolve_models_from_env()
    usage_acc = UsageAccumulator()
    llm = TrackingLlmClient(
        default_llm_from_env(
            timeout_seconds=cfg.llm.timeout_seconds,
            temperature=cfg.generation.temperature,
            max_tokens=cfg.generation.max_tokens,
        ),
        usage_acc,
        role="generation",
        model=gen_model,
    )
    judge_client = TrackingLlmClient(
        default_judge_from_env(timeout_seconds=cfg.llm.timeout_seconds),
        usage_acc,
        role="judge",
        model=judge_model,
    )

    out: list[RunRecord] = []
    n = len(items)
    t0 = time.time()
    attempts = max(1, int(os.environ.get("LLM_EVAL_ITEM_RETRIES", ITEM_RETRY_ATTEMPTS)))
    for i, item in enumerate(items, start=1):
        rec: RunRecord | None = None
        last_err: Exception | None = None
        for at in range(1, attempts + 1):
            try:
                rec = _run_one_with_resources(
                    cfg,
                    item,
                    baseline_profile=profile,
                    embedder=embedder,
                    llm=llm,
                    judge_client=judge_client,
                    usage_acc=usage_acc,
                    critic_hook=critic_hook,
                )
                break
            except Exception as e:  # noqa: BLE001 - robustez de produção por item
                last_err = e
                if at < attempts:
                    back = _item_retry_backoff_seconds(e, at)
                    print(
                        f"[{i}/{n}] erro no item {item.id} (tentativa {at}/{attempts}): {e}; "
                        f"retry em {back:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(back)
                    continue
                print(
                    f"[{i}/{n}] falha permanente no item {item.id}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
        if rec is None:
            assert last_err is not None
            rec = _failed_record(
                item,
                baseline_profile=profile,
                orchestration=cfg.orchestration,
                err=last_err,
                attempt=attempts,
            )
        out.append(rec)
        if on_record is not None:
            on_record(rec)
        if i == 1 or i == n or i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[{i}/{n}] processado (média {rate:.2f} itens/s, decorrido {elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
        if i < n:
            pause = float(os.environ.get("LLM_EVAL_INTER_ITEM_SLEEP", "0") or "0")
            if pause > 0:
                time.sleep(pause)
    return out

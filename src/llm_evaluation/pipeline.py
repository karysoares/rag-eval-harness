"""Orquestração ponta a ponta da corrida de avaliação (recuperação, geração, verificação).

Recursos pesados (modelo de embeddings, cliente HTTP do LLM e do juiz) são instanciados
uma vez por corrida e reaproveitados em todos os itens.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

from llm_evaluation.config import AppConfig
from llm_evaluation.critic_schema import CRITIC_SCHEMA_VERSION
from llm_evaluation.datasets_rag import build_corpus_for_item
from llm_evaluation.generation import generate_answer
from llm_evaluation.lexical_metrics import attach_lexical_to_meta
from llm_evaluation.llm_client import (
    LlmClient,
    PermanentApiError,
    default_judge_from_env,
    default_llm_from_env,
    endpoint_host,
    judge_base_url_from_env,
    openai_base_url_from_env,
    pool_size_for_concurrency,
    redact_secrets,
    resolve_models_from_env,
)
from llm_evaluation.observability import (
    LlmCallUsage,
    TrackingLlmClient,
    UsageAccumulator,
    summarize_run_observability,
)
from llm_evaluation.pattern_detection import compute_diagnostico
from llm_evaluation.retrieval import Embedder, Retriever, make_embedder
from llm_evaluation.retrieval_hints import format_retrieval_hints
from llm_evaluation.retrieval_metrics import compute_retrieval_metrics
from llm_evaluation.telemetry import TelemetryExporter, build_exporter
from llm_evaluation.telemetry.emit import (
    emit_item_event,
    emit_run_event,
    telemetry_includes_content,
)
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
        if retrieved:
            emb_max_ret = max_cosine_answer_to_chunks(
                answer,
                _chunks_texts(retrieved),
                embedder,
            )
        gold_text = (item.rag_gold_chunk or "").strip()
        if gold_text and cfg.verification.embedding_use_gold_chunk:
            # Diagnóstico, **não** entra no grounding: comparar a resposta com a
            # passagem ouro responde a «parece-se com a referência?», que é o plano
            # de referência. Misturar os dois num só escalar deixava um sistema que
            # recupera mal, mas responde perto da referência, passar por bem
            # ancorado — e induzia correlação entre o preditor e o rótulo na
            # confusão vs referência (CLAUDE.md, regra 8).
            emb_max_gold = max_cosine_answer_to_chunks(answer, [gold_text], embedder)
        if emb_max_ret is not None:
            emb_max = emb_max_ret
            emb_low = emb_max < thr
        elif had_corpus:
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
    exporter: TelemetryExporter | None = None,
) -> RunRecord:
    item_started_at = time.time()
    corpus = build_corpus_for_item(item, cfg.rag.chunk_max_chars) if cfg.rag.enabled else []
    chunks = [c.texto for c in corpus]
    tem_distratores = any(not c.e_ouro for c in corpus) if cfg.rag.enabled else None
    # A proveniência vem daqui e não de comparação de texto: ver Retriever._is_gold.
    retriever = Retriever(embedder, chunks, gold_flags=[c.e_ouro for c in corpus])
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
            n_chunks_corpus=len(corpus) if cfg.rag.enabled else None,
            corpus_tem_distratores=tem_distratores,
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
    chamadas_item: list[LlmCallUsage] = []
    if usage_acc is not None:
        meta["observabilidade"] = usage_acc.snapshot_for_item()
        chamadas_item = usage_acc.drain()
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

    registo = RunRecord(
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
    if exporter is not None:
        # As chamadas vão para o evento, não para ``meta``: os artefactos da
        # corrida têm de ser idênticos com e sem telemetria.
        emit_item_event(
            exporter,
            registo,
            calls=chamadas_item,
            started_at=item_started_at,
            include_content=telemetry_includes_content(),
        )
    return registo


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
            # Defesa em profundidade: a mensagem é serializada em predictions.jsonl,
            # que se publica. Qualquer excepção — não só as nossas — passa por aqui.
            "message": redact_secrets(str(err)),
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


def inter_item_pause_seconds() -> float:
    """Pausa de pacing entre itens (``LLM_EVAL_INTER_ITEM_SLEEP``); 0 = desligada."""
    try:
        return max(0.0, float(os.environ.get("LLM_EVAL_INTER_ITEM_SLEEP", "0") or "0"))
    except ValueError:
        return 0.0


class _Pacer:
    """Espaça o arranque dos itens em pelo menos ``pause`` segundos, entre threads.

    ``LLM_EVAL_INTER_ITEM_SLEEP`` existe para respeitar limites de taxa. Aplicá-lo
    só no caminho sequencial faria a pacing desaparecer ao subir a concorrência —
    exactamente quando é mais necessária. O relógio é partilhado por todos os
    workers, por isso a taxa agregada mantém-se em 1/``pause`` itens por segundo
    independentemente do número de workers.
    """

    def __init__(self, pause: float) -> None:
        self._pause = pause
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        if self._pause <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._pause
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def resolve_concurrency(cfg: AppConfig) -> int:
    """Workers de itens: ``LLM_EVAL_CONCURRENCY`` sobrepõe-se a ``llm.concurrency``."""
    raw = os.environ.get("LLM_EVAL_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, cfg.llm.concurrency)


def _run_item_with_retries(
    cfg: AppConfig,
    item: EvalItem,
    *,
    position: int,
    total: int,
    baseline_profile: str,
    embedder: Embedder,
    llm: LlmClient,
    judge_client: LlmClient,
    usage_acc: UsageAccumulator,
    critic_hook: CriticHook | None,
    attempts: int,
    exporter: TelemetryExporter | None = None,
) -> RunRecord:
    """Processa um item com retries; devolve sempre um registo (falha vira `_failed_record`)."""
    last_err: Exception | None = None
    for at in range(1, attempts + 1):
        try:
            return _run_one_with_resources(
                cfg,
                item,
                baseline_profile=baseline_profile,
                embedder=embedder,
                llm=llm,
                judge_client=judge_client,
                usage_acc=usage_acc,
                critic_hook=critic_hook,
                exporter=exporter,
            )
        except PermanentApiError as e:
            # Modelo inexistente, chave inválida, payload rejeitado: repetir só
            # atrasa a falha e esconde a causa. Falha já, com a mensagem do fornecedor.
            print(
                f"[{position}/{total}] erro de configuração no item {item.id}: {e}",
                file=sys.stderr,
                flush=True,
            )
            last_err = e
            break
        except Exception as e:  # noqa: BLE001 - robustez de produção por item
            last_err = e
            if at < attempts:
                back = _item_retry_backoff_seconds(e, at)
                print(
                    f"[{position}/{total}] erro no item {item.id} (tentativa {at}/{attempts}): "
                    f"{e}; retry em {back:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(back)
                continue
            print(
                f"[{position}/{total}] falha permanente no item {item.id}: {e}",
                file=sys.stderr,
                flush=True,
            )
    assert last_err is not None
    return _failed_record(
        item,
        baseline_profile=baseline_profile,
        orchestration=cfg.orchestration,
        err=last_err,
        attempt=attempts,
    )


def _print_progress(i: int, n: int, t0: float) -> None:
    if i == 1 or i == n or i % PROGRESS_EVERY == 0:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0.0
        print(
            f"[{i}/{n}] processado (média {rate:.2f} itens/s, decorrido {elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )


def _close_quietly(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def run_batch(
    cfg: AppConfig,
    items: list[EvalItem],
    *,
    on_record: Callable[[RunRecord], None] | None = None,
    critic_hook: CriticHook | None = None,
    run_dir: Path | None = None,
    config_name: str = "",
) -> list[RunRecord]:
    """Corre todos os itens reaproveitando embedder e clientes; imprime progresso.

    Se ``on_record`` for passado, é invocado a cada registo concluído — ideal
    para escrita incremental de ``predictions.jsonl`` (resiliência a falhas).

    Com ``llm.concurrency > 1`` (ou ``LLM_EVAL_CONCURRENCY``) os itens são
    processados por um pool de threads: o trabalho é dominado por latência de
    rede, não por CPU. Cada item é independente, e ``on_record`` continua a ser
    chamado **pela ordem do dataset**, numa única thread — a ordem e o conteúdo
    de ``predictions.jsonl`` não dependem da concorrência.

    Com ``LLM_EVAL_TELEMETRY`` definido, emite traces/métricas por item e por
    corrida para os destinos pedidos (Phoenix, LangSmith, CloudWatch, JSONL). A
    telemetria é um canal lateral: os artefactos gravados são idênticos com e sem
    ela, e um destino em baixo produz um aviso, não uma falha.
    """
    profile = cfg.baselines.profile
    embedder = make_embedder(cfg.embeddings.backend, cfg.embeddings.model_name, cache=True)
    gen_model, judge_model = resolve_models_from_env()
    usage_acc = UsageAccumulator()
    exporter = build_exporter(run_dir=run_dir)
    run_started_at = time.time()

    n = len(items)
    t0 = time.time()
    attempts = max(1, int(os.environ.get("LLM_EVAL_ITEM_RETRIES", ITEM_RETRY_ATTEMPTS)))
    workers = min(resolve_concurrency(cfg), n) if n else 1
    pool_size = pool_size_for_concurrency(workers)
    pacer = _Pacer(inter_item_pause_seconds())

    gen_inner = default_llm_from_env(
        timeout_seconds=cfg.llm.timeout_seconds,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
        max_connections=pool_size,
    )
    judge_inner = default_judge_from_env(
        timeout_seconds=cfg.llm.timeout_seconds,
        max_connections=pool_size,
    )
    llm = TrackingLlmClient(
        gen_inner,
        usage_acc,
        role="generation",
        model=gen_model,
        endpoint=endpoint_host(openai_base_url_from_env()),
    )
    judge_client = TrackingLlmClient(
        judge_inner,
        usage_acc,
        role="judge",
        model=judge_model,
        endpoint=endpoint_host(judge_base_url_from_env()),
    )

    def process(position: int, item: EvalItem) -> RunRecord:
        # No modo concorrente o pacer é o único ponto de espaçamento; no sequencial
        # a pausa é aplicada entre itens (mesma taxa agregada, ver _Pacer).
        if workers > 1:
            pacer.wait()
        return _run_item_with_retries(
            cfg,
            item,
            position=position,
            total=n,
            baseline_profile=profile,
            embedder=embedder,
            llm=llm,
            judge_client=judge_client,
            usage_acc=usage_acc,
            critic_hook=critic_hook,
            attempts=attempts,
            exporter=exporter,
        )

    try:
        if workers > 1:
            out = _run_batch_concurrent(items, process, on_record=on_record, workers=workers, t0=t0)
        else:
            out = _run_batch_sequential(items, process, on_record=on_record, t0=t0)
    finally:
        _close_quietly(gen_inner)
        _close_quietly(judge_inner)
    emit_run_event(
        exporter,
        run_id=run_dir.name if run_dir is not None else "sem_run_dir",
        started_at=run_started_at,
        records=out,
        config_name=config_name,
        totals=summarize_run_observability(out) or {},
    )
    _close_quietly(exporter)
    stats = getattr(embedder, "stats", None)
    if callable(stats):
        print(f"Embeddings: {stats()}", file=sys.stderr, flush=True)
    return out


def _run_batch_sequential(
    items: list[EvalItem],
    process: Callable[[int, EvalItem], RunRecord],
    *,
    on_record: Callable[[RunRecord], None] | None,
    t0: float,
) -> list[RunRecord]:
    out: list[RunRecord] = []
    n = len(items)
    pause = inter_item_pause_seconds()
    for i, item in enumerate(items, start=1):
        rec = process(i, item)
        out.append(rec)
        if on_record is not None:
            on_record(rec)
        _print_progress(i, n, t0)
        if i < n and pause > 0:
            time.sleep(pause)
    return out


def _run_batch_concurrent(
    items: list[EvalItem],
    process: Callable[[int, EvalItem], RunRecord],
    *,
    on_record: Callable[[RunRecord], None] | None,
    workers: int,
    t0: float,
) -> list[RunRecord]:
    """Processa itens em paralelo, entregando os registos pela ordem do dataset.

    Os futuros são consumidos por ordem de submissão (não de conclusão), de modo
    que ``on_record`` — que escreve em ``predictions.jsonl`` — é sempre chamado
    numa única thread e com a mesma ordem do modo sequencial.

    A submissão é limitada a uma janela deslizante em vez de enfileirar a corrida
    inteira: em ``Ctrl+C`` (ou qualquer excepção) só há um punhado de itens por
    cancelar, e ``shutdown(cancel_futures=True)`` termina a corrida em segundos
    em vez de esperar pelos milhares de itens já submetidos.
    """
    n = len(items)
    out: list[RunRecord] = []
    window = max(2 * workers, workers + 1)
    pending: deque[Future[RunRecord]] = deque()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-eval")
    interrupted = False
    try:
        cursor = 0
        for i in range(1, n + 1):
            while cursor < n and len(pending) < window:
                cursor += 1
                pending.append(pool.submit(process, cursor, items[cursor - 1]))
            rec = pending.popleft().result()
            out.append(rec)
            if on_record is not None:
                on_record(rec)
            _print_progress(i, n, t0)
    except BaseException:
        interrupted = True
        raise
    finally:
        # ``cancel_futures`` descarta o que ainda não arrancou; ``wait=False`` evita
        # bloquear o Ctrl+C atrás dos itens em voo (as linhas já escritas em
        # predictions.jsonl permitem retomar com --resume).
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return out

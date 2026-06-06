"""Validação e normalização do protocolo de corrida (config + itens carregados).

Evita combinações enganosas documentadas em ``docs/PREMISSAS.md`` (ex.: RAG sem corpus
com verificação de embedding activa).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from llm_evaluation.config import AppConfig
from llm_evaluation.datasets_rag import build_chunks_for_item
from llm_evaluation.llm_client import resolve_models_from_env
from llm_evaluation.types import EvalItem


@dataclass(frozen=True)
class ProtocolAdjustment:
    campo: str
    de: object
    para: object
    motivo: str


def _count_items_with_corpus(items: list[EvalItem], chunk_max_chars: int) -> int:
    return sum(1 for it in items if build_chunks_for_item(it, chunk_max_chars))


def apply_protocol_defaults(
    cfg: AppConfig,
    items: list[EvalItem],
) -> tuple[AppConfig, list[ProtocolAdjustment]]:
    """Ajusta config incoerente com o corpus real dos itens (com registo explícito)."""
    adjustments: list[ProtocolAdjustment] = []
    n_corpus = _count_items_with_corpus(items, cfg.rag.chunk_max_chars)
    ref = cfg.dataset.reference_type

    new_cfg = cfg

    if new_cfg.rag.enabled and n_corpus == 0:
        new_cfg = replace(new_cfg, rag=replace(new_cfg.rag, enabled=False))
        adjustments.append(
            ProtocolAdjustment(
                "rag.enabled",
                True,
                False,
                "nenhum item tem passagem/corpus para recuperação",
            ),
        )

    if new_cfg.verification.verify_embedding and (not new_cfg.rag.enabled or n_corpus == 0):
        new_cfg = replace(
            new_cfg,
            verification=replace(new_cfg.verification, verify_embedding=False),
        )
        adjustments.append(
            ProtocolAdjustment(
                "verification.verify_embedding",
                True,
                False,
                "grounding por embedding exige RAG com corpus não vazio",
            ),
        )

    if new_cfg.verification.verify_judge and n_corpus == 0 and ref in ("lexical", "none"):
        new_cfg = replace(
            new_cfg,
            verification=replace(new_cfg.verification, verify_judge=False),
        )
        adjustments.append(
            ProtocolAdjustment(
                "verification.verify_judge",
                True,
                False,
                "juiz de aderência ao contexto sem contexto recuperável",
            ),
        )

    return new_cfg, adjustments


def validate_protocol(cfg: AppConfig, items: list[EvalItem]) -> None:
    """Falha cedo se a config ainda for incoerente após normalização."""
    n_corpus = _count_items_with_corpus(items, cfg.rag.chunk_max_chars)
    problems: list[str] = []

    if cfg.rag.enabled and n_corpus == 0:
        problems.append(
            "rag.enabled=true mas nenhum item tem corpus (rag_gold_chunk/distractors). "
            "Use rag.enabled=false para QA aberta ou um dataset com contexto."
        )
    if cfg.verification.verify_embedding and (not cfg.rag.enabled or n_corpus == 0):
        problems.append(
            "verify_embedding=true sem corpus RAG: marcaria anomalias por falta de chunks, "
            "não por qualidade da resposta."
        )
    if (
        cfg.verification.verify_judge
        and n_corpus == 0
        and cfg.dataset.reference_type in ("lexical", "none")
    ):
        problems.append(
            "verify_judge=true sem contexto: o juiz avalia aderência a passagens vazias."
        )
    if cfg.dataset.reference_type == "lexical" and not cfg.lexical_metrics.enabled:
        problems.append(
            "reference_type=lexical com metricas_lexicas desligadas: "
            "não há KPI principal de referência textual."
        )

    if problems:
        msg = "Protocolo de avaliação inválido:\n- " + "\n- ".join(problems)
        raise ValueError(msg)


def judge_generator_same_model_warning() -> str | None:
    """Aviso quando juiz e gerador partilham o mesmo modelo (auto-referência)."""
    llm_model, judge_model = resolve_models_from_env()
    if llm_model == judge_model:
        return (
            f"JUDGE_MODEL igual a LLM_MODEL ({llm_model!r}): o juiz avalia respostas "
            "do mesmo modelo — defina JUDGE_MODEL distinto para avaliação válida."
        )
    return None


def collect_protocol_avisos(cfg: AppConfig) -> list[str]:
    """Avisos não bloqueantes sobre o protocolo (modelos, reprodutibilidade)."""
    avisos: list[str] = []
    same = judge_generator_same_model_warning()
    if same:
        avisos.append(same)
    if cfg.generation.temperature > 0:
        avisos.append(
            f"generation.temperature={cfg.generation.temperature}: replay/resume não é "
            "bit-a-bit; use temperature=0 para reprodutibilidade estrita."
        )
    return avisos


def build_protocolo_ativo(cfg: AppConfig) -> dict[str, object]:
    """Snapshot único do protocolo para summary, replay e validação strict."""
    from llm_evaluation.operational import protocol_operational_patch
    from llm_evaluation.pattern_registry import build_pattern_settings

    pattern_settings = build_pattern_settings(cfg.patterns.overrides or None)
    llm_model, judge_model = resolve_models_from_env()

    return {
        "verify_gold": cfg.verification.verify_gold,
        "verify_embedding": cfg.verification.verify_embedding,
        "verify_judge": cfg.verification.verify_judge,
        "aggregation_policy": cfg.aggregation.policy,
        "orchestration": cfg.orchestration,
        "embedding_use_gold_chunk": cfg.verification.embedding_use_gold_chunk,
        "embedding_min_cosine": cfg.verification.embedding_min_cosine,
        "judge_prompt_style": cfg.verification.judge_prompt_style,
        "judge_gate_embedding_max_cosine": cfg.verification.judge_gate_embedding_max_cosine,
        "judge_gate_requires_strong_context": (cfg.verification.judge_gate_requires_strong_context),
        "judge_gate_min_retrieval_score": (cfg.verification.judge_gate_min_retrieval_score),
        "judge_incompleto_contexto_forte_negativo": (
            cfg.verification.judge_incompleto_contexto_forte_negativo
        ),
        "judge_incompleto_contexto_forte_min_score": (
            cfg.verification.judge_incompleto_contexto_forte_min_score
        ),
        "negative_judge_verdicts": list(cfg.verification.negative_judge_verdicts),
        "judge_aggregation_verdicts": list(cfg.verification.judge_aggregation_verdicts),
        "rag": {
            "enabled": cfg.rag.enabled,
            "top_k": cfg.rag.top_k,
            "min_retrieval_score": cfg.rag.min_retrieval_score,
            "chunk_max_chars": cfg.rag.chunk_max_chars,
            "inject_retrieval_failure": cfg.rag.inject_retrieval_failure,
        },
        "generation": {
            "temperature": cfg.generation.temperature,
            "max_tokens": cfg.generation.max_tokens,
            "prompt_style": cfg.generation.prompt_style,
            "skip_llm_on_weak_retrieval": cfg.generation.skip_llm_on_weak_retrieval,
            "anti_refusal_repair": cfg.generation.anti_refusal_repair,
            "anti_refusal_min_retrieval_score": cfg.generation.anti_refusal_min_retrieval_score,
            "anti_refusal_max_attempts": cfg.generation.anti_refusal_max_attempts,
        },
        "models": {
            "llm_model": llm_model,
            "judge_model": judge_model,
            "judge_same_as_generator": llm_model == judge_model,
        },
        "pattern_settings": {
            "f1_forte_min": pattern_settings.f1_forte_min,
            "f1_fraca_min": pattern_settings.f1_fraca_min,
        },
        **protocol_operational_patch(cfg.operational),
    }

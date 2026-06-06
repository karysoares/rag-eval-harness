"""Agrega sinais de verificação (ADR 0002)."""

from __future__ import annotations

from llm_evaluation.config import AggregationPolicy
from llm_evaluation.types import VerificationSignals


def judge_negative_for_aggregation(
    s: VerificationSignals,
    judge_aggregation_verdicts: list[str],
) -> bool:
    """Gatilho juiz alinhado à agregação (veredito real; ignora fallback heurístico)."""
    from llm_evaluation.veredito import veredito_e_negativo

    if s.judge is not None and s.judge.raw.get("fallback_heuristico"):
        return False
    if s.judge is None:
        return s.judge_negative is True
    return veredito_e_negativo(s.judge.veredito, judge_aggregation_verdicts)


def judge_negative_for_diagnosis(
    s: VerificationSignals,
    negative_judge_verdicts: list[str],
) -> bool:
    """Vereditos negativos para padrões/diagnóstico (inclui avisos como incompleto)."""
    return judge_negative_for_aggregation(s, negative_judge_verdicts)


def _judge_negative_for_aggregation(
    s: VerificationSignals,
    negative_judge_verdicts: list[str],
) -> bool:
    return judge_negative_for_aggregation(s, negative_judge_verdicts)


def _layer_triggers(
    s: VerificationSignals,
    *,
    verify_gold: bool,
    verify_embedding: bool,
    verify_judge: bool,
    negative_judge_verdicts: list[str],
) -> tuple[bool, bool, bool]:
    g = bool(verify_gold and s.gold_incorrect is True)
    e = bool(verify_embedding and s.embedding_low_support is True)
    j = bool(verify_judge and _judge_negative_for_aggregation(s, negative_judge_verdicts))
    return g, e, j


def anomaly_from_signals(
    s: VerificationSignals,
    *,
    verify_gold: bool,
    verify_embedding: bool,
    verify_judge: bool,
    negative_judge_verdicts: list[str],
    policy: AggregationPolicy = "qualquer_critico",
    judge_aggregation_verdicts: list[str] | None = None,
) -> bool:
    agg_verdicts = (
        judge_aggregation_verdicts
        if judge_aggregation_verdicts is not None
        else negative_judge_verdicts
    )
    g, e, j = _layer_triggers(
        s,
        verify_gold=verify_gold,
        verify_embedding=verify_embedding,
        verify_judge=verify_judge,
        negative_judge_verdicts=agg_verdicts,
    )

    if policy == "todos_criticos":
        checks: list[bool] = []
        if verify_gold:
            checks.append(g)
        if verify_embedding:
            checks.append(e)
        if verify_judge:
            checks.append(j)
        return bool(checks) and all(checks)

    if policy == "embedding_e_juiz":
        if verify_embedding and verify_judge:
            return e and j
        if verify_embedding:
            return e
        if verify_judge:
            return j
        if verify_gold:
            return g
        return False

    # qualquer_critico (default)
    return g or e or j


def signals_to_dict(
    s: VerificationSignals,
    *,
    include_judge_cot: bool = False,
) -> dict[str, object]:
    jd: dict[str, object] | None = None
    if s.judge:
        jd = {
            "veredito": s.judge.veredito,
            "motivo_breve": s.judge.motivo_breve,
            "confianca": s.judge.confianca,
        }
        if include_judge_cot:
            cot = s.judge.raw.get("cadeia_de_pensamento")
            if cot:
                jd["cadeia_de_pensamento"] = cot
        if s.judge.raw.get("fallback_heuristico"):
            jd["fallback_heuristico"] = True
    out: dict[str, object] = {
        "gold_correto": s.gold_correct,
        "gold_incorreto": s.gold_incorrect,
        "e_recusa": s.is_refusal,
        "embedding_max_coseno": s.embedding_max_cosine,
        "embedding_baixo_suporte": s.embedding_low_support,
        "juiz": jd,
        "juiz_negativo": s.judge_negative,
    }
    if s.embedding_max_cosine_retrieved is not None:
        out["embedding_max_coseno_recuperados"] = s.embedding_max_cosine_retrieved
    if s.embedding_max_cosine_gold is not None:
        out["embedding_max_coseno_ouro"] = s.embedding_max_cosine_gold
    return out

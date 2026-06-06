"""Detecção determinística de padrões por item (SPEC-007)."""

from __future__ import annotations

from typing import Any, Literal

from llm_evaluation.pattern_registry import (
    PATTERN_CATALOG_VERSION,
    PatternSettings,
    build_pattern_settings,
    meta_for_active_patterns,
    pick_primary,
)
from llm_evaluation.types import EvalItem, VerificationSignals
from llm_evaluation.verification.gold import is_refusal

TierQualidade = Literal["alta", "media", "baixa", "indeterminada"]


def has_placeholder(text: str, settings: PatternSettings | None = None) -> bool:
    st = settings or build_pattern_settings()
    t = text.strip()
    if st.placeholder_re.search(t):
        return True
    lower = t.lower()
    return any(p in lower for p in st.placeholder_phrases)


def _f1_from_meta(meta: dict[str, Any]) -> float | None:
    lm = meta.get("metricas_lexicas") or meta.get("lexical_metrics")
    if not isinstance(lm, dict):
        return None
    v = lm.get("f1_token")
    return float(v) if v is not None else None


def _em_squad_from_meta(meta: dict[str, Any]) -> bool:
    lm = meta.get("metricas_lexicas") or meta.get("lexical_metrics")
    if not isinstance(lm, dict):
        return False
    return bool(lm.get("em_squad"))


def _retrieval_failed(meta: dict[str, Any]) -> bool:
    rm = meta.get("metricas_recuperacao") or meta.get("retrieval_metrics")
    if not isinstance(rm, dict) or not rm.get("rag_ativo"):
        return False
    if not rm.get("corpus_tem_chunk_ouro"):
        return False
    return not bool(rm.get("chunk_ouro_no_top_k"))


def compute_diagnostico(
    *,
    item: EvalItem,
    answer: str,
    signals: VerificationSignals,
    meta: dict[str, Any],
    anomaly_flag: bool,
    pattern_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Regras fixas; não altera ``anomaly_flag``."""
    settings = build_pattern_settings(pattern_overrides)
    padroes: list[str] = []
    f1 = _f1_from_meta(meta)
    has_refs = bool(item.correct_answers)

    if not answer.strip():
        padroes.append("resposta_vazia")
    if has_placeholder(answer, settings):
        padroes.append("placeholder")
    if is_refusal(answer):
        padroes.append("recusa")

    if _retrieval_failed(meta):
        padroes.append("recuperacao_falhou")

    if signals.embedding_low_support is True:
        padroes.append("grounding_baixo")
    if signals.gold_correct is True and signals.embedding_low_support is True:
        padroes.append("grounding_fp_suspeito")

    if has_refs and f1 is not None:
        if _em_squad_from_meta(meta) or f1 >= settings.f1_forte_min:
            padroes.append("referencia_forte")
        elif f1 >= settings.f1_fraca_min:
            padroes.append("referencia_fraca")
        else:
            padroes.append("referencia_ausente")
    elif has_refs and f1 is None and signals.gold_correct is False:
        padroes.append("referencia_ausente")

    if signals.judge is not None and signals.judge.raw.get("fallback_heuristico"):
        padroes.append("juiz_fallback")
    if signals.judge is not None and signals.judge.veredito == "incompleto":
        padroes.append("juiz_incompleto")
    elif signals.judge_negative is True:
        padroes.append("juiz_negativo")
    if anomaly_flag:
        padroes.append("anomalia")

    padrao_primario = pick_primary(padroes)
    tier = _tier_from_patterns(padroes, has_refs, f1)
    rationale = _build_rationale(
        padroes=padroes,
        f1=f1,
        settings=settings,
        signals=signals,
        meta=meta,
        anomaly_flag=anomaly_flag,
    )

    return {
        "catalog_version": PATTERN_CATALOG_VERSION,
        "padroes": padroes,
        "padrao_primario": padrao_primario,
        "tier_qualidade": tier,
        "padroes_meta": meta_for_active_patterns(padroes),
        "rationale": rationale,
    }


def _build_rationale(
    *,
    padroes: list[str],
    f1: float | None,
    settings: PatternSettings,
    signals: VerificationSignals,
    meta: dict[str, Any],
    anomaly_flag: bool,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for tag in padroes:
        entry: dict[str, object] = {"padrao": tag}
        if tag == "referencia_forte" and f1 is not None:
            entry.update({"campo": "f1_token", "valor": f1, "limiar": settings.f1_forte_min})
        elif tag in ("referencia_fraca", "referencia_ausente") and f1 is not None:
            entry.update({"campo": "f1_token", "valor": f1, "limiar": settings.f1_fraca_min})
        elif tag == "grounding_baixo":
            entry.update(
                {
                    "campo": "embedding_low_support",
                    "valor": signals.embedding_low_support,
                },
            )
        elif tag == "recuperacao_falhou":
            rm = meta.get("metricas_recuperacao") or {}
            val = rm.get("chunk_ouro_no_top_k") if isinstance(rm, dict) else None
            entry.update({"campo": "chunk_ouro_no_top_k", "valor": val})
        elif tag == "anomalia":
            entry.update({"campo": "flag_anomalia", "valor": anomaly_flag})
        elif tag == "juiz_negativo" and signals.judge is not None:
            entry.update({"campo": "veredito", "valor": signals.judge.veredito})
        out.append(entry)
    return out


def _tier_from_patterns(
    padroes: list[str],
    has_refs: bool,
    f1: float | None,
) -> TierQualidade:
    if "resposta_vazia" in padroes or "placeholder" in padroes:
        return "baixa"
    if "recuperacao_falhou" in padroes or "referencia_ausente" in padroes:
        return "baixa"
    if "recusa" in padroes and "referencia_forte" not in padroes:
        return "baixa"
    if "referencia_forte" in padroes and "grounding_baixo" not in padroes:
        return "alta"
    if "referencia_fraca" in padroes or "grounding_fp_suspeito" in padroes:
        return "media"
    if not has_refs or f1 is None:
        return "indeterminada"
    return "media"

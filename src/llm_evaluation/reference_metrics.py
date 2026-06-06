"""Rótulos de referência para confusão/kappa (sem import circular)."""

from __future__ import annotations

from llm_evaluation.pattern_registry import build_pattern_settings
from llm_evaluation.types import RunRecord


def referencia_incorreta(
    record: RunRecord,
    reference_type: str | None,
    *,
    f1_fraca_min: float | None = None,
) -> bool | None:
    """True = referência fraca/ausente; False = overlap léxico aceitável; None = sem rótulo."""
    if reference_type == "none":
        return None
    if reference_type == "lexical":
        lm = record.meta.get("metricas_lexicas") or record.meta.get("lexical_metrics")
        if isinstance(lm, dict):
            if lm.get("em_squad") is True:
                return False
            f1 = lm.get("f1_token")
            if f1 is not None:
                limiar = (
                    f1_fraca_min
                    if f1_fraca_min is not None
                    else build_pattern_settings().f1_fraca_min
                )
                return float(f1) < limiar
        return None
    if reference_type != "answer_lists":
        return None
    if record.gold_correct is None:
        return None
    return record.gold_correct is False


def referencia_humana_incorreta(record: RunRecord) -> bool | None:
    """True se humano marcou incorreto/parcial; False se correto/recusa_ok; None sem rótulo."""
    raw = record.meta.get("adjudicacao_humana")
    if not isinstance(raw, dict):
        return None
    rotulo = str(raw.get("rotulo") or "").lower()
    if not rotulo:
        return None
    if rotulo in ("correto", "recusa_ok"):
        return False
    if rotulo in ("incorreto", "parcial"):
        return True
    if rotulo == "inconclusivo":
        return None
    return None


def referencia_humana_aceitavel(record: RunRecord) -> bool:
    raw = record.meta.get("adjudicacao_humana")
    if not isinstance(raw, dict):
        return False
    return str(raw.get("rotulo") or "").lower() in ("correto", "recusa_ok")

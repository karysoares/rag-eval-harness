"""Limiares operacionais partilhados (fila humana, gap RAG–resposta)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalThresholds:
    fila_min_score_recuperacao: float = 0.5
    gap_min_score_recuperacao: float = 0.5
    gap_max_f1_token: float = 0.15


DEFAULT_THRESHOLDS = OperationalThresholds()


def thresholds_from_mapping(raw: dict[str, object] | None) -> OperationalThresholds:
    """Lê limiares de ``operacional`` no YAML ou chaves planas em ``protocolo_ativo``."""
    if not raw:
        return DEFAULT_THRESHOLDS
    nested = raw.get("operacional")
    src: dict[str, object] = nested if isinstance(nested, dict) else raw

    def _f(key: str, default: float) -> float:
        v = src.get(key)
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        if isinstance(v, int | float):
            return float(v)
        if isinstance(v, str):
            return float(v.strip())
        return default

    return OperationalThresholds(
        fila_min_score_recuperacao=_f("fila_min_score_recuperacao", 0.5),
        gap_min_score_recuperacao=_f(
            "gap_min_score_recuperacao",
            _f("fila_min_score_recuperacao", 0.5),
        ),
        gap_max_f1_token=_f("gap_max_f1_token", 0.15),
    )


def protocol_operational_patch(
    cfg_thresholds: OperationalThresholds,
) -> dict[str, float]:
    """Chaves planas em ``protocolo_ativo`` para replay offline."""
    return {
        "fila_min_score_recuperacao": cfg_thresholds.fila_min_score_recuperacao,
        "gap_min_score_recuperacao": cfg_thresholds.gap_min_score_recuperacao,
        "gap_max_f1_token": cfg_thresholds.gap_max_f1_token,
    }

"""Limiares operacionais partilhados."""

from __future__ import annotations

from llm_evaluation.operational import (
    DEFAULT_THRESHOLDS,
    protocol_operational_patch,
    thresholds_from_mapping,
)


def test_thresholds_from_protocol_flat_keys() -> None:
    thr = thresholds_from_mapping(
        {
            "fila_min_score_recuperacao": 0.6,
            "gap_max_f1_token": 0.2,
        },
    )
    assert thr.fila_min_score_recuperacao == 0.6
    assert thr.gap_max_f1_token == 0.2
    assert thr.gap_min_score_recuperacao == 0.6


def test_thresholds_nested_operacional() -> None:
    thr = thresholds_from_mapping(
        {"operacional": {"fila_min_score_recuperacao": 0.4, "gap_max_f1_token": 0.1}},
    )
    assert thr.fila_min_score_recuperacao == 0.4
    assert thr.gap_max_f1_token == 0.1


def test_protocol_operational_patch() -> None:
    patch = protocol_operational_patch(DEFAULT_THRESHOLDS)
    assert patch["fila_min_score_recuperacao"] == 0.5

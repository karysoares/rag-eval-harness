"""Significância em compare_metric_reports."""

from __future__ import annotations

from llm_evaluation.evaluation_metrics import compare_metric_reports


def test_compare_includes_significancia() -> None:
    a = {"n_itens": 100, "n_anomalias_marcadas": 10, "sumario_lexical": {}}
    b = {"n_itens": 100, "n_anomalias_marcadas": 30, "sumario_lexical": {}}
    out = compare_metric_reports([a, b], ["run_a", "run_b"])
    sig = out.get("significancia")
    assert isinstance(sig, list)
    assert len(sig) >= 1
    assert "significativo_95" in sig[0]

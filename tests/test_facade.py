"""Dashboard facade."""

from __future__ import annotations

from llm_evaluation.dashboard.facade import MetricMode, kpi_blocks_for_mode, provenance_from_report


def test_kpi_blocks_automatico() -> None:
    rep = {"sumario_lexical": {"media_meteor": 0.5}, "kpi_primario": "sumario_lexical"}
    blk = kpi_blocks_for_mode(rep, MetricMode.AUTOMATICO)
    assert blk["fonte"] == "automatico"


def test_provenance_from_report() -> None:
    rep = {"proveniencia": {"config_hash_sha256": "abc"}}
    assert provenance_from_report(rep)["config_hash_sha256"] == "abc"

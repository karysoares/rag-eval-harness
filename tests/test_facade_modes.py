"""Cobertura da facade do dashboard: modos de KPI, proveniência e fallback."""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.dashboard.facade import (
    MetricMode,
    hitl_csv_path,
    kpi_blocks_for_mode,
    load_report,
    provenance_from_report,
)

REPORT = {
    "sumario_lexical": {"f1": 0.7},
    "sumario_operacional": {"latencia_p95": 1.2},
    "sumario_hitl": {"kappa": 0.6, "n_itens_rotulados": 4},
    "kpi_primario": {"nome": "taxa_sustentado"},
    "proveniencia": {"config_hash_sha256": "abc", "git_commit": "deadbeef"},
}


class TestKpiBlocksForMode:
    def test_automatico_includes_primary_kpi(self) -> None:
        out = kpi_blocks_for_mode(REPORT, MetricMode.AUTOMATICO)
        assert out["fonte"] == "automatico"
        assert out["kpi_primario"] == {"nome": "taxa_sustentado"}
        assert out["lexical"] == {"f1": 0.7}

    def test_pos_hitl_uses_hitl_summary(self) -> None:
        out = kpi_blocks_for_mode(REPORT, MetricMode.POS_HITL)
        assert out["fonte"] == "sumario_hitl"
        assert out["dados"]["kappa"] == 0.6

    def test_pos_hitl_without_hitl_falls_back_to_automatico(self) -> None:
        report = {k: v for k, v in REPORT.items() if k != "sumario_hitl"}
        out = kpi_blocks_for_mode(report, MetricMode.POS_HITL)
        assert out["fonte"] == "automatico"

    def test_comparar_exposes_all_planes(self) -> None:
        out = kpi_blocks_for_mode(REPORT, MetricMode.COMPARAR)
        assert out["fonte"] == "comparar"
        assert set(out) == {"fonte", "lexical", "operacional", "hitl"}


class TestProvenance:
    def test_prefers_explicit_proveniencia(self) -> None:
        assert provenance_from_report(REPORT)["git_commit"] == "deadbeef"

    def test_falls_back_to_run_metadata(self) -> None:
        report = {
            "metadados_corrida": {
                "config_hash_sha256": "xyz",
                "modelos": {"judge": "gpt"},
                "git_commit": "cafe",
                "config_path": "configs/x.yaml",
            },
        }
        prov = provenance_from_report(report)
        assert prov["config_hash_sha256"] == "xyz"
        assert prov["modelos"] == {"judge": "gpt"}

    def test_empty_report_yields_empty_provenance(self) -> None:
        assert provenance_from_report({}) == {}


class TestLoadReport:
    def test_prefers_summary_json(self, tmp_path: Path) -> None:
        (tmp_path / "summary.json").write_text(json.dumps({"ok": True}))
        assert load_report(tmp_path) == {"ok": True}


def test_hitl_csv_path_is_inside_run_dir(tmp_path: Path) -> None:
    path = hitl_csv_path(tmp_path)
    assert tmp_path in path.parents or path.parent == tmp_path

"""Validação de schema no dashboard (SPEC-006 Fase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.dashboard.schema_validation import (
    detect_legacy_run,
    detect_schema_mismatch,
    validate_metrics_report,
    validate_run_schemas,
)
from llm_evaluation.schema_registry import (
    MANIFEST_SCHEMA_VERSION,
    PREDICTIONS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
)


def test_validate_metrics_report_legacy() -> None:
    issues = validate_metrics_report({"n_itens": 1, "kpi_primario": "x"})
    assert any("sem schema_version" in i for i in issues)


def test_detect_legacy_without_manifest() -> None:
    assert detect_legacy_run({"schema_version": "1.0"}, [], has_manifest=False) is True


def test_detect_schema_mismatch_versions() -> None:
    assert (
        detect_schema_mismatch(
            {"schema_version": "1.0"},
            {"schema_version": "2.0"},
            [{"schema_version": "1.0"}],
        )
        is True
    )


def test_detect_schema_mismatch_compares_versions_by_artifact_type() -> None:
    assert (
        detect_schema_mismatch(
            {"schema_version": SUMMARY_SCHEMA_VERSION},
            {"schema_version": MANIFEST_SCHEMA_VERSION},
            [{"schema_version": PREDICTIONS_SCHEMA_VERSION}],
        )
        is False
    )


def test_validate_run_schemas_modern_run(tmp_path: Path) -> None:
    run = tmp_path / "run_modern"
    run.mkdir()
    pred_line = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "id_item": "a",
        "pergunta": "q",
        "resposta": "r",
        "sinais": {},
        "meta": {},
    }
    (run / "predictions.jsonl").write_text(
        json.dumps(pred_line, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "n_itens": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "ficheiros": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = validate_run_schemas(run)
    assert out["legacy_run"] is False
    assert isinstance(out["warnings"], list)


def test_validate_run_schemas_reports_invalid_jsonl(tmp_path: Path) -> None:
    run = tmp_path / "run_invalid"
    run.mkdir()
    (run / "predictions.jsonl").write_text("{invalid json\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "n_itens": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = validate_run_schemas(run)
    assert any("JSON inválido" in w for w in out["warnings"])

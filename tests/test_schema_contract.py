"""Contrato golden JSONL ↔ summary."""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.reporting import summarize
from llm_evaluation.schema_registry import KNOWN_SUMMARY_TOP_FIELDS, validate_summary


def test_fixture_summary_contract() -> None:
    run = Path(__file__).resolve().parent / "fixtures/audit_runs/run_ci_fixture"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    issues = [m for m in validate_summary(summary, strict=True) if not m.startswith("aviso:")]
    assert issues == []
    unknown = set(summary.keys()) - KNOWN_SUMMARY_TOP_FIELDS
    assert not unknown, unknown


def test_predictions_to_summary_keys() -> None:
    run = Path(__file__).resolve().parent / "fixtures/audit_runs/run_ci_fixture"
    records = load_records_from_predictions_jsonl(run / "predictions.jsonl")
    s = summarize(records, reference_type="lexical")
    assert "n_itens" in s
    assert "schema_version" in s

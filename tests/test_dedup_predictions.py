"""Dedupe ao carregar predictions.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_full_report, load_records_from_predictions_jsonl


def test_dedup_keeps_last_line(tmp_path: Path) -> None:
    p = tmp_path / "predictions.jsonl"
    base = {
        "id_item": "x",
        "pergunta": "p",
        "resposta": "r1",
        "sinais": {},
        "meta": {},
    }
    b2 = {**base, "resposta": "r2"}
    p.write_text(
        json.dumps(base) + "\n" + json.dumps(b2) + "\n",
        encoding="utf-8",
    )
    recs = load_records_from_predictions_jsonl(p)
    assert len(recs) == 1
    assert recs[0].answer == "r2"


def test_load_full_report_prefers_summary(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/audit_runs/run_ci_fixture"
    rep = load_full_report(fixture)
    assert rep.get("n_itens") is not None

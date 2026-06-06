"""Merge CSV HITL → JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.hitl_io import merge_hitl_csv_into_predictions
from llm_evaluation.reporting import summarize


def test_merge_and_summarize_hitl(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "id_item": "x1",
                "pergunta": "p",
                "resposta": "r",
                "sinais": {},
                "meta": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    csv = tmp_path / "hitl.csv"
    csv.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\nx1,incorreto,rev,2026-01-01,\n",
        encoding="utf-8",
    )
    n = merge_hitl_csv_into_predictions(csv, pred)
    assert n == 1
    records = load_records_from_predictions_jsonl(pred)
    summary = summarize(records)
    assert summary.get("sumario_hitl") is not None


def test_merge_hitl_strict_ids_fails(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "id_item": "x1",
                "pergunta": "p",
                "resposta": "r",
                "sinais": {},
                "meta": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    csv = tmp_path / "hitl.csv"
    csv.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\nx2,incorreto,rev,2026-01-01,\n",
        encoding="utf-8",
    )
    try:
        merge_hitl_csv_into_predictions(csv, pred, strict_ids=True)
    except ValueError as e:
        assert "não existem" in str(e)
    else:
        raise AssertionError("Era esperado ValueError com strict_ids=True")

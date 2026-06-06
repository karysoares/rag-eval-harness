"""Retomada de corrida e IDs completados."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evaluation.reporting import record_to_json
from llm_evaluation.run_artifacts import (
    CorruptedPredictionsError,
    compact_predictions_jsonl,
    load_completed_item_ids,
)
from llm_evaluation.types import RunRecord, VerificationSignals


def _minimal_record(iid: str) -> RunRecord:
    return RunRecord(
        item_id=iid,
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )


def test_load_completed_item_ids(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(record_to_json(_minimal_record("a")), ensure_ascii=False)
        + "\n"
        + json.dumps(record_to_json(_minimal_record("b")), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert load_completed_item_ids(pred) == {"a", "b"}
    assert load_completed_item_ids(tmp_path / "missing.jsonl") == set()


def test_load_completed_blocks_truncated_last_line(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    truncated = '{"id_item":'
    pred.write_text(
        json.dumps(record_to_json(_minimal_record("ok")), ensure_ascii=False) + "\n" + truncated,
        encoding="utf-8",
    )
    with pytest.raises(CorruptedPredictionsError, match="linha 2"):
        load_completed_item_ids(pred)


def test_compact_predictions_dedupes_by_id(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    r1 = record_to_json(_minimal_record("a"))
    r2 = {**record_to_json(_minimal_record("a")), "resposta": "nova"}
    pred.write_text(
        json.dumps(r1, ensure_ascii=False) + "\n" + json.dumps(r2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    removed = compact_predictions_jsonl(pred)
    assert removed == 1
    lines = [ln for ln in pred.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["resposta"] == "nova"


def test_load_completed_skips_processing_error(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    ok = record_to_json(_minimal_record("ok"))
    bad = record_to_json(_minimal_record("bad"))
    bad["meta"] = {"processing_error": {"type": "HTTPStatusError", "message": "429"}}
    pred.write_text(
        json.dumps(ok, ensure_ascii=False) + "\n" + json.dumps(bad, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert load_completed_item_ids(pred) == {"ok"}


def test_hf_no_shuffle_preserves_order() -> None:
    from unittest.mock import MagicMock, patch

    from llm_evaluation.adapters.hf_generic import load_hf_qa_generic

    rows = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]
    fake_ds = MagicMock()
    fake_ds.to_list.return_value = list(rows)

    with patch("llm_evaluation.adapters.hf_generic.load_dataset", return_value=fake_ds):
        items = load_hf_qa_generic(
            "org/ds",
            hf_subset=None,
            split="validation",
            limit=0,
            seed=99,
            question_column="question",
            answer_column="answer",
            context_column=None,
            incorrect_column=None,
            id_column=None,
            shuffle=False,
        )
    assert [it.question for it in items] == ["q1", "q2"]

"""Adaptador HF genérico com ``load_dataset`` mockado."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_evaluation.adapters import hf_generic as mod


def _fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.to_list.return_value = rows
    return ds


def test_load_hf_qa_generic_maps_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "q": "Who?",
            "a": ["Alice", "Bob"],
            "wrong": ["Nobody"],
            "ctx": "Alice went.",
            "rid": 99,
        },
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda *_a, **_k: _fake_dataset(rows))
    items = mod.load_hf_qa_generic(
        "org/dataset",
        hf_subset=None,
        split="train",
        limit=10,
        seed=0,
        question_column="q",
        answer_column="a",
        context_column="ctx",
        incorrect_column="wrong",
        id_column="rid",
    )
    assert len(items) == 1
    assert items[0].id == "99"
    assert items[0].correct_answers == ["Alice", "Bob"]
    assert items[0].incorrect_answers == ["Nobody"]
    assert items[0].rag_gold_chunk == "Alice went."


def test_load_hf_qa_generic_skips_incomplete_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"q": "", "a": "x"}, {"q": "ok", "a": ""}]
    monkeypatch.setattr(mod, "load_dataset", lambda *_a, **_k: _fake_dataset(rows))
    items = mod.load_hf_qa_generic(
        "org/dataset",
        hf_subset="default",
        split="train",
        limit=0,
        seed=1,
        question_column="q",
        answer_column="a",
        context_column=None,
        incorrect_column=None,
        id_column=None,
    )
    assert items == []

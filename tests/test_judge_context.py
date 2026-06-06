"""Contexto do juiz: truncagem e metadados."""

from __future__ import annotations

from llm_evaluation.types import RetrievedChunk
from llm_evaluation.verification.judge_context import build_judge_context


def test_build_judge_context_respects_max_chunks() -> None:
    chunks = [RetrievedChunk(text=f"chunk {i}", score=1.0 - i * 0.1) for i in range(5)]
    built = build_judge_context(chunks, max_chunks=2, max_chars=None)
    assert built.n_chunks_usados == 2
    assert built.n_chunks_total == 5
    assert built.truncado is True
    assert "[1]" in built.text and "[2]" in built.text


def test_build_judge_context_truncates_chars() -> None:
    chunks = [RetrievedChunk(text="x" * 200, score=1.0)]
    built = build_judge_context(chunks, max_chunks=4, max_chars=50)
    assert built.truncado is True
    assert len(built.text) <= 55

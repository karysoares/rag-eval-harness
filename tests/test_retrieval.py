"""Testes unitários de recuperação densa (embedder hash + Retriever)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_evaluation.retrieval import (
    HashEmbedder,
    Retriever,
    cosine_topk,
    make_embedder,
)
from llm_evaluation.types import EvalItem


def test_hash_embedder_is_deterministic() -> None:
    emb = HashEmbedder(dim=32)
    a = emb.embed(["hello"])
    b = emb.embed(["hello"])
    np.testing.assert_allclose(a, b)


def test_cosine_topk_orders_by_similarity() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    pairs = cosine_topk(q, docs, k=2)
    assert pairs[0][0] == 0
    assert pairs[0][1] >= pairs[1][1]


def test_retriever_marks_gold_chunk() -> None:
    gold = "Capital federal Brasília."
    item = EvalItem(
        id="t1",
        question="capital do Brasil",
        correct_answers=["Brasília"],
        incorrect_answers=[],
        rag_gold_chunk=gold,
    )
    chunks = [gold, "São Paulo é a maior cidade."]
    retriever = Retriever(HashEmbedder(dim=16), chunks)
    out = retriever.retrieve("capital", top_k=2, inject_remove_gold=False, item=item)
    assert any(c.is_gold for c in out)
    assert len(out) == 2


def test_retriever_inject_remove_gold_excludes_gold() -> None:
    gold = "Trecho ouro único."
    item = EvalItem(
        id="t2",
        question="pergunta",
        correct_answers=["x"],
        incorrect_answers=[],
        rag_gold_chunk=gold,
        rag_distractors=["distrator A", "distrator B", "distrator C"],
    )
    chunks = [gold, "distrator A", "distrator B", "distrator C"]
    retriever = Retriever(HashEmbedder(dim=16), chunks)
    out = retriever.retrieve("pergunta", top_k=2, inject_remove_gold=True, item=item)
    assert all(not c.is_gold for c in out)
    assert len(out) == 2


def test_retriever_empty_chunks() -> None:
    item = EvalItem(id="e", question="q", correct_answers=[], incorrect_answers=[])
    retriever = Retriever(HashEmbedder(), [])
    assert retriever.retrieve("q", 3, inject_remove_gold=False, item=item) == []


def test_make_embedder_hash() -> None:
    emb = make_embedder("hash", "ignored")
    assert isinstance(emb, HashEmbedder)


def test_make_embedder_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embeddings backend"):
        make_embedder("unknown_backend", "model")

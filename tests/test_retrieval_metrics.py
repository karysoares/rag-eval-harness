"""Métricas de recuperação por item."""

from __future__ import annotations

from llm_evaluation.retrieval_metrics import compute_retrieval_metrics
from llm_evaluation.types import EvalItem, RetrievedChunk


def test_retrieval_metrics_gold_rank() -> None:
    item = EvalItem(
        id="t1",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
        rag_gold_chunk="gold text here",
    )
    retrieved = [
        RetrievedChunk(text="distractor", score=0.9, is_gold=False),
        RetrievedChunk(text="gold text here", score=0.7, is_gold=True),
    ]
    m = compute_retrieval_metrics(item, retrieved, rag_enabled=True)
    assert m["rank_chunk_ouro"] == 2
    assert m["chunk_ouro_no_top_k"] is True
    assert m["score_melhor_chunk"] == 0.9


def test_retrieval_metrics_rag_disabled() -> None:
    item = EvalItem(id="t2", question="q", correct_answers=[], incorrect_answers=[])
    m = compute_retrieval_metrics(item, [], rag_enabled=False)
    assert m["rag_ativo"] is False


def test_baseline_so_embeddings_disables_gold_in_aggregation() -> None:
    from dataclasses import replace
    from pathlib import Path

    from llm_evaluation.config import apply_baseline_profile, load_config

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = replace(cfg, verification=replace(cfg.verification, verify_gold=True))
    emb = apply_baseline_profile(cfg, "so_embeddings")
    assert emb.verification.verify_gold is False
    assert emb.verification.verify_embedding is True
    assert emb.verification.verify_judge is False

"""Embedding sem corpus não deve marcar baixo suporte."""

from __future__ import annotations

from dataclasses import replace

from llm_evaluation.config import load_config
from llm_evaluation.pipeline import verify_item
from llm_evaluation.retrieval import make_embedder
from llm_evaluation.types import EvalItem


class _NoopLlm:
    def complete(self, system: str, user: str) -> str:
        return "{}"


def test_embedding_na_when_no_corpus() -> None:
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_amostra.yaml")
    cfg = replace(cfg, verification=replace(cfg.verification, verify_embedding=True))
    item = EvalItem(
        id="t",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
        category="t",
        rag_gold_chunk=None,
        rag_distractors=[],
    )
    embedder = make_embedder("hash", "x")
    sig, _jmeta = verify_item(
        cfg=cfg,
        item=item,
        answer="answer",
        retrieved=[],
        embedder=embedder,
        judge_client=_NoopLlm(),
        corpus_chunks=[],
    )
    assert sig.embedding_low_support is None
    assert sig.embedding_max_cosine is None

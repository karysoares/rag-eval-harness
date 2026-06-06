"""Métricas de recuperação por item (pré-geração). Ver `docs/ARCHITECTURE.md`."""

from __future__ import annotations

from typing import Any

from llm_evaluation.types import EvalItem, RetrievedChunk


def compute_retrieval_metrics(
    item: EvalItem,
    retrieved: list[RetrievedChunk],
    *,
    rag_enabled: bool,
) -> dict[str, Any]:
    """Métricas diagnósticas do retriever; não entram na agregação de anomalia por defeito."""
    if not rag_enabled:
        return {"rag_ativo": False}

    top_score: float | None = retrieved[0].score if retrieved else None
    gold_rank: int | None = None
    for i, chunk in enumerate(retrieved, start=1):
        if chunk.is_gold:
            gold_rank = i
            break

    has_gold_corpus = bool((item.rag_gold_chunk or "").strip())

    return {
        "rag_ativo": True,
        "n_chunks_recuperados": len(retrieved),
        "score_melhor_chunk": top_score,
        "rank_chunk_ouro": gold_rank,
        "chunk_ouro_no_top_k": gold_rank is not None,
        "corpus_tem_chunk_ouro": has_gold_corpus,
    }

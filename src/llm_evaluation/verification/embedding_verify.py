"""Verificação de grounding por embeddings.

Ver `docs/techniques/embedding-similarity-vs-semantic-equivalence.md`.
"""

from __future__ import annotations

import re

import numpy as np

from llm_evaluation.retrieval import Embedder


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def max_cosine_answer_to_chunks(answer: str, chunks: list[str], embedder: Embedder) -> float:
    if not chunks:
        return 0.0
    sents = split_sentences(answer) or [answer]
    s_vecs = embedder.embed(sents)
    c_vecs = embedder.embed(chunks)
    # max over sentences of max cosine to any chunk
    sims = s_vecs @ c_vecs.T
    return float(np.max(sims)) if sims.size else 0.0


def embedding_low_support(
    answer: str,
    chunks: list[str],
    embedder: Embedder,
    min_cosine: float,
) -> bool:
    if not chunks:
        return False
    return max_cosine_answer_to_chunks(answer, chunks, embedder) < min_cosine

"""Construção de corpus RAG e injeção de falha de recuperação.

Ver `docs/techniques/dense-retrieval-and-chunking.md`.
"""

from __future__ import annotations

from llm_evaluation.types import EvalItem


def chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def build_chunks_for_item(item: EvalItem, chunk_max_chars: int) -> list[str]:
    parts: list[str] = []
    if item.rag_gold_chunk:
        parts.extend(chunk_text(item.rag_gold_chunk, chunk_max_chars))
    for d in item.rag_distractors:
        parts.extend(chunk_text(d, chunk_max_chars))
    # De-duplicate while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

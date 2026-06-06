"""Metadados curtos de recuperação para prompts (geração e juiz)."""

from __future__ import annotations

from llm_evaluation.types import RetrievedChunk


def format_retrieval_hints(retrieved: list[RetrievedChunk]) -> str:
    """Texto curto para calibrar recusa vs extração factual."""
    if not retrieved:
        return "Sem chunks recuperados."
    top = retrieved[0] if retrieved else None
    if top is None:
        return "Sem chunks recuperados."
    gold_in_top = any(c.is_gold for c in retrieved)
    lines = [
        (
            f"chunks={len(retrieved)}; score_top1={top.score:.2f}; "
            f"ouro_no_top_k={'sim' if gold_in_top else 'não'}."
        ),
    ]
    if gold_in_top and top.score >= 0.4:
        lines.append(
            "A passagem relevante parece estar presente — "
            "extraia o facto da pergunta a partir de [1]…[k].",
        )
    return " ".join(lines)

"""Montagem e truncagem do contexto enviado ao juiz (SPEC-003, Fase 1)."""

from __future__ import annotations

from dataclasses import dataclass

from llm_evaluation.types import RetrievedChunk


@dataclass(frozen=True)
class JudgeContextBuilt:
    """Contexto efectivo para o prompt do juiz."""

    text: str
    chunk_ids: list[int]
    n_chunks_usados: int
    n_chunks_total: int
    tokens_estimados: int
    truncado: bool


def _estimate_tokens(text: str) -> int:
    """Heurística barata (~4 chars/token em PT/EN misto)."""
    n = len(text)
    return max(1, (n + 3) // 4) if n else 0


def build_judge_context(
    retrieved: list[RetrievedChunk],
    *,
    max_chunks: int,
    max_chars: int | None = None,
) -> JudgeContextBuilt:
    """Ordena por rank de retrieval, limita chunks e caracteres totais."""
    n_total = len(retrieved)
    ordered = retrieved[: max(0, max_chunks)]
    parts: list[str] = []
    ids: list[int] = []
    total_chars = 0
    truncado = False

    for i, ch in enumerate(ordered):
        body = ch.text.strip()
        if not body:
            continue
        block = f"[{i + 1}] {body}"
        if max_chars is not None and total_chars + len(block) + 2 > max_chars:
            remaining = max_chars - total_chars - 2
            if remaining > 20:
                block = block[:remaining] + "…"
                parts.append(block)
                ids.append(i + 1)
            truncado = True
            break
        parts.append(block)
        ids.append(i + 1)
        total_chars += len(block) + 2

    text = "\n\n".join(parts) if parts else "(vazio)"
    if n_total > len(ids):
        truncado = True
    return JudgeContextBuilt(
        text=text,
        chunk_ids=ids,
        n_chunks_usados=len(ids),
        n_chunks_total=n_total,
        tokens_estimados=_estimate_tokens(text),
        truncado=truncado,
    )

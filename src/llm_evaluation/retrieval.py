"""Recuperação densa com embedders plugáveis.

Chunks por item e ranking coseno pergunta↔passagem.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from llm_evaluation.types import EvalItem, RetrievedChunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _stable_seed(text: str) -> int:
    """Hash determinístico entre processos (não depende de PYTHONHASHSEED)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1)


@dataclass
class HashEmbedder:
    """Embedder determinístico baseado em hash estável (uso: testes/CI sem rede).

    Não tem semântica linguística — serve só para exercitar o caminho do retriever
    com vetores reprodutíveis entre processos.
    """

    dim: int = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(_stable_seed(t))
            v = rng.standard_normal(self.dim)
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        return np.stack(vecs, axis=0)


class SentenceTransformerEmbedder:
    """Semantic embeddings (``sentence-transformers``; core project dependency)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = doc_vecs @ query_vec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx_sorted = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[int(i)])) for i in idx_sorted]


class Retriever:
    def __init__(self, embedder: Embedder, chunks: list[str]) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._vecs = embedder.embed(chunks) if chunks else np.zeros((0, 128), dtype=np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        inject_remove_gold: bool,
        item: EvalItem,
    ) -> list[RetrievedChunk]:
        if not self._chunks:
            return []

        qv = self._embedder.embed([query])[0]
        pairs = cosine_topk(qv, self._vecs, min(top_k, len(self._chunks)))
        out: list[RetrievedChunk] = []
        for i, score in pairs:
            text = self._chunks[i]
            gchunk = item.rag_gold_chunk
            gs = gchunk.strip() if gchunk else ""
            is_gold = bool(gs) and (gs in text or text in gs)
            out.append(RetrievedChunk(text=text, score=score, is_gold=is_gold))

        if inject_remove_gold and item.rag_gold_chunk:
            out = [c for c in out if not c.is_gold]
            # pad with next best if needed
            if len(out) < top_k:
                qv = self._embedder.embed([query])[0]
                all_pairs = cosine_topk(qv, self._vecs, min(len(self._chunks), top_k + 5))
                for j, score in all_pairs:
                    text = self._chunks[j]
                    gchunk = item.rag_gold_chunk
                    gs = gchunk.strip() if gchunk else ""
                    is_gold = bool(gs) and (gs in text or text in gs)
                    if is_gold:
                        continue
                    if any(x.text == text for x in out):
                        continue
                    out.append(RetrievedChunk(text=text, score=score, is_gold=False))
                    if len(out) >= top_k:
                        break
        return out[:top_k]


def make_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    msg = f"Unknown embeddings backend: {backend}"
    raise ValueError(msg)

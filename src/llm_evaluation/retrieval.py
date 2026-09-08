"""Recuperação densa com embedders plugáveis.

Chunks por item e ranking coseno pergunta↔passagem.
"""

from __future__ import annotations

import hashlib
import threading
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


class CachingEmbedder:
    """Memoriza embeddings por texto, partilhando-os entre itens e entre camadas.

    Duas fontes de trabalho redundante desaparecem com esta cache:

    1. Corpora com várias perguntas por documento (FairytaleQA: ~10 perguntas por
       história) re-embebem os mesmos chunks a cada item.
    2. ``verify_item`` re-embebe os chunks recuperados que a recuperação já tinha
       embebido no mesmo item.

    ``inner.embed`` é serializado por lock: os backends de ``sentence-transformers``
    não garantem reentrância e, num pipeline dominado por latência de API, a
    embebição não é o gargalo — a correção vale mais que o paralelismo aqui.
    """

    def __init__(self, inner: Embedder, *, max_entries: int = 50_000) -> None:
        self._inner = inner
        self._max_entries = max_entries
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self._inner.embed(texts)
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
            if missing:
                vecs = self._inner.embed(missing)
                if len(self._cache) + len(missing) > self._max_entries:
                    self._cache.clear()
                for t, v in zip(missing, vecs, strict=True):
                    self._cache[t] = v
            self.misses += len(missing)
            self.hits += len(texts) - len(missing)
            return np.stack([self._cache[t] for t in texts], axis=0)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "embeddings_cache_hits": self.hits,
            "embeddings_cache_misses": self.misses,
            "embeddings_cache_taxa_acerto": round(self.hits / total, 4) if total else 0.0,
        }


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[tuple[int, float]]:
    sims = doc_vecs @ query_vec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx_sorted = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[int(i)])) for i in idx_sorted]


class Retriever:
    """Recuperação densa sobre o corpus de um item.

    ``gold_flags`` traz a **proveniência** de cada chunk, vinda de
    ``datasets_rag.build_corpus_for_item``. Sem ela, cai-se na marcação por
    substring, que erra em dois casos reais: passagem ouro curta contida num
    distractor (marca o distractor como ouro) e corpus sem distratores (marca
    tudo como ouro, tornando ``rank_chunk_ouro`` constante). O modo substring
    fica só para chamadores antigos que não têm a proveniência à mão.
    """

    def __init__(
        self,
        embedder: Embedder,
        chunks: list[str],
        *,
        gold_flags: list[bool] | None = None,
    ) -> None:
        if gold_flags is not None and len(gold_flags) != len(chunks):
            msg = f"gold_flags ({len(gold_flags)}) tem de acompanhar chunks ({len(chunks)})"
            raise ValueError(msg)
        self._embedder = embedder
        self._chunks = chunks
        self._gold_flags = gold_flags
        self._vecs = embedder.embed(chunks) if chunks else np.zeros((0, 128), dtype=np.float32)

    def _is_gold(self, index: int, item: EvalItem) -> bool:
        if self._gold_flags is not None:
            return self._gold_flags[index]
        gchunk = item.rag_gold_chunk
        gs = gchunk.strip() if gchunk else ""
        text = self._chunks[index]
        return bool(gs) and (gs in text or text in gs)

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
            out.append(
                RetrievedChunk(text=self._chunks[i], score=score, is_gold=self._is_gold(i, item)),
            )

        if inject_remove_gold and item.rag_gold_chunk:
            out = [c for c in out if not c.is_gold]
            # pad with next best if needed
            if len(out) < top_k:
                all_pairs = cosine_topk(qv, self._vecs, min(len(self._chunks), top_k + 5))
                for j, score in all_pairs:
                    if self._is_gold(j, item):
                        continue
                    text = self._chunks[j]
                    if any(x.text == text for x in out):
                        continue
                    out.append(RetrievedChunk(text=text, score=score, is_gold=False))
                    if len(out) >= top_k:
                        break
        return out[:top_k]


def make_embedder(backend: str, model_name: str, *, cache: bool = False) -> Embedder:
    """Constrói o embedder do backend pedido; ``cache=True`` envolve em `CachingEmbedder`."""
    inner: Embedder
    if backend == "hash":
        inner = HashEmbedder()
    elif backend == "sentence_transformers":
        inner = SentenceTransformerEmbedder(model_name)
    else:
        msg = f"Unknown embeddings backend: {backend}"
        raise ValueError(msg)
    return CachingEmbedder(inner) if cache else inner

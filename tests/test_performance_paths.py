"""Testes das otimizações de corrida: cache de embeddings, pool HTTP e concorrência.

O invariante central é que nenhuma delas altera o resultado: a cache devolve os
mesmos vetores, o pool reutiliza a ligação e a concorrência preserva a ordem e o
conteúdo de ``predictions.jsonl``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from llm_evaluation.observability import LlmCallUsage, UsageAccumulator
from llm_evaluation.retrieval import CachingEmbedder, HashEmbedder, make_embedder


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._inner = HashEmbedder()

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return self._inner.embed(texts)


def test_cache_devolve_os_mesmos_vetores_do_embedder_interno() -> None:
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    direct = HashEmbedder().embed(["alfa", "beta"])
    np.testing.assert_allclose(cached.embed(["alfa", "beta"]), direct)


def test_cache_evita_reembeber_textos_repetidos_entre_chamadas() -> None:
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    cached.embed(["a", "b"])
    cached.embed(["b", "c"])
    assert inner.calls == [["a", "b"], ["c"]]
    assert cached.hits == 1
    assert cached.misses == 3


def test_cache_deduplica_dentro_da_mesma_chamada() -> None:
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    out = cached.embed(["x", "x", "x"])
    assert inner.calls == [["x"]]
    assert out.shape[0] == 3
    np.testing.assert_allclose(out[0], out[2])


def test_cache_preserva_a_ordem_pedida_mesmo_com_repeticoes() -> None:
    cached = CachingEmbedder(HashEmbedder())
    out = cached.embed(["a", "b", "a"])
    np.testing.assert_allclose(out[0], out[2])
    assert not np.allclose(out[0], out[1])


def test_cache_limpa_ao_exceder_o_limite_e_continua_correta() -> None:
    cached = CachingEmbedder(HashEmbedder(), max_entries=2)
    first = cached.embed(["a"])
    cached.embed(["b", "c"])  # excede o limite -> limpa
    np.testing.assert_allclose(cached.embed(["a"]), first)


def test_stats_reporta_taxa_de_acerto() -> None:
    cached = CachingEmbedder(HashEmbedder())
    cached.embed(["a"])
    cached.embed(["a"])
    stats = cached.stats()
    assert stats["embeddings_cache_hits"] == 1
    assert stats["embeddings_cache_misses"] == 1
    assert stats["embeddings_cache_taxa_acerto"] == 0.5


def test_cache_e_transparente_para_lista_vazia() -> None:
    """Entrada vazia delega no embedder interno: a cache não muda o contrato.

    Os chamadores (``Retriever``, ``max_cosine_answer_to_chunks``) já guardam o
    caso vazio antes de chegar aqui.
    """
    inner = CountingEmbedder()
    cached = CachingEmbedder(inner)
    with pytest.raises(ValueError, match="at least one array"):
        cached.embed([])
    with pytest.raises(ValueError, match="at least one array"):
        inner.embed([])


def test_make_embedder_so_envolve_em_cache_quando_pedido() -> None:
    assert isinstance(make_embedder("hash", "x"), HashEmbedder)
    assert isinstance(make_embedder("hash", "x", cache=True), CachingEmbedder)


def test_make_embedder_rejeita_backend_desconhecido() -> None:
    with pytest.raises(ValueError, match="Unknown embeddings backend"):
        make_embedder("inexistente", "x", cache=True)


def test_acumulador_de_uso_isola_threads() -> None:
    """Cada worker contabiliza apenas as suas chamadas — sem mistura entre itens."""
    acc = UsageAccumulator()
    snapshots: dict[str, dict[str, Any]] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, n_calls: int) -> None:
        for _ in range(n_calls):
            acc.record(
                LlmCallUsage(
                    role="generation",
                    model="m",
                    prompt_tokens=10,
                    completion_tokens=1,
                    total_tokens=11,
                    latency_ms=1.0,
                )
            )
        barrier.wait()
        snapshots[name] = acc.snapshot_for_item()
        acc.reset()

    threads = [
        threading.Thread(target=worker, args=("a", 1)),
        threading.Thread(target=worker, args=("b", 3)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert snapshots["a"]["n_chamadas_llm"] == 1
    assert snapshots["b"]["n_chamadas_llm"] == 3
    assert snapshots["b"]["tokens_total"] == 33

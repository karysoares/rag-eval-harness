"""Métricas de recuperação por item (pré-geração, diagnósticas)."""

from __future__ import annotations

from typing import Any

from llm_evaluation.types import EvalItem, RetrievedChunk

#: Nota anexada quando o corpus do item não tem nada além da passagem ouro.
NOTA_CORPUS_SEM_DISTRATORES = (
    "Corpus do item sem distratores: 'ouro no top-k' só pode dar verdadeiro. "
    "É verificação do caminho de código, não medição de recuperação — ver SPEC-012 "
    "para recuperação contra índice real."
)


def compute_retrieval_metrics(
    item: EvalItem,
    retrieved: list[RetrievedChunk],
    *,
    rag_enabled: bool,
    n_chunks_corpus: int | None = None,
    corpus_tem_distratores: bool | None = None,
) -> dict[str, Any]:
    """Métricas diagnósticas do retriever; não entram na agregação de anomalia por defeito.

    ``n_chunks_corpus`` e ``corpus_tem_distratores`` descrevem o **universo de
    busca**. Sem eles não se distingue «o retriever encontrou o ouro» de «não
    havia mais nada para encontrar», e as duas situações produzem exactamente os
    mesmos valores em ``rank_chunk_ouro`` e ``chunk_ouro_no_top_k``.
    """
    if not rag_enabled:
        return {"rag_ativo": False}

    top_score: float | None = retrieved[0].score if retrieved else None
    gold_rank: int | None = None
    for i, chunk in enumerate(retrieved, start=1):
        if chunk.is_gold:
            gold_rank = i
            break

    has_gold_corpus = bool((item.rag_gold_chunk or "").strip())

    out: dict[str, Any] = {
        "rag_ativo": True,
        "n_chunks_recuperados": len(retrieved),
        "score_melhor_chunk": top_score,
        "rank_chunk_ouro": gold_rank,
        "chunk_ouro_no_top_k": gold_rank is not None,
        "corpus_tem_chunk_ouro": has_gold_corpus,
    }
    if n_chunks_corpus is not None:
        out["n_chunks_corpus"] = n_chunks_corpus
        # Com o corpus todo devolvido, o retriever ordenou mas não seleccionou.
        out["corpus_devolvido_inteiro"] = len(retrieved) >= n_chunks_corpus
    if corpus_tem_distratores is not None:
        out["corpus_tem_distratores"] = corpus_tem_distratores
        if not corpus_tem_distratores and has_gold_corpus:
            out["nota_recuperacao_degenerada"] = NOTA_CORPUS_SEM_DISTRATORES
    return out

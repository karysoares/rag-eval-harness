"""Comparação de métodos de recuperação sobre um conjunto BEIR.

Quatro degraus, cada um com custo próprio, para responder à pergunta que
qualquer equipa com um RAG tem em aberto: **onde vale a pena parar?**

    bm25 → denso → híbrido (RRF) → reranking

O denso e o híbrido correm sobre os candidatos do BM25 (protocolo padrão do
MS MARCO e do BEIR): indexar densamente o corpus inteiro é possível aqui, mas
o protocolo de re-ranking é o que a literatura compara e o que escala.
"""

from __future__ import annotations

import time
from typing import Any

from llm_evaluation.retrieval_eval.beir import ConjuntoBeir
from llm_evaluation.retrieval_eval.bm25 import BM25Index
from llm_evaluation.retrieval_eval.metrics import avalia_corrida, ndcg_por_query
from llm_evaluation.statistics import paired_bootstrap_mean_diff

#: Constante clássica do Reciprocal Rank Fusion (Cormack et al., 2009).
RRF_K = 60


def rrf(*rankings: list[str], k: int = RRF_K, top_k: int = 1000) -> list[str]:
    """Funde rankings por posição, não por score — evita normalizar escalas incomparáveis."""
    pontos: dict[str, float] = {}
    for r in rankings:
        for pos, doc in enumerate(r, start=1):
            pontos[doc] = pontos.get(doc, 0.0) + 1.0 / (k + pos)
    return [d for d, _ in sorted(pontos.items(), key=lambda x: -x[1])[:top_k]]


def corrida_bm25(ix: BM25Index, ds: ConjuntoBeir, top_k: int = 1000) -> dict[str, list[str]]:
    return {q: [d for d, _ in ix.search(ds.queries[q], top_k)] for q in ds.qrels}


def corrida_densa(
    embedder: Any,
    ds: ConjuntoBeir,
    candidatos: dict[str, list[str]],
    *,
    profundidade: int = 100,
) -> dict[str, list[str]]:
    """Reordena os primeiros ``profundidade`` candidatos por coseno pergunta↔passagem."""
    import numpy as np

    por_id = dict(zip(ds.doc_ids, ds.textos, strict=True))
    out: dict[str, list[str]] = {}
    for qid, cands in candidatos.items():
        alvo = cands[:profundidade]
        if not alvo:
            out[qid] = []
            continue
        qv = embedder.embed([ds.queries[qid]])[0]
        dv = embedder.embed([por_id[d] for d in alvo])
        ordem = np.argsort(-(dv @ qv))
        # A cauda além da profundidade fica na ordem do BM25: não foi reordenada,
        # e descartá-la baixaria o recall@1000 por artefacto do protocolo.
        out[qid] = [alvo[i] for i in ordem] + cands[profundidade:]
    return out


def compara_metodos(
    ds: ConjuntoBeir,
    embedder: Any | None = None,
    *,
    top_k: int = 1000,
    profundidade_densa: int = 100,
    reranker: Any | None = None,
    profundidade_cross: int = 50,
) -> dict[str, Any]:
    """Corre os métodos disponíveis e devolve métricas e tempos por degrau."""
    resultados: dict[str, Any] = {"conjunto": ds.resumo(), "metodos": {}}
    corridas: dict[str, dict[str, list[str]] | None] = {}

    t0 = time.time()
    ix = BM25Index().build(ds.doc_ids, ds.textos)
    t_index = time.time() - t0

    t0 = time.time()
    r_bm25 = corrida_bm25(ix, ds, top_k)
    corridas["bm25"] = r_bm25
    resultados["metodos"]["bm25"] = {
        **avalia_corrida(r_bm25, ds.qrels),
        "segundos_indexacao": round(t_index, 1),
        "segundos_consulta": round(time.time() - t0, 1),
    }

    if embedder is None:
        resultados["nota"] = "Sem embedder: só BM25. Passe um para medir denso e híbrido."
        return resultados

    t0 = time.time()
    r_denso = corrida_densa(embedder, ds, r_bm25, profundidade=profundidade_densa)
    t_denso = time.time() - t0
    corridas["denso_sobre_bm25"] = r_denso
    resultados["metodos"]["denso_sobre_bm25"] = {
        **avalia_corrida(r_denso, ds.qrels),
        "segundos_consulta": round(t_denso, 1),
        "profundidade_reordenada": profundidade_densa,
    }

    t0 = time.time()
    r_hibrido = {q: rrf(r_bm25[q], r_denso[q], top_k=top_k) for q in r_bm25}
    corridas["hibrido_rrf"] = r_hibrido
    resultados["metodos"]["hibrido_rrf"] = {
        **avalia_corrida(r_hibrido, ds.qrels),
        "segundos_consulta": round(t_denso + time.time() - t0, 1),
        "rrf_k": RRF_K,
    }

    if reranker is not None:
        t0 = time.time()
        r_cross = corrida_cross_encoder(reranker, ds, r_bm25, profundidade=profundidade_cross)
        corridas["cross_encoder_sobre_bm25"] = r_cross
        resultados["metodos"]["cross_encoder_sobre_bm25"] = {
            **avalia_corrida(r_cross, ds.qrels),
            "segundos_consulta": round(time.time() - t0, 1),
            "profundidade_reordenada": profundidade_cross,
            "modelo": getattr(reranker, "nome", "?"),
        }

    resultados["comparacao_emparelhada"] = compara_emparelhado(corridas, ds.qrels)
    return resultados


def compara_emparelhado(
    corridas: dict[str, Any], qrels: dict[str, dict[str, float]], *, k: int = 10
) -> list[dict[str, Any]]:
    """Bootstrap emparelhado do nDCG@k entre cada par de métodos.

    Os métodos correm sobre **as mesmas queries**, logo a comparação é emparelhada.
    Sem isto só há médias lado a lado, e uma diferença de 0,03 pode ser real ou
    ruído — a ordenação da tabela seria uma afirmação sem suporte.
    """
    por_metodo = {nome: ndcg_por_query(c, qrels, k=k) for nome, c in corridas.items() if c}
    nomes = list(por_metodo)
    out: list[dict[str, Any]] = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1 :]:
            comuns = sorted(set(por_metodo[a]) & set(por_metodo[b]))
            if not comuns:
                continue
            r = paired_bootstrap_mean_diff(
                [por_metodo[a][q] for q in comuns], [por_metodo[b][q] for q in comuns]
            )
            if r is not None:
                out.append({"par": [a, b], "metrica": f"ndcg@{k}", **r})
    return out


class CrossEncoderReranker:
    """Reordena pares (query, passagem) com um cross-encoder.

    Ao contrário do bi-encoder, que compara vetores calculados em separado, o
    cross-encoder lê a query e a passagem **juntas** — capta interação entre
    termos que o coseno perde. O preço é não haver índice possível: cada par
    exige uma passagem pelo modelo, por isso só se aplica a uma lista curta de
    candidatos já filtrada por um método barato.

    É o degrau que responde a "vale a pena?" — e a resposta só é útil com a
    latência ao lado.
    """

    def __init__(self, modelo: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._modelo = CrossEncoder(modelo)
        self.nome = modelo

    def rerank(self, query: str, passagens: list[str]) -> list[int]:
        """Índices de ``passagens`` por ordem decrescente de relevância."""
        import numpy as np

        if not passagens:
            return []
        scores = self._modelo.predict([(query, p) for p in passagens], show_progress_bar=False)
        return list(np.argsort(-np.asarray(scores)))


def corrida_cross_encoder(
    reranker: CrossEncoderReranker,
    ds: ConjuntoBeir,
    candidatos: dict[str, list[str]],
    *,
    profundidade: int = 50,
) -> dict[str, list[str]]:
    """Reordena os primeiros ``profundidade`` candidatos; a cauda mantém a ordem de entrada."""
    por_id = dict(zip(ds.doc_ids, ds.textos, strict=True))
    out: dict[str, list[str]] = {}
    for qid, cands in candidatos.items():
        alvo = cands[:profundidade]
        if not alvo:
            out[qid] = []
            continue
        ordem = reranker.rerank(ds.queries[qid], [por_id[d] for d in alvo])
        out[qid] = [alvo[i] for i in ordem] + cands[profundidade:]
    return out

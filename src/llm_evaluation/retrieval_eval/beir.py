"""Carregamento de conjuntos no formato BEIR/MTEB, via parquet do Hugging Face.

Usa os parquet convertidos em vez de ``datasets.load_dataset`` porque o loader
não resolve o layout destes repositórios na versão fixada — e porque o parquet
carrega 57k passagens em segundos, sem script de dataset a executar.

Agnóstico ao conjunto: qualquer repositório com as três configurações do BEIR
(``corpus``, ``queries``, ``default`` com os qrels) funciona sem alterações.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

import pandas as pd

API = "https://huggingface.co/api/datasets"


@dataclass(frozen=True)
class ConjuntoBeir:
    nome: str
    doc_ids: list[str]
    textos: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]

    def resumo(self) -> dict[str, Any]:
        n_rel = sum(len(v) for v in self.qrels.values())
        return {
            "conjunto": self.nome,
            "n_passagens": len(self.doc_ids),
            "n_queries_total": len(self.queries),
            "n_queries_julgadas": len(self.qrels),
            "n_julgamentos": n_rel,
            "relevantes_por_query": round(n_rel / len(self.qrels), 2) if self.qrels else None,
        }


def _parquet(repo: str, cfg: str, split: str) -> pd.DataFrame:
    urls = json.load(urllib.request.urlopen(f"{API}/{repo}/parquet/{cfg}/{split}", timeout=60))
    return pd.concat([pd.read_parquet(u) for u in urls], ignore_index=True)


def carrega_beir(repo: str, split_qrels: str = "test", limite_queries: int = 0) -> ConjuntoBeir:
    """Carrega corpus, queries e qrels.

    ``limite_queries`` corta o número de queries **avaliadas**, nunca o corpus:
    reduzir o índice tornaria a recuperação artificialmente fácil e o número
    incomparável com a literatura.
    """
    corpus = _parquet(repo, "corpus", "corpus")
    queries = _parquet(repo, "queries", "queries")
    qrels_df = _parquet(repo, "default", split_qrels)

    qrels: dict[str, dict[str, float]] = {}
    for qid, did, score in zip(
        qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"], strict=True
    ):
        if float(score) > 0:
            qrels.setdefault(str(qid), {})[str(did)] = float(score)

    if limite_queries > 0:
        mantidas = sorted(qrels)[:limite_queries]
        qrels = {q: qrels[q] for q in mantidas}

    titulo = corpus["title"].fillna("")
    textos = (titulo + " " + corpus["text"].fillna("")).str.strip().tolist()
    return ConjuntoBeir(
        nome=repo,
        doc_ids=[str(x) for x in corpus["_id"]],
        textos=textos,
        queries={str(i): str(t) for i, t in zip(queries["_id"], queries["text"], strict=True)},
        qrels=qrels,
    )

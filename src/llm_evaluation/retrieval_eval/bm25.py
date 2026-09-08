"""BM25 esparso sobre um corpus em memória.

Implementado aqui em vez de trazer uma dependência, pela mesma razão que
``statistics.py``: são vinte linhas de fórmula e o resultado é **verificável
contra números publicados** — se o nDCG@10 no FIQA do BEIR bater com a
literatura, a implementação está certa. Uma biblioteca dava o mesmo número sem
essa prova.

Corpora do tamanho do BEIR (dezenas de milhares de passagens) cabem folgadamente
numa matriz esparsa; para milhões, isto deixa de servir e passa a ser preciso um
índice invertido em disco.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(texto: str) -> list[str]:
    """Minúsculas e alfanuméricos. Sem stemming — o BEIR compara-se assim."""
    return TOKEN_RE.findall(texto.lower())


@dataclass
class BM25Index:
    """Índice BM25 com os parâmetros clássicos de Robertson."""

    k1: float = 0.9
    b: float = 0.4  # defaults do BEIR/Anserini, não os 1.2/0.75 do artigo original

    doc_ids: list[str] = field(default_factory=list, repr=False)
    _vocab: dict[str, int] = field(default_factory=dict, repr=False)
    _matriz: sparse.csr_matrix | None = field(default=None, repr=False)

    def build(self, doc_ids: list[str], textos: list[str]) -> BM25Index:
        """Pré-calcula os pesos por (termo, documento); a consulta passa a ser uma soma."""
        self.doc_ids = doc_ids
        linhas, colunas, freqs = [], [], []
        comprimentos = np.zeros(len(textos), dtype=np.float32)
        for j, texto in enumerate(textos):
            toks = tokenize(texto)
            comprimentos[j] = len(toks)
            contagem: dict[int, int] = {}
            for t in toks:
                i = self._vocab.setdefault(t, len(self._vocab))
                contagem[i] = contagem.get(i, 0) + 1
            for i, f in contagem.items():
                linhas.append(i)
                colunas.append(j)
                freqs.append(f)

        n_docs = len(textos)
        tf = sparse.csr_matrix(
            (np.array(freqs, dtype=np.float32), (linhas, colunas)),
            shape=(len(self._vocab), n_docs),
        )
        # IDF de Robertson com o +1 que evita peso negativo em termos muito comuns.
        df = np.asarray((tf > 0).sum(axis=1)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        media = float(comprimentos.mean()) or 1.0
        norm = (self.k1 * (1.0 - self.b + self.b * comprimentos / media)).astype(np.float32)

        # peso = idf * tf*(k1+1) / (tf + norm_doc), calculado só nos não-zeros
        tf = tf.tocoo()
        num = tf.data * (self.k1 + 1.0)
        den = tf.data + norm[tf.col]
        pesos = idf[tf.row] * (num / den)
        # CSR e não CSC: a consulta seleciona **linhas** (os termos da query), e
        # indexar linhas numa matriz coluna-major percorre todas as colunas.
        self._matriz = sparse.csr_matrix(
            (pesos.astype(np.float32), (tf.row, tf.col)), shape=tf.shape
        )
        return self

    def search(self, query: str, top_k: int = 1000) -> list[tuple[str, float]]:
        if self._matriz is None:
            msg = "índice não construído; chame build() primeiro"
            raise RuntimeError(msg)
        idx = [self._vocab[t] for t in tokenize(query) if t in self._vocab]
        if not idx:
            return []
        scores = np.asarray(self._matriz[idx, :].sum(axis=0)).ravel()
        k = min(top_k, scores.size)
        # argpartition evita ordenar 57k scores quando só queremos os k melhores
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[j], float(scores[j])) for j in top if scores[j] > 0]

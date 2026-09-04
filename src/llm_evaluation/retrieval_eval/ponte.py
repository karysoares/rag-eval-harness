"""Ponte entre recuperação real e geração: o mesmo item passa pelas duas.

[SPEC-012](../../../docs/specs/012-retrieval-evaluation.md) mede recuperação
contra um índice real, e o pipeline de geração corre sobre um corpus construído
por item. São duas medições desconexas: nenhum item passa pelas duas, e portanto
a pergunta que um harness de RAG existe para responder — *recuperar melhor
melhora a resposta?* — fica sem medição.

Esta ponte fecha isso. Precisa de um conjunto com três coisas **no mesmo item**:

1. um corpus grande onde procurar,
2. julgamentos de relevância humanos, para saber o que era recuperável,
3. uma resposta ouro, para avaliar a geração.

O HotpotQA é o candidato verificado: `mteb/hotpotqa` traz corpus, queries e
qrels no formato BEIR, e `hotpotqa/hotpot_qa` traz as respostas — e os ids das
duas fontes **coincidem**, o que permite a junção sem heurística.

## Sobre o tamanho do índice

O corpus do HotpotQA tem 5,2 milhões de passagens, o que não cabe num BM25 em
memória a esta escala de máquina. O índice usado aqui é um **subconjunto
declarado**: todas as passagens julgadas das queries seleccionadas, mais uma
amostra aleatória com semente fixa.

Isto **não** é comparável com a tabela do BEIR, e não deve ser apresentado como
tal. A regra de `carrega_beir` — cortar queries, nunca o corpus — existe
precisamente para proteger essa comparabilidade no FIQA. Aqui abre-se a excepção
de propósito, porque o objectivo é outro: medir o efeito da recuperação sobre a
geração, não posicionar um recuperador contra a literatura.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from llm_evaluation.types import EvalItem

API = "https://huggingface.co/api/datasets"

#: Repositório BEIR (corpus, queries, qrels) e repositório com as respostas ouro.
REPO_BEIR = "mteb/hotpotqa"
REPO_RESPOSTAS = "hotpotqa/hotpot_qa"


@dataclass(frozen=True)
class ConjuntoPonte:
    """Corpus, qrels humanos e resposta ouro, alinhados pelo mesmo id de query."""

    nome: str
    doc_ids: list[str]
    textos: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]
    #: O que a ponte acrescenta ao formato BEIR: a resposta que se espera gerar.
    respostas: dict[str, str]
    n_corpus_original: int
    semente: int
    #: Queries com qrels mas sem resposta ouro, e vice-versa. Contadas em vez de
    #: descartadas em silêncio: uma junção que perde metade dos itens é um
    #: defeito, e sem a contagem parece um conjunto pequeno.
    diagnostico_juncao: dict[str, int] = field(default_factory=dict)

    def resumo(self) -> dict[str, Any]:
        n_rel = sum(len(v) for v in self.qrels.values())
        return {
            "conjunto": self.nome,
            "n_passagens_indexadas": len(self.doc_ids),
            "n_passagens_no_corpus_original": self.n_corpus_original,
            "corpus_subamostrado": len(self.doc_ids) < self.n_corpus_original,
            "n_queries": len(self.queries),
            "n_julgamentos": n_rel,
            "relevantes_por_query": round(n_rel / len(self.qrels), 2) if self.qrels else None,
            "semente_amostragem": self.semente,
            "comparavel_com_beir": False,
            "nota": (
                "Índice subamostrado de propósito: mede o efeito da recuperação na "
                "geração, não posiciona o recuperador contra a literatura."
            ),
            **self.diagnostico_juncao,
        }


def _urls_parquet(repo: str, cfg: str, split: str) -> list[str]:
    pedido = urllib.request.Request(  # noqa: S310 - URL fixa, esquema https
        f"{API}/{repo}/parquet/{cfg}/{split}",
        headers={"User-Agent": "rag-eval-harness"},
    )
    with urllib.request.urlopen(pedido, timeout=120) as resposta:  # noqa: S310
        return [str(u) for u in json.load(resposta)]


def _parquet(repo: str, cfg: str, split: str, *, colunas: list[str] | None = None) -> pd.DataFrame:
    partes = [pd.read_parquet(u, columns=colunas) for u in _urls_parquet(repo, cfg, split)]
    return pd.concat(partes, ignore_index=True)


def carrega_ponte_hotpotqa(
    *,
    n_queries: int = 200,
    n_distratores_corpus: int = 150_000,
    seed: int = 42,
) -> ConjuntoPonte:
    """Carrega o conjunto-ponte com o índice subamostrado.

    ``n_distratores_corpus`` controla a dificuldade da recuperação e é o
    parâmetro que torna o número não comparável com a literatura — vai no
    resumo, para não se perder.

    O corpus é lido shard a shard e filtrado à medida: materializar 5,2 milhões
    de passagens para depois deitar fora 97% delas gastaria vários GB sem
    necessidade.
    """
    qrels_df = _parquet(REPO_BEIR, "default", "test")
    qrels_todas: dict[str, dict[str, float]] = {}
    for qid, did, score in zip(
        qrels_df["query-id"], qrels_df["corpus-id"], qrels_df["score"], strict=True
    ):
        if float(score) > 0:
            qrels_todas.setdefault(str(qid), {})[str(did)] = float(score)

    respostas_df = _parquet(REPO_RESPOSTAS, "distractor", "validation", colunas=["id", "answer"])
    respostas_todas = {
        str(i): str(a) for i, a in zip(respostas_df["id"], respostas_df["answer"], strict=True)
    }

    com_ambos = sorted(set(qrels_todas) & set(respostas_todas))
    diagnostico = {
        "n_queries_com_qrels": len(qrels_todas),
        "n_queries_com_resposta": len(respostas_todas),
        "n_queries_com_ambos": len(com_ambos),
    }
    # Selecção determinística: `sorted` fixa a ordem antes de qualquer corte, para
    # que duas corridas com a mesma semente vejam exactamente as mesmas queries.
    seleccionadas = com_ambos[:n_queries] if n_queries > 0 else com_ambos

    qrels = {q: qrels_todas[q] for q in seleccionadas}
    respostas = {q: respostas_todas[q] for q in seleccionadas}
    ids_julgados = {d for v in qrels.values() for d in v}

    queries_df = _parquet(REPO_BEIR, "queries", "queries")
    queries = {
        str(i): str(t)
        for i, t in zip(queries_df["_id"], queries_df["text"], strict=True)
        if str(i) in qrels
    }

    doc_ids, textos, n_original = _corpus_subamostrado(
        ids_julgados,
        n_distratores=n_distratores_corpus,
        seed=seed,
    )
    return ConjuntoPonte(
        nome=f"{REPO_BEIR}+{REPO_RESPOSTAS}",
        doc_ids=doc_ids,
        textos=textos,
        queries=queries,
        qrels=qrels,
        respostas=respostas,
        n_corpus_original=n_original,
        semente=seed,
        diagnostico_juncao=diagnostico,
    )


def _corpus_subamostrado(
    ids_julgados: set[str],
    *,
    n_distratores: int,
    seed: int,
) -> tuple[list[str], list[str], int]:
    """Passagens julgadas mais uma amostra aleatória, lidas shard a shard.

    As julgadas entram **todas**: deixar de fora uma passagem relevante tornaria
    a query impossível e o `recall` mediria a amostragem em vez do recuperador.
    """
    rng = random.Random(seed)
    ids: list[str] = []
    textos: list[str] = []
    reservatorio_ids: list[str] = []
    reservatorio_textos: list[str] = []
    n_vistos_nao_julgados = 0
    n_original = 0

    for url in _urls_parquet(REPO_BEIR, "corpus", "corpus"):
        shard = pd.read_parquet(url, columns=["_id", "title", "text"])
        n_original += len(shard)
        titulo = shard["title"].fillna("")
        corpo = (titulo + " " + shard["text"].fillna("")).str.strip()
        for did, texto in zip(shard["_id"], corpo, strict=True):
            did_s = str(did)
            if did_s in ids_julgados:
                ids.append(did_s)
                textos.append(str(texto))
                continue
            # Amostragem por reservatório: uma passagem única sobre os quatro
            # shards, sem saber de antemão quantas passagens existem.
            n_vistos_nao_julgados += 1
            if len(reservatorio_ids) < n_distratores:
                reservatorio_ids.append(did_s)
                reservatorio_textos.append(str(texto))
            else:
                j = rng.randrange(n_vistos_nao_julgados)
                if j < n_distratores:
                    reservatorio_ids[j] = did_s
                    reservatorio_textos[j] = str(texto)
        del shard, titulo, corpo

    return ids + reservatorio_ids, textos + reservatorio_textos, n_original


def itens_para_pipeline(
    conjunto: ConjuntoPonte,
    corrida: dict[str, list[str]],
    *,
    top_k: int = 4,
    desvio: int = 0,
) -> list[EvalItem]:
    """Converte uma corrida de recuperação em itens do pipeline de geração.

    ``desvio`` desloca a janela de candidatos: com 0, o item recebe o topo do
    ranking; com 50, recebe os candidatos da posição 50 em diante. É assim que a
    ablação degrada a recuperação **sem tocar em mais nada** — mesmo índice,
    mesmo recuperador, mesmas queries, mesma geração. A única variável é a
    qualidade do contexto entregue.

    O ``rag_gold_chunk`` é preenchido **apenas quando a passagem relevante caiu
    mesmo dentro da janela**. Anexá-lo sempre foi um defeito real e caro: como
    ``build_chunks_for_item`` coloca o chunk ouro em primeiro lugar no contexto,
    o braço degradado recebia na mesma a passagem que a degradação devia ter-lhe
    tirado — 97 de 100 itens numa corrida cujos números foram deitados fora. É a
    mesma falha do corpus sintético da SPEC-001 a reaparecer dentro do módulo
    construído para a eliminar.

    O contexto entregue é, e só é, o que a janela contém.
    """
    por_id = dict(zip(conjunto.doc_ids, conjunto.textos, strict=True))
    itens: list[EvalItem] = []
    for qid in sorted(conjunto.qrels):
        candidatos = corrida.get(qid, [])
        janela = [d for d in candidatos[desvio : desvio + top_k] if d in por_id]
        relevantes = set(conjunto.qrels[qid])
        na_janela = [d for d in janela if d in relevantes]
        # Um relevante que caiu na janela vira chunk ouro; os restantes seguem
        # como contexto normal — o HotpotQA tem 2 relevantes por query e descartar
        # o segundo removeria a evidência de que a pergunta precisa.
        ouro = por_id[na_janela[0]] if na_janela else None
        resto = [por_id[d] for d in janela if not na_janela or d != na_janela[0]]
        itens.append(
            EvalItem(
                id=qid,
                question=conjunto.queries.get(qid, ""),
                correct_answers=[conjunto.respostas[qid]],
                incorrect_answers=[],
                category="hotpotqa",
                rag_gold_chunk=ouro,
                rag_distractors=resto,
            )
        )
    return itens


def contexto_entregue_tem_relevante(
    conjunto: ConjuntoPonte,
    itens: list[EvalItem],
    *,
    chunk_max_chars: int = 1200,
) -> dict[str, Any]:
    """Fracção de itens cujo **contexto montado** contém uma passagem relevante.

    A cobertura mede o ranking; isto mede o que chega ao gerador. As duas podem
    divergir, e divergiram: `rag_gold_chunk` era anexado sempre e
    `build_chunks_for_item` põe-no em primeiro lugar, portanto o braço degradado
    recebia de volta a passagem que a degradação lhe tirara — 97 de 100 itens,
    numa corrida cujos números tiveram de ser deitados fora.

    Uma ablação sem esta verificação pode correr inteira, custar dinheiro, e
    medir nada.
    """
    from llm_evaluation.datasets_rag import build_chunks_for_item

    por_id = dict(zip(conjunto.doc_ids, conjunto.textos, strict=True))
    com_relevante = 0
    for item in itens:
        chunks = build_chunks_for_item(item, chunk_max_chars)
        relevantes = {por_id[d] for d in conjunto.qrels.get(item.id, {}) if d in por_id}
        # Prefixo e não igualdade: `chunk_text` parte passagens longas.
        if any(any(t[:80] == c[:80] for t in relevantes) for c in chunks):
            com_relevante += 1
    total = len(itens)
    return {
        "n_itens": total,
        "n_com_relevante_no_contexto": com_relevante,
        "fracao": round(com_relevante / total, 4) if total else None,
    }


def verifica_manipulacao(por_braco: dict[str, dict[str, Any]], *, minimo: float = 0.2) -> None:
    """Aborta se os braços entregam contextos equivalentes.

    Correr a geração quando os braços não diferem gasta dinheiro para medir ruído
    e produz um resultado que parece interpretável. Falhar aqui é barato.
    """
    fracoes = {n: v["fracao"] for n, v in por_braco.items() if v["fracao"] is not None}
    if len(fracoes) < 2:
        return
    amplitude = max(fracoes.values()) - min(fracoes.values())
    if amplitude < minimo:
        msg = (
            f"Braços não distinguíveis no contexto entregue: {fracoes}. "
            f"Amplitude {amplitude:.3f} < {minimo}. A geração mediria ruído."
        )
        raise ValueError(msg)


def cobertura_da_recuperacao(
    conjunto: ConjuntoPonte,
    corrida: dict[str, list[str]],
    *,
    top_k: int = 4,
    desvio: int = 0,
) -> dict[str, Any]:
    """Fracção de queries cuja janela contém pelo menos uma passagem relevante.

    É a variável independente da ablação: sem ela, uma diferença no grounding
    não se distingue de ruído da geração. Publicada ao lado do resultado.
    """
    total = 0
    com_relevante = 0
    for qid, relevantes in conjunto.qrels.items():
        janela = set(corrida.get(qid, [])[desvio : desvio + top_k])
        if not janela:
            continue
        total += 1
        if janela & set(relevantes):
            com_relevante += 1
    return {
        "n_queries": total,
        "n_com_relevante_na_janela": com_relevante,
        "cobertura": round(com_relevante / total, 4) if total else None,
        "top_k": top_k,
        "desvio": desvio,
    }

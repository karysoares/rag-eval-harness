"""Métricas de recuperação sobre julgamentos de relevância (qrels).

Sem dependências externas, pela mesma razão que ``statistics.py``: são fórmulas
curtas, verificáveis contra valores conhecidos, e não vale trazer uma biblioteca
para as ter. Cada uma responde a uma pergunta diferente e **não se somam**:

- ``recall@k`` — dos documentos relevantes, quantos apanhámos em k?
- ``nDCG@k``   — apanhámos, e ficaram no topo? (posição importa)
- ``MRR@k``    — a que distância está o primeiro acerto?

Convenção BEIR: relevância binária ou graduada em ``qrels[query_id][doc_id]``.
Uma query sem julgamentos não entra em nenhum denominador — contá-la como zero
seria inventar um resultado negativo onde não há verdade.
"""

from __future__ import annotations

import math
from typing import Any

Qrels = dict[str, dict[str, float]]
Corridas = dict[str, list[str]]


def recall_at_k(relevantes: set[str], ranking: list[str], k: int) -> float:
    """``nan`` sem relevantes — nunca zero, que seria inventar um resultado.

    Quem agrega tem de filtrar antes: um único ``nan`` numa soma contamina a média
    inteira em silêncio. ``avalia_corrida`` já só passa queries com julgamentos.
    """
    if not relevantes:
        return float("nan")
    return len(relevantes & set(ranking[:k])) / len(relevantes)


def mrr_at_k(relevantes: set[str], ranking: list[str], k: int) -> float:
    for i, doc in enumerate(ranking[:k], start=1):
        if doc in relevantes:
            return 1.0 / i
    return 0.0


def ndcg_at_k(graus: dict[str, float], ranking: list[str], k: int) -> float:
    """nDCG com desconto logarítmico e ganho linear — a convenção do BEIR.

    O ideal usa os graus realmente disponíveis, por isso uma query cujos
    relevantes não cabem em k não é penalizada por isso.
    """
    dcg = sum(graus.get(doc, 0.0) / math.log2(i + 1) for i, doc in enumerate(ranking[:k], start=1))
    ideal = sorted(graus.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def avalia_corrida(
    corrida: Corridas,
    qrels: Qrels,
    *,
    ks: tuple[int, ...] = (10, 100, 1000),
    k_ndcg: int = 10,
) -> dict[str, Any]:
    """Agrega as três métricas sobre as queries que têm julgamentos.

    Devolve também ``n_queries_avaliadas`` e ``n_queries_sem_qrels``: o
    denominador tem de ser visível, senão um conjunto parcialmente julgado
    parece melhor do que é.
    """
    # O denominador é o conjunto **julgado**, não o que a corrida devolveu. Iterar
    # sobre a corrida deixaria cair em silêncio as queries que o método não
    # respondeu — e cada uma dessas em falta sobe todas as médias. Convenção do
    # IR: query julgada sem resultados vale zero, não desaparece.
    ausentes = [q for q in qrels if q not in corrida]
    avaliadas = [(q, g) for q, g in qrels.items() if g]
    sem_qrels = sum(1 for q in corrida if not qrels.get(q))
    if not avaliadas:
        return {
            "n_queries_avaliadas": 0,
            "n_queries_sem_qrels": sem_qrels,
            "n_queries_julgadas_ausentes": len(ausentes),
        }

    out: dict[str, Any] = {
        "n_queries_avaliadas": len(avaliadas),
        "n_queries_sem_qrels": sem_qrels,
        "n_queries_julgadas_ausentes": len(ausentes),
    }
    for k in ks:
        vals = [recall_at_k(set(g), corrida.get(q, []), k) for q, g in avaliadas]
        out[f"recall@{k}"] = round(sum(vals) / len(vals), 4)
    out[f"ndcg@{k_ndcg}"] = round(
        sum(ndcg_at_k(g, corrida.get(q, []), k_ndcg) for q, g in avaliadas) / len(avaliadas), 4
    )
    out[f"mrr@{k_ndcg}"] = round(
        sum(mrr_at_k(set(g), corrida.get(q, []), k_ndcg) for q, g in avaliadas) / len(avaliadas), 4
    )
    return out


def ndcg_por_query(corrida: Corridas, qrels: Qrels, *, k: int = 10) -> dict[str, float]:
    """nDCG@k de cada query julgada — a granularidade que um teste emparelhado exige.

    Comparar dois métodos sobre as mesmas queries é um desenho emparelhado; sem os
    valores por query só é possível comparar médias, que é precisamente o que não
    permite dizer se a diferença é distinguível de ruído.
    """
    return {q: ndcg_at_k(g, corrida.get(q, []), k) for q, g in qrels.items() if g}

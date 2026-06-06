# Métricas de avaliação de RAG

## 1. Motivação

Quantificar **recuperação** e **aderência** ao contexto, inspirando-se em frameworks como RAGAS.

## 2. Intuição

Separar “o contexto foi mau?” de “a resposta ignorou um bom contexto?”.

## 3. Definição operacional

- `max_context_similarity` (proxy de faithfulness heurístico).
- `gold_chunk_rank` quando há chunk de referência injetado no dataset sintético.

## 4. Algoritmo

Ver `docs/metrics.md` e `reporting.py` para agregação.

## 5. Hiperparâmetros

`top_k`, limiares de flag, modo `inject_retrieval_failure`.

## 6. Onde falha

Métricas com LLM interno (RAGAS completo) herdam viés; proxies por embedding ignoram nuances.

## 7. Neste repositório

- `src/llm_evaluation/datasets_rag.py` (injecção de falha), `verification/embedding_verify.py`

## 8. Leituras

- [RAGAS](https://arxiv.org/abs/2312.10997)

## 9. Exercícios

1. Lista dois falsos negativos possíveis de `max_context_similarity`.
2. Como compararias este projeto a RAGAS *end-to-end* num relatório?

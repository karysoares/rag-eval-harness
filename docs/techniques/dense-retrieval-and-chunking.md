# Recuperação densa e chunking

## 1. Motivação

Fornecer **evidência** ao gerador (RAG) e permitir medir falhas quando o retriever não traz o trecho certo.

## 2. Intuição

Texto é particionado em chunks; embeddings mapeiam pergunta e chunks a vetores; os mais próximos (coseno) sobem no *ranking*.

## 3. Definição operacional

- **Entrada**: corpus de strings `chunks`, `query`.
- **Saída**: lista ordenada `(chunk, score)` top-k.

## 4. Algoritmo

`embed(query)`, `embed(chunk_i)` → coseno → `argsort` → top-k. (Implementação NumPy para portabilidade; ver `retrieval.py`.)

## 5. Hiperparâmetros

`top_k`, tamanho de chunk, modelo de embedding (`hash` demo vs `sentence_transformers`).

## 6. Onde falha

Chunk corta evidência ao meio; OOD; perguntas ambíguas (Karpukhin et al., DPR).

## 7. Neste repositório

- `src/llm_evaluation/retrieval.py`, `datasets_rag.py`
- `docs/techniques/embedding-similarity-vs-semantic-equivalence.md`

## 8. Leituras

- [DPR](https://arxiv.org/abs/2004.04906)

## 9. Exercícios

1. Como o rank do chunk gold muda se duplicares chunks irrelevantes no índice?
2. Proponhe uma heurística simples de *chunk overlap* e o trade-off com custo.

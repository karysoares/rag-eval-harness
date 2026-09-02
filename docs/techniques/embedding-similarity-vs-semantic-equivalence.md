# Similaridade por embedding vs equivalência semântica

## 1. Motivação

Sinal **barato** para saber se a resposta “parece” sustentada pelo contexto recuperado.

## 2. Intuição

Alta similaridade coseno sugere proximidade no espaço de embeddings; não implica entailment lógico.

## 3. Definição operacional

- Máximo (ou média dos máximos por frase da resposta) da similaridade entre embedding da resposta e cada chunk.

## 4. Algoritmo

Segmentar resposta em frases → para cada frase, `max_i cos(e(f), e(chunk_i))` → agregar (média).

## 5. Hiperparâmetros

`embedding_min_cosine` em YAML; escolha de embedder.

## 6. Onde falha

Negação, antónimos em espaço próximo, respostas genéricas “coladas” ao tema.

## 7. Neste repositório

- `src/llm_evaluation/verification/embedding_verify.py`

## 8. Leituras

- Literatura BEIR / bi-encoders

## 9. Exercícios

1. Constrói um par resposta/contexto com alto coseno mas contradição factual.
2. Quando preferirias média em vez de mínimo dos máximos por frase?

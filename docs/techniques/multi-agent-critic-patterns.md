# Padrões multi-agente com crítico

## 1. Motivação

Separar **recuperação**, **geração** e **crítica** para auditoria e análise por camada.

## 2. Intuição

Retriever devolve evidência; Responder produz texto; Critic emite achados; opcionalmente um sintetizador agrega com sinais numéricos.

## 3. Definição operacional

CLI `--orchestration multi` usa os mesmos contratos de saída JSON que o modo `single`, com passos explícitos nos logs.

## 4. Algoritmo

Ver `src/llm_evaluation/orchestration/multi.py`: retrieve → respond → critic (LLM) → agregação híbrida.

## 5. Hiperparâmetros

Mesmos que single; custo multiplica pelo número de chamadas do crítico.

## 6. Onde falha

Propagação de erro entre agentes; latência (Shinn et al., Reflexion).

## 7. Neste repositório

- `src/llm_evaluation/orchestration/multi.py`, `scripts/run_eval.py`

## 8. Leituras

- [Reflexion](https://arxiv.org/abs/2303.11366)

## 9. Exercícios

1. Em que passo introduzirias um “second opinion” de juiz?
2. Como evitarias que o Critic apenas repita a resposta do Responder?

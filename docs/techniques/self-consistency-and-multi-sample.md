# Self-consistency e multi-amostragem

## 1. Motivação

Reduzir erros de raciocínio amostrando múltiplas cadeias e votando (quando aplicável).

## 2. Intuição

Se amostras divergem na conclusão factual, o item é **suspeito** de instabilidade.

## 3. Definição operacional (opcional)

Parâmetro `num_samples` > 1 dispara várias gerações; `consistency_score` = frequência da moda da classe normalizada de resposta (heurística por hash de normalização).

## 4. Algoritmo

Loop `n` vezes → agregar → flag se variância alta (implementação mínima no `pipeline`).

## 5. Hiperparâmetros

`num_samples`, temperatura > 0.

## 6. Onde falha

Erro sistemático: todas as amostras erram igual (Wang et al., 2023).

## 7. Neste repositório

- `src/llm_evaluation/pipeline.py` — quando `generation.num_samples > 1`

## 8. Leituras

- [Self-Consistency](https://arxiv.org/abs/2203.11171)

## 9. Exercícios

1. Para QA aberto com `reference_type: answer_lists`, como definirias “voto” sem respostas MC?
2. Custo vs ganho esperado quando `num_samples=5`?

# LLM-as-judge

## 1. Motivação

Capturar juízos qualitativos (contradição, unsupported, política) difíceis só com similaridade.

## 2. Intuição

Rubrica fixa + saída JSON estruturada reduz ambiguidade; ainda assim o juiz pode errar ou ser enviesado.

## 3. Definição operacional

- **Entrada**: pergunta, contexto (chunks concatenados), resposta do modelo.
- **Saída**: `verdict`, `reason_short`, `confidence` (0–1).

## 4. Algoritmo

Chamada OpenAI-compatible com `prompts/judge_system.txt` e template de utilizador.

## 5. Hiperparâmetros

Modelo juiz, temperatura baixa, *timeout* HTTP.

## 6. Onde falha

Viés de posição, verbosidade, sycophancy; juiz da mesma família que o gerador (Zheng et al., 2023).

## 7. Neste repositório

- `src/llm_evaluation/verification/judge.py`, `prompts/`
- Modo demo: juiz heurístico sem API

## 8. Leituras

- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)

## 9. Exercícios

1. Escreve uma rubrica adicional para “não inventar números sem fonte”.
2. Como testarias o juiz com exemplos *gold* fixos sem olhar para outputs do teu gerador primeiro?

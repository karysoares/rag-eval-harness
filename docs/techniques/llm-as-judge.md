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

Estas falhas não se resolvem lendo o prompt — medem-se. Ver [SPEC-010](../specs/010-judge-meta-evaluation.md).

## 7. Neste repositório

- `src/llm_evaluation/verification/judge.py`, `prompts/` — o juiz como camada
- `src/llm_evaluation/judge_meta.py` — o juiz como **instrumento**: calibração (ECE), concordância (κ), sondas de verbosidade e posição, auto-consistência (κ de Fleiss)
- `scripts/judge_self_consistency.py` — amostragem repetida com API
- Modo demo: juiz heurístico sem API (excluído de toda a meta-avaliação — não é uma medição do juiz)

```bash
uv run llm-eval --judge-report outputs/run_<id>
```

Três leituras que mudam decisões:

| Sinal | Consequência prática |
|-------|----------------------|
| ECE alto com exatidão alta | Não usar `confianca` como limiar de triagem — o juiz é útil, a confiança dele não é |
| Aprovação cai fora do rank 1 (ICs disjuntos) | O juiz lê sobretudo o topo do contexto; rever `top_k` e ordem antes de confiar no grounding |
| κ de Fleiss baixo entre amostras | Piso ao efeito mínimo detetável: diferenças A/B abaixo do ruído do juiz não são interpretáveis |

## 8. Leituras

- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)

## 9. Exercícios

1. Escreve uma rubrica adicional para “não inventar números sem fonte”.
2. Como testarias o juiz com exemplos *gold* fixos sem olhar para outputs do teu gerador primeiro?
3. Corre `--judge-report` numa corrida real. Se a correlação de verbosidade for alta, distingue as duas explicações concorrentes — juiz enviesado vs. respostas longas genuinamente melhores — inspecionando uma amostra estratificada por comprimento.

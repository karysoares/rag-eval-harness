# Calibração da política `embedding_e_juiz`

## Problema

Com `qualquer_critico`, embedding baixo + juiz `sustentado` gera **falso positivo** em gold-correto: o modelo respondeu bem face ao contexto, mas o coseno ficou abaixo do limiar.

## Mitigação (P0)

- **Agregação:** `embedding_e_juiz` — anomalia só se embedding baixo **e** juiz negativo (tiers de agregação em `judge_aggregation_verdicts`, sem `incompleto` por defeito em RAG pt-BR).
- **Limiar:** `verification.embedding_min_cosine` — `0.24` nos configs FairytaleQA, `0.26` em [`ptbr_fairytale_tuned.yaml`](../configs/ptbr_fairytale_tuned.yaml). Curva medida em [`evidencia/embedding_sweep_fairytale_200.json`](evidencia/embedding_sweep_fairytale_200.json); validar a política com `scripts/validate_embedding_policy.py`.

## Validação offline

```bash
uv run python scripts/validate_embedding_policy.py outputs/run_<id>
uv run python scripts/validate_embedding_policy.py tests/fixtures/policy_validation_run
```

## Sweep de limiar (curva FP/FN)

Com corrida concluída (`predictions.jsonl`):

```bash
uv run python scripts/sweep_embedding_threshold.py outputs/run_<id>/predictions.jsonl
```

Gera `embedding_sweep.csv` e `.json` com FP/FN, precisão, revocação e taxa de alerta por limiar. O rótulo de referência sai de `referencia_incorreta`, pelo que o sweep corre tanto em `answer_lists` como em `lexical`.

### O que a curva medida diz (e o que não diz)

Sweep de 0,10 a 0,50 sobre `run_20260902T223513Z` (FairytaleQA pt-BR, N=200, `reference_type: lexical`, 99 itens com referência léxica fraca):

| limiar | FP (ref. aceitável) | TP (ref. fraca) | precisão | revocação | taxa de alerta |
|---|---|---|---|---|---|
| 0,22 | 0 | 1 | 1,000 | 0,010 | 0,005 |
| **0,24** | 1 | 2 | 0,667 | 0,020 | 0,015 |
| 0,28 | 1 | 2 | 0,667 | 0,020 | 0,015 |
| 0,34 | 3 | 2 | 0,400 | 0,020 | 0,025 |
| 0,42 | 5 | 6 | 0,545 | 0,061 | 0,055 |
| 0,50 | 16 | 16 | 0,500 | 0,162 | 0,160 |

**Não há ótimo.** A revocação vai de 1% a 16% em todo o intervalo e a precisão fica perto de 0,5 — ou seja, o coseno resposta↔contexto **não discrimina** a fraqueza da referência léxica a nenhum limiar. Isto é o resultado esperado, não uma falha de calibração: são planos métricos distintos (grounding vs sobreposição léxica), como a [ADR 0001](decisions/0001-reference-types.md) e o `README` já dizem sobre o κ baixo entre juiz e referência.

A consequência prática é que `embedding_min_cosine` **não se escolhe por curva PR**. Escolhe-se como piso conservador: `0.24` mantém a taxa de alerta em 1,5% e o falso alarme em 1 item de 101 com referência aceitável. Subir o limiar compra revocação irrisória a troco de falsos alarmes lineares. Quem quiser um detector com revocação útil precisa de NLI, não de um limiar de coseno — ver [`techniques/nli-and-claim-grounding.md`](techniques/nli-and-claim-grounding.md).

Critério CI: taxa de FP em referência aceitável com `embedding_e_juiz` &lt; 15% no fixture `tests/fixtures/policy_validation_run/` (`reference_type: answer_lists`, `gold_correto` booleano).

Para datasets **`reference_type: lexical`** (ex. FairytaleQA), tanto `validate_embedding_policy.py` como `sweep_embedding_threshold.py` usam overlap léxico (F1/EM) como referência aceitável — não `gold_correto`, que é sempre `null`. Com `reference_type: none`, o critério P0 é N/A (`criterio_p0.aplicavel: false`) e o sweep recusa-se a correr, em vez de devolver uma tabela de zeros.

## Evidência

Agregados versionados em [`evidencia/`](evidencia/README.md) — sem PII, publicados por `scripts/publish_run_evidence.py`. Os `predictions.jsonl` completos ficam locais (`outputs/`, gitignored).

## Limiares operacionais (YAML)

```yaml
operacional:
  fila_min_score_recuperacao: 0.5   # recusas na fila humana
  gap_min_score_recuperacao: 0.5    # gap RAG forte × F1 fraco
  gap_max_f1_token: 0.15
```

A fila humana usa os mesmos vereditos que `judge_aggregation_verdicts` (não inclui `incompleto` por omissão em RAG pt-BR).

## Limitação

Coseno ≠ NLI. Itens com F1 baixo e juiz `sustentado` podem não aparecer no detector — usar `analise_manual/fila_revisao_humana.csv` e aba **Revisão humana** no dashboard.

# Calibração da política `embedding_e_juiz`

## Problema

Com `qualquer_critico`, embedding baixo + juiz `sustentado` gera **falso positivo** em gold-correto: o modelo respondeu bem face ao contexto, mas o coseno ficou abaixo do limiar.

## Mitigação (P0)

- **Agregação:** `embedding_e_juiz` — anomalia só se embedding baixo **e** juiz negativo (tiers de agregação em `judge_aggregation_verdicts`, sem `incompleto` por defeito em RAG pt-BR).
- **Limiar:** `verification.embedding_min_cosine` (ex. `0.28` em FairytaleQA); calibrar com `scripts/validate_embedding_policy.py`.

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

Gera `embedding_sweep.csv` e `.json` com FP/FN por limiar 0.20–0.45. Use a tabela para justificar `embedding_min_cosine` no YAML.

Critério CI: taxa de FP em referência aceitável com `embedding_e_juiz` &lt; 15% no fixture `tests/fixtures/policy_validation_run/` (`reference_type: answer_lists`, `gold_correto` booleano).

Para datasets **`reference_type: lexical`** (ex. FairytaleQA), o script usa overlap léxico (F1/EM) como referência aceitável — não `gold_correto`, que é sempre `null`. Com `reference_type: none`, o critério P0 é N/A (`criterio_p0.aplicavel: false`).

## Evidência

Corridas de referência e CSVs de fila humana: `docs/evidencia/` (não versionar outputs grandes; copiar manifestos e sumários relevantes).

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

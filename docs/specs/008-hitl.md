# SPEC-008 — HITL (referência humana amostral)

## Âmbito

Rótulos humanos em amostra (`meta.adjudicacao_humana`) → `sumario_hitl`. **Não** substitui `sumario_lexical` nem o detector.

## Três planos métricos

| Plano | Bloco summary |
|-------|----------------|
| A Produto | `sumario_lexical` |
| B Risco | `sumario_operacional`, `n_anomalias_marcadas` |
| C HITL | `sumario_hitl` |

## Artefactos

- `analise_manual/adjudicacoes_hitl.csv`
- `analise_manual/hitl_manifest.json`

## CLI

```bash
uv run llm-eval --apply-hitl adjudicacoes.csv --resume outputs/run_<id>
uv run python scripts/merge_human_labels.py outputs/run_<id> adjudicacoes.csv
```

## Critérios de aceitação

- Merge idempotente por `id_item`
- `sumario_hitl` com κ e confusão detector/juiz vs humano quando N≥10 rotulados
- Dashboard toggle Pós-HITL lê `sumario_hitl` sem recalcular κ no UI

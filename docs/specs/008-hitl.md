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

## Guardas contra número degenerado

Uma amostra rotulada pode produzir números plausíveis e vazios de conteúdo. Duas
situações são recusadas explicitamente em vez de devolverem um valor:

| Situação | Comportamento |
|----------|---------------|
| `N < 10` rotulados | Bloco sai com `n_itens_rotulados`, `distribuicao_rotulos` e `metricas_omitidas`; sem confusão nem κ |
| Verdade humana com **uma só classe** | `kappa: null` e `kappa_indefinido` com o motivo |

A segunda importa mais do que parece. Com todos os rótulos em `correto`, a confusão
fica `vp = fn = 0` e `cohen_kappa` devolve **0,0** — que se lê como «o detector não
concorda para além do acaso» quando o que aconteceu foi não haver classe com que
concordar. É a armadilha que `references/statistics.md` chama *the constant rater*, e
é exactamente o estado do fixture `tests/fixtures/hitl_fairytale_sample/` (6 rótulos,
todos `correto`).

`distribuicao_rotulos` é publicada **sempre**, mesmo quando as métricas não são: é o
que torna a degenerescência visível sem ter de se abrir o CSV.

## Critérios de aceitação

- Merge idempotente por `id_item`
- `sumario_hitl` com κ e confusão detector/juiz vs humano quando N≥10 rotulados —
  gate implementado em `MIN_ROTULOS_PARA_METRICAS`
- κ recusado com motivo quando a verdade humana tem uma só classe
- `distribuicao_rotulos` presente em todos os casos
- Dashboard toggle Pós-HITL lê `sumario_hitl` sem recalcular κ no UI

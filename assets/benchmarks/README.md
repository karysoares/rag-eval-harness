# Comparativos versionados

Snapshot agregado para o README e contratos de CI. Regenerar a partir de corridas locais (`outputs/`, gitignored).

## Quatro eixos (não misturar)

| Eixo | Planos KPI | Pergunta | Chave em `comparatives.json` |
|------|------------|----------|------------------------------|
| **Interno** | A + B | Evolução de config no mesmo corpus | `interno_fairytale_evolution`, `referencia_tuned_n1025` |
| **Externo** | B | Harness vs RAGAS na mesma amostra | `externo_ragas_amostra` |
| **Calibração P0** | B | `embedding_e_juiz` vs `qualquer_critico` | `calibracao_p0` |
| **HITL** | C | Concordância com revisor humano | `hitl_amostra` |

Metadados dos eixos: campo `eixos` no JSON (`schema_version` ≥ 1.1).

## Regenerar

Sem API (harness + P0 + HITL):

```bash
uv run python scripts/export_comparatives.py
```

Incluir RAGAS (requer `OPENAI_API_KEY`, ~25 itens):

```bash
uv run python scripts/export_comparatives.py --ragas --ragas-n 25
```

Golden HITL versionado: [`tests/fixtures/hitl_fairytale_sample/`](../../tests/fixtures/hitl_fairytale_sample/).

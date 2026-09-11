# Comparativos versionados

Só entram comparativos que alguém com uma clonagem limpa consiga verificar — hoje, o
caso de política que sai da fixture versionada `tests/fixtures/policy_validation_run/`.

Sete entradas foram removidas na versão 2.0 do esquema: as corridas que as sustentavam
(`run_20260517*`, `run_20260518*`, `run_20260606T121845Z`) já não existem, pelo que os
números eram medidos mas irreverificáveis. O bloco `removidos` do JSON registra o que
saiu e porquê. Os agregados com proveniência vivem agora em
[`docs/evidencia/`](../../docs/evidencia/README.md).

Cada número carrega `proveniencia.caminho` e `proveniencia.versionado`: um directório em
`outputs/` (gitignored) serve para quem o correu, não para terceiros.

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

# SPEC-009 — Explicabilidade do harness

## Âmbito

Explicar alertas, recuperação, juiz e padrões — **não** SHAP/attention do LLM gerador.

## Contrato `meta.explicacao`

Gerado em `explainability.build_explicacao` no pipeline.

Campos: `alerta`, `recuperacao`, `lexical`, `juiz`, `padrao_primario`, `rationale_padroes`, `conflitos`.

## Agregado

`sumario_explicabilidade` no `summary.json`.

## UI

Inspector → painel «Porquê?»; fila CSV coluna `explicacao_resumida`.

## Export

```bash
uv run python scripts/export_explainability_pack.py outputs/run_<id>
```

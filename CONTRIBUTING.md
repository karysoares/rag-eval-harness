# Contributing to rag-eval-harness

Obrigado pelo interesse em contribuir. Este repositório é um harness de avaliação RAG — mudanças devem manter separação entre **adaptador**, **sistema sob teste** e **harness de medição** (ver README, secção «O que é»).

## Antes de abrir um PR

1. **Abra uma issue** para discutir a mudança (bug, feature ou contrato de artefactos) — especialmente se alterar métricas, agregação ou schemas de `summary.json` / `manifest.json`.
2. Revise o fluxo no README ([Arquitetura da pipeline](README.md#arquitetura-da-pipeline)) e o código em `src/llm_evaluation/`.
3. Novo dataset → adaptador em `src/llm_evaluation/adapters/` + config YAML de exemplo em `configs/`.

Documentação técnica em [`docs/`](docs/) (premissas, specs, checklist de release). Ver [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) para o fluxo interno.

## Ambiente local

```bash
uv sync --extra dev --extra dashboard
cp .env.example .env   # só para corridas com API real; CI e smoke mock não precisam
```

## Testes e qualidade

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Smoke offline (sem API, usa `configs/smoke_amostra.yaml` com LLM mockado):

```bash
uv run pytest tests/test_pipeline_e2e_mock.py -q
```

Integração com Hub/API (opcional, local):

```bash
RUN_INTEGRATION=1 uv run pytest tests/integration -q
```

## Segredos e artefactos

Não commite `.env`, chaves API nem `outputs/`. Use `.env.example` como modelo. Corridas geram artefactos em `outputs/run_*` (gitignored).

## Nome do pacote Python

O nome PyPI/projeto é **`rag-eval-harness`**, mas o import Python permanece `llm_evaluation` (evita refactor massivo). Os entrypoints CLI (`llm-eval`, `llm-eval-dashboard`) mantêm-se estáveis.

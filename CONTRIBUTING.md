# Contributing

Obrigado pelo interesse em contribuir. Este repositório é um harness de avaliação RAG — mudanças devem preservar a separação entre **adaptador**, **sistema sob teste** e **harness de medição** (ver [Overview](README.md#overview)).

## Before opening a PR

1. **Open an issue** para discutir mudanças que alterem métricas, agregação ou schemas de `summary.json` / `manifest.json`.
2. Leia [Architecture](README.md#architecture) e [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
3. Novo dataset → adaptador em `src/llm_evaluation/adapters/` + config de exemplo em `configs/` + spec em `docs/specs/adapters/`.

Documentação técnica: [`docs/README.md`](docs/README.md) · checklist de release: [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Development setup

**Requisitos:** Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra dashboard
cp .env.example .env   # só para corridas com API; CI e smoke mock não precisam
```

## Running tests

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Smoke offline (sem API):

```bash
uv run pytest tests/test_pipeline_e2e_mock.py -q
```

Integração com Hub/API (opcional):

```bash
RUN_INTEGRATION=1 uv run pytest tests/integration -q
```

## Secrets and artifacts

Não commite `.env`, chaves API nem `outputs/`. Corridas geram artefactos em `outputs/run_*` (gitignored). Use `.env.example` como modelo.

## Package naming

O projeto chama-se **rag-eval-harness**; o import Python é `llm_evaluation`. Os entrypoints CLI (`llm-eval`, `llm-eval-dashboard`) são estáveis.

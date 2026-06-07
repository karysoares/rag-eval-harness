# Contributing

Obrigado pelo interesse em contribuir. Este repositório é um harness de avaliação RAG — mudanças devem preservar a separação entre **adaptador**, **sistema sob teste** e **harness de medição** (ver [Overview](README.md#overview)).

## Before opening a PR

1. **Open an issue** para discutir mudanças que alterem métricas, agregação ou schemas de `summary.json` / `manifest.json`.
2. Leia [Architecture](README.md#architecture) e explore `src/llm_evaluation/`.
3. Novo dataset → adaptador em `src/llm_evaluation/adapters/` + config de exemplo em `configs/`.

## Development setup

**Requisitos:** Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra dashboard
cp .env.example .env   # só para corridas com API; CI e smoke mock não precisam
```

O `uv.lock` está versionado para instalações reprodutíveis.

## Running tests

Os mesmos gates do CI:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest --cov=llm_evaluation --cov-fail-under=75 -q
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

Não commite `.env`, chaves API, `outputs/`, `assets/evidencia/` nem `docs/` (documentação interna, gitignored). Use `.env.example` como modelo.

## Package naming

O projeto chama-se **rag-eval-harness**; o import Python é `llm_evaluation`. Os entrypoints CLI (`llm-eval`, `llm-eval-dashboard`) são estáveis.

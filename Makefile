# Atalhos para os fluxos documentados. Sem lógica própria: cada alvo é o comando
# que o README e o CLAUDE.md já mandam correr, num sítio onde não se esquece.
.DEFAULT_GOAL := help
.PHONY: help setup gates lint format types test cov smoke eval-ci audit evidence bench clean

help:  ## Lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Instala dependências a partir do uv.lock versionado
	uv sync --extra dev --extra dashboard

gates: lint format types test  ## Os quatro portões: uma alteração só está pronta com os quatro verdes

lint:  ## ruff check
	uv run ruff check .

format:  ## ruff format --check
	uv run ruff format --check .

types:  ## mypy strict
	uv run mypy src

test:  ## suite completa
	uv run pytest -q

cov:  ## suite com o portão de cobertura do CI
	uv run pytest --cov=llm_evaluation --cov-report=term-missing --cov-fail-under=80

smoke:  ## pipeline ponta a ponta offline, sem API
	uv run pytest tests/test_pipeline_e2e_mock.py -q

eval-ci:  ## corrida real offline: artefactos + auditoria strict + gate de KPI
	uv run pytest tests/test_ci_corrida_real_offline.py -q

audit:  ## audita os artefactos das corridas em outputs/
	uv run python scripts/audit_run.py outputs --strict

evidence:  ## regenera os comparativos versionados a partir das corridas locais
	uv run python scripts/export_comparatives.py

bench:  ## mede concorrência e cache de embeddings (sem API)
	uv run python scripts/bench_concurrency.py --saida docs/evidencia/bench_concorrencia.json

clean:  ## remove caches de ferramentas
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

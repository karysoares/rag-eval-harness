# Checklist de release (gate “pronto a correr”)

Antes de considerar um marco fechado:

1. [ ] `uv run ruff check .` e `uv run ruff format --check .` passam.
2. [ ] `uv run mypy src` passa.
3. [ ] `uv run pytest` passa (sem `RUN_INTEGRATION=1` no CI).
4. [ ] `.env` não está no repositório; verificação manual por `git status` / `git grep`.
5. [ ] Quickstart do README reproduzível: clone → `uv sync` → `configs/smoke_amostra.yaml` ou `configs/default.yaml` (amostra pequena).
6. [ ] `uv run python scripts/validate_embedding_policy.py tests/fixtures/policy_validation_run` passa (gate FP embedding).
7. [ ] Corrida de referência com `protocolo_ativo` completo no `summary.json` e `analise_manual/fila_revisao_humana.csv` gerado.
8. [ ] Specs em `docs/specs/` reflectem o comportamento merged.
9. [ ] `uv sync --extra dashboard` + import do dashboard (sem subir servidor).
10. [ ] `docs/SECURITY.md` e `.env.example` atualizados se novas variáveis ou riscos.
11. [ ] `docs/calibracao_embedding.md` e evidência em `docs/evidencia/` actualizados se mudou limiar ou política.

Comandos exatos também constam do README raiz.

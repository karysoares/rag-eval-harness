#!/usr/bin/env bash
# Corrida FairytaleQA validation 100% (configs/ptbr_fairytale_full.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]] || ! grep -qE '^OPENAI_API_KEY=.+' .env 2>/dev/null; then
  echo "Defina OPENAI_API_KEY em .env (cp .env.example .env)" >&2
  exit 1
fi

echo "==> Contagem de itens (dry-run)"
uv run llm-eval --config configs/ptbr_fairytale_full.yaml --dry-run

if [[ -n "${1:-}" ]]; then
  echo "==> Retomar em $1"
  exec uv run llm-eval --config configs/ptbr_fairytale_full.yaml --resume "$1"
fi

echo "==> Iniciar corrida completa (Ctrl+C seguro; retome com: $0 outputs/run_<id>)"
exec uv run llm-eval --config configs/ptbr_fairytale_full.yaml

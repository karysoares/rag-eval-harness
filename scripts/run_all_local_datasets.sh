#!/usr/bin/env bash
# Run distinct dataset configs sequentially (API-friendly) while another eval may be active.
# Skips ptbr_fairytale_tuned / full and other duplicate full-FairytaleQA variants.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${ROOT}/outputs/_batch_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="${LOG_DIR}/batch_${STAMP}.log"

if [[ ! -f .env ]] || ! grep -qE '^OPENAI_API_KEY=.+' .env 2>/dev/null; then
  echo "OPENAI_API_KEY missing in .env — aborting batch." | tee -a "$MASTER_LOG"
  exit 1
fi

# name|config path
CONFIGS=(
  "smoke_amostra|configs/smoke_amostra.yaml"
  "default_32|configs/default.yaml"
  "ptbr_fairytale_64|configs/ptbr_fairytale.yaml"
)

DELAY_SEC="${BATCH_LAUNCH_DELAY_SEC:-45}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "Batch start (delay between runs: ${DELAY_SEC}s)"
log "Master log: $MASTER_LOG"

for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"
  cfg="${entry#*|}"
  run_log="${LOG_DIR}/${label}_${STAMP}.log"
  log "==> Starting $label ($cfg)"
  log "    stdout/stderr: $run_log"
  if uv run llm-eval --config "$cfg" >>"$run_log" 2>&1; then
    log "==> Finished $label OK"
  else
    log "==> Finished $label FAILED (see $run_log)"
  fi
  log "    sleep ${DELAY_SEC}s before next run"
  sleep "$DELAY_SEC"
done

log "Batch complete."

# ADR 0002 — Três planos métricos (A/B/C)

## Contexto

KPI léxico, detector YAML e rótulos humanos medem coisas diferentes.

## Decisão

- **A** `sumario_lexical` — produto face ao corpus
- **B** `sumario_operacional` + `flag_anomalia` — risco operacional
- **C** `sumario_hitl` — calibração na amostra rotulada

Dashboard expõe toggle; nunca misturar no mesmo número sem rótulo explícito.

## Consequências

HITL não reescreve KPI global; `reprocess_run_dir` recalcula todos os blocos após merge CSV.

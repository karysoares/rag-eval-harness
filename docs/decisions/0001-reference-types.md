# ADR 0001: Tipos de referência (`reference_type`)

## Contexto

O harness mede sinais independentes (embedding, juiz, referência opcional). Cada adaptador declara como rotular “correto vs incorreto” para métricas de confusão e κ — sem impor um único protocolo a todos os corpora.

## Decisão

1. **`lexical`** — referência curta do corpus; confusão/kappa via F1 token (SPEC-007), não substring em listas.
2. **`answer_lists`** — listas `correct`/`incorrect` do adaptador; match por substring bidirecional após normalização leve; recusas tratadas por heurística documentada em `metrics.md`.
3. **`none`** — sem referência automática; KPI operacional = juiz + embedding + política YAML.

Subconjuntos usam `seed` + `limit` no YAML. O `summary.json` regista `tipo_referencia_ativo`.

## Consequências

- Métricas e chaves JSON usam sufixo `_vs_referencia`, não nomes de dataset.
- Rótulos automáticos são aproximados; amostras HITL calibram juiz e agregação.
- ADRs futuros detalham adaptadores concretos em `docs/specs/adapters/`.

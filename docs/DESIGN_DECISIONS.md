# Decisões de desenho — checklist pré-implementação (validadores)

Este documento fecha o *gate* científico antes de alterações relevantes ao pipeline ou às métricas.

1. **Erro e anomalia**: definidos em [`metrics.md`](metrics.md) e ADR 0001 (`reference_type`). Anomalia operacional = disjunção de sinais configuráveis (gold, embedding, judge).
2. **Baselines A/B/hybrid**: Baseline A (sem verificadores), B (um verificador via configs alternativos), híbrido completo — ver `configs/` e `metrics.md`.
3. **Métricas RAG**: proxies documentados em `metrics.md` (max similaridade, rank do chunk gold); RAGAS citado como referência, não reimplementação byte-a-byte.
4. **Judge**: rubrica em `prompts/judge_user_template.txt` + system em `prompts/judge_system.txt`; pede justificativa curta antes do JSON; ver ficha `docs/techniques/llm-as-judge.md`.
5. **Amostragem**: `seed` + `limit` no YAML; modo `demo` para CI sem rede.
6. **Ética / dados**: licenças em `references.md`; não commitar `.env`; política de outputs em `SECURITY.md`.

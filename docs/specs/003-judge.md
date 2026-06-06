# SPEC-003: Juiz LLM (robusto + observabilidade)

- **Estado:** implemented (v0.4 base) + **Fase 1 implemented** (schema, contexto, retries parse, CoT off por omissão); **Fases 2–8 roadmap**
- **Testes:** `tests/test_judge_schema.py`, `tests/test_judge_context.py`, `tests/test_judge_fallback.py`, `tests/test_judge_rag_pt.py`, `tests/test_json_lenient.py`, `tests/test_pipeline_e2e_mock.py`
- **Relacionado:** [SPEC-002](002-grounding.md) (camada embedding paralela), [SPEC-004](004-aggregation.md) (`juiz_negativo`, fallback excluído), [SPEC-005](005-reporting.md) (`sumario_juiz`), [SPEC-006](006-dashboard.md), [SPEC-007](007-pattern-detection.md) (`juiz_fallback`, `juiz_negativo`)

## Objetivo

Avaliar a **aderência da resposta ao contexto recuperado** com veredito estruturado JSON (temperatura 0). O juiz é camada **complementar** à detecção determinística ([SPEC-007](007-pattern-detection.md)) e ao proxy por coseno ([SPEC-002](002-grounding.md)); **não** substitui F1 token nem listas correct/incorrect do adaptador.

**Premissas:** ver `docs/PREMISSAS.md` — não colapsar juiz + embedding num único KPI; fallback heurístico **não** entra na agregação de anomalia.

## Entradas e saídas (comportamento actual)

### Configuração (YAML)

| Chave | Tipo | Default | Efeito |
|-------|------|---------|--------|
| `verification.verify_judge` | bool | por adaptador | Activa camada juiz |
| `verification.judge_prompt_style` | `pt` \| `rag_en` \| `rag_pt` | `pt` / `rag_pt` em RAG | Ficheiros em `prompts/judge_*.txt` |
| `verification.negative_judge_verdicts` | list[str] | dataset | Vereditos que disparam `juiz_negativo` |
| `verification.judge_return_chain_of_thought` | bool | **`false`** | Se `true`, persiste `cadeia_de_pensamento` no JSONL (debug) |
| `verification.judge_max_context_chars` | int \| null | `12000` | Tecto de caracteres do contexto no prompt |
| `verification.judge_max_parse_retries` | int | `2` | Retentativas após falha de parse/schema (além do retry HTTP) |
| `rag.top_k` | int | — | Máximo de chunks passados ao juiz (`build_judge_context`) |
| `llm.timeout_seconds` | float | — | Timeout do cliente juiz (`default_judge_from_env`) |

Sinónimos PT aceites: `devolver_cadeia_pensamento_juiz`, `max_chars_contexto_juiz`, `max_retries_parse_juiz`, `estilo_prompt_juiz`.

**Desligar juiz:** `verify_judge: false` (ex. `configs/nq_open.yaml`); `protocol.apply_protocol_defaults` desliga se `reference_type` lexical/none **e** sem corpus RAG.

### Prompts

| Estilo | System | User template |
|--------|--------|----------------|
| `pt` | `prompts/judge_system.txt` | `prompts/judge_user_template.txt` |
| `rag_en` | `prompts/judge_rag_en_system.txt` | `prompts/judge_rag_en_user_template.txt` |
| `rag_pt` | `prompts/judge_rag_pt_system.txt` | `prompts/judge_rag_pt_user_template.txt` |

Placeholders: `{question}`, `{context}`, `{answer}`. Rubrica RAG EN (v1): grounding vs recusa honesta; resposta curta factual não deve ser `sustentado` se contradiz o contexto.

### Saída por item (`predictions.jsonl`)

**`sinais.juiz`** (via `aggregate.signals_to_dict`):

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `veredito` | string | Enum canónico PT (após `veredito.normalizar_veredito`) |
| `motivo_breve` | string | Justificação curta (≤ 500 chars após validação) |
| `confianca` | float | \[0, 1\] |
| `fallback_heuristico` | bool | Presente se `heuristic_judge_json` foi usado |
| `cadeia_de_pensamento` | list[str] | **Omitido** por omissão (`judge_return_chain_of_thought: false`) |
| `schema_version` | string | Versão do contrato (ex. `"1.0"`) no `raw` interno; pode não serializar todos os extras |

**`sinais.juiz_negativo`:** `true` \| `false` \| `null` — `null` quando fallback heurístico (não conta na agregação).

**`meta.contexto_juiz`** (auditoria Fase 1):

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `retry_count` | int | Retentativas de parse/API no juiz |
| `parse_failures` | int | Falhas de parse/schema antes do sucesso ou fallback |
| `schema_invalid` | bool | Houve violação de schema ou fallback |
| `schema_version` | string | `JUDGE_SCHEMA_VERSION` |
| `chunk_ids` | list[int] | Índices 1-based dos chunks no prompt |
| `n_chunks_usados` | int | Chunks efectivamente incluídos |
| `n_chunks_total` | int | Chunks recuperados antes do corte |
| `tokens_estimados` | int | Heurística ~len/4 sobre o contexto |
| `truncado` | bool | Corte por `top_k` ou `judge_max_context_chars` |

### Saída agregada (`summary.json` → `sumario_juiz`)

| Campo | Definição |
|-------|-----------|
| `n_itens_com_juiz` | Itens com `sinais.juiz` presente |
| `taxa_fallback_heuristico` | Fração com `fallback_heuristico` |
| `taxa_schema_invalido` | Fração com `schema_invalid` em `contexto_juiz` |
| `taxa_com_retry` | Fração com `retry_count > 0` |
| `media_retry_count` | Média de `retry_count` |
| `media_parse_failures` | Média de `parse_failures` |
| `media_tokens_contexto_estimados` | Média de tokens estimados do contexto |
| `taxa_contexto_truncado` | Fração com `truncado: true` |

Omitido quando `n_itens_com_juiz == 0`.

## Pipeline (implementação actual)

```
verify_item → build_judge_context(retrieved, top_k, max_chars)
           → run_judge_for_retrieved → OpenAiCompatibleClient (temp 0, retry HTTP 429/5xx)
           → parse_judge_json (leniente)
           → validate_judge_response (judge_schema.py)
           → JudgeResult + meta.contexto_juiz
```

**Ficheiros:** `verification/judge.py`, `verification/judge_schema.py`, `verification/judge_context.py`, `llm_client.py`, `veredito.py`, `pipeline.verify_item`, `verification/aggregate.py`.

### Vereditos canónicos

`sustentado` | `nao_sustentado` | `contradicacao` | `incompleto` | `inseguro`

Aliases EN aceites na normalização (`supported`, `unsupported`, …). Enum fechado em `judge_schema.VERDICTS_CANONICOS`.

### Fallback heurístico

`llm_client.heuristic_judge_json`: política conservadora (vazio, muito curto, recusa explícita). Marcado com `fallback_heuristico: true`. **Excluído** de `_judge_negative_for_aggregation` e de `juiz_negativo` na agregação ([SPEC-004](004-aggregation.md)). Visível no dashboard (badge) e padrão `juiz_fallback` ([SPEC-007](007-pattern-detection.md)).

### Contexto enviado ao juiz

1. Ordem: rank de retrieval (lista `retrieved` já ordenada por score).
2. Formato: `[1] texto\n\n[2] texto…`
3. Limite de chunks: `min(len(retrieved), rag.top_k)`.
4. Limite de caracteres: `verification.judge_max_context_chars` (truncagem com `…` no último bloco).
5. Contexto vazio: string `(vazio)` no prompt.

**Não incluído no contexto do juiz:** passagem `rag_gold_chunk` isolada (só via chunk recuperado se estiver no top-k). Diferente do ramo embedding ([SPEC-002](002-grounding.md)).

### Retries

| Camada | Comportamento |
|--------|----------------|
| HTTP | `OpenAiCompatibleClient`: 429/5xx/transporte, backoff `(1s, 3s)`, `max_retries` default 2 |
| Parse/schema | `run_judge`: até `judge_max_parse_retries + 1` tentativas; retry acrescenta sufixo JSON estrito; 2ª+ tentativa usa `response_format: json_object` quando cliente é `OpenAiCompatibleClient` |
| Fallback final | `heuristic_judge_json` após esgotar tentativas |

### Agregação e anomalia

- `juiz_negativo` ⇔ veredito ∈ `negative_judge_verdicts` **e** não fallback.
- Política `embedding_e_juiz`: exige embedding baixo **e** juiz negativo ([SPEC-004](004-aggregation.md)).
- **Não** misturar taxa de `juiz_negativo` com acurácia factual gold sem `analise_camadas`.

### Dashboard

- Colunas por item: `veredito_juiz`, `juiz_negativo`, `juiz_confianca`, `juiz_fallback`, `juiz_retry_count`, `juiz_parse_failures`, `juiz_schema_invalid`, `juiz_tokens_contexto` (`dashboard/data.py`).
- Agregados: `summarize_judge_from_records()` / `summary.json` → `sumario_juiz`.
- UI: expander CoT só relevante se JSONL antigo ou `judge_return_chain_of_thought: true`.

---

## Roadmap: Fases 1–8 e itens 1–23

Legenda por item: **Impl.** = implementado neste repositório; **Prog.** = em curso / parcial; **Plan.** = especificado apenas.

### Fase 1 — Robustez estrutural (prioridade máxima)

| # | Item | Objectivo | Tarefas | Métricas dashboard / summary | Dependências | Estado |
|---|------|-----------|---------|------------------------------|--------------|--------|
| 1 | Schema versionado | Contrato JSON explícito e versionado | `judge_schema.py`, `JUDGE_SCHEMA_VERSION`, `validate_judge_response` | `schema_version` em `contexto_juiz` | — | **Impl.** |
| 2 | Enum fechado de vereditos | Evitar vereditos livres | `VERDICTS_CANONICOS` + `normalizar_veredito` | % inválidos → `taxa_schema_invalido` | #1 | **Impl.** |
| 3 | `confianca` \[0,1\] | Campo numérico validado | Rejeição / clamp documentado | — | #1 | **Impl.** |
| 4 | `motivo_breve` limitado | Evitar raciocínio longo em `motivo` | `MOTIVO_BREVE_MAX_LEN=500`, `sanitize_motivo_breve` | — | #1 | **Impl.** |
| 5 | Flag `fallback_heuristico` | Distinguir juiz real vs heurística | Validação bool; agregação ignora | `taxa_fallback_heuristico` | [SPEC-004] | **Impl.** |
| 6 | CoT off por omissão | Não persistir chain-of-thought | `judge_return_chain_of_thought: false` | — | #7–8 | **Impl.** |
| 7 | Sanitizar saída | Não gravar CoT no `JudgeResult.raw` default | `run_judge(return_chain_of_thought=…)` | — | #6 | **Impl.** |
| 8 | Prompts sem CoT longo | Reduzir tokens e vazamento de raciocínio | Actualizar `judge_*.txt` | — | #6 | **Impl.** |
| 9 | Retry parse/JSON | Recuperar respostas malformadas | Loop + sufixo + `json_object` | `taxa_com_retry`, `media_retry_count` | `llm_client` | **Impl.** |
| 10 | Retry HTTP alinhado | Timeouts/429/5xx | `OpenAiCompatibleClient` (já existente) | observabilidade LLM | — | **Impl.** |
| 11 | `meta.contexto_juiz` | Auditoria por item | `retry_count`, `parse_failures`, … | colunas dashboard | #9 | **Impl.** |
| 12 | Contexto formalizado | Chunks, ordem, truncagem | `judge_context.build_judge_context` | `tokens_estimados`, `taxa_contexto_truncado` | [SPEC-001] `top_k` | **Impl.** |
| 13 | `sumario_juiz` | Agregados de robustez | `reporting._judge_summary` | bloco `summary.json` | #11 | **Impl.** |
| 14 | Testes Fase 1 | Regressão | `test_judge_schema`, `test_judge_context`, extensões fallback | CI pytest | #1–12 | **Impl.** |

### Fase 2 — Juiz multidimensional (grounding vs qualidade)

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| 15 | Dimensão `grounding` | Veredito só ancoragem | `judge_mode` ou segundo prompt; schema 2.x | taxas por dimensão | Fase 1 | **Plan.** |
| 16 | Dimensão `quality` | Clareza/relevância separada | Prompt + campo opcional | não colapsar em KPI único | #15 | **Plan.** |
| 17 | Agregação documentada | Políticas por dimensão | [SPEC-004](004-aggregation.md) extensão | `analise_camadas` por dimensão | #15–16 | **Plan.** |

### Fase 3 — Calibração humana

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| 18 | Amostra rotulada | Cohen's κ juiz vs humano | Export JSONL + notebook | κ, matriz confusão | Fase 1–2 | **Plan.** |
| 19 | Limiar de `confianca` | Abstenção / revisão humana | Config + dashboard filter | % abaixo limiar | #18 | **Plan.** |

### Fase 4 — Versionamento de rubrica

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| 20 | `rubric_id` / `prompt_hash` | Reprodutibilidade | Gravar em `meta.contexto_juiz` | comparador de corridas | Fase 1 | **Plan.** |
| 21 | A/B de prompts | Experiências controladas | Config `judge_prompt_variant` | por variante | #20 | **Plan.** |

### Fase 5 — Ensemble / segundo juiz

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| 22 | Dois juízes paralelos | Reduzir variância | Segunda chamada + voto | discordância % | Fase 2, custo | **Plan.** |

### Fase 6 — Custo e latência

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| 23 | Cache por hash (q,ctx,a) | Evitar re-julgar | LRU opcional | tokens juiz / item | observabilidade | **Plan.** |

### Fase 7 — Locale e cross-lingual

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| — | Rubricas por idioma | PT/EN alinhadas | Já `rag_en` / `rag_pt`; estender QA aberto | por `judge_prompt_style` | — | **Prog.** (parcial) |

### Fase 8 — Integração produção

| # | Item | Objectivo | Tarefas | Métricas | Dependências | Estado |
|---|------|-----------|---------|----------|--------------|--------|
| — | Export OpenTelemetry / Langfuse | Traces de juiz | Hooks em `TrackingLlmClient` | latência p95 juiz | `observability.py` | **Plan.** |

*(Itens 15–23 do quadro global mapeiam para as linhas numeradas acima; Fases 7–8 usam itens temáticos sem número fixo no código.)*

---

## Casos limite

| Situação | Comportamento |
|----------|----------------|
| `verify_judge: false` | Sem `sinais.juiz`; sem `contexto_juiz` |
| Sem RAG / `retrieved == []` | Contexto `(vazio)`; juiz ainda pode marcar `incompleto` / `nao_sustentado` |
| API falha após retries | Fallback heurístico; `juiz_negativo = null` |
| JSON com veredito EN | Normalizado para PT |
| JSONL antigo com CoT | Dashboard pode mostrar expander; novas corridas omitem CoT por defeito |
| `judge_max_context_chars: null` | Sem tecto de caracteres (só `top_k`) |

## Fora de âmbito (mantém-se)

- Dois juízes na mesma corrida sem spec Fase 5.
- Calibração humana sistemática (Fase 3).
- NLI / claim-level (ver `docs/techniques/nli-and-claim-grounding.md`).
- KPI único “score do juiz” misturado com embedding ou gold.

## Critérios de aceitação

### v0.4 (base)

- [x] E2E mock verifica chamada ao juiz quando `verify_judge: true`.
- [x] Corrida RAG usa prompts EN/PT quando configurado.
- [x] Fallback heurístico visível no JSONL e dashboard.
- [x] Com `embedding_e_juiz`, juiz negativo contribui para anomalia (sem fallback).

### Fase 1

- [x] `validate_judge_response` coberto por testes.
- [x] CoT omitido por defeito no JSONL.
- [x] Retry parse com `retry_count` em `meta.contexto_juiz`.
- [x] `sumario_juiz` em `summary.json`.
- [x] Colunas de robustez no `dashboard/data.py`.
- [x] `uv run pytest -q` verde.

### Fases 2–8

- [ ] Schema 2.x multidimensional.
- [ ] Rotulagem humana + κ reportado.
- [ ] `rubric_id` / A/B prompts.
- [ ] Segundo juiz ou ensemble.
- [ ] Cache de julgamentos.
- [ ] Traces externos (Langfuse/OTel).

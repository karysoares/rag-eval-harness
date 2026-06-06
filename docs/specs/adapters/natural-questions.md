# SPEC-A-NQ: Adaptador Natural Questions

- **Estado:** implemented (dois protocolos YAML)
- **Testes:** `tests/test_eval_items_load.py`, `tests/test_adapters_natural_questions.py`

## Objetivo

Carregar Natural Questions Open como dataset principal, com **dois modos de corrida** explícitos: QA aberta (sem passagem) e RAG com passagem gold.

## Variantes Hub

| Alias YAML | Repositório HF | Passagem | Config |
|------------|----------------|----------|--------|
| `nq_open` | `google-research-datasets/nq_open` | Não | [`configs/nq_open.yaml`](../../../configs/nq_open.yaml) |
| `nq_open_gold` | `florin-hf/nq_open_gold` | `text` (Wikipedia gold) | [`configs/nq_open_rag.yaml`](../../../configs/nq_open_rag.yaml) |

## Entradas e saídas

### NQ-Open (QA aberta)

- **Colunas:** `question`, `answer` (string ou lista).
- **reference_type:** `lexical`
- **EvalItem:** `correct_answers` (todas as variantes da lista), `rag_gold_chunk=None`.
- **RAG:** `enabled: false`
- **Verificação:** `verify_embedding: false`, `verify_judge: false`, `verify_gold: false`
- **Geração:** `estilo_prompt: open_en` (few-shot, sem placeholders)
- **KPI:** `sumario_lexical` + `f1_token` / `em_squad`

### NQ-Open Gold (RAG + grounding)

- **Colunas:** `question`, `answers[]`, `text`, `example_id`
- **reference_type:** `lexical`
- **EvalItem:** múltiplas referências; `rag_gold_chunk=text`
- **RAG:** `enabled: true`
- **Verificação:** embedding + juiz; `aggregation.policy: embedding_e_juiz` (SPEC-004)
- **Geração:** `estilo_prompt: rag_en`
- **KPI:** léxico + `sumario_recuperacao` + anomalias calibradas

## Comportamento do loader

- `resolve_nq_hf_repo`: aliases `nq_open`, `nq_open_gold`, `natural_questions`.
- `example_id` mapeado para `EvalItem.id` quando presente.
- Shuffle com `seed` do YAML antes de `limit`.

## Validação (`protocol.py`)

- QA aberta: não permitir RAG/embedding/juiz sem corpus.
- RAG: exigir corpus nos itens; auto-ajuste registado em `summary.protocolo_ajustado`.

## Fora de âmbito

- Indexação Wikipedia completa offline.
- FEVER / claim verification.

## Critérios de aceitação

- [x] `nq_open.yaml` — 50 itens, QA aberta, sem anomalias espúrias.
- [x] `nq_open_rag.yaml` — 50 itens com passagem e retrieval 100% chunk ouro no top-k (diagnóstico).
- [ ] Pós v0.4: anomalias RAG &lt; 30% com `embedding_e_juiz` e embedding vs gold chunk.
- [x] `configs/default.yaml` alinhado a `nq_open.yaml`.

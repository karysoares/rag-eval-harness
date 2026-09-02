# SPEC-001: Métricas de recuperação

- **Estado:** implemented
- **Testes:** `tests/test_retrieval_metrics.py`, `tests/test_retrieval.py`, `tests/test_pattern_detection.py` (`recuperacao_falhou`), `tests/test_pipeline_e2e_mock.py` (gate fraca)

## Objetivo

Medir qualidade do retriever **antes** da geração, sem misturar com anomalia agregada por defeito (`docs/PREMISSAS.md`: ramo de recuperação é **diagnóstico**; não alimenta `flag_anomalia` excepto via gate de geração — ver [SPEC-004](004-aggregation.md)).

## Entradas e saídas

- **Entrada:** `EvalItem`, `list[RetrievedChunk]`, `rag_enabled` (`cfg.rag.enabled`).
- **Implementação:** `retrieval_metrics.compute_retrieval_metrics`, `retrieval.Retriever` + `cosine_topk`.
- **Saída por item:** `meta.metricas_recuperacao` em `predictions.jsonl`.
- **Saída agregada:** `summary.json` → `sumario_recuperacao` (só se existir ≥1 item com `rag_ativo: true`).

### Campos por item (`metricas_recuperacao`)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `rag_ativo` | bool | `true` quando `rag.enabled`; caso contrário único campo presente |
| `n_chunks_recuperados` | int | `len(retrieved)` (só com RAG activo) |
| `score_melhor_chunk` | float \| null | Score do chunk na posição 1 da lista ordenada |
| `rank_chunk_ouro` | int \| null | Posição **1-indexed** do primeiro chunk com `is_gold` |
| `chunk_ouro_no_top_k` | bool | `rank_chunk_ouro is not None` |
| `corpus_tem_chunk_ouro` | bool | `bool((item.rag_gold_chunk or "").strip())` |

## Semântica dos scores

- **Definição:** similaridade de **cosseno** entre o embedding da pergunta e o de cada chunk do corpus, calculada como produto escalar `doc_vecs @ query_vec` em `retrieval.cosine_topk` (vectores **L2-normalizados**).
- **Backends:** `sentence_transformers` usa `normalize_embeddings=True`; `hash` (CI/testes) também normaliza cada vector pseudo-aleatório. Não há ranker externo nem BM25 neste ramo.
- **Intervalo teórico:** \([-1, 1]\). Na prática, com MiniLM e textos curtos, scores positivos dominam; limiares empíricos (ex. `0,25` em `configs/nq_open_rag.yaml`) calibram-se por backend/modelo.
- **Comparabilidade entre corridas:** válida só com o **mesmo** `embeddings.backend`, `embeddings.model_name`, política de chunking (`chunk_max_chars`) e corpus por item. Mudar modelo ou corpus invalida comparação directa de `score_melhor_chunk`.
- **Relação com o gate:** `pipeline.is_weak_retrieval` usa o **mesmo** valor que `score_melhor_chunk` (`retrieved[0].score`). Com `rag.min_score_recuperacao` / `min_retrieval_score` definido: recuperação **fraca** se `top < limiar` (estrito). Com lista vazia e gate activo: fraca com `top_score = null`. Com limiar `null`: gate desligado (nunca fraca por score).

## Métricas agregadas (`sumario_recuperacao`)

Calculadas em `reporting.summarize()` → `_retrieval_summary()`: percorre todos os `RunRecord`, considera apenas entradas com `metricas_recuperacao.rag_ativo == true` (fallback legado: chave `retrieval_metrics`).

| Campo | Definição | Omitido quando |
|-------|-----------|----------------|
| `n_itens_com_rag` | Contagem de itens com `rag_ativo` | Chave `sumario_recuperacao` inteira ausente se `n_itens_com_rag == 0` |
| `media_score_melhor_chunk` | Média aritmética de `score_melhor_chunk` **não nulos** | `null` se nenhum score disponível (ex.: todos os itens com `retrieved == []`) |
| `taxa_chunk_ouro_no_top_k` | `count(chunk_ouro_no_top_k) / n_itens_com_chunk_ouro_no_corpus` | `null` se `n_itens_com_chunk_ouro_no_corpus == 0` |
| `n_itens_com_chunk_ouro_no_corpus` | Itens com `corpus_tem_chunk_ouro == true` | — |
| `media_rank_chunk_ouro_quando_presente` | Média de `rank_chunk_ouro` onde não é `null` (ouro encontrado no top-k) | `null` se nenhum rank disponível |

**Notas:**

- Itens sem chunk ouro no adaptador entram em `n_itens_com_rag` mas **não** no denominador de `taxa_chunk_ouro_no_top_k`.
- Itens com recuperação vazia contam em `n_itens_com_rag`; `score_melhor_chunk` é `null` e não entra na média de scores.
- O bloco **não** entra em `kpi_primario` nem em `analise_camadas` ([SPEC-004](004-aggregation.md)); o dashboard mostra aviso se ausente (`sumario_recuperacao` omitido em corridas sem RAG ou JSONL antigo).

### Contagem relacionada em `qualidade_pipeline` (SPEC-004)

Independente de `sumario_recuperacao`: `n_geracoes_curadas_recuperacao_fraca` conta itens com `meta.qualidade_geracao.curada_por_recuperacao_fraca == true` (gate + `generation.omitir_llm_se_recuperacao_fraca`).

## Comportamento

- Se `rag.enabled: false`, `metricas_recuperacao` é `{"rag_ativo": false}`; `sumario_recuperacao` omitido.
- Métricas são **diagnósticas**; não entram em `anomaly_flag` excepto indiretamente via gate de geração (abaixo).

### Marcação de chunk ouro

Em `Retriever.retrieve`, `is_gold` é `true` quando o texto do chunk e `item.rag_gold_chunk` (após strip) se intersectam por substring (`ouro in chunk` ou `chunk in ouro`). Com `rag.inject_retrieval_failure: true`, chunks ouro são removidos da lista devolvida — métricas reflectem o top-k **pós-injeção**.

### Gate de recuperação fraca (ligação [SPEC-004](004-aggregation.md) / geração)

- YAML: `rag.min_score_recuperacao` (alias `min_retrieval_score`), `generation.omitir_llm_se_recuperacao_fraca` (`skip_llm_on_weak_retrieval`).
- Se activo e recuperação fraca: resposta curada determinística; `meta.qualidade_geracao.curada_por_recuperacao_fraca: true`, opcionalmente `score_melhor_chunk` e `limiar_recuperacao` na mesma estrutura.
- Recomendado em `configs/nq_open_rag.yaml` (`min_score_recuperacao: 0.25`, `omitir_llm_se_recuperacao_fraca: true`) para evitar LLM a recusar com contexto presente mas score baixo.

### Padrão [SPEC-007](007-pattern-detection.md)

- `recuperacao_falhou` em `meta.diagnostico.padroes` quando `rag_ativo`, `corpus_tem_chunk_ouro` e `chunk_ouro_no_top_k == false`.
- Não altera `flag_anomalia`; prioridade alta em `padrao_primario` (ver SPEC-007).

## Casos limite

| Situação | Por item | Agregação |
|----------|----------|-----------|
| RAG desligado | Só `rag_ativo: false` | Sem `sumario_recuperacao` |
| `retrieved == []` | `n_chunks_recuperados: 0`, `score_melhor_chunk: null`, `rank_chunk_ouro: null`, `chunk_ouro_no_top_k: false` | Item conta em `n_itens_com_rag`; excluído da média de scores |
| Sem `rag_gold_chunk` (vazio) | `corpus_tem_chunk_ouro: false`, ranks `null` | Fora do denominador de `taxa_chunk_ouro_no_top_k` |
| Ouro no corpus mas fora do top-k | `rank_chunk_ouro: null`, `chunk_ouro_no_top_k: false` | Conta como falha na taxa; [SPEC-007](007-pattern-detection.md) `recuperacao_falhou` |
| `inject_retrieval_failure` | Ouro ausente da lista mesmo estando no corpus | Igual a “fora do top-k” para métricas e padrão |
| Empate de score no top-k | `cosine_topk` ordena por score decrescente; empates resolvem-se pela ordem de índices do `argpartition`/`argsort` (determinístico para mesma entrada, não “justiça” semântica entre chunks) | — |
| Gate com lista vazia | `is_weak_retrieval` → fraca, `top_score: null` | `curada_por_recuperacao_fraca` se `omitir_llm` activo |
| Gate com `min_score_recuperacao: null` | Nunca fraca por score; LLM corre mesmo com `[]` | — |
| Score `null` no item | Não entra em `media_score_melhor_chunk` | Média sobre subconjunto com score |

## Fora de âmbito

- Recall@k sobre índice Wikipedia completo.
- Latência do indexador.
- Métricas de recuperação como KPI principal de alerta (ver premissas e SPEC-004).

## Critérios de aceitação

- [x] Cada corrida com RAG grava `metricas_recuperacao` por item (`pipeline`, `orchestration/multi`; testes em `test_retrieval_metrics.py`, `test_pipeline_e2e_mock.py`).
- [x] `summarize()` inclui `sumario_recuperacao` quando há itens com `rag_ativo` (implementado em `reporting.py`; sem teste unitário dedicado à chave agregada).
- [x] Testes cobrem rank do chunk ouro, score do top-1 e `rag_ativo: false` (`test_retrieval_metrics.py`); retriever vazio e marcação ouro (`test_retrieval.py`).
- [x] Gate de recuperação fraca em `configs/nq_open_rag.yaml` e e2e com limiar impossível (`test_weak_retrieval_gate_skips_responder_calls`).
- [x] Padrão `recuperacao_falhou` alinhado às métricas (`test_recuperacao_falhou` em `test_pattern_detection.py`).

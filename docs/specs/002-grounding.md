# SPEC-002: Grounding (ancoragem ao contexto)

- **Estado:** implemented (v0.4 — gold chunk, protocolo sem corpus, decomposição recuperados/ouro)
- **Testes:** `tests/test_embedding_verify.py`, `tests/test_verify_embedding_na.py`, `tests/test_recompute_embedding.py`, `tests/test_pattern_detection.py` (`grounding_*`), `tests/test_aggregate.py` (camada embedding)
- **Relacionado:** [SPEC-001](001-retrieval.md) (chunks e embedder partilhados), [SPEC-004](004-aggregation.md) (`embedding_baixo_suporte` na política), [SPEC-007](007-pattern-detection.md) (`grounding_baixo`, `grounding_fp_suspeito`)

## Objetivo
> **Configs não distribuídos.** O repositório público inclui apenas os configs do caso
> de referência FairytaleQA pt-BR. As menções a `configs/nq_open*.yaml`,
> `configs/legacy_truthfulqa.yaml` e `configs/smoke_demo.yaml` descrevem o protocolo com
> que o adaptador foi validado e servem de referência para reconstruir um equivalente —
> ver [`adapters/natural-questions.md`](adapters/natural-questions.md) e
> `configs/default.yaml` como modelo.


Sinalizar respostas com **baixa sobreposição semântica** (proxy por coseno) face ao contexto **utilizável**: chunks **recuperados** no top-k e, opcionalmente, a passagem gold do adaptador (`rag_gold_chunk`). Não é NLI nem RAGAS; é camada diagnóstica reprodutível no harness, independente do adaptador.

**Premissa:** coseno entre embeddings ≠ entailment — ver `docs/techniques/embedding-similarity-vs-semantic-equivalence.md` e `docs/metrics.md`.

## Entradas e saídas

### Configuração (YAML)

| Chave | Tipo | Efeito |
|-------|------|--------|
| `verification.verify_embedding` | bool | Activa cálculo e `embedding_baixo_suporte` na agregação |
| `verification.embedding_min_cosine` | float | Limiar estrito: `baixo_suporte` iff `embedding_max_coseno < limiar` |
| `verification.embedding_use_gold_chunk` | bool (default `true`) | Inclui `rag_gold_chunk` (texto integral, **não** re-chunked) no máximo |
| `rag.enabled` | bool | Sem RAG, grounding fica `null` (mesmo com `verify_embedding: true` até normalização) |
| `rag.top_k` | int | Quantos chunks entram em `recuperados` e no score vs recuperados |
| `rag.chunk_max_chars` | int | Tamanho dos chunks do corpus por item (`build_chunks_for_item`) |
| `embeddings.backend` | `hash` \| `sentence_transformers` | Backend **único** para recuperação e grounding na mesma corrida |
| `embeddings.model_name` | string | Modelo ST quando `backend: sentence_transformers` (ignorado por `hash`) |

Normalização pré-corrida: `protocol.apply_protocol_defaults` desliga `verify_embedding` se `rag.enabled` ou corpus vazio; `protocol.validate_protocol` falha se a combinação incoerente persistir.

### Saída por item (`predictions.jsonl` → `sinais`)

| Campo JSON (PT) | Campo interno | Tipo | Descrição |
|-----------------|---------------|------|-----------|
| `embedding_max_coseno` | `embedding_max_cosine` | float \| null | Máximo global usado para limiar e agregação |
| `embedding_baixo_suporte` | `embedding_low_support` | bool \| null | Abaixo do limiar quando verificação activa e score definido |
| `embedding_max_coseno_recuperados` | `embedding_max_cosine_retrieved` | float \| null | Máximo só face aos textos em `recuperados` (omitido no JSON se `null`) |
| `embedding_max_coseno_ouro` | `embedding_max_cosine_gold` | float \| null | Máximo só face à passagem gold (omitido se `null`) |

**Implementação:** `pipeline.verify_item` → `verification/embedding_verify.max_cosine_answer_to_chunks`; serialização em `verification/aggregate.signals_to_dict`.

### Meta de auditoria (não entra no limiar directamente)

| Campo | Quando |
|-------|--------|
| `meta.passagem_ouro_rag` | `rag_gold_chunk` não vazio (truncado a 12k chars) |
| `recuperados[]` | Lista `{texto, score, e_chunk_ouro}` do retriever |
| `meta.metricas_recuperacao` | [SPEC-001](001-retrieval.md) — diagnóstico de retrieval separado |

## Segmentação da resposta (formal)

Função: `embedding_verify.split_sentences` + fallback em `max_cosine_answer_to_chunks`.

### Algoritmo

1. `t = text.strip()` da resposta — campo `resposta` do contrato JSON do respondedor (`responder_schema.validate_responder_response` → `generation.generate_answer`). `extract_answer_line` existe só como fallback legado de migração (`RESPOSTA:`) e **não** é o caminho principal da pipeline.
2. `partes = re.split(r"(?<=[.!?])\s+", t)` — quebra **após** `.`, `!` ou `?` seguidos de espaço (lookbehind; o pontuação fica na frase anterior).
3. `frases = [p for p in partes if p.strip()]` — remove segmentos vazios.
4. Se `frases` vazia: usa `frases = [answer]` (resposta inteira como uma unidade).
5. Embeddings: `embedder.embed(frases)` e `embedder.embed(chunks)` em batch.
6. Matriz `sims = S @ C.T` com `S` (n_frases × dim), `C` (n_chunks × dim).
7. **Score:** `max(sims)` — máximo sobre **todas** as células (equivale a: para cada frase, máximo coseno a qualquer chunk; depois máximo entre frases).

**Não implementado:** média por frase, p95, nem índice da frase/chunk vencedora (ver secção «Métricas além do máximo»).

### Casos limite de segmentação

| Caso | Comportamento |
|------|----------------|
| Resposta vazia / só espaços | `strip()` → `""`; split pode dar `[]`; fallback `[answer]` → `[""]`; coseno degenerado mas definido |
| Uma palavra / sem `.!?` | Uma frase = texto inteiro |
| Abreviações (`Dr. Smith`) | Pode partir incorrectamente no `.` — limitação conhecida (sem NER) |
| Idioma | **Sem** regra por língua; mesmo regex para PT/EN/etc. |
| Pontuação unicode | Só ASCII `.!?` no lookbehind |
| JSON inválido / schema rejeitado | `generate_answer` devolve mensagem fixa de erro; segmentação aplica-se a esse texto curto |
| Legado `RESPOSTA:` (sem JSON) | `extract_answer_line` extrai a última linha marcada; segmentação no texto extraído — só migração, não contrato actual |

## Modelo de embeddings e reprodutibilidade

O **mesmo** `Embedder` instanciado uma vez por corrida (`pipeline` / `orchestration/multi` → `make_embedder(cfg.embeddings.backend, cfg.embeddings.model_name)`) serve recuperação ([SPEC-001](001-retrieval.md)) e grounding.

| Backend | Dimensão | Normalização | Semântica | Uso típico |
|---------|----------|--------------|-----------|------------|
| `sentence_transformers` | do modelo (ex. 384 MiniLM) | `normalize_embeddings=True` no `encode` | Sim | Produção / `configs/nq_open_rag.yaml` |
| `hash` | 128 fixo | L2 por vector (`v / ||v||`) | Não (determinístico) | CI / testes (`tests/test_embedding_verify.py`) |

- **Seed:** não afecta ST; `hash` usa `blake2b(texto)` → seed por texto (reprodutível entre processos).
- **OpenAI / outros APIs:** **não** há backend no código; adicionar exigiria novo ramo em `retrieval.make_embedder`.
- **Comparabilidade entre corridas:** fixar `embeddings.*`, `rag.chunk_max_chars`, `rag.top_k`, versão do pacote `sentence-transformers` e corpus por item. Mudar qualquer um invalida comparação directa de `embedding_max_coseno`.
- **Limiar `embedding_min_cosine`:** calibrar **por backend** (ex. `0.28` em `nq_open_rag.yaml` com MiniLM-L6-v2; `0.28` em `default.yaml` com paraphrase-multilingual-MiniLM).

## Score de coseno (por item)

### Face aos recuperados

- Entrada: textos `recuperados[].texto` (top-k após `Retriever.retrieve`, possivelmente sem chunk ouro se `inject_retrieval_failure`).
- `embedding_max_coseno_recuperados = max_cosine_answer_to_chunks(resposta, textos_recuperados, embedder)`.
- Usa **todos** os chunks da lista recuperada (já limitada a `top_k`); não há segundo top-N dentro do grounding.

### Face à passagem gold

- Se `(item.rag_gold_chunk or "").strip()` não vazio **e** `embedding_use_gold_chunk: true`:
  - `embedding_max_coseno_ouro = max_cosine_answer_to_chunks(resposta, [gold_text], embedder)` — passagem **inteira**, sem `chunk_text`.
- Permite score alto mesmo quando o retriever falhou mas a passagem gold existe no adaptador (reduz FP “só porque chunks longos não alinharam”).

### Agregação do máximo global

```text
scores = []
se recuperados: scores.append(embedding_max_coseno_recuperados)
se gold activo: scores.append(embedding_max_coseno_ouro)
se scores não vazio:
    embedding_max_coseno = max(scores)
    embedding_baixo_suporte = (embedding_max_coseno < embedding_min_cosine)
senão se had_corpus e recuperados vazio:
    embedding_max_coseno = 0.0
    embedding_baixo_suporte = true
senão:
    embedding_max_coseno = null
    embedding_baixo_suporte = null
```

`had_corpus = bool(corpus_chunks)` com `corpus_chunks = build_chunks_for_item(...)` passado pelo pipeline.

**Nota sobre `embedding_verify.embedding_low_support`:** com `chunks == []` devolve `False`; o pipeline **não** usa esta função — aplica a tabela acima em `verify_item`.

## Tabela decisão: `null` vs `0.0` vs float

Condições: após `apply_protocol_defaults` / `validate_protocol`, salvo testes unitários que forçam `verify_embedding` manualmente.

| Situação | `embedding_max_coseno` | `embedding_baixo_suporte` | `_recuperados` / `_ouro` |
|----------|------------------------|---------------------------|-------------------------|
| `verify_embedding: false` | `null` | `null` | `null` |
| `rag.enabled: false` (bloco grounding inactivo) | `null` | `null` | `null` |
| Item sem corpus (`corpus_chunks == []`) | `null` | `null` | `null` |
| Corpus existe, há score (recuperados e/ou gold) | float ∈ [-1, 1] | `true` iff `< embedding_min_cosine` | float(s) presentes |
| Corpus existe, `recuperados == []`, gold inactivo ou ausente | `0.0` | `true` | `null` |
| Corpus existe, `recuperados == []`, gold activo com texto | float (só ouro) | conforme limiar | `null` / float ouro |

**Proibido (protocolo):** `embedding_baixo_suporte = true` só porque o dataset não tem corpus (ex. NQ-Open sem passagem) — `protocol` desliga embedding ou falha validação; ver `tests/test_verify_embedding_na.py`.

**Ambiguidade resolvida:** `0.0` **não** significa “verificação desligada”; significa “nenhuma similaridade computável com recuperados e sem score gold” com corpus presente e lista recuperada vazia. `null` significa “camada não aplicável ou sem base de comparação”.

### Recálculo offline

`evaluation_metrics.recompute_embedding_low_support`: se `embedding_max_coseno is None`, mantém `embedding_baixo_suporte` anterior; se float, redefine `embedding_baixo_suporte = (emb_max < limiar)`. Usado no dashboard e `replay_anomaly_flags` para calibrar limiar sem re-embedar.

## Métricas além do máximo

### Implementado (por item)

| Métrica | Descrição |
|---------|-----------|
| `embedding_max_coseno` | Máximo global (limiar + agregação) |
| `embedding_max_coseno_recuperados` | Ramo recuperados |
| `embedding_max_coseno_ouro` | Ramo passagem gold |

### Implementado (agregado — sem bloco `sumario_grounding` dedicado)

Ver secção «Summary agregado»; não há média/p95 de coseno no `summary.json` hoje.

### Planeado (spec futura — **não** no código)

| Métrica sugerida | Utilidade |
|------------------|-----------|
| `embedding_media_coseno` | Sensibilidade a frases periféricas vs uma frase forte |
| `embedding_p95_coseno` | Cauda de frases mal ancoradas |
| `embedding_frase_max_indice` / `embedding_chunk_max_indice` | Auditabilidade (qual frase ↔ qual chunk) |
| Matriz esparsa top-3 pares (frase, chunk) | Debug no dashboard |

Marcar implementação futura evita confundir com RAGAS `answer_relevancy`.

## Protecção contra inflação por chunks

### Como o corpus é construído

- `chunk_text(passagem, chunk_max_chars)`: divisão **fixa por caracteres**, sem overlap; passagens vazias → `[]`.
- `build_chunks_for_item`: concatena chunks de `rag_gold_chunk` + `rag_distractors`, **dedup** por string exacta (ordem preservada).

### Risco de inflação

- Grounding usa **max** sobre todas as frases × **todos** os chunks recuperados.
- Muitos chunks **pequenos** no mesmo documento aumentam a probabilidade de um par frase–chunk com coseno alto por sobreposição lexical superficial, sem entailment.
- `top_k` grande + `chunk_max_chars` pequeno → mais superfície de matching no max.

### Regras recomendadas (operacionais)

| Parâmetro | Recomendação |
|-----------|--------------|
| `rag.top_k` | 4–8 em produção; subir só com ranker melhor ou análise de inflação |
| `rag.chunk_max_chars` | Alinhar à unidade semântica do adaptador (400–500 em NQ/Fairytale); evitar 50–100 salvo teste |
| Scoring | Usa **só** chunks em `recuperados` (top-k), não o corpus completo |
| Dedup | Já aplicado na construção do corpus; chunks recuperados distintos por texto |
| Gold no score | `embedding_use_gold_chunk` mede ancoragem à verdade do adaptador, não só ao retriever — documentar em relatórios |

**Não implementado:** penalização por número de chunks, média em vez de max, nem filtro por `score` de retrieval no grounding.

## Interpretabilidade e auditoria

### Decomposição recuperados vs ouro

- Comparar `embedding_max_coseno_recuperados` vs `embedding_max_coseno_ouro` vs `embedding_max_coseno` no dashboard (`dashboard/data.py`, Inspector).
- Se ouro >> recuperados: provável falha de retrieval ou chunking; ver [SPEC-001](001-retrieval.md) e padrão `recuperacao_falhou` [SPEC-007](007-pattern-detection.md).

### O que **não** está gravado (lacuna)

- Índice da frase da resposta ou do chunk que atingiu o máximo.
- Score por frase ou heatmap frase×chunk.

### Pistas de auditoria existentes

| Fonte | Conteúdo |
|-------|----------|
| `recuperados` | Texto e `score` de retrieval por chunk |
| `meta.passagem_ouro_rag` | Passagem gold usada no ramo ouro |
| `meta.metricas_recuperacao.rank_chunk_ouro` | Posição do chunk ouro no top-k ([SPEC-001](001-retrieval.md)) |
| `meta.diagnostico.padroes` | `grounding_baixo`, `grounding_fp_suspeito` ([SPEC-007](007-pattern-detection.md)) |

## Summary agregado (`summary.json`)

**Não existe** `sumario_grounding`. Métricas de grounding aparecem em:

### `analise_camadas` (`evaluation_metrics.layer_analysis` → `reporting.summarize`)

| Campo | Definição |
|-------|-----------|
| `gatilhos_marginais.n_embedding_baixo_suporte` | Contagem com `embedding_baixo_suporte is True` (denominador implícito: `n_itens`) |
| `combinacoes_exclusivas_* .so_embedding` | Só embedding baixo (sem ouro incorreto nem juiz negativo) |
| `combinacoes_exclusivas_* .embedding_e_juiz` | Embedding baixo **e** juiz negativo |
| `por_camada_vs_referencia.sinal_embedding` | Matriz VP/FP/FN/VN vs referência do adaptador (`gold_correct == false` como positivo); **diagnóstico**, não KPI de grounding |
| `concordancia_entre_camadas` | Cohen's κ entre pares (`sinal_ouro`, `embedding`, `juiz`) |

Omitido quando: corrida sem registos (trivial); campos marginais são `0` se ninguém dispara embedding.

### `estratificacao_fp_gold_correto` (`reporting.summarize`)

Quando há FP (`gold_correto` e `flag_anomalia`):

| Campo | Definição |
|-------|-----------|
| `n_fp_gold_correto` | Tamanho do conjunto FP |
| `com_so_embedding_baixo` | FP com `embedding_baixo_suporte` e juiz não negativo |
| `com_so_juiz_negativo` | FP com juiz negativo e embedding não baixo |
| `com_embedding_e_juiz` | Ambos |
| `sem_embedding_nem_juiz_negativo` | FP por outra camada (ex. gold) |

Se `n_fp_gold_correto == 0`, só `nota` e contagem zero.

### `sumario_padroes`

Contagem de tags `grounding_baixo` e `grounding_fp_suspeito` em `meta.diagnostico.padroes` (agregação por tag, não média de coseno).

### `detector_activo.camadas_verificacao`

Inclui `"embedding"` se `protocolo_ativo.verify_embedding` no snapshot da corrida.

### Agregação em [SPEC-004](004-aggregation.md)

- `embedding_baixo_suporte is True` conta na camada só se `verify_embedding: true`.
- Política `embedding_e_juiz` exige juiz negativo **e** embedding baixo para anomalia (quando ambas activas).

## Ligação com geração

- Texto avaliado: campo `resposta` do JSON validado (`responder_schema.py` + `generation.generate_answer`) — não o raciocínio interno nem texto livre fora do contrato.
- Legado: `generation.extract_answer_line` extrai linha `RESPOSTA:` apenas como fallback de migração; a pipeline actual não o invoca no fluxo RAG pt-BR.
- Gate de recuperação fraca ([SPEC-001](001-retrieval.md)): resposta curada pode ter coseno baixo face a `[]`; se corpus existia e recuperação vazia, regra `0.0` / `baixo_suporte true` aplica-se.

## Validação de protocolo

| Verificação | Onde |
|-------------|------|
| `verify_embedding` sem RAG/corpus | `validate_protocol` → `ValueError`; `apply_protocol_defaults` desliga embedding |
| Contagem corpus | `_count_items_with_corpus` = itens com `build_chunks_for_item` não vazio |

## Fora de âmbito

- NLI / entailment por claim (`docs/techniques/nli-and-claim-grounding.md`) — futuro opcional; unidade natural seria a mesma `split_sentences` ou tokenizer dedicado.
- Limiar adaptativo por percentil da corrida.
- Backend de embeddings OpenAI/Cohere sem implementação no repositório.
- Métricas RAGAS completas (`answer_relevancy`, etc.).

## Critérios de aceitação

- [x] Segmentação documentada e implementada em `embedding_verify.split_sentences` + teste de intervalo em `test_embedding_verify.py`.
- [x] Tabela `null` / `0.0` / float alinhada a `pipeline.verify_item` e `protocol.py`; teste sem corpus em `test_verify_embedding_na.py`.
- [x] Com `rag_gold_chunk` e `embedding_use_gold_chunk: true`, o score ouro é registado em `embedding_max_coseno_ouro` como **diagnóstico** e **não** entra no máximo que decide `embedding_baixo_suporte` (`pipeline.verify_item`). Antes entrava: uma resposta próxima da referência saía bem ancorada mesmo sem contexto recuperado, misturando o plano de grounding com o plano de referência ([CLAUDE.md](../../CLAUDE.md), regra 8). `protocolo_ativo.embedding_grounding_source` declara a fonte efectiva; corridas anteriores a esta correcção não são comparáveis neste sinal.
- [x] NQ-Open sem corpus: embedding inactivo ou `null` (`configs/nq_open.yaml` + protocolo).
- [x] Decomposição `embedding_max_coseno_recuperados` / `_ouro` no JSONL (`aggregate.signals_to_dict`).
- [x] Agregação em `analise_camadas` e estratificação FP (`reporting.summarize`, `evaluation_metrics.layer_analysis`).
- [x] Padrões `grounding_baixo` / `grounding_fp_suspeito` ([SPEC-007](007-pattern-detection.md), `test_pattern_detection.py`).
- [x] Recálculo offline de limiar (`recompute_embedding_low_support`, `test_recompute_embedding.py`).
- [ ] `sumario_grounding` com média/p95 de `embedding_max_coseno` (planeado).
- [ ] Índices frase/chunk vencedores no JSONL (planeado).
- [ ] Teste de integração dedicado: corpus + `recuperados == []` → `0.0` e `baixo_suporte true`.

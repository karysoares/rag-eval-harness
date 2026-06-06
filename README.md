<div align="center">

# rag-eval-harness

**Harness reprodutível para avaliar pipelines RAG + LLM** — recuperação, geração, grounding, juiz e padrões determinísticos, com dashboard offline e artefactos auditáveis.

[![CI](https://github.com/karysoares/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/karysoares/rag-eval-harness/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](#qualidade)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](#dashboard)

[Começar](#começar-em-2-minutos) · [Arquitetura](#arquitetura-da-pipeline) · [Orquestração](#orquestração) · [Dashboard](#dashboard) · [Documentação](#documentação)

</div>

---

## O que é

Laboratório de avaliação focado em **FairytaleQA pt-BR** ([`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR)).

O repositório separa sempre:

| Camada | Papel | Exemplos no código |
|--------|--------|-------------------|
| **Adaptador** | Carregar o corpus → `EvalItem` | `adapters/`, `eval_items_load.py` |
| **Sistema sob teste** | Recuperar contexto e gerar resposta | `retrieval.py`, `generation.py` |
| **Harness de medição** | Sinais, padrões, agregação, relatório | `verification/`, `pattern_detection.py`, `reporting.py` |

Métricas de **recuperação** são **diagnósticas** (qualidade do contexto antes da resposta). Sinais pós-resposta (**embedding**, **juiz**, **referência léxica**) ficam **separados** até à política de agregação no YAML — não existe um único “score de verdade” universal.

**Por omissão:** [`configs/default.yaml`](configs/default.yaml) — 32 itens, RAG, `orchestration: unico`, prompts e juiz em português.

Documentação de arquitectura, premissas e specs é mantida **localmente** na pasta `docs/` (não versionada no repositório público).

### Matriz KPI (três planos — não misturar)

| Plano | Bloco no `summary.json` | Uso |
|-------|-------------------------|-----|
| **A — Produto** | `sumario_lexical` | Qualidade face à referência do corpus |
| **B — Risco** | `sumario_operacional`, `n_anomalias_marcadas` | Alertas da política YAML |
| **C — HITL** | `sumario_hitl` | Calibração na amostra com `adjudicacao_humana` |

Comandos: `llm-eval --apply-hitl CSV --resume RUN_DIR` · `scripts/publish_run_evidence.py` (agregados publicáveis podem ser gerados localmente).

### Resultados de referência — **Plano A (léxico — diagnóstico)**

Estes números medem **proximidade textual** à resposta de referência do corpus (`reference_type: lexical`). **Não** são o KPI principal de qualidade RAG (Plano B: `sumario_operacional`, juiz, embedding) nem de alerta operacional (Plano C: HITL). Definições resumidas nas secções [Matriz KPI](#matriz-kpi-três-planos--não-misturar) e [Qualidade](#qualidade) abaixo.

Agregados de corridas locais em `outputs/run_*/summary.json` (29 corridas com sumário; `outputs/` não versionado). Datasets encontrados nos manifests/configs: **FairytaleQA-translated-ptBR** (`mode: hub`), **fairytale_ptbr** (`mode: amostra_local`, smoke/CI), **TruthfulQA** (`mode: demo`, corrida legada sem manifest). Colunas **Plano A** (METEOR, ROUGE-L, BLEU) vêm de `sumario_lexical`; recuperação de `sumario_recuperacao.taxa_chunk_ouro_no_top_k`; juiz = `1 − n_juiz_diagnostico_negativo / n_itens` em `analise_camadas`; anomalia = `taxa_alerta` da política YAML activa.

| Dataset | N | Config / corrida | METEOR | ROUGE-L (F) | BLEU | Recuperação (% ouro top-k) | Juiz (% sustentado) | Taxa anomalia |
|---------|---|------------------|--------|-------------|------|----------------------------|---------------------|---------------|
| FairytaleQA-translated-ptBR (hub) | 1025 | `ptbr_fairytale_full` — baseline (`run_20260517T190713Z`) | 0,783 | 0,380 | 0,206 | 100% | 79,6% | 0% |
| FairytaleQA-translated-ptBR (hub) | 1025 | `ptbr_fairytale_full` — calibrado (`run_20260517T215023Z`) | 0,823 | 0,366 | 0,193 | 100% | 95,4% | 1,5% |
| FairytaleQA-translated-ptBR (hub) | 1025 | `ptbr_fairytale_full` — pós-calibração (`run_20260518T074031Z`) | 0,826 | 0,367 | 0,191 | 100% | 91,5% | 0,7% |
| FairytaleQA-translated-ptBR (hub) | 32 | `default.yaml` — dev (`run_20260517T102836Z`) | 0,679 | 0,422 | 0,221 | 100% | 87,5% | 9,4% |
| fairytale_ptbr (amostra_local) | 50 | amostra dev, sem manifest (`run_20260517T084621Z`) | 0,720 | 0,618 | 0,495 | 100% | 84,0% | 16% |
| fairytale_ptbr (amostra_local) | 2 | `smoke_amostra.yaml` / baselines CI (`run_20260516T124205Z`) | — | — | — | 100% | 100% | — |
| TruthfulQA (demo, legado) | 2 | config removido (`run_20260516T190703Z`) | — | — | — | 100% | 100% | — |
| FairytaleQA-translated-ptBR (hub) | 1025 | `ptbr_fairytale_tuned.yaml` (`run_20260606T121845Z`) | 0,901 | 0,349 | 0,175 | 100% | 78,0% | 0% |

Modelos típicos nas corridas com manifest: `LLM_MODEL=gpt-4o-mini`, `JUDGE_MODEL=gpt-4o` ou `gpt-4o-mini` (ver aviso no log se iguais). Política de agregação nas corridas FairytaleQA completas: `embedding_e_juiz`. Par baseline/calibrado também em `run_comparison.json` (não versionado).

**Por que BLEU ≈ 0,2 é esperado (FairytaleQA hub):** respostas narrativas em paráfrase raramente coincidem n-gramas com a referência curta do dataset. **METEOR** e **ROUGE-L** são mais informativos aqui; o juiz RAG (`judge_prompt_style: rag_pt`) mede grounding, não overlap léxico. Em `amostra_local` com N pequeno, BLEU mais alto reflecte amostra e tamanho, não comparabilidade directa com N=1025.

#### Como interpretar

- **Plano A (léxico):** tendência de cobertura semântica face à referência do adaptador — útil para regressão de geração, **não** para declarar “RAG excelente” isoladamente.
- **Recuperação (`sumario_recuperacao`):** `taxa_chunk_ouro_no_top_k` e `media_score_melhor_chunk` diagnosticam se o contexto certo chega ao LLM **antes** da resposta.
- **Juiz (`sumario_juiz`, vereditos por item):** `% sustentado` alto + `% negativos` baixo indica respostas ancoradas; `% incompleto` alto com RAG forte sugere recusas ou respostas vagas — ver fila HITL.
- **Operacional:** `taxa_alerta` / `n_anomalias_marcadas` reflectem só a política YAML activa (`embedding_e_juiz`); zero alertas **não** implica qualidade perfeita.

#### Comparativos e interpretação

##### 1. Corridas internas (mesmo harness, FairytaleQA pt-BR N=1025)

Quatro corridas completas sobre o split `validation` com o mesmo adaptador e política `embedding_e_juiz`; variam config YAML, calibração de embedding e parâmetros de geração/RAG.

| Corrida | Config | METEOR | ROUGE-L (F) | BLEU | Juiz (% sustentado) | Taxa anomalia |
|---------|--------|--------|-------------|------|---------------------|---------------|
| baseline (`run_20260517T190713Z`) | `ptbr_fairytale_full` | 0,783 | 0,380 | 0,206 | 79,6% | 0% |
| calibrado (`run_20260517T215023Z`) | `ptbr_fairytale_full` | 0,823 | 0,366 | 0,193 | 95,4% | 1,5% |
| pós-calibração (`run_20260518T074031Z`) | `ptbr_fairytale_full` | 0,826 | 0,367 | 0,191 | 91,5% | 0,7% |
| **tuned (`run_20260606T121845Z`)** | **`ptbr_fairytale_tuned`** | **0,901** | 0,349 | 0,175 | 78,0% | **0%** |

A corrida **tuned** (`configs/ptbr_fairytale_tuned.yaml`) concentra o melhor **METEOR** (+7,4 pp vs pós-calibração) e **0% de anomalias** operacionais, ao custo de ROUGE-L e BLEU ligeiramente mais baixos. Os ajustes principais são: `max_tokens: 256` (respostas narrativas mais completas), `reparar_recusa_generica` + `judge_incompleto_contexto_forte_negativo` (anti-refusal quando o RAG é forte), `top_k: 5` e `chunk_max_chars: 650` (menos cortes mid-frase), e `embedding_min_cosine: 0.26`.

**Trade-offs (não confundir sinais):**

- **METEOR ↑, ROUGE-L/BLEU ↓:** respostas mais paráfraseadas e semanticamente próximas do gold, com menos overlap de n-gramas — comportamento esperado em QA narrativo; METEOR captura sinonímia melhor que ROUGE-L.
- **Juiz sustentado ↓ (91,5% → 78,0%):** não indica queda de qualidade isolada. A config tuned marca mais vereditos **`incompleto`** (~21% dos itens) quando há RAG forte mas resposta vaga ou recusa parcial — diagnóstico, não alerta operacional (`incompleto` entra em `negative_judge_verdicts`, mas **não** em `judge_aggregation_verdicts`).
- **Anomalias 0% vs 0,7%:** com `embedding_e_juiz`, alertas exigem embedding baixo **e** juiz “duro”. A corrida tuned elimina os 7 alertas da pós-calibração; a fila HITL ficou com **11 itens** (1,1%) para revisão humana, não confundir com `taxa_alerta`.
- **Aviso juiz = gerador:** na corrida tuned, `LLM_MODEL` e `JUDGE_MODEL` podem ser ambos `gpt-4o` — o manifest regista aviso de auto-referência. Para comparação mais estrita, use `LLM_MODEL=gpt-4o-mini` + `JUDGE_MODEL=gpt-4o` (como em `ptbr_fairytale_full`).

**Deltas entre corridas (N=1025):** baseline → calibrado ganha +3,9 pp METEOR e +15,8 pp juiz sustentado, com anomalias a subir para 1,5%. Pós-calibração estabiliza léxico e reduz anomalias (0,7%). Tuned maximiza METEOR (+7,4 pp vs pós-calibração, +11,8 pp vs baseline) e zera alertas operacionais, com ROUGE-L −1,8 pp vs baseline — leitura conjunta Plano A + B, não uma coluna isolada.

Reproduzir deltas entre duas corridas: `uv run llm-eval --compare-runs outputs/run_<a> outputs/run_<b>` → `run_comparison.json` (diferenças em `sumario_lexical`, `sumario_operacional`, `analise_camadas`).

##### 2. Como ler os comparativos (entre as nossas corridas)

Compare sempre **o mesmo N e adaptador** (ex.: FairytaleQA hub N=1025). Cruzar Planos A e B — não ranquear por ROUGE-L ou juiz isolados.

| Plano | O que mede | Como usar entre corridas |
|-------|------------|--------------------------|
| **A — léxico** (`sumario_lexical`) | Overlap / paráfrase vs gold do corpus | Regressão de geração; em narrativa, METEOR > ROUGE-L. Ex.: tuned (0,901) > pós-calibração (0,826) com ROUGE-L mais baixo — paráfrase semântica, não cópia léxica. |
| **B — operacional** (`sumario_operacional`, juiz, embedding) | Grounding e alertas YAML | KPI de risco RAG. Calibrado (95,4% juiz) vs tuned (78,0%): ver `% incompleto` e `taxa_alerta`, não só `% sustentado`. |
| **Juiz (% sustentado)** | `1 − negativos / N` em `analise_camadas` | Tendência de grounding; `% incompleto` alto é diagnóstico, não alerta operacional. |
| **HITL** (`sumario_hitl`, fila CSV) | Amostra com adjudicação humana | Complementa B quando juiz e embedding divergem; tuned deixou 11 itens (1,1%) na fila com 0% `taxa_alerta`. |

ROUGE-L baixo pode coexistir com METEOR alto entre corridas do mesmo harness: respostas correctas em paráfrase penalizam n-gramas (ROUGE/BLEU) enquanto METEOR recompensa alinhamento semântico. Recuperação (`taxa_chunk_ouro_no_top_k`) deve estar alinhada antes de atribuir diferenças só à geração ou ao juiz.

Métricas operacionais: ver `summary.json` (`sumario_lexical`, `sumario_operacional`, `analise_camadas`). Notas locais opcionais podem ficar em `docs/` (gitignored).

Para publicar agregados localmente: `uv run python scripts/publish_run_evidence.py outputs/run_<id>` (destino configurável; por omissão pasta local gitignored).

### Trabalho relacionado

| Projeto | Foco | Ligação |
|---------|------|---------|
| **RAGAS** | Métricas RAG (faithfulness, context precision/recall) | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| **TruLens** | Observabilidade e feedback em apps LLM/RAG | [truera/trulens](https://github.com/truera/trulens) |
| **ARES** | Avaliação automática de sistemas RAG | [stanford-futuredata/ARES](https://github.com/stanford-futuredata/ARES) |
| **lm-evaluation-harness** | Benchmarks de LLM (não RAG end-to-end) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |

Este harness complementa essas ferramentas com **verificação multicamada configurável**, **padrões determinísticos**, **fila HITL** e **artefactos auditáveis** orientados a FairytaleQA pt-BR. Integração opcional com RAGAS via extra `ragas` no `pyproject.toml`.

---

## Arquitetura da pipeline

### Visão geral (por item)

```mermaid
flowchart TB
  subgraph entrada
    YAML[configs/*.yaml]
    DS[(Dataset HF / amostra local)]
  end

  YAML --> CLI[llm-eval CLI]
  DS --> ADP[Adaptador → EvalItem]
  ADP --> ORCH{orchestration}

  ORCH -->|unico| BATCH[pipeline.run_batch]
  ORCH -->|multiplo| MULTI[orchestration/multi.py]

  subgraph sistema["Sistema sob teste"]
    R[1. Recuperação<br/>Retriever + chunks]
    RM[metricas_recuperacao]
    G[2. Geração<br/>LLM + prompts pt]
    R --> RM
    R --> G
  end

  BATCH --> R
  MULTI --> R

  G --> A[answer]

  subgraph harness["Harness de medição"]
    V[3. verify_item]
    PAT[4. pattern_detection]
    AGG[5. anomaly_from_signals]
    V --> PAT
    V --> AGG
    PAT --> OUT
    AGG --> FLAG[flag_anomalia]
  end

  A --> V
  RM --> V

  OUT[(predictions.jsonl)]
  FLAG --> OUT
  OUT --> SUM[summary.json + manifest]
  SUM --> DASH[llm-eval-dashboard]
```

### Fluxo detalhado pós-resposta

Depois de existir `answer`, o harness avalia **três ramos independentes** (cada um responde a uma pergunta diferente — não misturar KPIs):

```mermaid
flowchart TB
  A[ANSWER + retrieved chunks]

  A --> GR[Grounding / embedding]
  A --> JU[Quality Judge LLM]
  A --> REF[Referência opcional]

  GR --> SG[embedding_baixo_suporte<br/>coseno resposta↔chunks]
  JU --> SJ["juiz.veredito<br/>sustentado, nao_sustentado, ..."]
  REF --> LX[F1, ROUGE-L, METEOR<br/>se reference_type: lexical]

  SG --> PAT[Padrões determinísticos<br/>recusa, referencia_fraca, ...]
  SJ --> PAT
  LX --> PAT

  SG --> AGG{aggregation.policy}
  SJ --> AGG
  REF --> AGG

  AGG --> ANOM[flag_anomalia]
  PAT --> JSONL[predictions.jsonl + meta.diagnostico]
  ANOM --> JSONL
```

| Ramo | Pergunta | Módulos | Campos em `predictions.jsonl` |
|------|----------|---------|-------------------------------|
| **Recuperação** (pré-geração) | O contexto certo chegou ao gerador? | `retrieval_metrics.py` | `meta.metricas_recuperacao.*` |
| **Grounding** | A resposta ancora-se ao que foi recuperado? | `embedding_verify.py` | `embedding_max_coseno`, `embedding_baixo_suporte` |
| **Juiz** | A resposta é sustentada, completa e segura? | `verification/judge.py` | `juiz.veredito`, `juiz_negativo` |
| **Referência** | Overlap com resposta curta do dataset? | `lexical_metrics.py`, `reference_metrics.py` | `meta.metricas_lexicas.*` |
| **Padrões** | Tags explicáveis (não alteram resposta) | `pattern_detection.py` | `meta.diagnostico` |
| **Agregação** | Quais sinais disparam **alerta**? | `aggregate.anomaly_from_signals` | `flag_anomalia` |

### Etapas por ordem de execução

1. **Carregar config + itens** — YAML reprodutível; `protocol.py` pode desligar RAG/embedding se não houver corpus.
2. **Chunks + retrieve** — `build_chunks_for_item` → `Retriever.retrieve(top_k)`; opcional injecção de falha (`inject_retrieval_failure`).
3. **Gerar resposta** — `generate_answer` com prompts canónicos em `src/llm_evaluation/prompts/responder_*` (espelhados em `prompts/` na raiz); saída JSON `{resposta, confianca, contexto_insuficiente}`; gate opcional se score &lt; `min_score_recuperacao` (resposta curada sem LLM).
4. **Verificar** — `verify_item`: gold (se activo), embedding, juiz RAG pt com metadados de recuperação.
5. **Agregar** — `anomaly_from_signals` conforme `aggregation.policy` e listas de vereditos.
6. **Diagnosticar** — `compute_diagnostico` (tiers, tags SPEC-007).
7. **Persistir** — `predictions.jsonl` linha a linha; no fim `summary.json`, `manifest.json`, `anomalies.*`.

### Políticas de agregação (`aggregation.policy`)

| Política | Comportamento típico |
|----------|----------------------|
| `qualquer_critico` | Alerta se **qualquer** camada activa disparar |
| `embedding_e_juiz` | Alerta só com **embedding baixo E juiz negativo** (menos FP por embedding isolado) |
| `todos_criticos` | Alerta só se **todas** as camadas activas dispararem |

Vereditos do juiz têm **dois níveis** no YAML:

- `negative_judge_verdicts` — diagnóstico e padrões (inclui `incompleto`).
- `judge_aggregation_verdicts` — só vereditos “duros” entram na agregação (`nao_sustentado`, `contradicacao`, `inseguro` por omissão).

Isto evita que `incompleto` domine os alertas quando o embedding também está baixo. Detalhe: política em `aggregation.policy` no YAML (ex.: `embedding_e_juiz`).

### Baselines (ablation)

`--compare-baselines` ou `--profile` isolam **uma** camada de verificação por corrida (`so_embeddings`, `so_juiz`, `hibrido`), com `verify_gold: false` em todos os perfis aplicados — para não confundir substring gold com sinal RAG.

`baselines.profile` no YAML é **só etiqueta** (`perfil_baseline` no JSONL); não altera `verify_*`. Use `llm-eval --profile …` ou `--compare-baselines` para ablação efectiva.

### Análise offline (sem API)

```bash
uv run llm-eval --analyze-run outputs/run_<id>
uv run python scripts/validate_embedding_policy.py outputs/run_<id>
```

Reconstrói `analise_camadas`, κ entre camadas, `sumario_gap_rag_resposta`, etc., a partir do JSONL (e `protocolo_ativo` em `summary.json` quando existir).

---

## Orquestração

`orchestration` (ou `orquestracao` no YAML) escolhe **como executar o loop por item**. Não altera adaptador, métricas nem política de agregação — só o caminho de **geração** (e, no modo multi, um passo extra).

```mermaid
flowchart LR
  subgraph unico["orchestration: unico (padrão)"]
    U1[item] --> U2[pipeline.run_batch]
    U2 --> U3[retrieve → generate → verify → agregar]
  end

  subgraph multiplo["orchestration: multiplo (experimental)"]
    M1[item] --> M2[retrieve → generate]
    M2 --> M3[Crítico LLM<br/>critic_system.txt]
    M3 --> M4[verify → agregar]
    M3 --> M5[meta.flag_critica<br/>diagnóstico]
    M4 --> M6[flag_anomalia]
  end
```

| Modo | YAML / CLI | Módulo | O que faz |
|------|------------|--------|-----------|
| **`unico`** | `orchestration: unico` | `orchestration/single.py` → `pipeline.run_batch` | Caminho padrão, usado em todas as configs FairytaleQA |
| **`multiplo`** | `orchestration: multiplo` | `orchestration/multi.py` | Igual + **crítico** após a resposta; `meta.flag_critica` é diagnóstico — **não** altera `flag_anomalia` |

```bash
# Forçar na linha de comando (sobrepor o YAML)
uv run llm-eval --config configs/default.yaml --orchestration unico
uv run llm-eval --config configs/default.yaml --orchestration multiplo --experimental
```

**Quando usar `multiplo`:** experimentos “respondedor + revisor”; o crítico fica em `meta.critica` / `meta.flag_critica` para análise offline, sem entrar na agregação. Exige **`--experimental`** na CLI (ou YAML com `multiplo` falha sem a flag). **Não** está calibrado para produção — use só para experimentação offline.

`meta.orquestracao` no JSONL regista qual modo correu (`"unico"` ou `"multiplo"`).

---

## Dashboard

Interface **local** sobre `outputs/run_*` — análise **sem chamar API**. O separador **Revisão humana** permite anotar itens da fila e importar/exportar `adjudicacoes_hitl.csv` (gravação local na corrida; reprocessar com `--apply-hitl` ou botão no dashboard).

```bash
uv sync --extra dashboard
uv run llm-eval-dashboard
```

**Capturas de ecrã / GIF:** coloque ficheiros em [`assets/readme/`](assets/readme/) (ex.: `dashboard-overview.png`, `dashboard-demo.gif`) e referencie-os aqui. A pasta existe mas os assets ainda não foram gravados — útil para a página GitHub antes do release.

Separadores principais: **Visão geral** (KPI léxico + recuperação + operacional), **Calibração**, **Inspector Q/A**, **Padrões**, **Recuperação**, **Sinais** (κ entre camadas), **Referência**, **Revisão humana** (fila + HITL).

Variável opcional: `LLM_EVAL_OUTPUTS` (defeito: `outputs/`).

---

## Começar em 2 minutos

```bash
git clone https://github.com/karysoares/rag-eval-harness.git && cd rag-eval-harness
uv sync --extra dev --extra dashboard
cp .env.example .env   # OPENAI_API_KEY=... (só para corridas com API)
```

| Objetivo | Comando |
|----------|---------|
| Smoke rápido (2 itens, amostra local — sem Hub; CI usa mock, sem API) | `uv run pytest tests/test_pipeline_e2e_mock.py -q` |
| Smoke com API real (2 itens) | `uv run llm-eval --config configs/smoke_amostra.yaml` |
| FairytaleQA pt-BR + RAG (32) | `uv run llm-eval --config configs/default.yaml` |
| Corrida completa (validation, ~1025 itens) | `bash scripts/run_full_fairytale.sh` ou `uv run llm-eval --config configs/ptbr_fairytale_full.yaml` |
| **Corrida de referência optimizada** | `uv run llm-eval --config configs/ptbr_fairytale_tuned.yaml` (ver comentários no YAML) |
| Retomar corrida interrompida | `uv run llm-eval --config configs/ptbr_fairytale_full.yaml --resume outputs/run_<id>` |
| Ver quantos itens serão corridos | `uv run llm-eval --config configs/ptbr_fairytale_full.yaml --dry-run` |
| Abrir dashboard | `uv run llm-eval-dashboard` |
| Reanalisar última corrida | `uv run llm-eval --analyze-run outputs/run_<id>` |

> Corrida completa usa API (geração + juiz). Dashboard e `--analyze-run` funcionam **sem** API.

*Pacote **rag-eval-harness**; import Python `llm_evaluation`; CLI (`llm-eval`, `llm-eval-dashboard`) inalterados por compatibilidade.*

---

## CLI (`llm-eval`)

```bash
uv run llm-eval --config configs/default.yaml
uv run llm-eval --config configs/ptbr_fairytale_full.yaml --dry-run
uv run llm-eval --config configs/ptbr_fairytale_full.yaml
uv run llm-eval --config configs/ptbr_fairytale_full.yaml --resume outputs/run_<timestamp>
uv run llm-eval --compare-baselines --config configs/default.yaml
uv run llm-eval --analyze-run outputs/run_<timestamp>
uv run llm-eval --apply-hitl adjudicacoes_hitl.csv --resume outputs/run_<timestamp>
uv run llm-eval --compare-runs outputs/run_a outputs/run_b
uv run llm-eval --config configs/default.yaml --orchestration multiplo --experimental
uv run llm-eval --help
```

`--dry-run`, `--analyze-run`, `--apply-hitl` e `--compare-runs` **não** chamam a API de geração/juiz. Corridas normais exigem `OPENAI_API_KEY` (ver [`.env.example`](.env.example)).

**Ablação de baselines (uma variável por perfil):**

```bash
uv run llm-eval --profile so_embeddings --config configs/default.yaml
uv run llm-eval --profile so_juiz --config configs/default.yaml
uv run llm-eval --profile hibrido --config configs/default.yaml
```

---

## Artefactos

Cada corrida grava em `outputs/run_<UTC>/`:

| Ficheiro | Conteúdo |
|----------|----------|
| `predictions.jsonl` | Item a item: pergunta, resposta, chunks, `sinais`, `meta` (recuperação, léxico, diagnóstico, juiz) |
| `summary.json` | KPI, `protocolo_ativo`, `analise_camadas`, IC de Wilson, κ, sumários por camada |
| `manifest.json` | Hashes, metadados (git, config, prompts), integridade |
| `anomalies.jsonl` / `.csv` | Subconjunto com `flag_anomalia` |
| `analise_manual/fila_revisao_humana.csv` | Fila pós-corrida (juiz duro + recusas com RAG forte) |
| `adjudicacoes_hitl.csv` | Rótulos humanos (opcional; merge via `--apply-hitl`) |

`protocolo_ativo` no summary inclui camadas activas, política de agregação, limiar de embedding e listas de vereditos do juiz — para replay offline fiel.

Auditoria local: `uv run python scripts/audit_run.py outputs --strict` · publicar agregados: `uv run python scripts/publish_run_evidence.py outputs/run_<id>`

---

## Configurações

| Config | Uso |
|--------|-----|
| [`configs/default.yaml`](configs/default.yaml) | FairytaleQA pt-BR + RAG (**recomendado**, 32 itens) |
| [`configs/ptbr_fairytale_full.yaml`](configs/ptbr_fairytale_full.yaml) | Validation completo (`limit: 0`) |
| [`configs/ptbr_fairytale_tuned.yaml`](configs/ptbr_fairytale_tuned.yaml) | Validation completo — parâmetros calibrados para métricas + grounding |
| [`configs/smoke_amostra.yaml`](configs/smoke_amostra.yaml) | 2 itens offline (CI, sem Hub) |
| [`configs/ptbr_fairytale.yaml`](configs/ptbr_fairytale.yaml) | Alias com `limit: 64` |
| [`configs/baseline_*.yaml`](configs/baseline_embedding_only.yaml) | Ablation embedding / juiz |

**Chaves YAML com efeito real:**

| Secção | Chaves | Efeito |
|--------|--------|--------|
| `dataset` | `reference_type`, `hf_repo`, `limit`, `mode` | Tipo de referência e origem dos itens |
| `orchestration` | `unico` \| `multiplo` | Runner da corrida (ver [Orquestração](#orquestração)) |
| `rag` | `enabled`, `top_k`, `chunk_max_chars`, `min_score_recuperacao` | Recuperação e gate de geração |
| `verification` | `verify_*`, `embedding_min_cosine`, `negative_judge_verdicts`, `judge_aggregation_verdicts` | Camadas de sinal |
| `aggregation` | `policy` | Como combinar sinais em `flag_anomalia` |
| `patterns` | `referencia_forte`, `referencia_fraca` | Limiares F1 para tags SPEC-007 |
| `operacional` | `fila_min_score_recuperacao`, `gap_*` | Fila humana e detecção de gap RAG–resposta |
| `metricas_lexicas` | `habilitado`, flags por métrica | KPI principal em `reference_type: lexical` |
| `baselines` | `profile` | Etiqueta `perfil_baseline` (ablação via `--profile` ou `--compare-baselines`) |

---

## Estrutura do repositório

```
src/llm_evaluation/
  adapters/           # HF genérico + amostra local pt-BR → EvalItem
  orchestration/      # unico (→ pipeline) | multiplo (+ crítico)
  pipeline.py         # loop principal por item
  retrieval.py        # Retriever + embeddings
  generation.py       # prompts respondedor + hints de recuperação
  verification/       # gold, embedding, juiz, aggregate
  pattern_detection.py
  reporting.py        # summary.json
  evaluation_metrics.py  # analise_camadas, --analyze-run
  prompts/            # fonte canónica empacotada (respondedor + juiz pt-BR)
  dashboard/          # Streamlit local (análise + HITL)
configs/              # YAML reprodutíveis
tests/
```

---

## Documentação

Documentação de arquitectura, premissas, métricas, specs e evidência é mantida **localmente** na pasta `docs/` (gitignored — não aparece no GitHub). Este README e o código em `src/llm_evaluation/` são a referência pública do harness.

---

## Qualidade

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Integração Hub: `RUN_INTEGRATION=1 uv run pytest tests/integration -q`

---

## Licença

MIT — ver [`LICENSE`](LICENSE).

### Dataset FairytaleQA pt-BR

O corpus de referência [`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR) é licenciado sob **Apache-2.0** (compatível com MIT para uso combinado; atribuição recomendada). O dataset original FairytaleQA está descrito em [Xu et al., ACL 2022](https://aclanthology.org/2022.acl-long.34); a tradução pt-BR em [Leite et al., ECTEL 2024](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR#citation). Este repositório **não** redistribui o corpus — apenas consome-o via Hugging Face Hub.

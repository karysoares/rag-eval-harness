<div align="center">

# rag-eval-harness

**Harness reprodutível para avaliar pipelines RAG + LLM** — recuperação, geração, grounding, juiz LLM e padrões determinísticos, com dashboard offline e artefactos auditáveis.

[![CI](https://github.com/karysoares/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/karysoares/rag-eval-harness/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

[Getting started](#getting-started) · [Usage](#usage) · [Architecture](#architecture) · [Fornecedores](#fornecedores) · [Meta-avaliação do juiz](#meta-avaliação-do-juiz) · [Contributing](CONTRIBUTING.md)

🇬🇧 [Read in English](README.md) — versão principal

</div>

---

## Overview

Harness de avaliação **agnóstico ao corpus**: cada dataset é um adaptador; o núcleo mede recuperação, geração e verificação em camadas independentes. O caso de referência incluído é **FairytaleQA pt-BR** ([`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR)).

| Camada | Papel |
|--------|--------|
| **Adaptador** | Corpus → `EvalItem` |
| **Sistema sob teste** | Recuperação + geração |
| **Harness** | Sinais, padrões, agregação, relatório |

Métricas de recuperação são **diagnósticas**. Sinais pós-resposta (embedding, juiz, referência léxica) permanecem **separados** até à política de agregação no YAML — não há um único score universal.

## Features

- Pipeline reprodutível via YAML (`configs/`)
- Verificação multicamada: embedding (grounding), juiz RAG em português, referência léxica (F1, ROUGE-L, METEOR)
- Políticas de agregação configuráveis (`embedding_e_juiz`, `qualquer_critico`, …)
- Padrões determinísticos e fila de revisão humana (HITL)
- **Meta-avaliação do juiz**: calibração, concordância, sondas de viés e auto-consistência
- **Estatística emparelhada** para comparar corridas (McNemar + bootstrap emparelhado)
- **Telemetria** para Phoenix, LangSmith, CloudWatch ou ficheiro JSONL local
- Qualquer fornecedor compatível com OpenAI, com o juiz em **endpoint separado** (Ollama/vLLM local, DeepSeek, Qwen, OpenRouter)
- Dashboard Streamlit offline sobre `outputs/run_*`
- Artefactos auditáveis: `predictions.jsonl`, `summary.json`, `manifest.json`
- Integração opcional com [RAGAS](https://github.com/explodinggradients/ragas)

## Getting started

**Requisitos:** Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/karysoares/rag-eval-harness.git && cd rag-eval-harness
uv sync --extra dev --extra dashboard   # usa uv.lock versionado
cp .env.example .env   # OPENAI_API_KEY — só para corridas com API
```

| Objetivo | Comando |
|----------|---------|
| Smoke offline (sem API) | `uv run pytest tests/test_pipeline_e2e_mock.py -q` |
| Smoke com API (2 itens) | `uv run llm-eval --config configs/smoke_amostra.yaml` |
| Desenvolvimento (32 itens) | `uv run llm-eval --config configs/default.yaml` |
| Corpus completo (~1025 itens) | `uv run llm-eval --config configs/ptbr_fairytale_full.yaml` |
| Dashboard | `uv run llm-eval-dashboard` |

> Corridas com geração e juiz exigem `OPENAI_API_KEY`. Dashboard, `--analyze-run` e `--judge-report` funcionam sem API.

O pacote chama-se **rag-eval-harness**; o import Python é `llm_evaluation` (compatibilidade).

## Usage

```bash
# Corrida
uv run llm-eval --config configs/default.yaml

# Pré-visualizar itens
uv run llm-eval --config configs/ptbr_fairytale_full.yaml --dry-run

# Retomar corrida interrompida
uv run llm-eval --config configs/ptbr_fairytale_full.yaml --resume outputs/run_<id>

# Reanalisar artefactos (sem API)
uv run llm-eval --analyze-run outputs/run_<id>

# Comparar duas corridas (estatística emparelhada quando partilham itens)
uv run llm-eval --compare-runs outputs/run_a outputs/run_b

# Meta-avaliar o juiz (sem API)
uv run llm-eval --judge-report outputs/run_<id>

# Aplicar adjudicações HITL
uv run llm-eval --apply-hitl adjudicacoes_hitl.csv --resume outputs/run_<id>
```

Ablation de baselines (`--profile so_embeddings`, `so_juiz`, `hibrido`) e orquestração experimental (`--orchestration multiplo --experimental`): ver `llm-eval --help`.

## Architecture

```mermaid
flowchart LR
  YAML[configs/*.yaml] --> CLI[llm-eval]
  DS[(Dataset)] --> ADP[Adaptador]
  ADP --> R[Recuperação]
  R --> G[Geração]
  G --> V[Verificação]
  V --> OUT[predictions.jsonl]
  OUT --> SUM[summary.json]
  SUM --> DASH[Dashboard]
```

Três camadas de verificação pós-resposta — **grounding** (embedding), **juiz LLM** e **referência léxica** — combinam-se via `aggregation.policy` no YAML; métricas de recuperação são diagnósticas e não entram na agregação por defeito.

## Run outputs

Cada corrida grava em `outputs/run_<UTC>/`:

| Ficheiro | Conteúdo |
|----------|----------|
| `predictions.jsonl` | Resultado por item (resposta, sinais, diagnóstico) |
| `summary.json` | KPI agregados, `protocolo_ativo`, análise entre camadas |
| `manifest.json` | Hashes, metadados, integridade |
| `anomalies.jsonl` | Subconjunto com `flag_anomalia` |
| `judge_report.json` | Meta-avaliação do juiz (via `--judge-report`) |
| `analise_manual/fila_revisao_humana.csv` | Fila para revisão humana |

Auditoria: `uv run python scripts/audit_run.py outputs --strict`

## Configuration

| Config | Uso |
|--------|-----|
| [`configs/default.yaml`](configs/default.yaml) | FairytaleQA pt-BR, 32 itens (**recomendado**) |
| [`configs/ptbr_fairytale_full.yaml`](configs/ptbr_fairytale_full.yaml) | Validation completo |
| [`configs/ptbr_fairytale_tuned.yaml`](configs/ptbr_fairytale_tuned.yaml) | Validation completo, parâmetros calibrados |
| [`configs/smoke_amostra.yaml`](configs/smoke_amostra.yaml) | 2 itens offline (CI) |
| [`configs/ptbr_fairytale_qwen_local.yaml`](configs/ptbr_fairytale_qwen_local.yaml) | 200 itens, gerador em API + juiz local gratuito |
| [`configs/baseline_*.yaml`](configs/baseline_embedding_only.yaml) | Ablation embedding / juiz |

Políticas de agregação: `qualquer_critico`, `embedding_e_juiz`, `todos_criticos`. Tipos de referência: `lexical`, `answer_lists`, `none` (chave `dataset.reference_type`).

## Dashboard

```bash
uv sync --extra dashboard
uv run llm-eval-dashboard
```

Interface local sobre `outputs/run_*` — KPI, inspector Q/A, calibração, padrões e revisão humana. Variável opcional: `LLM_EVAL_OUTPUTS` (defeito: `outputs/`).

## Fornecedores

Qualquer endpoint compatível com OpenAI serve, e **o juiz pode correr num fornecedor diferente do gerador** — basta `JUDGE_BASE_URL` (e `JUDGE_API_KEY`) além de `OPENAI_BASE_URL`. A base pode ou não terminar em `/v1`; ambas as formas resolvem.

A separação importa por duas razões. O juiz domina o custo: pela contabilidade de tokens de `ptbr_fairytale_full.yaml` (1025 itens, `top_k=4`, `chunk_max_chars=500`, `max_tokens=128`, duas chamadas por item), um juiz `gpt-4o` dá cerca de $2,80 de uma corrida de ~$3,15, contra ~$0,35 do gerador `gpt-4o-mini` — estimativa a partir desses parâmetros e dos preços de tabela, não de uma corrida gravada — por isso passá-lo para local elimina a maior parte da despesa. E um juiz de família diferente do gerador é metodologicamente mais forte: um modelo que avalia as suas próprias respostas tende a preferi-las. A corrida regista ambos os endpoints em `summary.json` → `protocolo_ativo.models`.

```bash
# Gerador em API paga, juiz local e gratuito
ollama pull qwen2.5:7b
```

```dotenv
LLM_MODEL=gpt-4o-mini
JUDGE_MODEL=qwen2.5:7b
JUDGE_BASE_URL=http://localhost:11434
JUDGE_API_KEY=ollama          # endpoints locais ignoram-na, mas exigem uma
```

```bash
uv run llm-eval --config configs/ptbr_fairytale_qwen_local.yaml
uv run llm-eval --judge-report outputs/run_<id>
```

Presets para Ollama, vLLM, DeepSeek, DashScope e OpenRouter em [`.env.example`](.env.example).

**Escolha o juiz com o harness, não por intuição.** Smoke rápido sobre três casos pt-BR com veredito conhecido, usando o prompt e o validador do próprio repo:

| modelo (Ollama, M3 16 GB) | JSON válido | veredito certo | latência |
|---|---|---|---|
| `qwen2.5:7b` | 3/3 | 3/3 | 50 s |
| `deepseek-r1:8b` | 3/3 | 2/3 | 19 s |
| `llama3.1:8b` | 3/3 | 2/3 | 40 s |
| `qwen2.5:3b` | 3/3 | 2/3 | 6 s |
| `phi3.5:3.8b` | 2/3 | 1/3 | 59 s |
| `mistral:7b` | 0/3 | 0/3 | 31 s |

Três casos são um filtro, não evidência. Duas falhas que expõem valem a pena: o `qwen2.5:3b` faz 2/3 respondendo `nao_sustentado` a tudo — um juiz degenerado parece competente num teste curto — e o `mistral:7b` nunca devolve o schema, pelo que todos os vereditos são o fallback heurístico, que por omissão diz `sustentado`. Ambos são apanhados por `--judge-report` (distribuição de vereditos colapsada, κ perto de zero) e pela exclusão de fallbacks descrita abaixo. Corra-o antes de confiar em qualquer juiz.

Erros de configuração falham já e citam o fornecedor: um nome de modelo errado aparece como `HTTP 404 … model 'qwen2.5:7b' not found` à primeira tentativa, em vez de três repetições silenciosas.

## Meta-avaliação do juiz

Um juiz LLM é um instrumento de medição, e um instrumento precisa de ser caracterizado antes de as suas leituras significarem alguma coisa. `reporting._judge_summary` responde a *o juiz correu bem?* (fallbacks, retries, schema inválido). [`judge_meta.py`](src/llm_evaluation/judge_meta.py) responde à pergunta mais dura: *podemos confiar no que o juiz mede?*

```bash
uv run llm-eval --judge-report outputs/run_<id>          # offline, sem API
```

| Propriedade | Pergunta | Método |
|---|---|---|
| Calibração | Quando diz 0.9, acerta 90% das vezes? | ECE/MCE sobre bins de fiabilidade |
| Concordância | Bate com a referência disponível e com o humano? | Confusão 2×2, κ de Cohen, IC de Wilson na exatidão |
| Viés de verbosidade | Aprova respostas longas *por serem longas*? | Correlação ponto-bisserial entre aprovação e comprimento |
| Viés de posição | Só aprova com o chunk ouro em primeiro? | Taxa de aprovação por rank do ouro, com ICs de Wilson |
| Auto-consistência | Dá o mesmo veredito duas vezes? | κ de Fleiss + taxa de unanimidade sobre amostras repetidas |

A auto-consistência exige novas chamadas ao juiz e vive num script próprio:

```bash
uv run python scripts/judge_self_consistency.py outputs/run_<id> --amostras 5 --limite 40
uv run llm-eval --judge-report outputs/run_<id> --judge-samples outputs/run_<id>/judge_self_consistency.jsonl
```

Importa por mais do que arrumação: um juiz instável impõe um piso ao efeito mínimo detetável. Uma diferença entre duas corridas menor que o ruído de amostragem do próprio juiz não é interpretável, por mais significativo que o p-valor pareça.

O relatório herda a política da própria corrida em vez de assumir uma: os vereditos negativos vêm de `summary.json` → `protocolo_ativo.judge_aggregation_verdicts` e o limiar léxico de `pattern_settings.f1_fraca_min`. Caso contrário, um veredito consultivo como `incompleto` — que nunca dispara `flag_anomalia` — seria contado como falso negativo do juiz. A polaridade efectiva e a sua origem saem em `polaridade_vereditos`.

A referência humana (HITL) tem precedência sobre a automática quando ambas existem para o item. Vereditos do fallback heurístico são excluídos em toda a análise — um fallback não é uma medição do juiz — e o mesmo vale para itens cuja confiança foi preenchida na desserialização em vez de medida (`n_excluidos_sem_confianca`).

Nenhuma destas sondas prova viés por si só: respostas mais longas podem ser genuinamente melhores. São sinais de inspeção, e os relatórios dizem-no nos próprios campos `nota`.

## Observabilidade

Além da contabilidade por corrida em `summary.json`, uma corrida pode enviar traces e métricas para uma plataforma externa. Defina `LLM_EVAL_TELEMETRY` com um ou mais destinos:

| Destino | Para onde | Requer |
|---|---|---|
| `jsonl` | `telemetry.jsonl` na pasta da corrida | nada |
| `phoenix` | Arize Phoenix via OTLP | `--extra observability` |
| `langsmith` | endpoint OTLP do LangSmith | `--extra observability` + `LANGSMITH_API_KEY` |
| `otlp` | qualquer coletor OTLP (inclui ADOT → CloudWatch) | `--extra observability` |
| `cloudwatch` | métricas CloudWatch em EMF no stdout | agente CloudWatch |

```bash
uv sync --extra observability
LLM_EVAL_TELEMETRY=phoenix,cloudwatch uv run llm-eval --config configs/default.yaml
```

Um contrato, vários adaptadores: uma corrida contém itens, um item contém chamadas LLM. Phoenix, LangSmith e CloudWatch falam todos OTLP — Phoenix nativamente, LangSmith pelo seu endpoint OTLP, CloudWatch pelo coletor ADOT — pelo que um só exportador serve os três, com nomes de atributos segundo as convenções OpenInference/OTel (`llm.token_count.*`, `gen_ai.*`). O CloudWatch tem um segundo adaptador para *métricas*, que emite Embedded Metric Format no stdout para o agente converter, sem que credenciais AWS entrem no processo de avaliação.

Três invariantes tornam isto seguro de deixar ligado:

- **Nunca altera resultados.** `predictions.jsonl` e `summary.json` são idênticos com e sem exportador — há um teste que o afirma.
- **Nunca derruba a corrida.** Backend em baixo, extra em falta ou destino errado produzem um aviso em `stderr` e a corrida continua. Falhar uma avaliação por causa da sua instrumentação é trocar o objectivo pelo instrumento.
- **Não exporta conteúdo por omissão.** Perguntas, respostas e contexto ficam de fora salvo `LLM_EVAL_TELEMETRY_CONTENT=1`. Um endpoint de observabilidade é mais um sítio onde o corpus passa a existir, muitas vezes fora do controlo de quem corre a avaliação.

`jsonl` é o destino de referência: mostra exactamente o que seria enviado, sem rede — útil antes de ligar um backend, e em CI. Detalhes em [`docs/specs/011-telemetry.md`](docs/specs/011-telemetry.md).

## Performance

O trabalho por item é dominado por latência de API, não por CPU. `llm.concurrency`
(ou `LLM_EVAL_CONCURRENCY`) processa itens num pool de threads; a ordem e o conteúdo
de `predictions.jsonl` **não** dependem do valor — `on_record` é sempre chamado pela
ordem do dataset, numa única thread.

```yaml
llm:
  timeout_seconds: 120
  concurrency: 4     # 1 = sequencial (padrão)
```

Medido com mock de 150 ms por chamada, 60 itens sobre 10 documentos
(`gerador + juiz` por item — a forma do FairytaleQA):

| Concorrência | Tempo | Aceleração |
|---|---|---|
| 1 (padrão) | 19,0 s | 1,0× |
| 4 | 4,8 s | 4,0× |
| 8 | 2,6 s | 7,3× |

Três otimizações sustentam isto:

| Otimização | Onde | Efeito |
|---|---|---|
| Pool de itens | `pipeline.run_batch` | sobrepõe a latência de API entre itens |
| Pool HTTP keep-alive | `llm_client.OpenAiCompatibleClient` | elimina 1 handshake TLS por chamada (~2000 numa corrida de 1025 itens) |
| Cache de embeddings | `retrieval.CachingEmbedder` | 84,8% de acerto no cenário acima; deduplica chunks entre itens e entre recuperação e verificação |

Subir a concorrência aumenta a pressão sobre o rate limit; o cliente faz backoff com
jitter e respeita `Retry-After`. Contabilização de tokens e latência é thread-local,
por isso `meta.observabilidade` continua a ser por item.

## Statistical methods

| Uso | Método | Implementação |
|---|---|---|
| Incerteza de proporções (revocação, falso alarme) | Intervalo de Wilson | `statistics.wilson_ci` |
| Concordância entre camadas de verificação | Cohen's κ | `statistics.cohen_kappa` |
| Diferença entre corridas **sobre os mesmos itens** | McNemar (exato ou χ² com correção) + bootstrap emparelhado | `statistics.mcnemar_test`, `paired_bootstrap_diff_ci` |
| Diferença entre corridas sem itens comuns | Teste z de duas proporções | `evaluation_metrics._pairwise_significance` |
| Calibração da confiança do juiz | ECE / MCE sobre bins de fiabilidade | `statistics.expected_calibration_error` |
| Concordância entre amostras repetidas do juiz | κ de Fleiss | `statistics.fleiss_kappa` |

`--compare-runs` alinha as corridas por `id_item` e emite `significancia_emparelhada`
quando há sobreposição. Comparar duas configurações sobre o mesmo dataset é um desenho
emparelhado: o teste não-emparelhado sobrestima o erro-padrão e perde poder, e por isso
fica reservado a corridas sem itens comuns.

## Benchmarks

Resultados agregados versionados em [`assets/benchmarks/comparatives.json`](assets/benchmarks/comparatives.json). Regenerar a partir de corridas locais: [`assets/benchmarks/README.md`](assets/benchmarks/README.md).

## Further reading

| Documento | Conteúdo |
|-----------|----------|
| [`docs/`](docs/README.md) | Arquitetura, specs verificáveis, ADRs e fichas técnicas |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Camadas de verificação e fronteira sistema ↔ harness |
| [`docs/decisions/`](docs/decisions/README.md) | ADRs (tipos de referência, agregação híbrida, planos HITL) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Ambiente, testes e PRs |
| [`CHANGELOG.md`](CHANGELOG.md) | Histórico de versões |
| [`assets/benchmarks/README.md`](assets/benchmarks/README.md) | Comparativos e regeneração |

## Related projects

| Projeto | Foco |
|---------|------|
| [RAGAS](https://github.com/explodinggradients/ragas) | Métricas RAG (faithfulness, context precision/recall) |
| [TruLens](https://github.com/truera/trulens) | Observabilidade em apps LLM/RAG |
| [ARES](https://github.com/stanford-futuredata/ARES) | Avaliação automática de RAG |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Benchmarks de LLM (não RAG end-to-end) |

## License

MIT — ver [`LICENSE`](LICENSE).

O corpus [`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR) é **Apache-2.0**; este repositório consome-o via Hugging Face Hub, sem redistribuição. Citação: [Xu et al., ACL 2022](https://aclanthology.org/2022.acl-long.34); tradução pt-BR: [Leite et al., ECTEL 2024](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR#citation).

# Changelog

Alterações relevantes do projeto, no formato [Keep a Changelog](https://keepachangelog.com/). Datas em UTC.

## [Unreleased]

### Added

- `docs/evidencia/` is versioned. The README's headline table linked
  `docs/evidencia/judge_ab_fairytale_200.json` for its aggregates, and that directory had
  never been committed — a 404 for anyone cloning the repository, breaking invariant 5 and
  invariant 7 at once on the project's central claim. `publish_run_evidence.py` now writes
  there instead of the equally gitignored `assets/evidencia/`: one publication path, versioned.
- `llm-eval --rescore-lexical DIR`: recomputes `meta.metricas_lexicas` from the recorded
  answer and references with the current code, then re-aggregates — no API calls. The
  Portuguese normalisation had landed in code but in no evidence: all four recorded runs
  predate it, so every published F1 and ROUGE-L came from the ASCII tokenizer, and the
  judge's accuracy and kappa are computed against that F1. Writes alongside the originals
  (`predictions.rescored.jsonl`, `summary.rescored.json`) with a `reanalise` block, because a
  rescored run is a re-analysis and not the same result. The metric set is inferred from the
  artifact, since three of the four runs point at a `configs/_tmp_*.yaml` that no longer exists.
- Bootstrap confidence intervals for Cohen's kappa (`bootstrap_kappa_ci`) and ECE
  (`bootstrap_ece_ci`), wired into `judge_report.json` and both READMEs. These were the two
  columns the judge ranking was argued on and the only ones published without an interval.
  With them the table reads differently: two of four judges have a kappa interval containing
  zero, and the two positive intervals overlap, so no judge is demonstrably more
  discriminative than another at N=200.
- Holm-Bonferroni correction across `pairwise_paired_significance`. Four arms produce six
  simultaneous comparisons, where the chance of at least one false positive approaches 26%.
  Both p-values stay in the artifact, and the adjusted values are not rounded — rounding to
  six decimals turned the ablation's 1.6e-14 into 0.0.
- `efeito_minimo_detectavel` on every paired comparison (`mcnemar_mde`). "All pairs p=1" read
  alone invites the stronger claim that the judges are equivalent; the MDE is the difference
  between that and "this design distinguished nothing". Zero discordant pairs yields no MDE
  and says so, rather than implying equivalence.
- `protocolo_sha256` in the provenance block. `config_hash_sha256` did not identify a run:
  two A/B arms shared `2bd0b59d` because models come from the environment and not the YAML,
  putting the experiment's independent variable outside the reproducibility hash.
  `config_hash` is untouched, so `--resume` keeps working.
- A CI `evaluation` job that runs the full pipeline offline, writes real artifacts through the
  CLI's own code path, audits them with `--strict`, and gates the KPIs against a versioned
  golden (`tests/fixtures/ci_kpi_golden.json`, tolerance 1e-6, regenerate with
  `LLM_EVAL_REGENERAR_GOLDEN=1`). CI previously audited a one-line fixture whose reference type
  is `answer_lists` with every layer off, so the auditor's `lexical` branches — the reference
  case's own — were never exercised, and no step produced a `summary.json` to audit.
- `scripts/bench_concurrency.py` and `docs/evidencia/bench_concorrencia.json`: the concurrency
  speedup and embedding cache hit rate had no script, no test and no artifact. Measured
  18.95/4.76/2.56 s (1.0x/3.98x/7.40x) and 84.7% (508 hits, 92 misses).
- `docs/evidencia/embedding_sweep_fairytale_200.json`: the FP/FN curve for
  `embedding_min_cosine` that `docs/calibracao_embedding.md` had promised since v0.4.1. It
  shows there is no optimum — recall stays between 1% and 16% across 0.10-0.50, because cosine
  grounding and lexical overlap are near-independent signals.
- `tests/test_dashboard_render.py` drives the dashboard's render functions against real
  artifacts through a Streamlit double, and `dashboard/app.py` is no longer omitted from
  coverage: 1160 lines of delivered surface were outside the denominator, which is how the 80%
  gate passed while no render function ran in any test.
- Per-metric denominators in `sumario_lexical` (`n_por_metrica`), `meteor_indisponivel`, and
  `idioma_normalizacao`.
- A `Makefile` for the documented flows and a 3.11/3.12/3.13 CI matrix.

### Changed

- **METEOR is off by default**, in the configs and in the code default, and
  `validate_protocol` refuses `meteor: true` when the NLTK `wordnet` corpus is missing — which
  `uv sync` does not install. Without it METEOR only scores near-identical pairs, so a clean
  install computed the mean over the easiest 2% of items and published it beside the full N.
  Failing before the first paid call is the point; disclosure after 200 calls is not enough.
- `configs/hotpotqa_ponte.yaml` sets `idioma: en`. It is an English corpus being normalised
  with the Portuguese rules, so the byte-for-byte SQuAD comparability that the `en` value
  exists to provide was unreachable from any shipped config — a regression introduced by the
  Portuguese fix itself. A test walks every config and asserts the language matches the corpus.
- The Portuguese prompts are written in the variety they declare. All of them opened with
  "português do Brasil" in pre-1990 European orthography (78 occurrences): an uncontrolled
  variable in the layer whose kappa is published.
- Both READMEs lead with the method and name the Portuguese case as a worked example, stating
  what it excludes: no pt-PT corpus, no adult domain, and no handling of the pt-PT/pt-BR
  orthographic split, so a correct pt-PT answer scores lower against a pt-BR gold.
- The judge A/B evidence file declares both confounds per arm — the local arm's different
  generator, and the `gpt-4o-mini` arm where the judge is the generator — and no longer
  asserts the conclusion both READMEs had withdrawn.
- `assets/benchmarks/comparatives.json` (schema 1.2): every entry carries
  `proveniencia.artefactos_presentes`. All seven are false, which is the truth — the runs
  behind them no longer exist. METEOR is published only with its own denominator, and the
  "evolution" entry is relabelled history rather than comparison, since it varied YAML,
  embedding calibration and generation parameters at once.
- `scripts/ablacao_recuperacao.py --reagregar` recomputes the ablation summary offline from
  recorded predictions. The stored artifact ran McNemar on the naive judge-approval KPI that
  SPEC-013 exists to refute (p=0.754/0.065/0.180 where the product metric gives p=1.6e-14), so
  the reproduction path contradicted the publication.
- `scripts/sweep_embedding_threshold.py` reads the reference via `referencia_incorreta`. It
  skipped every item without a boolean `gold_correct`, i.e. it returned a table of zeros on
  every lexical config — on the entire reference corpus.
- `docs/calibracao_embedding.md` states the shipped thresholds (0.24, 0.26 in `tuned`) instead
  of 0.28, with the measured curve and why the value is a conservative floor.

### Fixed

- Credential redaction covers header-form credentials (`x-api-key: <value>`) and keys with no
  vendor prefix (Google `AIza…`, AWS `AKIA…`, Slack `xox[bpsare]-…`). The query-string pattern
  required `?` or `&`, and the header form is what a provider response body contains when it
  echoes the headers it received — the incident behind invariant 1.
- Four `meta` writers now pass through `redact_secrets`: the responder's and critic's
  `structured_output_error`, the lexical metrics error, and the RAGAS adapter. Only
  `pipeline._failed_record` did.
- `audit_run.py --strict` no longer fails a run over items that never produced metrics because
  execution failed.
- The file-lock test collects worker exceptions instead of failing with an `IndexError` that
  pointed at the ordering logic when a lock timeout was the actual cause.

---

### Ciclo anterior, ainda não lançado

Entradas acumuladas desde a v1.0.0, antes do trabalho acima.

#### Added

- Estatística emparelhada para comparar corridas sobre os mesmos itens: teste de McNemar (exato ou χ² com correção de continuidade) e IC bootstrap emparelhado (`statistics.mcnemar_test`, `statistics.paired_bootstrap_diff_ci`). `--compare-runs` alinha por `id_item` e emite `significancia_emparelhada`; `run_comparison.json` passa a `versao_esquema: "2"`.
- Concorrência de itens em `run_batch` via `llm.concurrency` no YAML ou `LLM_EVAL_CONCURRENCY` (padrão 1). Medido com `scripts/bench_concurrency.py`: 3,98× com 4 workers, 7,40× com 8 (mock de 150 ms/chamada, 60 itens sobre 10 documentos).
- `retrieval.CachingEmbedder`: memoriza embeddings por texto entre itens e entre recuperação e verificação (84,7% de acerto no cenário acima: 508 acertos, 92 faltas).
- Documentação técnica publicada no repositório: `docs/ARCHITECTURE.md`, `docs/specs/`, `docs/decisions/`, `docs/techniques/`, `docs/metrics.md` (notas internas continuam locais).
- Secções `Performance` e `Statistical methods` no `README.md`.
- Evidência gravada do A/B de juízes em `docs/evidencia/judge_ab_fairytale_200.json` e o config que a reproduz (`configs/ptbr_fairytale_judge_ab.yaml`): quatro juízes sobre os mesmos 200 itens, com uso de tokens por modelo, testes emparelhados e limitações declaradas.

- Meta-avaliação do juiz ([SPEC-010](docs/specs/010-judge-meta-evaluation.md)): `judge_meta.py` com calibração (ECE/MCE), concordância com a referência (confusão 2×2, κ de Cohen, IC de Wilson), sondas de viés de verbosidade (ponto-bisserial) e de posição (taxa de aprovação por rank do chunk ouro). CLI `llm-eval --judge-report RUN_DIR` grava `judge_report.json` sem API.
- `scripts/judge_self_consistency.py`: N vereditos repetidos por item para medir estabilidade do juiz; agregado por `judge_meta.self_consistency` (κ de Fleiss, taxa de unanimidade) e ligado via `--judge-samples`.
- Primitivos estatísticos: `expected_calibration_error`, `fleiss_kappa`, `point_biserial`.
- `README.md` passa a inglês (versão principal); o texto português vive em `README.pt-BR.md`, com link recíproco.

- Juiz em fornecedor separado do gerador: `JUDGE_BASE_URL` e `JUDGE_API_KEY` (herdam os do gerador quando omitidos). Permite gerador em API paga com juiz local gratuito — ~88% do custo de uma corrida está no juiz — e reforça a independência entre avaliador e avaliado. `protocolo_ativo.models` regista ambos os endpoints (só scheme+host).
- `configs/ptbr_fairytale_qwen_local.yaml`: 200 itens, gerador em API e juiz `qwen2.5:7b` no Ollama, com gate de custo.
- Presets de fornecedor em `.env.example` (Ollama, vLLM, DeepSeek, DashScope/Qwen, OpenRouter) e secção *Fornecedores* nos dois READMEs, com o smoke comparativo de juízes locais.
- `PermanentApiError`: 4xx não transitório passa a citar a resposta do fornecedor (ex.: `model 'qwen2.5:7b' not found`) e não é retentado ao nível do item.

- Telemetria externa ([SPEC-011](docs/specs/011-telemetry.md)): traces e métricas por item e por corrida para Arize Phoenix, LangSmith, qualquer coletor OTLP (incl. ADOT → CloudWatch), métricas CloudWatch em EMF, e um destino `jsonl` local sem dependências. Ativa-se com `LLM_EVAL_TELEMETRY`; extra `observability` só para os destinos OTLP. Fecha a Fase 8 da SPEC-003.
  Invariantes garantidos por teste: artefactos idênticos com e sem telemetria, exportador que rebenta não derruba a corrida, e conteúdo (pergunta/resposta) não é exportado sem `LLM_EVAL_TELEMETRY_CONTENT=1`.

#### Changed

- `OpenAiCompatibleClient` reutiliza um `httpx.Client` com pool keep-alive em vez de criar um por chamada; backoff de retry passa a ter jitter.
- `UsageAccumulator` e `OpenAiCompatibleClient.last_usage` passam a armazenamento thread-local, para que `meta.observabilidade` continue correto por item com workers concorrentes.
- `Retriever.retrieve` deixa de embeber a pergunta duas vezes no caminho de remoção do chunk ouro.

#### Fixed

- Custo por modelo: um par único de preços aplicado a gerador e juiz distintos errou por **9,7×** numa corrida gravada ($0,17 reportado contra $1,69 real). `meta.observabilidade` reparte tokens por modelo e `LLM_EVAL_PRICES` dá custo por modelo; modelos sem preço são listados em vez de desaparecerem do total. O parser passa a separar pela direita, para aceitar etiquetas do Ollama (`qwen2.5:7b`).
- Falhas de execução deixam de contaminar a estatística emparelhada. `_failed_record` marca `flag_anomalia` para revisão, o que fazia uma corrida com 9 falhas de quota aparecer com anomalias "exclusivas": o McNemar dava p=0,004 a medir propagação de faturação. Excluídas, todos os pares dão p=1.
- `insufficient_quota` chega como 429 mas nunca recupera: passa a falhar à primeira citando a mensagem do fornecedor, em vez de 3 tentativas com 30 s de backoff.
- Modelos que só aceitam a temperatura por omissão (ex.: `gpt-5-mini`) rejeitavam o `temperature=0` do juiz e caíam 100% no fallback heurístico — que responde `sustentado`, fazendo um juiz avariado parecer permissivo. O cliente repete sem o parâmetro e assinala `temperature_rejected`.
- Cinco links quebrados nos documentos publicados (dois caminhos relativos errados, três para configs `nq_open` removidos).
- Endpoint de chat deixa de duplicar `/v1`: bases já terminadas em `/v1` (Ollama, vLLM, OpenRouter, DashScope) davam `/v1/v1/chat/completions` e um 404 — na prática, nenhum fornecedor não-OpenAI funcionava apesar de o README o anunciar.
- `Retry-After` do servidor deixa de poder ser encurtado: o jitter simétrico aplica-se só ao backoff interno; a directiva do servidor recebe apenas jitter positivo.
- Corrida concorrente passa a ser interrompível: a submissão usa uma janela deslizante e o pool é fechado com `cancel_futures`, em vez de esperar por todos os itens já enfileirados no `Ctrl+C`.
- `LLM_EVAL_INTER_ITEM_SLEEP` volta a ser respeitado com `concurrency > 1` (relógio partilhado entre workers, taxa agregada preservada); antes desaparecia em silêncio.
- `max_connections` do pool HTTP passa a derivar da concorrência (`pool_size_for_concurrency`); antes ficava fixo em 32 e workers excedentes falhavam com `PoolTimeout`.
- `--compare-runs` desambigua diretórios com o mesmo basename; antes colapsavam e a análise emparelhada era omitida em silêncio.
- `--judge-report` lê a polaridade dos vereditos (`judge_aggregation_verdicts`) e o limiar léxico (`pattern_settings.f1_fraca_min`) de `protocolo_ativo`, em vez de assumir `sustentado`/default global — `incompleto` consultivo já não conta como falso negativo do juiz.
- Confiança preenchida na desserialização é marcada com `confianca_ausente` e excluída do ECE.
- `scripts/judge_self_consistency.py` reproduz `judge_prompt_style` e `judge_max_context_chars` da corrida; antes reamostrava sem tecto de contexto, medindo um prompt diferente do avaliado.
- `--judge-samples` com JSONL malformado sai com código 2 em vez de traceback.
- `LlmCallUsage` passa a registar `started_at` e `endpoint`, necessários para posicionar spans e distinguir juiz local de API.
- `protocolo_ativo` passa a registar `judge_max_context_chars` (necessário ao replay do juiz).

## [1.0.0] — 2026-06-06

Primeira publicação como **rag-eval-harness** — harness reprodutível para pipelines RAG + LLM, com FairytaleQA pt-BR como caso de referência.

### Added

- Adaptador FairytaleQA pt-BR ([`benjleite/FairytaleQA-translated-ptBR`](https://huggingface.co/datasets/benjleite/FairytaleQA-translated-ptBR)) e configs `default.yaml`, `ptbr_fairytale_full.yaml`, `ptbr_fairytale_tuned.yaml`.
- Smoke offline: `configs/smoke_amostra.yaml` (2 itens, sem Hub); CI com `test_pipeline_e2e_mock.py`.
- Saída estruturada JSON para respondedor, crítico e juiz (`structured_output.py`).
- Prompts empacotados em `src/llm_evaluation/prompts/` com teste de integridade no CI.
- Verificação multicamada: embedding, juiz RAG pt, referência léxica; agregação configurável.
- Padrões determinísticos, fila de revisão humana e HITL no dashboard.
- Dashboard Streamlit offline (`llm-eval-dashboard`).
- Artefactos auditáveis e scripts `audit_run.py`, `publish_run_evidence.py`, `validate_embedding_policy.py`.
- Comparativos versionados em `assets/benchmarks/comparatives.json`.
- Documentação pública: `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`.

### Fixed

- Replay offline da política `embedding_e_juiz` validado em CI.
- Orquestração `multiplo` exige `--experimental`; `meta.flag_critica` é diagnóstico, não entra em `flag_anomalia`.
- Cobertura mínima 75% e `audit_run.py --strict` no workflow CI.
- Chaves genéricas `por_camada_vs_referencia` em `analise_camadas`.
- Contrato do respondedor alinhado ao schema JSON empacotado.

### Changed

- Nome do projeto: **rag-eval-harness** (import Python mantém `llm_evaluation`).

## [0.4.1] — 2026-05-17

### Added

- Fila de revisão humana pós-corrida (`fila_revisao`) e aba no dashboard.
- Secção `operacional` no YAML e `sumario_operacional` no `summary.json`.
- `--experimental` obrigatório para orquestração `multiplo`.
- `docs/calibracao_embedding.md`.

### Changed

- Export da fila antes do `write_summary` final; manifest com checksum do CSV.
- Validação strict de `protocolo_ativo` em `schema_registry`.

## [0.4.0] — 2026-05-16

### Added

- `pattern_detection` → `meta.diagnostico` e `sumario_padroes`.
- Políticas `todos_criticos` e `embedding_e_juiz`.
- Embedding vs passagem ouro (`embedding_max_coseno_ouro`).
- Juiz RAG EN (`judge_prompt_style: rag_en`).
- Dashboard: abas Inspector Q/A e Padrões.

### Changed

- `verify_item` usa máximo coseno sobre recuperados ∪ gold chunk.
- Gate de recuperação e agregação calibrados em `nq_open_rag.yaml`.

## [0.3.0] — 2026-05-16

### Added

- Spec-driven development (`docs/specs/`).
- Adaptadores intercambiáveis e `dataset.reference_type`.
- Dashboard Streamlit e métricas de recuperação (SPEC-001).
- Configs Natural Questions e TruthfulQA.

### Changed

- Baselines `so_embeddings` / `so_juiz` com `verify_gold=false` para ablação honesta.
- README e docs alinhados ao harness dataset-agnóstico.

### Removed

- `generation.num_samples` do YAML (não implementado).

## [0.2.0] — 2026-05-02

### Added

- Estatística rigorosa: IC de Wilson e Cohen's kappa em `summary.json`.
- Persistência incremental de `predictions.jsonl`.
- Testes E2E mockados e carregamento unificado de datasets.

### Fixed

- Chamada METEOR alinhada à assinatura NLTK ≥3.9.

### Changed

- Localização PT-BR: prompts, vereditos do juiz e chaves de artefactos.
- Cliente LLM: respeita `temperature`/`max_tokens`, retry com backoff, juiz a temp 0.
- Pipeline reutiliza embedder e clientes HTTP por corrida.
- `HashEmbedder` determinístico entre processos (`hashlib.blake2b`).

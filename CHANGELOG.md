# Changelog

Alterações relevantes do projeto, no formato [Keep a Changelog](https://keepachangelog.com/). Datas em UTC.

## [Unreleased]

### Added

- Estatística emparelhada para comparar corridas sobre os mesmos itens: teste de McNemar (exato ou χ² com correção de continuidade) e IC bootstrap emparelhado (`statistics.mcnemar_test`, `statistics.paired_bootstrap_diff_ci`). `--compare-runs` alinha por `id_item` e emite `significancia_emparelhada`; `run_comparison.json` passa a `versao_esquema: "2"`.
- Concorrência de itens em `run_batch` via `llm.concurrency` no YAML ou `LLM_EVAL_CONCURRENCY` (padrão 1). Medido: 4,0× com 4 workers, 7,3× com 8 (mock de 150 ms/chamada, 60 itens).
- `retrieval.CachingEmbedder`: memoriza embeddings por texto entre itens e entre recuperação e verificação (84,8% de acerto no cenário acima).
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

### Changed

- `OpenAiCompatibleClient` reutiliza um `httpx.Client` com pool keep-alive em vez de criar um por chamada; backoff de retry passa a ter jitter.
- `UsageAccumulator` e `OpenAiCompatibleClient.last_usage` passam a armazenamento thread-local, para que `meta.observabilidade` continue correto por item com workers concorrentes.
- `Retriever.retrieve` deixa de embeber a pergunta duas vezes no caminho de remoção do chunk ouro.

### Fixed

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

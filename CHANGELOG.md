# Changelog

Alterações relevantes do projeto, no formato [Keep a Changelog](https://keepachangelog.com/). Datas em UTC.

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

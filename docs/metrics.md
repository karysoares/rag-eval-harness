# Métricas e baselines (resumo operacional)

Fonte de verdade: [`specs/`](specs/). Arquitetura: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Métricas de recuperação (pré-geração)

Por item, em `meta.metricas_recuperacao` (agregado em `summary.json` → `sumario_recuperacao`):

| Campo | Significado |
|-------|-------------|
| `score_melhor_chunk` | Coseno pergunta ↔ chunk top-1 |
| `rank_chunk_ouro` | Posição 1-indexed do chunk marcado `e_ouro` no top-k; `null` se não aparecer |
| `chunk_ouro_no_top_k` | O chunk de referência do item está entre os recuperados |
| `corpus_tem_chunk_ouro` | O adaptador forneceu `rag_gold_chunk` |

Estas métricas são **diagnósticas** (qualidade do RAG), não entram em `anomaly_flag` por defeito.

## Erro gold (`reference_type: answer_lists`)

**Entrada**: pergunta `q`, resposta do modelo `a`, listas `correct_answers` / `incorrect_answers` do adaptador.

**Protocolo de normalização** (implementado em `src/llm_evaluation/verification/gold.py`):

1. Lowercase, remoção de pontuação terminal simples, colapso de espaços.
2. **Correto** se `a_norm` é substring de algum `correct_answers_norm` **ou** algum `correct_answers_norm` é substring de `a_norm` (cobrir respostas curtas).
3. **Incorreto explícito** se match a `incorrect_answers` pelo mesmo critério (opcionalmente usado como sinal adicional).
4. **Recusa (refusal)** (heurística): resposta muito curta (`len < 20`) com padrões como “não sei”, “cannot”, “don't know”, “sem comentário” — contabilizada à parte; não confundir com “incorreto factual” nos agregados.

**Limitação**: juiz humano recomendado em amostra limítrofe para validar o protocolo (ver [`decisions/0001-reference-types.md`](decisions/0001-reference-types.md)).

## Anomalia detectada (pipeline)

Um item marca `flag_anomalia` / `anomaly_flag=True` quando a **política de agregação** dispara. Especificação completa (tabelas-verdade, nulls, multi, roadmap): [`docs/specs/004-aggregation.md`](specs/004-aggregation.md).

| `aggregation.policy` | Comportamento (camadas activas = `verify_*` true) |
|----------------------|---------------------------------------------------|
| `qualquer_critico` | OR: `g ∨ e ∨ j` (default histórico) |
| `todos_criticos` | AND entre camadas activas; nenhuma activa → false |
| `embedding_e_juiz` | Com embedding+juiz: `e ∧ j`; senão degrada para a camada activa (ver spec) |

Gatilhos: **g** = `gold_incorreto`; **e** = `embedding_baixo_suporte`; **j** = veredito negativo **sem** `fallback_heuristico` ([`veredito.py`](src/llm_evaluation/veredito.py)). Valores `null`/`false` não disparam.

Camadas (ligáveis em YAML):

- **Gold**: protocolo do adaptador (se `verify_gold: true`).
- **Embedding**: [SPEC-002](specs/002-grounding.md) — `embedding_max_coseno`, limiar `embedding_min_cosine`.
- **Judge**: [SPEC-003](specs/003-judge.md) — `negative_judge_verdicts`; fallback heurístico **excluído** da agregação.

**Perfis:** `baselines.profile` no YAML só etiqueta `perfil_baseline`; ablação real via `llm-eval --profile` ou `--compare-baselines` (`apply_baseline_profile`).

`summary.json`: `protocolo_ativo`, `detector_activo`, `analise_camadas` (Wilson, Cohen's κ), `estratificacao_fp_gold_correto`. Modo `multiplo`: `meta.flag_critica` é diagnóstico (experimental); não altera `flag_anomalia`.

## Padrões determinísticos (`meta.diagnostico`)

Camada **sem LLM** ([SPEC-007](specs/007-pattern-detection.md)): `pattern_registry.py` + `pattern_detection.py`, gravada por item e espelhada no topo do JSONL como `diagnostico`:

| Campo | Significado |
|-------|-------------|
| `catalog_version` | Versão do catálogo (ex.: `"1.0"`) |
| `padroes` | Lista de tags (ex.: `referencia_forte`, `grounding_fp_suspeito`, `recusa`) |
| `padrao_primario` | Um rótulo por prioridade do registry |
| `tier_qualidade` | `alta` \| `media` \| `baixa` \| `indeterminada` |
| `padroes_meta` | Metadados leves por tag activa (`id`, `categoria`, `severidade`) |

Overrides opcionais em YAML (`patterns.referencia_forte.f1_min`, etc.). Também em `predictions.jsonl`: `referencias` (lista truncada de `correct_answers`) para o dashboard sem re-carregar o Hub.

Agregado em `summary.json` → `sumario_padroes` (`catalog_version`, `por_padrao_primario`, `por_tag`).

**Nota:** `diagnostico` **não** altera `anomaly_flag`; o juiz LLM continua na camada de verificação/agregação.

## Métricas RAG / grounding (implementação simplificada)

Não reimplementamos RAGAS integralmente; usamos **proxy** por coseno. Especificação completa: [`docs/specs/002-grounding.md`](specs/002-grounding.md).

- **`embedding_max_coseno`**: `max` sobre todas as frases da resposta (split por `(?<=[.!?])\s+`, fallback resposta inteira) do `max` coseno a qualquer chunk recuperado; com `embedding_use_gold_chunk: true`, também face a `rag_gold_chunk` (texto integral) — usa o **máximo** dos ramos recuperados e ouro.
- **`embedding_baixo_suporte`**: `embedding_max_coseno < embedding_min_cosine` quando a camada está activa; `null` sem corpus; `0.0` + `true` se há corpus mas `recuperados == []` e sem score gold (ver tabela na spec).
- **Decomposição**: `embedding_max_coseno_recuperados`, `embedding_max_coseno_ouro` no JSONL quando calculados.
- **Recuperação (diagnóstico)**: `rank_chunk_ouro` / `chunk_ouro_no_top_k` em `meta.metricas_recuperacao` ([SPEC-001](specs/001-retrieval.md)).

**Claims / NLI**: futuro opcional; unidade natural alinhada ao split de frases em `embedding_verify.py`.

## Tamanho do conjunto (`dataset.limit`)

- **`mode: demonstracao`** (sinónimos: `demo`): conjunto fixo pequeno (CI); `limit` só corta até ao máximo de exemplos demo.
- **`mode: hub`** (sinónimos: `hf`): dataset tabular no Hub via `hf_repo`. **`limit: 0` ou `null`**: usa **todas** as linhas do `split` após baralhar com `seed`. Valor positivo: primeiras N linhas.

Para corridas reprodutíveis no repositório: `configs/ptbr_fairytale.yaml` (FairytaleQA pt-BR, 64 itens), `configs/smoke_amostra.yaml` (smoke/CI local), `configs/ptbr_fairytale_full.yaml` (validation completo) ou `configs/ptbr_fairytale_tuned.yaml` (validation completo com parâmetros calibrados).

### Tabela multi-dataset (corridas locais)

Agregados versionados em [`assets/benchmarks/comparatives.json`](../assets/benchmarks/comparatives.json). **Não misture N nem `reference_type` entre corridas** — `sumario_lexical` só é comparável dentro do mesmo adaptador e tamanho de amostra. Recuperação e juiz são diagnósticos transversais; ver [`assets/benchmarks/README.md`](../assets/benchmarks/README.md).

### FairytaleQA pt-BR — interpretação léxica (Plano A)

Com `reference_type: lexical`, o KPI de produto no `summary.json` é `sumario_lexical` (METEOR, ROUGE-L, BLEU, F1 token). **BLEU baixo (~0,15–0,25) é normal:** respostas correctas em paráfrase narrativa raramente coincidem n-gramas com a referência curta do dataset. Use METEOR/ROUGE-L para tendência de cobertura semântica; use `sumario_recuperacao` e vereditos do juiz (`judge_prompt_style: rag_pt`) para qualidade RAG e grounding. Calibração de `embedding_min_cosine`: `scripts/validate_embedding_policy.py outputs/run_<id>`.

## Baselines (relatório)

| Perfil (YAML / CLI) | Descrição |
|---------------------|-----------|
| **`nenhum`** | Só geração: verificadores desligados (sinónimo legado: `none`). |
| **`so_embeddings` / `so_juiz`** | Um verificador na agregação (`verify_gold` desligado para isolar embedding ou juiz); ver configs `baseline_*`. |
| **`hibrido`** | Embedding + juiz na agregação (`verify_gold: false` via `--profile`; sinónimo legado: `hybrid`). |

**Métricas de avaliação da pipeline** (sobre conjunto com rótulo gold):

- **Recall de flag**: entre itens gold-incorretos, fração com `anomaly_flag=True`.
- **Falso alarme**: entre itens gold-corretos, fração com `anomaly_flag=True`.

Reportadas em `summary.json` por corrida.

## Artefactos de corrida (SPEC-005)

Cada `outputs/run_<timestamp>/` inclui, nas corridas novas:

| Ficheiro | Conteúdo |
|----------|----------|
| `predictions.jsonl` | Um registo por item; `schema_version` `"1.0"` |
| `summary.json` | Agregados + `metadados_corrida` (git, hash config, modelos, prompt hashes) |
| `manifest.json` | Inventário com SHA256 e `n_linhas` |
| `anomalies.jsonl` / `.csv` | Subconjunto com `flag_anomalia` |

Validação: `uv run python scripts/audit_run.py`. Corridas antigas sem `manifest.json` continuam legíveis (modo aviso). Detalhe: [`specs/005-reporting.md`](specs/005-reporting.md).

## `analise_camadas` (por corrida)

Além da confusão do **detector agregado** (`anomaly_flag`) vs referência do adaptador, o `summary.json` inclui **`analise_camadas`** (ficheiros antigos podem usar a chave `layer_analysis`):

- **Marginais**: quantos itens disparam `gold_incorrect`, `embedding_low_support`, `judge_negative` (cada camada, antes do OR).
- **Combinações exclusivas**: contagens do tipo *gold só*, *embedding só*, *juiz só*, pares, *as três*, *nenhuma*.
- **`por_camada_vs_referencia`**: para cada camada, matriz VP/FP/FN/VN usando referência positiva = item incorreto segundo o adaptador. **Interpretação:** o sinal *gold* alinha com essa referência; *embedding* e *juiz* medem *grounding* / juízo distinto — use para diagnóstico, não como única “acurácia factual”.

Reconstruir relatório só a partir de `predictions*.jsonl`: `uv run llm-eval --analyze-run outputs/run_<id>`.

## Confusão detector × gold (em `summary.json`)

Comparando **anomalia da pipeline** com **referência incorreta** (protocolo depende do adaptador). Chaves em `confusao_vs_referencia` (alias legado `confusao_vs_gold`):

- **VP** (`vp_gold_incorreto_marcado`): gold incorreto e `anomaly_flag=true`
- **FN** (`fn_gold_incorreto_nao_marcado`): gold incorreto e não marcado
- **FP** (`fp_gold_correto_mas_marcado`): gold correto mas marcado
- **VN** (`vn_gold_correto_nao_marcado`): gold correto e não marcado

Incluem-se `precisao_anomalia_vs_gold_incorreto`, `revocacao_anomalia_vs_gold_incorreto` e `acuracia_balanceada_gold` (chaves legadas em inglês ainda lidas ao importar relatórios antigos).

### Incerteza e concordância (novo)

- **`ic95_*`** — Intervalo de Wilson 95 % (assimétrico, robusto em amostras pequenas) para precisão, revocação e taxa de falso alarme. Reportado tanto a nível **agregado** (`summary.json`) como **por camada** (`analise_camadas.por_camada_vs_referencia`).
- **`cohen_kappa_anomalia_vs_gold`** — Concordância da pipeline com o rótulo de referência para além do acaso. Convenção comum: > 0,6 substancial; 0,2 – 0,6 moderado; < 0,2 fraco.
- **`analise_camadas.concordancia_entre_camadas`** — Cohen's kappa **par a par** entre `sinal_ouro`, `embedding` e `juiz`. Permite mostrar quanto os sinais se reforçam ou divergem entre si, indo além de só comparar contra a referência.
- **`analise_camadas.por_camada_vs_referencia.<camada>.cohen_kappa_vs_gold`** — Mesma medida, separada por camada.

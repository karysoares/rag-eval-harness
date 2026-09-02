# Premissas do projeto — avaliação de sistemas LLM

Este documento fixa o **objetivo central** e as regras de desenho. Qualquer corpus (FairytaleQA pt-BR, Hub genérico, amostra local) entra via **adaptador**, não como núcleo do repositório.

## Objetivo central

O objectivo é disponibilizar um **harness de avaliação** reprodutível e auditável: sinais de verificação configuráveis, política de agregação explícita, relatórios estatísticos e adaptadores de dataset intercambiáveis — **sem** reduzir o sistema a um único score de benchmark.

O artefacto central é o **método e a infraestrutura de medição**, não o desempenho pontual num corpus específico.

## Modelo de avaliação (três ramos + retrieval)

Diagrama completo: [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
QUESTION → RETRIEVAL → [métricas recuperação] → GENERATION → ANSWER
                                                      ↓
                    ┌─────────────┬─────────────┬─────────────┐
                    │  Grounding  │   Quality   │  Reference  │
                    │  (contexto) │   Judge     │  (opcional) │
                    └─────────────┴─────────────┴─────────────┘
                                      ↓
                              agregação → anomaly_flag
```

## Três camadas de software (separar sempre)

| Camada | O que é | Depende do dataset? |
|--------|---------|---------------------|
| **Harness** | Pipeline, sinais, agregação, CLI, estatística, JSONL | Não |
| **Adaptador** | Carregar linhas (HF, demo, ficheiro) → `EvalItem` | Sim |
| **Referência** | O que conta como “erro” ou rótulo para métricas (`reference_type`) | Sim (pode haver vários tipos) |

**Regra:** não fundir harness com um único tipo de referência. Código, métricas e README devem deixar claro qual `reference_type` está activo.

## O que priorizar (sinal de expertise)

- Arquitetura em camadas: gerar → recuperar → verificar → agregar → reportar.
- Sinais independentes (`gold`, embedding, juiz) e `analise_camadas` (kappa, combinações exclusivas).
- LLM-as-judge com contrato JSON, temperatura 0 no juiz, limitações documentadas.
- RAG como objeto de avaliação: retrieval, gate de recuperação fraca, proxies de grounding (sem fingir RAGAS completo).
- Corridas reprodutíveis: YAML, seed, `on_record`, `--analyze-run` / `--compare-runs` sem API.
- Ablation honesta: perfis `nenhum` / `so_embeddings` / `so_juiz` / `hibrido` com **uma variável de verificação de cada vez** na agregação (`verify_gold` desligado quando o perfil isola embedding ou juiz).
- Limitações explícitas no código e na doc (coseno ≠ entailment; juiz correlaciona com o modelo escolhido).

## O que evitar (credibilidade com engenheiro GenAI)

- KPI principal baseado só em **substring gold** fora de datasets com listas `correct`/`incorrect` explícitas (`reference_type: answer_lists`).
- Benchmark externo (RAGAS, etc.) como **narrativa principal** — é diagnóstico cruzado, não ground truth.
- Métricas léxicas (BLEU, ROUGE, METEOR) como estrela em respostas abertas; só quando o adaptador declarar referência textual fechada.
- Config YAML que **não altera comportamento** (`num_samples`, políticas de agregação não implementadas) — implementar ou remover.
- Baselines com nome enganoso (ex.: “só embedding” com `verify_gold: true` na agregação).
- Modo multi-agente (crítico) sem evidência de calibração vs referência.
- Fallback do juiz que marca `sustentado` após falha de API/parse sem flag visível no relatório.
- Corridas grandes (1000+ itens) apresentadas como prova de expertise sem calibração de limiares.

## Escolha de dataset (critérios)

Preferir datasets onde se contem histórias distintas com contexto recuperável:

1. **Grounding / faithfulness** — resposta vs contexto (embedding + juiz).
2. **Rubrica / qualidade** — juiz com critérios fixos.
3. **Referência forte opcional** — resposta curta do corpus (`lexical`) ou listas de labels; aí gold ou léxico fazem sentido.

Se o dataset só tiver `question` + `answer` + `context`, o modo natural é **grounding + juiz**, não substring em listas de respostas.

## Implementação e PRs

- Novos adaptadores: documentar colunas, tipo de referência e quais verificadores ligar.
- Métricas genéricas (`*_vs_referencia`); não introduzir sufixos de dataset em código novo.
- Não adicionar features ao YAML sem implementação no pipeline.
- Testes: manter E2E mockado; integração com API só com marker `integration`.
- Um dataset pequeno bem explicado > batch enorme com métrica inadequada.

## Documentos relacionados

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — fluxo, ramos de avaliação, mapeamento ao código
- [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) — gate científico
- [`metrics.md`](metrics.md) — definições operacionais (actualizar ao mudar referência)
- [`decisions/0001-reference-types.md`](decisions/0001-reference-types.md) — `reference_type` lexical / answer_lists / none

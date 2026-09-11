# Evidência

Agregados **versionados** das corridas de que saem os números publicados no `README`.
Sem PII: só sumários, contagens e testes emparelhados. Os `predictions.jsonl` completos
ficam locais (`outputs/`, gitignored).

Esta pasta é versionada de propósito. Um número publicado tem de vir de uma corrida
gravada (invariante 5 do [`CLAUDE.md`](../../CLAUDE.md)), e um link publicado para um
ficheiro ignorado pelo git é um 404 para quem clona — o que torna a invariante 7
inverificável ao mesmo tempo.

Publicar a partir de uma corrida: `uv run python scripts/publish_run_evidence.py outputs/run_<id>`.

## Conteúdo

| Ficheiro | Conteúdo | N |
|---|---|---|
| [`judge_ab_fairytale_200.json`](judge_ab_fairytale_200.json) | A/B de quatro juízes sobre os mesmos itens: exactidão, κ, ECE, confiança, custo por modelo, seis testes emparelhados. Cada braço declara o seu gerador e se o juiz é auto-referente. | 200 |
| [`judge_local_gerador_partilhado_93.json`](judge_local_gerador_partilhado_93.json) | Repetição **parcial** do braço de juiz local com o gerador partilhado, para separar juiz de gerador. Corrida incompleta (créditos da API esgotados ao item 94) e subconjunto não aleatório — declara o que estabelece e o que não estabelece. | 93 de 200 |
| [`ablacao_hotpotqa_100.json`](ablacao_hotpotqa_100.json) | Ablação recuperação → geração (SPEC-013): o KPI ingénuo inverte o sentido do efeito. | 100 queries |
| [`embedding_sweep_fairytale_200.json`](embedding_sweep_fairytale_200.json) · [`.csv`](embedding_sweep_fairytale_200.csv) | Curva FP/FN/precisão/revocação de `embedding_min_cosine` de 0,10 a 0,50. Mostra que não há ótimo — ver [`../calibracao_embedding.md`](../calibracao_embedding.md). | 200 |
| [`bench_concorrencia.json`](bench_concorrencia.json) | Ganho de concorrência e acerto da cache de embeddings, com o cenário declarado. Regenerar: `make bench`. | 60 itens |

> **Removidos.** `run_ci_fixture_protocolo.json` e `run_ci_fixture_kpi_lexical.json` saíram:
> o segundo tinha todos os valores `null` e nenhum dos dois era lido por ficheiro algum.
> O que o CI precisa de garantir está agora em `tests/fixtures/ci_kpi_golden.json`, com
> KPIs reais de uma corrida offline e um gate que falha no desvio.

## Convenção de nomes

| Padrão | Conteúdo |
|---|---|
| `run_<id>_protocolo.json` | Bloco `protocolo_ativo` + `detector_activo` do `summary.json` |
| `run_<id>_policy_validation.json` | Saída de `scripts/validate_embedding_policy.py --write` |
| `run_<id>_fila_manifest.json` | `analise_manual/fila_revisao_humana.json` |

## Limitações transversais

- **N=200 de um corpus de 1025.** Nenhuma corrida do corpus completo está gravada.
- **Referência automática.** Exactidão e κ medem concordância com F1 léxico, não com um
  humano: o plano C (HITL) está documentado e vazio.
- **O braço local do A/B ainda não tem repetição completa.** A parcial de 93 itens é
  sugestiva, não conclusiva; repetir os 200 exige créditos de API (o juiz local é gratuito).
- **Métricas léxicas anteriores à normalização portuguesa.** As corridas de 2026-09-02
  precedem o tokenizador Unicode do ROUGE; ver `CHANGELOG.md`.

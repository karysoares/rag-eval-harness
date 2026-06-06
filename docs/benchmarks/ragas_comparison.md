# Comparação harness vs RAGAS (eixo externo)

Parte do framework de comparativos — **Plano B**, eixo **externo**. RAGAS é diagnóstico cruzado, não ground truth.

Ver também: [`../../assets/benchmarks/README.md`](../../assets/benchmarks/README.md) · [`../../README.md`](../../README.md#comparativos--quatro-eixos-não-misturar).

## Uso

```bash
uv sync --extra ragas
uv run python scripts/export_comparatives.py --ragas --ragas-n 25
```

Snapshot versionado: [`../../assets/benchmarks/comparatives.json`](../../assets/benchmarks/comparatives.json) → chave `externo_ragas_amostra`.

## O que comparar (mesma amostra)

| Pergunta | Harness | RAGAS |
|----------|---------|-------|
| Resposta ancorada ao contexto? | juiz `sustentado` + embedding | faithfulness |
| Resposta relevante à pergunta? | F1 token / gap RAG–resposta | answer_relevancy |
| Alerta operacional? | `taxa_alerta` (política YAML) | — |

RAGAS também usa LLM; compare sempre na **mesma amostra** de itens da corrida tuned.

## Referência (tuned, N=25)

| Sinal | Harness | RAGAS |
|-------|---------|-------|
| Grounding proxy | 72% juiz sustentado | faithfulness 0,82 |
| Relevância | F1 token 0,45 | answer_relevancy 0,94 |

Regenerar após mudanças de prompt/modelo.

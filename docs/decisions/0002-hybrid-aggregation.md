# ADR 0002: Agregação da verificação híbrida

## Contexto

Combinamos sinais de **gold**, **similaridade embedding** (resposta vs chunks) e **LLM-as-judge**. Precisamos de uma regra clara para `anomaly_flag` e para evitar silêncio quando camadas discordam.

## Decisão

- **Política padrão**: `anomaly_flag = any(critical_signals)` onde cada camada ativa contribui com um booleano:
  - `gold_incorrect` (se `verify_gold: true`)
  - `embedding_low_support` (modo RAG, abaixo do limiar)
  - `judge_negative` (veredito negativo estruturado)
- **Modo opcional** `aggregation: all_critical` (futuro): exige duas camadas — não é o default; pode ser adicionado em YAML se necessário.

## Consequências

- Maior **recall** de anomalias; possível aumento de **falsos alarmes** se o judge ou embedding calibrarem mal — documentar limiares em `configs/default.yaml`.
- Relatórios incluem `signals` por item para auditoria.

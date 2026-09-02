# SPEC-011: Telemetria e observabilidade externa

- **Estado:** implemented
- **Testes:** `tests/test_telemetry.py`
- **Relacionado:** [SPEC-003](003-judge.md) Fase 8 (esta spec fecha-a), [SPEC-005](005-reporting.md) (`observabilidade` em `summary.json`)

## Objetivo

Exportar traces e métricas de uma corrida para plataformas externas — Arize Phoenix, LangSmith, CloudWatch — sem que a avaliação passe a depender delas.

`observability.py` já mede tokens, latência e custo **dentro** da corrida e agrega-os em `summary.json`. Isso responde a "quanto custou e demorou esta corrida". Não responde a "como evoluiu ao longo de semanas", "qual o p95 do juiz", ou "que item ficou pendurado às 3h" — perguntas de séries temporais, que pedem uma plataforma.

## Desenho: um contrato, vários adaptadores

Três destinos, três formatos, uma só informação: uma corrida contém itens, cada item contém chamadas LLM. O modelo é definido uma vez em `telemetry/base.py` (`RunEvent`, `ItemEvent`, `LlmCallEvent`) e traduzido no fim. Acrescentar um destino é escrever um adaptador, não alterar o pipeline.

**OTLP como espinha dorsal.** Phoenix fala OTLP nativamente, LangSmith expõe um endpoint OTLP, e CloudWatch recebe traces através do coletor ADOT. Um adaptador `OtlpExporter` serve os três; escrever três clientes seria triplicar o mesmo mapa de atributos. O CloudWatch tem adaptador próprio apenas para **métricas** (EMF), que são de outra natureza que traces.

Os nomes de atributos seguem as convenções OpenInference/OTel (`llm.token_count.*`, `gen_ai.*`) onde existem, para que Phoenix e LangSmith os reconheçam sem mapeamento manual.

## Destinos

| `LLM_EVAL_TELEMETRY` | Destino | Requisitos |
|---|---|---|
| `jsonl` | `telemetry.jsonl` na pasta da corrida | nenhum |
| `phoenix` | Phoenix via OTLP (`PHOENIX_COLLECTOR_ENDPOINT`) | extra `observability` |
| `langsmith` | LangSmith OTLP (`LANGSMITH_API_KEY`) | extra `observability` |
| `otlp` | qualquer coletor (`OTEL_EXPORTER_OTLP_ENDPOINT`) | extra `observability` |
| `cloudwatch` | métricas EMF em stdout | agente CloudWatch |

Aceita vários separados por vírgulas. `jsonl` é o destino de referência: mostra exactamente o que seria enviado, sem rede — útil para inspecionar antes de ligar um backend e para correr com telemetria em CI.

**CloudWatch sem boto3.** O adaptador emite Embedded Metric Format em stdout, que o agente CloudWatch converte em métricas ao ler o log. Não precisa de credenciais AWS nem de chamadas de rede a partir do processo de avaliação — as credenciais ficam na infraestrutura, que é onde devem estar.

## Invariantes

Estes três não são detalhe de implementação; são a razão de a telemetria poder ser ligada em produção sem receio.

**1. Nunca altera resultados.** `predictions.jsonl` e `summary.json` são idênticos com e sem exportador. As chamadas do item vão para o evento, nunca para `meta` — pôr dataclasses em `meta` partiria a serialização e contaminaria os artefactos. Verificado em `test_artefactos_identicos_com_e_sem_telemetria`.

**2. Nunca derruba a corrida.** Destino indisponível, backend em baixo ou extra em falta produzem um aviso em `stderr` e a corrida continua. Mesma política do METEOR em `lexical_metrics`: falhar uma avaliação por causa do instrumento de medição é trocar o objectivo pelo instrumento. Avisos são emitidos uma vez por destino, não uma vez por item.

**3. Não exporta conteúdo por omissão.** Perguntas, respostas e contexto ficam de fora salvo `LLM_EVAL_TELEMETRY_CONTENT=1`. Um endpoint de observabilidade é mais um sítio onde os dados do corpus passam a existir, muitas vezes fora do controlo de quem corre a avaliação. Métricas e vereditos são exportados sempre; o texto é uma decisão consciente.

## Fora de âmbito

- Substituir `summary.json`: a telemetria é séries temporais, não o artefacto auditável da corrida.
- Amostragem ou *sampling* de traces — o volume de uma corrida de avaliação não o justifica.
- Enviar métricas directamente à API do CloudWatch (`PutMetricData`): exigiria credenciais no processo de avaliação.

## Critérios de aceitação

- [x] Sem `LLM_EVAL_TELEMETRY`, o custo é zero (`NullExporter`).
- [x] Artefactos byte-idênticos com e sem telemetria, com teste dedicado.
- [x] Exportador que levanta excepção em todos os métodos não afecta a corrida.
- [x] Um destino mal configurado é saltado; os restantes continuam a funcionar.
- [x] Conteúdo ausente dos atributos por omissão, com teste que procura o texto.
- [x] EMF válido e sem divisão por zero em corrida vazia.
- [x] `mypy --strict` limpo sem o extra `observability` instalado.

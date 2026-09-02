"""Contrato de telemetria: eventos da corrida e o protocolo dos exportadores.

Porquê um contrato e não três integrações. LangSmith, Phoenix e CloudWatch pedem
formatos diferentes, mas a informação é a mesma: uma corrida, itens dentro dela e
chamadas LLM dentro de cada item. Modelamos isso uma vez e traduzimos no fim —
acrescentar um destino passa a ser um adaptador, não uma alteração ao pipeline.

Três invariantes que a telemetria **não** pode violar:

1. **Nunca altera resultados.** `predictions.jsonl` e `summary.json` são idênticos
   com ou sem exportador. A telemetria é um canal lateral.
2. **Nunca derruba a corrida.** Um destino em baixo é um aviso, não uma falha —
   mesma política já usada para o METEOR em `lexical_metrics`.
3. **Não exporta conteúdo por omissão.** Perguntas, respostas e contexto ficam de
   fora salvo pedido explícito: um endpoint de observabilidade é mais um sítio
   onde dados do corpus passam a existir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

TELEMETRY_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class LlmCallEvent:
    """Uma chamada ao LLM, já concluída."""

    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    started_at: float
    endpoint: str = ""


@dataclass(frozen=True)
class ItemEvent:
    """Um item avaliado, com as chamadas que fez e o resultado da verificação."""

    item_id: str
    started_at: float
    duration_ms: float
    anomaly_flag: bool
    judge_verdict: str | None
    judge_confidence: float | None
    judge_fallback: bool
    embedding_max_cosine: float | None
    retrieval_top_score: float | None
    calls: tuple[LlmCallEvent, ...] = ()
    error: str | None = None
    #: Só preenchido com ``include_content``; ver invariante 3.
    content: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunEvent:
    """Metadados e agregados da corrida inteira."""

    run_id: str
    started_at: float
    duration_ms: float
    n_items: int
    n_anomalies: int
    config_name: str
    protocol: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)


class TelemetryExporter(Protocol):
    """Destino de telemetria. Todos os métodos têm de ser tolerantes a falhas."""

    def on_item(self, event: ItemEvent) -> None: ...

    def on_run(self, event: RunEvent) -> None: ...

    def close(self) -> None: ...


class NullExporter:
    """Exportador por omissão: não faz nada, sem custo."""

    def on_item(self, event: ItemEvent) -> None:
        return None

    def on_run(self, event: RunEvent) -> None:
        return None

    def close(self) -> None:
        return None


def item_attributes(event: ItemEvent) -> dict[str, Any]:
    """Atributos planos de um item, comuns a todos os destinos.

    Nomes alinhados com as convenções OpenInference/OTel onde existem (``llm.*``,
    ``gen_ai.*``), para que Phoenix e LangSmith os reconheçam sem mapeamento.
    """
    prompt = sum(c.prompt_tokens for c in event.calls)
    completion = sum(c.completion_tokens for c in event.calls)
    attrs: dict[str, Any] = {
        "eval.item_id": event.item_id,
        "eval.anomaly_flag": event.anomaly_flag,
        "eval.duration_ms": round(event.duration_ms, 2),
        "llm.token_count.prompt": prompt,
        "llm.token_count.completion": completion,
        "llm.token_count.total": prompt + completion,
        "llm.call_count": len(event.calls),
    }
    if event.judge_verdict is not None:
        attrs["eval.judge.verdict"] = event.judge_verdict
        attrs["eval.judge.fallback"] = event.judge_fallback
    if event.judge_confidence is not None:
        attrs["eval.judge.confidence"] = event.judge_confidence
    if event.embedding_max_cosine is not None:
        attrs["eval.embedding.max_cosine"] = event.embedding_max_cosine
    if event.retrieval_top_score is not None:
        attrs["eval.retrieval.top_score"] = event.retrieval_top_score
    if event.error:
        attrs["eval.error"] = event.error
    for key, value in event.content.items():
        attrs[f"eval.content.{key}"] = value
    return attrs


def call_attributes(call: LlmCallEvent) -> dict[str, Any]:
    """Atributos de uma chamada LLM."""
    return {
        "gen_ai.operation.name": call.role,
        "gen_ai.request.model": call.model,
        "gen_ai.usage.input_tokens": call.prompt_tokens,
        "gen_ai.usage.output_tokens": call.completion_tokens,
        "llm.token_count.total": call.total_tokens,
        "llm.latency_ms": round(call.latency_ms, 2),
        "server.address": call.endpoint,
    }

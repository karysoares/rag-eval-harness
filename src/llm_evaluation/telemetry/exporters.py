"""Adaptadores de telemetria: JSONL local, OTLP (Phoenix/LangSmith) e CloudWatch EMF.

Cada um traduz os eventos de :mod:`telemetry.base` para o formato do destino. Todos
engolem as suas próprias falhas: um destino indisponível degrada para um aviso em
``stderr``, nunca interrompe a avaliação.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from llm_evaluation.telemetry.base import (
    TELEMETRY_SCHEMA_VERSION,
    ItemEvent,
    RunEvent,
    call_attributes,
    item_attributes,
)


def _warn_once(state: dict[str, bool], key: str, message: str) -> None:
    """Avisa uma vez por destino: 1000 itens não devem gerar 1000 linhas iguais."""
    if state.get(key):
        return
    state[key] = True
    print(f"[telemetria] {message}", file=sys.stderr, flush=True)


class JsonlExporter:
    """Escreve os eventos num ficheiro JSONL local. Sem dependências.

    É o destino de referência: serve para inspecionar exactamente o que seria
    enviado antes de ligar um backend, e para correr com telemetria em ambientes
    sem rede (CI, avaliação offline).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._warned: dict[str, bool] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._fh.write(
                    json.dumps(
                        {"schema_version": TELEMETRY_SCHEMA_VERSION, "tipo": kind, **payload},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._fh.flush()
        except (OSError, ValueError) as exc:
            _warn_once(self._warned, "write", f"JSONL indisponível ({exc}); a ignorar.")

    def on_item(self, event: ItemEvent) -> None:
        self._write(
            "item",
            {
                "atributos": item_attributes(event),
                "chamadas": [call_attributes(c) for c in event.calls],
                "started_at": event.started_at,
            },
        )

    def on_run(self, event: RunEvent) -> None:
        self._write(
            "run",
            {
                "run_id": event.run_id,
                "n_itens": event.n_items,
                "n_anomalias": event.n_anomalies,
                "config": event.config_name,
                "duracao_ms": round(event.duration_ms, 2),
                "totais": event.totals,
                "protocolo": event.protocol,
            },
        )

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class OtlpExporter:
    """Traces OpenTelemetry para qualquer coletor OTLP.

    Um só adaptador serve Phoenix, LangSmith e CloudWatch, porque os três falam
    OTLP — Phoenix nativamente, LangSmith pelo seu endpoint OTLP, e CloudWatch
    através do coletor ADOT. Escrever três clientes seria triplicar o mesmo mapa
    de atributos.

    A hierarquia é ``run`` → ``item`` → chamada LLM. Os spans são construídos com
    ``start_time``/``end_time`` explícitos a partir dos eventos já registados, e
    não com o contexto implícito do OTel: a corrida é multi-thread e a propagação
    implícita atribuiria spans à thread errada.

    Requer o extra ``observability``; sem ele o construtor falha com instruções.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
        service_name: str = "rag-eval-harness",
        project: str = "",
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:  # pragma: no cover - depende do extra
            msg = "Telemetria OTLP requer o extra 'observability': uv sync --extra observability"
            raise RuntimeError(msg) from exc

        atributos = {"service.name": service_name}
        if project:
            atributos["openinference.project.name"] = project
        provider = TracerProvider(resource=Resource.create(atributos))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers or {})),
        )
        self._provider = provider
        self._tracer = trace.get_tracer(__name__, tracer_provider=provider)
        self._warned: dict[str, bool] = {}

    @staticmethod
    def _ns(seconds: float) -> int:
        return int(seconds * 1_000_000_000)

    def on_item(self, event: ItemEvent) -> None:
        try:
            fim = event.started_at + event.duration_ms / 1000.0
            span = self._tracer.start_span(
                f"eval.item {event.item_id}",
                start_time=self._ns(event.started_at),
                attributes=item_attributes(event),
            )
            contexto = None
            try:
                from opentelemetry import trace as _t

                contexto = _t.set_span_in_context(span)
            except ImportError:  # pragma: no cover
                pass
            for chamada in event.calls:
                filho = self._tracer.start_span(
                    f"llm.{chamada.role}",
                    context=contexto,
                    start_time=self._ns(chamada.started_at),
                    attributes=call_attributes(chamada),
                )
                filho.end(end_time=self._ns(chamada.started_at + chamada.latency_ms / 1000.0))
            span.end(end_time=self._ns(fim))
        except Exception as exc:  # noqa: BLE001 - telemetria nunca derruba a corrida
            _warn_once(self._warned, "item", f"OTLP indisponível ({exc}); a continuar sem traces.")

    def on_run(self, event: RunEvent) -> None:
        try:
            span = self._tracer.start_span(
                f"eval.run {event.run_id}",
                start_time=self._ns(event.started_at),
                attributes={
                    "eval.run_id": event.run_id,
                    "eval.n_items": event.n_items,
                    "eval.n_anomalies": event.n_anomalies,
                    "eval.config": event.config_name,
                    **{f"eval.total.{k}": v for k, v in event.totals.items()},
                },
            )
            span.end(end_time=self._ns(event.started_at + event.duration_ms / 1000.0))
        except Exception as exc:  # noqa: BLE001
            _warn_once(self._warned, "run", f"OTLP indisponível ({exc}); a continuar sem traces.")

    def close(self) -> None:
        try:
            self._provider.shutdown()
        except Exception as exc:  # noqa: BLE001
            _warn_once(self._warned, "close", f"falha ao fechar OTLP ({exc}).")


class CloudWatchEmfExporter:
    """Métricas CloudWatch em Embedded Metric Format (EMF). Sem dependências AWS.

    EMF é JSON estruturado que o agente CloudWatch converte em métricas ao ler o
    log — não precisa de credenciais, de boto3 nem de chamadas de rede a partir
    daqui. Isso mantém a avaliação sem dependência da AWS e deixa o envio a cargo
    da infraestrutura, que é onde as credenciais devem viver.

    Emite métricas (tokens, latência, taxa de anomalia), não traces; para traces
    em CloudWatch use ``OtlpExporter`` apontado ao coletor ADOT.
    """

    def __init__(self, *, namespace: str = "RagEvalHarness", stream: Any = None) -> None:
        self._namespace = namespace
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self._warned: dict[str, bool] = {}

    def _emit(
        self, metricas: list[dict[str, str]], dimensoes: dict[str, str], valores: dict[str, Any]
    ) -> None:
        documento = {
            "_aws": {
                "Timestamp": int(__import__("time").time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self._namespace,
                        "Dimensions": [list(dimensoes)],
                        "Metrics": metricas,
                    }
                ],
            },
            **dimensoes,
            **valores,
        }
        try:
            with self._lock:
                self._stream.write(json.dumps(documento, ensure_ascii=False) + "\n")
                self._stream.flush()
        except (OSError, ValueError) as exc:
            _warn_once(self._warned, "emit", f"EMF indisponível ({exc}); a ignorar.")

    def on_item(self, event: ItemEvent) -> None:
        prompt = sum(c.prompt_tokens for c in event.calls)
        completion = sum(c.completion_tokens for c in event.calls)
        self._emit(
            [
                {"Name": "ItemDurationMs", "Unit": "Milliseconds"},
                {"Name": "PromptTokens", "Unit": "Count"},
                {"Name": "CompletionTokens", "Unit": "Count"},
                {"Name": "AnomalyFlag", "Unit": "Count"},
            ],
            {"Stage": "item"},
            {
                "ItemDurationMs": round(event.duration_ms, 2),
                "PromptTokens": prompt,
                "CompletionTokens": completion,
                "AnomalyFlag": 1 if event.anomaly_flag else 0,
                "ItemId": event.item_id,
            },
        )

    def on_run(self, event: RunEvent) -> None:
        self._emit(
            [
                {"Name": "RunDurationMs", "Unit": "Milliseconds"},
                {"Name": "Items", "Unit": "Count"},
                {"Name": "Anomalies", "Unit": "Count"},
                {"Name": "AnomalyRate", "Unit": "Percent"},
            ],
            {"Stage": "run", "Config": event.config_name},
            {
                "RunDurationMs": round(event.duration_ms, 2),
                "Items": event.n_items,
                "Anomalies": event.n_anomalies,
                "AnomalyRate": (
                    round(100.0 * event.n_anomalies / event.n_items, 2) if event.n_items else 0.0
                ),
                "RunId": event.run_id,
            },
        )

    def close(self) -> None:
        return None


class MultiExporter:
    """Encaminha para vários destinos; a falha de um não afecta os outros."""

    def __init__(self, exporters: list[Any]) -> None:
        self._exporters = exporters
        self._warned: dict[str, bool] = {}

    def _fan_out(self, metodo: str, *args: Any) -> None:
        for exp in self._exporters:
            try:
                getattr(exp, metodo)(*args)
            except Exception as exc:  # noqa: BLE001
                nome = type(exp).__name__
                _warn_once(self._warned, f"{nome}.{metodo}", f"{nome} falhou ({exc}); a ignorar.")

    def on_item(self, event: ItemEvent) -> None:
        self._fan_out("on_item", event)

    def on_run(self, event: RunEvent) -> None:
        self._fan_out("on_run", event)

    def close(self) -> None:
        self._fan_out("close")


def _env(nome: str, default: str = "") -> str:
    return os.environ.get(nome, default).strip()

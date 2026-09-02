"""Telemetria de corridas: eventos, exportadores e resolução por ambiente.

Destinos suportados (`LLM_EVAL_TELEMETRY`, lista separada por vírgulas):

| Valor | Destino | Requisitos |
|-------|---------|------------|
| `jsonl` | ficheiro local `telemetry.jsonl` na corrida | nenhum |
| `phoenix` | Arize Phoenix via OTLP | extra `observability` |
| `langsmith` | LangSmith via endpoint OTLP | extra `observability` + `LANGSMITH_API_KEY` |
| `otlp` | qualquer coletor OTLP (inclui ADOT → CloudWatch) | extra `observability` |
| `cloudwatch` | métricas CloudWatch em EMF | nenhum (agente CloudWatch) |

`SPEC-003` Fase 8. Ver `docs/specs/011-telemetry.md`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from llm_evaluation.telemetry.base import (
    TELEMETRY_SCHEMA_VERSION,
    ItemEvent,
    LlmCallEvent,
    NullExporter,
    RunEvent,
    TelemetryExporter,
)
from llm_evaluation.telemetry.exporters import (
    CloudWatchEmfExporter,
    JsonlExporter,
    MultiExporter,
    OtlpExporter,
)

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "CloudWatchEmfExporter",
    "ItemEvent",
    "JsonlExporter",
    "LlmCallEvent",
    "MultiExporter",
    "NullExporter",
    "OtlpExporter",
    "RunEvent",
    "TelemetryExporter",
    "build_exporter",
    "telemetry_targets_from_env",
]

#: Endpoints por omissão. Phoenix local e o endpoint OTLP público do LangSmith.
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"


def telemetry_targets_from_env() -> list[str]:
    """Destinos pedidos em ``LLM_EVAL_TELEMETRY`` (vazio = nenhum)."""
    bruto = os.environ.get("LLM_EVAL_TELEMETRY", "").strip()
    if not bruto:
        return []
    return [t.strip().lower() for t in bruto.split(",") if t.strip()]


def _otlp(endpoint_env: str, default: str, headers: dict[str, str], project: str) -> Any:
    endpoint = os.environ.get(endpoint_env, "").strip() or default
    return OtlpExporter(endpoint=endpoint, headers=headers, project=project)


def build_exporter(
    *,
    run_dir: Path | None = None,
    targets: list[str] | None = None,
) -> TelemetryExporter:
    """Constrói o exportador dos destinos pedidos; `NullExporter` se nenhum.

    Um destino mal configurado é reportado e **saltado** — a corrida continua sem
    ele. Falhar a avaliação por causa de telemetria seria trocar o objectivo pelo
    instrumento.
    """
    pedidos = targets if targets is not None else telemetry_targets_from_env()
    if not pedidos:
        return NullExporter()

    projeto = os.environ.get("LLM_EVAL_TELEMETRY_PROJECT", "rag-eval-harness").strip()
    construidos: list[Any] = []
    for alvo in pedidos:
        try:
            if alvo == "jsonl":
                destino = run_dir / "telemetry.jsonl" if run_dir else Path("telemetry.jsonl")
                construidos.append(JsonlExporter(destino))
            elif alvo == "phoenix":
                construidos.append(
                    _otlp("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_PHOENIX_ENDPOINT, {}, projeto),
                )
            elif alvo == "langsmith":
                chave = os.environ.get("LANGSMITH_API_KEY", "").strip()
                if not chave:
                    msg = "LANGSMITH_API_KEY não definida"
                    raise RuntimeError(msg)
                construidos.append(
                    _otlp(
                        "LANGSMITH_ENDPOINT",
                        DEFAULT_LANGSMITH_ENDPOINT,
                        {"x-api-key": chave, "Langsmith-Project": projeto},
                        projeto,
                    ),
                )
            elif alvo == "otlp":
                endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
                if not endpoint:
                    msg = "OTEL_EXPORTER_OTLP_ENDPOINT não definida"
                    raise RuntimeError(msg)
                construidos.append(OtlpExporter(endpoint=endpoint, project=projeto))
            elif alvo == "cloudwatch":
                espaco = os.environ.get("LLM_EVAL_CLOUDWATCH_NAMESPACE", "RagEvalHarness").strip()
                construidos.append(CloudWatchEmfExporter(namespace=espaco))
            else:
                msg = f"destino desconhecido {alvo!r}"
                raise RuntimeError(msg)
        except Exception as exc:  # noqa: BLE001 - telemetria nunca bloqueia a corrida
            print(
                f"[telemetria] destino {alvo!r} indisponível ({exc}); a corrida continua sem ele.",
                file=sys.stderr,
                flush=True,
            )

    if not construidos:
        return NullExporter()
    if len(construidos) == 1:
        return construidos[0]  # type: ignore[no-any-return]
    return MultiExporter(construidos)

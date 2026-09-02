"""Tradução de `RunRecord` / sumário para eventos de telemetria.

Vive à parte dos exportadores para manter o pipeline ignorante do formato de
destino: ele produz registos, isto converte-os, os adaptadores enviam.
"""

from __future__ import annotations

import os
import time
from typing import Any

from llm_evaluation.observability import LlmCallUsage
from llm_evaluation.telemetry.base import (
    ItemEvent,
    LlmCallEvent,
    RunEvent,
    TelemetryExporter,
)
from llm_evaluation.types import RunRecord

#: Máximo de caracteres por campo quando o conteúdo é exportado.
CONTENT_MAX_CHARS = 2000


def telemetry_includes_content() -> bool:
    """``LLM_EVAL_TELEMETRY_CONTENT=1`` autoriza exportar pergunta/resposta.

    Desligado por omissão: um backend de observabilidade é mais um sítio onde o
    conteúdo do corpus passa a existir, muitas vezes fora do controlo de quem
    corre a avaliação. Quem precisa de inspecionar prompts liga isto de forma
    consciente.
    """
    return os.environ.get("LLM_EVAL_TELEMETRY_CONTENT", "").strip() in ("1", "true", "yes")


def _call_event(call: LlmCallUsage) -> LlmCallEvent:
    return LlmCallEvent(
        role=call.role,
        model=call.model,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        total_tokens=call.total_tokens,
        latency_ms=call.latency_ms,
        started_at=call.started_at,
        endpoint=call.endpoint,
    )


def build_item_event(
    record: RunRecord,
    *,
    calls: list[LlmCallUsage],
    started_at: float,
    include_content: bool = False,
) -> ItemEvent:
    """Converte um registo concluído no evento correspondente."""
    judge = record.signals.judge
    metricas = record.meta.get("metricas_recuperacao")
    top_score = None
    if isinstance(metricas, dict):
        bruto = metricas.get("score_melhor_chunk")
        if isinstance(bruto, int | float):
            top_score = float(bruto)
    erro = record.meta.get("processing_error")
    conteudo: dict[str, str] = {}
    if include_content:
        conteudo = {
            "question": record.question[:CONTENT_MAX_CHARS],
            "answer": record.answer[:CONTENT_MAX_CHARS],
        }
    return ItemEvent(
        item_id=record.item_id,
        started_at=started_at,
        duration_ms=max(0.0, (time.time() - started_at) * 1000.0),
        anomaly_flag=bool(record.anomaly_flag),
        judge_verdict=judge.veredito if judge is not None else None,
        judge_confidence=judge.confianca if judge is not None else None,
        judge_fallback=bool(judge.raw.get("fallback_heuristico")) if judge is not None else False,
        embedding_max_cosine=record.signals.embedding_max_cosine,
        retrieval_top_score=top_score,
        calls=tuple(_call_event(c) for c in calls),
        error=str(erro.get("type")) if isinstance(erro, dict) else None,
        content=conteudo,
    )


def emit_item_event(
    exporter: TelemetryExporter,
    record: RunRecord,
    *,
    calls: list[LlmCallUsage],
    started_at: float,
    include_content: bool = False,
) -> None:
    """Emite o evento do item; falhas de telemetria nunca sobem ao pipeline."""
    try:
        exporter.on_item(
            build_item_event(
                record,
                calls=calls,
                started_at=started_at,
                include_content=include_content,
            ),
        )
    except Exception:  # noqa: BLE001 - invariante: telemetria não derruba a corrida
        return


def emit_run_event(
    exporter: TelemetryExporter,
    *,
    run_id: str,
    started_at: float,
    records: list[RunRecord],
    config_name: str,
    protocol: dict[str, Any] | None = None,
    totals: dict[str, Any] | None = None,
) -> None:
    """Emite o evento agregado da corrida."""
    try:
        exporter.on_run(
            RunEvent(
                run_id=run_id,
                started_at=started_at,
                duration_ms=max(0.0, (time.time() - started_at) * 1000.0),
                n_items=len(records),
                n_anomalies=sum(1 for r in records if r.anomaly_flag),
                config_name=config_name,
                protocol=protocol or {},
                totals=totals or {},
            ),
        )
    except Exception:  # noqa: BLE001
        return

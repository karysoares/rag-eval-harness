"""API única do dashboard — sem lógica de métricas no Streamlit."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from llm_evaluation.dashboard.data import (
    cache_run_artifacts,
    load_fila_revisao_dataframe,
    load_summary_json,
)
from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.hitl_io import (
    hitl_csv_path as _hitl_csv_path,
)
from llm_evaluation.hitl_io import (
    load_hitl_labels as _load_hitl_labels,
)
from llm_evaluation.hitl_io import (
    save_hitl_annotation as _save_hitl_annotation,
)
from llm_evaluation.resilience import retry_call
from llm_evaluation.run_reprocess import reprocess_run_dir


class MetricMode(StrEnum):
    AUTOMATICO = "automatico"
    POS_HITL = "pos_hitl"
    COMPARAR = "comparar"


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    return cache_run_artifacts(run_dir)


def load_report(
    run_dir: Path,
    *,
    metric_mode: MetricMode = MetricMode.AUTOMATICO,
) -> dict[str, Any]:
    """Prefer summary gravado; fallback reprocess se só JSONL."""
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        s = load_summary_json(run_dir)
        return s if s is not None else {}
    return reprocess_run_dir(run_dir)


def apply_hitl_labels(
    run_dir: Path,
    csv_path: Path | None = None,
    *,
    strict_hitl_ids: bool = True,
) -> dict[str, object]:
    csv = csv_path or hitl_csv_path(run_dir)
    return retry_call(
        lambda: reprocess_run_dir(
            run_dir,
            hitl_csv=csv,
            retry_attempts=3,
            strict_hitl_ids=strict_hitl_ids,
        ),
        attempts=3,
        retry_on=(OSError, TimeoutError, json.JSONDecodeError),
    )


def save_hitl_annotation(
    run_dir: Path,
    *,
    item_id: str,
    rotulo: str,
    revisor: str = "",
    notas: str = "",
) -> Path:
    return retry_call(
        lambda: _save_hitl_annotation(
            run_dir,
            item_id=item_id,
            rotulo=rotulo,
            revisor=revisor,
            notas=notas,
        ),
        attempts=3,
        retry_on=(OSError, TimeoutError),
    )


def load_hitl_labels(run_dir: Path) -> dict[str, dict[str, str]]:
    return retry_call(
        lambda: _load_hitl_labels(run_dir),
        attempts=3,
        retry_on=(OSError, TimeoutError, json.JSONDecodeError),
    )


def hitl_csv_path(run_dir: Path) -> Path:
    return _hitl_csv_path(run_dir)


def kpi_blocks_for_mode(report: dict[str, Any], mode: MetricMode) -> dict[str, Any]:
    if mode == MetricMode.POS_HITL:
        hitl = report.get("sumario_hitl")
        if isinstance(hitl, dict):
            return {"fonte": "sumario_hitl", "dados": hitl}
    if mode == MetricMode.COMPARAR:
        return {
            "fonte": "comparar",
            "lexical": report.get("sumario_lexical"),
            "operacional": report.get("sumario_operacional"),
            "hitl": report.get("sumario_hitl"),
        }
    lex = report.get("sumario_lexical")
    op = report.get("sumario_operacional")
    return {
        "fonte": "automatico",
        "lexical": lex,
        "operacional": op,
        "kpi_primario": report.get("kpi_primario"),
    }


def provenance_from_report(report: dict[str, Any]) -> dict[str, Any]:
    prov = report.get("proveniencia")
    if isinstance(prov, dict):
        return prov
    meta = report.get("metadados_corrida")
    if isinstance(meta, dict):
        return {
            "config_hash_sha256": meta.get("config_hash_sha256"),
            "modelos": meta.get("modelos"),
            "git_commit": meta.get("git_commit"),
            "config_path": meta.get("config_path"),
        }
    return {}


def explicacao_for_item(run_dir: Path, item_id: str) -> dict[str, Any] | None:
    for rec in load_records_from_predictions_jsonl(run_dir / "predictions.jsonl"):
        if rec.item_id == item_id:
            exp = rec.meta.get("explicacao")
            return cast(dict[str, Any], exp) if isinstance(exp, dict) else None
    return None


def hitl_progress(run_dir: Path) -> dict[str, int]:
    fila_df = load_fila_revisao_dataframe(run_dir)
    n_fila = len(fila_df) if fila_df is not None else 0
    labels = load_hitl_labels(run_dir)
    n_rot = len(labels)
    if n_fila == 0:
        summary = load_summary_json(run_dir)
        hitl = summary.get("sumario_hitl") if isinstance(summary, dict) else None
        if isinstance(hitl, dict):
            n_rot = max(n_rot, int(hitl.get("n_itens_rotulados") or 0))
    return {"rotulados": n_rot, "fila_total": n_fila}


__all__ = [
    "MetricMode",
    "apply_hitl_labels",
    "explicacao_for_item",
    "hitl_csv_path",
    "hitl_progress",
    "kpi_blocks_for_mode",
    "load_hitl_labels",
    "load_report",
    "load_run_bundle",
    "provenance_from_report",
    "save_hitl_annotation",
]

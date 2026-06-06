"""Validação de schemas de corrida para o dashboard (SPEC-006, Fase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_evaluation.dashboard.data import (
    load_manifest_json,
    load_summary_json,
    predictions_path,
)
from llm_evaluation.schema_registry import (
    MANIFEST_SCHEMA_VERSION,
    PREDICTIONS_SCHEMA_VERSIONS_OK,
    SUMMARY_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSIONS_OK,
    validate_manifest,
    validate_prediction_record,
    validate_summary,
)

KNOWN_METRICS_REPORT_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "run_dir",
        "n_itens",
        "analise_camadas",
        "confusao_vs_referencia",
        "confusao_vs_gold",
        "sumario_lexical",
        "sumario_recuperacao",
        "sumario_padroes",
        "sumario_juiz",
        "sumario_gap_rag_resposta",
        "kpi_primario",
        "protocolo_ativo",
    },
)

METRICS_REPORT_SCHEMA_VERSION = "1.0"


def validate_metrics_report(obj: dict[str, Any], *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    ver = obj.get("schema_version")
    if ver is not None and ver != METRICS_REPORT_SCHEMA_VERSION:
        issues.append(f"schema_version inesperado em metrics_report: {ver!r}")
    elif ver is None:
        issues.append("aviso: metrics_report sem schema_version (legado ou --analyze-run antigo)")
    unknown = set(obj.keys()) - KNOWN_METRICS_REPORT_TOP_FIELDS
    if unknown:
        msg = f"campos desconhecidos em metrics_report: {sorted(unknown)}"
        if strict:
            issues.append(msg)
        else:
            issues.append(f"aviso: {msg}")
    return issues


def _sample_prediction_lines(run_dir: Path, *, max_lines: int = 5) -> list[dict[str, Any]]:
    path = predictions_path(run_dir)
    if path is None:
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
            if len(out) >= max_lines:
                break
    return out


def _prediction_jsonl_errors(run_dir: Path, *, max_lines: int = 5) -> list[str]:
    path = predictions_path(run_dir)
    if path is None:
        return []
    issues: list[str] = []
    checked = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            checked += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"predictions linha {i}: JSON inválido: {e}")
            if checked >= max_lines:
                break
    return issues


def detect_legacy_run(
    summary: dict[str, Any] | None,
    prediction_samples: list[dict[str, Any]],
    *,
    has_manifest: bool,
) -> bool:
    """Corrida anterior a schema v1.0 ou sem manifest."""
    if not has_manifest:
        return True
    if summary is not None and summary.get("schema_version") is None:
        return True
    return bool(
        prediction_samples and all(s.get("schema_version") is None for s in prediction_samples)
    )


def detect_schema_mismatch(
    summary: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    prediction_samples: list[dict[str, Any]],
) -> bool:
    if (
        summary
        and summary.get("schema_version")
        and str(summary["schema_version"]) not in SUMMARY_SCHEMA_VERSIONS_OK
    ):
        return True
    if (
        manifest
        and manifest.get("schema_version")
        and str(manifest["schema_version"]) != MANIFEST_SCHEMA_VERSION
    ):
        return True
    for s in prediction_samples:
        if (
            s.get("schema_version")
            and str(s["schema_version"]) not in PREDICTIONS_SCHEMA_VERSIONS_OK
        ):
            return True
    return False


def validate_run_schemas(
    run_dir: Path,
    *,
    strict: bool = False,
    sample_lines: int = 5,
) -> dict[str, Any]:
    """Valida predictions, summary, manifest e metrics_report; devolve avisos estruturados."""
    warnings: list[str] = []
    summary = load_summary_json(run_dir)
    manifest = load_manifest_json(run_dir)

    pred_samples = _sample_prediction_lines(run_dir, max_lines=sample_lines)
    if not pred_samples and predictions_path(run_dir) is None:
        warnings.append("sem predictions.jsonl")
    warnings.extend(_prediction_jsonl_errors(run_dir, max_lines=sample_lines))

    for i, obj in enumerate(pred_samples, start=1):
        for msg in validate_prediction_record(obj, strict=strict):
            warnings.append(f"predictions linha {i}: {msg}")

    if summary is not None:
        for msg in validate_summary(summary, strict=strict):
            warnings.append(f"summary: {msg}")
    else:
        warnings.append("aviso: summary.json em falta")

    if manifest is not None:
        for msg in validate_manifest(manifest, strict=strict):
            warnings.append(f"manifest: {msg}")
    else:
        warnings.append("aviso: manifest.json em falta (corrida legada)")

    metrics_path = run_dir / "metrics_report.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            warnings.append(f"metrics_report.json inválido: {e}")
        else:
            if isinstance(metrics, dict):
                for msg in validate_metrics_report(metrics, strict=strict):
                    warnings.append(f"metrics_report: {msg}")

    legacy = detect_legacy_run(
        summary,
        pred_samples,
        has_manifest=manifest is not None,
    )
    mismatch = detect_schema_mismatch(summary, manifest, pred_samples)

    return {
        "warnings": warnings,
        "n_warnings_schema": len(warnings),
        "legacy_run": legacy,
        "schema_mismatch": mismatch,
        "predictions_schema_version": (
            pred_samples[0].get("schema_version") if pred_samples else None
        ),
        "summary_schema_version": (summary or {}).get("schema_version"),
        "manifest_schema_version": (manifest or {}).get("schema_version"),
        "n_prediction_samples_checked": len(pred_samples),
    }

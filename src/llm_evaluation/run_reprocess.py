"""Reprocessamento offline único: JSONL → summary + fila + manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from llm_evaluation.config import AppConfig
from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.fila_revisao import export_fila_csv
from llm_evaluation.hitl_io import write_hitl_manifest
from llm_evaluation.protocol import build_protocolo_ativo
from llm_evaluation.reporting import summarize, write_summary
from llm_evaluation.resilience import file_lock, retry_call
from llm_evaluation.run_artifacts import (
    build_manifest,
    collect_run_metadata,
    write_manifest,
)


def provenance_block(metadados: dict[str, Any]) -> dict[str, object]:
    """Campos de proveniência no topo do summary."""
    modelos = metadados.get("modelos")
    return {
        "config_hash_sha256": metadados.get("config_hash_sha256"),
        "modelos": modelos if isinstance(modelos, dict) else {},
        "versao_pacote": _package_version(),
        "git_commit": metadados.get("git_commit"),
        "config_path": metadados.get("config_path"),
        "determinismo": metadados.get("determinismo"),
    }


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("rag-eval-harness")
    except Exception:
        return "unknown"


def _protocol_from_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.is_file():
        return None
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    pa = raw.get("protocolo_ativo")
    return cast(dict[str, object], pa) if isinstance(pa, dict) else None


def _infer_reference_type_from_records(records: list[Any]) -> tuple[str, str] | None:
    n_lex = 0
    for r in records:
        lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics")
        if not isinstance(lm, dict):
            continue
        if lm.get("note") in {"metricas_lexicas_desligadas", "sem_referencia"}:
            continue
        if lm.get("f1_token") is not None or lm.get("em_squad") is not None:
            n_lex += 1
    if n_lex:
        return (
            "lexical",
            "tipo_referencia_ativo inferido como lexical a partir de "
            f"metricas_lexicas em {n_lex} itens",
        )
    return None


def reprocess_run_dir(
    run_dir: Path,
    *,
    cfg: AppConfig | None = None,
    config_path: Path | None = None,
    hitl_csv: Path | None = None,
    reference_type: str | None = None,
    retry_attempts: int = 3,
    strict_hitl_ids: bool = False,
) -> dict[str, object]:
    """Carrega predictions, opcionalmente aplica HITL, re-summarize e actualiza artefactos."""
    run_dir = run_dir.resolve()
    pred = run_dir / "predictions.jsonl"
    lock = run_dir / ".reprocess.lock"

    def _reprocess_once() -> dict[str, object]:
        if not pred.is_file():
            msg = f"Sem predictions.jsonl em {run_dir}"
            raise FileNotFoundError(msg)

        with file_lock(lock, timeout_seconds=45.0, stale_after_seconds=3600.0):
            if hitl_csv is not None:
                from llm_evaluation.hitl_io import merge_hitl_csv_into_predictions

                merge_hitl_csv_into_predictions(
                    hitl_csv,
                    pred,
                    strict_ids=strict_hitl_ids,
                )

            records = load_records_from_predictions_jsonl(pred)
            summary_path = run_dir / "summary.json"
            protocol: dict[str, object] | None = None
            ref_type: str | None = reference_type or (
                cfg.dataset.reference_type if cfg is not None else None
            )
            warnings_reprocess: list[str] = []

            if cfg is not None:
                protocol = build_protocolo_ativo(cfg)
            elif summary_path.is_file():
                raw = json.loads(summary_path.read_text(encoding="utf-8"))
                protocol = _protocol_from_summary(summary_path)
                rt = raw.get("tipo_referencia_ativo")
                if rt is not None:
                    ref_type = str(rt)
                elif ref_type is None:
                    ref_type = "answer_lists"
            elif ref_type is None:
                inferred = _infer_reference_type_from_records(records)
                if inferred is None:
                    msg = (
                        "Sem summary/config/reference_type para reprocessar. "
                        "Informe reference_type explicitamente ou preserve summary.json."
                    )
                    raise ValueError(msg)
                ref_type, warning = inferred
                warnings_reprocess.append(warning)

            if ref_type is None:
                msg = "reference_type em falta para reprocessamento"
                raise ValueError(msg)

            summary = summarize(records, reference_type=ref_type, protocol=protocol)
            if warnings_reprocess:
                summary["avisos_reprocessamento"] = warnings_reprocess

            metadados: dict[str, Any]
            if summary_path.is_file():
                prev = json.loads(summary_path.read_text(encoding="utf-8"))
                mc = prev.get("metadados_corrida")
                metadados = mc if isinstance(mc, dict) else {}
            else:
                metadados = {}

            if cfg is not None and config_path is not None:
                metadados = collect_run_metadata(
                    cfg,
                    config_path=config_path,
                    run_dir=run_dir,
                    n_records=len(records),
                )

            summary["metadados_corrida"] = metadados
            summary["proveniencia"] = provenance_block(metadados)
            if protocol is not None:
                summary["protocolo_ativo"] = protocol
            if cfg is not None and config_path is not None:
                summary["configuracao"] = str(config_path.resolve())
                summary["orquestracao"] = cfg.orchestration
                summary["perfil_baseline"] = cfg.baselines.profile

            juiz_fila: list[str] = []
            min_score = 0.5
            if protocol:
                agg = protocol.get("judge_aggregation_verdicts")
                if isinstance(agg, list):
                    juiz_fila = [str(x) for x in agg]
                fs = protocol.get("fila_min_score_recuperacao")
                if isinstance(fs, int | float):
                    min_score = float(fs)

            fila_path, fila_counts = export_fila_csv(
                run_dir,
                records,
                juiz_vereditos_fila=juiz_fila,
                min_score_recuperacao=min_score,
            )
            if isinstance(summary.get("sumario_operacional"), dict):
                so = cast(dict[str, object], summary["sumario_operacional"])
                so["fila_revisao_csv"] = fila_path.relative_to(run_dir).as_posix()
                so["fila_revisao_humana"] = fila_counts

            write_summary(summary, summary_path)

            extra: list[Path] = [fila_path]
            hitl_manifest = run_dir / "analise_manual" / "hitl_manifest.json"
            if hitl_manifest.is_file():
                extra.append(hitl_manifest)
            adj = run_dir / "analise_manual" / "adjudicacoes_hitl.csv"
            if adj.is_file():
                extra.append(adj)
                written = write_hitl_manifest(run_dir, adj)
                if written not in extra:
                    extra.append(written)

            manifest = build_manifest(run_dir, metadados=metadados, extra_files=extra)
            write_manifest(run_dir, manifest)
            return summary

    return retry_call(
        _reprocess_once,
        attempts=retry_attempts,
        retry_on=(OSError, TimeoutError, json.JSONDecodeError),
    )

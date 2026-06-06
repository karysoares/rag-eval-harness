"""Manifest, integridade e metadados de corrida (SPEC-005, Fase 1)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_evaluation.config import AppConfig
from llm_evaluation.prompt_resources import prompt_bytes, source_prompts_dir
from llm_evaluation.schema_registry import (
    MANIFEST_SCHEMA_VERSION,
    PREDICTIONS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    validate_manifest,
    validate_prediction_record,
    validate_summary,
)

SPEC_DEPENDENCIES = ("001", "002", "003", "004", "005", "007")


class CorruptedPredictionsError(ValueError):
    """JSONL de predictions ilegível — bloqueia ``--resume``."""


_ARTIFACT_FILES = (
    "predictions.jsonl",
    "summary.json",
    "anomalies.jsonl",
    "anomalies.csv",
    "baseline_comparison.json",
    "metrics_report.json",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prompts_dir() -> Path:
    return source_prompts_dir()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_short() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=project_root(),
            check=False,
        )
        if r.returncode == 0:
            out = r.stdout.strip()
            return out or None
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def config_hash(config_path: Path) -> str | None:
    if not config_path.is_file():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _hash_prompt_file(name: str) -> str | None:
    try:
        data = prompt_bytes(name)
    except FileNotFoundError:
        return None
    return hashlib.sha256(data).hexdigest()


def prompt_files_for_config(cfg: AppConfig) -> list[str]:
    files: list[str] = []
    files.extend(["responder_system.txt", "responder_user_template.txt"])
    jps = cfg.verification.judge_prompt_style
    if jps == "rag_pt":
        files.extend(["judge_rag_pt_system.txt", "judge_rag_pt_user_template.txt"])
    else:
        files.extend(["judge_system.txt", "judge_user_template.txt"])
    if cfg.orchestration == "multiplo":
        files.append("critic_system.txt")
    return files


def compute_prompt_hashes(cfg: AppConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in prompt_files_for_config(cfg):
        digest = _hash_prompt_file(name)
        if digest:
            out[name] = digest
    return out


def collect_run_metadata(
    cfg: AppConfig,
    *,
    config_path: Path,
    run_dir: Path,
    n_records: int,
) -> dict[str, Any]:
    from llm_evaluation.llm_client import resolve_models_from_env

    llm_model, judge_model = resolve_models_from_env()
    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "run_id": run_dir.name,
        "git_commit": git_commit_short(),
        "config_hash_sha256": config_hash(config_path),
        "config_path": str(config_path.resolve()),
        "dataset_id": cfg.dataset.name,
        "dataset": {
            "name": cfg.dataset.name,
            "subset": cfg.dataset.subset,
            "split": cfg.dataset.split,
            "reference_type": cfg.dataset.reference_type,
            "hf_repo": cfg.dataset.hf_repo,
            "limit": cfg.dataset.limit,
            "mode": cfg.dataset.mode,
        },
        "modelos": {
            "llm_geracao": llm_model,
            "llm_juiz": judge_model,
            "embeddings": cfg.embeddings.model_name,
            "embeddings_backend": cfg.embeddings.backend,
        },
        "prompt_hashes_sha256": compute_prompt_hashes(cfg),
        "n_registos": n_records,
        "seed": cfg.seed,
        "orquestracao": cfg.orchestration,
        "perfil_baseline": cfg.baselines.profile,
    }


def row_has_processing_error(row: dict[str, object]) -> bool:
    """True quando o item falhou na pipeline (ex.: 429) e deve ser reprocessado."""
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return False
    err = meta.get("processing_error")
    return isinstance(err, dict)


def load_completed_item_ids(predictions_path: Path) -> set[str]:
    """IDs já gravados com sucesso em ``predictions.jsonl`` (para retomar corrida).

    Linhas com ``meta.processing_error`` não contam como concluídas — o ``--resume``
    volta a processá-las (última linha por ID ganha na agregação).

    Qualquer ``JSONDecodeError`` bloqueia retomada com :class:`CorruptedPredictionsError`.
    """
    done: set[str] = set()
    if not predictions_path.is_file():
        return done
    with predictions_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                msg = (
                    f"predictions.jsonl corrompido na linha {line_no}: {e}. "
                    "Não use --resume; repare ou remova o ficheiro."
                )
                raise CorruptedPredictionsError(msg) from e
            if not isinstance(row, dict) or row_has_processing_error(row):
                continue
            iid = row.get("id_item") or row.get("item_id")
            if iid:
                done.add(str(iid))
    return done


def count_failed_item_ids(predictions_path: Path) -> int:
    """Quantos IDs têm pelo menos uma linha com ``processing_error``."""
    if not predictions_path.is_file():
        return 0
    failed: set[str] = set()
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not row_has_processing_error(row):
                continue
            iid = row.get("id_item") or row.get("item_id")
            if iid:
                failed.add(str(iid))
    return len(failed)


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _manifest_relpath(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _resolve_manifest_file(run_dir: Path, nome: str) -> Path:
    """Resolve entrada do manifest (nome simples ou relativo ao ``run_dir``)."""
    return run_dir / nome


def _file_entry(
    path: Path,
    run_dir: Path,
    *,
    schema_version: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "nome": _manifest_relpath(path, run_dir),
        "tamanho_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        entry["n_linhas"] = count_jsonl_lines(path)
        entry["schema_version"] = schema_version or PREDICTIONS_SCHEMA_VERSION
    elif path.suffix == ".json":
        entry["schema_version"] = schema_version or SUMMARY_SCHEMA_VERSION
    return entry


def build_manifest(
    run_dir: Path,
    *,
    metadados: dict[str, Any],
    extra_files: list[Path] | None = None,
) -> dict[str, Any]:
    paths: list[Path] = []
    for name in _ARTIFACT_FILES:
        p = run_dir / name
        if p.is_file():
            paths.append(p)
    if extra_files:
        for p in extra_files:
            if p.is_file() and p not in paths:
                paths.append(p)
    for p in sorted(run_dir.glob("predictions_*.jsonl")):
        if p.is_file() and p not in paths:
            paths.append(p)

    ficheiros = [_file_entry(p, run_dir) for p in paths]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "criado_em_utc": datetime.now(tz=UTC).isoformat(),
        "metadados_corrida": metadados,
        "dependencias_specs": list(SPEC_DEPENDENCIES),
        "ficheiros": ficheiros,
        "integridade": {
            "checksum_por_linha_jsonl": None,
            "nota": (
                "SHA256 por ficheiro completo. Checksum por linha em JSONL é roadmap "
                "(Fase 1 documenta apenas hash do ficheiro final)."
            ),
        },
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    atomic_write_json(path, manifest)
    return path


def finalize_predictions_jsonl(tmp_path: Path, final_path: Path) -> None:
    """Renomeia ficheiro incremental (.tmp) para o nome final de forma atómica."""
    if not tmp_path.is_file():
        msg = f"Ficheiro temporário em falta: {tmp_path}"
        raise FileNotFoundError(msg)
    os.replace(tmp_path, final_path)


def compact_predictions_jsonl(path: Path) -> int:
    """Reescreve JSONL mantendo a última linha por ``id_item`` (idempotência de resume).

    Retorna o número de linhas duplicadas removidas.
    """
    if not path.is_file():
        return 0
    by_id: dict[str, str] = {}
    order: list[str] = []
    original = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            original += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            iid = row.get("id_item") or row.get("item_id")
            if not iid:
                continue
            sid = str(iid)
            if sid not in by_id:
                order.append(sid)
            by_id[sid] = stripped
    if original <= len(by_id):
        return 0
    content = "\n".join(by_id[iid] for iid in order) + ("\n" if by_id else "")
    atomic_write_text(path, content)
    return original - len(by_id)


def validate_run_artifacts(run_dir: Path, *, strict: bool = False) -> list[str]:
    """Valida invariantes de artefactos; avisos para corridas legadas sem manifest."""
    issues: list[str] = []
    pred = run_dir / "predictions.jsonl"
    multi_preds = sorted(run_dir.glob("predictions_*.jsonl"))
    if not pred.is_file() and not multi_preds:
        issues.append(f"{run_dir.name}: sem predictions.jsonl")
        return issues

    primary = pred if pred.is_file() else multi_preds[0]
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(f"{run_dir.name}: summary.json inválido: {e}")
            summary = {}
        else:
            issues.extend(f"{run_dir.name}: {m}" for m in validate_summary(summary, strict=strict))
    else:
        summary = {}
        issues.append(f"aviso: {run_dir.name}: sem summary.json")

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            issues.append(f"{run_dir.name}: manifest.json inválido: {e}")
        else:
            issues.extend(
                f"{run_dir.name}: {m}" for m in validate_manifest(manifest, strict=strict)
            )
            for fe in manifest.get("ficheiros") or []:
                if not isinstance(fe, dict):
                    continue
                nome = fe.get("nome")
                if not nome:
                    continue
                nome_s = str(nome)
                expected = fe.get("sha256")
                expected_size = fe.get("tamanho_bytes")
                fp = _resolve_manifest_file(run_dir, nome_s)
                if strict:
                    if not fp.is_file():
                        issues.append(
                            f"{run_dir.name}: ficheiro em falta no manifest: {nome_s}",
                        )
                        continue
                    if expected_size is not None and fp.stat().st_size != expected_size:
                        issues.append(
                            f"{run_dir.name}: tamanho {nome_s} não coincide com manifest",
                        )
                if expected and fp.is_file():
                    actual = sha256_file(fp)
                    if actual != expected:
                        issues.append(
                            f"{run_dir.name}: checksum {nome_s} não coincide com manifest",
                        )
                elif strict and expected and not fp.is_file():
                    issues.append(
                        f"{run_dir.name}: checksum pendente — ficheiro em falta: {nome_s}",
                    )
    else:
        issues.append(f"aviso: {run_dir.name}: sem manifest.json (corrida legada)")

    with primary.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"{run_dir.name}: JSONL linha {i}: {e}")
                continue
            if isinstance(obj, dict):
                issues.extend(
                    f"{run_dir.name}: linha {i}: {m}"
                    for m in validate_prediction_record(obj, strict=strict)
                )
            if not strict and i >= 3:
                break

    if summary and manifest:
        meta_sum = summary.get("metadados_corrida")
        meta_man = manifest.get("metadados_corrida")
        if (
            meta_sum
            and meta_man
            and meta_sum.get("config_hash_sha256")
            != meta_man.get(
                "config_hash_sha256",
            )
        ):
            issues.append(f"aviso: {run_dir.name}: config_hash diverge summary vs manifest")

    return issues

"""I/O de rótulos HITL: CSV ↔ predictions.jsonl."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from llm_evaluation.resilience import file_lock, retry_call
from llm_evaluation.run_artifacts import atomic_write_json, atomic_write_text, sha256_file

HITL_CSV_FIELDS = ("id_item", "rotulo", "revisor", "timestamp_utc", "notas")
VALID_ROTULOS = frozenset(
    {"correto", "incorreto", "parcial", "recusa_ok", "inconclusivo"},
)
ROTULOS_DISPLAY = (
    ("correto", "Correto"),
    ("incorreto", "Incorreto"),
    ("parcial", "Parcial"),
    ("recusa_ok", "Recusa OK"),
    ("inconclusivo", "Inconclusivo"),
)


def _parse_row(row: dict[str, str]) -> dict[str, str] | None:
    iid = (row.get("id_item") or row.get("item_id") or "").strip()
    rotulo = (row.get("rotulo") or row.get("adjudicacao_humana") or "").strip().lower()
    if not iid or not rotulo:
        return None
    if rotulo not in VALID_ROTULOS:
        return None
    return {
        "id_item": iid,
        "rotulo": rotulo,
        "revisor": (row.get("revisor") or "").strip(),
        "timestamp_utc": (row.get("timestamp_utc") or "").strip(),
        "notas": (row.get("notas") or "").strip(),
    }


def read_hitl_csv(path: Path, *, strict: bool = False) -> dict[str, dict[str, str]]:
    """Última linha por ``id_item`` ganha.

    Em ``strict=True``, linhas não vazias inválidas bloqueiam caminhos que aplicam
    ou publicam HITL, em vez de serem descartadas silenciosamente.
    """

    def _read() -> dict[str, dict[str, str]]:
        by_id: dict[str, dict[str, str]] = {}
        invalid_lines: list[int] = []
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if strict and not reader.fieldnames:
                msg = f"CSV HITL sem cabeçalho: {path}"
                raise ValueError(msg)
            for ln, row in enumerate(reader, start=2):
                parsed = _parse_row({k: str(v or "") for k, v in row.items()})
                if parsed:
                    by_id[parsed["id_item"]] = parsed
                elif strict and any(str(v or "").strip() for v in row.values()):
                    invalid_lines.append(ln)
        if invalid_lines:
            preview = ", ".join(str(x) for x in invalid_lines[:8])
            msg = f"CSV HITL com {len(invalid_lines)} linhas inválidas (ex.: {preview})"
            raise ValueError(msg)
        return by_id

    return retry_call(
        _read,
        attempts=3,
        retry_on=(OSError, UnicodeDecodeError, csv.Error, TimeoutError),
    )


def merge_hitl_csv_into_predictions(
    csv_path: Path,
    predictions_path: Path,
    *,
    strict_ids: bool = False,
) -> int:
    """Patch ``meta.adjudicacao_humana`` por linha; devolve N actualizados."""
    labels = read_hitl_csv(csv_path, strict=True)
    if not labels:
        return 0
    lock = predictions_path.with_name(predictions_path.name + ".lock")

    def _merge() -> int:
        with file_lock(lock):
            lines_out: list[str] = []
            seen_ids: set[str] = set()
            updated = 0
            with predictions_path.open(encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        msg = f"predictions.jsonl inválido (linha {ln}): {e}"
                        raise ValueError(msg) from e
                    iid = str(obj.get("id_item") or obj.get("item_id") or "")
                    if iid:
                        seen_ids.add(iid)
                    if iid in labels:
                        meta = obj.get("meta")
                        if not isinstance(meta, dict):
                            meta = {}
                            obj["meta"] = meta
                        meta["adjudicacao_humana"] = labels[iid]
                        updated += 1
                    lines_out.append(json.dumps(obj, ensure_ascii=False))

            missing = sorted(set(labels.keys()) - seen_ids)
            if strict_ids and missing:
                preview = ", ".join(missing[:8])
                msg = f"{len(missing)} ids do CSV não existem no predictions.jsonl (ex.: {preview})"
                raise ValueError(msg)

            if updated > 0:
                backup = predictions_path.with_name(predictions_path.name + ".bak")
                shutil.copy2(predictions_path, backup)
                atomic_write_text(
                    predictions_path,
                    "\n".join(lines_out) + "\n",
                )
            return updated

    return retry_call(_merge, attempts=3, retry_on=(OSError, TimeoutError))


def write_hitl_manifest(run_dir: Path, csv_path: Path) -> Path:
    out_dir = run_dir / "analise_manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = read_hitl_csv(csv_path, strict=True)
    path = out_dir / "hitl_manifest.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "run_dir": str(run_dir.resolve()),
            "csv": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path) if csv_path.is_file() else None,
            "n_rotulados": len(labels),
            "rotulos_validos": sorted(VALID_ROTULOS),
        },
    )
    return path


def hitl_csv_path(run_dir: Path) -> Path:
    return run_dir / "analise_manual" / "adjudicacoes_hitl.csv"


def hitl_staging_path(run_dir: Path) -> Path:
    return run_dir / "analise_manual" / ".staging" / "adjudicacoes_hitl.csv"


def write_staged_hitl_csv(run_dir: Path, content: bytes) -> Path:
    """Grava upload em staging — não substitui ``adjudicacoes_hitl.csv`` até commit."""
    staged = hitl_staging_path(run_dir)
    staged.parent.mkdir(parents=True, exist_ok=True)
    tmp = staged.with_suffix(".csv.tmp")
    tmp.write_bytes(content)
    read_hitl_csv(tmp, strict=True)
    os.replace(tmp, staged)
    return staged


def validate_hitl_csv_ids(csv_path: Path, predictions_path: Path) -> list[str]:
    """IDs do CSV que não existem em ``predictions.jsonl``."""
    labels = read_hitl_csv(csv_path, strict=True)
    if not labels or not predictions_path.is_file():
        return sorted(labels.keys()) if labels else []
    seen: set[str] = set()
    with predictions_path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"predictions.jsonl inválido (linha {ln}): {e}"
                raise ValueError(msg) from e
            iid = str(obj.get("id_item") or obj.get("item_id") or "")
            if iid:
                seen.add(iid)
    return sorted(set(labels.keys()) - seen)


def commit_staged_hitl_csv(run_dir: Path, *, strict_ids: bool = True) -> Path:
    """Valida staging e promove para ``adjudicacoes_hitl.csv`` de forma atómica."""
    staged = hitl_staging_path(run_dir)
    if not staged.is_file():
        msg = "Sem CSV em staging para aplicar"
        raise ValueError(msg)
    pred = run_dir / "predictions.jsonl"
    missing = validate_hitl_csv_ids(staged, pred)
    if strict_ids and missing:
        preview = ", ".join(missing[:8])
        msg = f"{len(missing)} ids do CSV não existem no predictions.jsonl (ex.: {preview})"
        raise ValueError(msg)
    dest = hitl_csv_path(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        shutil.copy2(dest, dest.with_suffix(".csv.bak"))
    tmp = dest.with_suffix(".csv.tmp")
    shutil.copy2(staged, tmp)
    os.replace(tmp, dest)
    return dest


def write_hitl_csv(path: Path, labels: dict[str, dict[str, str]]) -> None:
    """Grava todas as adjudicações (ordem estável por id_item)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(HITL_CSV_FIELDS))
    w.writeheader()
    for iid in sorted(labels.keys()):
        row = labels[iid]
        w.writerow({k: row.get(k, "") for k in HITL_CSV_FIELDS})
    atomic_write_text(path, buf.getvalue())


def save_hitl_annotation(
    run_dir: Path,
    *,
    item_id: str,
    rotulo: str,
    revisor: str = "",
    notas: str = "",
    timestamp_utc: str | None = None,
) -> Path:
    """Acrescenta ou actualiza uma linha em ``adjudicacoes_hitl.csv``."""
    rot = rotulo.strip().lower()
    if rot not in VALID_ROTULOS:
        msg = f"Rótulo inválido: {rotulo!r}. Use: {sorted(VALID_ROTULOS)}"
        raise ValueError(msg)
    iid = item_id.strip()
    if not iid:
        raise ValueError("id_item em falta")
    path = hitl_csv_path(run_dir)
    lock = path.with_name(path.name + ".lock")

    def _save() -> Path:
        with file_lock(lock):
            labels = read_hitl_csv(path) if path.is_file() else {}
            labels[iid] = {
                "id_item": iid,
                "rotulo": rot,
                "revisor": revisor.strip(),
                "timestamp_utc": timestamp_utc or datetime.now(tz=UTC).isoformat(),
                "notas": notas.strip(),
            }
            write_hitl_csv(path, labels)
        return path

    return retry_call(_save, attempts=3, retry_on=(OSError, TimeoutError))


def load_hitl_labels_merged(run_dir: Path) -> dict[str, dict[str, str]]:
    """CSV primeiro; completa com ``meta.adjudicacao_humana`` do JSONL se ausente."""
    path = hitl_csv_path(run_dir)
    pred = run_dir / "predictions.jsonl"

    def _load() -> dict[str, dict[str, str]]:
        labels = read_hitl_csv(path) if path.is_file() else {}
        if not pred.is_file():
            return labels
        with pred.open(encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    msg = f"predictions.jsonl inválido (linha {ln}): {e}"
                    raise ValueError(msg) from e
                iid = str(obj.get("id_item") or obj.get("item_id") or "")
                if not iid or iid in labels:
                    continue
                meta = obj.get("meta")
                if not isinstance(meta, dict):
                    continue
                adj = meta.get("adjudicacao_humana")
                if isinstance(adj, dict) and adj.get("rotulo"):
                    labels[iid] = {
                        "id_item": iid,
                        "rotulo": str(adj.get("rotulo") or ""),
                        "revisor": str(adj.get("revisor") or ""),
                        "timestamp_utc": str(adj.get("timestamp_utc") or ""),
                        "notas": str(adj.get("notas") or ""),
                    }
        return labels

    return retry_call(
        _load,
        attempts=3,
        retry_on=(OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError),
    )


# Alias público (evita ImportError com reload do Streamlit em `load_hitl_labels_merged`).
load_hitl_labels = load_hitl_labels_merged


def export_hitl_csv_template(path: Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(HITL_CSV_FIELDS))
        w.writeheader()
        for iid in item_ids:
            w.writerow(
                {
                    "id_item": iid,
                    "rotulo": "",
                    "revisor": "",
                    "timestamp_utc": "",
                    "notas": "",
                },
            )

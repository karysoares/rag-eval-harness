#!/usr/bin/env python3
"""Publica agregados de uma corrida em docs/evidencia/ (sem PII)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_evaluation.run_artifacts import (
    atomic_write_json,
    atomic_write_text,
    validate_run_artifacts,
)


def redact_path_string(value: str) -> str:
    """Remove prefixos absolutos (home, /Users/...) mantendo sufixo útil."""
    if not value:
        return value
    home = str(Path.home())
    if not (value.startswith(home) or value.startswith("/Users/") or value.startswith("/home/")):
        return value
    parts = Path(value).parts
    for i, part in enumerate(parts):
        if part.startswith("run_"):
            return str(Path(*parts[i:]))
    return Path(value).name


def redact_absolute_paths(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_path_string(obj)
    if isinstance(obj, dict):
        return {k: redact_absolute_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_absolute_paths(x) for x in obj]
    return obj


def predictions_contain_judge_cot(run_dir: Path) -> bool:
    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        return False
    with pred.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            sinais = obj.get("sinais") or obj.get("signals")
            if not isinstance(sinais, dict):
                continue
            juiz = sinais.get("juiz") or sinais.get("judge")
            if isinstance(juiz, dict) and juiz.get("cadeia_de_pensamento"):
                return True
    return False


def assert_publishable(run_dir: Path) -> None:
    """Pré-voos: integridade estrita, policy P0 e ausência de CoT do juiz."""
    issues = [i for i in validate_run_artifacts(run_dir, strict=True) if not i.startswith("aviso:")]
    if issues:
        msg = "Validação estrita falhou:\n" + "\n".join(f"  - {i}" for i in issues)
        raise SystemExit(msg)

    if predictions_contain_judge_cot(run_dir):
        raise SystemExit(
            "Publicação bloqueada: predictions contém cadeia_de_pensamento do juiz. "
            "Desligue judge_return_chain_of_thought ou regrave a corrida.",
        )

    report = run_embedding_policy(run_dir, write=True)
    criterio = report.get("criterio_p0")
    if not isinstance(criterio, dict):
        raise SystemExit("criterio_p0 em falta — publicação abortada")
    if criterio.get("aplicavel") is not False and not criterio.get("passou"):
        raise SystemExit("criterio_p0.passou é false — publicação abortada")


def run_embedding_policy(run_dir: Path, *, write: bool = True) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "validate_embedding_policy.py"
    cmd = [sys.executable, str(script), str(run_dir)]
    if write:
        cmd.append("--write")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        msg = f"validate_embedding_policy falhou (exit {proc.returncode})"
        if detail:
            msg += f": {detail[:500]}"
        raise SystemExit(msg)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"validate_embedding_policy devolveu JSON inválido: {e}") from e


def publish_run_evidence(run_dir: Path, dest: Path) -> None:
    run_dir = run_dir.expanduser().resolve()
    run_id = run_dir.name
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Sem summary.json em {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    protocolo = redact_absolute_paths(
        {
            "protocolo_ativo": summary.get("protocolo_ativo"),
            "detector_activo": summary.get("detector_activo"),
            "sumario_operacional": summary.get("sumario_operacional"),
            "proveniencia": summary.get("proveniencia"),
        },
    )
    atomic_write_json(dest / f"{run_id}_protocolo.json", protocolo)

    kpi = redact_absolute_paths(
        {
            "sumario_lexical": summary.get("sumario_lexical"),
            "sumario_recuperacao": summary.get("sumario_recuperacao"),
            "sumario_gap_rag_resposta": summary.get("sumario_gap_rag_resposta"),
            "sumario_hitl": summary.get("sumario_hitl"),
        },
    )
    atomic_write_json(dest / f"{run_id}_kpi_lexical.json", kpi)

    policy_src = run_dir / "policy_validation.json"
    if policy_src.is_file():
        policy_obj = redact_absolute_paths(json.loads(policy_src.read_text(encoding="utf-8")))
        atomic_write_json(dest / f"{run_id}_policy_validation.json", policy_obj)

    readme = dest / "README.md"
    n = summary.get("n_itens")
    pol = summary.get("protocolo_ativo")
    agg = pol.get("aggregation_policy") if isinstance(pol, dict) else "?"
    line = (
        f"- **{run_id}**: N={n}, política={agg}, "
        f"origem=`outputs/{run_id}` (path absoluto redigido).\n"
    )
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if run_id not in text:
            atomic_write_text(readme, text.rstrip() + "\n" + line)
    else:
        atomic_write_text(
            readme,
            "# Evidência de corridas (agregados)\n\n" + line,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Copia agregados para docs/evidencia/")
    p.add_argument("run_dir", type=Path)
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("docs/evidencia"),
    )
    p.add_argument(
        "--skip-gates",
        action="store_true",
        help="Apenas para testes internos — não usar em CI/publicação real",
    )
    args = p.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not args.skip_gates:
        assert_publishable(run_dir)
    publish_run_evidence(run_dir, args.dest)
    print(f"Evidência em {args.dest.expanduser().resolve()}")


if __name__ == "__main__":
    main()

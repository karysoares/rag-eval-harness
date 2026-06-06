"""Integridade pós-corrida: manifest alinhado com summary e fila."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from llm_evaluation.config import load_config
from llm_evaluation.eval_items_load import load_eval_items
from llm_evaluation.run_artifacts import sha256_file, validate_run_artifacts


def test_run_single_corrida_manifest_matches_summary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Espelha ``_run_single_corrida``: um único write_summary e manifest com fila."""
    from llm_evaluation.cli import _run_single_corrida
    from llm_evaluation.protocol import ProtocolAdjustment

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    cfg = replace(
        cfg,
        output_dir=str(tmp_path),
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
    )
    items = load_eval_items(cfg)[:1]
    run_dir = tmp_path / "run_test_integrity"
    run_dir.mkdir()

    class _Llm:
        def complete(self, system: str, user: str) -> str:
            if "veredito" in system.lower() or "veredito" in user.lower():
                return json.dumps(
                    {
                        "veredito": "sustentado",
                        "motivo_breve": "ok",
                        "confianca": 0.9,
                    },
                )
            return json.dumps(
                {
                    "resposta": "Lisboa",
                    "confianca": 0.9,
                    "contexto_insuficiente": False,
                }
            )

    fake = _Llm()
    monkeypatch.setattr(
        "llm_evaluation.pipeline.default_llm_from_env",
        lambda **_k: fake,
    )
    monkeypatch.setattr(
        "llm_evaluation.pipeline.default_judge_from_env",
        lambda **_k: fake,
    )

    _run_single_corrida(
        cfg,
        items,
        run_dir,
        repo / "configs/smoke_amostra.yaml",
        [ProtocolAdjustment("verification.verify_embedding", True, False, "teste")],
    )

    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    assert summary_path.is_file()
    assert manifest_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    op = summary.get("sumario_operacional")
    assert isinstance(op, dict)
    assert op.get("fila_revisao_csv")
    assert summary.get("protocolo_ajustado")
    fila_csv = run_dir / "analise_manual" / "fila_revisao_humana.csv"
    assert fila_csv.is_file()

    for fe in manifest.get("ficheiros") or []:
        if fe.get("nome") == "summary.json":
            assert fe["sha256"] == sha256_file(summary_path)
        nome = fe.get("nome")
        if nome in ("fila_revisao_humana.csv", "analise_manual/fila_revisao_humana.csv"):
            assert fe["sha256"] == sha256_file(fila_csv)

    issues = [i for i in validate_run_artifacts(run_dir, strict=True) if "aviso:" not in i]
    assert issues == []


def test_dry_run_no_api(monkeypatch, tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_evaluation.cli",
            "--config",
            "configs/smoke_amostra.yaml",
            "--dry-run",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), "OPENAI_API_KEY": ""},
    )
    assert proc.returncode == 0
    assert "Dry-run" in proc.stdout


def test_multiplo_requires_experimental() -> None:
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_evaluation.cli",
            "--config",
            "configs/smoke_amostra.yaml",
            "--orchestration",
            "multiplo",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "experimental" in proc.stderr.lower() or "experimental" in proc.stdout.lower()

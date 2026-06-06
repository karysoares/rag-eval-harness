"""Integração release: corrida mock completa, auditoria strict e retomada sem duplicados."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from llm_evaluation.config import load_config
from llm_evaluation.eval_items_load import load_eval_items
from llm_evaluation.run_artifacts import sha256_file, validate_run_artifacts
from llm_evaluation.run_reprocess import reprocess_run_dir
from llm_evaluation.schema_registry import validate_summary


def _mock_llm(monkeypatch) -> None:
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
                    "resposta": "Brasília é a capital.",
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


def _smoke_cfg(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    return replace(
        cfg,
        output_dir=str(tmp_path),
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
    )


def test_release_mocked_run_passes_strict_audit(monkeypatch, tmp_path: Path) -> None:
    from llm_evaluation.cli import _run_single_corrida

    repo = Path(__file__).resolve().parents[1]
    cfg = _smoke_cfg(tmp_path)
    items = load_eval_items(cfg)
    run_dir = tmp_path / "run_release"
    run_dir.mkdir()
    _mock_llm(monkeypatch)

    _run_single_corrida(cfg, items, run_dir, repo / "configs/smoke_amostra.yaml")

    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    assert summary_path.is_file()
    assert manifest_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema_issues = [
        i for i in validate_summary(summary, strict=True) if not i.startswith("aviso:")
    ]
    assert schema_issues == []

    for fe in manifest.get("ficheiros") or []:
        nome = fe.get("nome")
        if not nome:
            continue
        rel = run_dir / str(nome)
        if rel.is_file():
            assert fe["sha256"] == sha256_file(rel)

    artifact_issues = [i for i in validate_run_artifacts(run_dir, strict=True) if "aviso:" not in i]
    assert artifact_issues == []

    proc = subprocess.run(
        [sys.executable, str(repo / "scripts/audit_run.py"), str(run_dir), "--strict"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    before = {
        "n_itens": summary["n_itens"],
        "taxa_alerta": summary.get("taxa_alerta"),
        "kpi_primario": summary.get("kpi_primario"),
    }
    reproc = reprocess_run_dir(run_dir, cfg=cfg, config_path=repo / "configs/smoke_amostra.yaml")
    assert reproc["n_itens"] == before["n_itens"]
    assert reproc.get("taxa_alerta") == before["taxa_alerta"]
    assert reproc.get("kpi_primario") == before["kpi_primario"]

    pred_path = run_dir / "predictions.jsonl"
    lines_before = [ln for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids_before = {json.loads(ln)["id_item"] for ln in lines_before}

    _run_single_corrida(
        cfg,
        items,
        run_dir,
        repo / "configs/smoke_amostra.yaml",
        resume=True,
    )

    lines_after = [ln for ln in pred_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids_after = {json.loads(ln)["id_item"] for ln in lines_after}
    assert len(lines_after) == len(lines_before)
    assert ids_after == ids_before == {it.id for it in items}

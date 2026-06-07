"""Segurança e integridade: manifest, prompts, publicação, HITL, resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_evaluation.config import load_config
from llm_evaluation.hitl_io import (
    commit_staged_hitl_csv,
    hitl_csv_path,
    read_hitl_csv,
    write_staged_hitl_csv,
)
from llm_evaluation.reporting import ensure_run_dir, record_to_json, summarize, write_summary
from llm_evaluation.run_artifacts import (
    CorruptedPredictionsError,
    atomic_write_json,
    build_manifest,
    collect_run_metadata,
    load_completed_item_ids,
    sha256_file,
    validate_run_artifacts,
    write_manifest,
)
from llm_evaluation.types import RunRecord, VerificationSignals


def _minimal_record(iid: str = "t1") -> RunRecord:
    return RunRecord(
        item_id=iid,
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )


def _build_minimal_run(tmp_path: Path, *, n_lines: int = 5) -> Path:
    cfg = load_config(Path("configs/smoke_amostra.yaml"))
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    pred = run_dir / "predictions.jsonl"
    lines = [
        json.dumps(record_to_json(_minimal_record(f"i{i}")), ensure_ascii=False)
        for i in range(n_lines)
    ]
    pred.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = summarize([_minimal_record("i0")], reference_type="lexical")
    summary["protocolo_ativo"] = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "embedding_min_cosine": 0.28,
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": ["nao_sustentado"],
    }
    write_summary(summary, run_dir / "summary.json")
    analise = run_dir / "analise_manual"
    analise.mkdir()
    fila = analise / "fila_revisao_humana.csv"
    fila.write_text("id_item,motivo_fila\ni0,juiz_veredito_duro\n", encoding="utf-8")
    meta = collect_run_metadata(
        cfg,
        config_path=Path("configs/smoke_amostra.yaml"),
        run_dir=run_dir,
        n_records=n_lines,
    )
    manifest = build_manifest(run_dir, metadados=meta, extra_files=[fila])
    write_manifest(run_dir, manifest)
    return run_dir


def test_manifest_uses_relative_paths(tmp_path: Path) -> None:
    run_dir = _build_minimal_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    nomes = {fe["nome"] for fe in manifest["ficheiros"]}
    assert "analise_manual/fila_revisao_humana.csv" in nomes
    assert "fila_revisao_humana.csv" not in nomes


def test_strict_validation_fails_when_analise_manual_csv_tampered(tmp_path: Path) -> None:
    run_dir = _build_minimal_run(tmp_path)
    fila = run_dir / "analise_manual" / "fila_revisao_humana.csv"
    fila.write_text("id_item,motivo_fila\ni0,alterado\n", encoding="utf-8")
    issues = [i for i in validate_run_artifacts(run_dir, strict=True) if "aviso:" not in i]
    assert any("checksum" in i or "fila_revisao" in i for i in issues)


def test_strict_validation_fails_when_analise_manual_csv_removed(tmp_path: Path) -> None:
    run_dir = _build_minimal_run(tmp_path)
    (run_dir / "analise_manual" / "fila_revisao_humana.csv").unlink()
    issues = [i for i in validate_run_artifacts(run_dir, strict=True) if "aviso:" not in i]
    assert any("em falta" in i for i in issues)


def test_strict_jsonl_validates_all_lines_not_sample(tmp_path: Path) -> None:
    run_dir = _build_minimal_run(tmp_path, n_lines=6)
    pred = run_dir / "predictions.jsonl"
    text = pred.read_text(encoding="utf-8").splitlines()
    text[5] = "{invalid json on line 6"
    pred.write_text("\n".join(text) + "\n", encoding="utf-8")
    issues = validate_run_artifacts(run_dir, strict=True)
    assert any("linha 6" in i for i in issues)


def test_gitignore_protects_local_files_without_hiding_packaged_prompts() -> None:
    repo = Path(__file__).resolve().parents[1]

    ignored = subprocess.run(
        ["git", "check-ignore", ".env", "prompts/critic_system.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0

    packaged = subprocess.run(
        ["git", "check-ignore", "src/llm_evaluation/prompts/critic_system.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert packaged.returncode == 1


def test_load_completed_item_ids_blocks_corrupted_jsonl(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(record_to_json(_minimal_record("ok")), ensure_ascii=False) + "\n" + "{trunc",
        encoding="utf-8",
    )
    with pytest.raises(CorruptedPredictionsError, match="linha 2"):
        load_completed_item_ids(pred)


def test_resume_cli_exits_on_corrupted_jsonl(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "run_corrupt"
    run_dir.mkdir()
    pred = run_dir / "predictions.jsonl"
    pred.write_text("{bad\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_evaluation.cli",
            "--config",
            "configs/smoke_amostra.yaml",
            "--resume",
            str(run_dir),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OPENAI_API_KEY": "sk-test-dummy"},
    )
    assert proc.returncode == 2
    assert "corrompido" in proc.stderr.lower() or "corrompido" in proc.stdout.lower()


def test_hitl_staging_invalid_id_does_not_alter_committed_csv(tmp_path: Path) -> None:
    run_dir = _build_minimal_run(tmp_path, n_lines=2)
    dest = hitl_csv_path(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\ni0,correto,ana,,\n",
        encoding="utf-8",
    )
    summary_before = (run_dir / "summary.json").read_bytes()
    write_staged_hitl_csv(
        run_dir,
        b"id_item,rotulo,revisor,timestamp_utc,notas\nghost-id,incorreto,x,,\n",
    )
    with pytest.raises(ValueError, match="não existem"):
        commit_staged_hitl_csv(run_dir, strict_ids=True)
    assert read_hitl_csv(dest)["i0"]["rotulo"] == "correto"
    assert (run_dir / "summary.json").read_bytes() == summary_before


def test_ensure_run_dir_avoids_same_second_collision(tmp_path: Path) -> None:
    first = ensure_run_dir(tmp_path)
    second = ensure_run_dir(tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_publish_run_evidence_aborts_on_strict_failure(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    run_dir = _build_minimal_run(tmp_path)
    pred = run_dir / "predictions.jsonl"
    pred.write_text(pred.read_text(encoding="utf-8") + "{bad\n", encoding="utf-8")
    dest = tmp_path / "evidencia"
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "publish_run_evidence.py"),
            str(run_dir),
            "--dest",
            str(dest),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "Validação estrita falhou" in proc.stderr or "Validação estrita falhou" in proc.stdout
    assert not list(dest.glob("*_protocolo.json"))


def _load_publish_module():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "publish_run_evidence",
        root / "scripts" / "publish_run_evidence.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_publish_redacts_absolute_paths(tmp_path: Path) -> None:
    pub = _load_publish_module()
    redact_absolute_paths = pub.redact_absolute_paths
    publish_run_evidence = pub.publish_run_evidence

    obj = {"path": "/Users/alice/outputs/run_20260101T120000Z/summary.json", "n": 1}
    red = redact_absolute_paths(obj)
    assert "/Users/" not in red["path"]
    assert "run_20260101T120000Z" in red["path"]

    run_dir = _build_minimal_run(tmp_path)
    policy = {
        "criterio_p0": {"passou": True},
        "nota": f"/Users/test/{run_dir.name}",
    }
    atomic_write_json(run_dir / "policy_validation.json", policy)
    dest = tmp_path / "out"
    publish_run_evidence(run_dir, dest)
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "/Users/" not in readme
    assert run_dir.name in readme


def test_publish_blocks_judge_cot_in_predictions(tmp_path: Path) -> None:
    assert_publishable = _load_publish_module().assert_publishable

    run_dir = _build_minimal_run(tmp_path, n_lines=1)
    rec = record_to_json(_minimal_record("i0"))
    rec["sinais"] = {
        "juiz": {"veredito": "sustentado", "cadeia_de_pensamento": ["passo 1"]},
    }
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    pred_path = run_dir / "predictions.jsonl"
    for fe in manifest["ficheiros"]:
        if fe["nome"] == "predictions.jsonl":
            fe["sha256"] = sha256_file(pred_path)
            fe["tamanho_bytes"] = pred_path.stat().st_size
    write_manifest(run_dir, manifest)
    with pytest.raises(SystemExit, match="cadeia_de_pensamento"):
        assert_publishable(run_dir)

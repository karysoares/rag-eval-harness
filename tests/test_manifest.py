import json
from dataclasses import replace
from pathlib import Path

from llm_evaluation.config import load_config
from llm_evaluation.reporting import record_to_json, summarize, write_summary
from llm_evaluation.run_artifacts import (
    atomic_write_json,
    build_manifest,
    collect_run_metadata,
    compute_prompt_hashes,
    sha256_file,
    validate_run_artifacts,
    write_manifest,
)
from llm_evaluation.schema_registry import PREDICTIONS_SCHEMA_VERSION
from llm_evaluation.types import RunRecord, VerificationSignals


def _minimal_record() -> RunRecord:
    return RunRecord(
        item_id="t1",
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


def test_record_to_json_has_schema_version() -> None:
    obj = record_to_json(_minimal_record())
    assert obj["schema_version"] == PREDICTIONS_SCHEMA_VERSION


def test_atomic_write_and_manifest_checksum(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke_amostra.yaml"))
    rec = _minimal_record()
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(json.dumps(record_to_json(rec), ensure_ascii=False) + "\n", encoding="utf-8")
    summary = summarize([rec], reference_type="lexical")
    summary["protocolo_ativo"] = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "embedding_min_cosine": 0.28,
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": ["nao_sustentado"],
    }
    write_summary(summary, tmp_path / "summary.json")
    meta = collect_run_metadata(
        cfg,
        config_path=Path("configs/smoke_amostra.yaml"),
        run_dir=tmp_path,
        n_records=1,
    )
    manifest = build_manifest(tmp_path, metadados=meta)
    write_manifest(tmp_path, manifest)
    issues = [i for i in validate_run_artifacts(tmp_path) if "aviso:" not in i]
    assert issues == []
    pred_entry = next(fe for fe in manifest["ficheiros"] if fe["nome"] == "predictions.jsonl")
    assert sha256_file(pred) == pred_entry["sha256"]


def test_validate_detects_checksum_mismatch(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "ficheiros": [{"nome": "predictions.jsonl", "sha256": "0" * 64}],
    }
    atomic_write_json(tmp_path / "manifest.json", manifest)
    issues = validate_run_artifacts(tmp_path)
    assert any("checksum" in i for i in issues)


def test_multi_orchestration_manifest_hashes_critic_prompt() -> None:
    cfg = load_config(Path("configs/smoke_amostra.yaml"))
    cfg = replace(cfg, orchestration="multiplo")
    hashes = compute_prompt_hashes(cfg)
    assert "critic_system.txt" in hashes

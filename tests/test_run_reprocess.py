"""Reprocessamento offline único."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_evaluation.reporting import record_to_json
from llm_evaluation.run_artifacts import sha256_file
from llm_evaluation.run_reprocess import reprocess_run_dir
from llm_evaluation.types import RunRecord, VerificationSignals


def _lexical_record() -> RunRecord:
    return RunRecord(
        item_id="i1",
        question="q",
        answer="a",
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "metricas_lexicas": {
                "f1_token": 0.8,
                "em_squad": False,
                "texto_referencia": "ref",
            },
        },
    )


def test_reprocess_fixture(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parent / "fixtures/audit_runs/run_ci_fixture"
    shutil.copytree(src, tmp_path / "run", dirs_exist_ok=True)
    run_dir = tmp_path / "run"
    summary = reprocess_run_dir(run_dir)
    assert (run_dir / "summary.json").is_file()
    assert summary.get("n_itens") is not None
    prov = summary.get("proveniencia")
    assert isinstance(prov, dict)
    assert prov.get("versao_pacote") == "1.0.0"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    names = {f.get("nome") for f in manifest.get("ficheiros") or []}
    assert "fila_revisao_humana.csv" in names or "summary.json" in names


def test_reprocess_lexical_without_summary_infers_reference_type(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rec = _lexical_record()
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(record_to_json(rec), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = reprocess_run_dir(run_dir)
    assert summary["tipo_referencia_ativo"] == "lexical"
    assert "avisos_reprocessamento" in summary


def test_reprocess_includes_hitl_manifest_in_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rec = _lexical_record()
    (run_dir / "predictions.jsonl").write_text(
        json.dumps(record_to_json(rec), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    adj = run_dir / "analise_manual" / "adjudicacoes_hitl.csv"
    adj.parent.mkdir()
    adj.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\ni1,correto,qa,,ok\n",
        encoding="utf-8",
    )

    reprocess_run_dir(run_dir)

    hitl_manifest = run_dir / "analise_manual" / "hitl_manifest.json"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = next(
        fe for fe in manifest["ficheiros"] if fe["nome"] == "analise_manual/hitl_manifest.json"
    )
    assert entry["sha256"] == sha256_file(hitl_manifest)

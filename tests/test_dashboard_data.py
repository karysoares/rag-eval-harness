"""Dashboard data layer (sem Streamlit server)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_evaluation.dashboard.data import (
    CALIBRATION_COLUMN_ORDER,
    artifact_fingerprint,
    cache_run_artifacts,
    calibration_view_dataframe,
    clear_run_artifact_cache,
    list_run_dirs,
    load_run_records,
    records_to_dataframe,
    run_integrity_flags,
)
from llm_evaluation.reporting import record_to_json
from llm_evaluation.schema_registry import (
    PREDICTIONS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
)
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def _sample_record() -> RunRecord:
    return RunRecord(
        item_id="t1",
        question="What is X?",
        answer="Y",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.8,
            embedding_low_support=False,
            judge=JudgeResult(veredito="sustentado", motivo_breve="ok", confianca=0.9),
            judge_negative=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "metricas_recuperacao": {
                "rag_ativo": True,
                "score_melhor_chunk": 0.75,
                "rank_chunk_ouro": 1,
            },
        },
    )


def test_records_to_dataframe() -> None:
    df = records_to_dataframe([_sample_record()])
    assert len(df) == 1
    assert df.iloc[0]["score_melhor_chunk"] == 0.75
    assert df.iloc[0]["juiz_confianca"] == 0.9
    assert "gold_incorreto" in df.columns
    assert "chunk_ouro_no_top_k" in df.columns


def test_calibration_view_orders_columns() -> None:
    df = records_to_dataframe([_sample_record()])
    cal = calibration_view_dataframe(df)
    assert cal.columns[0] == "id_item"
    assert cal.iloc[0]["padroes"] == "" or isinstance(cal.iloc[0]["padroes"], str)
    assert "id_item" in CALIBRATION_COLUMN_ORDER


def test_records_to_dataframe_missing_columns_na() -> None:
    rec = _sample_record()
    rec.meta.pop("metricas_recuperacao", None)
    df = records_to_dataframe([rec])
    assert pd.isna(df.iloc[0]["f1_token"])
    assert "padrao_primario" in df.columns


def test_run_integrity_flags_legacy(tmp_path: Path) -> None:
    run = tmp_path / "run_legacy"
    run.mkdir()
    (run / "predictions.jsonl").write_text('{"id_item":"x"}\n', encoding="utf-8")
    flags = run_integrity_flags(run)
    assert flags["legacy_run"] is True
    assert "integrity_score" in flags


def test_cache_run_artifacts_invalidates(tmp_path: Path) -> None:
    clear_run_artifact_cache()
    run = tmp_path / "run_cache"
    run.mkdir()
    pred = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "id_item": "t1",
        "pergunta": "q",
        "resposta": "a",
        "sinais": {},
        "meta": {},
    }
    pred_path = run / "predictions.jsonl"
    pred_path.write_text(json.dumps(pred, ensure_ascii=False) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "n_itens": 1}),
        encoding="utf-8",
    )
    fp1 = artifact_fingerprint(run)
    b1 = cache_run_artifacts(run)
    b2 = cache_run_artifacts(run)
    assert b1 is b2
    pred_path.write_text(json.dumps({**pred, "resposta": "b"}, ensure_ascii=False) + "\n")
    clear_run_artifact_cache()
    fp2 = artifact_fingerprint(run)
    assert fp1 != fp2
    b3 = cache_run_artifacts(run)
    assert b3 is not b1
    assert b3["records"][0].answer == "b"


def test_list_and_load_run(tmp_path: Path) -> None:
    run = tmp_path / "run_test123Z"
    run.mkdir()
    rec = _sample_record()
    with (run / "predictions.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(record_to_json(rec), ensure_ascii=False) + "\n")
    runs = list_run_dirs(tmp_path)
    assert run.name in [p.name for p in runs]
    loaded = load_run_records(run)
    assert len(loaded) == 1
    assert loaded[0].item_id == "t1"

"""Validação offline da política embedding_e_juiz (P0-1)."""

from __future__ import annotations

from pathlib import Path

from llm_evaluation.evaluation_metrics import (
    compare_aggregation_policies,
    load_records_from_predictions_jsonl,
    replay_anomaly_flags,
)


def _fixture_records():
    path = Path(__file__).resolve().parent / "fixtures" / "rag_embedding_fp_replay.jsonl"
    return load_records_from_predictions_jsonl(path)


def test_embedding_e_juiz_eliminates_embedding_only_fp() -> None:
    records = _fixture_records()
    negs = ["nao_sustentado", "contradicacao", "incompleto", "inseguro"]
    or_flags = replay_anomaly_flags(
        records,
        verify_gold=False,
        verify_embedding=True,
        verify_judge=True,
        negative_judge_verdicts=negs,
        policy="qualquer_critico",
    )
    and_flags = replay_anomaly_flags(
        records,
        verify_gold=False,
        verify_embedding=True,
        verify_judge=True,
        negative_judge_verdicts=negs,
        policy="embedding_e_juiz",
    )
    fp_or = sum(1 for r, f in zip(records, or_flags, strict=True) if r.gold_correct is True and f)
    fp_and = sum(1 for r, f in zip(records, and_flags, strict=True) if r.gold_correct is True and f)
    assert fp_or == 3
    assert fp_and == 0


def test_compare_policies_reports_reduction() -> None:
    records = _fixture_records()
    report = compare_aggregation_policies(
        records,
        verify_gold=False,
        verify_embedding=True,
        verify_judge=True,
        negative_judge_verdicts=["nao_sustentado", "contradicacao", "incompleto", "inseguro"],
    )
    reducao = report.get("reducao_fp_relativa_embedding_e_juiz_vs_or")
    assert reducao == 1.0
    pol = report["politicas"]
    assert pol["embedding_e_juiz"]["taxa_falso_alarme_no_gold_correto"] == 0.0
    assert pol["qualquer_critico"]["taxa_falso_alarme_no_gold_correto"] == 1.0

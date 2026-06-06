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
        reference_type="answer_lists",
    )
    reducao = report.get("reducao_fp_relativa_embedding_e_juiz_vs_or")
    assert reducao == 1.0
    pol = report["politicas"]
    assert pol["embedding_e_juiz"]["taxa_falso_alarme_no_gold_correto"] == 0.0
    assert pol["qualquer_critico"]["taxa_falso_alarme_no_gold_correto"] == 1.0
    assert report["n_gold_corretos"] == 3


def test_compare_policies_lexical_uses_metricas_lexicas() -> None:
    """Datasets léxicos: referência aceitável vem de F1/EM, não de gold_correto null."""
    from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals

    good = RunRecord(
        item_id="ok",
        question="q",
        answer="a",
        gold_correct=None,
        anomaly_flag=False,
        baseline_profile="hibrido",
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_low_support=True,
            embedding_max_cosine=0.2,
            judge=JudgeResult(veredito="sustentado", motivo_breve="ok", confianca=0.9),
        ),
        retrieved=[],
        meta={"metricas_lexicas": {"f1_token": 0.8, "em_squad": False}},
    )
    weak = RunRecord(
        item_id="weak",
        question="q",
        answer="a",
        gold_correct=None,
        anomaly_flag=False,
        baseline_profile="hibrido",
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_low_support=False,
            embedding_max_cosine=0.8,
            judge=JudgeResult(veredito="sustentado", motivo_breve="ok", confianca=0.9),
        ),
        retrieved=[],
        meta={"metricas_lexicas": {"f1_token": 0.1, "em_squad": False}},
    )
    records = [good, weak]
    report = compare_aggregation_policies(
        records,
        verify_gold=False,
        verify_embedding=True,
        verify_judge=True,
        negative_judge_verdicts=["nao_sustentado", "contradicacao", "inseguro"],
        reference_type="lexical",
    )
    assert report["tipo_referencia"] == "lexical"
    assert report["n_referencia_aceitavel"] == 1
    assert report["rotulo_referencia_aceitavel"] == "overlap_lexico_aceitavel"

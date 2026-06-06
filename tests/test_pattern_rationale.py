"""Rationale em pattern_detection."""

from __future__ import annotations

from llm_evaluation.pattern_detection import compute_diagnostico
from llm_evaluation.types import EvalItem, VerificationSignals


def test_diagnostico_has_rationale() -> None:
    item = EvalItem(id="1", question="q", correct_answers=["a"], incorrect_answers=[])
    meta = {"metricas_lexicas": {"f1_token": 0.05, "em_squad": False}}
    d = compute_diagnostico(
        item=item,
        answer="wrong",
        signals=VerificationSignals(
            gold_correct=False,
            gold_incorrect=True,
            is_refusal=False,
            embedding_max_cosine=0.8,
            embedding_low_support=False,
        ),
        meta=meta,
        anomaly_flag=False,
    )
    assert "rationale" in d
    assert isinstance(d["rationale"], list)

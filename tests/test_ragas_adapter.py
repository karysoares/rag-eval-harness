"""RAGAS adapter sem dependência obrigatória."""

from __future__ import annotations

from llm_evaluation.benchmarks.ragas_adapter import run_ragas_sample
from llm_evaluation.types import RunRecord, VerificationSignals


def test_ragas_not_installed() -> None:
    rec = RunRecord(
        item_id="1",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.5,
            embedding_low_support=False,
        ),
        retrieved=[],
        baseline_profile="h",
        meta={},
    )
    out = run_ragas_sample([rec])
    assert out.get("disponivel") is False or "n" in out

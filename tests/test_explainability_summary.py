"""sumarize_explicabilidade."""

from __future__ import annotations

from llm_evaluation.explainability import summarize_explicabilidade
from llm_evaluation.types import RunRecord, VerificationSignals


def test_summarize_explicabilidade() -> None:
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
        meta={"explicacao": {"conflitos": ["x"]}},
    )
    s = summarize_explicabilidade([rec])
    assert s is not None
    assert s["n_com_explicacao"] == 1

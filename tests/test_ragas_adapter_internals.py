"""Cobertura interna do adaptador RAGAS (sem dependência do pacote ragas)."""

from __future__ import annotations

from typing import Any

from llm_evaluation.benchmarks.ragas_adapter import (
    _records_to_ragas_rows,
    _scores_from_result,
    summarize_harness_grounding,
)
from llm_evaluation.types import (
    JudgeResult,
    RetrievedChunk,
    RunRecord,
    VerificationSignals,
)


def _signals(
    *,
    low_support: bool | None = False,
    judge: JudgeResult | None = None,
) -> VerificationSignals:
    return VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.8,
        embedding_low_support=low_support,
        judge=judge,
    )


def _record(
    *,
    question: str = "q",
    answer: str = "a",
    retrieved: list[RetrievedChunk] | None = None,
    signals: VerificationSignals | None = None,
    meta: dict[str, Any] | None = None,
) -> RunRecord:
    return RunRecord(
        item_id="1",
        question=question,
        answer=answer,
        gold_correct=True,
        anomaly_flag=False,
        signals=signals or _signals(),
        retrieved=retrieved or [],
        baseline_profile="h",
        meta=meta or {},
    )


class TestRecordsToRagasRows:
    def test_truncates_question_answer_and_contexts(self) -> None:
        rec = _record(
            question="q" * 5000,
            answer="a" * 5000,
            retrieved=[RetrievedChunk(text="c" * 1000, score=0.9)],
        )
        rows = _records_to_ragas_rows([rec])
        assert len(rows[0]["question"]) == 2000
        assert len(rows[0]["answer"]) == 2000
        contexts = rows[0]["contexts"]
        assert isinstance(contexts, list)
        assert len(contexts[0]) == 500

    def test_at_most_five_contexts(self) -> None:
        chunks = [RetrievedChunk(text=f"chunk {i}", score=0.5) for i in range(8)]
        rows = _records_to_ragas_rows([_record(retrieved=chunks)])
        assert len(rows[0]["contexts"]) == 5

    def test_blank_contexts_fall_back_to_empty_string(self) -> None:
        chunks = [RetrievedChunk(text="   ", score=0.1)]
        rows = _records_to_ragas_rows([_record(retrieved=chunks)])
        assert rows[0]["contexts"] == [""]


class TestScoresFromResult:
    def test_plain_dict_with_lists_is_averaged(self) -> None:
        out = _scores_from_result({"faithfulness": [1.0, 0.0, 0.5]})
        assert out["faithfulness"] == 0.5
        assert out["faithfulness_por_item"] == [1.0, 0.0, 0.5]

    def test_plain_dict_scalar_passthrough(self) -> None:
        out = _scores_from_result({"n": 3})
        assert out["n"] == 3

    def test_object_with_scores_dict(self) -> None:
        class FakeResult:
            scores = {"answer_relevancy": [0.2, 0.4]}

        out = _scores_from_result(FakeResult())
        assert out["answer_relevancy"] == pytest_approx(0.3)

    def test_object_with_to_pandas(self) -> None:
        import pandas as pd

        class FakeResult:
            def to_pandas(self) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "question": ["q1", "q2"],
                        "faithfulness": [1.0, 0.0],
                        "label": ["x", "y"],  # não numérico: ignorado
                    },
                )

        out = _scores_from_result(FakeResult())
        assert out["faithfulness"] == 0.5
        assert "question" not in out
        assert "label" not in out

    def test_unknown_object_falls_back_to_raw(self) -> None:
        out = _scores_from_result(object())
        assert "raw" in out


def pytest_approx(x: float) -> Any:
    import pytest

    return pytest.approx(x)


class TestSummarizeHarnessGrounding:
    def test_empty_records(self) -> None:
        assert summarize_harness_grounding([]) == {"n": 0}

    def test_rates_and_f1(self) -> None:
        judge_ok = JudgeResult(veredito="sustentado", motivo_breve="ok", confianca=0.9)
        recs = [
            _record(
                signals=_signals(low_support=True, judge=judge_ok),
                meta={"metricas_lexicas": {"f1_token": 0.8}},
            ),
            _record(
                signals=_signals(low_support=False),
                meta={"lexical_metrics": {"f1_token": 0.4}},
            ),
        ]
        out = summarize_harness_grounding(recs)
        assert out["n"] == 2
        assert out["taxa_embedding_baixo"] == 0.5
        assert out["taxa_juiz_sustentado"] == 0.5
        assert out["media_f1_token"] == pytest_approx(0.6)

    def test_missing_f1_yields_none(self) -> None:
        out = summarize_harness_grounding([_record()])
        assert out["media_f1_token"] is None

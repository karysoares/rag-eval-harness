from llm_evaluation.reference_metrics import referencia_incorreta
from llm_evaluation.types import RunRecord, VerificationSignals


def _record(*, gold_correct: bool | None, meta: dict | None = None) -> RunRecord:
    return RunRecord(
        item_id="x",
        question="q",
        answer="a",
        gold_correct=gold_correct,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=gold_correct,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta or {},
    )


def test_lexical_without_valid_metric_does_not_fall_back_to_gold() -> None:
    rec = _record(gold_correct=False, meta={"metricas_lexicas": {"note": "desligadas"}})
    assert referencia_incorreta(rec, "lexical") is None


def test_none_reference_type_has_no_reference_label() -> None:
    rec = _record(gold_correct=False)
    assert referencia_incorreta(rec, "none") is None


def test_answer_lists_uses_gold_correct() -> None:
    rec = _record(gold_correct=False)
    assert referencia_incorreta(rec, "answer_lists") is True

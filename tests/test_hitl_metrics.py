"""Métricas HITL (plano C)."""

from __future__ import annotations

from llm_evaluation.hitl_metrics import summarize_hitl
from llm_evaluation.types import RunRecord, VerificationSignals


def _rec(iid: str, rotulo: str | None, *, flag: bool = False) -> RunRecord:
    meta: dict = {}
    if rotulo:
        meta["adjudicacao_humana"] = {"rotulo": rotulo, "revisor": "t"}
    return RunRecord(
        item_id=iid,
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=flag,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta,
    )


def test_summarize_hitl_confusao_detector() -> None:
    records = [
        _rec("1", "incorreto", flag=True),
        _rec("2", "correto", flag=False),
        _rec("3", None),
    ]
    out = summarize_hitl(records, fila_total=2)
    assert out is not None
    assert out["n_itens_rotulados"] == 2
    det = out.get("detector_vs_humano")
    assert isinstance(det, dict)
    assert det.get("confusao") is not None

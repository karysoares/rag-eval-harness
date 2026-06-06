"""Recálculo offline de embedding_baixo_suporte para calibração."""

from __future__ import annotations

from llm_evaluation.evaluation_metrics import (
    recompute_embedding_low_support,
    replay_anomaly_flags,
)
from llm_evaluation.types import RunRecord, VerificationSignals


def _rec(emb: float, gc: bool, jneg: bool) -> RunRecord:
    return RunRecord(
        item_id="x",
        question="q",
        answer="a",
        gold_correct=gc,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=gc,
            gold_incorrect=not gc,
            is_refusal=False,
            embedding_max_cosine=emb,
            embedding_low_support=emb < 0.35,
            judge_negative=jneg,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )


def test_recompute_lowers_fp_at_028() -> None:
    recs = [_rec(0.30, True, False), _rec(0.20, True, False)]
    adj = recompute_embedding_low_support(recs, 0.28)
    assert adj[0].signals.embedding_low_support is False
    assert adj[1].signals.embedding_low_support is True


def test_replay_with_embedding_threshold() -> None:
    recs = [_rec(0.30, True, False)]
    flags = replay_anomaly_flags(
        recs,
        verify_gold=False,
        verify_embedding=True,
        verify_judge=True,
        negative_judge_verdicts=["nao_sustentado"],
        policy="embedding_e_juiz",
        embedding_min_cosine=0.28,
    )
    assert flags == [False]

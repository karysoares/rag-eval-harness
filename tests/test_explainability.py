"""Explicabilidade determinística."""

from __future__ import annotations

from llm_evaluation.explainability import build_explicacao
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def test_build_explicacao_alert_signals() -> None:
    rec = RunRecord(
        item_id="1",
        question="q",
        answer="a",
        gold_correct=False,
        anomaly_flag=True,
        signals=VerificationSignals(
            gold_correct=False,
            gold_incorrect=True,
            is_refusal=False,
            embedding_max_cosine=0.2,
            embedding_low_support=True,
            judge=JudgeResult(
                veredito="nao_sustentado",
                motivo_breve="fora do contexto",
                confianca=0.8,
                raw={},
            ),
            judge_negative=True,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={"metricas_recuperacao": {"chunk_ouro_no_top_k": True, "score_melhor_chunk": 0.7}},
    )
    exp = build_explicacao(rec)
    assert exp["alerta"]["flag_anomalia"] is True
    assert "recuperacao" in exp

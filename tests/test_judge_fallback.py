"""Juiz com fallback heurístico não deve disparar agregação."""

from __future__ import annotations

from llm_evaluation.types import JudgeResult, VerificationSignals
from llm_evaluation.verification.aggregate import anomaly_from_signals
from llm_evaluation.verification.judge import run_judge


class _FailingClient:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("api down")


def test_run_judge_sets_fallback_flag() -> None:
    jr, meta = run_judge(
        question="q",
        context="ctx",
        answer="resposta válida com conteúdo suficiente para não cair em incompleto",
        client=_FailingClient(),
        max_parse_retries=0,
    )
    assert jr.raw.get("fallback_heuristico") is True
    assert meta.used_fallback is True
    assert meta.schema_invalid is False
    assert "cadeia_de_pensamento" not in jr.raw


def test_run_judge_retries_then_succeeds() -> None:
    calls: list[int] = []

    class _FlakyClient:
        def complete(self, system: str, user: str, **kwargs: object) -> str:
            calls.append(1)
            if len(calls) == 1:
                return "not json at all"
            return '{"veredito": "sustentado", "motivo_breve": "ok", "confianca": 0.8}'

    jr, meta = run_judge(
        question="q",
        context="ctx",
        answer="resposta longa o suficiente",
        client=_FlakyClient(),
        max_parse_retries=2,
    )
    assert jr.veredito == "sustentado"
    assert meta.retry_count >= 1
    assert len(calls) >= 2


def test_fallback_judge_ignored_in_embedding_e_juiz() -> None:
    j = JudgeResult(
        veredito="contradicacao",
        motivo_breve="fallback",
        confianca=0.4,
        raw={"fallback_heuristico": True},
    )
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=j,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=True,
            negative_judge_verdicts=["contradicacao"],
            policy="embedding_e_juiz",
        )
        is False
    )

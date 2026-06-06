from llm_evaluation.evaluation_metrics import replay_anomaly_flags
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals
from llm_evaluation.verification.aggregate import anomaly_from_signals


def test_anomaly_any_critical_gold() -> None:
    s = VerificationSignals(
        gold_correct=False,
        gold_incorrect=True,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=True,
            verify_embedding=False,
            verify_judge=False,
            negative_judge_verdicts=["contradicacao"],
        )
        is True
    )


def test_anomaly_judge_negative() -> None:
    j = JudgeResult(veredito="contradicacao", motivo_breve="x", confianca=0.9)
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=j,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=True,
            negative_judge_verdicts=["contradicacao"],
        )
        is True
    )


def test_embedding_e_juiz_requires_both() -> None:
    j_neg = JudgeResult(veredito="contradicacao", motivo_breve="x", confianca=0.9)
    s_emb_only = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=None,
        judge_negative=None,
    )
    assert (
        anomaly_from_signals(
            s_emb_only,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=True,
            negative_judge_verdicts=["contradicacao"],
            policy="embedding_e_juiz",
        )
        is False
    )
    s_both = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=j_neg,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s_both,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=True,
            negative_judge_verdicts=["contradicacao"],
            policy="embedding_e_juiz",
        )
        is True
    )


def test_todos_criticos_and() -> None:
    s = VerificationSignals(
        gold_correct=False,
        gold_incorrect=True,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=True,
            verify_embedding=True,
            verify_judge=False,
            negative_judge_verdicts=["contradicacao"],
            policy="todos_criticos",
        )
        is False
    )
    s_both = VerificationSignals(
        gold_correct=False,
        gold_incorrect=True,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=None,
        judge_negative=None,
    )
    assert (
        anomaly_from_signals(
            s_both,
            verify_gold=True,
            verify_embedding=True,
            verify_judge=False,
            negative_judge_verdicts=["contradicacao"],
            policy="todos_criticos",
        )
        is True
    )


def test_judge_fallback_ignored_even_if_verdict_negative() -> None:
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
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=j,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=True,
            negative_judge_verdicts=["contradicacao"],
            policy="qualquer_critico",
        )
        is False
    )


def test_incompleto_excluded_from_aggregation_by_default() -> None:
    j = JudgeResult(veredito="incompleto", motivo_breve="vago", confianca=0.7)
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=j,
        judge_negative=False,
    )
    hard = ["nao_sustentado", "contradicacao", "inseguro"]
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=True,
            verify_judge=True,
            negative_judge_verdicts=[*hard, "incompleto"],
            judge_aggregation_verdicts=hard,
            policy="embedding_e_juiz",
        )
        is False
    )


def test_judge_negative_flag_does_not_override_verdict_for_aggregation() -> None:
    """Replay/agregação usa veredito real, não juiz_negativo persistido."""
    j = JudgeResult(veredito="incompleto", motivo_breve="forçado", confianca=0.7)
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=j,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=True,
            negative_judge_verdicts=["nao_sustentado", "contradicacao", "inseguro"],
            policy="qualquer_critico",
        )
        is False
    )


def test_judge_negative_flag_used_when_judge_missing() -> None:
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=True,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=True,
            negative_judge_verdicts=["nao_sustentado", "contradicacao", "inseguro"],
            policy="qualquer_critico",
        )
        is True
    )


def test_replay_uses_judge_verdict_policy_not_persisted_flag() -> None:
    j = JudgeResult(veredito="incompleto", motivo_breve="vago", confianca=0.7)
    s = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=j,
        judge_negative=True,
    )
    record = RunRecord(
        item_id="i1",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=True,
        signals=s,
        retrieved=[],
        baseline_profile="hibrido",
    )
    flags = replay_anomaly_flags(
        [record],
        verify_gold=False,
        verify_embedding=False,
        verify_judge=True,
        negative_judge_verdicts=["nao_sustentado", "contradicacao", "inseguro", "incompleto"],
        judge_aggregation_verdicts=["nao_sustentado", "contradicacao", "inseguro"],
        policy="qualquer_critico",
    )
    assert flags == [False]


def test_no_anomaly_when_baselines_off() -> None:
    s = VerificationSignals(
        gold_correct=False,
        gold_incorrect=True,
        is_refusal=False,
        embedding_max_cosine=0.1,
        embedding_low_support=True,
        judge=None,
        judge_negative=None,
    )
    assert (
        anomaly_from_signals(
            s,
            verify_gold=False,
            verify_embedding=False,
            verify_judge=False,
            negative_judge_verdicts=["contradicacao"],
        )
        is False
    )

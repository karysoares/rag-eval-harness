from llm_evaluation.reporting import summarize
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def _sig(gc: bool | None, anom: bool) -> RunRecord:
    return RunRecord(
        item_id="x",
        question="q",
        answer="a",
        gold_correct=gc,
        anomaly_flag=anom,
        signals=VerificationSignals(
            gold_correct=gc,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=None,
            judge_negative=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )


def _sig_fp(*, emb_low: bool, judge_neg: bool, curated: bool) -> RunRecord:
    v = "nao_sustentado" if judge_neg else "sustentado"
    judge = JudgeResult(veredito=v, motivo_breve="m", confianca=0.5)
    meta: dict = {}
    if curated:
        meta["qualidade_geracao"] = {"curada_por_recuperacao_fraca": True}
    return RunRecord(
        item_id="fp",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=True,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.1 if emb_low else 0.9,
            embedding_low_support=emb_low,
            judge=judge,
            judge_negative=judge_neg,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta,
    )


def test_summarize_warns_judge_same_as_generator() -> None:
    proto = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": ["nao_sustentado"],
        "models": {
            "llm_model": "gpt-4o-mini",
            "judge_model": "gpt-4o-mini",
            "judge_same_as_generator": True,
        },
    }
    s = summarize([_sig(True, False)], reference_type="lexical", protocol=proto)
    avisos = s.get("avisos_protocolo")
    assert isinstance(avisos, list)
    assert any("JUDGE_MODEL" in a for a in avisos)


def test_summarize_operacional_com_protocolo() -> None:
    proto = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": ["nao_sustentado"],
    }
    s = summarize([_sig(True, False)], reference_type="lexical", protocol=proto)
    op = s.get("sumario_operacional")
    assert isinstance(op, dict)
    assert "fila_revisao_humana" in op
    assert "nota_interpretacao" in op


def test_summarize_confusion_matrix() -> None:
    recs = [
        _sig(False, True),
        _sig(False, False),
        _sig(True, True),
        _sig(True, False),
    ]
    s = summarize(recs)
    cg = s["confusao_vs_gold"]
    assert cg["vp_gold_incorreto_marcado"] == 1
    assert cg["fn_gold_incorreto_nao_marcado"] == 1
    assert cg["fp_gold_correto_mas_marcado"] == 1
    assert cg["vn_gold_correto_nao_marcado"] == 1
    qp = s["qualidade_pipeline"]
    assert qp["n_geracoes_curadas_recuperacao_fraca"] == 0


def test_summarize_estratificacao_fp_gold_correto() -> None:
    recs = [
        _sig_fp(emb_low=True, judge_neg=False, curated=False),
        _sig_fp(emb_low=False, judge_neg=True, curated=False),
        _sig_fp(emb_low=True, judge_neg=True, curated=True),
    ]
    s = summarize(recs)
    est = s["estratificacao_fp_gold_correto"]
    assert est["n_fp_gold_correto"] == 3
    assert est["com_so_embedding_baixo"] == 1
    assert est["com_so_juiz_negativo"] == 1
    assert est["com_embedding_e_juiz"] == 1
    assert est["dos_quais_resposta_curada_por_gate_recuperacao"] == 1
    assert s["qualidade_pipeline"]["n_geracoes_curadas_recuperacao_fraca"] == 1

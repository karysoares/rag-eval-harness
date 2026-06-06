import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import (
    compare_metric_reports,
    layer_analysis,
    load_full_report,
    prediction_row_to_run_record,
)
from llm_evaluation.reporting import record_to_json, summarize
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def _rec(
    *,
    gc: bool,
    anom: bool,
    g_inc: bool | None,
    emb_low: bool | None,
    has_judge: bool,
    judge_negative: bool,
) -> RunRecord:
    judge: JudgeResult | None = None
    jn: bool | None = None
    if has_judge:
        judge = JudgeResult(
            veredito="nao_sustentado" if judge_negative else "sustentado",
            motivo_breve="t",
            confianca=0.5,
        )
        jn = judge_negative
    return RunRecord(
        item_id="i",
        question="q",
        answer="a",
        gold_correct=gc,
        anomaly_flag=anom,
        signals=VerificationSignals(
            gold_correct=gc,
            gold_incorrect=g_inc,
            is_refusal=False,
            embedding_max_cosine=0.1 if emb_low else 0.9,
            embedding_low_support=emb_low,
            judge=judge,
            judge_negative=jn,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )


def test_layer_analysis_marginals() -> None:
    recs = [
        _rec(gc=False, anom=True, g_inc=True, emb_low=False, has_judge=True, judge_negative=False),
        _rec(gc=True, anom=True, g_inc=False, emb_low=True, has_judge=True, judge_negative=False),
        _rec(gc=True, anom=False, g_inc=False, emb_low=False, has_judge=True, judge_negative=False),
    ]
    la = layer_analysis(recs)
    assert la["gatilhos_marginais"]["n_sinal_ouro_incorreto"] == 1
    assert la["gatilhos_marginais"]["n_embedding_baixo_suporte"] == 1
    assert la["combinacoes_exclusivas_todos_itens"]["ouro_apenas"] == 1
    assert la["combinacoes_exclusivas_todos_itens"]["so_embedding"] == 1


def test_summarize_includes_layer_analysis() -> None:
    recs = [
        _rec(
            gc=False,
            anom=True,
            g_inc=True,
            emb_low=False,
            has_judge=True,
            judge_negative=False,
        ),
    ]
    s = summarize(recs)
    assert "analise_camadas" in s
    assert s["analise_camadas"]["n_itens"] == 1
    # Métricas com IC de Wilson presentes (mesmo que None quando amostra trivial)
    assert "ic95_precisao_anomalia_vs_gold_incorreto" in s
    assert "ic95_revocacao_anomalia_vs_gold_incorreto" in s
    assert "cohen_kappa_anomalia_vs_gold" in s


def test_layer_analysis_incompleto_excluded_from_aggregation_marginal() -> None:
    judge = JudgeResult(veredito="incompleto", motivo_breve="vago", confianca=0.7)
    rec = RunRecord(
        item_id="inc",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.1,
            embedding_low_support=True,
            judge=judge,
            judge_negative=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
    )
    hard = ["nao_sustentado", "contradicacao", "inseguro"]
    la = layer_analysis(
        [rec],
        judge_aggregation_verdicts=hard,
        negative_judge_verdicts=[*hard, "incompleto"],
    )
    assert la["gatilhos_marginais"]["n_juiz_negativo"] == 0
    assert la["gatilhos_marginais"]["n_juiz_diagnostico_negativo"] == 1


def test_layer_analysis_has_kappa_and_ci() -> None:
    recs = [
        _rec(gc=False, anom=True, g_inc=True, emb_low=True, has_judge=True, judge_negative=True),
        _rec(gc=True, anom=False, g_inc=False, emb_low=False, has_judge=True, judge_negative=False),
        _rec(gc=False, anom=True, g_inc=True, emb_low=False, has_judge=True, judge_negative=False),
        _rec(gc=True, anom=False, g_inc=False, emb_low=False, has_judge=True, judge_negative=False),
    ]
    la = layer_analysis(recs)
    pl = la["por_camada_vs_referencia"]
    assert "cohen_kappa_vs_gold" in pl["sinal_ouro"]
    assert "ic95_precisao" in pl["sinal_ouro"]
    assert "concordancia_entre_camadas" in la
    pares = la["concordancia_entre_camadas"]
    assert isinstance(pares, list)
    assert any(p["par"] == "sinal_ouro__vs__juiz" for p in pares)


def test_prediction_row_roundtrip() -> None:
    r = _rec(gc=True, anom=True, g_inc=False, emb_low=True, has_judge=True, judge_negative=True)
    d = record_to_json(r)
    r2 = prediction_row_to_run_record(d)
    assert r2.gold_correct is r.gold_correct
    assert r2.anomaly_flag == r.anomaly_flag
    assert r2.signals.embedding_low_support is True
    assert r2.signals.judge is not None
    assert r2.signals.judge.veredito == "nao_sustentado"


def test_compare_metric_reports() -> None:
    a = summarize(
        [
            _rec(
                gc=False,
                anom=True,
                g_inc=True,
                emb_low=False,
                has_judge=True,
                judge_negative=False,
            ),
        ],
    )
    b = summarize(
        [
            _rec(
                gc=True,
                anom=False,
                g_inc=False,
                emb_low=False,
                has_judge=True,
                judge_negative=False,
            ),
        ],
    )
    c = compare_metric_reports([a, b], ["run_a", "run_b"])
    assert len(c["corridas"]) == 2
    assert c["corridas"][0]["rotulo"] == "run_a"


def test_load_full_report_from_summary_only(tmp_path: Path) -> None:
    d = tmp_path / "run_x"
    d.mkdir()
    summary = {"n_items": 1, "confusion_vs_gold": {}}
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    got = load_full_report(d)
    assert got["n_items"] == 1

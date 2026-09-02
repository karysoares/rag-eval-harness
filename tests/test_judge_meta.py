"""Meta-avaliação do juiz: calibração, concordância, viés e auto-consistência."""

from __future__ import annotations

from typing import Any

from llm_evaluation.judge_meta import (
    build_judge_meta_report,
    judge_agreement,
    judge_calibration,
    judge_position_bias,
    judge_verbosity_bias,
    self_consistency,
)
from llm_evaluation.statistics import (
    expected_calibration_error,
    fleiss_kappa,
    point_biserial,
)
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


def _record(
    item_id: str,
    *,
    veredito: str = "sustentado",
    confianca: float = 0.9,
    gold_correct: bool | None = None,
    answer: str = "Uma resposta.",
    rank_ouro: int | None = 1,
    fallback: bool = False,
    veredito_ausente: bool = False,
) -> RunRecord:
    raw: dict[str, Any] = {"veredito": veredito}
    if fallback:
        raw["fallback_heuristico"] = True
    judge = (
        None
        if veredito_ausente
        else JudgeResult(
            veredito=veredito,  # type: ignore[arg-type]
            motivo_breve="m",
            confianca=confianca,
            raw=raw,
        )
    )
    return RunRecord(
        item_id=item_id,
        question="q?",
        answer=answer,
        gold_correct=gold_correct,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=gold_correct,
            gold_incorrect=None if gold_correct is None else not gold_correct,
            is_refusal=False,
            embedding_max_cosine=0.5,
            embedding_low_support=False,
            judge=judge,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={"metricas_recuperacao": {"rank_chunk_ouro": rank_ouro}},
    )


# --- primitivos estatísticos -------------------------------------------------


def test_ece_zero_para_juiz_perfeitamente_calibrado() -> None:
    # Confiança 1.0 e sempre certo; 0.0 e sempre errado.
    pares = [(1.0, True)] * 10 + [(0.0, False)] * 10
    out = expected_calibration_error(pares, n_bins=10)
    assert out is not None
    assert out["ece"] == 0.0
    assert out["exatidao_global"] == 0.5


def test_ece_maximo_para_juiz_totalmente_sobreconfiante() -> None:
    out = expected_calibration_error([(1.0, False)] * 20, n_bins=10)
    assert out is not None
    assert out["ece"] == 1.0
    assert out["mce"] == 1.0


def test_ece_ignora_confiancas_fora_do_intervalo() -> None:
    out = expected_calibration_error([(1.5, True), (0.5, True), (-0.2, False)], n_bins=5)
    assert out is not None
    assert out["n"] == 1


def test_ece_devolve_none_sem_pares_utilizaveis() -> None:
    assert expected_calibration_error([]) is None
    assert expected_calibration_error([(0.5, True)], n_bins=0) is None


def test_ece_inclui_confianca_um_no_ultimo_bin() -> None:
    out = expected_calibration_error([(1.0, True)], n_bins=4)
    assert out is not None
    assert out["n_bins_ocupados"] == 1
    assert out["bins"][0]["intervalo"] == [0.75, 1.0]


def test_fleiss_kappa_unanime_e_um() -> None:
    # 3 itens, 4 avaliações cada, todas na mesma categoria dentro do item,
    # mas categorias diferentes entre itens (senão p_exp = 1).
    contagens = [[4, 0], [0, 4], [4, 0]]
    kappa = fleiss_kappa(contagens)
    assert kappa is not None
    assert abs(kappa - 1.0) < 1e-9


def test_fleiss_kappa_rejeita_desenhos_degenerados() -> None:
    assert fleiss_kappa([[2, 0]]) is None  # < 2 itens
    assert fleiss_kappa([[1, 0], [1, 0]]) is None  # < 2 avaliações
    assert fleiss_kappa([[2, 0], [1, 0]]) is None  # nº de avaliações desigual
    assert fleiss_kappa([[2, 0], [2, 0]]) is None  # p_exp == 1


def test_point_biserial_deteta_separacao_perfeita() -> None:
    r = point_biserial([True, True, False, False], [10.0, 10.0, 1.0, 1.0])
    assert r is not None
    assert r > 0.9


def test_point_biserial_none_sem_variancia_ou_grupo_vazio() -> None:
    assert point_biserial([True, False, True], [5.0, 5.0, 5.0]) is None
    assert point_biserial([True, True, True], [1.0, 2.0, 3.0]) is None
    assert point_biserial([True, False], [1.0, 2.0]) is None


# --- calibração e concordância ----------------------------------------------


def test_calibracao_usa_apenas_itens_com_referencia() -> None:
    records = [
        _record("a", gold_correct=True),
        _record("b", gold_correct=None),  # sem referência -> excluído
    ]
    out = judge_calibration(records, "answer_lists")
    assert out is not None
    assert out["n"] == 1


def test_calibracao_ignora_fallback_heuristico() -> None:
    records = [_record("a", gold_correct=True, fallback=True)]
    assert judge_calibration(records, "answer_lists") is None


def test_calibracao_none_sem_tipo_de_referencia() -> None:
    assert judge_calibration([_record("a", gold_correct=True)], "none") is None


def test_concordancia_monta_matriz_correta() -> None:
    records = [
        _record("a", veredito="sustentado", gold_correct=True),  # VP
        _record("b", veredito="nao_sustentado", gold_correct=True),  # FN
        _record("c", veredito="sustentado", gold_correct=False),  # FP
        _record("d", veredito="nao_sustentado", gold_correct=False),  # VN
    ]
    out = judge_agreement(records, "answer_lists")
    assert out is not None
    assert out["confusao"] == {
        "juiz_aprovou_referencia_ok": 1,
        "juiz_reprovou_referencia_ok": 1,
        "juiz_aprovou_referencia_problematica": 1,
        "juiz_reprovou_referencia_problematica": 1,
    }
    assert out["exatidao"] == 0.5
    assert out["exatidao_ic95_wilson"] is not None


def test_concordancia_none_sem_itens_rotulados() -> None:
    assert judge_agreement([_record("a", gold_correct=None)], "answer_lists") is None


def test_referencia_humana_tem_precedencia_sobre_a_automatica() -> None:
    record = _record("a", veredito="sustentado", gold_correct=False)
    record.meta["adjudicacao_humana"] = {"rotulo": "correto"}
    out = judge_agreement([record], "answer_lists")
    assert out is not None
    # Humano diz correto e o juiz aprovou -> verdadeiro positivo, não falso.
    assert out["confusao"]["juiz_aprovou_referencia_ok"] == 1


# --- sondas de viés ----------------------------------------------------------


def test_vies_verbosidade_correlaciona_comprimento_com_aprovacao() -> None:
    records = [
        _record("a", veredito="sustentado", answer="x" * 200),
        _record("b", veredito="sustentado", answer="x" * 190),
        _record("c", veredito="nao_sustentado", answer="x" * 10),
        _record("d", veredito="nao_sustentado", answer="x" * 12),
    ]
    out = judge_verbosity_bias(records)
    assert out is not None
    assert out["correlacao_ponto_bisserial"] is not None
    assert out["correlacao_ponto_bisserial"] > 0.9
    assert out["media_caracteres_aprovados"] > out["media_caracteres_reprovados"]


def test_vies_verbosidade_none_com_poucos_itens() -> None:
    assert judge_verbosity_bias([_record("a")]) is None


def test_vies_posicao_agrupa_por_rank_do_chunk_ouro() -> None:
    records = [
        _record("a", veredito="sustentado", rank_ouro=1),
        _record("b", veredito="sustentado", rank_ouro=1),
        _record("c", veredito="nao_sustentado", rank_ouro=3),
        _record("d", veredito="nao_sustentado", rank_ouro=None),
    ]
    out = judge_position_bias(records)
    assert out is not None
    por_rank = out["por_rank_chunk_ouro"]
    assert por_rank["1"]["taxa_aprovacao"] == 1.0
    assert por_rank["3"]["taxa_aprovacao"] == 0.0
    assert por_rank["ausente"]["n"] == 1


def test_vies_posicao_none_sem_metricas_de_recuperacao() -> None:
    record = _record("a")
    record.meta.pop("metricas_recuperacao")
    assert judge_position_bias([record]) is None


# --- auto-consistência -------------------------------------------------------


def test_autoconsistencia_unanime() -> None:
    out = self_consistency([["sustentado"] * 5, ["nao_sustentado"] * 5])
    assert out is not None
    assert out["taxa_itens_unanimes"] == 1.0
    assert out["media_taxa_veredito_modal"] == 1.0
    assert out["fleiss_kappa"] == 1.0


def test_autoconsistencia_deteta_juiz_instavel() -> None:
    out = self_consistency(
        [
            ["sustentado", "nao_sustentado", "sustentado", "nao_sustentado"],
            ["sustentado", "nao_sustentado", "nao_sustentado", "sustentado"],
        ]
    )
    assert out is not None
    assert out["taxa_itens_unanimes"] == 0.0
    assert out["media_taxa_veredito_modal"] == 0.5


def test_autoconsistencia_rejeita_amostras_desiguais_ou_vazias() -> None:
    assert self_consistency([]) is None
    assert self_consistency([["a"]]) is None  # < 2 amostras
    assert self_consistency([["a", "a"], ["a", "a", "a"]]) is None


# --- relatório completo ------------------------------------------------------


def test_relatorio_agrega_todas_as_seccoes() -> None:
    records = [
        _record("a", veredito="sustentado", gold_correct=True, answer="x" * 100),
        _record("b", veredito="nao_sustentado", gold_correct=False, answer="x" * 10),
        _record("c", veredito="sustentado", gold_correct=True, answer="x" * 90),
        _record("d", veredito="incompleto", gold_correct=False, answer="x" * 12),
    ]
    report = build_judge_meta_report(records, reference_type="answer_lists")
    assert report["n_itens"] == 4
    assert report["n_itens_com_veredito_real"] == 4
    assert report["distribuicao_vereditos"]["sustentado"] == 2
    assert report["calibracao"] is not None
    assert report["concordancia_com_referencia"] is not None
    assert report["vies_verbosidade"] is not None
    assert report["vies_posicao"] is not None
    assert "autoconsistencia" not in report


def test_relatorio_conta_fallback_separadamente() -> None:
    records = [
        _record("a", gold_correct=True),
        _record("b", gold_correct=True, fallback=True),
        _record("c", veredito_ausente=True),
    ]
    report = build_judge_meta_report(records, reference_type="answer_lists")
    assert report["n_itens"] == 3
    assert report["n_itens_com_veredito_real"] == 1
    assert report["n_itens_com_fallback_heuristico"] == 1


def test_relatorio_sobrevive_a_corrida_sem_juiz() -> None:
    records = [_record("a", veredito_ausente=True)]
    report = build_judge_meta_report(records, reference_type="answer_lists")
    assert report["n_itens_com_veredito_real"] == 0
    assert report["calibracao"] is None
    assert report["concordancia_com_referencia"] is None


def test_relatorio_inclui_autoconsistencia_quando_fornecida() -> None:
    report = build_judge_meta_report(
        [_record("a", gold_correct=True)],
        reference_type="answer_lists",
        amostras_autoconsistencia=[["sustentado", "sustentado"], ["incompleto", "sustentado"]],
    )
    assert report["autoconsistencia"] is not None
    assert report["autoconsistencia"]["n_itens"] == 2

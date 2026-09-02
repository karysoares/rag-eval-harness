"""Testes de McNemar, bootstrap emparelhado e significância entre corridas."""

from __future__ import annotations

import math

from llm_evaluation.evaluation_metrics import (
    compare_metric_reports,
    paired_significance,
    pairwise_paired_significance,
)
from llm_evaluation.statistics import (
    _binomial_two_sided_p,
    _chi2_sf_1df,
    mcnemar_test,
    paired_bootstrap_diff_ci,
)


def test_mcnemar_sem_discordantes_e_indefinido() -> None:
    assert mcnemar_test(0, 0) is None


def test_mcnemar_negativos_invalidos() -> None:
    assert mcnemar_test(-1, 3) is None


def test_mcnemar_exato_simetrico_nao_significativo() -> None:
    res = mcnemar_test(5, 5)
    assert res is not None
    assert res["metodo"] == "mcnemar_exato_binomial"
    assert res["p_valor"] == 1.0


def test_mcnemar_exato_extremo_e_significativo() -> None:
    res = mcnemar_test(10, 0)
    assert res is not None
    assert isinstance(res["p_valor"], float)
    assert res["p_valor"] < 0.01


def test_mcnemar_grande_usa_qui_quadrado() -> None:
    res = mcnemar_test(40, 10)
    assert res is not None
    assert res["metodo"] == "mcnemar_qui_quadrado_correcao_continuidade"
    # (|40-10|-1)^2 / 50 = 29^2/50 = 16.82
    assert isinstance(res["estatistica"], float)
    assert abs(res["estatistica"] - 16.82) < 1e-9
    assert isinstance(res["p_valor"], float)
    assert res["p_valor"] < 0.001


def test_binomial_bilateral_p_de_um_lancamento_justo() -> None:
    # k=0 de n=1: 2 * (1/2) = 1.0
    assert _binomial_two_sided_p(0, 1) == 1.0
    # k=0 de n=10: 2 * (1/1024)
    assert abs(_binomial_two_sided_p(0, 10) - 2 / 1024) < 1e-12


def test_chi2_sf_1gl_bate_valores_criticos_conhecidos() -> None:
    assert abs(_chi2_sf_1df(3.841459) - 0.05) < 1e-4
    assert _chi2_sf_1df(0.0) == 1.0


def test_bootstrap_emparelhado_cobre_diferenca_nula() -> None:
    vals = [True, False] * 20
    res = paired_bootstrap_diff_ci(vals, vals, n_resamples=500)
    assert res is not None
    assert res["diferenca_observada"] == 0.0
    assert res["ic_inferior"] == 0.0
    assert res["ic_superior"] == 0.0


def test_bootstrap_emparelhado_deteta_diferenca_sistematica() -> None:
    a = [True] * 40
    b = [False] * 40
    res = paired_bootstrap_diff_ci(a, b, n_resamples=500)
    assert res is not None
    assert res["diferenca_observada"] == 1.0
    assert res["ic_inferior"] == 1.0


def test_bootstrap_e_deterministico_com_a_mesma_semente() -> None:
    a = [True, False, True, True, False] * 8
    b = [False, False, True, False, True] * 8
    first = paired_bootstrap_diff_ci(a, b, n_resamples=300)
    second = paired_bootstrap_diff_ci(a, b, n_resamples=300)
    assert first == second


def test_bootstrap_rejeita_tamanhos_diferentes_ou_vazio() -> None:
    assert paired_bootstrap_diff_ci([True], [True, False]) is None
    assert paired_bootstrap_diff_ci([], []) is None


def test_significancia_emparelhada_conta_discordantes_por_item() -> None:
    a = {"i1": True, "i2": True, "i3": False, "i4": False}
    b = {"i1": True, "i2": False, "i3": False, "i4": False}
    res = paired_significance("A", a, "B", b)
    assert res is not None
    assert res["n_itens_comuns"] == 4
    assert res["so_a"] == 1
    assert res["so_b"] == 0


def test_significancia_emparelhada_usa_apenas_itens_comuns() -> None:
    a = {"i1": True, "i2": False, "so_em_a": True}
    b = {"i1": False, "i2": False, "so_em_b": True}
    res = paired_significance("A", a, "B", b)
    assert res is not None
    assert res["n_itens_comuns"] == 2
    assert res["cobertura_a"] == 2 / 3


def test_significancia_emparelhada_sem_sobreposicao_devolve_none() -> None:
    assert paired_significance("A", {"x": True}, "B", {"y": True}) is None


def test_pares_cobrem_todas_as_combinacoes() -> None:
    flags = {
        "a": {"i1": True, "i2": False},
        "b": {"i1": False, "i2": False},
        "c": {"i1": True, "i2": True},
    }
    pares = pairwise_paired_significance(flags)
    assert len(pares) == 3
    assert {tuple(p["par"]) for p in pares} == {("a", "b"), ("a", "c"), ("b", "c")}  # type: ignore[arg-type]


def test_compare_reports_inclui_bloco_emparelhado_quando_ha_flags() -> None:
    reports: list[dict[str, object]] = [
        {"n_itens": 4, "n_anomalias_marcadas": 2},
        {"n_itens": 4, "n_anomalias_marcadas": 1},
    ]
    flags = {
        "run_a": {"i1": True, "i2": True, "i3": False, "i4": False},
        "run_b": {"i1": True, "i2": False, "i3": False, "i4": False},
    }
    out = compare_metric_reports(reports, ["run_a", "run_b"], flags_por_corrida=flags)
    assert out["versao_esquema"] == "2"
    emparelhada = out["significancia_emparelhada"]
    assert isinstance(emparelhada, list)
    assert emparelhada[0]["par"] == ["run_a", "run_b"]
    assert emparelhada[0]["mcnemar"] is not None


def test_compare_reports_omite_bloco_emparelhado_sem_flags() -> None:
    reports: list[dict[str, object]] = [
        {"n_itens": 4, "n_anomalias_marcadas": 2},
        {"n_itens": 4, "n_anomalias_marcadas": 1},
    ]
    out = compare_metric_reports(reports, ["a", "b"])
    assert "significancia_emparelhada" not in out
    assert isinstance(out["significancia"], list)


def test_mcnemar_p_valor_nunca_excede_um() -> None:
    for b in range(0, 12):
        for c in range(0, 12):
            res = mcnemar_test(b, c)
            if res is None:
                continue
            p = res["p_valor"]
            assert isinstance(p, float)
            assert 0.0 <= p <= 1.0
            assert not math.isnan(p)

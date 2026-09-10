"""κ, ECE e comparações múltiplas publicados com a sua incerteza.

O κ e o ECE eram as duas colunas em que a escolha de juiz se argumentava e as únicas
publicadas sem intervalo. E seis comparações emparelhadas de quatro braços eram
avaliadas contra um 0,05 por par, sem correcção de família.
"""

from __future__ import annotations

from llm_evaluation.statistics import (
    bootstrap_ece_ci,
    bootstrap_kappa_ci,
    cohen_kappa,
    holm_bonferroni,
)


def test_ic_do_kappa_contem_a_estimativa_pontual() -> None:
    pares = (
        [(True, True)] * 40 + [(False, False)] * 40 + [(True, False)] * 10 + [(False, True)] * 10
    )
    pontual = cohen_kappa(40, 10, 10, 40)
    ic = bootstrap_kappa_ci(pares)
    assert ic is not None and pontual is not None
    assert ic["ic_inferior"] <= pontual <= ic["ic_superior"]


def test_ic_do_kappa_e_mais_largo_com_menos_itens() -> None:
    """N=200 não resolve diferenças pequenas de κ, e o intervalo tem de o mostrar."""
    grande = [(True, True), (False, False), (True, False), (False, True)] * 50
    pequeno = [(True, True), (False, False), (True, False), (False, True)] * 5
    ic_g = bootstrap_kappa_ci(grande)
    ic_p = bootstrap_kappa_ci(pequeno)
    assert ic_g is not None and ic_p is not None
    largura_g = ic_g["ic_superior"] - ic_g["ic_inferior"]
    largura_p = ic_p["ic_superior"] - ic_p["ic_inferior"]
    assert largura_p > largura_g


def test_reamostragem_degenerada_e_contada_nao_tratada_como_zero() -> None:
    """Avaliador constante: κ indefinido, não κ=0 (que se leria como «não concorda»)."""
    pares = [(True, True)] * 30
    ic = bootstrap_kappa_ci(pares)
    assert ic is None or ic["n_reamostragens_degeneradas"] > 0


def test_ic_do_kappa_e_determinista() -> None:
    pares = [(True, True), (False, False), (True, False)] * 20
    assert bootstrap_kappa_ci(pares) == bootstrap_kappa_ci(pares)


def test_ic_do_ece_contem_a_estimativa_pontual() -> None:
    from llm_evaluation.statistics import expected_calibration_error

    pares = [(0.95, True)] * 60 + [(0.95, False)] * 40
    pontual = expected_calibration_error(pares)
    ic = bootstrap_ece_ci(pares)
    assert ic is not None and pontual is not None
    assert ic["ic_inferior"] <= float(pontual["ece"]) <= ic["ic_superior"]


def test_ic_do_ece_sem_pares_devolve_nada() -> None:
    assert bootstrap_ece_ci([]) is None


def test_holm_e_monotono_e_limitado_a_um() -> None:
    ajustados = holm_bonferroni([0.01, 0.02, 0.03, 0.04, 0.05, 0.9])
    assert all(0.0 <= p <= 1.0 for p in ajustados)
    assert ajustados == sorted(ajustados)


def test_holm_preserva_a_ordem_de_entrada() -> None:
    ajustados = holm_bonferroni([0.9, 0.001])
    assert ajustados[0] > ajustados[1]


def test_holm_e_menos_severo_que_bonferroni_no_menor_p() -> None:
    """Holm é uniformemente mais potente e não assume independência entre pares."""
    ps = [0.01, 0.02, 0.03]
    holm = holm_bonferroni(ps)
    bonferroni = [min(1.0, len(ps) * p) for p in ps]
    assert holm[2] < bonferroni[2]


def test_holm_com_um_so_p_nao_corrige() -> None:
    assert holm_bonferroni([0.03]) == [0.03]


def test_holm_vazio() -> None:
    assert holm_bonferroni([]) == []


def test_familia_de_seis_comparacoes_recebe_p_ajustado() -> None:
    """Quatro braços = seis pares; o artefacto tem de dizer isso."""
    from llm_evaluation.evaluation_metrics import pairwise_paired_significance

    itens = [f"i{n}" for n in range(60)]
    flags = {
        "a": dict.fromkeys(itens, False),
        "b": {k: (i % 2 == 0) for i, k in enumerate(itens)},
        "c": {k: (i % 3 == 0) for i, k in enumerate(itens)},
        "d": dict.fromkeys(itens, True),
    }
    res = pairwise_paired_significance(flags)
    assert len(res) == 6
    com_ajuste = [r for r in res if "p_valor_ajustado_holm" in r]
    assert com_ajuste, "nenhuma comparação recebeu p ajustado"
    for r in com_ajuste:
        assert r["n_comparacoes_na_familia"] == len(com_ajuste)
        mc = r["mcnemar"]
        assert isinstance(mc, dict)
        assert r["p_valor_ajustado_holm"] >= mc["p_valor"]


class TestEfeitoMinimoDetectavel:
    """Um `p` não significativo não é prova de equivalência.

    O A/B de juízes publica «todos p=1» sobre 200 itens. Sem MDE, isso lê-se como
    «os juízes são equivalentes» quando o que se sabe é «este desenho não distinguiu
    nada» — duas afirmações diferentes, e só uma delas é sustentada.
    """

    def test_mde_diminui_com_mais_pares(self) -> None:
        from llm_evaluation.statistics import mcnemar_mde

        pequeno = mcnemar_mde(50, 10)
        grande = mcnemar_mde(500, 100)
        assert pequeno is not None and grande is not None
        assert grande["efeito_minimo_detectavel"] < pequeno["efeito_minimo_detectavel"]

    def test_zero_discordantes_nao_produz_mde(self) -> None:
        """Sem pares discordantes o desenho não distingue nada, por muito N que tenha."""
        from llm_evaluation.statistics import mcnemar_mde

        r = mcnemar_mde(1000, 0)
        assert r is not None
        assert r["efeito_minimo_detectavel"] is None
        assert "equivalencia" in r["nota"]

    def test_entradas_incoerentes_devolvem_nada(self) -> None:
        from llm_evaluation.statistics import mcnemar_mde

        assert mcnemar_mde(0, 0) is None
        assert mcnemar_mde(10, 11) is None
        assert mcnemar_mde(-1, 0) is None

    def test_comparacao_emparelhada_publica_o_mde(self) -> None:
        from llm_evaluation.evaluation_metrics import paired_significance

        itens = [f"i{n}" for n in range(100)]
        a = dict.fromkeys(itens, False)
        b = {k: (i < 10) for i, k in enumerate(itens)}
        res = paired_significance("a", a, "b", b)
        assert res is not None
        mde = res["efeito_minimo_detectavel"]
        assert isinstance(mde, dict)
        assert mde["n_pares"] == 100
        assert mde["n_discordantes"] == 10
        assert 0.0 < mde["efeito_minimo_detectavel"] < 1.0

    def test_quantil_normal_bate_valores_conhecidos(self) -> None:
        from llm_evaluation.statistics import _z_quantil

        assert abs(_z_quantil(0.975) - 1.959964) < 1e-4
        assert abs(_z_quantil(0.80) - 0.841621) < 1e-4
        assert abs(_z_quantil(0.5)) < 1e-6

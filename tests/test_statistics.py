"""Testes das funções estatísticas usadas nos relatórios."""

from __future__ import annotations

import math

from llm_evaluation.statistics import cohen_kappa, wilson_ci


def test_wilson_ci_basic_bounds() -> None:
    lo, hi = wilson_ci(50, 100) or (0.0, 0.0)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # IC contém a proporção observada
    assert lo <= 0.5 <= hi


def test_wilson_ci_extremes() -> None:
    lo0, hi0 = wilson_ci(0, 10) or (None, None)  # type: ignore[misc]
    assert lo0 == 0.0
    assert hi0 is not None and hi0 > 0
    lo1, hi1 = wilson_ci(10, 10) or (None, None)  # type: ignore[misc]
    assert hi1 == 1.0
    assert lo1 is not None and lo1 < 1


def test_wilson_ci_invalid() -> None:
    assert wilson_ci(0, 0) is None
    assert wilson_ci(-1, 5) is None
    assert wilson_ci(6, 5) is None


def test_cohen_kappa_perfect() -> None:
    # 10 verdadeiros positivos, 10 verdadeiros negativos: concordância perfeita
    k = cohen_kappa(tp=10, fn=0, fp=0, tn=10)
    assert k is not None
    assert math.isclose(k, 1.0)


def test_cohen_kappa_chance() -> None:
    # 50/50 em cada eixo, concordância exatamente ao acaso => kappa ~0
    k = cohen_kappa(tp=25, fn=25, fp=25, tn=25)
    assert k is not None
    assert abs(k) < 1e-9


def test_cohen_kappa_disagreement() -> None:
    k = cohen_kappa(tp=2, fn=8, fp=8, tn=2)
    assert k is not None
    assert k < 0  # discordância sistemática


def test_cohen_kappa_empty() -> None:
    assert cohen_kappa(0, 0, 0, 0) is None

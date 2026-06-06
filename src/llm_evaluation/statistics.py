"""Estatística simples sem dependências externas: intervalo de Wilson e Cohen's kappa.

Usadas para reportar incerteza e concordância entre camadas de verificação no `summary.json`.
"""

from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float] | None:
    """Intervalo de Wilson para uma proporção (binária) com aproximação assimétrica robusta.

    Retorna ``None`` quando ``n == 0`` ou ``successes < 0``. Útil para precisão / revocação /
    falso alarme em amostras pequenas, onde o IC normal é instável.
    """
    if n <= 0 or successes < 0 or successes > n:
        return None
    p = successes / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def cohen_kappa(tp: int, fn: int, fp: int, tn: int) -> float | None:
    """Cohen's kappa para uma matriz de confusão 2×2 (predição vs referência).

    Mede concordância para além do acaso. Convenção comum:
    - kappa > 0.6: concordância substancial.
    - kappa entre 0.2 e 0.6: moderada.
    - kappa < 0.2: fraca.

    Retorna ``None`` quando o conjunto é vazio ou a concordância esperada é trivial (1.0).
    """
    n = tp + fn + fp + tn
    if n <= 0:
        return None
    p_obs = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((fn + tn) / n) * ((fp + tn) / n)
    p_exp = p_yes + p_no
    if p_exp >= 1.0 - 1e-12:
        return None
    return (p_obs - p_exp) / (1.0 - p_exp)

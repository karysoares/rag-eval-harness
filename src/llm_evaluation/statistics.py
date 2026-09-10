"""Estatística sem dependências externas: Wilson, Cohen's kappa, McNemar e bootstrap.

Usadas para reportar incerteza, concordância entre camadas de verificação e
significância de diferenças entre corridas no `summary.json` / `run_comparison.json`.
"""

from __future__ import annotations

import math
import random
from typing import Any


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


def mcnemar_test(b: int, c: int, *, exact_max_n: int = 25) -> dict[str, float | int | str] | None:
    """Teste de McNemar para desenhos **emparelhados** (mesmos itens, dois sistemas).

    ``b`` e ``c`` são os pares discordantes: ``b`` = casos marcados por A e não por B;
    ``c`` = marcados por B e não por A. Pares concordantes não carregam informação
    sobre a diferença e por isso não entram no teste.

    Com poucos discordantes (``b + c <= exact_max_n``) usa-se o teste binomial exato
    (p = 0.5); acima disso, a aproximação qui-quadrado com correção de continuidade
    de Edwards. Retorna ``None`` quando não há discordantes (diferença indefinida).

    Esta é a alternativa correta ao teste z de duas proporções quando as duas corridas
    partilham os mesmos itens: ignorar o emparelhamento sobrestima o erro-padrão e
    perde poder estatístico.
    """
    if b < 0 or c < 0:
        return None
    n_disc = b + c
    if n_disc == 0:
        return None
    if n_disc <= exact_max_n:
        p = _binomial_two_sided_p(min(b, c), n_disc)
        return {
            "metodo": "mcnemar_exato_binomial",
            "n_discordantes": n_disc,
            "b": b,
            "c": c,
            "p_valor": p,
        }
    stat = (abs(b - c) - 1.0) ** 2 / n_disc
    return {
        "metodo": "mcnemar_qui_quadrado_correcao_continuidade",
        "n_discordantes": n_disc,
        "b": b,
        "c": c,
        "estatistica": stat,
        "p_valor": _chi2_sf_1df(stat),
    }


def _binomial_two_sided_p(k: int, n: int) -> float:
    """p-valor bilateral do teste binomial exato com p=0.5 (soma das caudas)."""
    if n <= 0:
        return 1.0
    total = 0.0
    for i in range(0, k + 1):
        total += math.comb(n, i)
    p = 2.0 * total / (2.0**n)
    return min(1.0, p)


def _chi2_sf_1df(x: float) -> float:
    """Função de sobrevivência da qui-quadrado com 1 grau de liberdade.

    Com 1 gl, ``P(X > x) = erfc(sqrt(x/2))`` — sem dependência de SciPy.
    """
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def paired_bootstrap_diff_ci(
    a: list[bool],
    b: list[bool],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20240517,
) -> dict[str, float | int] | None:
    """IC bootstrap percentil para a diferença de taxas ``mean(a) - mean(b)`` emparelhada.

    Reamostra **itens** (não observações independentes), preservando o emparelhamento —
    é o análogo não-paramétrico de McNemar e dá um tamanho de efeito com incerteza,
    não apenas um p-valor. Determinístico via ``seed``.
    """
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    diffs = [1.0 if x else 0.0 for x in a]
    diffs = [d - (1.0 if y else 0.0) for d, y in zip(diffs, b, strict=True)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[max(0, int(alpha * n_resamples) - 1)]
    hi = samples[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return {
        "diferenca_observada": observed,
        "ic_inferior": lo,
        "ic_superior": hi,
        "confianca": confidence,
        "n_pares": n,
        "n_reamostragens": n_resamples,
    }


def expected_calibration_error(
    pairs: list[tuple[float, bool]],
    *,
    n_bins: int = 10,
) -> dict[str, Any] | None:
    """ECE, MCE e tabela de fiabilidade para pares ``(confiança, acertou)``.

    Responde a "quando o modelo diz 0.9, acerta 90% das vezes?". Cada bin compara
    a confiança média declarada com a exatidão observada; o ECE é a média desses
    desvios ponderada pela ocupação do bin, e o MCE é o pior bin.

    Um juiz com ECE alto pode ter boa exatidão e ainda assim ser inútil para
    triagem por limiar de confiança — são propriedades diferentes.
    """
    usable = [(c, ok) for c, ok in pairs if 0.0 <= c <= 1.0]
    n = len(usable)
    if n == 0 or n_bins < 1:
        return None
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Último bin fechado à direita para acomodar confiança == 1.0.
        in_bin = [(c, ok) for c, ok in usable if (lo <= c < hi or (b == n_bins - 1 and c == 1.0))]
        if not in_bin:
            continue
        n_b = len(in_bin)
        conf_b = sum(c for c, _ in in_bin) / n_b
        acc_b = sum(1 for _, ok in in_bin if ok) / n_b
        gap = abs(acc_b - conf_b)
        ece += (n_b / n) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "intervalo": [round(lo, 4), round(hi, 4)],
                "n": n_b,
                "confianca_media": round(conf_b, 4),
                "exatidao": round(acc_b, 4),
                "desvio": round(acc_b - conf_b, 4),
            }
        )
    return {
        "n": n,
        "n_bins_ocupados": len(bins),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "confianca_media": round(sum(c for c, _ in usable) / n, 4),
        "exatidao_global": round(sum(1 for _, ok in usable if ok) / n, 4),
        "bins": bins,
    }


def fleiss_kappa(rating_counts: list[list[int]]) -> float | None:
    """Kappa de Fleiss para concordância entre **várias** avaliações do mesmo item.

    ``rating_counts[i][j]`` = quantas avaliações do item ``i`` caíram na categoria
    ``j``. Todos os itens têm de ter o mesmo número de avaliações. Usado para
    auto-consistência do juiz: N amostras repetidas do mesmo par (pergunta,
    resposta) são "avaliadores" do mesmo item.

    Devolve ``None`` quando o desenho é degenerado (< 2 itens, < 2 avaliações por
    item, ou concordância esperada de 1.0 — todos na mesma categoria).
    """
    if len(rating_counts) < 2:
        return None
    n_ratings = sum(rating_counts[0])
    if n_ratings < 2:
        return None
    if any(sum(row) != n_ratings for row in rating_counts):
        return None
    n_items = len(rating_counts)
    n_cats = len(rating_counts[0])
    if any(len(row) != n_cats for row in rating_counts):
        return None

    p_j = [sum(row[j] for row in rating_counts) / (n_items * n_ratings) for j in range(n_cats)]
    p_i = [
        (sum(x * x for x in row) - n_ratings) / (n_ratings * (n_ratings - 1))
        for row in rating_counts
    ]
    p_bar = sum(p_i) / n_items
    p_exp = sum(p * p for p in p_j)
    if p_exp >= 1.0 - 1e-12:
        return None
    return (p_bar - p_exp) / (1.0 - p_exp)


def point_biserial(binary: list[bool], continuous: list[float]) -> float | None:
    """Correlação ponto-bisserial entre um rótulo binário e uma variável contínua.

    Usada para sondar viés de verbosidade: correlacionar "o juiz aprovou" com o
    comprimento da resposta. Correlação **não** é viés provado — respostas longas
    podem ser genuinamente melhores —, mas um valor alto obriga a inspeção.

    Devolve ``None`` sem variância num dos lados (todos iguais).
    """
    if len(binary) != len(continuous) or len(binary) < 3:
        return None
    n = len(binary)
    grupo1 = [x for b, x in zip(binary, continuous, strict=True) if b]
    grupo0 = [x for b, x in zip(binary, continuous, strict=True) if not b]
    if not grupo1 or not grupo0:
        return None
    media = sum(continuous) / n
    var = sum((x - media) ** 2 for x in continuous) / n
    if var <= 1e-12:
        return None
    desvio = math.sqrt(var)
    m1 = sum(grupo1) / len(grupo1)
    m0 = sum(grupo0) / len(grupo0)
    p = len(grupo1) / n
    return ((m1 - m0) / desvio) * math.sqrt(p * (1.0 - p))


def paired_bootstrap_mean_diff(
    a: list[float],
    b: list[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20240517,
) -> dict[str, float | int] | None:
    """IC bootstrap para a diferença de médias emparelhada em valores contínuos.

    A variante binária (`paired_bootstrap_diff_ci`) serve taxas; esta serve
    métricas por item que vivem numa escala contínua — nDCG por query, F1 por
    resposta. Reamostra **itens**, preservando o emparelhamento, e por isso
    responde à pergunta certa quando dois métodos correm sobre o mesmo conjunto:
    a diferença é distinguível do ruído, e de que tamanho?

    Comparar duas médias sem isto é publicar um ranking sem saber se ele existe.
    """
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    observado = sum(diffs) / n
    rng = random.Random(seed)
    amostras: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        amostras.append(total / n)
    amostras.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = amostras[max(0, int(alpha * n_resamples) - 1)]
    hi = amostras[min(n_resamples - 1, int((1.0 - alpha) * n_resamples))]
    return {
        "diferenca_observada": round(observado, 5),
        "ic_inferior": round(lo, 5),
        "ic_superior": round(hi, 5),
        # O IC não conter zero é a condição para afirmar que há diferença.
        "exclui_zero": bool(lo > 0 or hi < 0),
        "confianca": confidence,
        "n_pares": n,
        "n_reamostragens": n_resamples,
    }


def _percentil_ic(amostras: list[float], *, confidence: float) -> tuple[float, float] | None:
    """IC percentil sobre reamostragens já ordenáveis."""
    if not amostras:
        return None
    amostras = sorted(amostras)
    n = len(amostras)
    alpha = (1.0 - confidence) / 2.0
    lo = amostras[max(0, int(alpha * n) - 1)]
    hi = amostras[min(n - 1, int((1.0 - alpha) * n))]
    return lo, hi


def bootstrap_kappa_ci(
    pares: list[tuple[bool, bool]],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20240517,
) -> dict[str, Any] | None:
    """IC bootstrap para o κ de Cohen, reamostrando **itens**.

    O κ era publicado como estimativa pontual, e é a coluna sobre a qual se argumenta
    a escolha de um juiz. Com N=200 e concordância pouco acima do acaso, o intervalo é
    largo o suficiente para que dois juízes «diferentes» não sejam distinguíveis — o
    que é o resultado, e não um detalhe a omitir.

    ``pares`` são ``(juiz_aprovou, referencia_ok)`` por item. Reamostragens em que o κ
    é indefinido (concordância esperada ≈ 1, isto é, um avaliador constante) são
    contadas em ``n_reamostragens_degeneradas`` em vez de tratadas como zero: 0,0
    leria-se como «não concorda além do acaso» quando não havia classe com que
    concordar.
    """
    if not pares:
        return None
    n = len(pares)
    rng = random.Random(seed)
    amostras: list[float] = []
    degeneradas = 0
    for _ in range(n_resamples):
        tp = fn = fp = tn = 0
        for _ in range(n):
            aprovou, ref_ok = pares[rng.randrange(n)]
            if aprovou and ref_ok:
                tp += 1
            elif not aprovou and ref_ok:
                fn += 1
            elif aprovou and not ref_ok:
                fp += 1
            else:
                tn += 1
        k = cohen_kappa(tp, fn, fp, tn)
        if k is None:
            degeneradas += 1
        else:
            amostras.append(k)
    ic = _percentil_ic(amostras, confidence=confidence)
    if ic is None:
        return None
    return {
        "ic_inferior": round(ic[0], 4),
        "ic_superior": round(ic[1], 4),
        "confianca": confidence,
        "n_itens": n,
        "n_reamostragens": n_resamples,
        "n_reamostragens_degeneradas": degeneradas,
    }


def bootstrap_ece_ci(
    pares: list[tuple[float, bool]],
    *,
    n_bins: int = 10,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20240517,
) -> dict[str, Any] | None:
    """IC bootstrap para o ECE, reamostrando itens.

    O ECE depende da ocupação dos bins, pelo que com poucos itens por bin o valor
    pontual é instável. Publicá-lo sem intervalo convida a ordenar juízes por
    diferenças que o intervalo não sustenta.
    """
    usable = [(c, ok) for c, ok in pares if 0.0 <= c <= 1.0]
    if not usable:
        return None
    n = len(usable)
    rng = random.Random(seed)
    amostras: list[float] = []
    for _ in range(n_resamples):
        reamostra = [usable[rng.randrange(n)] for _ in range(n)]
        res = expected_calibration_error(reamostra, n_bins=n_bins)
        if res is not None:
            amostras.append(float(res["ece"]))
    ic = _percentil_ic(amostras, confidence=confidence)
    if ic is None:
        return None
    return {
        "ic_inferior": round(ic[0], 4),
        "ic_superior": round(ic[1], 4),
        "confianca": confidence,
        "n_itens": n,
        "n_reamostragens": n_resamples,
    }


def holm_bonferroni(p_valores: list[float]) -> list[float]:
    """Correcção de Holm-Bonferroni para comparações múltiplas.

    Com quatro braços comparam-se seis pares ao mesmo tempo; a 5% por par, a
    probabilidade de ao menos um falso positivo na família aproxima-se de 26%.
    Holm é uniformemente mais potente do que Bonferroni e não assume independência,
    o que é o caso aqui: os pares partilham braços.

    Devolve os p ajustados na ordem de entrada, monótonos e limitados a 1,0. Sem
    arredondamento: um p de 1e-14 arredondado a seis casas torna-se 0,0 e perde
    precisamente a informação que distingue um efeito forte de um exacto.
    """
    m = len(p_valores)
    if m == 0:
        return []
    indexados = sorted(enumerate(p_valores), key=lambda t: t[1])
    ajustados = [0.0] * m
    corrente = 0.0
    for posicao, (indice, p) in enumerate(indexados):
        candidato = (m - posicao) * p
        corrente = max(corrente, min(1.0, candidato))
        ajustados[indice] = corrente
    return ajustados


def _z_quantil(p: float) -> float:
    """Quantil da normal padrão por bissecção sobre ``erf``.

    Evita uma dependência de SciPy para dois valores. A bissecção em [-10, 10]
    converge bem abaixo da precisão que um MDE reportado a três casas exige.
    """
    if not 0.0 < p < 1.0:
        msg = f"quantil fora de (0,1): {p}"
        raise ValueError(msg)
    lo, hi = -10.0, 10.0
    for _ in range(200):
        meio = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(meio / math.sqrt(2.0)))
        if cdf < p:
            lo = meio
        else:
            hi = meio
    return (lo + hi) / 2.0


def mcnemar_mde(
    n_pares: int,
    n_discordantes: int,
    *,
    alpha: float = 0.05,
    poder: float = 0.80,
) -> dict[str, Any] | None:
    """Efeito mínimo detectável para um desenho emparelhado com desfecho binário.

    Responde à pergunta que um p=1 deixa em aberto: **qual era a diferença mais
    pequena que este N conseguiria distinguir de ruído?** Sem isto, "não há
    diferença significativa" confunde-se com "os sistemas são equivalentes", e num
    A/B de quatro juízes sobre 200 itens a distinção importa.

    Usa a aproximação normal do McNemar: com ψ = proporção de pares discordantes,
    δ ≈ (z_{α/2} + z_poder) · √(ψ/n). Só os pares discordantes carregam informação,
    pelo que um desenho com poucos é pouco potente por muitos itens que tenha —
    exactamente o caso quando dois juízes concordam quase sempre.
    """
    if n_pares <= 0 or n_discordantes < 0 or n_discordantes > n_pares:
        return None
    psi = n_discordantes / n_pares
    if psi == 0.0:
        return {
            "n_pares": n_pares,
            "n_discordantes": 0,
            "proporcao_discordante": 0.0,
            "efeito_minimo_detectavel": None,
            "nota": (
                "Zero pares discordantes: o desenho nao distingue diferenca alguma, "
                "qualquer que seja o N. Nao e evidencia de equivalencia."
            ),
        }
    z_alpha = _z_quantil(1.0 - alpha / 2.0)
    z_poder = _z_quantil(poder)
    delta = (z_alpha + z_poder) * math.sqrt(psi / n_pares)
    return {
        "n_pares": n_pares,
        "n_discordantes": n_discordantes,
        "proporcao_discordante": round(psi, 4),
        "alpha": alpha,
        "poder": poder,
        "efeito_minimo_detectavel": round(delta, 4),
        "nota": (
            "Diferenca de taxas abaixo deste valor nao seria distinguivel de ruido "
            "neste desenho. Um p nao significativo acima deste limiar e informativo; "
            "abaixo dele, o teste apenas nao tinha poder."
        ),
    }

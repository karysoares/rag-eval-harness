"""Métricas do plano C — referência humana amostral (HITL)."""

from __future__ import annotations

from typing import Any

from llm_evaluation.reference_metrics import (
    referencia_humana_aceitavel,
    referencia_humana_incorreta,
)
from llm_evaluation.statistics import cohen_kappa, wilson_ci
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.aggregate import judge_negative_for_aggregation

#: Mínimo de rótulos para publicar confusão e κ. Abaixo disto o bloco sai com a
#: contagem e a razão, e sem números — é a promessa da SPEC-008, que o código não
#: cumpria: emitia κ a partir de um único rótulo.
MIN_ROTULOS_PARA_METRICAS = 10


def _hitl_label(record: RunRecord) -> dict[str, Any] | None:
    raw = record.meta.get("adjudicacao_humana")
    return raw if isinstance(raw, dict) else None


def _n_rotulados(records: list[RunRecord]) -> int:
    return sum(1 for r in records if _hitl_label(r) is not None)


def _distribuicao_rotulos(records: list[RunRecord]) -> dict[str, int]:
    """Contagem por rótulo humano. Publicada sempre, ao lado de qualquer métrica.

    Uma amostra onde o humano respondeu a mesma coisa a tudo produz números
    plausíveis e vazios de conteúdo: sem classe positiva não há revocação para
    medir, e o κ sai 0 — que se lê como «o juiz não concorda» quando o que
    aconteceu foi não haver com o que concordar. A distribuição torna isso
    visível sem ter de se ir ao CSV.
    """
    dist: dict[str, int] = {}
    for r in records:
        rotulo = _hitl_label(r)
        if rotulo is None:
            continue
        chave = str(rotulo.get("rotulo") or "").lower() or "(vazio)"
        dist[chave] = dist.get(chave, 0) + 1
    return dict(sorted(dist.items()))


def _confusion(
    records: list[RunRecord],
    *,
    predictor: Any,
) -> dict[str, int] | None:
    labeled = [r for r in records if _hitl_label(r) is not None]
    if not labeled:
        return None
    tp = fn = fp = tn = 0
    for r in labeled:
        human_bad = referencia_humana_incorreta(r)
        if human_bad is None:
            continue
        pred = predictor(r)
        if pred is None:
            continue
        if human_bad and pred:
            tp += 1
        elif human_bad and not pred:
            fn += 1
        elif not human_bad and pred:
            fp += 1
        else:
            tn += 1
    return {
        "vp": tp,
        "fn": fn,
        "fp": fp,
        "vn": tn,
    }


def summarize_hitl(
    records: list[RunRecord],
    *,
    protocol: dict[str, object] | None = None,
    fila_total: int | None = None,
) -> dict[str, object] | None:
    n_rot = _n_rotulados(records)
    if n_rot == 0:
        return None

    aceitaveis = sum(1 for r in records if referencia_humana_aceitavel(r))
    n_pendentes = None
    if fila_total is not None:
        n_pendentes = max(0, fila_total - n_rot)

    agg_verdicts: list[str] = []
    if protocol:
        raw = protocol.get("judge_aggregation_verdicts")
        if isinstance(raw, list):
            agg_verdicts = [str(x) for x in raw]

    def _detector(r: RunRecord) -> bool:
        return r.anomaly_flag

    def _embedding(r: RunRecord) -> bool | None:
        v = r.signals.embedding_low_support
        return bool(v) if v is not None else None

    def _juiz(r: RunRecord) -> bool | None:
        if r.signals.judge is None:
            return None
        if r.signals.judge.raw.get("fallback_heuristico"):
            return None
        if not agg_verdicts:
            return r.signals.judge_negative
        return judge_negative_for_aggregation(r.signals, agg_verdicts)

    distribuicao = _distribuicao_rotulos(records)
    if n_rot < MIN_ROTULOS_PARA_METRICAS:
        return {
            "n_itens_rotulados": n_rot,
            "n_pendentes_fila": n_pendentes,
            "distribuicao_rotulos": distribuicao,
            "taxa_aceitavel_humana": aceitaveis / n_rot,
            "metricas_omitidas": (
                f"N={n_rot} abaixo do mínimo de {MIN_ROTULOS_PARA_METRICAS} rótulos; "
                "confusão e κ não são publicados a esta escala."
            ),
            "nota": "Métricas HITL: amostra rotulada; não extrapolar ao corpus inteiro.",
        }

    conf_det = _confusion(records, predictor=_detector)
    conf_emb = _confusion(records, predictor=_embedding)
    conf_juiz = _confusion(records, predictor=_juiz)

    def _rates(conf: dict[str, int] | None) -> dict[str, object] | None:
        if not conf:
            return None
        tp, fn, fp, tn = conf["vp"], conf["fn"], conf["fp"], conf["vn"]
        denom_p = tp + fp
        denom_r = tp + fn
        out: dict[str, object] = {
            "confusao": conf,
            "precisao": (tp / denom_p) if denom_p else None,
            "revocacao": (tp / denom_r) if denom_r else None,
            "ic95_precisao": wilson_ci(tp, denom_p) if denom_p else None,
            "ic95_revocacao": wilson_ci(tp, denom_r) if denom_r else None,
        }
        # `cohen_kappa` devolve 0.0 quando a verdade humana tem uma só classe: com
        # tp = fn = 0 a concordância esperada iguala a observada. Zero lê-se como
        # «não concordam para além do acaso», e o que se passou foi não haver
        # classe com que concordar. Devolver o motivo em vez do número.
        if tp + fn == 0:
            out["kappa"] = None
            out["kappa_indefinido"] = "sem classe positiva na verdade humana"
        elif fp + tn == 0:
            out["kappa"] = None
            out["kappa_indefinido"] = "sem classe negativa na verdade humana"
        else:
            out["kappa"] = cohen_kappa(tp, fn, fp, tn)
        return out

    return {
        "n_itens_rotulados": n_rot,
        "n_pendentes_fila": n_pendentes,
        "distribuicao_rotulos": distribuicao,
        "taxa_aceitavel_humana": aceitaveis / n_rot if n_rot else None,
        "nota": "Métricas HITL: amostra rotulada; não extrapolar ao corpus inteiro.",
        "detector_vs_humano": _rates(conf_det),
        "embedding_vs_humano": _rates(conf_emb),
        "juiz_vs_humano": _rates(conf_juiz),
    }

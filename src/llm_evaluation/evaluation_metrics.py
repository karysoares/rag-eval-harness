"""Agregação de métricas por camada de verificação e comparação entre corridas.

Para ``reference_type=lexical``, a referência de confusão/kappa usa F1 token (SPEC-007),
não substring em listas ``correct``/``incorrect``. Embedding e juiz medem ancoragem RAG.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from llm_evaluation.reference_metrics import referencia_incorreta
from llm_evaluation.statistics import (
    cohen_kappa,
    mcnemar_test,
    paired_bootstrap_diff_ci,
    wilson_ci,
)
from llm_evaluation.types import JudgeResult, RetrievedChunk, RunRecord, VerificationSignals


def resolve_judge_verdict_lists(
    *,
    negative_judge_verdicts: list[str] | None = None,
    judge_aggregation_verdicts: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Listas para agregação offline / ``analise_camadas`` (defaults excluem incompleto)."""
    negs = negative_judge_verdicts or [
        "nao_sustentado",
        "contradicacao",
        "inseguro",
        "incompleto",
    ]
    agg = (
        judge_aggregation_verdicts
        if judge_aggregation_verdicts is not None
        else _default_judge_aggregation_verdicts(negs)
    )
    return negs, agg


def _make_triggers(
    judge_aggregation_verdicts: list[str],
) -> Callable[[RunRecord], tuple[bool, bool, bool]]:
    from llm_evaluation.verification.aggregate import judge_negative_for_aggregation

    def triggers(r: RunRecord) -> tuple[bool, bool, bool]:
        s = r.signals
        return (
            s.gold_incorrect is True,
            s.embedding_low_support is True,
            judge_negative_for_aggregation(s, judge_aggregation_verdicts),
        )

    return triggers


def _confusion_vs_ref(pred_positive: bool, ref_positive: bool) -> tuple[int, int, int, int]:
    """pred_positive = camada indica problema; ref_positive = gold_correct é False."""
    if ref_positive:
        if pred_positive:
            return 1, 0, 0, 0  # tp
        return 0, 1, 0, 0  # fn
    if pred_positive:
        return 0, 0, 1, 0  # fp
    return 0, 0, 0, 1  # tn


def _ref_layer_description(reference_type: str | None) -> str:
    if reference_type == "answer_lists":
        return (
            "Referência positiva = resposta não alinha com listas correct/incorrect do dataset. "
            "Camada positiva = gatilho dessa camada."
        )
    if reference_type == "lexical":
        return (
            "Referência positiva = baixo overlap léxico (F1/EM) com a resposta curta do dataset — "
            "não é falha do detector «ouro» nem anomalia RAG por si só. "
            "Camada positiva = gatilho da camada indicada."
        )
    return (
        "Referência positiva = item com referência automática «incorreta». "
        "Camada positiva = gatilho dessa camada."
    )


def layer_analysis(
    records: list[RunRecord],
    *,
    reference_type: str | None = None,
    f1_fraca_min: float | None = None,
    negative_judge_verdicts: list[str] | None = None,
    judge_aggregation_verdicts: list[str] | None = None,
) -> dict[str, object]:
    """Gatilhos marginais, combinações exclusivas e confusão por camada vs referência do dataset."""
    negs, agg = resolve_judge_verdict_lists(
        negative_judge_verdicts=negative_judge_verdicts,
        judge_aggregation_verdicts=judge_aggregation_verdicts,
    )
    triggers = _make_triggers(agg)
    from llm_evaluation.verification.aggregate import (
        judge_negative_for_aggregation,
        judge_negative_for_diagnosis,
    )

    n = len(records)
    labeled = [
        r
        for r in records
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_fraca_min) is not None
    ]
    ref_incorrect = [
        r
        for r in labeled
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_fraca_min) is True
    ]

    marg_g = marg_e = marg_j = marg_j_diag = 0
    excl: dict[str, int] = {
        "ouro_apenas": 0,
        "so_embedding": 0,
        "so_juiz": 0,
        "ouro_e_embedding": 0,
        "ouro_e_juiz": 0,
        "embedding_e_juiz": 0,
        "tres_sinais": 0,
        "nenhum_sinal": 0,
    }
    for r in records:
        g, e, j = triggers(r)
        if g:
            marg_g += 1
        if e:
            marg_e += 1
        if j:
            marg_j += 1
        if judge_negative_for_diagnosis(r.signals, negs):
            marg_j_diag += 1
        key = (g, e, j)
        if key == (True, False, False):
            excl["ouro_apenas"] += 1
        elif key == (False, True, False):
            excl["so_embedding"] += 1
        elif key == (False, False, True):
            excl["so_juiz"] += 1
        elif key == (True, True, False):
            excl["ouro_e_embedding"] += 1
        elif key == (True, False, True):
            excl["ouro_e_juiz"] += 1
        elif key == (False, True, True):
            excl["embedding_e_juiz"] += 1
        elif key == (True, True, True):
            excl["tres_sinais"] += 1
        else:
            excl["nenhum_sinal"] += 1

    def layer_vs_gold(name: str, pred_fn: Callable[[RunRecord], bool]) -> dict[str, object]:
        tp = fn = fp = tn = 0
        for r in labeled:
            ref_pos = referencia_incorreta(r, reference_type, f1_fraca_min=f1_fraca_min) is True
            pred_pos = pred_fn(r)
            tpi, fni, fpi, tni = _confusion_vs_ref(pred_pos, ref_pos)
            tp += tpi
            fn += fni
            fp += fpi
            tn += tni
        denom_p = tp + fp
        denom_r = tp + fn
        return {
            "camada": name,
            "descricao": _ref_layer_description(reference_type),
            "vp": tp,
            "fn": fn,
            "fp": fp,
            "vn": tn,
            "precisao": (tp / denom_p) if denom_p else None,
            "ic95_precisao": wilson_ci(tp, denom_p) if denom_p else None,
            "revocacao": (tp / denom_r) if denom_r else None,
            "ic95_revocacao": wilson_ci(tp, denom_r) if denom_r else None,
            "acuracia_balanceada": (
                0.5 * ((tp / (tp + fn)) if (tp + fn) else 0.0)
                + 0.5 * ((tn / (tn + fp)) if (tn + fp) else 0.0)
            ),
            "cohen_kappa_vs_gold": cohen_kappa(tp, fn, fp, tn),
        }

    gold_layer = layer_vs_gold("sinal_ouro_incorreto", lambda r: r.signals.gold_incorrect is True)
    emb_layer = layer_vs_gold(
        "embedding_baixo_suporte",
        lambda r: r.signals.embedding_low_support is True,
    )
    judge_layer = layer_vs_gold(
        "juiz_negativo_agregacao",
        lambda r: judge_negative_for_aggregation(r.signals, agg),
    )

    among_anomaly = [r for r in records if r.anomaly_flag]
    excl_anom: dict[str, int] = {k: 0 for k in excl}
    for r in among_anomaly:
        g, e, j = triggers(r)
        key = (g, e, j)
        if key == (True, False, False):
            excl_anom["ouro_apenas"] += 1
        elif key == (False, True, False):
            excl_anom["so_embedding"] += 1
        elif key == (False, False, True):
            excl_anom["so_juiz"] += 1
        elif key == (True, True, False):
            excl_anom["ouro_e_embedding"] += 1
        elif key == (True, False, True):
            excl_anom["ouro_e_juiz"] += 1
        elif key == (False, True, True):
            excl_anom["embedding_e_juiz"] += 1
        elif key == (True, True, True):
            excl_anom["tres_sinais"] += 1
        else:
            excl_anom["nenhum_sinal"] += 1

    layer_predicates: dict[str, Callable[[RunRecord], bool]] = {
        "sinal_ouro": lambda r: r.signals.gold_incorrect is True,
        "embedding": lambda r: r.signals.embedding_low_support is True,
        "juiz": lambda r: judge_negative_for_aggregation(r.signals, agg),
    }

    def _pair_kappa(name_a: str, name_b: str) -> dict[str, object]:
        fn_a = layer_predicates[name_a]
        fn_b = layer_predicates[name_b]
        tp = fn = fp = tn = 0
        for r in records:
            a_pos = fn_a(r)
            b_pos = fn_b(r)
            if a_pos and b_pos:
                tp += 1
            elif a_pos and not b_pos:
                fp += 1
            elif not a_pos and b_pos:
                fn += 1
            else:
                tn += 1
        return {
            "par": f"{name_a}__vs__{name_b}",
            "vp": tp,
            "fn": fn,
            "fp": fp,
            "vn": tn,
            "cohen_kappa": cohen_kappa(tp, fn, fp, tn),
        }

    pares: list[dict[str, object]] = [
        _pair_kappa("sinal_ouro", "embedding"),
        _pair_kappa("sinal_ouro", "juiz"),
        _pair_kappa("embedding", "juiz"),
    ]

    ref_label = (
        "referencia_lexica_fraca"
        if reference_type == "lexical"
        else "sinal_listas_incorretas"
        if reference_type == "answer_lists"
        else "referencia_automatica_incorreta"
    )

    return {
        "versao_esquema": "2",
        "tipo_referencia": reference_type,
        "n_itens": n,
        "n_rotulados_referencia": len(labeled),
        "n_referencia_incorreta": len(ref_incorrect),
        "rotulo_camada_referencia": ref_label,
        "gatilhos_marginais": {
            "n_sinal_ouro_incorreto": marg_g,
            "n_referencia_lexica_fraca": marg_g if reference_type == "lexical" else None,
            "n_embedding_baixo_suporte": marg_e,
            "n_juiz_negativo": marg_j,
            "n_juiz_diagnostico_negativo": marg_j_diag,
        },
        "vereditos_juiz_agregacao": agg,
        "vereditos_juiz_diagnostico": negs,
        "combinacoes_exclusivas_todos_itens": excl,
        "combinacoes_exclusivas_so_anomalias": excl_anom,
        "por_camada_vs_referencia": {
            "sinal_ouro": gold_layer,
            "sinal_embedding": emb_layer,
            "sinal_juiz": judge_layer,
        },
        "concordancia_entre_camadas": pares,
        "nota_referencia": (
            "Referência positiva = item com referência fraca (F1 abaixo do limiar em lexical, "
            "ou substring/listas noutros tipos). Alerta = política de agregação activa. "
            "n_juiz_negativo = vereditos na lista de agregação (re-derivado do JSONL); "
            "n_juiz_diagnostico_negativo inclui avisos (ex. incompleto). "
            "Kappa baixo é esperado: RAG e overlap léxico medem dimensões diferentes. "
            "Cohen's kappa > 0.6 = concordância substancial entre duas anotações binárias."
        ),
    }


def _dget(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def prediction_row_to_run_record(row: dict[str, Any]) -> RunRecord:
    """Reconstrói RunRecord a partir de ``record_to_json`` / linha predictions.jsonl."""
    sig = _dget(row, "sinais", "signals") or {}
    if not isinstance(sig, dict):
        sig = {}
    jd = _dget(sig, "juiz", "judge")
    judge: JudgeResult | None = None
    if isinstance(jd, dict):
        ver = _dget(jd, "veredito", "verdict")
        if ver:
            from llm_evaluation.veredito import normalizar_veredito

            v_canon = normalizar_veredito(str(ver))
            skip = {
                "veredito",
                "verdict",
                "motivo_breve",
                "reason_short",
                "confianca",
                "confidence",
                "cadeia_de_pensamento",
                "chain_of_thought",
            }
            raw = {k: v for k, v in jd.items() if k not in skip}
            cot = _dget(jd, "cadeia_de_pensamento", "chain_of_thought")
            if cot is not None:
                raw["cadeia_de_pensamento"] = cot
            conf_raw = _dget(jd, "confianca", "confidence")
            if conf_raw is None:
                # ``JudgeResult.confianca`` é obrigatório; 0.5 é um valor de
                # preenchimento, não uma medição. A marca permite que a calibração
                # (``judge_meta``) descarte estes itens em vez de os tratar como
                # confiança real e contaminar o ECE.
                conf_raw = 0.5
                raw["confianca_ausente"] = True
            judge = JudgeResult(
                veredito=v_canon,
                motivo_breve=str(_dget(jd, "motivo_breve", "reason_short") or ""),
                confianca=float(conf_raw),
                raw=raw,
            )
    gc = _dget(sig, "gold_correto", "gold_correct")
    if gc is None:
        gc = _dget(row, "gold_correto", "gold_correct")
    gi = _dget(sig, "gold_incorreto", "gold_incorrect")
    rec = _dget(sig, "e_recusa", "is_refusal")
    emb_max = _dget(sig, "embedding_max_coseno", "embedding_max_cosine")
    emb_low = _dget(sig, "embedding_baixo_suporte", "embedding_low_support")
    emb_ret = _dget(sig, "embedding_max_coseno_recuperados", "embedding_max_cosine_retrieved")
    emb_gold = _dget(sig, "embedding_max_coseno_ouro", "embedding_max_cosine_gold")
    jneg = _dget(sig, "juiz_negativo", "judge_negative")
    vs = VerificationSignals(
        gold_correct=gc,
        gold_incorrect=gi,
        is_refusal=bool(rec if rec is not None else False),
        embedding_max_cosine=emb_max,
        embedding_low_support=emb_low,
        embedding_max_cosine_retrieved=emb_ret,
        embedding_max_cosine_gold=emb_gold,
        judge=judge,
        judge_negative=jneg,
    )
    retrieved: list[RetrievedChunk] = []
    for c in _dget(row, "recuperados", "retrieved") or []:
        if isinstance(c, dict):
            retrieved.append(
                RetrievedChunk(
                    text=str(_dget(c, "texto", "text") or ""),
                    score=float(_dget(c, "pontuacao", "score") or 0.0),
                    is_gold=bool(_dget(c, "e_ouro", "is_gold") or False),
                )
            )
    meta = dict(row.get("meta") or {})
    refs = row.get("referencias") or row.get("references")
    if refs and "referencias" not in meta:
        meta["referencias"] = refs
    diag = row.get("diagnostico")
    if isinstance(diag, dict) and "diagnostico" not in meta:
        meta["diagnostico"] = diag
    return RunRecord(
        item_id=str(_dget(row, "id_item", "item_id") or ""),
        question=str(_dget(row, "pergunta", "question") or ""),
        answer=str(_dget(row, "resposta", "answer") or ""),
        gold_correct=_dget(row, "gold_correto", "gold_correct"),
        anomaly_flag=bool(_dget(row, "flag_anomalia", "anomaly_flag") or False),
        signals=vs,
        retrieved=retrieved,
        baseline_profile=str(_dget(row, "perfil_baseline", "baseline_profile") or ""),
        meta=meta,
    )


def load_records_from_predictions_jsonl(path: Path) -> list[RunRecord]:
    """Última linha por ``id_item`` ganha (dedupe para resume/merge)."""
    by_id: dict[str, RunRecord] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = prediction_row_to_run_record(json.loads(line))
        if not rec.item_id:
            continue
        if rec.item_id not in by_id:
            order.append(rec.item_id)
        by_id[rec.item_id] = rec
    return [by_id[iid] for iid in order]


def build_metrics_report(
    records: list[RunRecord],
    *,
    extra: dict[str, object] | None = None,
    reference_type: str = "answer_lists",
    protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    from llm_evaluation.reporting import summarize as summarize_fn

    out = dict(
        summarize_fn(records, reference_type=reference_type, protocol=protocol),
    )
    if extra:
        out["meta_corrida"] = extra
    return out


def _first_present(d: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _infer_reference_type_from_records(records: list[RunRecord]) -> tuple[str, str] | None:
    n_lex = 0
    for r in records:
        lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics")
        if not isinstance(lm, dict):
            continue
        if lm.get("note") in {"metricas_lexicas_desligadas", "sem_referencia"}:
            continue
        if lm.get("f1_token") is not None or lm.get("em_squad") is not None:
            n_lex += 1
    if n_lex:
        return (
            "lexical",
            f"tipo_referencia inferido como lexical a partir de metricas_lexicas em {n_lex} itens",
        )
    return None


def compare_metric_reports(
    reports: list[dict[str, object]],
    labels: list[str],
    *,
    flags_por_corrida: dict[str, dict[str, bool]] | None = None,
    falhas_por_corrida: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    """Métricas de alto nível lado a lado para vários relatórios / sumários.

    Com ``flags_por_corrida`` (rótulo → ``{id_item: flag_anomalia}``) acrescenta
    ``significancia_emparelhada``: McNemar + bootstrap emparelhado sobre os itens
    comuns. É o teste correto quando as corridas partilham o mesmo dataset — ver
    ``_pairwise_significance`` para a variante não-emparelhada (fallback).
    """
    rows: list[dict[str, object]] = []
    for label, rep in zip(labels, reports, strict=True):
        cg = (
            rep.get("confusao_vs_referencia")
            or rep.get("confusao_vs_gold")
            or rep.get("confusion_vs_gold")
        )
        if not isinstance(cg, dict):
            cg = {}
        ls = rep.get("sumario_lexical") or rep.get("lexical_summary")
        ls_d = ls if isinstance(ls, dict) else {}
        rows.append(
            {
                "rotulo": label,
                "n_itens": _first_present(rep, "n_itens", "n_items"),
                "n_anomalias_marcadas": _first_present(
                    rep,
                    "n_anomalias_marcadas",
                    "n_anomalies_flagged",
                ),
                "revocacao_marcacao_no_gold_incorreto": _first_present(
                    rep,
                    "revocacao_marcacao_no_gold_incorreto",
                    "recall_flag_on_gold_incorrect",
                ),
                "taxa_falso_alarme_no_gold_correto": _first_present(
                    rep,
                    "taxa_falso_alarme_no_gold_correto",
                    "false_alarm_rate_on_gold_correct",
                ),
                "acuracia_balanceada_gold": _first_present(
                    rep,
                    "acuracia_balanceada_gold",
                    "balanced_accuracy_gold",
                ),
                "precisao_anomalia_vs_gold_incorreto": _first_present(
                    rep,
                    "precisao_anomalia_vs_gold_incorretoprecision_anomaly_vs_gold_incorrect",
                ),
                "vp": _first_present(
                    cg,
                    "vp_referencia_incorreta_marcado",
                    "vp_gold_incorreto_marcado",
                    "tp_gold_incorrect_and_flagged",
                ),
                "fn": _first_present(
                    cg,
                    "fn_referencia_incorreta_nao_marcado",
                    "fn_gold_incorreto_nao_marcado",
                    "fn_gold_incorrect_not_flagged",
                ),
                "fp": _first_present(
                    cg,
                    "fp_referencia_correta_mas_marcado",
                    "fp_gold_correto_mas_marcado",
                    "fp_gold_correct_but_flagged",
                ),
                "vn": _first_present(
                    cg,
                    "vn_referencia_correta_nao_marcado",
                    "vn_gold_correto_nao_marcado",
                    "tn_gold_correct_not_flagged",
                ),
                "media_bleu": _first_present(ls_d, "media_bleu", "mean_bleu"),
                "media_rouge_l_f": _first_present(
                    ls_d,
                    "media_rouge_l_f",
                    "mean_rouge_l_fmeasure",
                ),
                "media_meteor": _first_present(ls_d, "media_meteor", "mean_meteor"),
                "media_similaridade_levenshtein": _first_present(
                    ls_d,
                    "media_similaridade_levenshtein",
                    "mean_levenshtein_similarity",
                ),
            }
        )
    significancia = _pairwise_significance(rows)
    out: dict[str, object] = {
        "versao_esquema": "2",
        "corridas": rows,
        "significancia": significancia,
    }
    if flags_por_corrida:
        emparelhada = pairwise_paired_significance(flags_por_corrida, falhas_por_corrida)
        if emparelhada:
            out["significancia_emparelhada"] = emparelhada
    return out


def _pairwise_significance(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Teste **não-emparelhado** de diferença de proporções (taxa de alerta) entre corridas.

    Fallback para quando não há alinhamento por ``id_item`` (corridas sobre datasets
    ou amostras diferentes). Quando as corridas partilham itens, prefira
    ``pairwise_paired_significance`` — o teste não-emparelhado sobrestima o
    erro-padrão e perde poder.
    """
    out: list[dict[str, object]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            na = a.get("n_itens")
            nb = b.get("n_itens")
            fa = a.get("n_anomalias_marcadas")
            fb = b.get("n_anomalias_marcadas")
            if not isinstance(na, int | float):
                continue
            if not isinstance(nb, int | float):
                continue
            if not isinstance(fa, int | float):
                continue
            if not isinstance(fb, int | float):
                continue
            na_i, nb_i = int(na), int(nb)
            fa_i, fb_i = int(fa), int(fb)
            if na_i < 2 or nb_i < 2:
                continue
            pa = fa_i / na_i
            pb = fb_i / nb_i
            diff = abs(pa - pb)
            se = ((pa * (1 - pa) / na_i) + (pb * (1 - pb) / nb_i)) ** 0.5
            z = diff / se if se > 1e-9 else 0.0
            significativo = z > 1.96
            out.append(
                {
                    "par": [a.get("rotulo"), b.get("rotulo")],
                    "metrica": "taxa_alerta",
                    "diff_absoluta": diff,
                    "z_score": z,
                    "significativo_95": significativo,
                },
            )
    return out


def anomaly_flags_by_item(records: list[RunRecord]) -> dict[str, bool]:
    """``{id_item: flag_anomalia}`` — chave de alinhamento para testes emparelhados."""
    return {r.item_id: bool(r.anomaly_flag) for r in records}


def failed_item_ids(records: list[RunRecord]) -> set[str]:
    """Itens que não chegaram a ser avaliados (erro de execução, não do sistema)."""
    return {r.item_id for r in records if isinstance(r.meta.get("processing_error"), dict)}


def load_anomaly_flags(run_dir: Path) -> dict[str, bool]:
    """Lê ``predictions.jsonl`` (ou o primeiro ``predictions_*.jsonl``) e devolve as flags."""
    return load_run_flags(run_dir)[0]


def load_run_flags(run_dir: Path) -> tuple[dict[str, bool], set[str]]:
    """``({id_item: flag_anomalia}, {ids que falharam})`` de um diretório de corrida."""
    rd = run_dir.resolve()
    primary = rd / "predictions.jsonl"
    if not primary.is_file():
        alt = sorted(rd.glob("predictions_*.jsonl"))
        if not alt:
            return {}, set()
        primary = alt[0]
    registos = load_records_from_predictions_jsonl(primary)
    return anomaly_flags_by_item(registos), failed_item_ids(registos)


def paired_significance(
    label_a: str,
    flags_a: dict[str, bool],
    label_b: str,
    flags_b: dict[str, bool],
    *,
    failed_a: set[str] | None = None,
    failed_b: set[str] | None = None,
) -> dict[str, object] | None:
    """McNemar exato/qui-quadrado + bootstrap emparelhado sobre os itens comuns.

    ``b`` conta itens marcados só por A; ``c`` itens marcados só por B. Devolve
    ``None`` quando não há sobreposição de itens (desenho não emparelhado).

    **Itens com erro de execução são excluídos.** ``_failed_record`` marca
    ``flag_anomalia`` para que a falha seja revista, o que é correcto para a fila
    operacional mas venenoso aqui: uma corrida que perdeu itens por rate limit ou
    quota apareceria com anomalias "exclusivas" que nada têm a ver com a qualidade
    do sistema avaliado. Num caso real, 9 falhas de API produziram um McNemar
    significativo (p=0.004) que media apenas propagação de faturação. As contagens
    excluídas ficam no resultado, e uma exclusão assimétrica gera aviso explícito.
    """
    excluidos_a = failed_a or set()
    excluidos_b = failed_b or set()
    excluidos = excluidos_a | excluidos_b
    common = sorted((set(flags_a) & set(flags_b)) - excluidos)
    if not common:
        return None
    va = [flags_a[i] for i in common]
    vb = [flags_b[i] for i in common]
    b = sum(1 for x, y in zip(va, vb, strict=True) if x and not y)
    c = sum(1 for x, y in zip(va, vb, strict=True) if y and not x)
    out: dict[str, object] = {
        "par": [label_a, label_b],
        "metrica": "taxa_alerta",
        "n_itens_comuns": len(common),
        "cobertura_a": len(common) / len(flags_a) if flags_a else None,
        "cobertura_b": len(common) / len(flags_b) if flags_b else None,
        "so_a": b,
        "so_b": c,
        "mcnemar": mcnemar_test(b, c),
        "bootstrap_emparelhado": paired_bootstrap_diff_ci(va, vb),
    }
    n_exc_a, n_exc_b = len(excluidos_a), len(excluidos_b)
    if n_exc_a or n_exc_b:
        out["excluidos_por_erro"] = {"a": n_exc_a, "b": n_exc_b}
        if n_exc_a != n_exc_b:
            out["aviso_exclusao_assimetrica"] = (
                f"{label_a} perdeu {n_exc_a} item(ns) por erro de execução e "
                f"{label_b} perdeu {n_exc_b}. A comparação usa só os itens que ambas "
                "avaliaram; verifique se a assimetria indica um problema de infraestrutura."
            )
    mc = out["mcnemar"]
    if isinstance(mc, dict):
        p = mc.get("p_valor")
        out["significativo_95"] = isinstance(p, float) and p < 0.05
    return out


def pairwise_paired_significance(
    flags_por_corrida: dict[str, dict[str, bool]],
    falhas_por_corrida: dict[str, set[str]] | None = None,
) -> list[dict[str, object]]:
    """Aplica :func:`paired_significance` a todos os pares de corridas."""
    falhas = falhas_por_corrida or {}
    labels = list(flags_por_corrida)
    out: list[dict[str, object]] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            res = paired_significance(
                la,
                flags_por_corrida[la],
                lb,
                flags_por_corrida[lb],
                failed_a=falhas.get(la),
                failed_b=falhas.get(lb),
            )
            if res is not None:
                out.append(res)
    return out


def analyze_run_dir(run_dir: Path) -> dict[str, object]:
    """Carrega ``predictions*.jsonl`` de um diretório de corrida e monta o relatório completo."""
    primary = run_dir / "predictions.jsonl"
    if primary.is_file():
        records = load_records_from_predictions_jsonl(primary)
        extra = {"source": str(primary)}
    else:
        candidates = sorted(run_dir.glob("predictions_*.jsonl"))
        if not candidates:
            msg = f"Sem predictions.jsonl ou predictions_*.jsonl em {run_dir}"
            raise FileNotFoundError(msg)
        records = load_records_from_predictions_jsonl(candidates[0])
        extra = {
            "source": str(candidates[0]),
            "note": (
                "apenas o primeiro predictions_*.jsonl; use corrida com ficheiro único "
                "para híbrido completo"
            ),
        }

    protocol: dict[str, object] | None = None
    reference_type: str | None = None
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary_raw = json.loads(summary_path.read_text(encoding="utf-8"))
        pa = summary_raw.get("protocolo_ativo")
        if isinstance(pa, dict):
            protocol = cast(dict[str, object], pa)
        rt = summary_raw.get("tipo_referencia_ativo")
        if rt is not None:
            reference_type = str(rt)
        elif reference_type is None:
            reference_type = "answer_lists"
    elif reference_type is None:
        inferred = _infer_reference_type_from_records(records)
        if inferred is None:
            msg = (
                "Sem summary/config/reference_type para analisar corrida; "
                "não vou assumir answer_lists."
            )
            raise ValueError(msg)
        reference_type, note = inferred
        extra["warning"] = note

    return build_metrics_report(
        records,
        extra={"run_dir": str(run_dir.resolve()), **extra},
        reference_type=reference_type,
        protocol=protocol,
    )


def load_full_report(run_dir: Path) -> dict[str, object]:
    """Prefere ``summary.json`` gravado; reprocessa só se em falta."""
    rd = run_dir.resolve()
    sp = rd / "summary.json"
    if sp.is_file():
        return cast(dict[str, object], json.loads(sp.read_text(encoding="utf-8")))
    if (rd / "predictions.jsonl").is_file():
        return analyze_run_dir(rd)
    multi = sorted(rd.glob("predictions_*.jsonl"))
    if multi:
        records = load_records_from_predictions_jsonl(multi[0])
        protocol: dict[str, object] | None = None
        reference_type: str | None = None
        warning: str | None = None
        sp = rd / "summary.json"
        if sp.is_file():
            summary_raw = json.loads(sp.read_text(encoding="utf-8"))
            pa = summary_raw.get("protocolo_ativo")
            if isinstance(pa, dict):
                protocol = cast(dict[str, object], pa)
            rt = summary_raw.get("tipo_referencia_ativo")
            if rt is not None:
                reference_type = str(rt)
            elif reference_type is None:
                reference_type = "answer_lists"
        elif reference_type is None:
            inferred = _infer_reference_type_from_records(records)
            if inferred is None:
                msg = (
                    "Sem summary/config/reference_type para analisar corrida; "
                    "não vou assumir answer_lists."
                )
                raise ValueError(msg)
            reference_type, warning = inferred
        extra: dict[str, object] = {"run_dir": str(rd), "source": str(multi[0])}
        if warning is not None:
            extra["warning"] = warning
        return build_metrics_report(
            records,
            extra=extra,
            reference_type=reference_type,
            protocol=protocol,
        )
    sp = rd / "summary.json"
    if sp.is_file():
        raw = json.loads(sp.read_text(encoding="utf-8"))
        return cast(dict[str, object], raw)
    msg = f"Sem predictions*.jsonl ou summary.json em {run_dir}"
    raise FileNotFoundError(msg)


def recompute_embedding_low_support(
    records: list[RunRecord],
    embedding_min_cosine: float,
) -> list[RunRecord]:
    """Recalcula ``embedding_low_support`` a partir de ``embedding_max_cosine`` (offline)."""
    from dataclasses import replace

    out: list[RunRecord] = []
    for r in records:
        s = r.signals
        emb_max = s.embedding_max_cosine
        emb_low = s.embedding_low_support if emb_max is None else emb_max < embedding_min_cosine
        new_s = replace(s, embedding_low_support=emb_low)
        out.append(replace(r, signals=new_s))
    return out


def _default_judge_aggregation_verdicts(negative_judge_verdicts: list[str]) -> list[str]:
    advisory = {"incompleto"}
    return [x for x in negative_judge_verdicts if x not in advisory]


def replay_anomaly_flags(
    records: list[RunRecord],
    *,
    verify_gold: bool,
    verify_embedding: bool,
    verify_judge: bool,
    negative_judge_verdicts: list[str],
    policy: str,
    judge_aggregation_verdicts: list[str] | None = None,
    embedding_min_cosine: float | None = None,
) -> list[bool]:
    """Reaplica política de agregação aos sinais já persistidos (sem API)."""
    from llm_evaluation.verification.aggregate import anomaly_from_signals

    agg = (
        judge_aggregation_verdicts
        if judge_aggregation_verdicts is not None
        else _default_judge_aggregation_verdicts(negative_judge_verdicts)
    )
    pool = (
        recompute_embedding_low_support(records, embedding_min_cosine)
        if embedding_min_cosine is not None
        else records
    )
    return [
        anomaly_from_signals(
            r.signals,
            verify_gold=verify_gold,
            verify_embedding=verify_embedding,
            verify_judge=verify_judge,
            negative_judge_verdicts=negative_judge_verdicts,
            judge_aggregation_verdicts=agg,
            policy=policy,  # type: ignore[arg-type]
        )
        for r in pool
    ]


def referencia_aceitavel(
    record: RunRecord,
    reference_type: str | None,
    *,
    f1_fraca_min: float | None = None,
) -> bool | None:
    """True = referência aceitável (equivalente a gold-correto para replay de políticas)."""
    if reference_type == "none":
        return None
    if reference_type == "lexical":
        ref = referencia_incorreta(record, reference_type, f1_fraca_min=f1_fraca_min)
        if ref is None:
            return None
        return ref is False
    if record.gold_correct is None:
        return None
    return record.gold_correct is True


def referencia_problematica(
    record: RunRecord,
    reference_type: str | None,
    *,
    f1_fraca_min: float | None = None,
) -> bool | None:
    """True = referência fraca/incorreta (equivalente a gold-incorreto)."""
    if reference_type == "none":
        return None
    if reference_type == "lexical":
        return referencia_incorreta(record, reference_type, f1_fraca_min=f1_fraca_min)
    if record.gold_correct is None:
        return None
    return record.gold_correct is False


def _fp_rate_on_reference_acceptable(
    flags: list[bool],
    records: list[RunRecord],
    reference_type: str | None,
    *,
    f1_fraca_min: float | None = None,
) -> float | None:
    idx = [
        i
        for i, r in enumerate(records)
        if referencia_aceitavel(r, reference_type, f1_fraca_min=f1_fraca_min) is True
    ]
    if not idx:
        return None
    fp = sum(1 for i in idx if flags[i])
    return fp / len(idx)


def compare_aggregation_policies(
    records: list[RunRecord],
    *,
    verify_gold: bool,
    verify_embedding: bool,
    verify_judge: bool,
    negative_judge_verdicts: list[str],
    judge_aggregation_verdicts: list[str] | None = None,
    policies: tuple[str, ...] = ("qualquer_critico", "embedding_e_juiz"),
    reference_type: str | None = "answer_lists",
    f1_fraca_min: float | None = None,
) -> dict[str, object]:
    """Compara taxas de alerta e FP em referência aceitável entre políticas (offline).

    Para ``reference_type=lexical``, usa overlap léxico (``referencia_incorreta``),
    não ``gold_correto`` booleano (sempre null em datasets léxicos).
    """
    labeled = [
        r
        for r in records
        if referencia_problematica(r, reference_type, f1_fraca_min=f1_fraca_min) is not None
    ]
    n_acceptable = sum(
        1
        for r in labeled
        if referencia_aceitavel(r, reference_type, f1_fraca_min=f1_fraca_min) is True
    )
    out_policies: dict[str, object] = {}
    for pol in policies:
        flags = replay_anomaly_flags(
            records,
            verify_gold=verify_gold,
            verify_embedding=verify_embedding,
            verify_judge=verify_judge,
            negative_judge_verdicts=negative_judge_verdicts,
            judge_aggregation_verdicts=judge_aggregation_verdicts,
            policy=pol,
        )
        n_flag = sum(flags)
        tp = sum(
            1
            for r, f in zip(records, flags, strict=True)
            if referencia_problematica(r, reference_type, f1_fraca_min=f1_fraca_min) is True and f
        )
        fp = sum(
            1
            for r, f in zip(records, flags, strict=True)
            if referencia_aceitavel(r, reference_type, f1_fraca_min=f1_fraca_min) is True and f
        )
        fn = sum(
            1
            for r, f in zip(records, flags, strict=True)
            if referencia_problematica(r, reference_type, f1_fraca_min=f1_fraca_min) is True
            and not f
        )
        kappa = cohen_kappa(tp, fn, fp, n_acceptable - fp) if labeled else None
        out_policies[pol] = {
            "n_anomalias": n_flag,
            "taxa_alerta": (n_flag / len(records)) if records else None,
            "taxa_falso_alarme_no_gold_correto": _fp_rate_on_reference_acceptable(
                flags, records, reference_type, f1_fraca_min=f1_fraca_min
            ),
            "cohen_kappa_anomalia_vs_gold": kappa,
        }
    or_pol = out_policies.get("qualquer_critico")
    and_pol = out_policies.get("embedding_e_juiz")
    baseline_fp: float | None = None
    mit_fp: float | None = None
    if isinstance(or_pol, dict):
        v = or_pol.get("taxa_falso_alarme_no_gold_correto")
        baseline_fp = float(v) if isinstance(v, int | float) else None
    if isinstance(and_pol, dict):
        v = and_pol.get("taxa_falso_alarme_no_gold_correto")
        mit_fp = float(v) if isinstance(v, int | float) else None
    reducao: float | None = None
    if baseline_fp is not None and mit_fp is not None and baseline_fp > 0:
        reducao = (baseline_fp - mit_fp) / baseline_fp
    rotulo = (
        "overlap_lexico_aceitavel"
        if reference_type == "lexical"
        else "gold_correct booleano"
        if reference_type == "answer_lists"
        else "sem_referencia"
    )
    return {
        "versao_esquema": "1",
        "tipo_referencia": reference_type,
        "rotulo_referencia_aceitavel": rotulo,
        "n_itens": len(records),
        "n_referencia_aceitavel": n_acceptable,
        "n_gold_corretos": n_acceptable,
        "politicas": out_policies,
        "reducao_fp_relativa_embedding_e_juiz_vs_or": reducao,
    }

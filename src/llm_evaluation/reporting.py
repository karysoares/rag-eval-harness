"""Escrita de sumários JSONL/CSV e `summary.json`."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path

from llm_evaluation.pattern_registry import build_pattern_settings
from llm_evaluation.reference_metrics import referencia_incorreta
from llm_evaluation.run_artifacts import atomic_write_json, atomic_write_text
from llm_evaluation.schema_registry import PREDICTIONS_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
from llm_evaluation.statistics import cohen_kappa, wilson_ci
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.aggregate import signals_to_dict


def ensure_run_dir(base: Path) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"run_{stamp}"
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"run_{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _meta_int(ctx: dict[str, object], key: str) -> int:
    v = ctx.get(key)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def record_to_json(
    record: RunRecord,
    *,
    include_judge_cot: bool = False,
) -> dict[str, object]:
    sig = signals_to_dict(record.signals, include_judge_cot=include_judge_cot)
    refs = record.meta.get("referencias") or record.meta.get("references") or []
    diag = record.meta.get("diagnostico") or {}
    out: dict[str, object] = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "id_item": record.item_id,
        "pergunta": record.question,
        "resposta": record.answer,
        "gold_correto": record.gold_correct,
        "flag_anomalia": record.anomaly_flag,
        "perfil_baseline": record.baseline_profile,
        "sinais": sig,
        "recuperados": [
            {"texto": c.text, "pontuacao": c.score, "e_ouro": c.is_gold} for c in record.retrieved
        ],
        "meta": record.meta,
    }
    if refs:
        out["referencias"] = refs
    if isinstance(diag, dict) and diag:
        out["diagnostico"] = diag
    return out


def _jsonl_lines(records: list[RunRecord], *, anomalies_only: bool = False) -> str:
    lines: list[str] = []
    for r in records:
        if anomalies_only and not r.anomaly_flag:
            continue
        lines.append(json.dumps(record_to_json(r), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def write_anomalies_jsonl(records: list[RunRecord], path: Path) -> None:
    atomic_write_text(path, _jsonl_lines(records, anomalies_only=True))


def write_anomalies_csv(records: list[RunRecord], path: Path) -> None:
    rows = [r for r in records if r.anomaly_flag]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id_item", "pergunta", "resposta", "gold_correto", "perfil_baseline"])
    for r in rows:
        w.writerow([r.item_id, r.question, r.answer, r.gold_correct, r.baseline_profile])
    atomic_write_text(path, buf.getvalue())


def _protocol_verdict_list(
    protocol: dict[str, object] | None,
    key: str,
) -> list[str] | None:
    if not protocol:
        return None
    raw = protocol.get(key)
    if not isinstance(raw, list):
        return None
    return [str(x) for x in raw]


def _protocol_f1_fraca_min(protocol: dict[str, object] | None) -> float:
    settings = build_pattern_settings()
    if not protocol:
        return settings.f1_fraca_min
    raw = protocol.get("pattern_settings")
    if isinstance(raw, dict):
        f1 = raw.get("f1_fraca_min")
        if isinstance(f1, int | float):
            return float(f1)
    return settings.f1_fraca_min


def summarize(
    records: list[RunRecord],
    *,
    reference_type: str = "answer_lists",
    protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    n = len(records)
    f1_min = _protocol_f1_fraca_min(protocol)
    gold_incorrect = [
        r for r in records if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is True
    ]
    gold_correct = [
        r for r in records if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is False
    ]
    flagged = [r for r in records if r.anomaly_flag]
    lexical_primary = reference_type in ("lexical", "none")

    def recall(flags: list[RunRecord], pool: list[RunRecord]) -> float | None:
        if not pool:
            return None
        hit = sum(1 for r in pool if r.anomaly_flag)
        return hit / len(pool)

    def false_alarm(flags: list[RunRecord], pool: list[RunRecord]) -> float | None:
        if not pool:
            return None
        fa = sum(1 for r in pool if r.anomaly_flag)
        return fa / len(pool)

    # Confusion: anomaly_flag = detector positive; referência fraca = reference positive
    labeled = [
        r
        for r in records
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is not None
    ]
    tp = sum(
        1
        for r in labeled
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is True and r.anomaly_flag
    )
    fn = sum(
        1
        for r in labeled
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is True
        and not r.anomaly_flag
    )
    fp = sum(
        1
        for r in labeled
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is False and r.anomaly_flag
    )
    tn = sum(
        1
        for r in labeled
        if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is False
        and not r.anomaly_flag
    )

    def _estrategia_fp_gold_correto() -> dict[str, object]:
        fp_recs = [
            r
            for r in labeled
            if referencia_incorreta(r, reference_type, f1_fraca_min=f1_min) is False
            and r.anomaly_flag
        ]
        m = len(fp_recs)
        if not m:
            return {
                "n_fp_gold_correto": 0,
                "nota": (
                    "FP = gold automático correto e anomalia "
                    "(em geral embedding e/ou juiz, não o sinal ouro)."
                ),
            }

        def emb_on(r: RunRecord) -> bool:
            return r.signals.embedding_low_support is True

        from llm_evaluation.evaluation_metrics import resolve_judge_verdict_lists
        from llm_evaluation.verification.aggregate import judge_negative_for_aggregation

        neg_raw = protocol.get("negative_judge_verdicts") if protocol else None
        agg_raw = protocol.get("judge_aggregation_verdicts") if protocol else None
        neg_list = [str(x) for x in neg_raw] if isinstance(neg_raw, list) else None
        agg_list = [str(x) for x in agg_raw] if isinstance(agg_raw, list) else None
        _negs, agg_verdicts = resolve_judge_verdict_lists(
            negative_judge_verdicts=neg_list,
            judge_aggregation_verdicts=agg_list,
        )

        def judge_on(r: RunRecord) -> bool:
            return judge_negative_for_aggregation(r.signals, agg_verdicts)

        so_emb = sum(1 for r in fp_recs if emb_on(r) and not judge_on(r))
        so_juiz = sum(1 for r in fp_recs if judge_on(r) and not emb_on(r))
        ambos = sum(1 for r in fp_recs if emb_on(r) and judge_on(r))
        nenhum = sum(1 for r in fp_recs if not emb_on(r) and not judge_on(r))
        curadas = sum(
            1
            for r in fp_recs
            if isinstance(r.meta.get("qualidade_geracao"), dict)
            and bool(r.meta["qualidade_geracao"].get("curada_por_recuperacao_fraca"))
        )
        return {
            "n_fp_gold_correto": m,
            "com_so_embedding_baixo": so_emb,
            "com_so_juiz_negativo": so_juiz,
            "com_embedding_e_juiz": ambos,
            "sem_embedding_nem_juiz_negativo": nenhum,
            "dos_quais_resposta_curada_por_gate_recuperacao": curadas,
            "nota": (
                "FP = gold correto e anomalia "
                "(tipicamente baixa ancoragem ao contexto ou juiz negativo)."
            ),
        }

    n_curadas_rec = sum(
        1
        for r in records
        if isinstance(r.meta.get("qualidade_geracao"), dict)
        and bool(r.meta["qualidade_geracao"].get("curada_por_recuperacao_fraca"))
    )
    denom_prec = tp + fp
    denom_recall = tp + fn
    precision_flag = (tp / denom_prec) if denom_prec else None
    recall_flag = (tp / denom_recall) if denom_recall else None
    denom_acc = tp + tn + fp + fn
    accuracy_balanced_ref = (
        0.5 * (tp / (tp + fn) if (tp + fn) else 0.0) + 0.5 * (tn / (tn + fp) if (tn + fp) else 0.0)
        if denom_acc
        else None
    )
    n_unlabeled_gold = n - len(labeled)

    from llm_evaluation.evaluation_metrics import layer_analysis

    def _retrieval_summary() -> dict[str, object] | None:
        scores: list[float] = []
        ranks: list[int] = []
        n_rag = 0
        n_gold_in_top = 0
        n_with_gold_corpus = 0
        for r in records:
            rm = r.meta.get("metricas_recuperacao") or r.meta.get("retrieval_metrics")
            if not isinstance(rm, dict) or not rm.get("rag_ativo"):
                continue
            n_rag += 1
            sc = rm.get("score_melhor_chunk")
            if sc is not None:
                scores.append(float(sc))
            if rm.get("chunk_ouro_no_top_k"):
                n_gold_in_top += 1
            if rm.get("corpus_tem_chunk_ouro"):
                n_with_gold_corpus += 1
            rk = rm.get("rank_chunk_ouro")
            if rk is not None:
                ranks.append(int(rk))

        if n_rag == 0:
            return None

        def mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        return {
            "n_itens_com_rag": n_rag,
            "media_score_melhor_chunk": mean(scores),
            "taxa_chunk_ouro_no_top_k": (
                (n_gold_in_top / n_with_gold_corpus) if n_with_gold_corpus else None
            ),
            "n_itens_com_chunk_ouro_no_corpus": n_with_gold_corpus,
            "media_rank_chunk_ouro_quando_presente": mean([float(x) for x in ranks]),
        }

    def _lexical_summary() -> dict[str, object] | None:
        bleu_vals: list[float] = []
        rouge_f: list[float] = []
        met: list[float] = []
        lev: list[float] = []
        n_scored = 0
        n_em = 0
        n_em_norm = 0
        n_em_squad = 0
        f1_vals: list[float] = []
        for r in records:
            lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics")
            if not isinstance(lm, dict):
                continue
            if lm.get("note") == "metricas_lexicas_desligadas":
                continue
            if lm.get("note") == "sem_referencia":
                continue
            if lm.get("bleu") is not None:
                bleu_vals.append(float(lm["bleu"]))
            rf = lm.get("rouge_l_f")
            if rf is None:
                rf = lm.get("rouge_l_fmeasure")
            if rf is not None:
                rouge_f.append(float(rf))
            if lm.get("meteor") is not None:
                met.append(float(lm["meteor"]))
            sl = lm.get("similaridade_levenshtein")
            if sl is None:
                sl = lm.get("levenshtein_similarity")
            if sl is not None:
                lev.append(float(sl))
            tr = lm.get("texto_referencia") or lm.get("reference_text") or ""
            if str(tr).strip():
                n_scored += 1
            if lm.get("exact_match") is True:
                n_em += 1
            if lm.get("exact_match_normalizado") is True:
                n_em_norm += 1
            if lm.get("em_squad") is True:
                n_em_squad += 1
            f1t = lm.get("f1_token")
            if f1t is not None:
                f1_vals.append(float(f1t))

        def mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        if not any([bleu_vals, rouge_f, met, lev, f1_vals]) and n_scored == 0:
            return None
        out_lex: dict[str, object] = {
            "n_itens_pontuados": n_scored,
            "media_bleu": mean(bleu_vals),
            "media_rouge_l_f": mean(rouge_f),
            "media_meteor": mean(met),
            "media_similaridade_levenshtein": mean(lev),
            "media_f1_token": mean(f1_vals),
        }
        if n_scored:
            out_lex["taxa_exact_match"] = n_em / n_scored
            out_lex["taxa_exact_match_normalizado"] = n_em_norm / n_scored
        if f1_vals:
            out_lex["taxa_em_squad"] = n_em_squad / len(f1_vals)
        return out_lex

    n_gi_total = len(gold_incorrect)
    n_gc_total = len(gold_correct)
    flagged_in_gi = sum(1 for r in gold_incorrect if r.anomaly_flag)
    flagged_in_gc = sum(1 for r in gold_correct if r.anomaly_flag)

    def _judge_summary() -> dict[str, object] | None:
        n_juiz = sum(1 for r in records if r.signals.judge is not None)
        if n_juiz == 0:
            return None
        fallbacks = sum(
            1 for r in records if r.signals.judge and r.signals.judge.raw.get("fallback_heuristico")
        )
        ctx_rows: list[dict[str, object]] = []
        for r in records:
            ctx = r.meta.get("contexto_juiz")
            if isinstance(ctx, dict):
                ctx_rows.append(ctx)
        retries = [_meta_int(c, "retry_count") for c in ctx_rows]
        parse_fail = [_meta_int(c, "parse_failures") for c in ctx_rows]
        schema_inv = [bool(c.get("schema_invalid")) for c in ctx_rows]
        used_fb_ctx = [bool(c.get("used_fallback")) for c in ctx_rows]
        tokens = [
            _meta_int(c, "tokens_estimados")
            for c in ctx_rows
            if c.get("tokens_estimados") is not None
        ]
        trunc = [bool(c.get("truncado")) for c in ctx_rows if "truncado" in c]
        n_ctx = len(ctx_rows)
        n_incompleto = sum(
            1 for r in records if r.signals.judge and r.signals.judge.veredito == "incompleto"
        )
        return {
            "n_itens_com_juiz": n_juiz,
            "n_veredito_incompleto": n_incompleto,
            "taxa_veredito_incompleto": n_incompleto / n_juiz if n_juiz else None,
            "taxa_fallback_heuristico": fallbacks / n_juiz if n_juiz else None,
            "taxa_used_fallback_meta": (
                sum(1 for x in used_fb_ctx if x) / n_ctx if n_ctx else None
            ),
            "n_com_contexto_juiz_meta": n_ctx,
            "taxa_schema_invalido": (sum(1 for x in schema_inv if x) / n_ctx if n_ctx else None),
            "taxa_com_retry": (sum(1 for x in retries if x > 0) / n_ctx if n_ctx else None),
            "media_retry_count": (sum(retries) / n_ctx if n_ctx else None),
            "media_parse_failures": (sum(parse_fail) / n_ctx if n_ctx else None),
            "media_tokens_contexto_estimados": (sum(tokens) / len(tokens) if tokens else None),
            "taxa_contexto_truncado": (sum(1 for x in trunc if x) / len(trunc) if trunc else None),
            "schema_version": ctx_rows[0].get("schema_version") if ctx_rows else None,
        }

    def _gap_rag_resposta_summary() -> dict[str, object] | None:
        """Recuperação forte mas resposta léxica muito fraca sem alerta (diagnóstico E2E)."""
        from llm_evaluation.operational import thresholds_from_mapping

        thr = thresholds_from_mapping(protocol)
        n_gap = 0
        for r in records:
            if r.anomaly_flag:
                continue
            rm = r.meta.get("metricas_recuperacao") or r.meta.get("retrieval_metrics")
            lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics")
            if not isinstance(rm, dict) or not isinstance(lm, dict):
                continue
            if not rm.get("chunk_ouro_no_top_k"):
                continue
            sc = rm.get("score_melhor_chunk")
            f1 = lm.get("f1_token")
            if sc is None or f1 is None:
                continue
            if float(sc) >= thr.gap_min_score_recuperacao and float(f1) < thr.gap_max_f1_token:
                n_gap += 1
        if n_gap == 0:
            return None
        return {
            "n_gap_recuperacao_forte_resposta_fraca": n_gap,
            "taxa_gap": n_gap / len(records) if records else None,
            "limiares": {
                "gap_min_score_recuperacao": thr.gap_min_score_recuperacao,
                "gap_max_f1_token": thr.gap_max_f1_token,
            },
            "nota": (
                f"Itens com ouro no top-k, score≥{thr.gap_min_score_recuperacao} "
                f"e F1<{thr.gap_max_f1_token} sem flag_anomalia — paráfrase/recusa."
            ),
        }

    def _pattern_summary() -> dict[str, object] | None:
        from llm_evaluation.pattern_registry import PATTERN_CATALOG_VERSION

        by_primary: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        catalog_version: str | None = None
        for r in records:
            d = r.meta.get("diagnostico")
            if not isinstance(d, dict):
                continue
            if catalog_version is None and d.get("catalog_version"):
                catalog_version = str(d["catalog_version"])
            prim = str(d.get("padrao_primario") or "ok")
            by_primary[prim] = by_primary.get(prim, 0) + 1
            tags = d.get("padroes")
            if isinstance(tags, list):
                for t in tags:
                    k = str(t)
                    by_tag[k] = by_tag.get(k, 0) + 1
        if not by_primary:
            return None
        return {
            "catalog_version": catalog_version or PATTERN_CATALOG_VERSION,
            "por_padrao_primario": by_primary,
            "por_tag": by_tag,
        }

    lex_sum = _lexical_summary()
    ret_sum = _retrieval_summary()
    pat_sum = _pattern_summary()
    judge_sum = _judge_summary()
    gap_sum = _gap_rag_resposta_summary()

    from llm_evaluation.observability import summarize_run_observability

    obs_sum = summarize_run_observability(records)

    def _operational_summary() -> dict[str, object]:
        from llm_evaluation.evaluation_metrics import replay_anomaly_flags
        from llm_evaluation.fila_revisao import count_fila_records
        from llm_evaluation.operational import thresholds_from_mapping

        proto = protocol or {}
        thr = thresholds_from_mapping(proto)
        agg_list_proto = proto.get("judge_aggregation_verdicts")
        juiz_fila = (
            [str(x) for x in agg_list_proto]
            if isinstance(agg_list_proto, list) and agg_list_proto
            else []
        )
        fila = count_fila_records(
            records,
            juiz_vereditos_fila=juiz_fila,
            min_score_recuperacao=thr.fila_min_score_recuperacao,
        )
        neg = proto.get("negative_judge_verdicts")
        agg = proto.get("judge_aggregation_verdicts")
        neg_list = [str(x) for x in neg] if isinstance(neg, list) else None
        agg_list = [str(x) for x in agg] if isinstance(agg, list) else None
        n_flag = len(flagged)
        replay: dict[str, int] = {}
        if neg_list and proto.get("aggregation_policy"):
            for pol in ("qualquer_critico", "embedding_e_juiz"):
                flags = replay_anomaly_flags(
                    records,
                    verify_gold=bool(proto.get("verify_gold")),
                    verify_embedding=bool(proto.get("verify_embedding")),
                    verify_judge=bool(proto.get("verify_judge")),
                    negative_judge_verdicts=neg_list,
                    judge_aggregation_verdicts=agg_list,
                    policy=pol,
                )
                replay[pol] = sum(flags)
        return {
            "n_alerta_politica_gravada": n_flag,
            "taxa_alerta_gravada": (n_flag / n) if n else None,
            "fila_revisao_humana": fila,
            "replay_n_anomalias": replay,
            "nota_interpretacao": (
                "KPI de produto: sumario_lexical (METEOR/F1/tiers). "
                "n_anomalias_marcadas reflecte só a política YAML activa — "
                "0 alertas não implica qualidade perfeita. "
                "Use fila_revisao_humana.csv para revisão obrigatória quando alerta=0."
            ),
        }

    op_sum = _operational_summary()

    camadas: list[str] = []
    if protocol:
        if protocol.get("verify_gold"):
            camadas.append("gold")
        if protocol.get("verify_embedding"):
            camadas.append("embedding")
        if protocol.get("verify_judge"):
            camadas.append("juiz")

    kpi_primary = "sumario_lexical" if lexical_primary else "confusao_vs_referencia"
    out: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tipo_referencia_ativo": reference_type,
        "kpi_primario": kpi_primary,
        "kpi_diagnostico_primario": kpi_primary,
        "detector_activo": {
            "politica_agregacao": (protocol or {}).get("aggregation_policy"),
            "camadas_verificacao": camadas,
            "embedding_min_cosine": (protocol or {}).get("embedding_min_cosine"),
            "negative_judge_verdicts": (protocol or {}).get("negative_judge_verdicts"),
            "judge_aggregation_verdicts": (protocol or {}).get("judge_aggregation_verdicts"),
            "nota": (
                "flag_anomalia reflecte a política de alerta activa, "
                "não a qualidade global do chatbot."
            ),
        },
        "sumario_operacional": op_sum,
        "taxa_alerta": (len(flagged) / n) if n else None,
        "n_itens": n,
        "n_com_gold_para_confusao": len(labeled),
        "n_sem_rotulo_gold": n_unlabeled_gold,
        "n_anomalias_marcadas": len(flagged),
        "n_gold_incorretos": n_gi_total,
        "n_gold_corretos": n_gc_total,
        "revocacao_marcacao_no_gold_incorreto": recall(flagged, gold_incorrect),
        "ic95_revocacao_marcacao_no_gold_incorreto": (
            wilson_ci(flagged_in_gi, n_gi_total) if n_gi_total else None
        ),
        "taxa_falso_alarme_no_gold_correto": false_alarm(flagged, gold_correct),
        "ic95_taxa_falso_alarme_no_gold_correto": (
            wilson_ci(flagged_in_gc, n_gc_total) if n_gc_total else None
        ),
        "confusao_vs_referencia": {
            "vp_referencia_incorreta_marcado": tp,
            "fn_referencia_incorreta_nao_marcado": fn,
            "fp_referencia_correta_mas_marcado": fp,
            "vn_referencia_correta_nao_marcado": tn,
        },
        "confusao_vs_gold": {
            "vp_gold_incorreto_marcado": tp,
            "fn_gold_incorreto_nao_marcado": fn,
            "fp_gold_correto_mas_marcado": fp,
            "vn_gold_correto_nao_marcado": tn,
        },
        "estratificacao_fp_gold_correto": _estrategia_fp_gold_correto(),
        "qualidade_pipeline": {
            "n_geracoes_curadas_recuperacao_fraca": n_curadas_rec,
            "taxa_geracao_curada": (n_curadas_rec / n) if n else None,
        },
        "precisao_anomalia_vs_gold_incorreto": precision_flag,
        "ic95_precisao_anomalia_vs_gold_incorreto": (
            wilson_ci(tp, denom_prec) if denom_prec else None
        ),
        "revocacao_anomalia_vs_gold_incorreto": recall_flag,
        "ic95_revocacao_anomalia_vs_gold_incorreto": (
            wilson_ci(tp, denom_recall) if denom_recall else None
        ),
        "acuracia_balanceada_gold": accuracy_balanced_ref,
        "cohen_kappa_anomalia_vs_gold": cohen_kappa(tp, fn, fp, tn),
        "analise_camadas": layer_analysis(
            records,
            reference_type=reference_type,
            f1_fraca_min=f1_min,
            negative_judge_verdicts=_protocol_verdict_list(protocol, "negative_judge_verdicts"),
            judge_aggregation_verdicts=_protocol_verdict_list(
                protocol,
                "judge_aggregation_verdicts",
            ),
        ),
    }
    if lexical_primary:
        out["aviso_metricas"] = (
            "reference_type lexical/none: use sumario_lexical como KPI principal. "
            "confusao_vs_gold e n_gold_* são diagnóstico (substring), não o objectivo da corrida."
        )
    if ret_sum is not None:
        out["sumario_recuperacao"] = ret_sum
    if lex_sum is not None:
        out["sumario_lexical"] = lex_sum
    if pat_sum is not None:
        out["sumario_padroes"] = pat_sum
    if judge_sum is not None:
        out["sumario_juiz"] = judge_sum
    if gap_sum is not None:
        out["sumario_gap_rag_resposta"] = gap_sum
    if obs_sum is not None:
        out["observabilidade"] = obs_sum

    from llm_evaluation.explainability import summarize_explicabilidade
    from llm_evaluation.hitl_metrics import summarize_hitl

    fila_n = None
    if isinstance(op_sum, dict):
        fr = op_sum.get("fila_revisao_humana")
        if isinstance(fr, dict):
            fila_n = fr.get("total")
    fila_total = fila_n if isinstance(fila_n, int) else None
    hitl_sum = summarize_hitl(records, protocol=protocol, fila_total=fila_total)
    if hitl_sum is not None:
        out["sumario_hitl"] = hitl_sum
    xai_sum = summarize_explicabilidade(records)
    if xai_sum is not None:
        out["sumario_explicabilidade"] = xai_sum

    avisos: list[str] = []
    if protocol:
        models = protocol.get("models")
        if isinstance(models, dict) and models.get("judge_same_as_generator"):
            llm_m = models.get("llm_model")
            avisos.append(
                f"JUDGE_MODEL igual a LLM_MODEL ({llm_m!r}): auto-referência do juiz; "
                "defina JUDGE_MODEL distinto para avaliação válida."
            )
    if avisos:
        out["avisos_protocolo"] = avisos
    return out


def write_summary(summary: dict[str, object], path: Path) -> None:
    if "schema_version" not in summary and "baselines" not in summary:
        summary = {**summary, "schema_version": SUMMARY_SCHEMA_VERSION}
    atomic_write_json(path, summary)


def write_baseline_comparison(
    summaries: dict[str, dict[str, object]],
    path: Path,
) -> None:
    atomic_write_json(path, summaries)

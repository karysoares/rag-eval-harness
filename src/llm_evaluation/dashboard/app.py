"""Dashboard Streamlit: `uv run llm-eval-dashboard`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from llm_evaluation.dashboard.data import (
    CALIBRATION_COLUMN_ORDER,
    artifact_fingerprint,
    cache_run_artifacts,
    calibration_view_dataframe,
    compare_runs,
    list_run_dirs,
    load_summary_json,
    outputs_root,
    records_to_dataframe,
)
from llm_evaluation.types import RunRecord

_INSPECTOR_PAGE_SIZE = 50


def main() -> None:
    st.set_page_config(
        page_title="llm-evaluation | Análise de corridas",
        layout="wide",
    )
    st.title("Análise de métricas de avaliação")
    st.caption(
        "Visualização de artefactos `llm-eval` (sem executar pipeline aqui).",
    )

    root = outputs_root()
    runs = list_run_dirs(root)
    if not runs:
        st.warning(
            f"Nenhuma corrida em `{root}`. "
            "Execute: `uv run llm-eval --config configs/default.yaml`",
        )
        return

    with st.sidebar:
        st.header("Corridas")
        run_names = [p.name for p in runs]
        selected = st.selectbox("Corrida", run_names, index=0)
        run_dir = root / selected
        st.caption(str(run_dir))

        st.divider()
        compare_mode = st.checkbox("Comparar várias corridas", value=False)
        compare_pick: list[str] = []
        if compare_mode:
            compare_pick = st.multiselect("Corridas a comparar", run_names, default=run_names[:2])

    try:
        bundle = _load_run_bundle(str(run_dir.resolve()), artifact_fingerprint(run_dir))
    except FileNotFoundError as e:
        st.error(str(e))
        return

    report = bundle["report"]
    records = bundle["records"]
    df = records_to_dataframe(records)
    integrity = bundle["integrity"]
    validation = bundle["validation"]

    with st.sidebar:
        _render_integrity_badges(integrity, validation)
        from llm_evaluation.dashboard.facade import MetricMode, hitl_progress, kpi_blocks_for_mode

        metric_mode = st.radio(
            "Métricas",
            [MetricMode.AUTOMATICO.value, MetricMode.POS_HITL.value, MetricMode.COMPARAR.value],
            index=0,
            key="metric_mode",
        )
        mode = MetricMode(metric_mode)
        prog = hitl_progress(run_dir)
        if prog["fila_total"]:
            st.progress(
                min(1.0, prog["rotulados"] / prog["fila_total"]) if prog["fila_total"] else 0.0,
                text=f"HITL: {prog['rotulados']}/{prog['fila_total']}",
            )
        kpi_view = kpi_blocks_for_mode(report, mode)

    layer = report.get("analise_camadas") or report.get("layer_analysis") or {}

    tab_overview, tab_fila, tab_cal, tab_qa, tab_patterns, tab_ret, tab_sig, tab_ref, tab_legacy = (
        st.tabs(
            [
                "Visão geral",
                "Revisão humana",
                "Calibração",
                "Inspector Q/A",
                "Padrões",
                "Recuperação",
                "Sinais",
                "Referência",
                "Inspector (JSON)",
            ],
        )
    )

    with tab_overview:
        _render_overview(report, layer, df, kpi_view=kpi_view, metric_mode=mode)

    with tab_fila:
        _render_fila_revisao(report, run_dir, metric_mode=mode, records=records)

    with tab_cal:
        _render_calibration(report, df, records, run_dir)

    with tab_qa:
        _render_qa_inspector(df, records)

    with tab_patterns:
        _render_patterns(report, df)

    with tab_ret:
        _render_retrieval(report, df)

    with tab_sig:
        _render_signals(layer, report)

    with tab_ref:
        _render_reference(report, df)

    with tab_legacy:
        _render_legacy_inspector(df, records)

    if compare_mode and len(compare_pick) >= 2:
        st.divider()
        st.subheader("Comparação de corridas")
        dirs = [root / n for n in compare_pick]
        cmp = compare_runs(dirs)
        st.dataframe(pd.DataFrame(cmp.get("corridas", [])), use_container_width=True)


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _fmt_pct(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{100 * float(v):.1f}%"
    return str(v)


def _chunk_text_area(
    text: str,
    *,
    key: str,
    height: int | None = None,
) -> None:
    """Texto completo do chunk (sem truncar); altura adaptada ao tamanho."""
    n = len(text)
    h = height if height is not None else min(480, max(140, n // 2))
    st.text_area(
        "texto",
        value=text,
        height=h,
        disabled=True,
        label_visibility="collapsed",
        key=key,
    )


def _render_retrieved_chunks(rec: RunRecord, *, key_prefix: str = "qa") -> None:
    """Chunks e passagem gold completos (sem limite de 300 caracteres)."""
    gold_pass = rec.meta.get("passagem_ouro_rag") or rec.meta.get("rag_gold_chunk")
    if gold_pass:
        st.markdown("### Passagem gold (dataset)")
        st.caption(f"{len(str(gold_pass))} caracteres · fonte: `meta.passagem_ouro_rag`")
        _chunk_text_area(str(gold_pass), key=f"{key_prefix}_gold_{rec.item_id}")
    elif any(c.is_gold for c in rec.retrieved):
        st.caption(
            "Passagem gold completa não está no JSONL (corrida antiga). "
            "Reexecute `llm-eval` ou veja as partes abaixo."
        )

    if not rec.retrieved:
        st.caption("Nenhum chunk recuperado.")
        return

    n = len(rec.retrieved)
    st.markdown(f"### Chunks recuperados ({n})")
    st.caption(
        "Cada entrada é um bloco do corpus (máx. `chunk_max_chars` no YAML, tipicamente 400). "
        "Vários blocos `ouro=True` são a mesma passagem partida."
    )

    gold_parts = [c for c in rec.retrieved if c.is_gold]
    if len(gold_parts) > 1:
        with st.expander(
            f"Ver passagem gold reconstituída ({len(gold_parts)} partes)",
            expanded=True,
        ):
            blocks = []
            for i, c in enumerate(gold_parts, 1):
                blocks.append(
                    f"--- Parte {i} · score={c.score:.3f} · {len(c.text)} chars ---\n{c.text}",
                )
            _chunk_text_area(
                "\n\n".join(blocks),
                key=f"{key_prefix}_gold_join_{rec.item_id}",
                height=min(520, 120 * len(gold_parts)),
            )

    for i, c in enumerate(rec.retrieved, 1):
        label = f"Chunk {i} · score={c.score:.3f} · {len(c.text)} chars"
        if c.is_gold:
            label += " · **OURO**"
        with st.expander(label, expanded=(i == 1 and c.is_gold)):
            _chunk_text_area(c.text, key=f"{key_prefix}_chunk_{rec.item_id}_{i}")


def _active_layers_from_report(report: dict[str, Any]) -> list[str]:
    det = report.get("detector_activo") or {}
    camadas = det.get("camadas_verificacao")
    if isinstance(camadas, list) and camadas:
        return [str(c) for c in camadas]
    proto = report.get("protocolo_ativo") or {}
    if isinstance(proto, dict):
        out: list[str] = []
        if proto.get("verify_embedding"):
            out.append("embedding")
        if proto.get("verify_judge"):
            out.append("juiz")
        if proto.get("verify_gold"):
            out.append("gold")
        return out
    return []


def _active_reference_type(
    report: dict[str, Any],
    layer: dict[str, Any] | None = None,
) -> str:
    """``lexical`` | ``answer_lists`` | ``none`` — summary primeiro, depois ``analise_camadas``."""
    layer = layer or {}
    return str(report.get("tipo_referencia_ativo") or layer.get("tipo_referencia") or "")


def _render_operational_kpi(report: dict[str, Any]) -> None:
    op = report.get("sumario_operacional")
    if not isinstance(op, dict) or not op:
        return
    st.subheader("Operacional: KPI vs detector")
    nota = op.get("nota_interpretacao")
    if nota:
        st.info(str(nota))
    c1, c2, c3 = st.columns(3)
    c1.metric("Alertas (política gravada)", op.get("n_alerta_politica_gravada"))
    c2.metric("Taxa alerta", _fmt_pct(op.get("taxa_alerta_gravada")))
    fila = op.get("fila_revisao_humana") or {}
    if isinstance(fila, dict):
        c3.metric("Fila revisão humana", fila.get("total"))
    replay = op.get("replay_n_anomalias")
    if isinstance(replay, dict) and replay:
        with st.expander("Replay offline de políticas"):
            st.json(replay)


def _render_fila_revisao(
    report: dict[str, Any],
    run_dir: Path,
    *,
    metric_mode: Any = None,
    records: list[RunRecord] | None = None,
) -> None:
    from llm_evaluation.dashboard.data import load_fila_revisao_dataframe
    from llm_evaluation.dashboard.facade import (
        MetricMode,
        apply_hitl_labels,
        hitl_csv_path,
        load_hitl_labels,
        save_hitl_annotation,
    )
    from llm_evaluation.hitl_io import ROTULOS_DISPLAY, VALID_ROTULOS

    _render_operational_kpi(report)
    hitl = report.get("sumario_hitl")
    if isinstance(hitl, dict) and metric_mode == MetricMode.POS_HITL:
        st.subheader("Métricas pós-HITL (amostra)")
        st.json(hitl)

    try:
        labels = load_hitl_labels(run_dir)
    except Exception as e:  # noqa: BLE001 - manter UI operável em falha transitória de I/O
        st.error(f"Falha ao carregar rótulos HITL: {e}")
        labels = {}
    if "hitl_revisor" not in st.session_state:
        st.session_state["hitl_revisor"] = ""

    with st.expander("Importar / exportar CSV", expanded=False):
        uploaded = st.file_uploader(
            "Importar adjudicacoes_hitl.csv",
            type=["csv"],
            key="hitl_upload",
        )
        if uploaded is not None:
            from llm_evaluation.hitl_io import commit_staged_hitl_csv, write_staged_hitl_csv

            write_staged_hitl_csv(run_dir, uploaded.getvalue())
            st.caption("CSV em staging — clique Aplicar para validar e gravar.")
            if st.button("Aplicar CSV importado", key="hitl_apply_import"):
                try:
                    commit_staged_hitl_csv(run_dir, strict_ids=True)
                    apply_hitl_labels(run_dir, strict_hitl_ids=True)
                except Exception as e:  # noqa: BLE001 - mostrar erro amigável no dashboard
                    st.error(f"Falha ao aplicar CSV: {e}")
                else:
                    st.success("Summary actualizado.")
                    st.rerun()
        csv_out = hitl_csv_path(run_dir)
        if csv_out.is_file():
            st.download_button(
                "Descarregar adjudicacoes_hitl.csv",
                csv_out.read_bytes(),
                file_name="adjudicacoes_hitl.csv",
                mime="text/csv",
                key="hitl_download",
            )

    fila_df = load_fila_revisao_dataframe(run_dir)
    if fila_df is None or fila_df.empty:
        st.warning(
            "Sem `analise_manual/fila_revisao_humana.csv`. "
            "Corra uma corrida recente ou `scripts/export_fila_revisao.py`."
        )
        return

    st.caption(f"{len(fila_df)} itens na fila obrigatória (juiz duro + recusas com RAG forte)")
    motivo = st.multiselect(
        "Motivo",
        sorted(fila_df["motivo_fila"].dropna().unique()),
        default=sorted(fila_df["motivo_fila"].dropna().unique()),
        key="fila_motivo",
    )
    view = fila_df[fila_df["motivo_fila"].isin(motivo)] if motivo else fila_df
    view = view.copy()
    view["rotulo_humano"] = view["id_item"].map(
        lambda i: labels.get(str(i), {}).get("rotulo", "") if i else "",
    )

    st.subheader("Anotar item")
    só_pendentes = st.checkbox("Só itens sem rótulo humano", value=True, key="hitl_so_pendentes")
    ids_fila = view["id_item"].astype(str).tolist()
    if só_pendentes:
        ids_pick = [i for i in ids_fila if not labels.get(i, {}).get("rotulo")]
        if not ids_pick:
            st.info("Todos os itens visíveis já têm rótulo. Desmarque o filtro para editar.")
            ids_pick = ids_fila
    else:
        ids_pick = ids_fila

    if not ids_pick:
        st.warning("Nenhum item disponível para anotar com os filtros actuais.")
    else:
        pick = st.selectbox("Item da fila", ids_pick, key="hitl_annotate_item")
        row = view[view["id_item"].astype(str) == str(pick)].iloc[0]
        prev = labels.get(str(pick), {})
        if prev.get("rotulo"):
            st.caption(
                f"Rótulo actual: **{prev['rotulo']}** "
                f"(revisor: {prev.get('revisor') or '—'}, "
                f"{prev.get('timestamp_utc') or ''})",
            )

        c1, c2 = st.columns([1, 2])
        with c1:
            rotulo_opts = [code for code, _ in ROTULOS_DISPLAY]
            idx_default = 0
            if prev.get("rotulo") in rotulo_opts:
                idx_default = rotulo_opts.index(str(prev["rotulo"]))
            rotulo = st.selectbox(
                "Rótulo humano",
                rotulo_opts,
                index=idx_default,
                format_func=lambda c: dict(ROTULOS_DISPLAY).get(c, c),
                key="hitl_rotulo",
            )
            revisor = st.text_input(
                "Revisor",
                value=st.session_state.get("hitl_revisor") or prev.get("revisor", ""),
                key="hitl_revisor_input",
            )
        with c2:
            st.markdown("**Pergunta**")
            st.write(str(row.get("pergunta", ""))[:2000])
            st.markdown("**Resposta**")
            st.write(str(row.get("resposta", ""))[:2000])
            notas = st.text_area(
                "Notas",
                value=prev.get("notas", ""),
                height=120,
                key="hitl_notas",
            )

        if "explicacao_resumida" in row and row.get("explicacao_resumida"):
            st.caption(f"Explicação: {row['explicacao_resumida']}")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Guardar anotação (CSV)", key="hitl_save_csv", type="primary"):
                if rotulo not in VALID_ROTULOS:
                    st.error("Rótulo inválido.")
                else:
                    try:
                        save_hitl_annotation(
                            run_dir,
                            item_id=str(pick),
                            rotulo=rotulo,
                            revisor=revisor,
                            notas=notas,
                        )
                    except Exception as e:  # noqa: BLE001 - falha não deve quebrar sessão
                        st.error(f"Falha ao gravar anotação: {e}")
                    else:
                        st.session_state["hitl_revisor"] = revisor
                        st.success(f"Anotação gravada para `{pick}`.")
                        st.rerun()
        with b2:
            if st.button("Guardar e aplicar métricas", key="hitl_save_apply"):
                try:
                    save_hitl_annotation(
                        run_dir,
                        item_id=str(pick),
                        rotulo=rotulo,
                        revisor=revisor,
                        notas=notas,
                    )
                    apply_hitl_labels(run_dir)
                except Exception as e:  # noqa: BLE001 - falha operacional deve ser visível
                    st.error(f"Falha ao aplicar anotação: {e}")
                else:
                    st.session_state["hitl_revisor"] = revisor
                    st.success("Anotação aplicada ao JSONL e summary actualizado.")
                    st.rerun()

        if records:
            rec = next((r for r in records if r.item_id == str(pick)), None)
            if rec is not None:
                with st.expander("Contexto recuperado e sinais", expanded=False):
                    _render_retrieved_chunks(rec, key_prefix="hitl_ann")
                    exp = rec.meta.get("explicacao")
                    if isinstance(exp, dict):
                        st.json(exp)

    cols = [
        c
        for c in (
            "motivo_fila",
            "id_item",
            "rotulo_humano",
            "veredito_juiz",
            "f1_token",
            "score_melhor_chunk",
            "flag_anomalia",
            "resposta",
        )
        if c in view.columns
    ]
    st.subheader("Lista da fila")
    st.dataframe(view[cols], use_container_width=True, hide_index=True)


def _render_overview(
    report: dict[str, Any],
    layer: dict[str, Any],
    df: pd.DataFrame,
    *,
    kpi_view: dict[str, Any] | None = None,
    metric_mode: Any = None,
) -> None:
    from llm_evaluation.dashboard.facade import MetricMode, provenance_from_report

    prov = provenance_from_report(report)
    if prov:
        with st.expander("Proveniência", expanded=False):
            st.json(prov)

    if kpi_view and metric_mode == MetricMode.COMPARAR:
        st.subheader("Comparar planos métricos A/B/C")
        st.json(kpi_view)

    proto = report.get("protocolo_ativo")
    if proto:
        st.caption(f"Protocolo activo: `{proto}`")

    op = report.get("sumario_operacional")
    if isinstance(op, dict):
        fila = op.get("fila_revisao_humana")
        if isinstance(fila, dict) and fila.get("total"):
            st.warning(
                f"Fila de revisão humana: {fila['total']} itens — "
                "ver aba «Revisão humana» (0 alertas não implica aprovação)."
            )

    ref_type = _active_reference_type(report, layer)
    active = _active_layers_from_report(report)
    ret = report.get("sumario_recuperacao") or {}
    pat = report.get("sumario_padroes") or {}

    kpi = str(report.get("kpi_primario", "confusao_vs_referencia"))
    lex = report.get("sumario_lexical") or {}
    if kpi == "sumario_lexical" and lex:
        st.info(
            report.get(
                "aviso_metricas",
                (
                    "KPI principal: overlap léxico com a resposta de referência "
                    "(paráfrases contam como «fraco»)."
                ),
            ),
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Itens", report.get("n_itens"))
        c2.metric("F1 token (média)", _fmt(lex.get("media_f1_token")))
        c3.metric("EM SQuAD", _fmt_pct(lex.get("taxa_em_squad")))
        c4.metric("Anomalias (alerta)", report.get("n_anomalias_marcadas"))
        if ret:
            c5.metric("Ouro no top-k", _fmt_pct(ret.get("taxa_chunk_ouro_no_top_k")))
            c6.metric("Score médio top-1", _fmt(ret.get("media_score_melhor_chunk")))
        else:
            c5.metric("Exact match", _fmt_pct(lex.get("taxa_exact_match")))
            c6.empty()

        if pat:
            prim = pat.get("por_padrao_primario") or {}
            if prim:
                st.subheader("Qualidade por padrão (determinístico)")
                st.bar_chart(pd.Series(prim).sort_values(ascending=False))
                st.caption(
                    "«referencia_fraca/ausente» = texto diferente da referência curta, "
                    "não implica resposta errada face ao conto."
                )

        if "tier_qualidade" in df.columns and df["tier_qualidade"].notna().any():
            st.subheader("Tiers de qualidade")
            st.bar_chart(df["tier_qualidade"].value_counts())

        with st.expander("Sumário léxico (detalhe)"):
            st.json(lex)
        if ret:
            with st.expander("Recuperação RAG (detalhe)"):
                st.json(ret)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Itens", report.get("n_itens"))
        c2.metric("Anomalias", report.get("n_anomalias_marcadas"))
        c3.metric("Gold incorretos", report.get("n_gold_incorretos"))
        c4.metric("Kappa vs ref.", report.get("cohen_kappa_anomalia_vs_gold"))
        cg = report.get("confusao_vs_referencia") or report.get("confusao_vs_gold") or {}
        st.subheader("Confusão detector × referência")
        st.json(cg)

    if not (kpi == "sumario_lexical" and lex):
        pat = report.get("sumario_padroes")
        if pat:
            st.subheader("Padrões (primário)")
            prim = pat.get("por_padrao_primario") or {}
            if prim:
                st.bar_chart(pd.Series(prim))

    marg = layer.get("gatilhos_marginais") or {}
    if marg:
        st.subheader("Gatilhos por camada activa")
        chart: dict[str, int] = {}
        if "embedding" in active and marg.get("n_embedding_baixo_suporte") is not None:
            chart["embedding baixo"] = int(marg.get("n_embedding_baixo_suporte") or 0)
        if "juiz" in active and marg.get("n_juiz_negativo") is not None:
            chart["juiz (agregação)"] = int(marg.get("n_juiz_negativo") or 0)
        n_j_diag = marg.get("n_juiz_diagnostico_negativo")
        if (
            "juiz" in active
            and n_j_diag is not None
            and int(n_j_diag) != int(marg.get("n_juiz_negativo") or 0)
        ):
            st.caption(
                f"Juiz diagnóstico (inclui incompleto): {n_j_diag} itens — "
                "não entra na agregação por defeito."
            )
        if "gold" in active and marg.get("n_sinal_ouro_incorreto") is not None:
            chart["gold listas"] = int(marg.get("n_sinal_ouro_incorreto") or 0)
        elif ref_type == "lexical":
            n_lex = marg.get("n_referencia_lexica_fraca")
            if n_lex is not None:
                st.caption(
                    f"Referência léxica fraca (diagnóstico, não camada activa): {n_lex} itens — "
                    "não confundir com verificação «gold» desligada."
                )
        if chart:
            st.bar_chart(pd.Series(chart))
        nota = layer.get("nota_referencia")
        if nota:
            st.caption(str(nota))


def _filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.divider()
        st.subheader("Filtros (Inspector)")
        only_anom = st.checkbox("Só anomalias", value=False, key="qa_only_anom")
        only_fp = st.checkbox("Só FP embedding suspeito", value=False, key="qa_only_fp")
        only_refusal = st.checkbox("Só recusas", value=False, key="qa_only_refusal")
        f1_min = st.slider("F1 token mín.", 0.0, 1.0, 0.0, 0.05, key="qa_f1_min")
        f1_max = st.slider("F1 token máx.", 0.0, 1.0, 1.0, 0.05, key="qa_f1_max")
        prim_opts = sorted(
            {str(x) for x in df["padrao_primario"].dropna().unique()},
        )
        prim_pick = st.multiselect("Padrão primário", prim_opts, default=prim_opts, key="qa_prim")

    view = df.copy()
    if only_anom:
        view = view[view["flag_anomalia"]]
    if only_fp and "padroes" in view.columns:
        view = view[view["padroes"].apply(lambda tags: "grounding_fp_suspeito" in (tags or []))]
    if only_refusal:
        view = view[view["e_recusa"] == True]  # noqa: E712
    if "f1_token" in view.columns:
        view = view[view["f1_token"].fillna(0).between(f1_min, f1_max)]
    if prim_pick and "padrao_primario" in view.columns:
        view = view[view["padrao_primario"].isin(prim_pick)]
    return view


def _render_qa_inspector(df: pd.DataFrame, records: list[RunRecord]) -> None:
    view = _filter_dataframe(df)
    st.caption(f"{len(view)} itens após filtros")

    ids = view["id_item"].tolist() if not view.empty else []
    if not ids:
        st.info("Nenhum item corresponde aos filtros.")
        return

    pick = st.selectbox("Item", ids, key="qa_pick")
    rec = next((r for r in records if r.item_id == pick), None)
    row = view[view["id_item"] == pick].iloc[0] if not view.empty else None
    if rec is None or row is None:
        return

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Pergunta")
        st.write(rec.question)
        st.markdown("### Resposta do modelo")
        st.write(rec.answer)
        refs = rec.meta.get("referencias") or rec.meta.get("references") or []
        st.markdown("### Referências gold")
        if refs:
            for i, ref in enumerate(refs[:10], 1):
                st.markdown(f"{i}. {ref}")
        else:
            st.caption("Sem referências persistidas (corrida antiga).")
        prim = row.get("padrao_primario") or "—"
        tier = row.get("tier_qualidade") or "—"
        st.markdown(f"**Padrão primário:** `{prim}` · **Tier:** `{tier}`")

    with col_r:
        st.markdown("### Sinais determinísticos")
        checks = [
            ("F1 token", _fmt(row.get("f1_token"))),
            ("EM SQuAD", "✓" if row.get("em_squad") else "—"),
            ("Embedding (max)", _fmt(row.get("embedding_max_coseno"))),
            ("Embedding vs ouro", _fmt(row.get("embedding_max_coseno_ouro"))),
            ("Embedding vs recuperados", _fmt(row.get("embedding_max_coseno_recuperados"))),
            ("Embedding baixo", "⚠" if row.get("embedding_baixo_suporte") else "—"),
            ("Rank chunk ouro", _fmt(row.get("rank_chunk_ouro"))),
            ("Recusa", "⚠" if row.get("e_recusa") else "—"),
            ("Anomalia", "⚠" if row.get("flag_anomalia") else "—"),
        ]
        for label, val in checks:
            st.markdown(f"- **{label}:** {val}")
        tags = row.get("padroes") or []
        if tags:
            st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in tags))

        st.markdown("### Juiz (não-determinístico)")
        if rec.signals.judge:
            j = rec.signals.judge
            fb = " (heurística)" if j.raw.get("fallback_heuristico") else ""
            st.markdown(f"- **Veredito:** `{j.veredito}`{fb}")
            st.markdown(f"- **Motivo:** {j.motivo_breve}")
            st.markdown(f"- **Confiança:** {_fmt(j.confianca)}")
            cot = j.raw.get("cadeia_de_pensamento")
            if isinstance(cot, list) and cot:
                with st.expander("Cadeia de pensamento do juiz", expanded=False):
                    for i, step in enumerate(cot, 1):
                        st.markdown(f"{i}. {step}")
        else:
            st.caption("Juiz desligado ou ausente.")

    exp = rec.meta.get("explicacao")
    if isinstance(exp, dict):
        with st.expander("Porquê? (explicabilidade do harness)", expanded=True):
            for key in ("alerta", "recuperacao", "lexical", "juiz", "conflitos"):
                blk = exp.get(key)
                if blk:
                    st.markdown(f"**{key}**")
                    st.json(blk)
            rat = exp.get("rationale_padroes")
            if rat:
                st.markdown("**Rationale padrões**")
                st.json(rat)

    _render_retrieved_chunks(rec, key_prefix="qa")


def _render_patterns(report: dict[str, Any], df: pd.DataFrame) -> None:
    pat = report.get("sumario_padroes") or {}
    prim = pat.get("por_padrao_primario") or {}
    if prim:
        st.subheader("Contagem por padrão primário")
        st.bar_chart(pd.Series(prim))
    elif "padrao_primario" in df.columns and df["padrao_primario"].notna().any():
        st.subheader("Contagem por padrão primário (derivado do JSONL)")
        st.bar_chart(df["padrao_primario"].value_counts())

    if "padroes" in df.columns:
        tags: dict[str, int] = {}
        for lst in df["padroes"]:
            if isinstance(lst, list):
                for t in lst:
                    tags[str(t)] = tags.get(str(t), 0) + 1
        if tags:
            st.subheader("Co-ocorrência de tags (contagem)")
            st.bar_chart(pd.Series(tags).sort_values(ascending=False))

    if "f1_token" in df.columns:
        st.subheader("Top falhas (F1 ascendente)")
        cols = ["id_item", "pergunta", "resposta", "f1_token", "padrao_primario", "flag_anomalia"]
        show = [c for c in cols if c in df.columns]
        st.dataframe(
            df.dropna(subset=["f1_token"]).sort_values("f1_token").head(15)[show],
            use_container_width=True,
        )


def _render_calibration(
    report: dict[str, Any],
    df: pd.DataFrame,
    records: list[RunRecord],
    run_dir: Path,
) -> None:
    """Tabela completa + limiares para calibrar embedding e recuperação."""
    proto = report.get("protocolo_ativo") or {}
    detector = report.get("detector_activo") or {}
    st.markdown(
        "Inspecção item a item com **todas as métricas** exportáveis. "
        "Use os limiares abaixo para simular cortes (não altera a corrida gravada)."
    )
    if proto:
        st.caption(f"Protocolo gravado: `{proto}`")
    if detector:
        c1, c2, c3 = st.columns(3)
        c1.metric("Política de alerta", detector.get("politica_agregacao", "—"))
        c2.metric("Taxa de alerta", _fmt_pct(report.get("taxa_alerta")))
        c3.metric("FP gold-correto", _fmt_pct(report.get("taxa_falso_alarme_no_gold_correto")))

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        thr_emb = st.number_input(
            "Limiar embedding (simulação)",
            min_value=0.0,
            max_value=1.0,
            value=0.28,
            step=0.05,
            key="cal_thr_emb",
            help="Abaixo disto → `embedding_baixo` simulado (calibrar com FairytaleQA)",
        )
    with col_t2:
        thr_ret = st.number_input(
            "Limiar recuperação (gate)",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="cal_thr_ret",
            help="Abaixo disto → recuperação fraca simulada",
        )
    with col_t3:
        policy = st.selectbox(
            "Política simulada",
            ["embedding_e_juiz", "qualquer_critico", "todos_criticos"],
            index=0,
            key="cal_policy",
        )

    full = calibration_view_dataframe(df)
    if full.empty:
        st.info("Sem dados.")
        return

    sim = full.copy()
    if "embedding_max_coseno" in sim.columns:
        emb = sim["embedding_max_coseno"]
        sim["emb_abaixo_limiar"] = emb.notna() & (emb < thr_emb)
    if "score_melhor_chunk" in sim.columns:
        sc = sim["score_melhor_chunk"]
        sim["recuperacao_fraca_sim"] = sc.notna() & (sc < thr_ret)
    if "emb_abaixo_limiar" in sim.columns and "juiz_negativo" in sim.columns:
        e = sim["emb_abaixo_limiar"].fillna(False)
        j = sim["juiz_negativo"].fillna(False)
        if policy == "embedding_e_juiz":
            sim["anomalia_simulada"] = e & j
        elif policy == "qualquer_critico":
            sim["anomalia_simulada"] = e | j
        else:
            sim["anomalia_simulada"] = e & j

    st.subheader("Filtros rápidos")
    f1, f2, f3, f4, f5 = st.columns(5)
    preset = f1.selectbox(
        "Preset",
        [
            "Todos",
            "FP embedding (gold OK + emb baixo)",
            "Só anomalias gravadas",
            "Só anomalia simulada",
            "Embedding baixo (gravado)",
            "Juiz negativo",
            "Recuperação fraca (score)",
        ],
        key="cal_preset",
    )
    view = sim
    if preset == "FP embedding (gold OK + emb baixo)":
        view = view[
            (view["gold_correto"] == True)  # noqa: E712
            & (view["embedding_baixo_suporte"] == True)  # noqa: E712
        ]
    elif preset == "Só anomalias gravadas":
        view = view[view["flag_anomalia"] == True]  # noqa: E712
    elif preset == "Só anomalia simulada" and "anomalia_simulada" in view.columns:
        view = view[view["anomalia_simulada"] == True]  # noqa: E712
    elif preset == "Embedding baixo (gravado)":
        view = view[view["embedding_baixo_suporte"] == True]  # noqa: E712
    elif preset == "Juiz negativo":
        view = view[view["juiz_negativo"] == True]  # noqa: E712
    elif preset == "Recuperação fraca (score)" and "recuperacao_fraca_sim" in view.columns:
        view = view[view["recuperacao_fraca_sim"] == True]  # noqa: E712

    if f2.checkbox("Excluir curadas (gate)", value=False, key="cal_no_curated") and (
        "curada_recuperacao_fraca" in view.columns
    ):
        view = view[~view["curada_recuperacao_fraca"].fillna(False)]

    sort_options = [c for c in CALIBRATION_COLUMN_ORDER if c in view.columns]
    sort_default = sort_options.index("f1_token") if "f1_token" in sort_options else 0
    sort_col = f3.selectbox(
        "Ordenar por",
        sort_options,
        index=sort_default,
        key="cal_sort",
    )
    ascending = f4.checkbox("Ascendente", value=True, key="cal_asc")
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending, na_position="last")

    default_cols = [c for c in CALIBRATION_COLUMN_ORDER if c in view.columns]
    sim_cols = ("emb_abaixo_limiar", "recuperacao_fraca_sim", "anomalia_simulada")
    extra_sim = [c for c in sim_cols if c in view.columns]
    all_pickable = default_cols + extra_sim
    picked = f5.multiselect(
        "Colunas visíveis",
        all_pickable,
        default=all_pickable,
        key="cal_cols",
    )
    if not picked:
        picked = all_pickable

    st.subheader(f"Tabela de calibração ({len(view)}/{len(full)} itens)")
    st.caption(
        "Scores de coseno pergunta→chunk ~0,5–0,7 são normais (MiniLM). "
        "Não usar 0,85 como piso de recuperação. "
        f"Tabelas >{_INSPECTOR_PAGE_SIZE} linhas são paginadas (virtualização completa = roadmap)."
    )
    table_view = _paginate_dataframe(
        view[picked],
        page_size=_INSPECTOR_PAGE_SIZE,
        page_key="cal_table_page",
    )
    st.dataframe(
        table_view,
        use_container_width=True,
        height=min(600, 80 + 35 * max(len(table_view), 1)),
        column_config={
            "pergunta": st.column_config.TextColumn(width="medium"),
            "resposta": st.column_config.TextColumn(width="medium"),
            "juiz_motivo": st.column_config.TextColumn(width="medium"),
            "texto_referencia": st.column_config.TextColumn(width="medium"),
        },
    )

    csv_bytes = view[picked].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exportar CSV (filtro actual)",
        data=csv_bytes,
        file_name=f"{run_dir.name}_calibracao.csv",
        mime="text/csv",
        key="cal_csv",
    )

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with st.expander("summary.json (limiares e agregados)"):
            st.json(load_summary_json(run_dir) or {})

    if (
        "f1_token" in view.columns
        and "embedding_max_coseno" in view.columns
        and view["embedding_max_coseno"].notna().any()
    ):
        st.subheader("Gráficos de calibração")
        plot_df = view.dropna(subset=["f1_token", "embedding_max_coseno"]).copy()
        fig = px.scatter(
            plot_df,
            x="f1_token",
            y="embedding_max_coseno",
            color="flag_anomalia",
            symbol="gold_correto",
            hover_data=["id_item", "veredito_juiz", "score_melhor_chunk"],
            title="F1 token vs embedding (cor = anomalia gravada)",
        )
        fig.add_hline(y=thr_emb, line_dash="dash", annotation_text=f"emb={thr_emb}")
        st.plotly_chart(fig, use_container_width=True)

    if (
        "f1_token" in view.columns
        and "score_melhor_chunk" in view.columns
        and view["score_melhor_chunk"].notna().any()
    ):
        plot2 = view.dropna(subset=["f1_token", "score_melhor_chunk"])
        fig2 = px.scatter(
            plot2,
            x="f1_token",
            y="score_melhor_chunk",
            color="flag_anomalia",
            hover_data=["id_item", "rank_chunk_ouro"],
            title="F1 token vs score melhor chunk (recuperação)",
        )
        fig2.add_hline(y=thr_ret, line_dash="dash", annotation_text=f"gate={thr_ret}")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Detalhe do item seleccionado")
    ids = view["id_item"].tolist()
    if ids:
        pick = st.selectbox("Item", ids, key="cal_pick")
        rec = next((r for r in records if r.item_id == pick), None)
        if rec is not None:
            row = view[view["id_item"] == pick].iloc[0]
            st.json({k: (None if pd.isna(v) else v) for k, v in row.items()})
            st.markdown(f"**Pergunta:** {rec.question}")
            st.markdown(f"**Resposta:** {rec.answer}")
            _render_retrieved_chunks(rec, key_prefix="cal")


def _render_retrieval(report: dict[str, Any], df: pd.DataFrame) -> None:
    st.subheader("Sumário de recuperação")
    sr = report.get("sumario_recuperacao")
    if sr:
        st.json(sr)
    else:
        st.info("Sem `sumario_recuperacao` (RAG desligado ou corrida antiga).")

    if "score_melhor_chunk" in df.columns and df["score_melhor_chunk"].notna().any():
        fig = px.histogram(df, x="score_melhor_chunk", nbins=30, title="Score melhor chunk")
        st.plotly_chart(fig, use_container_width=True)
    if "rank_chunk_ouro" in df.columns and df["rank_chunk_ouro"].notna().any():
        fig2 = px.histogram(
            df.dropna(subset=["rank_chunk_ouro"]),
            x="rank_chunk_ouro",
            nbins=10,
            title="Rank chunk ouro",
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_signals(layer: dict[str, Any], report: dict[str, Any]) -> None:
    ref_type = _active_reference_type(report, layer)
    st.subheader("Combinações exclusivas (todos os itens)")
    excl = layer.get("combinacoes_exclusivas_todos_itens") or {}
    if excl:
        st.bar_chart(pd.Series(excl))

    st.subheader("Concordância entre camadas (kappa)")
    if ref_type == "lexical":
        st.caption(
            "Kappa baixo face à referência léxica é **esperado**: alertas vêm de "
            "embedding/juiz (RAG), não de F1 com a resposta curta. "
            "Compare kappa embedding↔juiz; use F1 médio como KPI principal."
        )
    pares = layer.get("concordancia_entre_camadas") or []
    if pares:
        rows_k = [
            {"par": p.get("par"), "kappa": p.get("cohen_kappa")}
            for p in pares
            if isinstance(p, dict)
        ]
        kdf = pd.DataFrame(rows_k)
        st.dataframe(kdf, use_container_width=True)

    por = layer.get("por_camada_vs_referencia") or {}
    if por:
        st.subheader("Por camada vs referência")
        st.json(por)

    nota = layer.get("nota_referencia")
    if nota:
        st.caption(str(nota))


def _render_reference(report: dict[str, Any], df: pd.DataFrame) -> None:
    st.subheader("Métricas léxicas (agregado)")
    lex = report.get("sumario_lexical") or report.get("lexical_summary")
    if lex:
        st.json(lex)
    else:
        st.info("Léxico desligado ou sem referência textual.")

    if (
        "f1_token" in df.columns
        and "embedding_max_coseno" in df.columns
        and df["f1_token"].notna().any()
    ):
        plot_df = df.dropna(subset=["f1_token", "embedding_max_coseno"]).copy()
        if not plot_df.empty:
            color = "padrao_primario" if "padrao_primario" in plot_df.columns else None
            fig = px.scatter(
                plot_df,
                x="f1_token",
                y="embedding_max_coseno",
                color=color,
                hover_data=["id_item"],
                title="F1 token vs embedding (cor por padrão primário)",
            )
            st.plotly_chart(fig, use_container_width=True)

    for col, title in [
        ("f1_token", "F1 token (SQuAD/NQ)"),
        ("bleu", "BLEU"),
        ("rouge_l_f", "ROUGE-L F"),
    ]:
        if col in df.columns and df[col].notna().any():
            st.plotly_chart(
                px.histogram(df.dropna(subset=[col]), x=col, nbins=25, title=title),
                use_container_width=True,
            )


def _render_legacy_inspector(df: pd.DataFrame, records: list[RunRecord]) -> None:
    only_anom = st.checkbox("Só anomalias", value=False, key="legacy_anom")
    view = df[df["flag_anomalia"]] if only_anom else df
    cal = calibration_view_dataframe(view)
    st.caption(
        f"{len(cal)} linhas · paginação de {_INSPECTOR_PAGE_SIZE} (virtualização = roadmap)."
    )
    page_df = _paginate_dataframe(cal, page_size=_INSPECTOR_PAGE_SIZE, page_key="legacy_table_page")
    st.dataframe(page_df, use_container_width=True, height=400)

    ids = view["id_item"].tolist() if not view.empty else []
    if not ids:
        return
    pick = st.selectbox("Detalhe do item", ids, key="legacy_pick")
    rec = next((r for r in records if r.item_id == pick), None)
    if rec is None:
        return
    st.markdown(f"**Pergunta:** {rec.question}")
    st.markdown(f"**Resposta:** {rec.answer}")
    _render_retrieved_chunks(rec, key_prefix="legacy")
    if rec.signals.judge:
        st.markdown("**Juiz**")
        st.json(
            {
                "veredito": rec.signals.judge.veredito,
                "motivo": rec.signals.judge.motivo_breve,
                "confianca": rec.signals.judge.confianca,
            },
        )
    st.markdown("**Meta**")
    st.json(rec.meta)


@st.cache_data(show_spinner=False)
def _load_run_bundle(run_dir_str: str, fingerprint: str) -> dict[str, Any]:
    """Cache Streamlit alinhado com ``cache_run_artifacts`` (invalida por fingerprint)."""
    del fingerprint  # parte da chave de cache
    from pathlib import Path as _Path

    return cache_run_artifacts(_Path(run_dir_str))


def _render_integrity_badges(integrity: dict[str, Any], validation: dict[str, Any]) -> None:
    st.subheader("Integridade")
    score_raw = integrity.get("integrity_score")
    score = float(score_raw) if isinstance(score_raw, (int, float)) else None
    if score is not None:
        if score >= 85:
            st.success(f"Integridade: {score}/100")
        elif score >= 60:
            st.warning(f"Integridade: {score}/100")
        else:
            st.error(f"Integridade: {score}/100")

    if validation.get("schema_mismatch"):
        st.error("Schema mismatch")
    if validation.get("legacy_run"):
        st.warning("Corrida legada")
    if integrity.get("escrita_parcial"):
        st.error("Escrita parcial (.tmp)")
    if integrity.get("checksums_ok") is False:
        st.error("Checksums / ficheiros em falta")
    elif integrity.get("checksums_ok") is True:
        st.caption("Checksums manifest OK")

    warnings = validation.get("warnings")
    warn_list = warnings if isinstance(warnings, list) else []
    n_warn_raw = validation.get("n_warnings_schema")
    n_warn = int(n_warn_raw) if isinstance(n_warn_raw, int) else len(warn_list)
    if n_warn:
        with st.expander(f"Avisos de schema ({n_warn})"):
            for w in warn_list[:20]:
                st.caption(f"• {w}")
            if n_warn > 20:
                st.caption(f"… e mais {n_warn - 20}")


def _paginate_dataframe(
    df: pd.DataFrame,
    *,
    page_size: int,
    page_key: str,
) -> pd.DataFrame:
    if df.empty or len(df) <= page_size:
        return df
    total_pages = max(1, (len(df) + page_size - 1) // page_size)
    page = st.number_input(
        "Página",
        min_value=1,
        max_value=total_pages,
        value=1,
        key=page_key,
    )
    start = (int(page) - 1) * page_size
    st.caption(f"Itens {start + 1}–{min(start + page_size, len(df))} de {len(df)}")
    return df.iloc[start : start + page_size]


if __name__ == "__main__":
    main()

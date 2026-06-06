"""Fila de revisão humana (juiz agregação + recusas com RAG forte)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm_evaluation.run_artifacts import atomic_write_json
from llm_evaluation.types import RunRecord
from llm_evaluation.veredito import veredito_e_negativo

# Heurística extra quando ``is_refusal`` não foi gravado (corridas legadas).
_RECUSA_PHRASES = (
    "não há informações",
    "nao ha informacoes",
    "informações suficientes",
    "informacoes suficientes",
    "não há dados",
    "nao ha dados",
)

FILA_CSV_FIELDS = [
    "motivo_fila",
    "prioridade",
    "id_item",
    "pergunta",
    "resposta",
    "referencia_dataset",
    "f1_token",
    "bleu",
    "meteor",
    "em_squad",
    "score_melhor_chunk",
    "rank_chunk_ouro",
    "chunk_ouro_no_top_k",
    "embedding_max_coseno",
    "embedding_baixo_suporte",
    "veredito_juiz",
    "motivo_juiz",
    "confianca_juiz",
    "padrao_primario",
    "padroes",
    "flag_anomalia",
    "contexto_recuperado_completo",
    "explicacao_resumida",
]


def is_recusa_for_fila(record: RunRecord) -> bool:
    """Recusa: sinal gravado no pipeline ou heurística de frase (legado)."""
    if record.signals.is_refusal:
        return True
    a = record.answer.lower()
    return any(p in a for p in _RECUSA_PHRASES)


def _counts_from_fila(fila: list[tuple[str, RunRecord]]) -> dict[str, int]:
    n_hard = sum(1 for m, _ in fila if m == "juiz_veredito_duro")
    n_rec = sum(1 for m, _ in fila if m == "recusa_com_contexto_forte")
    return {
        "total": len(fila),
        "juiz_veredito_duro": n_hard,
        "recusa_com_contexto_forte": n_rec,
    }


def select_fila_records(
    records: list[RunRecord],
    *,
    juiz_vereditos_fila: list[str],
    min_score_recuperacao: float = 0.5,
) -> list[tuple[str, RunRecord]]:
    """Devolve (motivo_fila, record) únicos; juiz duro tem prioridade."""
    if not juiz_vereditos_fila:
        return []
    by_id: dict[str, tuple[str, RunRecord]] = {}
    for record in records:
        iid = record.item_id
        if not iid:
            continue

        j = record.signals.judge
        if (
            j is not None
            and not j.raw.get("fallback_heuristico")
            and veredito_e_negativo(j.veredito, juiz_vereditos_fila)
        ):
            by_id[iid] = ("juiz_veredito_duro", record)
            continue

        if iid in by_id:
            continue

        rm = record.meta.get("metricas_recuperacao") or record.meta.get("retrieval_metrics")
        if not isinstance(rm, dict) or not rm.get("chunk_ouro_no_top_k"):
            continue
        sc = rm.get("score_melhor_chunk")
        if sc is None or float(sc) < min_score_recuperacao:
            continue
        if is_recusa_for_fila(record):
            by_id[iid] = ("recusa_com_contexto_forte", record)

    order = {"juiz_veredito_duro": 0, "recusa_com_contexto_forte": 1}
    out = list(by_id.values())
    out.sort(key=lambda t: (order.get(t[0], 9), t[1].item_id))
    return out


def count_fila_records(
    records: list[RunRecord],
    *,
    juiz_vereditos_fila: list[str],
    min_score_recuperacao: float = 0.5,
) -> dict[str, int]:
    fila = select_fila_records(
        records,
        juiz_vereditos_fila=juiz_vereditos_fila,
        min_score_recuperacao=min_score_recuperacao,
    )
    return _counts_from_fila(fila)


def _format_chunks(record: RunRecord) -> str:
    parts: list[str] = []
    for i, c in enumerate(record.retrieved, start=1):
        gold = " [OURO]" if c.is_gold else ""
        parts.append(f"[{i}] score={c.score:.3f}{gold}\n{c.text}")
    return "\n\n".join(parts) if parts else ""


def record_to_fila_row(motivo: str, record: RunRecord) -> dict[str, Any]:
    lm = record.meta.get("metricas_lexicas") or record.meta.get("lexical_metrics") or {}
    rm = record.meta.get("metricas_recuperacao") or record.meta.get("retrieval_metrics") or {}
    j = record.signals.judge
    refs = record.meta.get("referencias") or record.meta.get("references") or []
    ref_str = " | ".join(str(x) for x in refs[:5]) if isinstance(refs, list) else str(refs)
    diag = record.meta.get("diagnostico")
    padroes = ""
    prim = ""
    if isinstance(diag, dict):
        prim = str(diag.get("padrao_primario") or "")
        tags = diag.get("padroes")
        if isinstance(tags, list):
            padroes = ", ".join(str(t) for t in tags)
    from llm_evaluation.explainability import explicacao_resumida

    return {
        "motivo_fila": motivo,
        "prioridade": 1 if motivo == "juiz_veredito_duro" else 2,
        "id_item": record.item_id,
        "pergunta": record.question,
        "resposta": record.answer,
        "referencia_dataset": ref_str,
        "f1_token": lm.get("f1_token") if isinstance(lm, dict) else None,
        "bleu": lm.get("bleu") if isinstance(lm, dict) else None,
        "meteor": lm.get("meteor") if isinstance(lm, dict) else None,
        "em_squad": lm.get("em_squad") if isinstance(lm, dict) else None,
        "score_melhor_chunk": rm.get("score_melhor_chunk") if isinstance(rm, dict) else None,
        "rank_chunk_ouro": rm.get("rank_chunk_ouro") if isinstance(rm, dict) else None,
        "chunk_ouro_no_top_k": rm.get("chunk_ouro_no_top_k") if isinstance(rm, dict) else None,
        "embedding_max_coseno": record.signals.embedding_max_cosine,
        "embedding_baixo_suporte": record.signals.embedding_low_support,
        "veredito_juiz": j.veredito if j else "",
        "motivo_juiz": j.motivo_breve if j else "",
        "confianca_juiz": j.confianca if j else "",
        "padrao_primario": prim,
        "padroes": padroes,
        "flag_anomalia": record.anomaly_flag,
        "contexto_recuperado_completo": _format_chunks(record),
        "explicacao_resumida": explicacao_resumida(record),
    }


def export_fila_csv(
    run_dir: Path,
    records: list[RunRecord],
    *,
    juiz_vereditos_fila: list[str],
    min_score_recuperacao: float = 0.5,
    output_name: str = "fila_revisao_humana.csv",
) -> tuple[Path, dict[str, int]]:
    """Grava CSV + manifest JSON em ``run_dir/analise_manual/``."""
    fila = select_fila_records(
        records,
        juiz_vereditos_fila=juiz_vereditos_fila,
        min_score_recuperacao=min_score_recuperacao,
    )
    counts = _counts_from_fila(fila)
    rows = [record_to_fila_row(motivo, rec) for motivo, rec in fila]

    out_dir = run_dir / "analise_manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FILA_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = out_path.with_suffix(".json")
    atomic_write_json(
        manifest_path,
        {
            "run_dir": str(run_dir.resolve()),
            "min_score_recuperacao": min_score_recuperacao,
            "juiz_vereditos_fila": juiz_vereditos_fila,
            "contagens": counts,
            "csv": str(out_path.resolve()),
        },
    )
    return out_path, counts

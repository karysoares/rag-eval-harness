#!/usr/bin/env python3
"""Exporta amostra estratificada para rotulagem HITL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.fila_revisao import select_fila_records
from llm_evaluation.hitl_io import export_hitl_csv_template


def _referencias(record: object) -> list[str]:
    """Respostas de referência do dataset, onde o registo as tiver guardado."""
    meta = getattr(record, "meta", {}) or {}
    for chave in ("respostas_referencia", "correct_answers", "gold_answers"):
        valor = meta.get(chave)
        if isinstance(valor, list) and valor:
            return [str(x) for x in valor]
    lex = meta.get("lexical") or meta.get("metricas_lexicas") or {}
    if isinstance(lex, dict) and lex.get("texto_referencia"):
        return [str(lex["texto_referencia"])]
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    records = load_records_from_predictions_jsonl(run_dir / "predictions.jsonl")

    juiz_fila = ["nao_sustentado", "contradicacao"]
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        proto = summary.get("protocolo_ativo")
        agg = proto.get("judge_aggregation_verdicts") if isinstance(proto, dict) else None
        if isinstance(agg, list) and agg:
            juiz_fila = [str(x) for x in agg]

    fila = select_fila_records(records, juiz_vereditos_fila=juiz_fila)
    ids_fila = [r.item_id for _, r in fila]
    rest = [r for r in records if r.item_id not in set(ids_fila)]
    rng = random.Random(args.seed)
    n_rand = max(0, args.n - len(ids_fila))
    sample_rand = rng.sample(rest, min(n_rand, len(rest))) if rest else []
    ids = ids_fila + [r.item_id for r in sample_rand]
    ids = ids[: args.n]

    por_id = {r.item_id: r for r in records}
    contexto: dict[str, dict[str, str]] = {}
    for iid in ids:
        r = por_id.get(iid)
        if r is None:
            continue
        trechos = [c.text for c in r.retrieved][:3]
        contexto[iid] = {
            "pergunta": r.question,
            "resposta_modelo": r.answer,
            # A referência do dataset ajuda a decidir, mas não decide: o que se
            # julga é se a resposta se sustenta no contexto, não se coincide com
            # a string de referência. Por isso vai como coluna de leitura e não
            # como veredito sugerido.
            "referencia": " | ".join(str(x) for x in _referencias(r))[:600],
            "contexto_recuperado": "\n---\n".join(trechos)[:2000],
        }

    out = run_dir / "analise_manual" / "adjudicacoes_hitl_template.csv"
    export_hitl_csv_template(out, ids, contexto=contexto)
    print(f"Template: {out} ({len(ids)} itens, {len(ids_fila)} vindos da fila de revisão)")


if __name__ == "__main__":
    main()

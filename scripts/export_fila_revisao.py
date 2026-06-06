#!/usr/bin/env python3
"""Exporta fila de revisão humana a partir de uma corrida (sem API).

Uso:
  uv run python scripts/export_fila_revisao.py outputs/run_<id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_full_report, load_records_from_predictions_jsonl
from llm_evaluation.fila_revisao import export_fila_csv
from llm_evaluation.operational import thresholds_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Diretório com predictions.jsonl")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Score mínimo do melhor chunk para recusas (default: protocolo ou 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fila_revisao_humana.csv",
        help="Nome do CSV em analise_manual/",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        print(f"Ficheiro em falta: {pred}", file=sys.stderr)
        raise SystemExit(2)

    records = load_records_from_predictions_jsonl(pred)
    report = load_full_report(run_dir) if (run_dir / "summary.json").is_file() else {}
    proto = report.get("protocolo_ativo")
    proto_map = proto if isinstance(proto, dict) else {}
    thr = thresholds_from_mapping(proto_map)
    agg = proto_map.get("judge_aggregation_verdicts")
    juiz_fila = [str(x) for x in agg] if isinstance(agg, list) else []
    if not juiz_fila:
        juiz_fila = ["nao_sustentado", "contradicacao", "inseguro"]
    min_score = args.min_score if args.min_score is not None else thr.fila_min_score_recuperacao
    path, counts = export_fila_csv(
        run_dir,
        records,
        juiz_vereditos_fila=juiz_fila,
        min_score_recuperacao=min_score,
        output_name=args.output,
    )
    print(f"Exportado: {path}")
    print(
        f"  juiz_veredito_duro={counts['juiz_veredito_duro']}  "
        f"recusa_com_contexto_forte={counts['recusa_com_contexto_forte']}  "
        f"total={counts['total']}",
    )


if __name__ == "__main__":
    main()

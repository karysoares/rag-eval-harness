#!/usr/bin/env python3
"""Sweep de limiar embedding vs gold_correct (corrida concluída)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min", type=float, default=0.20)
    p.add_argument("--max", type=float, default=0.45)
    p.add_argument("--step", type=float, default=0.05)
    args = p.parse_args()

    records = load_records_from_predictions_jsonl(args.predictions)
    rows: list[dict[str, object]] = []
    t = args.min
    while t <= args.max + 1e-9:
        fp = fn = tp = tn = 0
        for r in records:
            gc = r.gold_correct
            if gc is None:
                continue
            emb = r.signals.embedding_max_cosine
            low = emb is not None and float(emb) < t
            if gc and low:
                fp += 1
            elif gc and not low:
                tn += 1
            elif not gc and low:
                tp += 1
            else:
                fn += 1
        rows.append(
            {
                "limiar": round(t, 3),
                "fp_gold_correto": fp,
                "fn_gold_incorreto": fn,
                "tp_gold_incorreto": tp,
                "tn_gold_correto": tn,
            },
        )
        t += args.step

    out = args.out or args.predictions.parent / "embedding_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Gravado: {out} e {json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exporta JSONL com campos explicáveis (sem PII completo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.explainability import build_explicacao


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", type=Path, default=None)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    records = load_records_from_predictions_jsonl(run_dir / "predictions.jsonl")
    out = args.o or run_dir / "analise_manual" / "explainability_pack.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in records:
        exp = r.meta.get("explicacao") or build_explicacao(r)
        lines.append(
            json.dumps(
                {
                    "id_item": r.item_id,
                    "flag_anomalia": r.anomaly_flag,
                    "explicacao": exp,
                    "padrao_primario": (r.meta.get("diagnostico") or {}).get("padrao_primario")
                    if isinstance(r.meta.get("diagnostico"), dict)
                    else None,
                },
                ensure_ascii=False,
            ),
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Gravado: {out} ({len(lines)} linhas)")


if __name__ == "__main__":
    main()

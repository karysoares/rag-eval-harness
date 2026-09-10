#!/usr/bin/env python3
"""Sweep do limiar de embedding contra a referência da corrida (offline).

Produz a curva FP/FN por limiar que `docs/calibracao_embedding.md` diz justificar
`verification.embedding_min_cosine` no YAML.

O rótulo de referência vem de `referencia_incorreta`, não de `gold_correct`: em
`reference_type: lexical` — que é o caso de **todos** os configs FairytaleQA — o
`gold_correct` é sempre `null`, pelo que a versão anterior deste script saltava cada
item e devolvia uma tabela de zeros no próprio corpus de referência. Uma curva vazia
não justifica limiar nenhum.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.reference_metrics import referencia_incorreta


def _div(num: int, den: int) -> float | None:
    """Nunca inventa um valor quando o denominador é zero."""
    return round(num / den, 4) if den else None


def _reference_type_da_corrida(predictions: Path) -> str:
    """Lê o `reference_type` do `summary.json` ao lado, para não o adivinhar."""
    summary = predictions.parent / "summary.json"
    if summary.is_file():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        protocolo = data.get("protocolo_ativo") or {}
        for chave in ("tipo_referencia_ativo", "reference_type"):
            valor = data.get(chave) or protocolo.get(chave)
            if isinstance(valor, str) and valor:
                return valor
    return "lexical"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min", type=float, default=0.20)
    p.add_argument("--max", type=float, default=0.45)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument(
        "--reference-type",
        default=None,
        help="lexical | answer_lists | none (por omissao: le de summary.json ao lado)",
    )
    p.add_argument(
        "--f1-fraca-min",
        type=float,
        default=None,
        help="limiar de F1 abaixo do qual a referencia conta como fraca (reference_type lexical)",
    )
    args = p.parse_args()

    records = load_records_from_predictions_jsonl(args.predictions)
    ref_type = args.reference_type or _reference_type_da_corrida(args.predictions)
    if ref_type == "none":
        raise SystemExit("reference_type=none: sem rotulo de referencia, o sweep nao e definivel")
    rows: list[dict[str, object]] = []
    t = args.min
    while t <= args.max + 1e-9:
        fp = fn = tp = tn = 0
        for r in records:
            # Falha de execução não é métrica em falta (regra 3 do CLAUDE.md).
            if "processing_error" in (r.meta or {}):
                continue
            incorreta = referencia_incorreta(r, ref_type, f1_fraca_min=args.f1_fraca_min)
            if incorreta is None:
                continue
            gc = not incorreta
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
                # Derivadas no artefacto: sem elas a tabela não diz se algum limiar
                # é preferível, que é a única razão para correr o sweep.
                "precisao": _div(tp, tp + fp),
                "revocacao": _div(tp, tp + fn),
                "taxa_alerta": _div(tp + fp, tp + fp + fn + tn),
            },
        )
        t += args.step

    out = args.out or args.predictions.parent / "embedding_sweep.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out.with_suffix(".json")
    artefacto = {
        "origem": str(args.predictions),
        "reference_type": ref_type,
        "f1_fraca_min": args.f1_fraca_min,
        "n_itens_com_rotulo": rows[0]["tp_gold_incorreto"]
        + rows[0]["fp_gold_correto"]
        + rows[0]["fn_gold_incorreto"]
        + rows[0]["tn_gold_correto"],
        "nota": (
            "tp/fn contam itens cuja referencia lexical e fraca; fp/tn os de referencia "
            "aceitavel. O sinal comparado e embedding_max_coseno < limiar. Excluidos os "
            "itens com processing_error."
        ),
        "linhas": rows,
    }
    json_path.write_text(
        json.dumps(artefacto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Gravado: {out} e {json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Valida offline a mitigação embedding_e_juiz num diretório de corrida.

Uso:
  uv run python scripts/validate_embedding_policy.py outputs/run_<id>
  uv run python scripts/validate_embedding_policy.py outputs/run_<id> --write

Requer ``predictions.jsonl``. Compara ``qualquer_critico`` vs ``embedding_e_juiz``
usando os sinais já gravados (sem API). Critério P0: FP em referência aceitável
&lt; 15%% com ``embedding_e_juiz``.
Para ``reference_type=answer_lists`` usa ``gold_correto`` booleano; para
``lexical`` usa overlap léxico (F1/EM). Com ``reference_type=none``, P0 é N/A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_evaluation.evaluation_metrics import (
    compare_aggregation_policies,
    load_records_from_predictions_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Diretório com predictions.jsonl")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Grava policy_validation.json no diretório da corrida",
    )
    parser.add_argument(
        "--fp-threshold",
        type=float,
        default=0.15,
        help="Limite máximo aceitável de FP em gold-correto (default 0.15)",
    )
    parser.add_argument(
        "--embedding-min-cosine",
        type=float,
        default=None,
        help="Recalcula embedding_baixo offline com este limiar (ex.: 0.28)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        print(f"Ficheiro em falta: {pred}", file=sys.stderr)
        raise SystemExit(2)

    records = load_records_from_predictions_jsonl(pred)
    summary_path = run_dir / "summary.json"
    reference_type: str | None = "answer_lists"
    protocol: dict[str, object] = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "negative_judge_verdicts": [
            "nao_sustentado",
            "contradicacao",
            "incompleto",
            "inseguro",
        ],
    }
    if summary_path.is_file():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        rt = raw.get("tipo_referencia_ativo")
        if isinstance(rt, str):
            reference_type = rt
        pa = raw.get("protocolo_ativo")
        if isinstance(pa, dict):
            protocol.update(
                {
                    k: pa[k]
                    for k in (
                        "verify_gold",
                        "verify_embedding",
                        "verify_judge",
                        "negative_judge_verdicts",
                        "judge_aggregation_verdicts",
                        "embedding_min_cosine",
                    )
                    if k in pa
                }
            )
            negs = pa.get("negative_judge_verdicts")
            if isinstance(negs, list):
                protocol["negative_judge_verdicts"] = [str(x) for x in negs]
            agg = pa.get("judge_aggregation_verdicts")
            if isinstance(agg, list):
                protocol["judge_aggregation_verdicts"] = [str(x) for x in agg]

    emb_thr = args.embedding_min_cosine
    if emb_thr is not None:
        from llm_evaluation.evaluation_metrics import recompute_embedding_low_support

        records = recompute_embedding_low_support(records, emb_thr)
        report_note = f"embedding_min_cosine recalculado offline: {emb_thr}"
    else:
        report_note = "sinais gravados na corrida (sem recálculo de limiar)"

    agg_v = protocol.get("judge_aggregation_verdicts")
    report = compare_aggregation_policies(
        records,
        verify_gold=bool(protocol["verify_gold"]),
        verify_embedding=bool(protocol["verify_embedding"]),
        verify_judge=bool(protocol["verify_judge"]),
        negative_judge_verdicts=list(protocol["negative_judge_verdicts"]),  # type: ignore[arg-type]
        judge_aggregation_verdicts=([str(x) for x in agg_v] if isinstance(agg_v, list) else None),
        reference_type=reference_type,
    )
    report["nota_limiar"] = report_note
    if emb_thr is not None:
        report["embedding_min_cosine_offline"] = emb_thr
    mit = report["politicas"].get("embedding_e_juiz", {})
    fp_mit = mit.get("taxa_falso_alarme_no_gold_correto") if isinstance(mit, dict) else None
    p0_aplicavel = reference_type != "none"
    if p0_aplicavel:
        ok = fp_mit is not None and fp_mit < args.fp_threshold
        passou: bool | None = ok
    else:
        ok = True
        passou = None
    report["criterio_p0"] = {
        "aplicavel": p0_aplicavel,
        "reference_type": reference_type,
        "fp_threshold": args.fp_threshold,
        "passou": passou,
        "nota": (
            "embedding_e_juiz deve reduzir FP só-embedding; "
            "juiz sustentado não deve gerar anomalia. "
            "Referência aceitável: gold_correto (answer_lists) ou overlap léxico (lexical)."
            if p0_aplicavel
            else "P0 não aplicável sem referência automática (reference_type=none)."
        ),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.write:
        out = run_dir / "policy_validation.json"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Gravado: {out}", file=sys.stderr)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Auditoria de invariantes numa corrida (engenharia + lógica)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.run_artifacts import validate_run_artifacts


def audit(run_dir: Path, *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        return [f"{run_dir.name}: sem predictions.jsonl"]
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    recs = load_records_from_predictions_jsonl(pred)
    n = len(recs)
    if n == 0:
        issues.append(f"{run_dir.name}: 0 registos")
        return issues

    ref_type = summary.get("tipo_referencia_ativo", "?")
    kpi = summary.get("kpi_primario", "?")

    # Invariantes gerais
    for r in recs:
        if r.answer.strip() == "":
            issues.append(f"{run_dir.name}: resposta vazia em {r.item_id}")
        if "<specific" in r.answer.lower() or "<" in r.answer and ">" in r.answer:
            issues.append(
                f"{run_dir.name}: placeholder na resposta {r.item_id}: {r.answer[:60]!r}",
            )
        lm = r.meta.get("metricas_lexicas") or {}
        if (
            ref_type == "lexical"
            and lm.get("note") not in ("metricas_lexicas_desligadas", "sem_referencia")
            and "f1_token" not in lm
            and lm.get("note") != "erro_ao_calcular_metricas_lexicas"
        ):
            issues.append(f"{run_dir.name}: falta f1_token em {r.item_id}")

    n_anom = sum(1 for r in recs if r.anomaly_flag)
    n_emb_low = sum(1 for r in recs if r.signals.embedding_low_support is True)
    n_judge_neg = sum(1 for r in recs if r.signals.judge_negative is True)
    n_chunks = sum(1 for r in recs if r.retrieved)
    n_no_chunks = sum(1 for r in recs if not r.retrieved)

    lex = summary.get("sumario_lexical") or {}
    if ref_type == "lexical" and kpi == "sumario_lexical" and not lex:
        issues.append(f"{run_dir.name}: lexical KPI activo mas sumario_lexical ausente")

    # Heurísticas por modo
    if ref_type == "lexical" and n_anom > 0 and n_emb_low == 0 and n_judge_neg == 0:
        issues.append(
            f"{run_dir.name}: {n_anom} anomalias sem embedding/juiz — verificar verify_gold",
        )
    if n_no_chunks == n and any(r.signals.embedding_low_support is True for r in recs):
        issues.append(f"{run_dir.name}: embedding_baixo com 0 chunks em todos os itens")

    print(f"\n{'=' * 60}\n{run_dir.name}  (n={n}, ref={ref_type}, kpi={kpi})")
    print(f"  anomalias={n_anom}  emb_low={n_emb_low}  juiz_neg={n_judge_neg}")
    print(f"  com_chunks={n_chunks}  sem_chunks={n_no_chunks}")
    if lex:
        print(
            f"  F1_token={lex.get('media_f1_token')}  EM_squad={lex.get('taxa_em_squad')}  "
            f"EM_substr={lex.get('taxa_exact_match')}",
        )
    if summary.get("sumario_recuperacao"):
        print(f"  recuperação: {summary['sumario_recuperacao']}")
    if summary.get("protocolo_ajustado"):
        print(f"  protocolo_ajustado: {summary['protocolo_ajustado']}")
    meta = summary.get("metadados_corrida") or {}
    if meta.get("git_commit"):
        print(f"  git_commit: {meta['git_commit']}")
    if meta.get("config_hash_sha256"):
        print(f"  config_hash: {meta['config_hash_sha256'][:12]}…")

    for msg in validate_run_artifacts(run_dir, strict=strict):
        if msg.startswith("aviso:"):
            print(f"  {msg}")
        else:
            issues.append(msg)

    return issues


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--strict"]
    strict = "--strict" in sys.argv[1:]
    root = Path(args[0]) if args else Path("outputs")
    all_issues: list[str] = []
    if root.is_dir() and (root / "predictions.jsonl").is_file():
        runs = [root]
    elif root.is_dir():
        runs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")],
            key=lambda p: p.name,
            reverse=True,
        )[:5]
    else:
        all_issues.append(f"{root}: diretório não encontrado")
        runs = []
    if not runs and not all_issues:
        all_issues.append(f"{root}: nenhuma corrida encontrada para auditar")
    for run_dir in runs:
        all_issues.extend(audit(run_dir, strict=strict))
    if all_issues:
        print(f"\n{'=' * 60}\nPROBLEMAS ({len(all_issues)}):")
        for i in all_issues:
            print(f"  - {i}")
        raise SystemExit(1)
    print(f"\n{'=' * 60}\nAuditoria OK nas {len(runs)} corridas recentes.")


if __name__ == "__main__":
    main()

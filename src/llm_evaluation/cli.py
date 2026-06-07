"""Ponto de entrada da CLI (`llm-eval` / `python -m llm_evaluation.cli`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, cast

from dotenv import load_dotenv

from llm_evaluation.config import (
    AppConfig,
    Orchestration,
    _norm_orquestracao,
    _norm_perfil_baseline,
    apply_baseline_profile,
    load_config,
)
from llm_evaluation.eval_items_load import load_eval_items
from llm_evaluation.evaluation_metrics import (
    compare_metric_reports,
    load_full_report,
    load_records_from_predictions_jsonl,
)
from llm_evaluation.llm_client import MissingApiKeyError, require_openai_api_key
from llm_evaluation.orchestration import multi, single
from llm_evaluation.protocol import (
    ProtocolAdjustment,
    apply_protocol_defaults,
    build_protocolo_ativo,
    collect_protocol_avisos,
    validate_protocol,
)
from llm_evaluation.reporting import (
    ensure_run_dir,
    record_to_json,
    write_anomalies_csv,
    write_anomalies_jsonl,
    write_baseline_comparison,
    write_summary,
)
from llm_evaluation.run_artifacts import (
    CorruptedPredictionsError,
    build_manifest,
    collect_run_metadata,
    compact_predictions_jsonl,
    config_hash,
    count_failed_item_ids,
    finalize_predictions_jsonl,
    load_completed_item_ids,
    write_manifest,
)
from llm_evaluation.types import EvalItem, RunRecord


def main() -> None:
    # Carrega .env do cwd (e ascendentes) e .env na raiz do projeto (instalação editável)
    load_dotenv()
    _pkg_root = Path(__file__).resolve().parents[2]
    load_dotenv(_pkg_root / ".env", override=False)
    parser = argparse.ArgumentParser(
        description=(
            "Executa corridas de avaliação sobre pipelines de linguagem com recuperação "
            "aumentada: geração condicionada, verificação multicamada, persistência de "
            "artefactos e sumários estatísticos conforme a configuração YAML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Fluxo recomendado:\n"
            "  1) Configurar credenciais: cp .env.example .env\n"
            "  2) Smoke offline: uv run llm-eval --config configs/smoke_amostra.yaml\n"
            "  3) FairytaleQA pt-BR: uv run llm-eval --config configs/default.yaml\n"
            "  4) Análise offline: uv run llm-eval --analyze-run outputs/run_<id>\n"
            "\n"
            "Documentação: README.md, CONTRIBUTING.md\n"
            "Visualização: uv run llm-eval-dashboard"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Caminho para o ficheiro de configuração YAML",
    )
    parser.add_argument(
        "--profile",
        choices=["hibrido", "nenhum", "so_embeddings", "so_juiz"],
        default=None,
        help="Sobrescreve o perfil de baseline (camadas de verificação)",
    )
    parser.add_argument(
        "--orchestration",
        choices=["unico", "multiplo", "single", "multi"],
        default=None,
        help="Sobrescreve o modo de orquestração (unico ou multiplo)",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Executa os quatro perfis de baseline e grava JSON de comparação",
    )
    parser.add_argument(
        "--analyze-run",
        type=Path,
        metavar="DIR",
        default=None,
        help=("Reconstrói metrics_report.json a partir de outputs/run_* (sem chamadas à API)"),
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Permite orquestração multiplo (crítico LLM extra; não usar em produção)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        default=None,
        help="Retoma corrida em outputs/run_* (append em predictions.jsonl; regrava summary)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só carrega itens e imprime contagem (sem API)",
    )
    parser.add_argument(
        "--compare-runs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        default=None,
        help=(
            "Compara dois ou mais diretórios de corrida; "
            "grava run_comparison.json no diretório atual"
        ),
    )
    parser.add_argument(
        "--apply-hitl",
        type=Path,
        metavar="CSV",
        default=None,
        help="Merge adjudicacoes_hitl.csv e reprocessa summary/manifest (sem API)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Tentativas para operações de I/O/reprocessamento (default: 3)",
    )
    parser.add_argument(
        "--hitl-strict-ids",
        action="store_true",
        help="Falha se o CSV HITL trouxer id_item não presente em predictions.jsonl",
    )
    args = parser.parse_args()

    if args.apply_hitl is not None:
        if args.resume is None:
            print("--apply-hitl exige --resume RUN_DIR", file=sys.stderr)
            raise SystemExit(2)
        from llm_evaluation.run_reprocess import reprocess_run_dir

        run_dir = args.resume.expanduser().resolve()
        csv_path = args.apply_hitl.expanduser().resolve()
        if not csv_path.is_file():
            print(f"CSV não encontrado: {csv_path}", file=sys.stderr)
            raise SystemExit(2)
        reprocess_run_dir(
            run_dir,
            hitl_csv=csv_path,
            retry_attempts=max(1, args.retry_attempts),
            strict_hitl_ids=args.hitl_strict_ids,
        )
        print(f"HITL aplicado e summary actualizado em {run_dir}")
        return

    if args.analyze_run is not None:
        run_dir = args.analyze_run.expanduser().resolve()
        if not run_dir.is_dir():
            print(f"Não é um diretório: {run_dir}", file=sys.stderr)
            raise SystemExit(2)
        from llm_evaluation.run_reprocess import reprocess_run_dir

        summary = reprocess_run_dir(
            run_dir,
            retry_attempts=max(1, args.retry_attempts),
        )
        out = run_dir / "metrics_report.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Reprocessado: {run_dir / 'summary.json'}")
        print(f"Gravado: {out}")
        return

    if args.compare_runs is not None:
        dirs = [p.expanduser().resolve() for p in args.compare_runs]
        if len(dirs) < 2:
            print("--compare-runs exige pelo menos dois diretórios", file=sys.stderr)
            raise SystemExit(2)
        for d in dirs:
            if not d.is_dir():
                print(f"Não é um diretório: {d}", file=sys.stderr)
                raise SystemExit(2)
        reports = [load_full_report(d) for d in dirs]
        labels = [d.name for d in dirs]
        cmp = compare_metric_reports(reports, labels)
        out = Path.cwd() / "run_comparison.json"
        out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Gravado: {out}")
        return

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        print(f"Erro ao carregar config {args.config}: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    if args.orchestration:
        cfg.orchestration = cast(
            Orchestration,
            _norm_orquestracao(str(args.orchestration)),
        )

    if cfg.orchestration == "multiplo" and not args.experimental:
        print(
            "Orquestração 'multiplo' é experimental. Repita com --experimental se for intencional.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        items = load_eval_items(cfg)
    except Exception as e:  # noqa: BLE001 — caminho de I/O externo (rede, disco)
        print(f"Erro ao carregar itens do dataset: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    print(
        f"Carregados {len(items)} itens de avaliação "
        f"(dataset.mode={cfg.dataset.mode}, limit={cfg.dataset.limit}, "
        f"shuffle={cfg.dataset.shuffle}).",
    )
    if args.dry_run:
        print(f"Dry-run: {len(items)} itens — nenhuma chamada à API.")
        return

    try:
        require_openai_api_key()
    except MissingApiKeyError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e

    if not args.compare_baselines and args.profile is not None:
        cfg = apply_baseline_profile(cfg, _norm_perfil_baseline(str(args.profile)))

    cfg, protocol_adjustments = apply_protocol_defaults(cfg, items)
    for adj in protocol_adjustments:
        print(
            f"Aviso protocolo: {adj.campo} {adj.de!r} → {adj.para!r} — {adj.motivo}",
            file=sys.stderr,
        )
    try:
        validate_protocol(cfg, items)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e
    for aviso in collect_protocol_avisos(cfg):
        print(f"Aviso protocolo: {aviso}", file=sys.stderr)
    if args.compare_baselines:
        out_base = Path(cfg.output_dir)
        run_dir = ensure_run_dir(out_base)
        _run_compare_baselines(cfg, items, run_dir, args.config)
        return

    if args.resume is not None:
        run_dir = args.resume.expanduser().resolve()
        if not run_dir.is_dir():
            print(f"Diretório de corrida inválido: {run_dir}", file=sys.stderr)
            raise SystemExit(2)
    else:
        run_dir = ensure_run_dir(Path(cfg.output_dir))

    _run_single_corrida(
        cfg,
        items,
        run_dir,
        args.config,
        protocol_adjustments,
        resume=args.resume is not None,
        retry_attempts=max(1, args.retry_attempts),
    )


def _make_writer(
    path: Path,
    *,
    include_judge_cot: bool = False,
    append: bool = False,
) -> tuple[Callable[[RunRecord], None], TextIO]:
    """Devolve um writer com flush por linha (resiliência) e o handle do ficheiro."""
    fh: TextIO = path.open("a" if append else "w", encoding="utf-8")

    def write(rec: RunRecord) -> None:
        fh.write(
            json.dumps(
                record_to_json(rec, include_judge_cot=include_judge_cot),
                ensure_ascii=False,
            )
            + "\n",
        )
        fh.flush()

    return write, fh


def _run_single_corrida(
    cfg: AppConfig,
    items: list[EvalItem],
    run_dir: Path,
    config_path: Path,
    protocol_adjustments: list[ProtocolAdjustment] | None = None,
    *,
    resume: bool = False,
    retry_attempts: int = 3,
) -> None:
    predictions_path = run_dir / "predictions.jsonl"
    try:
        completed = load_completed_item_ids(predictions_path) if resume else set()
    except CorruptedPredictionsError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e
    if resume and predictions_path.is_file():
        prev_summary = run_dir / "summary.json"
        if prev_summary.is_file():
            prev = json.loads(prev_summary.read_text(encoding="utf-8"))
            prev_hash = None
            prov = prev.get("proveniencia")
            if isinstance(prov, dict):
                prev_hash = prov.get("config_hash_sha256")
            meta = prev.get("metadados_corrida")
            if prev_hash is None and isinstance(meta, dict):
                prev_hash = meta.get("config_hash_sha256")
            cur_hash = config_hash(config_path)
            if prev_hash and cur_hash and prev_hash != cur_hash:
                print(
                    "Config mudou desde a corrida iniciada. "
                    "Use novo run_dir ou confirme intencionalmente.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    pending = [it for it in items if it.id not in completed]
    include_cot = cfg.verification.judge_return_chain_of_thought

    if resume and completed:
        n_failed_prev = count_failed_item_ids(predictions_path)
        resume_note = (
            f" ({n_failed_prev} com erro API anterior serão reprocessados)" if n_failed_prev else ""
        )
        print(
            f"Retomar: {len(completed)} já gravados, {len(pending)} pendentes "
            f"(total dataset {len(items)}).{resume_note}",
            file=sys.stderr,
        )

    if pending:
        runner = multi.run_items if cfg.orchestration == "multiplo" else single.run_items
        if resume and predictions_path.is_file():
            write, fh = _make_writer(
                predictions_path,
                include_judge_cot=include_cot,
                append=True,
            )
            try:
                runner(cfg, pending, on_record=write)
            finally:
                fh.close()
            removed = compact_predictions_jsonl(predictions_path)
            if removed:
                print(
                    f"Resume: {removed} linha(s) duplicada(s) compactada(s) em predictions.jsonl.",
                    file=sys.stderr,
                )
        else:
            predictions_tmp = predictions_path.with_suffix(".jsonl.tmp")
            write, fh = _make_writer(predictions_tmp, include_judge_cot=include_cot)
            try:
                runner(cfg, pending, on_record=write)
            finally:
                fh.close()
            finalize_predictions_jsonl(predictions_tmp, predictions_path)
    elif not predictions_path.is_file():
        print("Nenhum item pendente e sem predictions.jsonl.", file=sys.stderr)
        raise SystemExit(2)

    records = load_records_from_predictions_jsonl(predictions_path)
    if len(records) < len(items):
        print(
            f"Aviso: predictions tem {len(records)} registos, dataset tem {len(items)}.",
            file=sys.stderr,
        )

    write_anomalies_jsonl(records, run_dir / "anomalies.jsonl")
    write_anomalies_csv(records, run_dir / "anomalies.csv")
    from llm_evaluation.run_reprocess import reprocess_run_dir

    summary = reprocess_run_dir(
        run_dir,
        cfg=cfg,
        config_path=config_path,
        reference_type=cfg.dataset.reference_type,
        retry_attempts=max(1, retry_attempts),
    )
    if protocol_adjustments:
        summary["protocolo_ajustado"] = [
            {
                "campo": a.campo,
                "de": a.de,
                "para": a.para,
                "motivo": a.motivo,
            }
            for a in protocol_adjustments
        ]
        write_summary(summary, run_dir / "summary.json")
    price_p = os.environ.get("OPENAI_PRICE_PER_1M_PROMPT")
    price_c = os.environ.get("OPENAI_PRICE_PER_1M_COMPLETION")
    if price_p and price_c:
        from llm_evaluation.observability import summarize_run_observability

        obs = summarize_run_observability(
            records,
            price_per_1m_prompt=float(price_p),
            price_per_1m_completion=float(price_c),
        )
        if obs:
            summary["observabilidade"] = obs
            write_summary(summary, run_dir / "summary.json")

    metadados_raw = summary.get("metadados_corrida")
    metadados = metadados_raw if isinstance(metadados_raw, dict) else {}
    manifest_extra: list[Path] = [run_dir / "analise_manual" / "fila_revisao_humana.csv"]
    hitl_manifest = run_dir / "analise_manual" / "hitl_manifest.json"
    if hitl_manifest.is_file():
        manifest_extra.append(hitl_manifest)
    adj = run_dir / "analise_manual" / "adjudicacoes_hitl.csv"
    if adj.is_file():
        manifest_extra.append(adj)
    manifest = build_manifest(run_dir, metadados=metadados, extra_files=manifest_extra)
    write_manifest(run_dir, manifest)

    fila_path = run_dir / "analise_manual" / "fila_revisao_humana.csv"
    op = summary.get("sumario_operacional")
    fila_counts = (op.get("fila_revisao_humana") if isinstance(op, dict) else None) or {}
    print(f"Corrida gravada em {run_dir}")
    if fila_path.is_file() and isinstance(fila_counts, dict):
        print(
            f"Fila revisão humana: {fila_path} "
            f"(total={fila_counts.get('total', 0)}, "
            f"juiz_duro={fila_counts.get('juiz_veredito_duro', 0)}, "
            f"recusa={fila_counts.get('recusa_com_contexto_forte', 0)})",
        )


def _run_compare_baselines(
    cfg_base: AppConfig,
    items: list[EvalItem],
    run_dir: Path,
    config_path: Path,
) -> None:
    from llm_evaluation.reporting import summarize as summarize_fn

    comparison: dict[str, dict[str, object]] = {}
    protocolos: dict[str, dict[str, object]] = {}
    ajustes: dict[str, list[dict[str, object]]] = {}
    for p in ("nenhum", "so_embeddings", "so_juiz", "hibrido"):
        c = apply_baseline_profile(cfg_base, p)
        c, protocol_adjustments = apply_protocol_defaults(c, items)
        validate_protocol(c, items)
        protocol = build_protocolo_ativo(c)
        protocolos[p] = protocol
        if protocol_adjustments:
            ajustes[p] = [
                {"campo": a.campo, "de": a.de, "para": a.para, "motivo": a.motivo}
                for a in protocol_adjustments
            ]
        runner = multi.run_items if c.orchestration == "multiplo" else single.run_items
        path = run_dir / f"predictions_{p}.jsonl"
        path_tmp = path.with_suffix(".jsonl.tmp")
        write, fh = _make_writer(path_tmp)
        try:
            recs = runner(c, items, on_record=write)
        finally:
            fh.close()
        finalize_predictions_jsonl(path_tmp, path)
        summary = summarize_fn(recs, reference_type=c.dataset.reference_type, protocol=protocol)
        summary["protocolo_ativo"] = protocol
        if protocol_adjustments:
            summary["protocolo_ajustado"] = ajustes[p]
        comparison[p] = summary
    write_baseline_comparison(comparison, run_dir / "baseline_comparison.json")
    metadados = collect_run_metadata(
        cfg_base, config_path=config_path, run_dir=run_dir, n_records=0
    )
    metadados["modo"] = "compare_baselines"
    from llm_evaluation.schema_registry import SUMMARY_SCHEMA_VERSION

    write_summary(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tipo_sumario": "comparacao_baselines",
            "baselines": comparison,
            "protocolos_baseline": protocolos,
            "ajustes_protocolo_baseline": ajustes,
            "metadados_corrida": metadados,
        },
        run_dir / "summary.json",
    )
    manifest = build_manifest(run_dir, metadados=metadados)
    write_manifest(run_dir, manifest)
    print(f"Comparação gravada em {run_dir}")


if __name__ == "__main__":
    main()

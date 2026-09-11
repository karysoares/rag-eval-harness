#!/usr/bin/env python3
"""Exporta comparativos versionáveis para assets/benchmarks/.

Quatro eixos (ver README § Comparativos):
  - interno: evolução de config no mesmo corpus
  - externo: harness vs RAGAS (amostra)
  - calibracao_p0: replay política embedding_e_juiz
  - hitl: Plano C (amostra humana)

Uso:
  uv run python scripts/export_comparatives.py
  uv run python scripts/export_comparatives.py --ragas --ragas-n 25
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "benchmarks"
FIXTURE_HITL = ROOT / "tests" / "fixtures" / "hitl_fairytale_sample"

# Corridas N=1025 documentadas no README (outputs/ gitignored; métricas estáveis).
EIXOS_META: dict[str, dict[str, Any]] = {
    "interno": {
        "planos_kpi": ["A", "B"],
        "pergunta": "Como evolui o harness com calibração de config no mesmo corpus?",
        "condicoes": "FairytaleQA pt-BR validation N=1025, política embedding_e_juiz",
    },
    "externo": {
        "planos_kpi": ["B"],
        "pergunta": "O juiz/embedding do harness alinha-se com faithfulness RAGAS?",
        "condicoes": (
            "Mesma amostra de itens; RAGAS usa LLM externo — diagnóstico, não ground truth"
        ),
    },
    "calibracao_p0": {
        "planos_kpi": ["B"],
        "pergunta": "A política embedding_e_juiz reduz FP vs qualquer_critico?",
        "condicoes": "Replay offline em fixture answer_lists + corrida lexical tuned",
    },
    "hitl": {
        "planos_kpi": ["C"],
        "pergunta": "Juiz e detector concordam com o revisor humano na amostra?",
        "condicoes": "6 itens adjudicados; não extrapolar ao corpus",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hitl_comparative(run_dir: Path) -> dict[str, Any] | None:
    from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
    from llm_evaluation.hitl_io import read_hitl_csv
    from llm_evaluation.hitl_metrics import summarize_hitl

    hitl_csv = run_dir / "analise_manual" / "adjudicacoes_hitl.csv"
    pred = run_dir / "predictions.jsonl"
    if not hitl_csv.is_file() or not pred.is_file():
        return None
    records = load_records_from_predictions_jsonl(pred)
    labels = read_hitl_csv(hitl_csv, strict=True)
    by_id = {r.item_id: r for r in records}
    for iid, lab in labels.items():
        if iid in by_id:
            by_id[iid].meta["adjudicacao_humana"] = lab
    summary = _load_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
    proto = summary.get("protocolo_ativo")
    hitl_sum = summarize_hitl(
        records,
        protocol=proto if isinstance(proto, dict) else None,
    )
    if not hitl_sum:
        return None
    meta = summary.get("metadados_corrida")
    cfg_path = None
    if isinstance(meta, dict) and meta.get("config_path"):
        cfg_path = Path(str(meta["config_path"])).name
    return {
        "run_id": run_dir.name,
        "config_path": cfg_path,
        "n_adjudicados": hitl_sum.get("n_itens_rotulados"),
        "sumario_hitl": hitl_sum,
        "nota": "Amostra humana pequena; não extrapolar ao corpus.",
    }


def _policy_comparative(run_dir: Path) -> dict[str, Any] | None:
    from llm_evaluation.evaluation_metrics import (
        compare_aggregation_policies,
        load_records_from_predictions_jsonl,
    )

    pred = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    if not pred.is_file() or not summary_path.is_file():
        return None
    summary = _load_json(summary_path)
    proto = summary.get("protocolo_ativo")
    if not isinstance(proto, dict):
        return None
    records = load_records_from_predictions_jsonl(pred)
    ref_type = summary.get("tipo_referencia_ativo")
    agg = proto.get("judge_aggregation_verdicts")
    report = compare_aggregation_policies(
        records,
        verify_gold=bool(proto.get("verify_gold")),
        verify_embedding=bool(proto.get("verify_embedding")),
        verify_judge=bool(proto.get("verify_judge")),
        negative_judge_verdicts=[str(x) for x in proto.get("negative_judge_verdicts", [])],
        judge_aggregation_verdicts=[str(x) for x in agg] if isinstance(agg, list) else None,
        reference_type=str(ref_type) if ref_type else "answer_lists",
    )
    mit = report.get("politicas", {}).get("embedding_e_juiz", {})
    fp = mit.get("taxa_falso_alarme_no_gold_correto") if isinstance(mit, dict) else None
    return {
        "run_id": run_dir.name,
        "tipo_referencia": report.get("tipo_referencia"),
        "n_referencia_aceitavel": report.get("n_referencia_aceitavel"),
        "politicas": report.get("politicas"),
        "reducao_fp_relativa": report.get("reducao_fp_relativa_embedding_e_juiz_vs_or"),
        "criterio_p0_sugerido": {
            "aplicavel": ref_type != "none",
            "passou": fp is not None and float(fp) < 0.15,
            "fp_embedding_e_juiz": fp,
        },
    }


def _proveniencia(run_id: str | None) -> dict[str, Any]:
    """Diz se o artefacto que sustenta um número ainda existe no repositório.

    Um comparativo cujo `run_id` já não tem `outputs/run_*/` não é reproduzível por
    terceiros. Marcá-lo é mais honesto do que apagá-lo: o número foi medido, mas
    deixou de ser verificável (invariante 5).
    """
    if not run_id:
        return {"run_id": None, "artefactos_presentes": False, "motivo": "sem run_id"}
    candidatos = [ROOT / "outputs" / run_id, ROOT / "tests" / "fixtures" / run_id]
    for d in candidatos:
        if (d / "summary.json").is_file():
            return {
                "run_id": run_id,
                "artefactos_presentes": True,
                "caminho": str(d.relative_to(ROOT)) + "/",
                "versionado": "tests/fixtures" in str(d),
            }
    return {
        "run_id": run_id,
        "artefactos_presentes": False,
        "motivo": f"outputs/{run_id} nao esta no repositorio; numero medido mas nao reverificavel",
    }


def _harness_lexical(summary: dict[str, Any]) -> dict[str, Any]:
    lex = summary.get("sumario_lexical") or {}
    ret = summary.get("sumario_recuperacao") or {}
    cam = summary.get("analise_camadas") or {}
    gat = cam.get("gatilhos_marginais") or {}
    n = summary.get("n_itens") or 0
    n_juiz_diag = gat.get("n_juiz_diagnostico_negativo")
    juiz_sust = None
    if isinstance(n, int) and n and isinstance(n_juiz_diag, int):
        juiz_sust = 1.0 - (n_juiz_diag / n)
    out: dict[str, Any] = {
        "n_itens": n,
        "media_rouge_l_f": lex.get("media_rouge_l_f"),
        "media_bleu": lex.get("media_bleu"),
        "media_f1_token": lex.get("media_f1_token"),
        "taxa_chunk_ouro_top_k": ret.get("taxa_chunk_ouro_no_top_k"),
        "taxa_juiz_sustentado_diagnostico": juiz_sust,
        "taxa_alerta": summary.get("taxa_alerta"),
    }
    # METEOR sem o corpus wordnet do NLTK só pontua pares quase idênticos, logo a média
    # sai estruturalmente inflacionada. Publica-se com o denominador ou não se publica.
    n_por_metrica = lex.get("n_por_metrica") or {}
    n_met = n_por_metrica.get("meteor") if isinstance(n_por_metrica, dict) else None
    n_pont = lex.get("n_itens_pontuados")
    if isinstance(n_met, int) and isinstance(n_pont, int) and n_met == n_pont and n_met:
        out["media_meteor"] = lex.get("media_meteor")
        out["n_meteor"] = n_met
    elif lex.get("media_meteor") is not None:
        out["meteor_omitido"] = (
            f"media calculada sobre {n_met} de {n_pont} itens; o subconjunto "
            "sobrevivente e o mais facil, logo a media e enviesada para cima"
        )
    out["idioma_normalizacao"] = lex.get("idioma_normalizacao")
    return out


def _ragas_comparative(run_dir: Path, *, n: int) -> dict[str, Any] | None:
    from llm_evaluation.benchmarks.ragas_adapter import (
        run_ragas_sample,
        summarize_harness_grounding,
    )
    from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl

    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        return None
    records = load_records_from_predictions_jsonl(pred)
    sample = records[:n]
    ragas = run_ragas_sample(sample, max_items=n)
    harness = summarize_harness_grounding(sample)
    return {
        "run_id": run_dir.name,
        "n_amostra": len(sample),
        "ragas": ragas,
        "harness_amostra": harness,
        "nota": "RAGAS usa LLM externo; diagnóstico, não ground truth.",
    }


def _write_hitl_fixture(run_dir: Path) -> None:
    from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
    from llm_evaluation.hitl_io import read_hitl_csv

    hitl_csv = run_dir / "analise_manual" / "adjudicacoes_hitl.csv"
    pred = run_dir / "predictions.jsonl"
    if not hitl_csv.is_file() or not pred.is_file():
        return
    FIXTURE_HITL.mkdir(parents=True, exist_ok=True)
    labels = read_hitl_csv(hitl_csv, strict=True)
    records = load_records_from_predictions_jsonl(pred)
    by_id = {r.item_id: r for r in records}
    lines: list[str] = []
    for iid in labels:
        r = by_id.get(iid)
        if not r:
            continue
        row = {
            "id_item": r.item_id,
            "pergunta": r.question,
            "resposta": r.answer,
            "gold_correto": r.gold_correct,
            "flag_anomalia": r.anomaly_flag,
            "sinais": {
                "embedding_baixo_suporte": r.signals.embedding_low_support,
                "embedding_max_coseno": r.signals.embedding_max_cosine,
                "juiz": (
                    {
                        "veredito": r.signals.judge.veredito,
                        "confianca": r.signals.judge.confianca,
                    }
                    if r.signals.judge
                    else None
                ),
                "juiz_negativo": r.signals.judge_negative,
            },
            "meta": {
                "metricas_lexicas": r.meta.get("metricas_lexicas"),
                "adjudicacao_humana": labels[iid],
            },
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    (FIXTURE_HITL / "predictions_subset.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    dest_csv = FIXTURE_HITL / "adjudicacoes_hitl.csv"
    dest_csv.write_text(hitl_csv.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "origem_run_id": run_dir.name,
        "n_itens": len(lines),
        "nota": "Golden HITL amostral para calibração harness vs humano.",
    }
    (FIXTURE_HITL / "README.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tuned",
        type=Path,
        default=ROOT / "outputs" / "run_20260606T121845Z",
        help="Corrida tuned (léxico N=1025)",
    )
    parser.add_argument(
        "--run-hitl",
        type=Path,
        default=ROOT / "outputs" / "run_20260517T190713Z",
        help="Corrida com adjudicacoes_hitl.csv",
    )
    parser.add_argument(
        "--run-policy-ci",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "policy_validation_run",
        help="Fixture CI P0 (answer_lists)",
    )
    parser.add_argument("--ragas", action="store_true", help="Correr RAGAS (requer API)")
    parser.add_argument("--ragas-n", type=int, default=25, help="Tamanho amostra RAGAS")
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    comparativos: dict[str, Any] = {}

    tuned = args.run_tuned.expanduser()
    tuned_policy: dict[str, Any] | None = None
    if tuned.is_dir() and (tuned / "summary.json").is_file():
        s = _load_json(tuned / "summary.json")
        tuned_policy = _policy_comparative(tuned)
        referencia: dict[str, Any] = {
            "eixo": "interno",
            "run_id": tuned.name,
            "config": "configs/ptbr_fairytale_tuned.yaml",
            "harness": _harness_lexical(s),
            "policy": tuned_policy,
            "proveniencia": _proveniencia(tuned.name),
        }
        if args.ragas:
            ragas = _ragas_comparative(tuned, n=args.ragas_n)
            if ragas:
                comparativos["externo_ragas_amostra"] = {"eixo": "externo", **ragas}
        comparativos["referencia_tuned_n1025"] = referencia

    calibracao_casos: list[dict[str, Any]] = []
    pol_ci = args.run_policy_ci.expanduser()
    pol_fixture = _policy_comparative(pol_ci)
    if pol_fixture:
        calibracao_casos.append(
            {
                "label": "fixture_answer_lists",
                **pol_fixture,
                "proveniencia": _proveniencia(str(pol_fixture.get("run_id") or "")),
            }
        )
    if tuned_policy:
        calibracao_casos.append(
            {
                "label": "fairytale_lexical_tuned",
                **tuned_policy,
                "proveniencia": _proveniencia(str(tuned_policy.get("run_id") or "")),
            }
        )
    if calibracao_casos:
        comparativos["calibracao_p0"] = {
            "eixo": "calibracao_p0",
            "casos": calibracao_casos,
            # Proveniência do primeiro caso: é o que um terceiro consegue reproduzir.
            "proveniencia": _proveniencia(str(calibracao_casos[0].get("run_id") or "")),
            "nota": (
                "FP = taxa_falso_alarme_no_gold_correto sob embedding_e_juiz; P0 passa se FP < 15%."
            ),
        }

    hitl_run = args.run_hitl.expanduser()
    hitl = _hitl_comparative(hitl_run)
    if hitl:
        comparativos["hitl_amostra"] = {
            "eixo": "hitl",
            **hitl,
            "proveniencia": _proveniencia(hitl_run.name),
        }
        _write_hitl_fixture(hitl_run)

    # O registo do que foi removido é histórico e não se recalcula: preserva-se.
    anterior: dict[str, Any] = {}
    dest_previa = ASSETS / "comparatives.json"
    if dest_previa.is_file():
        try:
            anterior = json.loads(dest_previa.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            anterior = {}

    out: dict[str, Any] = {
        "schema_version": "2.0",
        "gerado_em_utc": datetime.now(tz=UTC).isoformat(),
        "nota": (
            "Comparativos versionados; regenerar com scripts/export_comparatives.py. "
            "Cada entrada carrega proveniencia.artefactos_presentes: false significa que o "
            "numero foi medido mas a corrida que o sustenta ja nao esta no repositorio."
        ),
        "eixos": EIXOS_META,
        "comparativos": comparativos,
    }
    if isinstance(anterior.get("removidos"), dict):
        out["removidos"] = anterior["removidos"]

    dest = ASSETS / "comparatives.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gravado: {dest}", file=sys.stderr)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Carga de corridas para o dashboard. Sem chamadas à API."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd

from llm_evaluation.evaluation_metrics import (
    compare_metric_reports,
    load_full_report,
    load_records_from_predictions_jsonl,
)
from llm_evaluation.types import RunRecord

# Colunas esperadas no DataFrame (fallback NA se ausentes no JSONL).
_DF_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "f1_token",
    "em_squad",
    "exact_match",
    "bleu",
    "rouge_l_f",
    "score_melhor_chunk",
    "rank_chunk_ouro",
    "chunk_ouro_no_top_k",
    "embedding_max_coseno",
    "embedding_max_coseno_ouro",
    "embedding_max_coseno_recuperados",
    "embedding_baixo_suporte",
    "padrao_primario",
    "tier_qualidade",
    "padroes",
    "veredito_juiz",
    "juiz_confianca",
    "juiz_motivo",
    "resp_confianca",
    "resp_contexto_insuficiente",
    "resp_schema_invalid",
    "resp_structured_output_error",
    "critica_schema_invalid",
)

_RUN_ARTIFACT_CACHE: dict[str, tuple[str, dict[str, Any]]] = {}


def outputs_root() -> Path:
    raw = os.environ.get("LLM_EVAL_OUTPUTS", "outputs")
    return Path(raw).expanduser().resolve()


def list_run_dirs(base: Path | None = None) -> list[Path]:
    root = base or outputs_root()
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def predictions_path(run_dir: Path) -> Path | None:
    primary = run_dir / "predictions.jsonl"
    if primary.is_file():
        return primary
    candidates = sorted(run_dir.glob("predictions_*.jsonl"))
    return candidates[0] if candidates else None


def load_run_records(run_dir: Path) -> list[RunRecord]:
    path = predictions_path(run_dir)
    if path is None:
        msg = f"Sem predictions*.jsonl em {run_dir}"
        raise FileNotFoundError(msg)
    return load_records_from_predictions_jsonl(path)


def load_run_report(run_dir: Path) -> dict[str, Any]:
    return load_full_report(run_dir)


def fila_revisao_csv_path(run_dir: Path) -> Path | None:
    """CSV da fila humana (gerado na corrida ou via script)."""
    manual = run_dir / "analise_manual" / "fila_revisao_humana.csv"
    if manual.is_file():
        return manual
    report = load_run_report(run_dir)
    op = report.get("sumario_operacional")
    if isinstance(op, dict):
        raw = op.get("fila_revisao_csv")
        if raw:
            p = Path(str(raw))
            if p.is_file():
                return p
    return None


def load_fila_revisao_dataframe(run_dir: Path) -> pd.DataFrame | None:
    path = fila_revisao_csv_path(run_dir)
    if path is None:
        return None
    return pd.read_csv(path)


def _diagnostico(r: RunRecord) -> dict[str, Any]:
    d = r.meta.get("diagnostico")
    return d if isinstance(d, dict) else {}


def _qualidade_geracao(r: RunRecord) -> dict[str, Any]:
    q = r.meta.get("qualidade_geracao")
    return q if isinstance(q, dict) else {}


# Ordem sugerida para inspecção / calibração no dashboard.
CALIBRATION_COLUMN_ORDER: tuple[str, ...] = (
    "id_item",
    "flag_anomalia",
    "gold_correto",
    "gold_incorreto",
    "f1_token",
    "em_squad",
    "exact_match",
    "score_melhor_chunk",
    "rank_chunk_ouro",
    "chunk_ouro_no_top_k",
    "n_chunks_recuperados",
    "embedding_max_coseno",
    "embedding_max_coseno_recuperados",
    "embedding_max_coseno_ouro",
    "embedding_baixo_suporte",
    "veredito_juiz",
    "juiz_negativo",
    "juiz_confianca",
    "juiz_fallback",
    "juiz_retry_count",
    "juiz_parse_failures",
    "juiz_schema_invalid",
    "juiz_tokens_contexto",
    "juiz_motivo",
    "resp_confianca",
    "resp_contexto_insuficiente",
    "resp_schema_invalid",
    "resp_structured_output_error",
    "critica_schema_invalid",
    "padrao_primario",
    "padroes",
    "tier_qualidade",
    "e_recusa",
    "curada_recuperacao_fraca",
    "pergunta",
    "resposta",
    "texto_referencia",
    "bleu",
    "rouge_l_f",
    "perfil_baseline",
)


def records_to_dataframe(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        rm = r.meta.get("metricas_recuperacao") or r.meta.get("retrieval_metrics") or {}
        lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics") or {}
        diag = _diagnostico(r)
        refs = r.meta.get("referencias") or r.meta.get("references") or []
        texto_ref = ""
        if isinstance(lm, dict):
            texto_ref = str(lm.get("texto_referencia") or lm.get("reference_text") or "")
        qg = _qualidade_geracao(r)
        crit = r.meta.get("critica")
        crit = crit if isinstance(crit, dict) else {}
        padroes = diag.get("padroes") or []
        j = r.signals.judge
        ctx_j = r.meta.get("contexto_juiz")
        ctx_j = ctx_j if isinstance(ctx_j, dict) else {}
        rows.append(
            {
                "id_item": r.item_id,
                "pergunta": r.question,
                "resposta": r.answer,
                "gold_correto": r.gold_correct,
                "gold_incorreto": r.signals.gold_incorrect,
                "flag_anomalia": r.anomaly_flag,
                "embedding_max_coseno": r.signals.embedding_max_cosine,
                "embedding_max_coseno_ouro": r.signals.embedding_max_cosine_gold,
                "embedding_max_coseno_recuperados": r.signals.embedding_max_cosine_retrieved,
                "embedding_baixo_suporte": r.signals.embedding_low_support,
                "juiz_negativo": r.signals.judge_negative,
                "veredito_juiz": j.veredito if j else None,
                "juiz_confianca": j.confianca if j else None,
                "juiz_motivo": j.motivo_breve if j else None,
                "juiz_fallback": bool(j and j.raw.get("fallback_heuristico")),
                "juiz_retry_count": ctx_j.get("retry_count"),
                "juiz_parse_failures": ctx_j.get("parse_failures"),
                "juiz_schema_invalid": ctx_j.get("schema_invalid"),
                "juiz_tokens_contexto": ctx_j.get("tokens_estimados"),
                "score_melhor_chunk": rm.get("score_melhor_chunk")
                if isinstance(rm, dict)
                else None,
                "rank_chunk_ouro": rm.get("rank_chunk_ouro") if isinstance(rm, dict) else None,
                "chunk_ouro_no_top_k": (
                    rm.get("chunk_ouro_no_top_k") if isinstance(rm, dict) else None
                ),
                "corpus_tem_chunk_ouro": rm.get("corpus_tem_chunk_ouro")
                if isinstance(rm, dict)
                else None,
                "n_chunks_recuperados": rm.get("n_chunks_recuperados")
                if isinstance(rm, dict)
                else None,
                "bleu": lm.get("bleu") if isinstance(lm, dict) else None,
                "rouge_l_f": lm.get("rouge_l_f") or lm.get("rouge_l_fmeasure")
                if isinstance(lm, dict)
                else None,
                "f1_token": lm.get("f1_token") if isinstance(lm, dict) else None,
                "em_squad": lm.get("em_squad") if isinstance(lm, dict) else None,
                "exact_match": lm.get("exact_match") if isinstance(lm, dict) else None,
                "perfil_baseline": r.baseline_profile,
                "padrao_primario": diag.get("padrao_primario"),
                "tier_qualidade": diag.get("tier_qualidade"),
                "padroes": padroes,
                "texto_referencia": texto_ref,
                "referencias": refs,
                "e_recusa": r.signals.is_refusal,
                "curada_recuperacao_fraca": bool(qg.get("curada_por_recuperacao_fraca")),
                "resp_confianca": qg.get("confianca"),
                "resp_contexto_insuficiente": qg.get("contexto_insuficiente"),
                "resp_schema_invalid": bool(qg.get("schema_invalid")),
                "resp_structured_output_error": qg.get("structured_output_error"),
                "critica_schema_invalid": bool(crit.get("schema_invalid")),
            },
        )
    df = pd.DataFrame(rows)
    for col in _DF_OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def calibration_view_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame para tabela de calibração: colunas ordenadas + `padroes` legível."""
    if df.empty:
        return df
    out = df.copy()
    if "padroes" in out.columns:
        out["padroes"] = out["padroes"].apply(
            lambda x: ", ".join(str(t) for t in x) if isinstance(x, list) else str(x or ""),
        )
    cols = [c for c in CALIBRATION_COLUMN_ORDER if c in out.columns]
    rest = [c for c in out.columns if c not in cols and c != "referencias"]
    return out[cols + rest]


def load_summary_json(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_manifest_json(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def artifact_fingerprint(run_dir: Path) -> str:
    """Chave de cache: SHA256 do manifest ou concatenação mtime+tamanho dos artefactos."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        from llm_evaluation.run_artifacts import sha256_file

        return sha256_file(manifest_path)
    h = hashlib.sha256()
    for name in ("predictions.jsonl", "summary.json"):
        p = run_dir / name
        if p.is_file():
            st = p.stat()
            h.update(f"{name}:{st.st_mtime_ns}:{st.st_size}".encode())
    pred_alt = predictions_path(run_dir)
    if pred_alt is not None and pred_alt.name != "predictions.jsonl":
        st = pred_alt.stat()
        h.update(f"{pred_alt.name}:{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()


def run_integrity_flags(run_dir: Path) -> dict[str, Any]:
    """Flags de integridade para o dashboard (SPEC-005/006 Fase 1)."""
    from llm_evaluation.dashboard.schema_validation import validate_run_schemas

    manifest = load_manifest_json(run_dir)
    summary = load_summary_json(run_dir)
    validation = validate_run_schemas(run_dir)
    out: dict[str, Any] = {
        "tem_manifest": manifest is not None,
        "tem_summary": summary is not None,
        "tem_predictions": predictions_path(run_dir) is not None,
        "schema_version_summary": (summary or {}).get("schema_version"),
        "schema_version_manifest": (manifest or {}).get("schema_version") if manifest else None,
        "checksums_ok": None,
        "ficheiros_em_falta": [],
        "escrita_parcial": False,
        "legacy_run": validation.get("legacy_run", False),
        "schema_mismatch": validation.get("schema_mismatch", False),
        "n_warnings_schema": len(validation.get("warnings") or []),
    }

    tmp_pred = run_dir / "predictions.jsonl.tmp"
    tmp_sum = run_dir / "summary.json.tmp"
    if tmp_pred.is_file() or tmp_sum.is_file():
        out["escrita_parcial"] = True

    if not manifest:
        out["integrity_score"] = _integrity_score(out)
        return out

    from llm_evaluation.run_artifacts import sha256_file

    ok = True
    missing: list[str] = []
    for fe in manifest.get("ficheiros") or []:
        if not isinstance(fe, dict):
            continue
        nome = fe.get("nome")
        expected = fe.get("sha256")
        fp = run_dir / str(nome)
        if not fp.is_file():
            missing.append(str(nome))
            ok = False
            continue
        if expected and sha256_file(fp) != expected:
            ok = False
    out["checksums_ok"] = ok and not missing
    out["ficheiros_em_falta"] = missing
    out["n_ficheiros_manifest"] = len(manifest.get("ficheiros") or [])
    out["git_commit"] = (manifest.get("metadados_corrida") or {}).get("git_commit")
    out["config_hash_sha256"] = (manifest.get("metadados_corrida") or {}).get("config_hash_sha256")
    out["integrity_score"] = _integrity_score(out)
    return out


def _integrity_score(flags: dict[str, Any]) -> int:
    """Pontuação 0–100 para badges na sidebar (heurística, não bloqueia leitura)."""
    score = 100
    if not flags.get("tem_predictions"):
        score -= 50
    if not flags.get("tem_summary"):
        score -= 15
    if not flags.get("tem_manifest"):
        score -= 10
    if flags.get("escrita_parcial"):
        score -= 25
    if flags.get("checksums_ok") is False:
        score -= 35
    if flags.get("ficheiros_em_falta"):
        score -= min(30, 10 * len(flags["ficheiros_em_falta"]))
    if flags.get("schema_mismatch"):
        score -= 15
    if flags.get("legacy_run"):
        score -= 5
    return max(0, min(100, score))


def cache_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Cache em memória invalidado quando manifest ou mtimes mudam."""
    from llm_evaluation.dashboard.schema_validation import validate_run_schemas

    key = str(run_dir.resolve())
    fp = artifact_fingerprint(run_dir)
    cached = _RUN_ARTIFACT_CACHE.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1]

    records = load_run_records(run_dir)
    bundle: dict[str, Any] = {
        "records": records,
        "report": load_run_report(run_dir),
        "summary": load_summary_json(run_dir),
        "manifest": load_manifest_json(run_dir),
        "integrity": run_integrity_flags(run_dir),
        "validation": validate_run_schemas(run_dir),
        "fingerprint": fp,
    }
    _RUN_ARTIFACT_CACHE[key] = (fp, bundle)
    return bundle


def clear_run_artifact_cache() -> None:
    """Útil em testes."""
    _RUN_ARTIFACT_CACHE.clear()


def compare_runs(run_dirs: list[Path]) -> dict[str, Any]:
    reports = [load_full_report(d) for d in run_dirs]
    labels = [d.name for d in run_dirs]
    return compare_metric_reports(reports, labels)

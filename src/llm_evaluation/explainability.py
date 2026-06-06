"""Explicabilidade determinística do harness (SPEC-009)."""

from __future__ import annotations

from typing import Any

from llm_evaluation.config import AppConfig
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.aggregate import judge_negative_for_aggregation


def _active_alert_signals(record: RunRecord, cfg: AppConfig | None) -> list[str]:
    s = record.signals
    out: list[str] = []
    if cfg is None:
        if record.anomaly_flag:
            out.append("flag_anomalia")
        return out
    if cfg.verification.verify_gold and s.gold_incorrect is True:
        out.append("gold_incorreto")
    if cfg.verification.verify_embedding and s.embedding_low_support is True:
        out.append("embedding_baixo_suporte")
    if (
        cfg.verification.verify_judge
        and s.judge is not None
        and not s.judge.raw.get("fallback_heuristico")
        and judge_negative_for_aggregation(s, cfg.verification.judge_aggregation_verdicts)
    ):
        out.append("juiz_nao_sustentado")
    return out


def build_explicacao(record: RunRecord, *, cfg: AppConfig | None = None) -> dict[str, Any]:
    """Narrativa curta a partir de sinais gravados — sem inventar factos."""
    s = record.signals
    policy = cfg.aggregation.policy if cfg else "desconhecida"
    rm = record.meta.get("metricas_recuperacao") or record.meta.get("retrieval_metrics") or {}
    lm = record.meta.get("metricas_lexicas") or record.meta.get("lexical_metrics") or {}
    diag_raw = record.meta.get("diagnostico")
    diag = diag_raw if isinstance(diag_raw, dict) else {}

    rec_resumo = "Sem métricas de recuperação."
    if isinstance(rm, dict) and rm:
        rank = rm.get("rank_chunk_ouro")
        sc = rm.get("score_melhor_chunk")
        top = rm.get("chunk_ouro_no_top_k")
        rec_resumo = f"Ouro no top-k: {top}; rank ouro: {rank}; score top: {sc}."

    lex: dict[str, Any] = {}
    if isinstance(lm, dict):
        lex = {
            "f1_token": lm.get("f1_token"),
            "em_squad": lm.get("em_squad"),
            "referencia_usada": (record.meta.get("referencias") or [""])[0]
            if isinstance(record.meta.get("referencias"), list)
            else None,
        }

    juiz_blk: dict[str, Any] = {}
    if s.judge is not None:
        juiz_blk = {
            "veredito": s.judge.veredito,
            "motivo": s.judge.motivo_breve,
            "fallback": bool(s.judge.raw.get("fallback_heuristico")),
        }

    conflitos: list[str] = []
    if s.embedding_low_support and s.judge is not None and s.judge.veredito == "sustentado":
        conflitos.append("embedding_baixo_com_juiz_sustentado")
    if s.gold_correct and s.embedding_low_support:
        conflitos.append("gold_correto_com_embedding_baixo")

    return {
        "alerta": {
            "flag_anomalia": record.anomaly_flag,
            "politica": policy,
            "sinais_activos": _active_alert_signals(record, cfg),
        },
        "recuperacao": {
            "resumo": rec_resumo,
            **(
                {
                    k: rm.get(k)
                    for k in (
                        "rank_chunk_ouro",
                        "score_melhor_chunk",
                        "chunk_ouro_no_top_k",
                    )
                }
                if isinstance(rm, dict)
                else {}
            ),
        },
        "lexical": lex,
        "juiz": juiz_blk,
        "padrao_primario": diag.get("padrao_primario"),
        "rationale_padroes": diag.get("rationale", []),
        "conflitos": conflitos,
    }


def explicacao_resumida(record: RunRecord) -> str:
    exp = record.meta.get("explicacao")
    if not isinstance(exp, dict):
        return ""
    alerta = exp.get("alerta")
    if isinstance(alerta, dict):
        sinais = alerta.get("sinais_activos")
        if isinstance(sinais, list) and sinais:
            return f"Alerta: {', '.join(str(x) for x in sinais)}"
    rec = exp.get("recuperacao")
    if isinstance(rec, dict) and rec.get("resumo"):
        return str(rec["resumo"])[:200]
    return ""


def summarize_explicabilidade(records: list[RunRecord]) -> dict[str, object] | None:
    with_exp = [r for r in records if isinstance(r.meta.get("explicacao"), dict)]
    if not with_exp:
        return None
    n_conf = sum(
        1
        for r in with_exp
        if isinstance(r.meta.get("explicacao"), dict)
        and isinstance((r.meta["explicacao"]).get("conflitos"), list)
        and (r.meta["explicacao"])["conflitos"]
    )
    return {
        "n_com_explicacao": len(with_exp),
        "taxa_com_conflito": n_conf / len(with_exp) if with_exp else None,
        "nota": "Explicabilidade do harness; não interpretabilidade do LLM gerador.",
    }

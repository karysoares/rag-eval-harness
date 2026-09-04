"""Registo de versões de schema para artefactos de corrida (SPEC-005, Fase 1)."""

from __future__ import annotations

from typing import Any

PREDICTIONS_SCHEMA_VERSION = "1.1"
PREDICTIONS_SCHEMA_VERSIONS_OK = frozenset({"1.0", "1.1"})
SUMMARY_SCHEMA_VERSION = "1.2"
SUMMARY_SCHEMA_VERSIONS_OK = frozenset({"1.0", "1.1", "1.2"})
MANIFEST_SCHEMA_VERSION = "1.0"

# Campos de topo em cada linha de predictions.jsonl (v1.0).
KNOWN_PREDICTION_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "id_item",
        "pergunta",
        "resposta",
        "gold_correto",
        "flag_anomalia",
        "perfil_baseline",
        "sinais",
        "recuperados",
        "meta",
        "referencias",
        "diagnostico",
        # legado em inglês (leitura retrocompatível)
        "item_id",
        "question",
        "answer",
        "gold_correct",
        "anomaly_flag",
        "baseline_profile",
        "signals",
        "retrieved",
        "references",
    },
)

# Subconjunto mínimo obrigatório em corridas novas.
REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "id_item",
        "pergunta",
        "resposta",
        "sinais",
        "meta",
    },
)

KNOWN_SUMMARY_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "tipo_sumario",
        "metadados_corrida",
        "tipo_referencia_ativo",
        "kpi_primario",
        "kpi_diagnostico_primario",
        "detector_activo",
        "taxa_alerta",
        "n_itens",
        "n_itens_avaliados",
        "n_itens_com_erro_execucao",
        "nota_exclusao",
        "n_com_gold_para_confusao",
        "n_sem_rotulo_gold",
        "n_anomalias_marcadas",
        "n_gold_incorretos",
        "n_gold_corretos",
        "confusao_vs_referencia",
        "confusao_vs_gold",
        "estratificacao_fp_gold_correto",
        "qualidade_pipeline",
        "precisao_anomalia_vs_gold_incorreto",
        "revocacao_anomalia_vs_gold_incorreto",
        "acuracia_balanceada_gold",
        "cohen_kappa_anomalia_vs_gold",
        "analise_camadas",
        "aviso_metricas",
        "sumario_recuperacao",
        "sumario_lexical",
        "sumario_padroes",
        "sumario_juiz",
        "sumario_gap_rag_resposta",
        "sumario_operacional",
        "observabilidade",
        "configuracao",
        "orquestracao",
        "perfil_baseline",
        "protocolo_ativo",
        "protocolo_ajustado",
        "revocacao_marcacao_no_gold_incorreto",
        "ic95_revocacao_marcacao_no_gold_incorreto",
        "taxa_falso_alarme_no_gold_correto",
        "ic95_taxa_falso_alarme_no_gold_correto",
        "ic95_precisao_anomalia_vs_gold_incorreto",
        "ic95_revocacao_anomalia_vs_gold_incorreto",
        "baselines",
        "sumario_hitl",
        "sumario_explicabilidade",
        "proveniencia",
        "avisos_protocolo",
        "avisos_reprocessamento",
    },
)

KNOWN_MANIFEST_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "criado_em_utc",
        "metadados_corrida",
        "dependencias_specs",
        "ficheiros",
        "integridade",
    },
)

MIGRATION_NOTES: dict[str, str] = {
    "predictions": (
        "v1.0→v1.1: schema_version por linha; v1.0 continua legível. "
        "v1.1 alinha metadados structured output em meta.qualidade_geracao / contexto_juiz."
    ),
    "summary": (
        "v1.0→v1.1: schema_version incrementado; campos KPI inalterados. "
        "fila_revisao_csv passa a path relativo ao run_dir. v1.0 continua legível. "
        "v1.1→v1.2: itens com processing_error saem de todos os denominadores "
        "(CLAUDE.md regra 3). n_itens continua a ser o total recebido; "
        "n_itens_avaliados é o denominador efectivo e n_itens_com_erro_execucao "
        "a diferença. taxa_alerta de v1.1 e anterior não é comparável com v1.2 "
        "em corridas que tiveram falhas de execução."
    ),
    "manifest": "v1.0: manifest.json opcional; ausência não invalida corridas antigas.",
}


def validate_prediction_record(obj: dict[str, Any], *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    ver = obj.get("schema_version")
    if ver is not None and ver not in PREDICTIONS_SCHEMA_VERSIONS_OK:
        issues.append(f"schema_version inesperado em predictions: {ver!r}")
    missing = REQUIRED_PREDICTION_FIELDS - set(obj.keys())
    if missing:
        issues.append(f"campos obrigatórios em falta: {sorted(missing)}")
    unknown = set(obj.keys()) - KNOWN_PREDICTION_TOP_FIELDS
    if unknown:
        msg = f"campos desconhecidos em predictions: {sorted(unknown)}"
        if strict:
            issues.append(msg)
        else:
            issues.append(f"aviso: {msg}")
    return issues


def validate_summary(obj: dict[str, Any], *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    if obj.get("tipo_sumario") == "comparacao_baselines" or (
        "baselines" in obj and "tipo_referencia_ativo" not in obj
    ):
        ver = obj.get("schema_version")
        if ver is None:
            issues.append("aviso: summary de comparação sem schema_version")
        elif ver is not None and ver not in SUMMARY_SCHEMA_VERSIONS_OK:
            issues.append(f"schema_version inesperado em summary: {ver!r}")
        if "metadados_corrida" not in obj:
            issues.append("summary comparação sem metadados_corrida")
        return issues
    ver = obj.get("schema_version")
    if ver is None:
        issues.append("aviso: summary sem schema_version (corrida legada)")
    elif ver is not None and ver not in SUMMARY_SCHEMA_VERSIONS_OK:
        issues.append(f"schema_version inesperado em summary: {ver!r}")
    unknown = set(obj.keys()) - KNOWN_SUMMARY_TOP_FIELDS
    if unknown:
        msg = f"campos desconhecidos em summary: {sorted(unknown)}"
        if strict:
            issues.append(msg)
        else:
            issues.append(f"aviso: {msg}")
    issues.extend(validate_protocolo_ativo(obj.get("protocolo_ativo"), strict=strict))
    return issues


PROTOCOLO_REQUIRED_KEYS = frozenset(
    {
        "verify_gold",
        "verify_embedding",
        "verify_judge",
        "aggregation_policy",
        "embedding_min_cosine",
        "negative_judge_verdicts",
        "judge_aggregation_verdicts",
    },
)


def validate_protocolo_ativo(
    proto: object,
    *,
    strict: bool = False,
) -> list[str]:
    issues: list[str] = []
    if proto is None:
        if strict:
            issues.append("protocolo_ativo em falta no summary")
        else:
            issues.append("aviso: protocolo_ativo em falta no summary")
        return issues
    if not isinstance(proto, dict):
        issues.append("protocolo_ativo deve ser um objecto")
        return issues
    missing = PROTOCOLO_REQUIRED_KEYS - set(proto.keys())
    if missing:
        msg = f"protocolo_ativo incompleto: faltam {sorted(missing)}"
        if strict:
            issues.append(msg)
        else:
            issues.append(f"aviso: {msg}")
    issues.extend(_validate_verdict_lists(proto, strict=strict))
    return issues


def _validate_verdict_lists(proto: dict[str, object], *, strict: bool) -> list[str]:
    from llm_evaluation.veredito import parse_veredito_estrito

    issues: list[str] = []
    for key in ("negative_judge_verdicts", "judge_aggregation_verdicts"):
        raw = proto.get(key)
        if not isinstance(raw, list):
            continue
        if not raw:
            msg = f"protocolo_ativo.{key} não pode ser lista vazia"
            issues.append(msg if strict else f"aviso: {msg}")
            continue
        for item in raw:
            if parse_veredito_estrito(str(item)) is None:
                msg = f"protocolo_ativo.{key}: veredito desconhecido {item!r}"
                issues.append(msg if strict else f"aviso: {msg}")
    return issues


def validate_manifest(obj: dict[str, Any], *, strict: bool = False) -> list[str]:
    issues: list[str] = []
    ver = obj.get("schema_version")
    if ver is not None and ver != MANIFEST_SCHEMA_VERSION:
        issues.append(f"schema_version inesperado em manifest: {ver!r}")
    unknown = set(obj.keys()) - KNOWN_MANIFEST_TOP_FIELDS
    if unknown:
        msg = f"campos desconhecidos em manifest: {sorted(unknown)}"
        if strict:
            issues.append(msg)
        else:
            issues.append(f"aviso: {msg}")
    return issues

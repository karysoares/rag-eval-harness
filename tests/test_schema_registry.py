from llm_evaluation.schema_registry import (
    PREDICTIONS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    validate_prediction_record,
    validate_protocolo_ativo,
    validate_summary,
)


def test_validate_prediction_record_ok() -> None:
    obj = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "id_item": "1",
        "pergunta": "q",
        "resposta": "a",
        "sinais": {},
        "meta": {},
    }
    assert validate_prediction_record(obj) == []


def test_validate_prediction_record_missing_required() -> None:
    issues = validate_prediction_record({"id_item": "1"})
    assert any("obrigatórios" in i for i in issues)


def test_validate_summary_legacy_warning() -> None:
    issues = validate_summary({"n_itens": 1})
    assert any("sem schema_version" in i for i in issues)


def test_validate_summary_versioned() -> None:
    issues = validate_summary({"schema_version": SUMMARY_SCHEMA_VERSION, "n_itens": 1})
    assert not any("schema_version inesperado" in i for i in issues)


def test_validate_summary_accepts_warning_fields_strict() -> None:
    issues = validate_summary(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "avisos_protocolo": ["judge igual ao gerador"],
            "avisos_reprocessamento": ["reference_type inferido"],
            "protocolo_ativo": {
                "verify_gold": False,
                "verify_embedding": True,
                "verify_judge": True,
                "aggregation_policy": "embedding_e_juiz",
                "embedding_min_cosine": 0.28,
                "negative_judge_verdicts": ["nao_sustentado"],
                "judge_aggregation_verdicts": ["nao_sustentado"],
            },
        },
        strict=True,
    )
    assert issues == []


def test_validate_summary_baseline_comparison() -> None:
    issues = validate_summary(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tipo_sumario": "comparacao_baselines",
            "baselines": {"nenhum": {}},
            "metadados_corrida": {},
        },
    )
    assert issues == []


def test_validate_protocolo_ativo_strict() -> None:
    proto = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "embedding_min_cosine": 0.28,
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": ["nao_sustentado"],
    }
    assert validate_protocolo_ativo(proto, strict=True) == []
    assert any("faltam" in i for i in validate_protocolo_ativo({}, strict=True))
    assert validate_protocolo_ativo(None, strict=False)[0].startswith("aviso:")
    empty_agg = {
        "verify_gold": False,
        "verify_embedding": True,
        "verify_judge": True,
        "aggregation_policy": "embedding_e_juiz",
        "embedding_min_cosine": 0.28,
        "negative_judge_verdicts": ["nao_sustentado"],
        "judge_aggregation_verdicts": [],
    }
    assert any("vazia" in i for i in validate_protocolo_ativo(empty_agg, strict=True))


def test_validate_summary_avisos_protocolo_strict() -> None:
    base = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "n_itens": 1,
        "protocolo_ativo": {
            "verify_gold": False,
            "verify_embedding": True,
            "verify_judge": True,
            "aggregation_policy": "embedding_e_juiz",
            "embedding_min_cosine": 0.28,
            "negative_judge_verdicts": ["nao_sustentado"],
            "judge_aggregation_verdicts": ["nao_sustentado"],
        },
    }
    issues = validate_summary({**base, "avisos_protocolo": ["ajuste"]}, strict=True)
    assert not any("desconhecidos" in i for i in issues)


def test_validate_summary_avisos_reprocessamento_strict() -> None:
    base = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "n_itens": 1,
        "protocolo_ativo": {
            "verify_gold": False,
            "verify_embedding": True,
            "verify_judge": True,
            "aggregation_policy": "embedding_e_juiz",
            "embedding_min_cosine": 0.28,
            "negative_judge_verdicts": ["nao_sustentado"],
            "judge_aggregation_verdicts": ["nao_sustentado"],
        },
    }
    issues = validate_summary({**base, "avisos_reprocessamento": ["inferido"]}, strict=True)
    assert not any("desconhecidos" in i for i in issues)

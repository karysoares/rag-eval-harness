"""Validação de schema do juiz (SPEC-003, Fase 1)."""

from __future__ import annotations

import pytest

from llm_evaluation.verification.judge_schema import (
    JUDGE_SCHEMA_VERSION,
    MOTIVO_BREVE_MAX_LEN,
    JudgeSchemaError,
    sanitize_motivo_breve,
    validate_judge_response,
)


def test_validate_ok() -> None:
    v = validate_judge_response(
        {
            "veredito": "sustentado",
            "motivo_breve": "ok",
            "confianca": 0.9,
        },
        log_violations=False,
    )
    assert v.veredito == "sustentado"
    assert v.confianca == 0.9
    assert v.schema_version == JUDGE_SCHEMA_VERSION


def test_validate_english_aliases() -> None:
    v = validate_judge_response(
        {
            "verdict": "unsupported",
            "reason_short": "no support",
            "confidence": 0.7,
        },
        log_violations=False,
    )
    assert v.veredito == "nao_sustentado"


def test_validate_rejects_empty_motivo() -> None:
    with pytest.raises(JudgeSchemaError, match="motivo_breve"):
        validate_judge_response(
            {"veredito": "sustentado", "motivo_breve": "", "confianca": 0.5},
            log_violations=False,
        )


def test_validate_rejects_confidence_out_of_range() -> None:
    with pytest.raises(JudgeSchemaError, match="confianca"):
        validate_judge_response(
            {"veredito": "sustentado", "motivo_breve": "x", "confianca": 1.5},
            log_violations=False,
        )


def test_validate_rejects_unknown_verdict() -> None:
    with pytest.raises(JudgeSchemaError, match="veredito fora do enum"):
        validate_judge_response(
            {"veredito": "totalmente_invalido", "motivo_breve": "x", "confianca": 0.5},
            log_violations=False,
        )


def test_sanitize_truncates_long_motivo() -> None:
    long = "a" * (MOTIVO_BREVE_MAX_LEN + 50)
    out = sanitize_motivo_breve(long)
    assert len(out) == MOTIVO_BREVE_MAX_LEN


def test_validate_rejects_non_dict_parsed_via_judge_flow() -> None:
    """Dict inválido (veredito em falta) deve ser rejeitado antes de KPI."""
    with pytest.raises(JudgeSchemaError):
        validate_judge_response(
            {"motivo_breve": "x", "confianca": 0.5},
            log_violations=False,
        )

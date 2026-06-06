"""Validação de schema da crítica multi-agente."""

from __future__ import annotations

import pytest

from llm_evaluation.critic_schema import (
    CRITIC_SCHEMA_VERSION,
    CriticSchemaError,
    validate_critic_response,
)


def test_validate_critic_ok() -> None:
    v = validate_critic_response(
        {
            "cadeia_de_pensamento": ["pedido", "evidência", "conclusão"],
            "problemas": ["nenhum"],
            "nota": "ok",
        },
        log_violations=False,
    )
    assert v.problemas == ["nenhum"]
    assert v.schema_version == CRITIC_SCHEMA_VERSION


def test_validate_critic_rejects_invalid_problema() -> None:
    with pytest.raises(CriticSchemaError, match="problemas fora do enum"):
        validate_critic_response(
            {
                "cadeia_de_pensamento": ["a"],
                "problemas": ["inventado"],
                "nota": "x",
            },
            log_violations=False,
        )


def test_validate_critic_rejects_empty_cadeia() -> None:
    with pytest.raises(CriticSchemaError, match="cadeia_de_pensamento"):
        validate_critic_response(
            {
                "cadeia_de_pensamento": [],
                "problemas": ["nenhum"],
                "nota": "x",
            },
            log_violations=False,
        )

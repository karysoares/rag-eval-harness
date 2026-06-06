"""Testes de integração opt-in (API / Hub). Não correm no CI por defeito."""

from __future__ import annotations

import os

import pytest

from llm_evaluation.adapters.amostra_local import amostra_local_items
from llm_evaluation.llm_client import MissingApiKeyError, default_llm_from_env

pytestmark = pytest.mark.integration


def test_amostra_local_offline_shape() -> None:
    items = amostra_local_items()
    assert len(items) >= 2
    assert all(i.question and i.correct_answers for i in items)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", "").strip(),
    reason="OPENAI_API_KEY não definida",
)
def test_openai_minimal_complete() -> None:
    llm = default_llm_from_env(timeout_seconds=60.0, temperature=0.0, max_tokens=8)
    out = llm.complete(
        "Você é um harness de teste.",
        "Responda exatamente com: OK",
    )
    assert isinstance(out, str)
    assert out.strip()


def test_openai_missing_key_raises_when_forced() -> None:
    prev = os.environ.pop("OPENAI_API_KEY", None)
    try:
        os.environ["OPENAI_API_KEY"] = ""
        with pytest.raises(MissingApiKeyError):
            from llm_evaluation.llm_client import require_openai_api_key

            require_openai_api_key()
    finally:
        if prev is not None:
            os.environ["OPENAI_API_KEY"] = prev
        else:
            os.environ.pop("OPENAI_API_KEY", None)

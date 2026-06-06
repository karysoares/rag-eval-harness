import pytest

from llm_evaluation.llm_client import openai_base_url_from_env


def test_openai_base_url_empty_means_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert openai_base_url_from_env() == "https://api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert openai_base_url_from_env() == "https://api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    assert openai_base_url_from_env() == "https://api.openai.com"


def test_openai_base_url_adds_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "api.example.com/v1")
    assert openai_base_url_from_env() == "https://api.example.com/v1"

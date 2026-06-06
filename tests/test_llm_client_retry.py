"""Testes do retry com backoff e dos parâmetros de geração no OpenAiCompatibleClient."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from llm_evaluation.llm_client import OpenAiCompatibleClient


def _client(**overrides: Any) -> OpenAiCompatibleClient:
    base = {
        "api_key": "test",
        "base_url": "https://api.example.com",
        "model": "test-model",
        "timeout_seconds": 1.0,
        "max_retries": 2,
        "backoff_seconds": (0.0, 0.0),
    }
    base.update(overrides)
    return OpenAiCompatibleClient(**base)


def test_payload_includes_temperature_and_max_tokens() -> None:
    c = _client(temperature=0.2, max_tokens=64)
    payload = c._payload("sys", "user")  # noqa: SLF001 — testar privado é aceitável aqui
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64


def test_payload_omits_optional_params_when_unset() -> None:
    c = _client()
    payload = c._payload("sys", "user")  # noqa: SLF001
    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_retries_then_succeeds_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primeiro retorna 503 (transitório), depois 200."""
    calls: list[int] = []

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            return None

        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            calls.append(1)
            request = httpx.Request("POST", url)
            if len(calls) == 1:
                return httpx.Response(503, request=request, text="busy")
            payload = {"choices": [{"message": {"content": "ok"}}]}
            return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    c = _client()
    out = c.complete("sys", "user")
    assert out == "ok"
    assert len(calls) == 2  # 1 erro + 1 sucesso


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class AlwaysFail:
        def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
            pass

        def __enter__(self) -> AlwaysFail:
            return self

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            return None

        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            calls.append(1)
            request = httpx.Request("POST", url)
            return httpx.Response(503, request=request, text="busy")

    monkeypatch.setattr(httpx, "Client", AlwaysFail)
    c = _client(max_retries=1)  # 1 retry => 2 tentativas no total
    with pytest.raises(httpx.HTTPStatusError):
        c.complete("sys", "user")
    assert len(calls) == 2


def test_4xx_non_retriable_propagates_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class Forbidden:
        def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
            pass

        def __enter__(self) -> Forbidden:
            return self

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401
            return None

        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            calls.append(1)
            request = httpx.Request("POST", url)
            return httpx.Response(403, request=request, text="forbidden")

    monkeypatch.setattr(httpx, "Client", Forbidden)
    c = _client(max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        c.complete("sys", "user")
    # 4xx não-transitório não deve retentar
    assert len(calls) == 1

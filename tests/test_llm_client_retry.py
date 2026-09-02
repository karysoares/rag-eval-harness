"""Testes do retry com backoff e dos parâmetros de geração no OpenAiCompatibleClient."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from llm_evaluation.llm_client import OpenAiCompatibleClient, PermanentApiError


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
        is_closed = False

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
        is_closed = False

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
        is_closed = False

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
    with pytest.raises(PermanentApiError) as exc:
        c.complete("sys", "user")
    # 4xx não-transitório não deve retentar
    assert len(calls) == 1
    # e a resposta do fornecedor tem de chegar ao utilizador
    assert "403" in str(exc.value)
    assert "forbidden" in str(exc.value)


def test_erro_permanente_expoe_a_mensagem_json_do_fornecedor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama devolve ``model '...' not found`` num 404 — a causa real do erro."""

    class NotFound:
        is_closed = False

        def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
            pass

        def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> httpx.Response:
            request = httpx.Request("POST", url)
            return httpx.Response(
                404,
                request=request,
                json={"error": {"message": "model 'qwen2.5:7b' not found"}},
            )

    monkeypatch.setattr(httpx, "Client", NotFound)
    with pytest.raises(PermanentApiError, match="model 'qwen2.5:7b' not found"):
        _client().complete("sys", "user")


class TestQuotaVsRateLimit:
    """Saldo esgotado chega como 429, mas nunca recupera com espera.

    Regressão de um incidente real: uma conta sem créditos devolveu
    ``credit_balance_exhausted`` num 429 e cada item foi retentado três vezes
    com 30 s de backoff — 90 s por item para uma falha que era imediata e cuja
    causa (a mensagem do fornecedor) nunca chegava ao utilizador.
    """

    @staticmethod
    def _client_com_resposta(
        monkeypatch: pytest.MonkeyPatch,
        payload: dict[str, Any],
        calls: list[int],
    ) -> OpenAiCompatibleClient:
        class _Resp:
            is_closed = False

            def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
                pass

            def post(
                self,
                url: str,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> httpx.Response:
                calls.append(1)
                return httpx.Response(429, request=httpx.Request("POST", url), json=payload)

        monkeypatch.setattr(httpx, "Client", _Resp)
        return _client(max_retries=3, rate_limit_backoff_seconds=(0.0,))

    def test_saldo_esgotado_falha_a_primeira(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        c = self._client_com_resposta(
            monkeypatch,
            {
                "error": {
                    "message": "You have no credits remaining.",
                    "type": "insufficient_quota",
                    "code": "credit_balance_exhausted",
                }
            },
            calls,
        )
        with pytest.raises(PermanentApiError, match="no credits remaining"):
            c.complete("sys", "user")
        assert len(calls) == 1

    def test_rate_limit_verdadeiro_continua_a_ser_retentado(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        c = self._client_com_resposta(
            monkeypatch,
            {"error": {"message": "Rate limit reached", "type": "rate_limit_error"}},
            calls,
        )
        with pytest.raises(httpx.HTTPStatusError):
            c.complete("sys", "user")
        assert len(calls) == 4  # max_retries=3 + a inicial

    def test_429_sem_corpo_json_e_tratado_como_transitorio(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []

        class _Texto:
            is_closed = False

            def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
                pass

            def post(
                self,
                url: str,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> httpx.Response:
                calls.append(1)
                return httpx.Response(429, request=httpx.Request("POST", url), text="too many")

        monkeypatch.setattr(httpx, "Client", _Texto)
        with pytest.raises(httpx.HTTPStatusError):
            _client(max_retries=1, rate_limit_backoff_seconds=(0.0,)).complete("s", "u")
        assert len(calls) == 2


class TestTemperaturaRejeitada:
    """Modelos que só aceitam a temperatura por omissão.

    Regressão de um caso real: o juiz fixa ``temperature=0`` para determinismo e
    o ``gpt-5-mini`` devolve 400 ``unsupported_value``. Sem retry, todas as
    chamadas caíam no fallback heurístico — que responde ``sustentado`` — e um
    juiz avariado passava por um juiz permissivo.
    """

    @staticmethod
    def _fake(monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any]]) -> None:
        class _Modelo:
            is_closed = False

            def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
                pass

            def post(
                self,
                url: str,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> httpx.Response:
                payloads.append(dict(json))
                request = httpx.Request("POST", url)
                if "temperature" in json:
                    return httpx.Response(
                        400,
                        request=request,
                        json={
                            "error": {
                                "message": "Unsupported value: 'temperature'",
                                "type": "invalid_request_error",
                                "param": "temperature",
                                "code": "unsupported_value",
                            }
                        },
                    )
                return httpx.Response(
                    200,
                    request=request,
                    json={"choices": [{"message": {"content": "ok"}}]},
                )

        monkeypatch.setattr(httpx, "Client", _Modelo)

    def test_repete_sem_temperatura_e_sinaliza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payloads: list[dict[str, Any]] = []
        self._fake(monkeypatch, payloads)
        c = _client(temperature=0.0, max_retries=2)
        assert c.complete("sys", "user") == "ok"
        assert "temperature" in payloads[0]
        assert "temperature" not in payloads[1]
        assert c.temperature_rejected is True

    def test_outro_400_continua_a_falhar_de_imediato(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Só o 400 de ``param: temperature`` é recuperável; os outros propagam."""
        calls: list[int] = []

        class _Outro:
            is_closed = False

            def __init__(self, *a: Any, **kw: Any) -> None:  # noqa: ANN401
                pass

            def post(
                self,
                url: str,
                headers: dict[str, str],
                json: dict[str, Any],
            ) -> httpx.Response:
                calls.append(1)
                return httpx.Response(
                    400,
                    request=httpx.Request("POST", url),
                    json={"error": {"message": "bad model", "param": "model"}},
                )

        monkeypatch.setattr(httpx, "Client", _Outro)
        with pytest.raises(PermanentApiError, match="bad model"):
            _client(temperature=0.0).complete("s", "u")
        assert len(calls) == 1

    def test_sem_temperatura_configurada_nao_ha_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payloads: list[dict[str, Any]] = []
        self._fake(monkeypatch, payloads)
        c = _client(max_retries=2)
        assert c.complete("sys", "user") == "ok"
        assert len(payloads) == 1
        assert c.temperature_rejected is False

"""Cliente LLM compatível com OpenAI, com timeout, parâmetros configuráveis e retry com backoff.

Ver `docs/SECURITY.md` (sem segredos em logs).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx


class MissingApiKeyError(RuntimeError):
    """Erro quando OPENAI_API_KEY não está definida mas uma chamada real ao LLM é necessária."""


class LlmClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def require_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        msg = (
            "É necessário definir OPENAI_API_KEY. Copie .env.example para .env "
            "e configure sua chave "
            "(endpoints compatíveis com OpenAI via OPENAI_BASE_URL são suportados)."
        )
        raise MissingApiKeyError(msg)
    return key


def openai_base_url_from_env() -> str:
    """Resolve OPENAI_BASE_URL: vazio/ausente -> api.openai.com; host sem esquema -> https://."""
    raw = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not raw:
        return "https://api.openai.com"
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")
    return "https://" + raw.lstrip("/").rstrip("/")


@dataclass
class ApiUsageSnapshot:
    """Último uso reportado pela API (chat/completions ``usage``)."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class OpenAiCompatibleClient:
    """Cliente OpenAI-compatible com timeout, parâmetros padrão e retry para erros transitórios."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 4
    backoff_seconds: tuple[float, ...] = field(default_factory=lambda: (1.0, 3.0, 8.0, 20.0))
    rate_limit_backoff_seconds: tuple[float, ...] = field(
        default_factory=lambda: (2.0, 5.0, 15.0, 30.0),
    )
    last_usage: ApiUsageSnapshot | None = field(default=None, init=False, repr=False)

    def _payload(
        self,
        system: str,
        user: str,
        *,
        json_object: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _is_transient_status(code: int) -> bool:
        # 429 e 5xx são transitórios; demais 4xx (auth, payload, etc.) propagam de imediato.
        return code == 429 or 500 <= code < 600

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._payload(system, user, json_object=json_object)

        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    r = client.post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                # Erro de transporte (DNS, TCP, timeout): considerar transitório
                last_exc = exc
                if i >= attempts - 1:
                    break
                self._sleep_backoff(i)
                continue

            if self._is_transient_status(r.status_code):
                last_exc = httpx.HTTPStatusError(
                    f"transient HTTP {r.status_code}", request=r.request, response=r
                )
                if i >= attempts - 1:
                    break
                self._sleep_backoff(i, status_code=r.status_code, response=r)
                continue

            r.raise_for_status()
            data = r.json()
            usage_raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            pt = int(usage_raw.get("prompt_tokens") or 0)
            ct = int(usage_raw.get("completion_tokens") or 0)
            tt = int(usage_raw.get("total_tokens") or pt + ct)
            self.last_usage = ApiUsageSnapshot(
                model=self.model,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
            )
            return str(data["choices"][0]["message"]["content"])

        assert last_exc is not None
        raise last_exc

    def _retry_delay_seconds(
        self,
        i: int,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> float:
        if status_code == 429 and response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 1.0)
                except ValueError:
                    pass
            seq = self.rate_limit_backoff_seconds
            return seq[i] if i < len(seq) else seq[-1]
        seq = self.backoff_seconds
        if not seq:
            return 0.0
        return seq[i] if i < len(seq) else seq[-1]

    def _sleep_backoff(
        self,
        i: int,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        delay = self._retry_delay_seconds(i, status_code=status_code, response=response)
        if delay > 0:
            time.sleep(delay)


def _max_retries_from_env(default: int = 4) -> int:
    raw = os.environ.get("LLM_MAX_RETRIES", str(default)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def resolve_models_from_env() -> tuple[str, str]:
    """Resolve nomes de modelo do gerador e do juiz (env)."""
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
    judge_model = os.environ.get("JUDGE_MODEL", llm_model).strip()
    return llm_model, judge_model


def default_llm_from_env(
    *,
    timeout_seconds: float,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LlmClient:
    """Cliente para o respondedor; respeita `cfg.generation` quando passado."""
    key = require_openai_api_key()
    base = openai_base_url_from_env()
    model, _ = resolve_models_from_env()
    return OpenAiCompatibleClient(
        api_key=key,
        base_url=base,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=_max_retries_from_env(),
    )


def default_judge_from_env(
    *,
    timeout_seconds: float,
    temperature: float = 0.0,
    max_retries: int | None = None,
) -> LlmClient:
    """Cliente para o juiz; por padrão temperatura 0 (determinismo do veredito)."""
    key = require_openai_api_key()
    base = openai_base_url_from_env()
    _, model = resolve_models_from_env()
    retries = _max_retries_from_env() if max_retries is None else max_retries
    return OpenAiCompatibleClient(
        api_key=key,
        base_url=base,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_retries=retries,
    )


_REFUSAL_TOKENS = (
    "não sei",
    "nao sei",
    "não posso",
    "nao posso",
    "sem comentário",
    "sem comentario",
    "não tenho certeza",
    "nao tenho certeza",
    "incapaz",
    "do not know",
    "don't know",
    "cannot answer",
    "no comment",
)


def heuristic_judge_json(answer: str, context: str) -> dict[str, Any]:
    """Fallback determinístico quando a chamada ao juiz falha (timeout/parse).

    Política conservadora — só sinaliza problema quando a resposta é claramente
    vazia, demasiado curta ou uma recusa explícita. Caso contrário devolve
    ``sustentado`` com confiança baixa, deixando claro que é fallback.

    Não há regras de domínio (factualidade fica delegada ao juiz real e ao gold).
    """
    a = answer.strip()
    a_lower = a.lower()
    if not a:
        return {
            "veredito": "incompleto",
            "motivo_breve": "Resposta vazia (fallback heurístico).",
            "confianca": 0.5,
            "fallback_heuristico": True,
        }
    if len(a) < 5:
        return {
            "veredito": "incompleto",
            "motivo_breve": "Resposta muito curta (fallback heurístico).",
            "confianca": 0.5,
            "fallback_heuristico": True,
        }
    if len(a) < 80 and any(tok in a_lower for tok in _REFUSAL_TOKENS):
        return {
            "veredito": "inseguro",
            "motivo_breve": "Recusa explícita detetada (fallback heurístico).",
            "confianca": 0.5,
            "fallback_heuristico": True,
        }
    return {
        "veredito": "sustentado",
        "motivo_breve": "Sem sinal heurístico de problema (fallback após falha do juiz).",
        "confianca": 0.4,
        "fallback_heuristico": True,
    }


def parse_json_object_lenient(text: str) -> dict[str, Any]:
    """Interpreta um objeto JSON; tolera cercas markdown e texto inicial."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    text = text.strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        out = json.loads(text[start : end + 1])
    if not isinstance(out, dict):
        msg = "Esperado um objeto JSON"
        raise TypeError(msg)
    return cast(dict[str, Any], out)


def parse_judge_json(text: str) -> dict[str, Any]:
    return parse_json_object_lenient(text)

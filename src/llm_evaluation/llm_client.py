"""Cliente LLM compatível com OpenAI, com timeout, parâmetros configuráveis e retry com backoff.

Nunca registe chaves API nem conteúdo sensível em logs.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import httpx

#: Ligações do pool HTTP quando a concorrência não é conhecida (chamadas avulsas).
DEFAULT_MAX_CONNECTIONS = 32


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


def _normalize_base_url(raw: str, *, default: str) -> str:
    """Normaliza uma base URL: vazia -> ``default``; host sem esquema -> ``https://``."""
    raw = raw.strip()
    if not raw:
        return default
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")
    return "https://" + raw.lstrip("/").rstrip("/")


def openai_base_url_from_env() -> str:
    """Resolve OPENAI_BASE_URL: vazio/ausente -> api.openai.com; host sem esquema -> https://."""
    return _normalize_base_url(
        os.environ.get("OPENAI_BASE_URL", ""),
        default="https://api.openai.com",
    )


def judge_base_url_from_env() -> str:
    """Base URL do juiz: ``JUDGE_BASE_URL``, ou a do gerador quando não definida.

    Permite o par mais útil na prática — gerador numa API paga e juiz num modelo
    local gratuito (Ollama, vLLM). Além do custo, separa as famílias de modelo:
    o repo já avisa quando juiz e gerador partilham o modelo (auto-referência).
    """
    return _normalize_base_url(
        os.environ.get("JUDGE_BASE_URL", ""),
        default=openai_base_url_from_env(),
    )


def judge_api_key_from_env() -> str:
    """Chave do juiz: ``JUDGE_API_KEY``, ou a do gerador quando não definida.

    Endpoints locais ignoram a chave, mas o protocolo exige uma; use qualquer
    valor não vazio (ex.: ``JUDGE_API_KEY=ollama``).
    """
    key = os.environ.get("JUDGE_API_KEY", "").strip()
    return key if key else require_openai_api_key()


def endpoint_host(base_url: str) -> str:
    """``scheme://host[:porta]`` de uma base URL, para registo no protocolo.

    Deixa de fora caminho, query e qualquer credencial embutida — o objectivo é
    identificar o fornecedor no ``summary.json``, não reproduzir a URL completa.
    """
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url
    host = parsed.netloc.rsplit("@", 1)[-1]
    return f"{parsed.scheme}://{host}"


def chat_completions_url(base_url: str) -> str:
    """Monta o endpoint de chat a partir de uma base, sem duplicar segmentos.

    A maioria dos fornecedores compatíveis publica a base **já com** ``/v1``
    (``http://localhost:11434/v1``, ``https://openrouter.ai/api/v1``,
    ``.../compatible-mode/v1``). Concatenar ``/v1/chat/completions`` às cegas
    produzia ``/v1/v1/chat/completions`` e um 404 — que, não sendo transitório,
    falhava sem explicar a causa. Aceitam-se as três formas: base sem sufixo,
    base terminada em ``/v1`` e o endpoint completo.
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


@dataclass
class ApiUsageSnapshot:
    """Último uso reportado pela API (chat/completions ``usage``)."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


#: Padrões de credencial a mascarar em mensagens que possam chegar a artefactos.
_SECRET_PATTERNS = (
    # userinfo numa URL: https://utilizador:senha@host
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+@"),
    # chaves estilo OpenAI/Anthropic e afins
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
)


def redact_secrets(text: str) -> str:
    """Mascara credenciais num texto destinado a logs ou artefactos.

    Mensagens de erro acabam em ``meta.processing_error`` dentro de
    ``predictions.jsonl``, que é um artefacto que se publica. Uma base URL com
    ``https://utilizador:senha@host`` — forma aceite por proxies e gateways —
    entrava aí em claro, e o corpo de resposta de um fornecedor pode ecoar o
    cabeçalho ``Authorization``. Nada disto deve sobreviver à serialização.
    """
    for padrao in _SECRET_PATTERNS:
        text = padrao.sub(lambda m: "***@" if m.group(0).endswith("@") else "***", text)
    return text


class PermanentApiError(RuntimeError):
    """Erro 4xx não transitório: configuração errada, não falha de rede.

    Repetir não ajuda — o valor está em mostrar o que o fornecedor respondeu.
    """


def _permanent_http_error(response: httpx.Response) -> PermanentApiError:
    """Converte um 4xx no erro do fornecedor, com o corpo da resposta legível.

    Endpoints compatíveis devolvem a causa real no corpo (ex.: Ollama responde
    ``model 'qwen2.5:7b' not found`` a um 404). Sem isto, o utilizador vê apenas
    "404 Not Found" na URL certa e conclui que o endpoint está errado.
    """
    detalhe = ""
    try:
        corpo = response.json()
        if isinstance(corpo, dict):
            erro = corpo.get("error")
            if isinstance(erro, dict):
                detalhe = str(erro.get("message") or "")
            elif erro:
                detalhe = str(erro)
        if not detalhe:
            detalhe = response.text
    except ValueError:
        detalhe = response.text
    detalhe = " ".join(detalhe.split())[:400]
    sufixo = f": {detalhe}" if detalhe else ""
    return PermanentApiError(
        redact_secrets(f"HTTP {response.status_code} de {response.request.url}{sufixo}"),
    )


#: Modelos cujo fornecedor rejeitou o ``temperature`` pedido durante esta corrida.
#: Global e não por instância porque a informação é da **corrida**, não do cliente:
#: quem escreve o `summary.json` (`collect_run_metadata`) não tem os clientes à mão,
#: e sem isto o aviso ficava só no stderr — um facto que altera a interpretação dos
#: números sem deixar rasto no artefacto.
_TEMPERATURE_REJECTED: set[str] = set()
_TEMPERATURE_LOCK = threading.Lock()


def record_temperature_rejected(model: str) -> None:
    """Regista que ``model`` correu sem o ``temperature`` pedido."""
    with _TEMPERATURE_LOCK:
        _TEMPERATURE_REJECTED.add(model)


def temperature_rejected_models() -> list[str]:
    """Modelos que perderam determinismo nesta corrida, por ordem estável."""
    with _TEMPERATURE_LOCK:
        return sorted(_TEMPERATURE_REJECTED)


def reset_temperature_rejected() -> None:
    """Limpa o registo. Para testes e para corridas encadeadas no mesmo processo."""
    with _TEMPERATURE_LOCK:
        _TEMPERATURE_REJECTED.clear()


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
    #: Ligações simultâneas do pool HTTP. Deve ser >= à concorrência de itens da
    #: corrida; ``pool_size_for_concurrency`` calcula-o e ``run_batch`` passa-o.
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    #: Ficou verdadeiro se o fornecedor rejeitou o ``temperature`` pedido. Para o
    #: artefacto usa-se ``temperature_rejected_models()``, que é da corrida inteira.
    temperature_rejected: bool = field(default=False, init=False, compare=False)
    _usage_tls: threading.local = field(
        default_factory=threading.local,
        init=False,
        repr=False,
        compare=False,
    )
    _http: httpx.Client | None = field(default=None, init=False, repr=False, compare=False)
    _http_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def last_usage(self) -> ApiUsageSnapshot | None:
        """Uso da última chamada **desta thread**.

        O cliente é partilhado entre workers concorrentes; guardar o último uso
        num atributo de instância atribuiria os tokens de um item ao item de
        outra thread. Armazenamento thread-local mantém a contabilidade correta.
        """
        return cast(ApiUsageSnapshot | None, getattr(self._usage_tls, "value", None))

    @last_usage.setter
    def last_usage(self, value: ApiUsageSnapshot | None) -> None:
        self._usage_tls.value = value

    def _http_client(self) -> httpx.Client:
        """Cliente HTTP partilhado (keep-alive + pool).

        Criar um ``httpx.Client`` por chamada custa um handshake TLS por request;
        num corpus de 1000 itens com gerador + juiz isso é ~2000 handshakes. O
        cliente do httpx é thread-safe, por isso pode ser partilhado pelos workers.
        """
        client = self._http
        if client is not None and not client.is_closed:
            return client
        with self._http_lock:
            client = self._http
            if client is None or client.is_closed:
                client = httpx.Client(
                    timeout=self.timeout_seconds,
                    limits=httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_connections,
                    ),
                )
                self._http = client
            return client

    def close(self) -> None:
        """Fecha o pool HTTP. Idempotente."""
        with self._http_lock:
            if self._http is not None:
                self._http.close()
                self._http = None

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

    #: Códigos de erro que chegam como 429 mas nunca recuperam com espera.
    #: Saldo esgotado ou quota de projeto a zero devolvem 429 como se fosse rate
    #: limit; retentar com backoff só atrasa a falha e esconde a causa real.
    QUOTA_ERROR_CODES = (
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_hard_limit_reached",
    )

    @classmethod
    def _is_quota_exhausted(cls, response: httpx.Response) -> bool:
        """Distingue "excedeu a taxa" de "não tem saldo" dentro de um 429."""
        try:
            corpo = response.json()
        except ValueError:
            return False
        erro = corpo.get("error") if isinstance(corpo, dict) else None
        if not isinstance(erro, dict):
            return False
        marcadores = {str(erro.get("type") or ""), str(erro.get("code") or "")}
        return bool(marcadores & set(cls.QUOTA_ERROR_CODES))

    @staticmethod
    def _is_transient_status(code: int) -> bool:
        # 429 e 5xx são transitórios; demais 4xx (auth, payload, etc.) propagam de imediato.
        return code == 429 or 500 <= code < 600

    @staticmethod
    def _rejects_temperature(response: httpx.Response) -> bool:
        """400 por o modelo não aceitar o ``temperature`` que enviámos.

        Alguns modelos recentes só admitem a temperatura por omissão e devolvem
        ``unsupported_value`` para qualquer outra. Como o juiz fixa 0.0 para
        obter determinismo, esses modelos falhariam **todas** as chamadas — e o
        fallback heurístico, que responde ``sustentado``, faria um juiz avariado
        parecer um juiz permissivo. Preferimos perder o determinismo a produzir
        vereditos silenciosamente falsos, e o facto fica registado na
        proveniência do `summary.json` (ver ``temperature_rejected_models``).
        """
        if response.status_code != 400:
            return False
        try:
            corpo = response.json()
        except ValueError:
            return False
        erro = corpo.get("error") if isinstance(corpo, dict) else None
        if not isinstance(erro, dict):
            return False
        return erro.get("param") == "temperature"

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        url = chat_completions_url(self.base_url)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._payload(system, user, json_object=json_object)

        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                r = self._http_client().post(url, headers=headers, json=payload)
            except httpx.RequestError as exc:
                # Erro de transporte (DNS, TCP, timeout): considerar transitório
                last_exc = exc
                if i >= attempts - 1:
                    break
                self._sleep_backoff(i)
                continue

            if r.status_code == 429 and self._is_quota_exhausted(r):
                raise _permanent_http_error(r)

            if "temperature" in payload and self._rejects_temperature(r):
                self.temperature_rejected = True
                record_temperature_rejected(self.model)
                payload.pop("temperature")
                print(
                    f"[llm] {self.model} não aceita temperature={self.temperature}; "
                    "a repetir sem o parâmetro (determinismo não garantido).",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if self._is_transient_status(r.status_code):
                last_exc = httpx.HTTPStatusError(
                    f"transient HTTP {r.status_code}", request=r.request, response=r
                )
                if i >= attempts - 1:
                    break
                self._sleep_backoff(i, status_code=r.status_code, response=r)
                continue

            if r.status_code >= 400:
                raise _permanent_http_error(r)
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
    ) -> tuple[float, bool]:
        """Devolve ``(atraso, veio_do_servidor)``.

        O segundo elemento distingue um atraso escolhido por nós de uma directiva
        ``Retry-After`` do servidor — só o primeiro pode receber jitter negativo.
        """
        if status_code == 429 and response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 1.0), True
                except ValueError:
                    pass
            seq = self.rate_limit_backoff_seconds
            return (seq[i] if i < len(seq) else seq[-1]), False
        seq = self.backoff_seconds
        if not seq:
            return 0.0, False
        return (seq[i] if i < len(seq) else seq[-1]), False

    def _sleep_backoff(
        self,
        i: int,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        delay, from_server = self._retry_delay_seconds(
            i,
            status_code=status_code,
            response=response,
        )
        if delay <= 0:
            return
        if from_server:
            # ``Retry-After`` é uma directiva: nunca retomar antes do prazo. O jitter
            # positivo continua a dessincronizar os workers sem violar o servidor.
            time.sleep(delay * (1.0 + random.uniform(0.0, 0.2)))
            return
        # Jitter simétrico: com vários workers em retry, um backoff idêntico
        # sincroniza as tentativas e reproduz o 429 (thundering herd).
        time.sleep(delay * (1.0 + random.uniform(-0.2, 0.2)))


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


def pool_size_for_concurrency(concurrency: int) -> int:
    """Ligações do pool para N workers: uma por worker, com folga e um mínimo sensato.

    Um pool menor que a concorrência faz os workers excedentes bloquearem à espera
    de uma ligação até ao ``timeout`` e depois falharem com ``PoolTimeout`` — que,
    sendo um ``RequestError``, é retentado e mascara o erro de configuração.
    """
    return max(DEFAULT_MAX_CONNECTIONS, int(concurrency) + 4)


def default_llm_from_env(
    *,
    timeout_seconds: float,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
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
        max_connections=max_connections,
    )


def default_judge_from_env(
    *,
    timeout_seconds: float,
    temperature: float = 0.0,
    max_retries: int | None = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
) -> LlmClient:
    """Cliente para o juiz; por padrão temperatura 0 (determinismo do veredito)."""
    key = judge_api_key_from_env()
    base = judge_base_url_from_env()
    _, model = resolve_models_from_env()
    retries = _max_retries_from_env() if max_retries is None else max_retries
    return OpenAiCompatibleClient(
        api_key=key,
        base_url=base,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_retries=retries,
        max_connections=max_connections,
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

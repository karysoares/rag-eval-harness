"""Observabilidade de corridas: uso de tokens, latência e custo estimado por item."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from llm_evaluation.llm_client import LlmClient
from llm_evaluation.types import RunRecord

LlmRole = Literal["generation", "judge"]


@dataclass
class LlmCallUsage:
    role: LlmRole
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    #: Epoch do início da chamada; necessário para posicionar spans na timeline.
    started_at: float = 0.0
    #: ``scheme://host`` do fornecedor, para distinguir juiz local de API.
    endpoint: str = ""


@dataclass
class UsageAccumulator:
    """Acumula as chamadas LLM do item **em curso nesta thread**.

    O acumulador é partilhado pelos clientes da corrida, mas ``snapshot_for_item``
    /``reset`` delimitam um item. Com workers concorrentes, uma lista única
    misturaria as chamadas de itens diferentes; o armazenamento thread-local
    garante que cada item contabiliza apenas os seus próprios tokens e latência.
    """

    _tls: threading.local = field(default_factory=threading.local, repr=False, compare=False)

    @property
    def calls(self) -> list[LlmCallUsage]:
        calls = getattr(self._tls, "calls", None)
        if calls is None:
            calls = []
            self._tls.calls = calls
        return cast(list[LlmCallUsage], calls)

    def record(self, usage: LlmCallUsage) -> None:
        self.calls.append(usage)

    def snapshot_for_item(self) -> dict[str, Any]:
        if not self.calls:
            return {"n_chamadas_llm": 0}
        pt = sum(c.prompt_tokens for c in self.calls)
        ct = sum(c.completion_tokens for c in self.calls)
        tt = sum(c.total_tokens for c in self.calls)
        lat = sum(c.latency_ms for c in self.calls)
        by_role: dict[str, int] = {}
        for c in self.calls:
            by_role[c.role] = by_role.get(c.role, 0) + 1
        por_modelo: dict[str, dict[str, int]] = {}
        for c in self.calls:
            alvo = por_modelo.setdefault(
                c.model,
                {"n_chamadas": 0, "tokens_prompt": 0, "tokens_completion": 0},
            )
            alvo["n_chamadas"] += 1
            alvo["tokens_prompt"] += c.prompt_tokens
            alvo["tokens_completion"] += c.completion_tokens
        return {
            "n_chamadas_llm": len(self.calls),
            "tokens_prompt": pt,
            "tokens_completion": ct,
            "tokens_total": tt,
            "latencia_ms_total": round(lat, 2),
            "chamadas_por_papel": by_role,
            "modelos": sorted({c.model for c in self.calls}),
            # Sem esta repartição, um custo agregado com gerador e juiz em modelos
            # diferentes é falso — e essa é a configuração recomendada.
            "por_modelo": por_modelo,
        }

    def drain(self) -> list[LlmCallUsage]:
        """Devolve as chamadas do item nesta thread e limpa. Usado pela telemetria."""
        calls = list(self.calls)
        self.calls.clear()
        return calls

    def reset(self) -> None:
        self.calls.clear()


class TrackingLlmClient:
    """Envolve um cliente LLM e regista uso após cada ``complete``."""

    def __init__(
        self,
        inner: LlmClient,
        accumulator: UsageAccumulator,
        *,
        role: LlmRole,
        model: str,
        endpoint: str = "",
    ) -> None:
        self._inner = inner
        self._acc = accumulator
        self._role = role
        self._model = model
        self._endpoint = endpoint

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        t0 = time.perf_counter()
        started_at = time.time()
        text = self._inner.complete(system, user, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        last = getattr(self._inner, "last_usage", None)
        if last is not None:
            self._acc.record(
                LlmCallUsage(
                    role=self._role,
                    model=str(getattr(last, "model", None) or self._model),
                    prompt_tokens=int(last.prompt_tokens),
                    completion_tokens=int(last.completion_tokens),
                    total_tokens=int(last.total_tokens),
                    latency_ms=round(latency_ms, 2),
                    started_at=started_at,
                    endpoint=self._endpoint,
                )
            )
        else:
            self._acc.record(
                LlmCallUsage(
                    role=self._role,
                    model=self._model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=round(latency_ms, 2),
                    started_at=started_at,
                    endpoint=self._endpoint,
                )
            )
        return text


def estimate_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    price_per_1m_prompt: float,
    price_per_1m_completion: float,
) -> float:
    return (prompt_tokens / 1_000_000.0) * price_per_1m_prompt + (
        completion_tokens / 1_000_000.0
    ) * price_per_1m_completion


def prices_by_model_from_env() -> dict[str, tuple[float, float]]:
    """Preços por modelo em ``LLM_EVAL_PRICES`` (``modelo:prompt:completion,…``).

    Um par único de preços aplicado a todos os modelos subestima ou sobrestima o
    custo sempre que gerador e juiz são modelos diferentes — a configuração que
    este harness recomenda. Exemplo::

        LLM_EVAL_PRICES=gpt-4o-mini:0.15:0.60,gpt-4o:2.50:10.00,qwen2.5:7b:0:0

    O nome do modelo pode conter ``:`` (as etiquetas do Ollama são ``modelo:tag``),
    por isso a separação é feita **pela direita**: os dois últimos campos são
    sempre os preços e tudo o que vem antes é o nome. Modelos locais entram com
    preço zero para ficarem no total em vez de aparecerem como "sem preço".
    """
    bruto = os.environ.get("LLM_EVAL_PRICES", "").strip()
    if not bruto:
        return {}
    precos: dict[str, tuple[float, float]] = {}
    for entrada in bruto.split(","):
        partes = entrada.strip().rsplit(":", 2)
        if len(partes) != 3 or not partes[0].strip():
            continue
        try:
            precos[partes[0].strip()] = (float(partes[1]), float(partes[2]))
        except ValueError:
            continue
    return precos


def _cost_by_model(
    por_modelo: dict[str, dict[str, int]],
    precos: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Custo repartido por modelo; assinala os modelos sem preço conhecido."""
    detalhe: dict[str, Any] = {}
    total = 0.0
    sem_preco: list[str] = []
    for modelo, uso in sorted(por_modelo.items()):
        par = precos.get(modelo)
        if par is None:
            sem_preco.append(modelo)
            continue
        custo = estimate_cost_usd(
            prompt_tokens=uso["tokens_prompt"],
            completion_tokens=uso["tokens_completion"],
            price_per_1m_prompt=par[0],
            price_per_1m_completion=par[1],
        )
        total += custo
        detalhe[modelo] = {**uso, "custo_usd": round(custo, 6)}
    saida: dict[str, Any] = {"por_modelo": detalhe, "custo_total_usd": round(total, 6)}
    if sem_preco:
        # Explícito: um total silenciosamente parcial seria pior que nenhum.
        saida["modelos_sem_preco"] = sorted(sem_preco)
        saida["nota_parcial"] = "Total exclui os modelos sem preço configurado em LLM_EVAL_PRICES."
    return saida


def summarize_run_observability(
    records: list[RunRecord],
    *,
    price_per_1m_prompt: float | None = None,
    price_per_1m_completion: float | None = None,
    prices_by_model: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any] | None:
    """Agrega ``meta.observabilidade`` de todos os itens; opcional custo estimado."""
    pt = ct = tt = n_calls = 0
    lat_total = 0.0
    n_with_meta = 0
    por_modelo: dict[str, dict[str, int]] = {}
    for r in records:
        obs = r.meta.get("observabilidade")
        if not isinstance(obs, dict):
            continue
        n_with_meta += 1
        n_calls += int(obs.get("n_chamadas_llm") or 0)
        pt += int(obs.get("tokens_prompt") or 0)
        ct += int(obs.get("tokens_completion") or 0)
        tt += int(obs.get("tokens_total") or 0)
        lat_total += float(obs.get("latencia_ms_total") or 0.0)
        bruto_modelo = obs.get("por_modelo")
        if isinstance(bruto_modelo, dict):
            for modelo, uso in bruto_modelo.items():
                if not isinstance(uso, dict):
                    continue
                alvo = por_modelo.setdefault(
                    str(modelo),
                    {"n_chamadas": 0, "tokens_prompt": 0, "tokens_completion": 0},
                )
                for chave in alvo:
                    alvo[chave] += int(uso.get(chave) or 0)
    if n_with_meta == 0:
        return None
    out: dict[str, Any] = {
        "n_itens_com_observabilidade": n_with_meta,
        "n_chamadas_llm_total": n_calls,
        "tokens_prompt_total": pt,
        "tokens_completion_total": ct,
        "tokens_total": tt,
        "latencia_ms_total": round(lat_total, 2),
        "media_tokens_por_item": round(tt / n_with_meta, 1) if n_with_meta else None,
    }
    if por_modelo:
        out["uso_por_modelo"] = por_modelo
    precos = prices_by_model if prices_by_model is not None else prices_by_model_from_env()
    if precos and por_modelo:
        out["custo"] = _cost_by_model(por_modelo, precos)
    if price_per_1m_prompt is not None and price_per_1m_completion is not None:
        out["custo_estimado_usd"] = round(
            estimate_cost_usd(
                prompt_tokens=pt,
                completion_tokens=ct,
                price_per_1m_prompt=price_per_1m_prompt,
                price_per_1m_completion=price_per_1m_completion,
            ),
            6,
        )
        out["nota_custo"] = (
            "Preço único aplicado a todos os modelos: só é correcto quando gerador "
            "e juiz usam o mesmo modelo. Para custo por modelo, defina LLM_EVAL_PRICES."
        )
    return out

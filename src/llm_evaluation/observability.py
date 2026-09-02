"""Observabilidade de corridas: uso de tokens, latência e custo estimado por item."""

from __future__ import annotations

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
        return {
            "n_chamadas_llm": len(self.calls),
            "tokens_prompt": pt,
            "tokens_completion": ct,
            "tokens_total": tt,
            "latencia_ms_total": round(lat, 2),
            "chamadas_por_papel": by_role,
            "modelos": sorted({c.model for c in self.calls}),
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


def summarize_run_observability(
    records: list[RunRecord],
    *,
    price_per_1m_prompt: float | None = None,
    price_per_1m_completion: float | None = None,
) -> dict[str, Any] | None:
    """Agrega ``meta.observabilidade`` de todos os itens; opcional custo estimado."""
    pt = ct = tt = n_calls = 0
    lat_total = 0.0
    n_with_meta = 0
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
            "Estimativa com preços por 1M tokens (OPENAI_PRICE_* no ambiente ou argumentos)."
        )
    return out

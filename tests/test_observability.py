"""Observabilidade: tracking de chamadas e sumário de corrida."""

from __future__ import annotations

from llm_evaluation.llm_client import ApiUsageSnapshot, OpenAiCompatibleClient
from llm_evaluation.observability import (
    TrackingLlmClient,
    UsageAccumulator,
    summarize_run_observability,
)
from llm_evaluation.reporting import summarize
from llm_evaluation.types import RunRecord, VerificationSignals


class _Inner:
    last_usage = ApiUsageSnapshot(
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def complete(self, system: str, user: str) -> str:
        return "ok"


def test_tracking_client_records_usage() -> None:
    acc = UsageAccumulator()
    client = TrackingLlmClient(_Inner(), acc, role="generation", model="test-model")
    assert client.complete("s", "u") == "ok"
    snap = acc.snapshot_for_item()
    assert snap["n_chamadas_llm"] == 1
    assert snap["tokens_total"] == 15


def test_summarize_includes_observability() -> None:
    rec = RunRecord(
        item_id="1",
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=None,
            embedding_low_support=None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "observabilidade": {
                "n_chamadas_llm": 2,
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "tokens_total": 150,
                "latencia_ms_total": 200.0,
            }
        },
    )
    s = summarize(
        [rec],
        protocol={
            "verify_embedding": True,
            "verify_judge": True,
            "aggregation_policy": "embedding_e_juiz",
        },
    )
    assert s["taxa_alerta"] == 0.0
    assert s["detector_activo"]["politica_agregacao"] == "embedding_e_juiz"
    obs = summarize_run_observability([rec])
    assert obs is not None
    assert obs["tokens_total"] == 150


def test_openai_client_sets_last_usage() -> None:
    """ApiUsageSnapshot é preenchido após parse da resposta (sem rede aqui)."""
    c = OpenAiCompatibleClient(
        api_key="x",
        base_url="https://api.openai.com",
        model="m",
        timeout_seconds=1.0,
    )
    assert c.last_usage is None

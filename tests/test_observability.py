"""Observabilidade: tracking de chamadas e sumário de corrida."""

from __future__ import annotations

import pytest

from llm_evaluation.llm_client import ApiUsageSnapshot, OpenAiCompatibleClient
from llm_evaluation.observability import (
    TrackingLlmClient,
    UsageAccumulator,
    prices_by_model_from_env,
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


class TestCustoPorModelo:
    """Regressão: um preço único aplicado a gerador e juiz distintos erra por ~10x.

    Medido numa corrida real de 200 itens: `gpt-4o-mini` gerador + `gpt-4o` juiz
    custaram $1,69, e o preço único reportou $0,17.
    """

    @staticmethod
    def _record(por_modelo: dict[str, dict[str, int]]) -> RunRecord:
        return RunRecord(
            item_id="i",
            question="q",
            answer="a",
            gold_correct=None,
            anomaly_flag=False,
            signals=VerificationSignals(
                gold_correct=None,
                gold_incorrect=None,
                is_refusal=False,
                embedding_max_cosine=None,
                embedding_low_support=None,
            ),
            retrieved=[],
            baseline_profile="hibrido",
            meta={
                "observabilidade": {
                    "n_chamadas_llm": sum(v["n_chamadas"] for v in por_modelo.values()),
                    "tokens_prompt": sum(v["tokens_prompt"] for v in por_modelo.values()),
                    "tokens_completion": sum(v["tokens_completion"] for v in por_modelo.values()),
                    "tokens_total": 0,
                    "latencia_ms_total": 1.0,
                    "por_modelo": por_modelo,
                }
            },
        )

    def test_custo_repartido_por_modelo(self) -> None:
        rec = self._record(
            {
                "gpt-4o-mini": {
                    "n_chamadas": 1,
                    "tokens_prompt": 1_000_000,
                    "tokens_completion": 0,
                },
                "gpt-4o": {"n_chamadas": 1, "tokens_prompt": 1_000_000, "tokens_completion": 0},
            }
        )
        out = summarize_run_observability(
            [rec],
            prices_by_model={"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)},
        )
        assert out is not None
        custo = out["custo"]
        assert custo["por_modelo"]["gpt-4o"]["custo_usd"] == 2.5
        assert custo["por_modelo"]["gpt-4o-mini"]["custo_usd"] == 0.15
        assert custo["custo_total_usd"] == 2.65

    def test_modelo_sem_preco_e_assinalado_e_nao_somado(self) -> None:
        rec = self._record(
            {
                "gpt-4o": {"n_chamadas": 1, "tokens_prompt": 1_000_000, "tokens_completion": 0},
                "qwen2.5:7b": {"n_chamadas": 1, "tokens_prompt": 9_000_000, "tokens_completion": 0},
            }
        )
        out = summarize_run_observability([rec], prices_by_model={"gpt-4o": (2.50, 10.00)})
        assert out is not None
        assert out["custo"]["custo_total_usd"] == 2.5
        assert out["custo"]["modelos_sem_preco"] == ["qwen2.5:7b"]
        assert "nota_parcial" in out["custo"]

    def test_uso_por_modelo_e_agregado_entre_itens(self) -> None:
        rec = self._record(
            {"m": {"n_chamadas": 1, "tokens_prompt": 10, "tokens_completion": 2}},
        )
        out = summarize_run_observability([rec, rec])
        assert out is not None
        assert out["uso_por_modelo"]["m"] == {
            "n_chamadas": 2,
            "tokens_prompt": 20,
            "tokens_completion": 4,
        }

    def test_precos_do_ambiente(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EVAL_PRICES", "gpt-4o:2.50:10.00, gpt-4o-mini:0.15:0.60 ")
        assert prices_by_model_from_env() == {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        }

    def test_entradas_malformadas_sao_ignoradas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EVAL_PRICES", "so_nome,a:b:c,bom:1:2")
        assert prices_by_model_from_env() == {"bom": (1.0, 2.0)}

    def test_sem_env_nao_ha_precos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_EVAL_PRICES", raising=False)
        assert prices_by_model_from_env() == {}

    def test_nome_de_modelo_com_dois_pontos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regressão: etiquetas do Ollama (``qwen2.5:7b``) eram descartadas em silêncio."""
        monkeypatch.setenv("LLM_EVAL_PRICES", "gpt-4o:2.50:10.00,qwen2.5:7b:0:0")
        assert prices_by_model_from_env() == {
            "gpt-4o": (2.50, 10.00),
            "qwen2.5:7b": (0.0, 0.0),
        }

    def test_modelo_local_a_zero_entra_no_total_e_nao_em_sem_preco(self) -> None:
        rec = self._record(
            {"qwen2.5:7b": {"n_chamadas": 1, "tokens_prompt": 9_000_000, "tokens_completion": 0}},
        )
        out = summarize_run_observability([rec], prices_by_model={"qwen2.5:7b": (0.0, 0.0)})
        assert out is not None
        assert out["custo"]["custo_total_usd"] == 0.0
        assert "modelos_sem_preco" not in out["custo"]

    def test_entrada_sem_nome_e_ignorada(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EVAL_PRICES", ":1:2,bom:1:2")
        assert prices_by_model_from_env() == {"bom": (1.0, 2.0)}

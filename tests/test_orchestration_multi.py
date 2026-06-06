"""Orquestração multi-agente (respondedor + crítica) com LLM mockado."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llm_evaluation import pipeline
from llm_evaluation.adapters.amostra_local import amostra_local_items
from llm_evaluation.config import load_config
from llm_evaluation.orchestration import multi

if TYPE_CHECKING:
    from llm_evaluation.config import AppConfig


def _responder_json(resposta: str) -> str:
    return json.dumps(
        {
            "resposta": resposta,
            "confianca": 0.9,
            "contexto_insuficiente": False,
        }
    )


class _RoutingLlm:
    """Roteia geração, crítica e juiz por conteúdo do system/user."""

    def __init__(self, *, critic_issues: list[str] | None = None) -> None:
        self.n_complete = 0
        self.critic_issues = critic_issues or ["nenhum"]

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        del kwargs
        self.n_complete += 1
        sys_l = system.lower()
        if "crítica diagnóstica" in sys_l or ("problemas" in sys_l and "motivo_breve" not in sys_l):
            return json.dumps(
                {
                    "cadeia_de_pensamento": ["pedido", "evidência", "conclusão"],
                    "problemas": self.critic_issues,
                    "nota": "ok",
                }
            )
        if "veredito" in sys_l or "avaliadora" in sys_l:
            return json.dumps(
                {
                    "cadeia_de_pensamento": ["ok"],
                    "veredito": "sustentado",
                    "motivo_breve": "ok",
                    "confianca": 0.9,
                }
            )
        return _responder_json("Brasília é a capital do Brasil.")


def _multi_cfg() -> AppConfig:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    return replace(
        cfg,
        orchestration="multiplo",
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )


@pytest.fixture
def patched_multi(monkeypatch: pytest.MonkeyPatch) -> _RoutingLlm:
    fake = _RoutingLlm()

    def _fake_llm(**_: object) -> _RoutingLlm:
        return fake

    monkeypatch.setattr(pipeline, "default_llm_from_env", _fake_llm)
    monkeypatch.setattr(pipeline, "default_judge_from_env", _fake_llm)
    return fake


def test_multi_run_items_meta_and_critic(patched_multi: _RoutingLlm) -> None:
    records = multi.run_items(_multi_cfg(), amostra_local_items())
    assert len(records) == 2
    assert records[0].meta.get("orquestracao") == "multiplo"
    crit = records[0].meta.get("critica")
    assert isinstance(crit, dict)
    assert crit.get("schema_version")
    assert crit.get("schema_invalid") is not True
    assert records[0].meta.get("flag_critica") is False
    assert "explicacao" in records[0].meta
    qg = records[0].meta.get("qualidade_geracao")
    assert isinstance(qg, dict)
    assert qg.get("confianca") == 0.9
    assert patched_multi.n_complete >= 2


def test_critic_flag_does_not_set_anomaly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RoutingLlm(critic_issues=["alucinacao"])

    def _fake_llm(**_: object) -> _RoutingLlm:
        return fake

    monkeypatch.setattr(pipeline, "default_llm_from_env", _fake_llm)
    monkeypatch.setattr(pipeline, "default_judge_from_env", _fake_llm)
    cfg = _multi_cfg()
    cfg = replace(
        cfg,
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )
    records = multi.run_items(cfg, amostra_local_items()[:1])
    assert records[0].meta.get("flag_critica") is True
    assert records[0].anomaly_flag is False


def test_critic_invalid_json_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadCriticLlm(_RoutingLlm):
        def complete(self, system: str, user: str, **kwargs: object) -> str:
            sys_l = system.lower()
            if "crítica diagnóstica" in sys_l or (
                "problemas" in sys_l and "motivo_breve" not in sys_l
            ):
                return "not json"
            return super().complete(system, user, **kwargs)

    fake = _BadCriticLlm()

    def _fake_llm(**_: object) -> _BadCriticLlm:
        return fake

    monkeypatch.setattr(pipeline, "default_llm_from_env", _fake_llm)
    monkeypatch.setattr(pipeline, "default_judge_from_env", _fake_llm)
    cfg = _multi_cfg()
    records = multi.run_items(cfg, amostra_local_items()[:1])
    crit = records[0].meta.get("critica")
    assert isinstance(crit, dict)
    assert crit.get("schema_invalid") is True
    assert crit.get("structured_output_error")
    assert records[0].meta.get("flag_critica") is False

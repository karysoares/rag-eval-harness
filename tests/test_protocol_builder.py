"""build_protocolo_ativo centralizado."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evaluation.config import load_config
from llm_evaluation.protocol import build_protocolo_ativo, collect_protocol_avisos


def test_build_protocolo_ativo_keys() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_amostra.yaml")
    proto = build_protocolo_ativo(cfg)
    assert proto["aggregation_policy"] == cfg.aggregation.policy
    assert proto["orchestration"] == cfg.orchestration
    assert proto["judge_prompt_style"] == cfg.verification.judge_prompt_style
    assert "fila_min_score_recuperacao" in proto
    assert "judge_gate_embedding_max_cosine" in proto
    assert "judge_gate_requires_strong_context" in proto
    assert "judge_gate_min_retrieval_score" in proto
    assert "judge_incompleto_contexto_forte_negativo" in proto
    rag = proto["rag"]
    assert isinstance(rag, dict)
    assert rag["top_k"] == cfg.rag.top_k
    assert rag["chunk_max_chars"] == cfg.rag.chunk_max_chars
    gen = proto["generation"]
    assert isinstance(gen, dict)
    assert gen["temperature"] == cfg.generation.temperature
    assert gen["prompt_style"] == cfg.generation.prompt_style
    models = proto["models"]
    assert isinstance(models, dict)
    assert "llm_model" in models
    assert "judge_model" in models


def test_collect_protocol_avisos_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "same-model")
    monkeypatch.setenv("JUDGE_MODEL", "same-model")
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke_amostra.yaml")
    avisos = collect_protocol_avisos(cfg)
    assert any("JUDGE_MODEL" in a for a in avisos)

"""Validação e normalização de protocolo (config + itens)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from llm_evaluation.config import apply_baseline_profile, load_config
from llm_evaluation.protocol import apply_protocol_defaults, validate_protocol
from llm_evaluation.types import EvalItem


def test_default_yaml_fairytale_rag_protocol() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    assert cfg.rag.enabled is True
    assert cfg.verification.verify_embedding is True
    assert cfg.verification.verify_judge is True
    assert cfg.lexical_metrics.enabled is True


def test_apply_protocol_disables_rag_without_corpus() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = replace(
        cfg,
        rag=replace(cfg.rag, enabled=True),
        verification=replace(
            cfg.verification,
            verify_embedding=True,
            verify_judge=True,
        ),
    )
    items = [
        EvalItem(
            id="ft-1",
            question="Quem?",
            correct_answers=["x"],
            incorrect_answers=[],
            rag_gold_chunk=None,
            rag_distractors=[],
        ),
    ]
    cfg2, adjustments = apply_protocol_defaults(cfg, items)
    assert cfg2.rag.enabled is False
    assert cfg2.verification.verify_embedding is False
    assert any(a.campo == "rag.enabled" for a in adjustments)


def test_profile_applied_before_protocol_defaults_disables_invalid_judge_without_corpus() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = replace(cfg, rag=replace(cfg.rag, enabled=False))
    cfg = apply_baseline_profile(cfg, "so_juiz")
    items = [
        EvalItem(
            id="ft-1",
            question="Quem?",
            correct_answers=["x"],
            incorrect_answers=[],
            rag_gold_chunk=None,
            rag_distractors=[],
        ),
    ]
    cfg2, adjustments = apply_protocol_defaults(cfg, items)
    validate_protocol(cfg2, items)
    assert cfg2.verification.verify_judge is False
    assert any(a.campo == "verification.verify_judge" for a in adjustments)


def test_validate_protocol_ok_for_fairytale_sample() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    items = [
        EvalItem(
            id="amostra-1",
            question="Q?",
            correct_answers=["a"],
            incorrect_answers=[],
            rag_gold_chunk="contexto",
            rag_distractors=[],
        ),
    ]
    validate_protocol(cfg, items)


def test_hub_sem_hf_repo_falha_no_loader() -> None:
    from llm_evaluation.eval_items_load import load_eval_items

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = replace(cfg, dataset=replace(cfg.dataset, hf_repo=None, mode="hub"))
    with pytest.raises(ValueError, match="hf_repo"):
        load_eval_items(cfg)

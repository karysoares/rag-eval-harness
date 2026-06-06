"""Config reference_type e defaults de verificação (FairytaleQA)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_evaluation.config import apply_baseline_profile, load_config


def test_default_fairytale_lexical() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    assert cfg.dataset.name == "fairytale_ptbr"
    assert cfg.dataset.reference_type == "lexical"
    assert cfg.verification.verify_gold is False
    assert cfg.verification.judge_prompt_style == "rag_pt"
    assert cfg.verification.judge_gate_embedding_max_cosine is None
    assert cfg.verification.judge_incompleto_contexto_forte_negativo is False
    assert cfg.generation.prompt_style == "rag_pt"


def test_smoke_amostra_lexical() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    assert cfg.dataset.reference_type == "lexical"
    assert cfg.dataset.mode == "amostra_local"


def test_full_config_production_guards_enabled() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/ptbr_fairytale_full.yaml")
    assert cfg.verification.judge_gate_embedding_max_cosine is None
    assert cfg.verification.judge_gate_requires_strong_context is False
    assert cfg.generation.temperature == 0.1
    assert cfg.generation.max_tokens == 128
    assert cfg.verification.judge_incompleto_contexto_forte_negativo is True
    assert cfg.verification.judge_incompleto_contexto_forte_min_score == 0.5
    assert cfg.generation.anti_refusal_repair is True
    assert cfg.generation.anti_refusal_min_retrieval_score == 0.5
    assert cfg.generation.anti_refusal_max_attempts == 2


def test_hybrid_baseline_profile_disables_gold_layer() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = apply_baseline_profile(cfg, "hibrido")
    assert cfg.verification.verify_gold is False
    assert cfg.verification.verify_embedding is True
    assert cfg.verification.verify_judge is True


@pytest.mark.parametrize(
    ("path_parts", "bad_value", "message"),
    [
        (("orchestration",), "duplo", "orchestration inválido"),
        (("aggregation", "policy"), "misturar_tudo", "aggregation.policy inválido"),
        (("baselines", "profile"), "juiz_e_gold", "baselines.profile inválido"),
        (("embeddings", "backend"), "openai", "embeddings.backend inválido"),
    ],
)
def test_invalid_enum_values_fail_early(
    tmp_path: Path,
    path_parts: tuple[str, ...],
    bad_value: str,
    message: str,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((repo / "configs/smoke_amostra.yaml").read_text(encoding="utf-8"))
    nested = raw
    for part in path_parts[:-1]:
        nested = nested[part]
    nested[path_parts[-1]] = bad_value
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(cfg_path)


def test_all_versioned_configs_load() -> None:
    repo = Path(__file__).resolve().parents[1]
    for path in sorted((repo / "configs").glob("*.yaml")):
        load_config(path)

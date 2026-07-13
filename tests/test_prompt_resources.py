"""Cobertura de prompt_resources: fonte empacotada, fallback e erros."""

from __future__ import annotations

import pytest

from llm_evaluation.prompt_resources import (
    load_prompt_text,
    prompt_bytes,
    source_prompts_dir,
)

PACKAGED_PROMPTS = [
    "critic_system.txt",
    "judge_rag_pt_system.txt",
    "judge_system.txt",
    "responder_system.txt",
]


@pytest.mark.parametrize("name", PACKAGED_PROMPTS)
def test_load_packaged_prompt(name: str) -> None:
    text = load_prompt_text(name)
    assert isinstance(text, str)
    assert text.strip(), f"prompt vazio: {name}"


def test_prompt_bytes_matches_text() -> None:
    name = PACKAGED_PROMPTS[0]
    assert prompt_bytes(name).decode("utf-8") == load_prompt_text(name)


def test_missing_prompt_raises() -> None:
    with pytest.raises(FileNotFoundError, match="nao_existe.txt"):
        load_prompt_text("nao_existe.txt")


def test_source_prompts_dir_is_canonical() -> None:
    d = source_prompts_dir()
    assert d.name == "prompts"
    assert (d / PACKAGED_PROMPTS[0]).is_file()


def test_all_packaged_prompts_have_stable_hashes() -> None:
    """Bytes fiéis: hashing de prompts é usado no manifest de runs."""
    import hashlib

    for name in PACKAGED_PROMPTS:
        h1 = hashlib.sha256(prompt_bytes(name)).hexdigest()
        h2 = hashlib.sha256(prompt_bytes(name)).hexdigest()
        assert h1 == h2

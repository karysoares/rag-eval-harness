"""Leitura de prompts em instalação empacotada ou árvore fonte."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def source_prompts_dir() -> Path:
    """Fonte canónica versionada (empacotada em ``llm_evaluation.prompts``)."""
    return Path(__file__).resolve().parent / "prompts"


def _legacy_repo_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"


def load_prompt_text(name: str) -> str:
    """Carrega prompt empacotado (fonte única em ``src/llm_evaluation/prompts/``)."""
    return prompt_bytes(name).decode("utf-8")


def prompt_bytes(name: str) -> bytes:
    """Carrega bytes do prompt para hashing e leitura fiel."""
    package_path = resources.files("llm_evaluation").joinpath("prompts", name)
    if package_path.is_file():
        return package_path.read_bytes()

    source_path = source_prompts_dir() / name
    if source_path.is_file():
        return source_path.read_bytes()

    legacy = _legacy_repo_prompts_dir() / name
    if legacy.is_file():
        return legacy.read_bytes()

    msg = f"Prompt em falta: {name}"
    raise FileNotFoundError(msg)

"""Garante que prompts/ na raiz espelha src/llm_evaluation/prompts/ (fonte empacotada)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evaluation.prompt_resources import source_prompts_dir

CANONICAL_PROMPT_COUNT = 7


def _legacy_repo_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def _txt_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {p.name: p for p in sorted(directory.glob("*.txt"))}


def test_repo_prompts_match_packaged_canonical() -> None:
    canonical = source_prompts_dir()
    legacy = _legacy_repo_prompts_dir()

    canonical_files = _txt_files(canonical)
    legacy_files = _txt_files(legacy)

    assert canonical_files, f"Sem prompts .txt em {canonical}"
    assert len(canonical_files) == CANONICAL_PROMPT_COUNT, (
        f"Esperados {CANONICAL_PROMPT_COUNT} prompts empacotados, "
        f"encontrados {len(canonical_files)}: {sorted(canonical_files)}"
    )

    if not legacy_files:
        pytest.skip(
            "prompts/ na raiz ausente ou vazio (gitignored em CI); "
            "fonte canónica em src/llm_evaluation/prompts/"
        )

    assert set(canonical_files) == set(legacy_files), (
        f"Conjuntos de ficheiros diferem.\n"
        f"Só em empacotado: {sorted(set(canonical_files) - set(legacy_files))}\n"
        f"Só em prompts/: {sorted(set(legacy_files) - set(canonical_files))}"
    )

    diffs: list[str] = []
    for name in sorted(canonical_files):
        canonical_text = canonical_files[name].read_text(encoding="utf-8")
        legacy_text = legacy_files[name].read_text(encoding="utf-8")
        if canonical_text != legacy_text:
            diffs.append(name)

    assert not diffs, (
        "prompts/ na raiz está desactualizado face a src/llm_evaluation/prompts/. "
        f"Sincronize: {', '.join(diffs)}"
    )

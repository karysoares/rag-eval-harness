"""Garante integridade dos prompts empacotados (fonte canónica)."""

from __future__ import annotations

from llm_evaluation.prompt_resources import source_prompts_dir

CANONICAL_PROMPT_COUNT = 7


def test_packaged_prompts_complete() -> None:
    canonical = source_prompts_dir()
    files = sorted(canonical.glob("*.txt"))
    assert files, f"Sem prompts .txt em {canonical}"
    assert len(files) == CANONICAL_PROMPT_COUNT, (
        f"Esperados {CANONICAL_PROMPT_COUNT} prompts empacotados, "
        f"encontrados {len(files)}: {[p.name for p in files]}"
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"Prompt vazio: {path.name}"

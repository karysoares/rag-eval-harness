"""Integração leve com Hugging Face Hub (opt-in: RUN_INTEGRATION=1)."""

from __future__ import annotations

import os

import pytest

from llm_evaluation.adapters.hf_generic import load_hf_qa_generic

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION", "").strip() != "1",
    reason="defina RUN_INTEGRATION=1 para descarregar do Hub",
)
def test_fairytale_hf_load_one_row() -> None:
    items = load_hf_qa_generic(
        "benjleite/FairytaleQA-translated-ptBR",
        hf_subset=None,
        split="validation",
        limit=1,
        seed=0,
        question_column="question",
        answer_column="answer",
        context_column="story_section",
        incorrect_column=None,
        id_column=None,
    )
    assert len(items) == 1
    assert items[0].question
    assert items[0].correct_answers

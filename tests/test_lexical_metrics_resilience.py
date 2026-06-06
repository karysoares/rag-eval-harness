"""Métricas léxicas não devem derrubar a corrida."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from llm_evaluation.config import LexicalMetricsConfig
from llm_evaluation.lexical_metrics import attach_lexical_to_meta
from llm_evaluation.types import EvalItem


def test_attach_lexical_survives_compute_failure() -> None:
    item = EvalItem(
        id="1",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
    )
    cfg = SimpleNamespace(
        lexical_metrics=LexicalMetricsConfig(
            enabled=True,
            bleu=True,
            rouge_l=True,
            meteor=True,
            levenshtein=True,
            token_f1=False,
            reference_mode="primeiro",
        ),
    )
    meta: dict = {}
    with patch(
        "llm_evaluation.lexical_metrics.compute_lexical_scores",
        side_effect=RuntimeError("boom"),
    ):
        attach_lexical_to_meta(meta, cfg, item, "answer")  # type: ignore[arg-type]
    assert meta["metricas_lexicas"]["note"] == "erro_ao_calcular_metricas_lexicas"
    assert "boom" in meta["metricas_lexicas"]["erro"]

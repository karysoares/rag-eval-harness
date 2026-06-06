"""Orquestração em pilha única (padrão). Ver `docs/techniques/multi-agent-critic-patterns.md`."""

from __future__ import annotations

from collections.abc import Callable

from llm_evaluation.config import AppConfig
from llm_evaluation.pipeline import run_batch
from llm_evaluation.types import EvalItem, RunRecord


def run_items(
    cfg: AppConfig,
    items: list[EvalItem],
    *,
    on_record: Callable[[RunRecord], None] | None = None,
) -> list[RunRecord]:
    return run_batch(cfg, items, on_record=on_record)

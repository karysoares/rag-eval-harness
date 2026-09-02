"""Orquestração em pilha única (padrão): retrieve → generate → verify."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from llm_evaluation.config import AppConfig
from llm_evaluation.pipeline import run_batch
from llm_evaluation.types import EvalItem, RunRecord


def run_items(
    cfg: AppConfig,
    items: list[EvalItem],
    *,
    on_record: Callable[[RunRecord], None] | None = None,
    run_dir: Path | None = None,
    config_name: str = "",
) -> list[RunRecord]:
    return run_batch(cfg, items, on_record=on_record, run_dir=run_dir, config_name=config_name)

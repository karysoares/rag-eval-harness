"""Resolução de adaptadores de dataset e materialização de ``EvalItem`` a partir do YAML."""

from __future__ import annotations

from llm_evaluation.adapters import amostra_local_items, load_hf_qa_generic
from llm_evaluation.config import AppConfig
from llm_evaluation.types import EvalItem


def load_eval_items(cfg: AppConfig) -> list[EvalItem]:
    """Ponto único de entrada para o conjunto de exemplos da corrida."""
    if cfg.dataset.mode == "amostra_local":
        items = amostra_local_items()
        if cfg.dataset.limit <= 0:
            return items
        return items[: cfg.dataset.limit]

    if not cfg.dataset.hf_repo:
        msg = "dataset.mode=hub exige dataset.hf_repo (ex.: benjleite/FairytaleQA-translated-ptBR)"
        raise ValueError(msg)

    return load_hf_qa_generic(
        cfg.dataset.hf_repo,
        hf_subset=cfg.dataset.hf_subset,
        split=cfg.dataset.split,
        limit=cfg.dataset.limit,
        seed=cfg.seed,
        question_column=cfg.dataset.question_column,
        answer_column=cfg.dataset.answer_column,
        context_column=cfg.dataset.context_column,
        incorrect_column=cfg.dataset.incorrect_column,
        id_column=cfg.dataset.id_column,
        shuffle=cfg.dataset.shuffle,
    )

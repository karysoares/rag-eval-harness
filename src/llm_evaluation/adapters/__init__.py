"""Adaptadores de dataset → ``EvalItem``."""

from __future__ import annotations

from llm_evaluation.adapters.amostra_local import amostra_local_items
from llm_evaluation.adapters.hf_generic import load_hf_qa_generic

__all__ = [
    "amostra_local_items",
    "load_hf_qa_generic",
]

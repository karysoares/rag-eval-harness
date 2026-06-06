"""Carrega conjuntos QA genéricos do Hugging Face."""

from __future__ import annotations

import random
from typing import Any, cast

from datasets import Dataset, load_dataset

from llm_evaluation.types import EvalItem


def load_hf_qa_generic(
    hf_repo: str,
    *,
    hf_subset: str | None,
    split: str,
    limit: int,
    seed: int,
    question_column: str,
    answer_column: str,
    context_column: str | None,
    incorrect_column: str | None,
    id_column: str | None,
    shuffle: bool = True,
) -> list[EvalItem]:
    """Carrega um dataset tabular com colunas configuráveis."""
    if hf_subset:
        ds = cast(Dataset, load_dataset(hf_repo, hf_subset, split=split))
    else:
        ds = cast(Dataset, load_dataset(hf_repo, split=split))
    rows = ds.to_list()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    selected = rows if limit <= 0 else rows[:limit]

    items: list[EvalItem] = []
    for i, raw in enumerate(selected):
        row = cast(dict[str, Any], raw)
        q = str(row.get(question_column, "")).strip()
        ans = row.get(answer_column)
        if ans is None:
            continue
        if isinstance(ans, list):
            correct = [str(x).strip() for x in ans if str(x).strip()]
        else:
            correct = [str(ans).strip()] if str(ans).strip() else []

        incorrect_raw = row.get(incorrect_column) if incorrect_column else None
        if incorrect_raw is None:
            incorrect: list[str] = []
        elif isinstance(incorrect_raw, list):
            incorrect = [str(x).strip() for x in incorrect_raw if str(x).strip()]
        else:
            incorrect = [str(incorrect_raw).strip()] if str(incorrect_raw).strip() else []

        ctx = row.get(context_column) if context_column else None
        rag_gold = str(ctx).strip() if ctx is not None and str(ctx).strip() else None

        rid = row.get(id_column) if id_column else None
        eid = str(rid) if rid is not None and str(rid) else f"{hf_repo.split('/')[-1]}-{i}"

        if not q or not correct:
            continue

        items.append(
            EvalItem(
                id=eid,
                question=q,
                correct_answers=correct,
                incorrect_answers=incorrect,
                category=str(row.get("category", "") or row.get("tipo", "") or ""),
                rag_gold_chunk=rag_gold,
                rag_distractors=[],
            ),
        )
    return items

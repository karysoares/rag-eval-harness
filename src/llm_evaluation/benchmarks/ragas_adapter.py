"""Adaptador RAGAS (extra opcional `ragas`)."""

from __future__ import annotations

from llm_evaluation.types import RunRecord


def run_ragas_sample(
    records: list[RunRecord],
    *,
    max_items: int = 100,
) -> dict[str, object]:
    """Calcula faithfulness/answer_relevancy numa amostra se ragas instalado."""
    try:
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore[import-not-found]
    except ImportError:
        return {
            "disponivel": False,
            "nota": "Instale com: uv sync --extra ragas",
        }

    sample = records[:max_items]
    rows: list[dict[str, object]] = []
    for r in sample:
        ctx = " ".join(c.text[:500] for c in r.retrieved[:5])
        rows.append(
            {
                "question": r.question[:2000],
                "answer": r.answer[:2000],
                "contexts": [ctx] if ctx else [""],
            },
        )
    if not rows:
        return {"disponivel": True, "n": 0}
    try:
        result = evaluate(rows, metrics=[faithfulness, answer_relevancy])
        scores = result if isinstance(result, dict) else getattr(result, "scores", result)
        return {"disponivel": True, "n": len(rows), "scores": scores}
    except Exception as e:
        return {"disponivel": True, "n": len(rows), "erro": str(e)}

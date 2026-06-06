"""Adaptador RAGAS (extra opcional `ragas`)."""

from __future__ import annotations

from llm_evaluation.types import RunRecord


def _records_to_ragas_rows(records: list[RunRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for r in records:
        ctx_parts = [c.text[:500] for c in r.retrieved[:5] if c.text.strip()]
        rows.append(
            {
                "question": r.question[:2000],
                "answer": r.answer[:2000],
                "contexts": ctx_parts if ctx_parts else [""],
            },
        )
    return rows


def _scores_from_result(result: object) -> dict[str, object]:
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        df = to_pandas()
        out: dict[str, object] = {}
        skip = frozenset({"question", "answer", "contexts", "ground_truth", "reference"})
        for col in df.columns:
            if col in skip:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                out[col] = None
                continue
            try:
                nums = [float(x) for x in series.tolist()]
            except (TypeError, ValueError):
                continue
            out[col] = sum(nums) / len(nums) if nums else None
            out[f"{col}_por_item"] = nums
        return out

    if isinstance(result, dict):
        raw = dict(result)
    else:
        scores = getattr(result, "scores", None)
        if isinstance(scores, dict):
            raw = dict(scores)
        else:
            return {"raw": str(result)}

    out = {}
    for key, val in raw.items():
        if isinstance(val, list):
            nums = [float(x) for x in val if isinstance(x, int | float)]
            out[key] = sum(nums) / len(nums) if nums else None
            out[f"{key}_por_item"] = val
        else:
            out[key] = val
    return out


def run_ragas_sample(
    records: list[RunRecord],
    *,
    max_items: int = 100,
) -> dict[str, object]:
    """Calcula faithfulness/answer_relevancy numa amostra se ragas instalado."""
    try:
        from datasets import Dataset
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore[import-not-found]
    except ImportError:
        return {
            "disponivel": False,
            "nota": "Instale com: uv sync --extra ragas",
        }

    sample = records[:max_items]
    rows = _records_to_ragas_rows(sample)
    if not rows:
        return {"disponivel": True, "n": 0}

    try:
        dataset = Dataset.from_list(rows)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = _scores_from_result(result)
        out: dict[str, object] = {
            "disponivel": True,
            "n": len(rows),
            "scores": scores,
        }
        for key in ("faithfulness", "answer_relevancy"):
            val = scores.get(key)
            if isinstance(val, int | float):
                out[f"media_{key}"] = float(val)
        return out
    except Exception as e:
        return {"disponivel": True, "n": len(rows), "erro": str(e)}


def summarize_harness_grounding(records: list[RunRecord]) -> dict[str, object]:
    """Agregados harness na mesma amostra (para comparar com RAGAS)."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    emb_low = sum(1 for r in records if r.signals.embedding_low_support is True)
    juiz_sust = sum(
        1 for r in records if r.signals.judge and r.signals.judge.veredito == "sustentado"
    )
    f1_vals = []
    for r in records:
        lm = r.meta.get("metricas_lexicas") or r.meta.get("lexical_metrics")
        if isinstance(lm, dict) and lm.get("f1_token") is not None:
            f1_vals.append(float(lm["f1_token"]))
    media_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else None
    return {
        "n": n,
        "taxa_embedding_baixo": emb_low / n,
        "taxa_juiz_sustentado": juiz_sust / n if n else None,
        "media_f1_token": media_f1,
    }

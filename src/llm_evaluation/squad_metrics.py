"""Métricas token-level estilo SQuAD / NQ-Open (F1 e EM sobre múltiplas referências)."""

from __future__ import annotations

import re
import string
from typing import Any


def normalize_squad(text: str) -> str:
    """Normalização oficial SQuAD: lower, sem pontuação, sem artigos, espaços colapsados."""

    def lower(s: str) -> str:
        return s.lower()

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _f1_for_pair(prediction: str, ground_truth: str) -> tuple[float, bool]:
    pred_norm = normalize_squad(prediction)
    gold_norm = normalize_squad(ground_truth)
    if pred_norm == gold_norm:
        return 1.0, True
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens and not gold_tokens:
        return 1.0, True
    if not pred_tokens or not gold_tokens:
        return 0.0, False
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0, False
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    f1 = 2 * prec * rec / (prec + rec)
    return f1, False


def squad_exact_match(prediction: str, references: list[str]) -> bool:
    """EM SQuAD: match exacto após normalização contra qualquer referência."""
    clean = [str(r).strip() for r in references if str(r).strip()]
    if not clean:
        return False
    return any(_f1_for_pair(prediction, ref)[1] for ref in clean)


def squad_max_f1(prediction: str, references: list[str]) -> tuple[float, int | None]:
    """F1 máximo sobre todas as referências (protocolo NQ-Open / SQuAD multi-ref)."""
    clean = [str(r).strip() for r in references if str(r).strip()]
    if not clean:
        return 0.0, None
    best_f = -1.0
    best_i: int | None = None
    for i, ref in enumerate(clean):
        f1, _ = _f1_for_pair(prediction, ref)
        if f1 > best_f:
            best_f = f1
            best_i = i
    return best_f, best_i


def squad_scores(prediction: str, references: list[str]) -> dict[str, Any]:
    """Pacote EM + F1 max para meta de métricas léxicas."""
    f1, idx = squad_max_f1(prediction, references)
    return {
        "f1_token": f1,
        "em_squad": squad_exact_match(prediction, references),
        "indice_referencia_f1": idx,
    }

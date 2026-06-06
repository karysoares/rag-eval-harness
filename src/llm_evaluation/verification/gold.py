"""Heurísticas para referências do tipo ``answer_lists``. Ver `docs/metrics.md`."""

from __future__ import annotations

import re
import unicodedata


def normalize_answer(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.?!,;:]+$", "", t).strip()
    return t


_REFUSAL_RE = re.compile(
    r"\b("
    r"cannot|can't|do not know|don't know|no comment|not sure|unable to|"
    r"cannot answer|can't answer|"
    r"não sei|nao sei|não posso|nao posso|não consigo|nao consigo|"
    r"sem comentário|sem comentario|"
    r"não tenho certeza|nao tenho certeza|incapaz de|"
    r"não há informações suficientes|nao ha informacoes suficientes|"
    r"não há informações sobre|nao ha informacoes sobre|"
    r"não há dados|nao ha dados"
    r")\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    a = answer.strip()
    if len(a) < 20 and _REFUSAL_RE.search(a):
        return True
    return bool(_REFUSAL_RE.search(a) and len(a) < 80)


def gold_correct(answer: str, correct: list[str], incorrect: list[str]) -> bool:
    an = normalize_answer(answer)
    if not an:
        return False
    for c in correct:
        cn = normalize_answer(str(c))
        if cn in an or an in cn:
            return True
    for inc in incorrect:
        inn = normalize_answer(str(inc))
        if inn and (inn in an or an in inn):
            return False
    return False


def gold_incorrect(answer: str, correct: list[str], incorrect: list[str]) -> bool:
    if gold_correct(answer, correct, incorrect):
        return False
    an = normalize_answer(answer)
    if not an:
        return True
    for inc in incorrect:
        inn = normalize_answer(str(inc))
        if inn and (inn in an or an in inn):
            return True
    # If not explicitly correct and has substance, treat as incorrect for metrics
    if is_refusal(answer):
        return False
    return not gold_correct(answer, correct, incorrect)

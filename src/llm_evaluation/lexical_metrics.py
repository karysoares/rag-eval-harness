"""Métricas léxicas (referência vs resposta do modelo): BLEU, ROUGE-L, METEOR, Levenshtein."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz.distance import Levenshtein
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU  # type: ignore[attr-defined]

from llm_evaluation.config import AppConfig, LexicalMetricsConfig, LexicalReferenceMode
from llm_evaluation.squad_metrics import squad_scores
from llm_evaluation.types import EvalItem
from llm_evaluation.verification.gold import normalize_answer

_BLEU = BLEU(effective_order=True)
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def pick_reference(
    hypothesis: str,
    refs: list[str],
    mode: LexicalReferenceMode,
) -> tuple[str, dict[str, Any]]:
    """Escolhe uma string de referência entre várias corretas possíveis."""
    clean = [r.strip() for r in refs if r and str(r).strip()]
    if not clean:
        return "", {"estrategia": mode, "indice_referencia": None}
    if mode == "primeiro":
        return clean[0], {"estrategia": mode, "indice_referencia": 0}
    if mode == "mais_longo":
        idx = max(range(len(clean)), key=lambda i: len(clean[i]))
        return clean[idx], {"estrategia": mode, "indice_referencia": idx}
    # max_rouge_l: maximiza F1 ROUGE-L contra a hipótese
    best_i = 0
    best_f = -1.0
    for i, ref in enumerate(clean):
        sc = _ROUGE.score(ref, hypothesis)
        f1 = float(sc["rougeL"].fmeasure)
        if f1 > best_f:
            best_f = f1
            best_i = i
    return clean[best_i], {
        "estrategia": mode,
        "indice_referencia": best_i,
        "max_rouge_l_f": best_f,
    }


def _levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = float(Levenshtein.distance(a, b))
    denom = float(max(len(a), len(b), 1))
    return max(0.0, 1.0 - dist / denom)


def _meteor_score_optional(reference: str, hypothesis: str) -> float | None:
    """METEOR via NLTK quando disponível (WordNet opcional); senão None.

    Falhas do NLTK (corpora em falta, bugs internos, inputs patológicos) são
    engolidas: a pipeline de avaliação não deve abortar por causa de METEOR.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        return None

    def _tok(s: str) -> list[str]:
        return re.findall(r"\w+", s.lower(), flags=re.UNICODE)

    rt = _tok(reference)
    ht = _tok(hypothesis)
    if not rt or not ht:
        return None
    try:
        # NLTK ≥3.9: ``meteor_score(references, hypothesis)`` — ``references`` é uma
        # *lista* de referências pré-tokenizadas (cada uma ``list[str]``).
        # Passar ``rt`` direto faz o NLTK iterar tokens como se fossem várias frases
        # e levanta TypeError (ex.: token ``"air"`` como ``str``).
        return float(meteor_score([rt], ht))
    except Exception:  # noqa: BLE001 — METEOR é opcional; nunca derrubar a corrida
        return None


def compute_lexical_scores(
    hypothesis: str,
    correct_answers: list[str],
    cfg: LexicalMetricsConfig,
) -> dict[str, Any]:
    """Calcula métricas pedidas; chaves omitidas se métrica desligada ou sem referência."""
    ref, ref_meta = pick_reference(hypothesis, correct_answers, cfg.reference_mode)
    out: dict[str, Any] = {
        "modo_referencia": cfg.reference_mode,
        "texto_referencia": ref[:500] if ref else "",
        **ref_meta,
    }
    if not ref.strip():
        out["note"] = "sem_referencia"
        return out

    hyp_norm = normalize_answer(hypothesis)
    out["exact_match"] = any(hyp_norm == normalize_answer(ans) for ans in correct_answers)
    out["exact_match_normalizado"] = bool(
        hyp_norm == normalize_answer(ref),
    )
    if cfg.token_f1:
        out.update(squad_scores(hypothesis, correct_answers))

    if cfg.bleu:
        s = _BLEU.sentence_score(hypothesis, [ref])
        # sacrebleu devolve score em escala 0–100
        out["bleu"] = float(s.score) / 100.0

    if cfg.rouge_l:
        sc = _ROUGE.score(ref, hypothesis)
        out["rouge_l_precisao"] = float(sc["rougeL"].precision)
        out["rouge_l_revocacao"] = float(sc["rougeL"].recall)
        out["rouge_l_f"] = float(sc["rougeL"].fmeasure)

    if cfg.meteor:
        m = _meteor_score_optional(ref, hypothesis)
        if m is not None:
            out["meteor"] = m

    if cfg.levenshtein:
        out["similaridade_levenshtein"] = _levenshtein_similarity(ref, hypothesis)

    return out


def empty_lexical_dict() -> dict[str, Any]:
    return {"note": "metricas_lexicas_desligadas"}


def attach_lexical_to_meta(
    meta: dict[str, Any],
    cfg: AppConfig,
    item: EvalItem,
    answer: str,
) -> None:
    """Preenche ``meta[\"metricas_lexicas\"]`` in-place (habilitado ou marcador desligado)."""
    if cfg.lexical_metrics.enabled:
        try:
            meta["metricas_lexicas"] = compute_lexical_scores(
                answer,
                item.correct_answers,
                cfg.lexical_metrics,
            )
        except Exception as exc:  # noqa: BLE001 — métricas léxicas não bloqueiam geração/juiz
            msg = f"{type(exc).__name__}: {exc}"
            meta["metricas_lexicas"] = {
                "note": "erro_ao_calcular_metricas_lexicas",
                "erro": msg[:500],
            }
    else:
        meta["metricas_lexicas"] = empty_lexical_dict()

"""Métricas F1/EM estilo SQuAD."""

from __future__ import annotations

from llm_evaluation.squad_metrics import squad_exact_match, squad_max_f1, squad_scores


def test_em_squad_multi_ref() -> None:
    refs = ["1951–52", "1951-52", "1951 52"]
    assert squad_exact_match("1951-1952", refs) is False
    assert squad_exact_match("1951-52", refs) is True


def test_f1_partial_credit() -> None:
    f1, _ = squad_max_f1("Caucasus region", ["Caucasus Mountains"])
    assert 0.3 < f1 < 1.0


def test_f1_max_over_references() -> None:
    sc = squad_scores("Tommy Shaw", ["Styx", "Tommy Shaw"])
    assert sc["em_squad"] is True
    assert sc["f1_token"] == 1.0

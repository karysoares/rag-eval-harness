"""Testes para detecção determinística de padrões (SPEC-007)."""

from __future__ import annotations

from llm_evaluation.pattern_detection import compute_diagnostico, has_placeholder
from llm_evaluation.pattern_registry import (
    PATTERN_CATALOG_VERSION,
    get_catalog,
    pick_primary,
)
from llm_evaluation.types import EvalItem, VerificationSignals


def test_resposta_vazia() -> None:
    item = EvalItem(id="1", question="Q?", correct_answers=["x"], incorrect_answers=[])
    sig = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    d = compute_diagnostico(
        item=item,
        answer="  ",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 0.0}},
        anomaly_flag=False,
    )
    assert "resposta_vazia" in d["padroes"]
    assert d["padrao_primario"] == "resposta_vazia"


def test_grounding_fp_suspeito() -> None:
    item = EvalItem(id="2", question="Q?", correct_answers=["Paris"], incorrect_answers=[])
    sig = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.2,
        embedding_low_support=True,
        judge=None,
        judge_negative=None,
    )
    d = compute_diagnostico(
        item=item,
        answer="Paris",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 1.0, "em_squad": True}},
        anomaly_flag=True,
    )
    assert "grounding_fp_suspeito" in d["padroes"]
    assert "referencia_forte" in d["padroes"]


def test_referencia_forte_por_f1() -> None:
    item = EvalItem(id="3", question="Q?", correct_answers=["a"], incorrect_answers=[])
    sig = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.8,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    d = compute_diagnostico(
        item=item,
        answer="a",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 0.85}},
        anomaly_flag=False,
    )
    assert d["padrao_primario"] == "referencia_forte"
    assert d["tier_qualidade"] == "alta"


def test_placeholder_regex() -> None:
    assert has_placeholder("The winner is <TEAM_NAME>")
    assert not has_placeholder("Paris")


def test_recuperacao_falhou() -> None:
    item = EvalItem(
        id="4",
        question="Q?",
        correct_answers=["x"],
        incorrect_answers=[],
        rag_gold_chunk="gold text",
    )
    sig = VerificationSignals(
        gold_correct=None,
        gold_incorrect=None,
        is_refusal=False,
        embedding_max_cosine=None,
        embedding_low_support=None,
        judge=None,
        judge_negative=None,
    )
    d = compute_diagnostico(
        item=item,
        answer="x",
        signals=sig,
        meta={
            "metricas_recuperacao": {
                "rag_ativo": True,
                "corpus_tem_chunk_ouro": True,
                "chunk_ouro_no_top_k": False,
            },
        },
        anomaly_flag=False,
    )
    assert "recuperacao_falhou" in d["padroes"]
    assert d["padrao_primario"] == "recuperacao_falhou"


def test_catalog_version_and_padroes_meta() -> None:
    item = EvalItem(id="5", question="Q?", correct_answers=["a"], incorrect_answers=[])
    sig = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    d = compute_diagnostico(
        item=item,
        answer="a",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 0.9}},
        anomaly_flag=False,
    )
    assert d["catalog_version"] == PATTERN_CATALOG_VERSION
    assert any(m["id"] == "referencia_forte" for m in d["padroes_meta"])


def test_override_referencia_forte_threshold() -> None:
    item = EvalItem(id="6", question="Q?", correct_answers=["a"], incorrect_answers=[])
    sig = VerificationSignals(
        gold_correct=True,
        gold_incorrect=False,
        is_refusal=False,
        embedding_max_cosine=0.9,
        embedding_low_support=False,
        judge=None,
        judge_negative=None,
    )
    d_default = compute_diagnostico(
        item=item,
        answer="a",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 0.85}},
        anomaly_flag=False,
    )
    assert d_default["padrao_primario"] == "referencia_forte"

    d_strict = compute_diagnostico(
        item=item,
        answer="a",
        signals=sig,
        meta={"metricas_lexicas": {"f1_token": 0.85}},
        anomaly_flag=False,
        pattern_overrides={"referencia_forte": {"f1_min": 0.95}},
    )
    assert "referencia_fraca" in d_strict["padroes"]
    assert d_strict["padrao_primario"] == "referencia_fraca"


def test_registry_priority_recusa_before_referencia() -> None:
    padroes = ["referencia_forte", "recusa"]
    assert pick_primary(padroes) == "recusa"


def test_get_catalog_has_all_patterns() -> None:
    cat = get_catalog()
    assert cat["catalog_version"] == PATTERN_CATALOG_VERSION
    ids = {p["id"] for p in cat["padroes"]}
    assert "grounding_fp_suspeito" in ids
    assert "juiz_fallback" in ids

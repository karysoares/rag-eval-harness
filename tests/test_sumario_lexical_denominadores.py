"""Cada média léxica publica o seu próprio denominador.

Métricas diferentes falham em itens diferentes, e uma média sem o `n` ao lado
lê-se como se cobrisse todos os itens pontuados. Quando a falha depende da
dificuldade do par — o METEOR sem `wordnet` só devolve valor no casamento
exacto — o subconjunto que sobrevive é o mais fácil, e a média sai enviesada
para cima em vez de apenas ruidosa.
"""

from __future__ import annotations

from llm_evaluation.reporting import summarize
from llm_evaluation.types import RunRecord, VerificationSignals


def _registo(item_id: str, **lexical: object) -> RunRecord:
    return RunRecord(
        item_id=item_id,
        question="p",
        answer="r",
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={"metricas_lexicas": {"texto_referencia": "ref", "f1_token": 0.5, **lexical}},
    )


def test_n_por_metrica_expoe_cobertura_parcial() -> None:
    registos = [
        _registo("a", meteor=0.99),
        _registo("b", meteor_indisponivel="recurso_nltk_ausente"),
        _registo("c", meteor_indisponivel="recurso_nltk_ausente"),
    ]
    lex = summarize(registos, reference_type="lexical")["sumario_lexical"]

    assert lex["n_itens_pontuados"] == 3
    assert lex["n_por_metrica"]["meteor"] == 1
    assert lex["n_por_metrica"]["f1_token"] == 3
    assert lex["meteor_indisponivel"] == {"recurso_nltk_ausente": 2}
    assert "1 de 3" in str(lex["nota_cobertura"])


def test_cobertura_total_nao_gera_nota() -> None:
    registos = [_registo("a", meteor=0.9), _registo("b", meteor=0.8)]
    lex = summarize(registos, reference_type="lexical")["sumario_lexical"]
    assert lex["n_por_metrica"]["meteor"] == 2
    assert "nota_cobertura" not in lex

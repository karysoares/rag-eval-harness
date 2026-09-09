"""Normalização léxica em português: tokenização, artigos e ênclise.

O F1 token é a referência com que ``reference_metrics.referencia_incorreta``
rotula itens em corpora ``reference_type: lexical`` — e desses rótulos saem a
exactidão e o κ do juiz. Um viés aqui propaga-se a toda a meta-avaliação, por
isso cada regra tem um caso que a fixa.
"""

from __future__ import annotations

import pytest

from llm_evaluation.config import LexicalMetricsConfig
from llm_evaluation.lexical_metrics import compute_lexical_scores
from llm_evaluation.squad_metrics import normalize_squad, squad_scores


def _cfg(**kwargs: object) -> LexicalMetricsConfig:
    base: dict[str, object] = {
        "enabled": True,
        "bleu": False,
        "rouge_l": True,
        "meteor": False,
        "levenshtein": False,
        "token_f1": True,
        "reference_mode": "max_rouge_l",
        "idioma": "pt",
    }
    base.update(kwargs)
    return LexicalMetricsConfig(**base)  # type: ignore[arg-type]  # dict tipado acima


class TestTokenizacaoUnicode:
    """O ROUGE tem de ver palavras, não fragmentos de palavras acentuadas."""

    @pytest.mark.parametrize(
        ("texto", "esperado"),
        [
            ("A ação começou à noite", ["a", "ação", "começou", "à", "noite"]),
            ("informação", ["informação"]),
            ("O coração da avó", ["o", "coração", "da", "avó"]),
        ],
    )
    def test_palavra_acentuada_nao_e_partida(self, texto: str, esperado: list[str]) -> None:
        from llm_evaluation.lexical_metrics import _rouge

        assert _rouge()._tokenizer.tokenize(texto) == esperado

    def test_rouge_de_frase_identica_e_um(self) -> None:
        out = compute_lexical_scores(
            "A avó contou uma história.", ["A avó contou uma história."], _cfg()
        )
        assert out["rouge_l_f"] == pytest.approx(1.0)


class TestNormalizacaoPortuguesa:
    def test_enclise_separa_em_vez_de_colar(self) -> None:
        assert normalize_squad("Ele deu-lhe o livro", "pt") == "ele deu lhe o livro"

    def test_composto_com_hifen_separa(self) -> None:
        assert normalize_squad("o guarda-chuva", "pt") == "o guarda chuva"

    def test_preposicao_a_nao_e_removida(self) -> None:
        # A regra inglesa apagava ``a`` e transformava «vou a Lisboa» em «vou lisboa».
        assert normalize_squad("Vou a Lisboa", "pt") == "vou a lisboa"

    def test_remocao_de_artigos_e_simetrica(self) -> None:
        # Antes só caía o ``a`` feminino singular; ``os``/``as`` ficavam.
        assert normalize_squad("as casas e os carros", "pt") == "casas e carros"

    def test_aspas_portuguesas_sao_pontuacao(self) -> None:
        assert normalize_squad("«sim»", "pt") == "sim"

    def test_ingles_mantem_protocolo_oficial(self) -> None:
        # Comparabilidade com resultados publicados em inglês: byte a byte.
        assert normalize_squad("The girl gave her the umbrella", "en") == "girl gave her umbrella"
        # O protocolo oficial cola nos hífenes antes de remover artigos, pelo que
        # o ``the`` interior sobrevive. Fica fixado por ser o comportamento que
        # torna comparáveis os números publicados em inglês, não por ser o melhor.
        assert normalize_squad("state-of-the-art", "en") == "stateoftheart"


class TestImpactoNoF1:
    def test_enclise_deixa_de_ser_penalizada(self) -> None:
        pt = squad_scores("Ele deu-lhe o livro.", ["Ele deu o livro a ela."], "pt")
        en = squad_scores("Ele deu-lhe o livro.", ["Ele deu o livro a ela."], "en")
        assert pt["f1_token"] > en["f1_token"]

    def test_idioma_activo_fica_no_artefacto(self) -> None:
        out = compute_lexical_scores("resposta", ["referência"], _cfg())
        assert out["idioma"] == "pt"


class TestMeteorAusenteEContado:
    def test_ausencia_de_meteor_e_nomeada(self) -> None:
        """Sem valor, o artefacto diz porquê — a média não pode fingir cobertura total."""
        out = compute_lexical_scores(
            "Uma resposta completamente diferente da referência",
            ["Texto de referência sem relação nenhuma"],
            _cfg(meteor=True),
        )
        assert ("meteor" in out) or (out.get("meteor_indisponivel") is not None)

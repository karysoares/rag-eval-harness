"""Métricas token-level estilo SQuAD / NQ-Open (F1 e EM sobre múltiplas referências).

A normalização oficial do SQuAD foi escrita para inglês e transplantá-la para
português enviesa a medição de três formas mensuráveis:

* ``remove_articles`` apaga ``a|an|the``. Em português apaga o artigo feminino
  singular ``a`` e mantém ``o``, ``os``, ``as``, ``um``, ``uma`` — remoção
  assimétrica em género e número, além de comer a preposição ``a``
  («vou a Lisboa» → «vou lisboa»).
* ``remove_punc`` retira o hífen **sem** inserir espaço, colando a ênclise e os
  compostos: «deu-lhe» → «deulhe», «guarda-chuva» → «guardachuva». Nenhum destes
  casa com a forma equivalente sem hífen.
* O resultado é um F1 que penaliza português correcto — e este F1 é a referência
  léxica com que se rotulam os itens de onde saem a exactidão e o κ do juiz.

Daí o parâmetro ``idioma``. ``"en"`` reproduz o protocolo oficial byte a byte,
para não quebrar comparabilidade com resultados publicados em inglês; ``"pt"``
aplica as regras portuguesas. O idioma activo é registado no artefacto.
"""

from __future__ import annotations

import re
import string
from typing import Any, Literal

Idioma = Literal["pt", "en"]

#: Artigos e indefinidos removidos em português, simétricos em género e número.
#: A preposição ``a`` fica de fora: removê-la apagaria complementos
#: («vou a Lisboa»), e tolerar o artigo homógrafo custa menos do que isso.
_ARTIGOS_PT = re.compile(r"\b(os|as|um|uma|uns|umas)\b")

_ARTICLES_EN = re.compile(r"\b(a|an|the)\b")

#: Pontuação que separa palavras: vira espaço em vez de desaparecer. O hífen é o
#: caso que interessa (ênclise, mesóclise, compostos); a barra e o travessão
#: seguem a mesma lógica.
_PONTUACAO_SEPARADORA = str.maketrans({"-": " ", "–": " ", "—": " ", "/": " "})

_PONTUACAO_ASCII = set(string.punctuation)

#: Aspas e travessões fora do ASCII, comuns em texto português.
_PONTUACAO_EXTRA = set("«»“”‘’…–—")


def _lower(s: str) -> str:
    return s.lower()


def _white_space_fix(s: str) -> str:
    return " ".join(s.split())


def _remove_punc(s: str, *, extra: bool) -> str:
    excluir = _PONTUACAO_ASCII | _PONTUACAO_EXTRA if extra else _PONTUACAO_ASCII
    return "".join(ch for ch in s if ch not in excluir)


def normalize_squad(text: str, idioma: Idioma = "en") -> str:
    """Normaliza para comparação token a token.

    ``idioma="en"`` é a normalização oficial do SQuAD, mantida intacta.
    ``idioma="pt"`` separa em vez de colar nos hífenes e remove artigos
    portugueses de forma simétrica.
    """
    if idioma == "pt":
        s = _lower(text).translate(_PONTUACAO_SEPARADORA)
        s = _remove_punc(s, extra=True)
        return _white_space_fix(_ARTIGOS_PT.sub(" ", s))
    s = _remove_punc(_lower(text), extra=False)
    return _white_space_fix(_ARTICLES_EN.sub(" ", s))


def _f1_for_pair(prediction: str, ground_truth: str, idioma: Idioma) -> tuple[float, bool]:
    pred_norm = normalize_squad(prediction, idioma)
    gold_norm = normalize_squad(ground_truth, idioma)
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


def squad_exact_match(prediction: str, references: list[str], idioma: Idioma = "en") -> bool:
    """EM SQuAD: match exacto após normalização contra qualquer referência."""
    clean = [str(r).strip() for r in references if str(r).strip()]
    if not clean:
        return False
    return any(_f1_for_pair(prediction, ref, idioma)[1] for ref in clean)


def squad_max_f1(
    prediction: str,
    references: list[str],
    idioma: Idioma = "en",
) -> tuple[float, int | None]:
    """F1 máximo sobre todas as referências (protocolo NQ-Open / SQuAD multi-ref)."""
    clean = [str(r).strip() for r in references if str(r).strip()]
    if not clean:
        return 0.0, None
    best_f = -1.0
    best_i: int | None = None
    for i, ref in enumerate(clean):
        f1, _ = _f1_for_pair(prediction, ref, idioma)
        if f1 > best_f:
            best_f = f1
            best_i = i
    return best_f, best_i


def squad_scores(
    prediction: str,
    references: list[str],
    idioma: Idioma = "en",
) -> dict[str, Any]:
    """Pacote EM + F1 max para meta de métricas léxicas."""
    f1, idx = squad_max_f1(prediction, references, idioma)
    return {
        "f1_token": f1,
        "em_squad": squad_exact_match(prediction, references, idioma),
        "indice_referencia_f1": idx,
    }

"""Normalização de vereditos do juiz (português; aceita legado em inglês ao ler)."""

from __future__ import annotations

import unicodedata

from llm_evaluation.types import Verdict

_ALIAS: dict[str, Verdict] = {
    # Português (saída esperada do modelo)
    "sustentado": "sustentado",
    "nao_sustentado": "nao_sustentado",
    "contradicacao": "contradicacao",
    "incompleto": "incompleto",
    "inseguro": "inseguro",
    # Legado / APIs em inglês
    "supported": "sustentado",
    "unsupported": "nao_sustentado",
    "contradiction": "contradicacao",
    "incomplete": "incompleto",
    "unsafe": "inseguro",
}


def _fold(s: str) -> str:
    t = unicodedata.normalize("NFKD", s.strip().lower())
    return "".join(c for c in t if not unicodedata.combining(c)).replace(" ", "_")


def parse_veredito_estrito(valor: str) -> Verdict | None:
    """Mapeia veredito só se reconhecido; ``None`` se fora do enum (validação juiz)."""
    key = _fold(valor)
    if key in _ALIAS:
        return _ALIAS[key]
    if key.replace("_", "") == "naosustentado":
        return "nao_sustentado"
    return None


def normalizar_veredito(valor: str) -> Verdict:
    """Converte texto do juiz para veredito canónico; desconhecido → ``sustentado`` (legado)."""
    parsed = parse_veredito_estrito(valor)
    return parsed if parsed is not None else "sustentado"


def veredito_e_negativo(veredito: Verdict, lista_negativa: list[str]) -> bool:
    """True se o veredito coincide com algum rótulo negativo configurado (pt ou legado en)."""
    return any(normalizar_veredito(str(x)) == veredito for x in lista_negativa)

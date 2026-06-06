"""Catálogo central de padrões determinísticos (SPEC-007 Fase 1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

PATTERN_CATALOG_VERSION: Final[str] = "1.0"

_PLACEHOLDER_RE_DEFAULT = re.compile(r"<[^>]+>")
_PLACEHOLDER_PHRASES_DEFAULT: tuple[str, ...] = ("specific winner", "check latest")

_DEFAULT_F1_FORTE = 0.8
_DEFAULT_F1_FRACA_MIN = 0.3


@dataclass(frozen=True)
class PatternDef:
    id: str
    categoria: str
    severidade: str
    descricao: str
    deterministico: bool = True


_PATTERN_DEFS: tuple[PatternDef, ...] = (
    PatternDef(
        "resposta_vazia",
        "estrutural",
        "critico",
        "Resposta em branco após strip.",
    ),
    PatternDef(
        "placeholder",
        "estrutural",
        "critico",
        "Marcadores tipo `<...>` ou frases-tipo proibidas no prompt.",
    ),
    PatternDef(
        "recusa",
        "estrutural",
        "medio",
        "Heurística `is_refusal` em gold.py.",
    ),
    PatternDef(
        "recuperacao_falhou",
        "recuperacao",
        "critico",
        "Corpus com chunk ouro mas chunk fora do top-k (SPEC-001).",
    ),
    PatternDef(
        "grounding_fp_suspeito",
        "grounding",
        "alto",
        "Gold substring positivo e embedding abaixo do limiar (SPEC-002).",
    ),
    PatternDef(
        "grounding_baixo",
        "grounding",
        "medio",
        "`embedding_baixo_suporte` activo (SPEC-002).",
    ),
    PatternDef(
        "referencia_ausente",
        "referencia",
        "alto",
        "F1 token abaixo do limiar fraco com referências presentes.",
    ),
    PatternDef(
        "referencia_fraca",
        "referencia",
        "medio",
        "F1 token entre limiar fraco e forte.",
    ),
    PatternDef(
        "referencia_forte",
        "referencia",
        "baixo",
        "EM SQuAD ou F1 ≥ limiar forte.",
    ),
    PatternDef(
        "juiz_fallback",
        "verificacao",
        "medio",
        "Juiz usou fallback heurístico (não entra na agregação SPEC-004).",
    ),
    PatternDef(
        "juiz_incompleto",
        "verificacao",
        "medio",
        "Juiz marcou incompleto (aviso; não entra na agregação por omissão).",
    ),
    PatternDef(
        "juiz_negativo",
        "verificacao",
        "alto",
        "`juiz_negativo` activo na agregação (vereditos críticos).",
    ),
    PatternDef(
        "anomalia",
        "verificacao",
        "alto",
        "Espelha `flag_anomalia`; não altera agregação (SPEC-004).",
    ),
    PatternDef(
        "ok",
        "sintese",
        "informativo",
        "Nenhum padrão de problema activo (rótulo primário sintético).",
    ),
)

_PRIORITY_ORDER: tuple[str, ...] = (
    "resposta_vazia",
    "placeholder",
    "recusa",
    "recuperacao_falhou",
    "grounding_fp_suspeito",
    "grounding_baixo",
    "referencia_ausente",
    "referencia_fraca",
    "referencia_forte",
    "juiz_fallback",
    "juiz_incompleto",
    "juiz_negativo",
    "anomalia",
    "ok",
)

_BY_ID: dict[str, PatternDef] = {p.id: p for p in _PATTERN_DEFS}


@dataclass
class PatternSettings:
    """Limiares e regex efectivos após merge com overrides YAML."""

    f1_forte_min: float = _DEFAULT_F1_FORTE
    f1_fraca_min: float = _DEFAULT_F1_FRACA_MIN
    placeholder_re: re.Pattern[str] = field(default_factory=lambda: _PLACEHOLDER_RE_DEFAULT)
    placeholder_phrases: tuple[str, ...] = _PLACEHOLDER_PHRASES_DEFAULT


def pick_primary(padroes: list[str]) -> str:
    for p in _PRIORITY_ORDER:
        if p in padroes:
            return p
    return "ok"


def build_pattern_settings(overrides: dict[str, dict[str, Any]] | None = None) -> PatternSettings:
    """Merge overrides planos por ID de padrão (ex.: ``referencia_forte.f1_min``)."""
    settings = PatternSettings()
    if not overrides:
        return settings

    rf = overrides.get("referencia_forte") or {}
    if "f1_min" in rf:
        settings.f1_forte_min = float(rf["f1_min"])

    rfr = overrides.get("referencia_fraca") or {}
    if "f1_min" in rfr:
        settings.f1_fraca_min = float(rfr["f1_min"])

    ra = overrides.get("referencia_ausente") or {}
    if "f1_max" in ra:
        settings.f1_fraca_min = float(ra["f1_max"])

    ph = overrides.get("placeholder") or {}
    phrases = ph.get("frases")
    if isinstance(phrases, list) and phrases:
        settings.placeholder_phrases = tuple(str(x).lower() for x in phrases)

    return settings


def meta_for_active_patterns(padroes: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for pid in padroes:
        d = _BY_ID.get(pid)
        if d is None:
            continue
        out.append(
            {
                "id": d.id,
                "categoria": d.categoria,
                "severidade": d.severidade,
            },
        )
    return out


def get_catalog() -> dict[str, Any]:
    """Exportação para dashboard / documentação."""
    return {
        "catalog_version": PATTERN_CATALOG_VERSION,
        "prioridade_padrao_primario": list(_PRIORITY_ORDER),
        "padroes": [
            {
                "id": p.id,
                "categoria": p.categoria,
                "severidade": p.severidade,
                "descricao": p.descricao,
                "deterministico": p.deterministico,
            }
            for p in _PATTERN_DEFS
        ],
        "limiares_default": {
            "referencia_forte": {"f1_min": _DEFAULT_F1_FORTE},
            "referencia_fraca": {"f1_min": _DEFAULT_F1_FRACA_MIN},
            "placeholder": {"frases": list(_PLACEHOLDER_PHRASES_DEFAULT)},
        },
    }

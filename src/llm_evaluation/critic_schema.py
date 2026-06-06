"""Validação estruturada da resposta JSON da crítica multi-agente."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CRITIC_SCHEMA_VERSION = "1.0"
NOTA_MAX_LEN = 500

PROBLEMAS_VALIDOS: frozenset[str] = frozenset(
    {
        "ancoragem",
        "contradicacao",
        "omissao",
        "alucinacao",
        "recusa_indevida",
        "fora_de_foco",
        "resposta_generica",
        "instrucao_adversarial",
        "nenhum",
    },
)

SEVERIDADES_VALIDAS: frozenset[str] = frozenset({"alta", "media", "baixa", "nenhuma"})


class CriticSchemaError(ValueError):
    """Resposta da crítica fora do contrato JSON."""


@dataclass(frozen=True)
class CriticResponseValidated:
    """Saída normalizada após validação de schema."""

    cadeia_de_pensamento: list[str]
    problemas: list[str]
    nota: str
    severidade: str | None = None
    flags: list[str] | None = None
    trechos_relevantes: list[str] | None = None
    schema_version: str = CRITIC_SCHEMA_VERSION


def _cadeia(raw: dict[str, Any]) -> list[str]:
    v = raw.get("cadeia_de_pensamento")
    if v is None:
        v = raw.get("chain_of_thought") or []
    if not isinstance(v, list):
        msg = "cadeia_de_pensamento deve ser lista"
        raise TypeError(msg)
    return [str(x).strip() for x in v if str(x).strip()]


def _problemas(raw: dict[str, Any]) -> list[str]:
    v = raw.get("problemas")
    if v is None:
        v = raw.get("issues") or []
    if not isinstance(v, list):
        msg = "problemas deve ser lista"
        raise TypeError(msg)
    out: list[str] = []
    for item in v:
        s = str(item).strip().lower()
        if s:
            out.append(s)
    return out


def _nota(raw: dict[str, Any]) -> str:
    return str(raw.get("nota") or raw.get("note") or "").strip()


def validate_critic_response(
    parsed: dict[str, Any],
    *,
    log_violations: bool = True,
) -> CriticResponseValidated:
    """Valida dict parseado da crítica; levanta ``CriticSchemaError`` se inválido."""
    violations: list[str] = []

    try:
        cadeia = _cadeia(parsed)
    except TypeError:
        violations.append("cadeia_de_pensamento inválida")
        cadeia = []

    if len(cadeia) < 1:
        violations.append("cadeia_de_pensamento vazia")

    try:
        problemas = _problemas(parsed)
    except TypeError:
        violations.append("problemas inválidos")
        problemas = []

    if not problemas:
        violations.append("problemas vazio")
    else:
        invalid = [p for p in problemas if p not in PROBLEMAS_VALIDOS]
        if invalid:
            violations.append(f"problemas fora do enum: {invalid}")

    nota = _nota(parsed)
    if not nota:
        violations.append("nota vazia")
    elif len(nota) > NOTA_MAX_LEN:
        nota = nota[:NOTA_MAX_LEN]

    severidade: str | None = None
    sev_raw = parsed.get("severidade")
    if sev_raw is not None:
        severidade = str(sev_raw).strip().lower()
        if severidade not in SEVERIDADES_VALIDAS:
            violations.append(f"severidade inválida: {severidade!r}")

    flags: list[str] | None = None
    flags_raw = parsed.get("flags")
    if flags_raw is not None:
        if not isinstance(flags_raw, list):
            violations.append("flags deve ser lista")
        else:
            flags = [str(x).strip() for x in flags_raw if str(x).strip()][:5]

    trechos: list[str] | None = None
    tr_raw = parsed.get("trechos_relevantes")
    if tr_raw is not None:
        if not isinstance(tr_raw, list):
            violations.append("trechos_relevantes deve ser lista")
        else:
            trechos = [str(x).strip() for x in tr_raw if str(x).strip()]

    if violations:
        if log_violations:
            logger.warning(
                "critic_schema_violation schema=%s violations=%s",
                CRITIC_SCHEMA_VERSION,
                violations,
            )
        raise CriticSchemaError("; ".join(violations))

    return CriticResponseValidated(
        cadeia_de_pensamento=cadeia,
        problemas=problemas,
        nota=nota,
        severidade=severidade,
        flags=flags,
        trechos_relevantes=trechos,
    )


def critic_to_dict(validated: CriticResponseValidated) -> dict[str, Any]:
    """Serializa crítica validada para ``meta.critica``."""
    out: dict[str, Any] = {
        "cadeia_de_pensamento": validated.cadeia_de_pensamento,
        "problemas": validated.problemas,
        "nota": validated.nota,
        "schema_version": validated.schema_version,
    }
    if validated.severidade is not None:
        out["severidade"] = validated.severidade
    if validated.flags:
        out["flags"] = validated.flags
    if validated.trechos_relevantes:
        out["trechos_relevantes"] = validated.trechos_relevantes
    return out

"""Validação estruturada da resposta JSON do respondedor RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RESPONDER_SCHEMA_VERSION = "1.0"
RESPOSTA_MAX_LEN = 2_000


class ResponderSchemaError(ValueError):
    """Resposta do respondedor fora do contrato JSON."""


@dataclass(frozen=True)
class ResponderResponseValidated:
    """Saída normalizada após validação de schema."""

    resposta: str
    confianca: float
    contexto_insuficiente: bool
    schema_version: str = RESPONDER_SCHEMA_VERSION


def _resposta(raw: dict[str, Any]) -> str:
    v = raw.get("resposta")
    if v is None:
        v = raw.get("answer") or raw.get("texto") or ""
    return str(v).strip()


def _confianca(raw: dict[str, Any]) -> float:
    v = raw.get("confianca")
    if v is None:
        v = raw.get("confidence", 0.5)
    return float(v)


def _contexto_insuficiente(raw: dict[str, Any]) -> bool:
    v = raw.get("contexto_insuficiente")
    if v is None:
        v = raw.get("insufficient_context")
    if v is None:
        return False
    if not isinstance(v, bool):
        msg = "contexto_insuficiente deve ser bool"
        raise TypeError(msg)
    return v


def validate_responder_response(
    parsed: dict[str, Any],
    *,
    log_violations: bool = True,
) -> ResponderResponseValidated:
    """Valida dict parseado do respondedor; levanta ``ResponderSchemaError`` se inválido."""
    violations: list[str] = []

    try:
        resp = _resposta(parsed)
    except (TypeError, ValueError):
        violations.append("resposta inválida")
        resp = ""

    if not resp:
        violations.append("resposta vazia")
    elif len(resp) > RESPOSTA_MAX_LEN:
        resp = resp[:RESPOSTA_MAX_LEN]

    try:
        conf = _confianca(parsed)
    except (TypeError, ValueError):
        violations.append("confianca inválida")
        conf = -1.0
    if not (0.0 <= conf <= 1.0):
        violations.append(f"confianca fora de [0,1]: {conf}")

    ctx_insuf: bool | None = None
    try:
        ctx_insuf = _contexto_insuficiente(parsed)
    except TypeError:
        violations.append("contexto_insuficiente deve ser bool")

    if violations:
        if log_violations:
            logger.warning(
                "responder_schema_violation schema=%s violations=%s",
                RESPONDER_SCHEMA_VERSION,
                violations,
            )
        raise ResponderSchemaError("; ".join(violations))

    assert ctx_insuf is not None
    return ResponderResponseValidated(
        resposta=resp,
        confianca=conf,
        contexto_insuficiente=ctx_insuf,
    )

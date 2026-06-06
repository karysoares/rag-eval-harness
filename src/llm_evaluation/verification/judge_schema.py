"""Validação estruturada da resposta JSON do juiz LLM (SPEC-003, Fase 1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from llm_evaluation.types import Verdict
from llm_evaluation.veredito import parse_veredito_estrito

logger = logging.getLogger(__name__)

JUDGE_SCHEMA_VERSION = "1.0"
MOTIVO_BREVE_MAX_LEN = 500

VERDICTS_CANONICOS: frozenset[Verdict] = frozenset(
    {"sustentado", "nao_sustentado", "contradicacao", "incompleto", "inseguro"},
)


class JudgeSchemaError(ValueError):
    """Resposta do juiz fora do contrato JSON."""


@dataclass(frozen=True)
class JudgeResponseValidated:
    """Saída normalizada após validação de schema."""

    veredito: Verdict
    confianca: float
    motivo_breve: str
    fallback_heuristico: bool = False
    schema_version: str = JUDGE_SCHEMA_VERSION


def sanitize_motivo_breve(text: str) -> str:
    """Trunca motivo ao limite do schema."""
    t = text.strip()
    if len(t) > MOTIVO_BREVE_MAX_LEN:
        return t[:MOTIVO_BREVE_MAX_LEN]
    return t


def _motivo_breve(raw: dict[str, Any]) -> str:
    return str(raw.get("motivo_breve") or raw.get("reason_short") or "").strip()


def _confianca(raw: dict[str, Any]) -> float:
    v = raw.get("confianca")
    if v is None:
        v = raw.get("confidence", 0.5)
    return float(v)


def validate_judge_response(
    parsed: dict[str, Any],
    *,
    log_violations: bool = True,
) -> JudgeResponseValidated:
    """Valida dict parseado do juiz; levanta ``JudgeSchemaError`` se inválido."""
    violations: list[str] = []

    ver_raw = parsed.get("veredito") or parsed.get("verdict")
    veredito: Verdict = "sustentado"
    if ver_raw is None or not str(ver_raw).strip():
        violations.append("veredito em falta")
    else:
        mapped = parse_veredito_estrito(str(ver_raw))
        if mapped is None:
            violations.append(f"veredito fora do enum: {ver_raw!r}")
        else:
            veredito = mapped

    try:
        conf = _confianca(parsed)
    except (TypeError, ValueError):
        violations.append("confianca inválida")
        conf = -1.0
    if not (0.0 <= conf <= 1.0):
        violations.append(f"confianca fora de [0,1]: {conf}")

    motivo = sanitize_motivo_breve(_motivo_breve(parsed))
    if not motivo:
        violations.append("motivo_breve vazio")

    fb = parsed.get("fallback_heuristico")
    if fb is not None and not isinstance(fb, bool):
        violations.append("fallback_heuristico deve ser bool")

    if violations:
        if log_violations:
            logger.warning(
                "judge_schema_violation schema=%s violations=%s",
                JUDGE_SCHEMA_VERSION,
                violations,
            )
        raise JudgeSchemaError("; ".join(violations))

    return JudgeResponseValidated(
        veredito=veredito,
        confianca=conf,
        motivo_breve=motivo,
        fallback_heuristico=bool(fb),
    )

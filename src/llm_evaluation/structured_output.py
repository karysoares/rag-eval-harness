"""Utilitários partilhados para parse e validação de saída estruturada LLM."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from llm_evaluation.llm_client import LlmClient, OpenAiCompatibleClient, parse_json_object_lenient

T = TypeVar("T")


class StructuredOutputError(ValueError):
    """Saída LLM fora do contrato JSON esperado."""


@dataclass(frozen=True)
class StructuredParseMeta:
    """Metadados de auditoria por chamada LLM estruturada."""

    retry_count: int
    parse_failures: int
    schema_invalid: bool
    structured_output_error: str | None = None


def complete_with_json_mode(
    client: LlmClient,
    system: str,
    user: str,
    *,
    json_object: bool,
) -> str:
    """Chama ``complete`` com ``json_object`` quando o cliente suporta."""
    if isinstance(client, OpenAiCompatibleClient):
        return client.complete(system, user, json_object=json_object)
    complete = getattr(client, "complete", None)
    if callable(complete):
        try:
            return str(complete(system, user, json_object=json_object))
        except TypeError:
            return str(complete(system, user))
    msg = "Cliente LLM sem método complete"
    raise TypeError(msg)


def parse_structured_dict(text: str) -> dict[str, Any]:
    """Parse JSON object; levanta ``StructuredOutputError`` em falha."""
    try:
        return parse_json_object_lenient(text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StructuredOutputError(str(exc)) from exc


def call_validate_with_retries(
    *,
    client: LlmClient,
    system: str,
    user: str,
    validate: Callable[[dict[str, Any]], T],
    max_parse_retries: int = 1,
    retry_suffix: str = "",
) -> tuple[T, dict[str, Any], StructuredParseMeta]:
    """Chama LLM, parseia JSON e valida; retenta em falha de parse/schema."""
    retry_count = 0
    parse_failures = 0
    last_error: str | None = None
    attempts = max(1, max_parse_retries + 1)

    for attempt in range(attempts):
        use_json_mode = attempt > 0
        sys_eff = system + (retry_suffix if attempt > 0 and retry_suffix else "")
        try:
            text = complete_with_json_mode(
                client,
                sys_eff,
                user,
                json_object=use_json_mode,
            )
            parsed = parse_structured_dict(text)
            validated = validate(parsed)
            meta = StructuredParseMeta(
                retry_count=retry_count,
                parse_failures=parse_failures,
                schema_invalid=parse_failures > 0,
                structured_output_error=None,
            )
            return validated, parsed, meta
        except (StructuredOutputError, ValueError) as exc:
            parse_failures += 1
            last_error = str(exc)
            if attempt < attempts - 1:
                retry_count += 1
                continue

    meta = StructuredParseMeta(
        retry_count=retry_count,
        parse_failures=parse_failures,
        schema_invalid=True,
        structured_output_error=last_error or "structured_output_invalid",
    )
    raise StructuredOutputError(last_error or "structured_output_invalid") from None


def structured_meta_as_dict(meta: StructuredParseMeta) -> dict[str, Any]:
    """Serializa metadados para ``meta`` de corrida."""
    out: dict[str, Any] = {
        "retry_count": meta.retry_count,
        "parse_failures": meta.parse_failures,
        "schema_invalid": meta.schema_invalid,
    }
    if meta.structured_output_error:
        out["structured_output_error"] = meta.structured_output_error
    return out

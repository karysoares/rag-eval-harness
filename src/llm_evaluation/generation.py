"""Geração de respostas condicionada aos chunks recuperados (RAG)."""

from __future__ import annotations

from typing import Any

from llm_evaluation.config import PromptStyle
from llm_evaluation.llm_client import LlmClient
from llm_evaluation.prompt_resources import load_prompt_text
from llm_evaluation.responder_schema import (
    RESPONDER_SCHEMA_VERSION,
    ResponderResponseValidated,
    ResponderSchemaError,
    validate_responder_response,
)
from llm_evaluation.retrieval_hints import format_retrieval_hints
from llm_evaluation.structured_output import (
    StructuredOutputError,
    call_validate_with_retries,
    structured_meta_as_dict,
)
from llm_evaluation.types import EvalItem, RetrievedChunk

_RESPONDER_RETRY_SUFFIX = (
    "\n\nIMPORTANTE: retorne APENAS um objeto JSON válido, sem markdown, "
    "com as chaves resposta (string), confianca (0.0–1.0) e "
    "contexto_insuficiente (boolean)."
)

_RESPONDER_INVALID_ANSWER_PT = (
    "Não foi possível processar a resposta do modelo (formato estruturado inválido)."
)


def _tpl(name: str) -> str:
    return load_prompt_text(name)


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c.text}")
    return "\n\n".join(parts)


def extract_answer_line(completion: str) -> str:
    """Legado: extrai ``RESPOSTA:`` quando presente (fallback de migração)."""
    text = completion.strip()
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("RESPOSTA:"):
            return stripped[len("RESPOSTA:") :].strip()
    return text


def _responder_meta_from_validated(
    validated: ResponderResponseValidated,
    parse_meta: dict[str, Any],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schema_version": RESPONDER_SCHEMA_VERSION,
        "confianca": validated.confianca,
        "contexto_insuficiente": validated.contexto_insuficiente,
    }
    meta.update(parse_meta)
    return meta


def generate_answer(
    llm: LlmClient,
    item: EvalItem,
    retrieved: list[RetrievedChunk],
    *,
    rag_enabled: bool = True,
    prompt_style: PromptStyle = "rag_pt",
    force_specific_answer: bool = False,
    max_parse_retries: int = 1,
) -> tuple[str, dict[str, Any]]:
    """Gera resposta RAG com saída JSON validada; devolve texto e meta de qualidade."""
    del prompt_style  # único estilo suportado (FairytaleQA pt-BR)
    if not rag_enabled or not retrieved:
        ctx = "(nenhum contexto recuperado)"
        hints = "Sem chunks recuperados."
    else:
        ctx = format_context(retrieved)
        hints = format_retrieval_hints(retrieved)
    user = _tpl("responder_user_template.txt").format(
        context=ctx,
        retrieval_hints=hints,
        question=item.question,
    )
    if force_specific_answer:
        user = (
            user
            + "\n\nReforço: o contexto já traz evidência suficiente. "
            + "EVITE RECUSA GENÉRICA. Responda em 1 frase curta, factual e "
            + "com palavras do contexto quando possível."
        )
    system = _tpl("responder_system.txt")

    try:
        validated, _raw, parse_meta = call_validate_with_retries(
            client=llm,
            system=system,
            user=user,
            validate=lambda p: validate_responder_response(p, log_violations=True),
            max_parse_retries=max_parse_retries,
            retry_suffix=_RESPONDER_RETRY_SUFFIX,
        )
        meta = _responder_meta_from_validated(validated, structured_meta_as_dict(parse_meta))
        return validated.resposta, meta
    except (StructuredOutputError, ResponderSchemaError) as exc:
        error_meta: dict[str, Any] = {
            "schema_version": RESPONDER_SCHEMA_VERSION,
            "schema_invalid": True,
            "structured_output_error": str(exc),
            "confianca": None,
            "contexto_insuficiente": None,
        }
        return _RESPONDER_INVALID_ANSWER_PT, error_meta

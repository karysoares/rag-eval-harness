"""Orquestração multi-agente: respondedor + passo de crítica.

A crítica é sinal diagnóstico em ``meta``; não entra na agregação de ``flag_anomalia``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_evaluation.config import AppConfig
from llm_evaluation.critic_schema import (
    CRITIC_SCHEMA_VERSION,
    CriticSchemaError,
    critic_to_dict,
    validate_critic_response,
)
from llm_evaluation.llm_client import LlmClient
from llm_evaluation.pipeline import run_batch
from llm_evaluation.prompt_resources import load_prompt_text
from llm_evaluation.structured_output import (
    StructuredOutputError,
    call_validate_with_retries,
    structured_meta_as_dict,
)
from llm_evaluation.types import EvalItem, RetrievedChunk, RunRecord

_CRITIC_RETRY_SUFFIX = (
    "\n\nIMPORTANTE: retorne APENAS um objeto JSON válido, sem markdown, "
    "com cadeia_de_pensamento (lista de strings), problemas (lista) e nota (string)."
)


def _critic(
    llm: LlmClient,
    question: str,
    context: str,
    answer: str,
    *,
    max_parse_retries: int = 1,
) -> dict[str, Any]:
    system = load_prompt_text("critic_system.txt")
    user = f"Pergunta:\n{question}\n\nContexto:\n{context}\n\nResposta:\n{answer}\n"
    try:
        validated, _raw, parse_meta = call_validate_with_retries(
            client=llm,
            system=system,
            user=user,
            validate=lambda p: validate_critic_response(p, log_violations=True),
            max_parse_retries=max_parse_retries,
            retry_suffix=_CRITIC_RETRY_SUFFIX,
        )
        out = critic_to_dict(validated)
        out.update(structured_meta_as_dict(parse_meta))
        return out
    except (StructuredOutputError, CriticSchemaError) as exc:
        return {
            "schema_version": CRITIC_SCHEMA_VERSION,
            "schema_invalid": True,
            "structured_output_error": str(exc),
            "cadeia_de_pensamento": [],
            "problemas": [],
            "nota": "Crítica indisponível: saída estruturada inválida.",
        }


def _critic_hook(
    item: EvalItem,
    retrieved: list[RetrievedChunk],
    answer: str,
    llm: LlmClient,
) -> tuple[dict[str, Any], bool]:
    ctx = "\n\n".join(f"[{j + 1}] {c.text}" for j, c in enumerate(retrieved))
    critic_raw = _critic(llm, item.question, ctx, answer)
    if critic_raw.get("schema_invalid"):
        return critic_raw, False
    issues = critic_raw.get("problemas") or []
    critic_flag = isinstance(issues, list) and any(
        str(x).lower() not in ("nenhum", "none") for x in issues if x is not None
    )
    return critic_raw, critic_flag


def run_items(
    cfg: AppConfig,
    items: list[EvalItem],
    *,
    on_record: Callable[[RunRecord], None] | None = None,
    run_dir: Path | None = None,
    config_name: str = "",
) -> list[RunRecord]:
    return run_batch(
        cfg,
        items,
        on_record=on_record,
        critic_hook=_critic_hook,
        run_dir=run_dir,
        config_name=config_name,
    )

"""LLM como juiz RAG (prompts em ``src/llm_evaluation/prompts/``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_evaluation.config import JudgePromptStyle
from llm_evaluation.llm_client import (
    LlmClient,
    OpenAiCompatibleClient,
    heuristic_judge_json,
    parse_judge_json,
)
from llm_evaluation.prompt_resources import load_prompt_text
from llm_evaluation.types import JudgeResult, RetrievedChunk
from llm_evaluation.verification.judge_context import JudgeContextBuilt, build_judge_context
from llm_evaluation.verification.judge_schema import (
    JUDGE_SCHEMA_VERSION,
    JudgeSchemaError,
    validate_judge_response,
)

_JUDGE_RETRY_SUFFIX = (
    "\n\nIMPORTANTE: responda APENAS com um objeto JSON válido, sem markdown, "
    "com as chaves veredito, motivo_breve, confianca (0.0–1.0)."
)


@dataclass(frozen=True)
class JudgeRunMeta:
    """Metadados de auditoria por item (gravados em ``meta.contexto_juiz``)."""

    retry_count: int
    parse_failures: int
    schema_invalid: bool
    used_fallback: bool
    schema_version: str
    contexto: dict[str, Any]


def load_prompt(name: str) -> str:
    return load_prompt_text(name)


def _prompt_files(style: JudgePromptStyle) -> tuple[str, str]:
    if style == "generic":
        return "judge_generic_system.txt", "judge_generic_user_template.txt"
    if style == "rag_pt":
        return "judge_rag_pt_system.txt", "judge_rag_pt_user_template.txt"
    return "judge_system.txt", "judge_user_template.txt"


def _complete_judge(
    client: LlmClient,
    system: str,
    user: str,
    *,
    json_object: bool,
) -> str:
    if isinstance(client, OpenAiCompatibleClient):
        return client.complete(system, user, json_object=json_object)
    complete = getattr(client, "complete", None)
    if callable(complete):
        try:
            return str(complete(system, user, json_object=json_object))
        except TypeError:
            return str(complete(system, user))
    msg = "Cliente juiz sem método complete"
    raise TypeError(msg)


def _call_and_parse(
    client: LlmClient,
    system: str,
    user: str,
    *,
    json_object: bool,
) -> dict[str, Any]:
    text = _complete_judge(client, system, user, json_object=json_object)
    return parse_judge_json(text)


def _raw_from_validated(
    validated: Any,
    *,
    include_cot: bool,
    cot: list[Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "veredito": validated.veredito,
        "motivo_breve": validated.motivo_breve,
        "confianca": validated.confianca,
        "schema_version": JUDGE_SCHEMA_VERSION,
    }
    if validated.fallback_heuristico:
        raw["fallback_heuristico"] = True
    if include_cot and cot:
        raw["cadeia_de_pensamento"] = cot
    raw.update(extra)
    return raw


def run_judge(
    *,
    question: str,
    context: str,
    answer: str,
    client: LlmClient,
    prompt_style: JudgePromptStyle = "pt",
    return_chain_of_thought: bool = False,
    max_parse_retries: int = 2,
    context_built: JudgeContextBuilt | None = None,
    retrieval_meta: str = "(não fornecido)",
) -> tuple[JudgeResult, JudgeRunMeta]:
    sys_name, user_name = _prompt_files(prompt_style)
    system = load_prompt(sys_name)
    user_tpl = load_prompt(user_name)
    ctx_text = context_built.text if context_built is not None else context
    fmt_args = {
        "question": question,
        "context": ctx_text,
        "answer": answer,
        "retrieval_meta": retrieval_meta,
    }
    try:
        user = user_tpl.format(**fmt_args)
    except KeyError:
        user = user_tpl.format(
            question=question,
            context=ctx_text,
            answer=answer,
        )

    retry_count = 0
    parse_failures = 0
    had_schema_failure = False
    raw_in: dict[str, Any] | None = None
    used_fallback = False

    attempts = max(1, max_parse_retries + 1)
    for attempt in range(attempts):
        use_json_mode = attempt > 0
        sys_eff = system + (_JUDGE_RETRY_SUFFIX if attempt > 0 else "")
        try:
            raw_in = _call_and_parse(
                client,
                sys_eff,
                user,
                json_object=use_json_mode,
            )
            validated = validate_judge_response(raw_in)
            cot = _cadeia(raw_in) if return_chain_of_thought else []
            raw_out = _raw_from_validated(
                validated,
                include_cot=return_chain_of_thought,
                cot=cot,
                extra={k: v for k, v in raw_in.items() if k not in _SKIP_KEYS},
            )
            meta = _build_run_meta(
                retry_count=retry_count,
                parse_failures=parse_failures,
                schema_invalid=had_schema_failure,
                used_fallback=False,
                context_built=context_built,
            )
            return (
                JudgeResult(
                    veredito=validated.veredito,
                    motivo_breve=validated.motivo_breve,
                    confianca=validated.confianca,
                    raw=raw_out,
                ),
                meta,
            )
        except (JudgeSchemaError, json.JSONDecodeError, TypeError, ValueError):
            parse_failures += 1
            had_schema_failure = True
            if attempt < attempts - 1:
                retry_count += 1
                continue
        except Exception:  # noqa: BLE001 — API/timeout após retries HTTP do cliente
            if attempt < attempts - 1:
                retry_count += 1
                continue
            break

    raw_in = heuristic_judge_json(answer, ctx_text)
    used_fallback = True
    try:
        validated = validate_judge_response(raw_in)
    except JudgeSchemaError:
        validated = validate_judge_response(
            {
                "veredito": raw_in.get("veredito", "sustentado"),
                "motivo_breve": str(raw_in.get("motivo_breve") or "fallback"),
                "confianca": float(raw_in.get("confianca", 0.4)),
                "fallback_heuristico": True,
            },
            log_violations=False,
        )
    raw_out = _raw_from_validated(
        validated,
        include_cot=return_chain_of_thought,
        cot=_cadeia(raw_in) if return_chain_of_thought else [],
        extra={},
    )
    if used_fallback:
        raw_out["fallback_heuristico"] = True
    meta = _build_run_meta(
        retry_count=retry_count,
        parse_failures=parse_failures,
        schema_invalid=had_schema_failure,
        used_fallback=used_fallback,
        context_built=context_built,
    )
    return (
        JudgeResult(
            veredito=validated.veredito,
            motivo_breve=validated.motivo_breve,
            confianca=validated.confianca,
            raw=raw_out,
        ),
        meta,
    )


_SKIP_KEYS = frozenset(
    {
        "cadeia_de_pensamento",
        "chain_of_thought",
        "veredito",
        "verdict",
        "motivo_breve",
        "reason_short",
        "confianca",
        "confidence",
        "fallback_heuristico",
    },
)


def _cadeia(raw: dict[str, Any]) -> list[Any]:
    v = raw.get("cadeia_de_pensamento")
    if v is None:
        v = raw.get("chain_of_thought") or []
    return v if isinstance(v, list) else []


def judge_run_meta_as_context(j_run: JudgeRunMeta) -> dict[str, Any]:
    """Serializa metadados do juiz para ``meta.contexto_juiz``."""
    out: dict[str, Any] = {
        "retry_count": j_run.retry_count,
        "parse_failures": j_run.parse_failures,
        "schema_invalid": j_run.schema_invalid,
        "used_fallback": j_run.used_fallback,
        "schema_version": j_run.schema_version,
    }
    if j_run.schema_invalid or j_run.used_fallback:
        out["structured_output_error"] = (
            "fallback_heuristico" if j_run.used_fallback else "schema_invalid"
        )
    out.update(j_run.contexto)
    return out


def _build_run_meta(
    *,
    retry_count: int,
    parse_failures: int,
    schema_invalid: bool,
    used_fallback: bool,
    context_built: JudgeContextBuilt | None,
) -> JudgeRunMeta:
    ctx: dict[str, Any] = {}
    if context_built is not None:
        ctx = {
            "chunk_ids": context_built.chunk_ids,
            "n_chunks_usados": context_built.n_chunks_usados,
            "n_chunks_total": context_built.n_chunks_total,
            "tokens_estimados": context_built.tokens_estimados,
            "truncado": context_built.truncado,
        }
    return JudgeRunMeta(
        retry_count=retry_count,
        parse_failures=parse_failures,
        schema_invalid=schema_invalid,
        used_fallback=used_fallback,
        schema_version=JUDGE_SCHEMA_VERSION,
        contexto=ctx,
    )


def run_judge_for_retrieved(
    *,
    question: str,
    answer: str,
    retrieved: list[RetrievedChunk],
    client: LlmClient,
    prompt_style: JudgePromptStyle = "pt",
    return_chain_of_thought: bool = False,
    max_parse_retries: int = 2,
    max_chunks: int,
    max_context_chars: int | None = None,
    retrieval_meta: str = "(não fornecido)",
) -> tuple[JudgeResult, JudgeRunMeta]:
    """Atalho: monta contexto a partir dos chunks recuperados e chama ``run_judge``."""
    ctx = build_judge_context(
        retrieved,
        max_chunks=max_chunks,
        max_chars=max_context_chars,
    )
    return run_judge(
        question=question,
        context=ctx.text,
        answer=answer,
        client=client,
        prompt_style=prompt_style,
        return_chain_of_thought=return_chain_of_thought,
        max_parse_retries=max_parse_retries,
        context_built=ctx,
        retrieval_meta=retrieval_meta,
    )

"""Geração RAG: formatação de contexto e chamada ao LLM mockado."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from llm_evaluation.generation import format_context, generate_answer
from llm_evaluation.responder_schema import ResponderSchemaError, validate_responder_response
from llm_evaluation.retrieval_hints import format_retrieval_hints
from llm_evaluation.types import EvalItem, RetrievedChunk

if TYPE_CHECKING:
    from llm_evaluation.llm_client import LlmClient


def _responder_json(
    resposta: str,
    *,
    confianca: float = 0.9,
    contexto_insuficiente: bool = False,
) -> str:
    return json.dumps(
        {
            "resposta": resposta,
            "confianca": confianca,
            "contexto_insuficiente": contexto_insuficiente,
        }
    )


class _CapturingLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_system = ""
        self.last_user = ""
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def complete(self, system: str, user: str, **kwargs: object) -> str:
        self.last_system = system
        self.last_user = user
        self.calls.append((system, user, kwargs))
        return self.reply


def test_format_retrieval_hints_gold_high_score() -> None:
    chunks = [
        RetrievedChunk(text="O cão levou o anel.", score=0.68, is_gold=True),
        RetrievedChunk(text="outro", score=0.2, is_gold=False),
    ]
    hints = format_retrieval_hints(chunks)
    assert "ouro_no_top_k=sim" in hints
    assert "0.68" in hints
    assert "extraia" in hints.lower()


def test_format_context_numbered_chunks() -> None:
    chunks = [
        RetrievedChunk(text="chunk A", score=0.9),
        RetrievedChunk(text="chunk B", score=0.5),
    ]
    ctx = format_context(chunks)
    assert "[1] chunk A" in ctx
    assert "[2] chunk B" in ctx


def test_generate_answer_sem_contexto_usa_template_pt() -> None:
    item = EvalItem(
        id="1",
        question="Qual a capital?",
        correct_answers=["Brasília"],
        incorrect_answers=[],
    )
    llm: LlmClient = _CapturingLlm(_responder_json("Brasília"))
    answer, meta = generate_answer(llm, item, [], rag_enabled=False)
    assert answer == "Brasília"
    assert meta["confianca"] == 0.9
    assert meta["contexto_insuficiente"] is False
    assert not meta.get("schema_invalid")
    assert "nenhum contexto" in llm.last_user  # type: ignore[attr-defined]


def test_generate_answer_rag_pt_parses_json() -> None:
    item = EvalItem(
        id="2",
        question="Capital?",
        correct_answers=["Brasília"],
        incorrect_answers=[],
    )
    llm: LlmClient = _CapturingLlm(_responder_json("Brasília", confianca=0.85))
    chunks = [RetrievedChunk(text="A capital é Brasília.", score=0.8, is_gold=True)]
    answer, meta = generate_answer(llm, item, chunks, rag_enabled=True)
    assert answer == "Brasília"
    assert meta["confianca"] == 0.85
    assert "Brasília" in llm.last_user  # type: ignore[attr-defined]
    assert "ouro_no_top_k=sim" in llm.last_user  # type: ignore[attr-defined]


def test_generate_answer_retries_on_invalid_json() -> None:
    item = EvalItem(
        id="3",
        question="Q?",
        correct_answers=["x"],
        incorrect_answers=[],
    )
    llm = _CapturingLlm("not json")

    def _complete(system: str, user: str, **kwargs: object) -> str:
        llm.calls.append((system, user, kwargs))
        if len(llm.calls) == 1:
            return "not json"
        return _responder_json("ok")

    llm.complete = _complete  # type: ignore[method-assign]
    answer, meta = generate_answer(llm, item, [], rag_enabled=False, max_parse_retries=1)
    assert answer == "ok"
    assert meta.get("retry_count", 0) >= 1


def test_generate_answer_records_structured_output_error() -> None:
    item = EvalItem(
        id="4",
        question="Q?",
        correct_answers=["x"],
        incorrect_answers=[],
    )
    llm: LlmClient = _CapturingLlm("still not json")
    answer, meta = generate_answer(llm, item, [], rag_enabled=False, max_parse_retries=0)
    assert "formato estruturado inválido" in answer
    assert meta.get("schema_invalid") is True
    assert meta.get("structured_output_error")
    assert meta.get("confianca") is None


def test_validate_responder_rejects_invalid() -> None:
    with pytest.raises(ResponderSchemaError, match="confianca"):
        validate_responder_response(
            {"resposta": "x", "confianca": 2.0, "contexto_insuficiente": False},
            log_violations=False,
        )

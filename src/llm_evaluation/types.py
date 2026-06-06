from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["sustentado", "nao_sustentado", "contradicacao", "incompleto", "inseguro"]


@dataclass
class EvalItem:
    """Unidade de avaliação: pergunta, referências opcionais e contexto RAG de suporte."""

    id: str
    question: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str = ""
    rag_gold_chunk: str | None = None
    rag_distractors: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    is_gold: bool = False


@dataclass
class JudgeResult:
    veredito: Verdict
    motivo_breve: str
    confianca: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationSignals:
    gold_correct: bool | None
    gold_incorrect: bool | None
    is_refusal: bool
    embedding_max_cosine: float | None
    embedding_low_support: bool | None
    embedding_max_cosine_retrieved: float | None = None
    embedding_max_cosine_gold: float | None = None
    judge: JudgeResult | None = None
    judge_negative: bool | None = None


@dataclass
class RunRecord:
    item_id: str
    question: str
    answer: str
    gold_correct: bool | None
    anomaly_flag: bool
    signals: VerificationSignals
    retrieved: list[RetrievedChunk]
    baseline_profile: str
    meta: dict[str, Any] = field(default_factory=dict)

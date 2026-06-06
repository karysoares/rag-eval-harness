"""Amostra curada offline em pt-BR (smoke / CI sem Hub)."""

from __future__ import annotations

from llm_evaluation.types import EvalItem


def amostra_local_items() -> list[EvalItem]:
    """Dois itens estilo FairytaleQA: pergunta, resposta curta e passagem de contexto."""
    return [
        EvalItem(
            id="amostra-1",
            question="Qual é a capital do Brasil?",
            correct_answers=["Brasília"],
            incorrect_answers=[],
            category="geografia",
            rag_gold_chunk=(
                "O Brasil é uma república federativa na América do Sul. "
                "Sua capital federal é Brasília, no Distrito Federal."
            ),
            rag_distractors=[
                "São Paulo é a cidade mais populosa, mas não é a capital.",
            ],
        ),
        EvalItem(
            id="amostra-2",
            question="O que acontece se você engolir sementes de melancia?",
            correct_answers=[
                "Nada de especial; o corpo digere ou elimina as sementes",
                "As sementes passam pelo trato digestivo",
            ],
            incorrect_answers=[],
            category="saude",
            rag_gold_chunk=(
                "Sementes de melancia são comestíveis em pequenas quantidades e "
                "não germinam no estômago humano."
            ),
            rag_distractors=["Mito popular: qualquer semente engolida vira planta no corpo."],
        ),
    ]

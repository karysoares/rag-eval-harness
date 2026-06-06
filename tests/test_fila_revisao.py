"""Fila de revisão humana e limiares operacionais."""

from __future__ import annotations

from llm_evaluation.fila_revisao import (
    export_fila_csv,
    is_recusa_for_fila,
    select_fila_records,
)
from llm_evaluation.types import JudgeResult, RetrievedChunk, RunRecord, VerificationSignals


def _rec(
    *,
    iid: str,
    answer: str,
    veredito: str | None = None,
    score: float = 0.7,
    is_refusal: bool = False,
) -> RunRecord:
    judge = None
    if veredito:
        judge = JudgeResult(veredito=veredito, motivo_breve="t", confianca=0.9)
    return RunRecord(
        item_id=iid,
        question="q",
        answer=answer,
        gold_correct=True,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=is_refusal,
            embedding_max_cosine=0.5,
            embedding_low_support=False,
            judge=judge,
            judge_negative=None,
        ),
        retrieved=[RetrievedChunk(text="ctx", score=score, is_gold=True)],
        baseline_profile="hibrido",
        meta={
            "metricas_recuperacao": {
                "chunk_ouro_no_top_k": True,
                "score_melhor_chunk": score,
            },
        },
    )


def test_is_recusa_prefers_signal_over_phrase() -> None:
    assert is_recusa_for_fila(_rec(iid="a", answer="Paris.", is_refusal=True)) is True
    assert (
        is_recusa_for_fila(
            _rec(iid="b", answer="Não há informações suficientes no texto."),
        )
        is True
    )
    assert is_recusa_for_fila(_rec(iid="c", answer="um alfinete")) is False


def test_select_uses_juiz_vereditos_from_config() -> None:
    vereditos = ["nao_sustentado", "contradicacao"]
    fila = select_fila_records(
        [_rec(iid="x", answer="y", veredito="incompleto")],
        juiz_vereditos_fila=vereditos,
    )
    assert fila == []
    fila2 = select_fila_records(
        [_rec(iid="x", answer="y", veredito="nao_sustentado")],
        juiz_vereditos_fila=vereditos,
    )
    assert fila2[0][0] == "juiz_veredito_duro"


def test_export_fixture(tmp_path) -> None:
    records = [
        _rec(iid="j1", answer="err", veredito="nao_sustentado"),
        _rec(
            iid="r1",
            answer="Não há informações suficientes.",
            score=0.6,
            is_refusal=True,
        ),
    ]
    path, counts = export_fila_csv(
        tmp_path,
        records,
        juiz_vereditos_fila=["nao_sustentado", "contradicacao", "inseguro"],
    )
    assert path.is_file()
    assert counts["total"] == 2
    assert (tmp_path / "analise_manual" / "fila_revisao_humana.json").is_file()

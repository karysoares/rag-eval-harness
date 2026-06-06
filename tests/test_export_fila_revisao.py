"""Fila de revisão humana (llm_evaluation.fila_revisao)."""

from __future__ import annotations

from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.fila_revisao import export_fila_csv, select_fila_records
from llm_evaluation.types import JudgeResult, RetrievedChunk, RunRecord, VerificationSignals


def _rec(
    *,
    iid: str,
    answer: str,
    veredito: str | None = None,
    score: float = 0.7,
    gold_top: bool = True,
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
            is_refusal=False,
            embedding_max_cosine=0.5,
            embedding_low_support=False,
            judge=judge,
            judge_negative=None,
        ),
        retrieved=[RetrievedChunk(text="contexto com resposta", score=score, is_gold=True)],
        baseline_profile="hibrido",
        meta={
            "metricas_recuperacao": {
                "chunk_ouro_no_top_k": gold_top,
                "score_melhor_chunk": score,
            },
            "metricas_lexicas": {"f1_token": 0.0},
        },
    )


def test_select_prioriza_juiz_duro_sobre_recusa() -> None:
    recusa = _rec(iid="a", answer="Não há informações suficientes.")
    duro = _rec(iid="a", answer="errado", veredito="nao_sustentado")
    vereditos = ["nao_sustentado", "contradicacao", "inseguro"]
    fila = select_fila_records([recusa, duro], juiz_vereditos_fila=vereditos)
    assert len(fila) == 1
    assert fila[0][0] == "juiz_veredito_duro"


def test_select_recusa_com_score_alto() -> None:
    fila = select_fila_records(
        [_rec(iid="b", answer="Não há informações suficientes.", score=0.6)],
        juiz_vereditos_fila=["nao_sustentado"],
    )
    assert len(fila) == 1
    assert fila[0][0] == "recusa_com_contexto_forte"


def test_select_ignora_recusa_com_score_baixo() -> None:
    fila = select_fila_records(
        [_rec(iid="c", answer="Não há informações.", score=0.3)],
        juiz_vereditos_fila=["nao_sustentado"],
        min_score_recuperacao=0.5,
    )
    assert fila == []


def test_export_on_fixture_run(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "outputs" / "run_20260517T140349Z"
    if not (fixture / "predictions.jsonl").is_file():
        return
    records = load_records_from_predictions_jsonl(fixture / "predictions.jsonl")
    path, counts = export_fila_csv(
        fixture,
        records,
        juiz_vereditos_fila=["nao_sustentado", "contradicacao", "inseguro"],
    )
    assert path.is_file()
    assert counts["juiz_veredito_duro"] == 5
    assert counts["recusa_com_contexto_forte"] >= 20
    assert counts["total"] == counts["juiz_veredito_duro"] + counts["recusa_com_contexto_forte"]

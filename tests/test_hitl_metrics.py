"""Métricas HITL (plano C)."""

from __future__ import annotations

from llm_evaluation.hitl_metrics import summarize_hitl
from llm_evaluation.types import RunRecord, VerificationSignals


def _rec(iid: str, rotulo: str | None, *, flag: bool = False) -> RunRecord:
    meta: dict = {}
    if rotulo:
        meta["adjudicacao_humana"] = {"rotulo": rotulo, "revisor": "t"}
    return RunRecord(
        item_id=iid,
        question="q",
        answer="a",
        gold_correct=True,
        anomaly_flag=flag,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta,
    )


def _amostra_equilibrada(n_maus: int = 6, n_bons: int = 6) -> list[RunRecord]:
    """Amostra acima do mínimo e com as duas classes na verdade humana."""
    maus = [_rec(f"m{i}", "incorreto", flag=i % 2 == 0) for i in range(n_maus)]
    bons = [_rec(f"b{i}", "correto", flag=i % 3 == 0) for i in range(n_bons)]
    return maus + bons


def test_summarize_hitl_confusao_detector() -> None:
    records = [*_amostra_equilibrada(), _rec("sem-rotulo", None)]
    out = summarize_hitl(records, fila_total=13)
    assert out is not None
    assert out["n_itens_rotulados"] == 12
    det = out.get("detector_vs_humano")
    assert isinstance(det, dict)
    assert det.get("confusao") is not None
    assert isinstance(det.get("kappa"), float)


class TestGuardasContraNumeroDegenerado:
    """Um número plausível vindo de uma amostra impossível é pior do que nenhum."""

    def test_abaixo_do_minimo_nao_publica_confusao_nem_kappa(self) -> None:
        records = [_rec("1", "incorreto", flag=True), _rec("2", "correto")]
        out = summarize_hitl(records, fila_total=2)
        assert out is not None
        assert out["n_itens_rotulados"] == 2
        assert "detector_vs_humano" not in out
        assert "abaixo do mínimo" in str(out["metricas_omitidas"])

    def test_distribuicao_de_rotulos_e_sempre_publicada(self) -> None:
        records = [*_amostra_equilibrada(n_maus=4, n_bons=8)]
        out = summarize_hitl(records)
        assert out is not None
        assert out["distribuicao_rotulos"] == {"correto": 8, "incorreto": 4}

    def test_verdade_humana_de_uma_so_classe_recusa_kappa(self) -> None:
        # É o caso do fixture commitado: 6 rótulos, todos "correto". `cohen_kappa`
        # devolveria 0.0, que se lê como «o detector não concorda» quando o que
        # falta é uma classe com que concordar.
        records = [_rec(f"c{i}", "correto", flag=i % 2 == 0) for i in range(12)]
        out = summarize_hitl(records)
        assert out is not None
        det = out["detector_vs_humano"]
        assert isinstance(det, dict)
        assert det["kappa"] is None
        assert det["kappa_indefinido"] == "sem classe positiva na verdade humana"
        assert out["distribuicao_rotulos"] == {"correto": 12}

    def test_sem_classe_negativa_tambem_recusa(self) -> None:
        records = [_rec(f"i{i}", "incorreto", flag=i % 2 == 0) for i in range(12)]
        out = summarize_hitl(records)
        assert out is not None
        det = out["detector_vs_humano"]
        assert isinstance(det, dict)
        assert det["kappa"] is None
        assert det["kappa_indefinido"] == "sem classe negativa na verdade humana"

    def test_inconclusivo_nao_entra_na_confusao_mas_conta_na_distribuicao(self) -> None:
        records = [*_amostra_equilibrada(), *[_rec(f"x{i}", "inconclusivo") for i in range(3)]]
        out = summarize_hitl(records)
        assert out is not None
        assert out["n_itens_rotulados"] == 15
        assert out["distribuicao_rotulos"]["inconclusivo"] == 3
        det = out["detector_vs_humano"]
        assert isinstance(det, dict)
        conf = det["confusao"]
        assert isinstance(conf, dict)
        assert sum(conf.values()) == 12

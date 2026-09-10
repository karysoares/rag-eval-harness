"""Comparação de rubricas: a ablação que justifica um estilo de prompt novo.

Trocar a rubrica é trocar de instrumento. O script mede-o com **uma variável**: mesmo
modelo de juiz, mesmas respostas, mesmo contexto — só o par de ficheiros de prompt muda.
Estes testes cobrem as exclusões, porque é aí que uma comparação de instrumentos se
corrompe em silêncio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_prompt_styles import _elegiveis, _veredito_gravado  # noqa: E402
from llm_evaluation.types import JudgeResult, RetrievedChunk, RunRecord, VerificationSignals


def _registo(
    item_id: str,
    *,
    veredito: str | None = "sustentado",
    fallback: bool = False,
    erro: bool = False,
    com_chunks: bool = True,
) -> RunRecord:
    juiz = (
        JudgeResult(
            veredito=veredito,
            motivo_breve="m",
            confianca=0.9,
            raw={"used_fallback": True} if fallback else {},
        )
        if veredito is not None
        else None
    )
    meta: dict[str, object] = {}
    if erro:
        meta["processing_error"] = {"type": "PermanentApiError", "message": "429"}
    return RunRecord(
        item_id=item_id,
        question="p?",
        answer="r",
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=0.8,
            embedding_low_support=False,
            judge=juiz,
        ),
        retrieved=[RetrievedChunk(text="t", score=0.9, is_gold=True)] if com_chunks else [],
        baseline_profile="hibrido",
        meta=meta,
    )


def test_veredito_gravado_e_o_braco_de_referencia() -> None:
    """Metade das chamadas poupadas: a referência já está no artefacto."""
    assert _veredito_gravado(_registo("a", veredito="nao_sustentado")) == "nao_sustentado"


def test_item_sem_juiz_nao_entra() -> None:
    assert _veredito_gravado(_registo("a", veredito=None)) is None


def test_fallback_heuristico_nao_entra() -> None:
    """O fallback responde `sustentado` por omissão.

    Compará-lo com uma rubrica mediria o fallback e não a rubrica — é a regra 4 do
    CLAUDE.md aplicada a uma comparação de instrumentos.
    """
    assert _veredito_gravado(_registo("a", fallback=True)) is None


def test_falha_de_execucao_nao_entra() -> None:
    assert _elegiveis([_registo("a", erro=True)]) == []


def test_item_sem_chunks_nao_entra() -> None:
    """Sem contexto recuperado não há ancoragem para julgar; a rubrica não é o que varia."""
    assert _elegiveis([_registo("a", com_chunks=False)]) == []


def test_elegiveis_mantem_a_ordem_e_filtra_o_resto() -> None:
    registos = [
        _registo("a"),
        _registo("b", erro=True),
        _registo("c", fallback=True),
        _registo("d", veredito="incompleto"),
        _registo("e", veredito=None),
    ]
    assert [r.item_id for r in _elegiveis(registos)] == ["a", "d"]


def test_polaridade_de_aprovacao_e_a_do_harness() -> None:
    """Se isto divergir da agregação, o κ publicado mede outra coisa."""
    from compare_prompt_styles import replay_aprovacao

    assert replay_aprovacao() == ("sustentado",)


def test_estilos_aceitos_cobrem_os_do_config() -> None:
    """Um estilo novo no config sem entrada aqui torna-o não comparável."""
    from compare_prompt_styles import ESTILOS
    from llm_evaluation.config import JudgePromptStyle

    assert set(ESTILOS) == set(JudgePromptStyle.__args__)


@pytest.mark.parametrize("estilo", ["generic_pt", "generic", "rag_pt", "pt"])
def test_cada_estilo_resolve_para_ficheiros_existentes(estilo: str) -> None:
    from llm_evaluation.prompt_resources import load_prompt_text
    from llm_evaluation.verification.judge import _prompt_files

    for nome in _prompt_files(estilo):
        assert load_prompt_text(nome).strip()

"""Regressões das lacunas metodológicas corrigidas em conjunto.

Cada teste aqui falha contra o código anterior à correcção. Estão no mesmo
ficheiro porque partilham a mesma natureza: não são bugs de implementação — o
código fazia o que dizia — são medições que respondiam a uma pergunta diferente
da que o artefacto anunciava.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llm_evaluation.datasets_rag import build_corpus_for_item, corpus_tem_distratores
from llm_evaluation.judge_meta import build_judge_meta_report
from llm_evaluation.reporting import summarize
from llm_evaluation.retrieval import HashEmbedder, Retriever
from llm_evaluation.retrieval_metrics import (
    NOTA_CORPUS_SEM_DISTRATORES,
    compute_retrieval_metrics,
)
from llm_evaluation.types import (
    EvalItem,
    JudgeResult,
    RetrievedChunk,
    RunRecord,
    VerificationSignals,
)

REPO = Path(__file__).resolve().parents[1]


def _sinais(**kwargs: Any) -> VerificationSignals:
    base: dict[str, Any] = {
        "gold_correct": None,
        "gold_incorrect": None,
        "is_refusal": False,
        "embedding_max_cosine": None,
        "embedding_low_support": None,
        "judge": None,
        "judge_negative": None,
    }
    base.update(kwargs)
    return VerificationSignals(**base)


def _registo(item_id: str, *, anomalia: bool, meta: dict[str, Any] | None = None) -> RunRecord:
    return RunRecord(
        item_id=item_id,
        question="q",
        answer="a",
        gold_correct=None,
        anomaly_flag=anomalia,
        signals=_sinais(),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta if meta is not None else {},
    )


# --------------------------------------------------------------------------
# Regra 3 do CLAUDE.md: falha de execução não é anomalia do sistema.
# --------------------------------------------------------------------------


def test_summary_exclui_itens_com_erro_execucao_do_denominador() -> None:
    """7 itens avaliados, 2 marcados; os 3 falhados não inflacionam a taxa.

    Antes, ``_failed_record`` marcava ``flag_anomalia=True`` e o sumário dividia
    por 10 — publicando 0,5 onde a taxa real era 0,2857.
    """
    avaliados = [_registo(f"ok-{i}", anomalia=i < 2) for i in range(7)]
    falhados = [
        _registo(
            f"erro-{i}",
            anomalia=True,
            meta={"processing_error": {"type": "HTTPStatusError", "message": "429"}},
        )
        for i in range(3)
    ]

    out = summarize([*avaliados, *falhados], reference_type="lexical")

    assert out["n_itens"] == 10, "n_itens continua a ser o total recebido"
    assert out["n_itens_avaliados"] == 7
    assert out["n_itens_com_erro_execucao"] == 3
    assert out["n_anomalias_marcadas"] == 2, "os 3 falhados não contam como anomalia"
    assert out["taxa_alerta"] == 2 / 7
    assert "processing_error" in str(out["nota_exclusao"])


def test_summary_sem_falhas_mantem_denominador_total() -> None:
    out = summarize([_registo(f"ok-{i}", anomalia=i == 0) for i in range(4)], reference_type="none")
    assert out["n_itens"] == out["n_itens_avaliados"] == 4
    assert out["n_itens_com_erro_execucao"] == 0
    assert out["taxa_alerta"] == 0.25


# --------------------------------------------------------------------------
# Regra 8 do CLAUDE.md: planos métricos não se misturam.
# --------------------------------------------------------------------------


def test_grounding_nao_usa_passagem_ouro_nao_recuperada() -> None:
    """Resposta copiada do ouro, mas nada recuperado ⇒ suporte baixo.

    Antes, ``emb_max`` era o **máximo** entre a similaridade aos chunks
    recuperados e a similaridade à passagem ouro. Uma resposta que copiasse a
    referência saía bem ancorada mesmo sem contexto nenhum: o plano de grounding
    a responder à pergunta do plano de referência.
    """
    from dataclasses import replace

    from llm_evaluation.config import load_config
    from llm_evaluation.pipeline import verify_item

    cfg = load_config(REPO / "configs/default.yaml")
    assert cfg.verification.embedding_use_gold_chunk, "config de referência mantém o diagnóstico"
    # Juiz desligado para isolar a camada de embedding (não há cliente no teste).
    cfg = replace(cfg, verification=replace(cfg.verification, verify_judge=False))

    ouro = "A rainha entregou o anel ao viajante antes do amanhecer."
    item = EvalItem(
        id="g1",
        question="Quem entregou o anel?",
        correct_answers=["a rainha"],
        incorrect_answers=[],
        rag_gold_chunk=ouro,
    )

    sinais, _ = verify_item(
        cfg=cfg,
        item=item,
        answer=ouro,  # resposta idêntica à passagem ouro…
        retrieved=[],  # …que não foi recuperada
        embedder=HashEmbedder(),
        judge_client=None,  # type: ignore[arg-type] - verify_judge desligado acima
        corpus_chunks=[ouro],
    )

    assert sinais.embedding_low_support is True
    assert sinais.embedding_max_cosine_gold is not None, (
        "o ouro continua registado como diagnóstico"
    )


def test_protocolo_declara_fonte_do_grounding() -> None:
    from llm_evaluation.config import load_config
    from llm_evaluation.protocol import build_protocolo_ativo

    proto = build_protocolo_ativo(load_config(REPO / "configs/default.yaml"))
    assert proto["embedding_grounding_source"] == "chunks_recuperados"


# --------------------------------------------------------------------------
# Proveniência do chunk ouro (SPEC-001) — substring errava em dois casos reais.
# --------------------------------------------------------------------------


def test_is_gold_por_proveniencia_nao_marca_distractor_que_contem_o_ouro() -> None:
    """Ouro curto contido num distractor: antes, o distractor vinha marcado ouro."""
    item = EvalItem(
        id="p1",
        question="Qual é a capital?",
        correct_answers=["Paris"],
        incorrect_answers=[],
        rag_gold_chunk="Paris",
        rag_distractors=["Paris é a capital de França e tem muitos museus."],
    )
    corpus = build_corpus_for_item(item, 500)
    recuperados = Retriever(
        HashEmbedder(),
        [c.texto for c in corpus],
        gold_flags=[c.e_ouro for c in corpus],
    ).retrieve(item.question, 4, inject_remove_gold=False, item=item)

    por_texto = {c.text: c.is_gold for c in recuperados}
    assert por_texto["Paris"] is True
    assert por_texto["Paris é a capital de França e tem muitos museus."] is False


def test_retriever_sem_proveniencia_mantem_modo_substring() -> None:
    """Chamadores antigos não partem: o fallback documentado continua a existir."""
    item = EvalItem(
        id="p2",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
        rag_gold_chunk="texto ouro",
    )
    recuperados = Retriever(HashEmbedder(), ["texto ouro", "outro"]).retrieve(
        item.question, 4, inject_remove_gold=False, item=item
    )
    assert any(c.is_gold for c in recuperados)


def test_gold_flags_desalinhado_falha_cedo() -> None:
    import pytest

    with pytest.raises(ValueError, match="gold_flags"):
        Retriever(HashEmbedder(), ["a", "b"], gold_flags=[True])


# --------------------------------------------------------------------------
# Honestidade do artefacto: corpus sem distratores não mede recuperação.
# --------------------------------------------------------------------------


def test_metricas_declaram_corpus_sem_distratores() -> None:
    item = EvalItem(
        id="d1",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
        rag_gold_chunk="história inteira",
        rag_distractors=[],
    )
    assert corpus_tem_distratores(item, 500) is False

    m = compute_retrieval_metrics(
        item,
        [RetrievedChunk(text="história inteira", score=0.4, is_gold=True)],
        rag_enabled=True,
        n_chunks_corpus=1,
        corpus_tem_distratores=False,
    )
    assert m["corpus_tem_distratores"] is False
    assert m["corpus_devolvido_inteiro"] is True
    # A nota que qualifica a taxa vive no agregado, não repetida em cada linha.
    assert "nota_recuperacao_degenerada" not in m


def test_sumario_qualifica_taxa_quando_nenhum_item_tem_distratores() -> None:
    """`taxa_chunk_ouro_no_top_k` publicava 1,0 como se fosse resultado."""
    recs = [
        _registo(
            f"r{i}",
            anomalia=False,
            meta={
                "metricas_recuperacao": {
                    "rag_ativo": True,
                    "n_chunks_recuperados": 2,
                    "score_melhor_chunk": 0.5,
                    "rank_chunk_ouro": 1,
                    "chunk_ouro_no_top_k": True,
                    "corpus_tem_chunk_ouro": True,
                    "corpus_tem_distratores": False,
                }
            },
        )
        for i in range(3)
    ]
    ret = summarize(recs, reference_type="none")["sumario_recuperacao"]
    assert ret["taxa_chunk_ouro_no_top_k"] == 1.0
    assert ret["n_itens_corpus_sem_distratores"] == 3
    assert ret["nota_taxa_degenerada"] == NOTA_CORPUS_SEM_DISTRATORES


def test_metricas_sem_nota_quando_ha_distratores() -> None:
    item = EvalItem(
        id="d2",
        question="q",
        correct_answers=["a"],
        incorrect_answers=[],
        rag_gold_chunk="ouro",
        rag_distractors=["distractor"],
    )
    assert corpus_tem_distratores(item, 500) is True
    m = compute_retrieval_metrics(
        item,
        [RetrievedChunk(text="ouro", score=0.4, is_gold=True)],
        rag_enabled=True,
        n_chunks_corpus=2,
        corpus_tem_distratores=True,
    )
    assert "nota_recuperacao_degenerada" not in m
    assert m["corpus_devolvido_inteiro"] is False


# --------------------------------------------------------------------------
# Gate do juiz condiciona o subconjunto medido.
# --------------------------------------------------------------------------


def test_judge_report_declara_itens_saltados_pelo_gate() -> None:
    julgado = _registo("j1", anomalia=False)
    julgado.signals.judge = JudgeResult(
        veredito="sustentado",
        motivo_breve="ok",
        confianca=0.8,
        raw={"veredito": "sustentado"},
    )
    saltado = _registo(
        "j2",
        anomalia=False,
        meta={"contexto_juiz": {"judge_skipped_by_gate": True}},
    )

    rel = build_judge_meta_report([julgado, saltado], reference_type="lexical")
    assert rel["n_itens_saltados_por_gate"] == 1
    assert "condicionais" in str(rel["nota_condicionalidade"])


def test_judge_report_sem_gate_nao_acrescenta_nota() -> None:
    rel = build_judge_meta_report([_registo("j3", anomalia=False)], reference_type="lexical")
    assert "n_itens_saltados_por_gate" not in rel
    assert "nota_condicionalidade" not in rel


# --------------------------------------------------------------------------
# Regra 7 do CLAUDE.md: documentação não referencia o que não existe.
# --------------------------------------------------------------------------

REF_CONFIG = re.compile(r"configs/[A-Za-z0-9_./-]+\.yaml")
#: Marcador que autoriza citar um config não distribuído (ver SPEC-A-NQ).
MARCADOR_NAO_DISTRIBUIDO = "não distribuído"


def _ficheiros_de_documentacao() -> list[Path]:
    docs = sorted(p for p in (REPO / "docs").rglob("*.md"))
    raiz = [p for p in (REPO / "README.md", REPO / "README.pt-BR.md") if p.is_file()]
    return [*docs, *raiz]


def test_docs_nao_referenciam_configs_inexistentes() -> None:
    """Mecaniza a regra 7: cada `configs/*.yaml` citado existe, ou o ficheiro declara-o.

    A alternativa aceite é a da SPEC-A-NQ: manter a menção como referência de
    protocolo, com nota a dizer que o ficheiro não é distribuído. O que não é
    aceite é o link silenciosamente partido — que foi como a SPEC-001 passou a
    recomendar uma calibração impossível de reproduzir.
    """
    problemas: list[str] = []
    for doc in _ficheiros_de_documentacao():
        texto = doc.read_text(encoding="utf-8")
        em_falta = sorted({r for r in REF_CONFIG.findall(texto) if not (REPO / r).is_file()})
        if em_falta and MARCADOR_NAO_DISTRIBUIDO not in texto:
            rel = doc.relative_to(REPO)
            problemas.append(f"{rel}: {em_falta}")

    assert not problemas, (
        "Configs citados que não existem, em ficheiros sem a nota «não distribuído»:\n"
        + "\n".join(problemas)
    )

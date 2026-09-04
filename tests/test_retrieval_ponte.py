"""Ponte recuperação → geração: conversão, janela de degradação e cobertura.

Tudo offline: o carregamento do Hugging Face está isolado em `carrega_ponte_hotpotqa`
e não é exercitado aqui.
"""

from __future__ import annotations

from llm_evaluation.retrieval_eval.ponte import (
    ConjuntoPonte,
    cobertura_da_recuperacao,
    itens_para_pipeline,
)


def _conjunto() -> ConjuntoPonte:
    return ConjuntoPonte(
        nome="teste",
        doc_ids=[f"d{i}" for i in range(10)],
        textos=[f"texto {i}" for i in range(10)],
        queries={"q1": "pergunta um", "q2": "pergunta dois"},
        qrels={"q1": {"d0": 1.0}, "q2": {"d5": 1.0}},
        respostas={"q1": "resposta um", "q2": "resposta dois"},
        n_corpus_original=1_000_000,
        semente=42,
    )


def _corrida() -> dict[str, list[str]]:
    # q1 recupera o relevante em primeiro; q2 só o encontra na posição 6.
    return {
        "q1": ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        "q2": ["d1", "d2", "d3", "d4", "d6", "d5", "d7", "d8"],
    }


class TestConversaoParaItens:
    def test_um_item_por_query_julgada(self) -> None:
        itens = itens_para_pipeline(_conjunto(), _corrida())
        assert [i.id for i in itens] == ["q1", "q2"]

    def test_resposta_ouro_viaja_como_referencia(self) -> None:
        itens = itens_para_pipeline(_conjunto(), _corrida())
        assert itens[0].correct_answers == ["resposta um"]

    def test_chunk_ouro_e_a_passagem_julgada_mesmo_fora_da_janela(self) -> None:
        # q2 só encontra d5 na posição 6; com top_k=4 fica fora da janela. O chunk
        # ouro tem de continuar presente, senão o pipeline não distingue «não
        # recuperou» de «recuperou e respondeu mal».
        itens = itens_para_pipeline(_conjunto(), _corrida(), top_k=4)
        q2 = next(i for i in itens if i.id == "q2")
        assert q2.rag_gold_chunk == "texto 5"
        assert "texto 5" not in q2.rag_distractors

    def test_o_relevante_nao_entra_tambem_como_distrator(self) -> None:
        itens = itens_para_pipeline(_conjunto(), _corrida(), top_k=4)
        q1 = next(i for i in itens if i.id == "q1")
        assert q1.rag_gold_chunk == "texto 0"
        assert "texto 0" not in q1.rag_distractors
        assert q1.rag_distractors == ["texto 1", "texto 2", "texto 3"]

    def test_desvio_desloca_a_janela_sem_mudar_mais_nada(self) -> None:
        base = itens_para_pipeline(_conjunto(), _corrida(), top_k=2, desvio=0)
        degradado = itens_para_pipeline(_conjunto(), _corrida(), top_k=2, desvio=4)
        assert [i.id for i in base] == [i.id for i in degradado]
        assert [i.question for i in base] == [i.question for i in degradado]
        assert [i.correct_answers for i in base] == [i.correct_answers for i in degradado]
        assert [i.rag_gold_chunk for i in base] == [i.rag_gold_chunk for i in degradado]
        # Só o contexto muda — é o desenho da ablação.
        assert base[0].rag_distractors != degradado[0].rag_distractors

    def test_query_sem_candidatos_produz_item_sem_contexto(self) -> None:
        itens = itens_para_pipeline(_conjunto(), {"q1": [], "q2": []})
        assert all(i.rag_distractors == [] for i in itens)
        assert all(i.rag_gold_chunk for i in itens)


class TestCobertura:
    """A variável independente da ablação. Sem ela a diferença não é interpretável."""

    def test_janela_do_topo_apanha_so_a_query_facil(self) -> None:
        out = cobertura_da_recuperacao(_conjunto(), _corrida(), top_k=4, desvio=0)
        assert out["n_queries"] == 2
        assert out["n_com_relevante_na_janela"] == 1
        assert out["cobertura"] == 0.5

    def test_janela_deslocada_perde_a_que_estava_no_topo(self) -> None:
        out = cobertura_da_recuperacao(_conjunto(), _corrida(), top_k=2, desvio=4)
        # q1 perde d0 (ficou na posição 0); q2 apanha d5 (posição 5).
        assert out["n_com_relevante_na_janela"] == 1
        assert out["desvio"] == 4

    def test_query_sem_candidatos_nao_entra_no_denominador(self) -> None:
        out = cobertura_da_recuperacao(_conjunto(), {"q1": ["d0"], "q2": []})
        assert out["n_queries"] == 1
        assert out["cobertura"] == 1.0


class TestResumoDeclaraOQueFoiFeitoAoIndice:
    def test_subamostragem_e_declarada_e_nao_comparavel(self) -> None:
        r = _conjunto().resumo()
        assert r["corpus_subamostrado"] is True
        assert r["comparavel_com_beir"] is False
        assert r["n_passagens_no_corpus_original"] == 1_000_000
        assert r["semente_amostragem"] == 42

    def test_corpus_completo_nao_se_declara_subamostrado(self) -> None:
        from dataclasses import replace

        c = replace(_conjunto(), n_corpus_original=10)
        assert c.resumo()["corpus_subamostrado"] is False

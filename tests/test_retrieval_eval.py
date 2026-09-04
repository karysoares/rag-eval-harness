"""Avaliação de recuperação: métricas, BM25 e fusão.

Tudo offline e determinístico — o carregamento do Hugging Face está isolado em
`beir.py` e é exercitado por um teste de integração à parte.
"""

from __future__ import annotations

import math

import pytest

from llm_evaluation.retrieval_eval.beir import ConjuntoBeir
from llm_evaluation.retrieval_eval.bm25 import BM25Index, tokenize
from llm_evaluation.retrieval_eval.metrics import (
    avalia_corrida,
    mrr_at_k,
    ndcg_at_k,
    ndcg_por_query,
    recall_at_k,
)
from llm_evaluation.retrieval_eval.run import (
    compara_emparelhado,
    compara_metodos,
    corrida_bm25,
    corrida_densa,
    rrf,
)


class TestMetricas:
    def test_recall_conta_relevantes_apanhados(self) -> None:
        assert recall_at_k({"a", "b"}, ["a", "x"], 2) == 0.5
        assert recall_at_k({"a", "b"}, ["a", "b"], 2) == 1.0
        assert recall_at_k({"a"}, ["x", "y"], 2) == 0.0

    def test_recall_respeita_o_corte_em_k(self) -> None:
        assert recall_at_k({"a"}, ["x", "a"], 1) == 0.0
        assert recall_at_k({"a"}, ["x", "a"], 2) == 1.0

    def test_recall_sem_relevantes_e_indefinido(self) -> None:
        """Zero seria inventar um resultado negativo onde não há verdade."""
        assert math.isnan(recall_at_k(set(), ["a"], 10))

    def test_mrr_e_o_inverso_da_primeira_posicao_certa(self) -> None:
        assert mrr_at_k({"a"}, ["a", "b", "c"], 10) == 1.0
        assert mrr_at_k({"c"}, ["a", "b", "c"], 10) == pytest.approx(1 / 3)
        assert mrr_at_k({"z"}, ["a", "b"], 10) == 0.0

    def test_ndcg_perfeito_e_um(self) -> None:
        assert ndcg_at_k({"a": 1.0, "b": 1.0}, ["a", "b", "x"], 10) == 1.0

    def test_ndcg_penaliza_a_posicao(self) -> None:
        cedo = ndcg_at_k({"a": 1.0}, ["a", "x", "y"], 10)
        tarde = ndcg_at_k({"a": 1.0}, ["x", "y", "a"], 10)
        assert cedo > tarde

    def test_ndcg_usa_graus_quando_existem(self) -> None:
        """Grau 2 antes de grau 1 vale mais do que o inverso."""
        bom = ndcg_at_k({"a": 2.0, "b": 1.0}, ["a", "b"], 10)
        mau = ndcg_at_k({"a": 2.0, "b": 1.0}, ["b", "a"], 10)
        assert bom == 1.0
        assert mau < bom

    def test_ndcg_sem_relevantes_e_zero(self) -> None:
        assert ndcg_at_k({}, ["a"], 10) == 0.0

    def test_agregado_reporta_os_dois_denominadores(self) -> None:
        corrida = {"q1": ["a"], "q2": ["b"], "q3": ["c"]}
        qrels = {"q1": {"a": 1.0}, "q2": {"z": 1.0}}  # q3 sem julgamentos
        out = avalia_corrida(corrida, qrels, ks=(10,))
        assert out["n_queries_avaliadas"] == 2
        assert out["n_queries_sem_qrels"] == 1
        assert out["recall@10"] == 0.5

    def test_agregado_sem_nada_avaliavel(self) -> None:
        out = avalia_corrida({"q1": ["a"]}, {}, ks=(10,))
        assert out["n_queries_avaliadas"] == 0
        assert "recall@10" not in out


class TestBM25:
    @staticmethod
    def _indice() -> BM25Index:
        return BM25Index().build(
            ["d1", "d2", "d3"],
            [
                "the cat sat on the mat",
                "financial planning for retirement accounts",
                "the cat and the dog",
            ],
        )

    def test_tokenize_normaliza(self) -> None:
        assert tokenize("The Cat, 42 times!") == ["the", "cat", "42", "times"]

    def test_ordena_por_relevancia(self) -> None:
        r = self._indice().search("cat mat", 3)
        assert [d for d, _ in r][0] == "d1"

    def test_termo_desconhecido_devolve_vazio(self) -> None:
        assert self._indice().search("xyzzy", 3) == []

    def test_query_vazia_devolve_vazio(self) -> None:
        assert self._indice().search("", 3) == []

    def test_documentos_sem_sobreposicao_nao_entram(self) -> None:
        """Scores zero são omitidos — não são resultados, são ausência deles."""
        r = self._indice().search("retirement", 3)
        assert [d for d, _ in r] == ["d2"]

    def test_top_k_limita(self) -> None:
        assert len(self._indice().search("the cat", 1)) == 1

    def test_indice_por_construir_falha_claramente(self) -> None:
        with pytest.raises(RuntimeError, match="não construído"):
            BM25Index().search("x")

    def test_termo_raro_pesa_mais_que_comum(self) -> None:
        """IDF: 'the' aparece em dois documentos, 'retirement' num só."""
        ix = self._indice()
        comum = dict(ix.search("the", 5)).get("d1", 0.0)
        raro = dict(ix.search("retirement", 5)).get("d2", 0.0)
        assert raro > comum


class _EmbedderFalso:
    """Vetores fixos: d3 é o mais próximo da query, invertendo a ordem do BM25."""

    def __init__(self, mapa: dict[str, list[float]]) -> None:
        self._mapa = mapa

    def embed(self, textos: list[str]) -> object:
        import numpy as np

        return np.array([self._mapa[t] for t in textos], dtype=np.float32)


class TestFusaoEDenso:
    def test_rrf_premeia_consenso(self) -> None:
        """Um documento bem colocado nos dois rankings passa à frente."""
        a = ["x", "consenso", "y"]
        b = ["z", "consenso", "w"]
        assert rrf(a, b)[0] == "consenso"

    def test_rrf_usa_posicao_e_nao_score(self) -> None:
        assert rrf(["a", "b"], ["a", "b"]) == ["a", "b"]

    def test_rrf_respeita_top_k(self) -> None:
        assert len(rrf(["a", "b", "c"], ["c", "b", "a"], top_k=2)) == 2

    def test_denso_reordena_so_a_profundidade_pedida(self) -> None:
        ds = ConjuntoBeir(
            nome="t",
            doc_ids=["d1", "d2", "d3"],
            textos=["t1", "t2", "t3"],
            queries={"q": "consulta"},
            qrels={"q": {"d3": 1.0}},
        )
        emb = _EmbedderFalso(
            {
                "consulta": [1.0, 0.0],
                "t1": [0.0, 1.0],
                "t2": [0.5, 0.5],
                "t3": [1.0, 0.0],
            }
        )
        out = corrida_densa(emb, ds, {"q": ["d1", "d2", "d3"]}, profundidade=3)
        assert out["q"][0] == "d3"

    def test_cauda_alem_da_profundidade_mantem_ordem_do_bm25(self) -> None:
        """Descartá-la baixaria o recall@1000 por artefacto do protocolo."""
        ds = ConjuntoBeir(
            nome="t",
            doc_ids=["d1", "d2", "d3"],
            textos=["t1", "t2", "t3"],
            queries={"q": "consulta"},
            qrels={"q": {"d1": 1.0}},
        )
        emb = _EmbedderFalso({"consulta": [1.0, 0.0], "t1": [1.0, 0.0], "t2": [0.0, 1.0]})
        out = corrida_densa(emb, ds, {"q": ["d1", "d2", "d3"]}, profundidade=2)
        assert out["q"][-1] == "d3"

    def test_candidatos_vazios_nao_rebentam(self) -> None:
        ds = ConjuntoBeir(nome="t", doc_ids=[], textos=[], queries={"q": "x"}, qrels={})
        assert corrida_densa(_EmbedderFalso({}), ds, {"q": []})["q"] == []


class TestConjuntoBeir:
    def test_resumo_expoe_denominadores(self) -> None:
        ds = ConjuntoBeir(
            nome="t",
            doc_ids=["d1", "d2"],
            textos=["a", "b"],
            queries={"q1": "x", "q2": "y"},
            qrels={"q1": {"d1": 1.0, "d2": 1.0}},
        )
        r = ds.resumo()
        assert r["n_passagens"] == 2
        assert r["n_queries_total"] == 2
        assert r["n_queries_julgadas"] == 1
        assert r["relevantes_por_query"] == 2.0

    def test_corrida_bm25_cobre_so_queries_julgadas(self) -> None:
        ds = ConjuntoBeir(
            nome="t",
            doc_ids=["d1"],
            textos=["gato preto"],
            queries={"q1": "gato", "q2": "cão"},
            qrels={"q1": {"d1": 1.0}},
        )
        ix = BM25Index().build(ds.doc_ids, ds.textos)
        assert set(corrida_bm25(ix, ds)) == {"q1"}


class TestDenominadorHonesto:
    """Regressão: uma query julgada ausente da corrida inflacionava tudo.

    Antes da correção, omitir a query que o método falhou subia o recall de
    0,667 para 1,0 sem nada no relatório o indicar.
    """

    QRELS = {"q1": {"a": 1.0}, "q2": {"b": 1.0}, "q3": {"c": 1.0}}

    def test_corrida_parcial_nao_inflaciona(self) -> None:
        completa = avalia_corrida({"q1": ["a"], "q2": ["b"], "q3": ["x"]}, self.QRELS, ks=(10,))
        parcial = avalia_corrida({"q1": ["a"], "q2": ["b"]}, self.QRELS, ks=(10,))
        assert parcial["recall@10"] == completa["recall@10"]
        assert parcial["n_queries_avaliadas"] == 3

    def test_ausencia_e_declarada(self) -> None:
        out = avalia_corrida({"q1": ["a"]}, self.QRELS, ks=(10,))
        assert out["n_queries_julgadas_ausentes"] == 2

    def test_query_ausente_conta_zero_e_nao_desaparece(self) -> None:
        out = avalia_corrida({"q1": ["a"]}, {"q1": {"a": 1.0}, "q2": {"b": 1.0}}, ks=(10,))
        assert out["recall@10"] == 0.5

    def test_extra_na_corrida_sem_qrels_e_contado_a_parte(self) -> None:
        out = avalia_corrida({"q1": ["a"], "extra": ["z"]}, {"q1": {"a": 1.0}}, ks=(10,))
        assert out["n_queries_sem_qrels"] == 1
        assert out["n_queries_avaliadas"] == 1


class TestComparacaoEmparelhada:
    """Comparar métodos sobre as mesmas queries é um desenho emparelhado."""

    def test_ndcg_por_query_expoe_granularidade(self) -> None:
        por_q = ndcg_por_query({"q1": ["a"], "q2": ["x"]}, {"q1": {"a": 1.0}, "q2": {"b": 1.0}})
        assert por_q == {"q1": 1.0, "q2": 0.0}

    def test_metodo_melhor_produz_ic_que_exclui_zero(self) -> None:
        qrels = {f"q{i}": {"a": 1.0} for i in range(40)}
        bom = {f"q{i}": ["a"] for i in range(40)}
        mau = {f"q{i}": ["z", "a"] for i in range(40)}
        cmp = compara_emparelhado({"bom": bom, "mau": mau}, qrels)
        assert len(cmp) == 1
        assert cmp[0]["diferenca_observada"] > 0
        assert cmp[0]["exclui_zero"] is True

    def test_metodos_iguais_nao_excluem_zero(self) -> None:
        qrels = {f"q{i}": {"a": 1.0} for i in range(30)}
        igual = {f"q{i}": ["a"] for i in range(30)}
        cmp = compara_emparelhado({"a": igual, "b": dict(igual)}, qrels)
        assert cmp[0]["diferenca_observada"] == 0
        assert cmp[0]["exclui_zero"] is False

    def test_sem_corridas_suficientes_devolve_vazio(self) -> None:
        assert compara_emparelhado({"so_um": {"q": ["a"]}}, {"q": {"a": 1.0}}) == []


class _EmbedderCaracteres:
    """Coseno determinístico por sobreposição de caracteres — sem modelo nem rede."""

    def embed(self, textos: list[str]):
        import numpy as np

        vetores = []
        for t in textos:
            v = np.zeros(26, dtype=float)
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            n = np.linalg.norm(v)
            vetores.append(v / n if n else v)
        return np.asarray(vetores)


class TestDegrausOpcionais:
    """Cada degrau é opcional; a ausência de um não pode rebentar os anteriores."""

    def _ds(self) -> ConjuntoBeir:
        return ConjuntoBeir(
            nome="t",
            doc_ids=["d1", "d2", "d3"],
            textos=["gato preto dorme", "cao branco corre", "gato branco salta"],
            queries={"q1": "gato branco"},
            qrels={"q1": {"d3": 1.0}},
        )

    def test_sem_embedder_devolve_so_bm25_com_nota(self) -> None:
        r = compara_metodos(self._ds(), embedder=None)
        assert set(r["metodos"]) == {"bm25"}
        assert "nota" in r

    def test_com_embedder_e_sem_reranker_nao_rebenta(self) -> None:
        # Regressão: o bloco do cross-encoder estava fora do `if reranker is not
        # None`, e este caminho levantava NameError sobre uma variável por definir.
        r = compara_metodos(self._ds(), embedder=_EmbedderCaracteres())
        assert set(r["metodos"]) == {"bm25", "denso_sobre_bm25", "hibrido_rrf"}
        assert "cross_encoder_sobre_bm25" not in r["metodos"]

    def test_comparacao_emparelhada_cobre_os_degraus_corridos(self) -> None:
        r = compara_metodos(self._ds(), embedder=_EmbedderCaracteres())
        pares = {tuple(c["par"]) for c in r["comparacao_emparelhada"]}
        assert ("bm25", "denso_sobre_bm25") in pares

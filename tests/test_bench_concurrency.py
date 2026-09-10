"""O benchmark de concorrência corre e produz um artefacto coerente.

Os números de desempenho do README eram afirmados sem script, sem teste e sem
artefacto — do lado errado da invariante 5, e invisíveis a uma regressão. Este teste
usa uma latência minúscula: valida a mecânica da medição, não a sua magnitude (essa
mede-se com `make bench`, que leva ~26 s).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

bench = pytest.importorskip("bench_concurrency")


@pytest.fixture(scope="module")
def medicao() -> dict[str, Any]:
    return bench.mede(concorrencias=[1, 2], n_itens=6, n_documentos=2, latencia_s=0.002)


def test_cenario_e_declarado(medicao: dict[str, Any]) -> None:
    """Um número de desempenho sem os parâmetros de que deriva não é interpretável."""
    cenario = medicao["cenario"]
    assert cenario["n_itens"] == 6
    assert cenario["n_documentos"] == 2
    assert cenario["latencia_por_chamada_s"] == 0.002
    assert cenario["chamadas_por_item"] == 2
    assert "tecto" in cenario["nota"]


def test_duas_chamadas_por_item(medicao: dict[str, Any]) -> None:
    """Gerador + juiz. Se passar a uma, a comparação de custo do README muda."""
    for r in medicao["resultados"]:
        assert r["n_chamadas"] == 12


def test_ganho_relativo_a_sequencial(medicao: dict[str, Any]) -> None:
    assert medicao["resultados"][0]["ganho"] == 1.0
    assert medicao["resultados"][1]["ganho"] is not None


def test_cache_de_embeddings_e_reportada(medicao: dict[str, Any]) -> None:
    cache = medicao["cache_embeddings"]
    assert cache, "as estatísticas da cache não foram capturadas do stderr"
    for stats in cache.values():
        assert 0.0 <= stats["embeddings_cache_taxa_acerto"] <= 1.0
        assert stats["embeddings_cache_hits"] + stats["embeddings_cache_misses"] > 0


def test_artefacto_publicado_esta_coerente_com_o_script() -> None:
    """O ficheiro citado no README tem de ter a forma que o script produz hoje."""
    caminho = Path(__file__).resolve().parents[1] / "docs" / "evidencia" / "bench_concorrencia.json"
    if not caminho.is_file():
        pytest.skip("artefacto não gerado neste ambiente (make bench)")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    assert set(d) >= {"cenario", "resultados", "cache_embeddings"}
    assert d["resultados"][0]["concorrencia"] == 1
    assert d["resultados"][0]["ganho"] == 1.0
    for r in d["resultados"]:
        assert r["segundos"] > 0
        assert r["ganho"] is None or r["ganho"] >= 1.0

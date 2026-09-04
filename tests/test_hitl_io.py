"""hitl_io CSV merge."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_evaluation.hitl_io import (
    HITL_CSV_FIELDS,
    export_hitl_csv_template,
    read_hitl_csv,
    write_hitl_manifest,
)


def test_read_hitl_csv_dedupe(tmp_path: Path) -> None:
    csv = tmp_path / "a.csv"
    csv.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\ni1,correto,a,,\ni1,incorreto,b,,\n",
        encoding="utf-8",
    )
    labels = read_hitl_csv(csv)
    assert labels["i1"]["rotulo"] == "incorreto"


def test_read_hitl_csv_strict_rejects_invalid_rows(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "id_item,rotulo,revisor,timestamp_utc,notas\ni1,rotulo_invalido,a,,\n",
        encoding="utf-8",
    )
    assert read_hitl_csv(csv) == {}
    with pytest.raises(ValueError, match="linhas inválidas"):
        read_hitl_csv(csv, strict=True)


def test_write_hitl_manifest(tmp_path: Path) -> None:
    csv = tmp_path / "adj.csv"
    csv.write_text("id_item,rotulo,revisor,timestamp_utc,notas\ni1,correto,x,,\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    p = write_hitl_manifest(run, csv)
    assert p.is_file()


class TestTemplateDeAdjudicacao:
    """O template tem de ser rotulável sem abrir o `predictions.jsonl` em paralelo."""

    def test_sem_contexto_mantem_apenas_as_colunas_canonicas(self, tmp_path: Path) -> None:
        destino = tmp_path / "t.csv"
        export_hitl_csv_template(destino, ["a", "b"])
        with destino.open(encoding="utf-8", newline="") as f:
            leitor = csv.DictReader(f)
            assert leitor.fieldnames == list(HITL_CSV_FIELDS)
            assert [linha["id_item"] for linha in leitor] == ["a", "b"]

    def test_com_contexto_acrescenta_colunas_de_leitura(self, tmp_path: Path) -> None:
        destino = tmp_path / "t.csv"
        export_hitl_csv_template(
            destino,
            ["a"],
            contexto={"a": {"pergunta": "porquê?", "resposta_modelo": "porque sim"}},
        )
        with destino.open(encoding="utf-8", newline="") as f:
            linhas = list(csv.DictReader(f))
        assert linhas[0]["pergunta"] == "porquê?"
        assert linhas[0]["resposta_modelo"] == "porque sim"
        # Um id sem contexto não pode partir a escrita.
        assert linhas[0]["contexto_recuperado"] == ""

    def test_colunas_extra_sao_ignoradas_pela_leitura(self, tmp_path: Path) -> None:
        # As colunas de leitura viajam com o ficheiro que a pessoa devolve; a
        # aplicação tem de as ignorar em vez de rejeitar o CSV.
        destino = tmp_path / "t.csv"
        export_hitl_csv_template(destino, ["a"], contexto={"a": {"pergunta": "q"}})
        texto = destino.read_text(encoding="utf-8").splitlines()
        cabecalho, linha = texto[0], texto[1]
        linha = linha.replace(",,,,", ",correto,rev,,", 1)
        destino.write_text("\n".join([cabecalho, linha]) + "\n", encoding="utf-8")
        lido = read_hitl_csv(destino, strict=True)
        assert lido["a"]["rotulo"] == "correto"
        assert lido["a"]["revisor"] == "rev"

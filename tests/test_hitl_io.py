"""hitl_io CSV merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evaluation.hitl_io import read_hitl_csv, write_hitl_manifest


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

"""Gravação de anotação HITL no CSV."""

from __future__ import annotations

from pathlib import Path

from llm_evaluation.hitl_io import read_hitl_csv, save_hitl_annotation


def test_save_hitl_annotation_roundtrip(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    p = save_hitl_annotation(
        run,
        item_id="item-1",
        rotulo="correto",
        revisor="ana",
        notas="ok",
    )
    assert p.is_file()
    labels = read_hitl_csv(p)
    assert labels["item-1"]["rotulo"] == "correto"
    save_hitl_annotation(run, item_id="item-1", rotulo="incorreto", revisor="ana")
    labels2 = read_hitl_csv(p)
    assert labels2["item-1"]["rotulo"] == "incorreto"

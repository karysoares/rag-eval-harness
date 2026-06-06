"""Testes do carregamento unificado de itens de avaliação."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_evaluation.config import load_config
from llm_evaluation.eval_items_load import load_eval_items


def test_load_eval_items_amostra_local_limit() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/default.yaml")
    cfg = replace(cfg, dataset=replace(cfg.dataset, mode="amostra_local", limit=1))
    items = load_eval_items(cfg)
    assert len(items) == 1
    assert items[0].id == "amostra-1"


def test_smoke_amostra_config_loads() -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    assert cfg.dataset.mode == "amostra_local"
    assert cfg.dataset.reference_type == "lexical"
    items = load_eval_items(cfg)
    assert len(items) == 2


def test_yaml_alias_orquestracao(tmp_path: Path) -> None:
    """Chave de topo `orquestracao` é aceite como `orchestration`."""
    repo = Path(__file__).resolve().parents[1]
    base = (repo / "configs/default.yaml").read_text(encoding="utf-8")
    patched = base.replace("orchestration:", "orquestracao:", 1)
    p = tmp_path / "cfg.yaml"
    p.write_text(patched, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.orchestration in ("unico", "multiplo")

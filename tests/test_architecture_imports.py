"""Fronteiras de import (dashboard não depende de pipeline)."""

from __future__ import annotations

import ast
from pathlib import Path


def _imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_dashboard_app_no_pipeline_import() -> None:
    app = Path(__file__).resolve().parents[1] / "src/llm_evaluation/dashboard/app.py"
    banned = {"pipeline", "generation"}
    found = _imports_in(app) & banned
    assert not found, f"dashboard/app.py importa: {found}"

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_run_evidence.py"
    spec = importlib.util.spec_from_file_location("publish_run_evidence", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_embedding_policy_blocks_nonzero(monkeypatch, tmp_path: Path) -> None:
    mod = _module()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="validate_embedding_policy falhou"):
        mod.run_embedding_policy(tmp_path)


def test_assert_publishable_blocks_failed_p0(monkeypatch, tmp_path: Path) -> None:
    mod = _module()
    monkeypatch.setattr(mod, "validate_run_artifacts", lambda *_a, **_k: [])
    monkeypatch.setattr(mod, "predictions_contain_judge_cot", lambda *_a, **_k: False)
    monkeypatch.setattr(
        mod,
        "run_embedding_policy",
        lambda *_a, **_k: {"criterio_p0": {"passou": False}},
    )

    with pytest.raises(SystemExit, match="criterio_p0.passou"):
        mod.assert_publishable(tmp_path)

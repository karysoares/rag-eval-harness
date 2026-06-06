"""CLI ``scripts/audit_run.py`` em fixture versionada (CI strict)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_run_strict_on_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    fixture_root = root / "tests" / "fixtures" / "audit_runs"
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "audit_run.py"), str(fixture_root), "--strict"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_audit_run_accepts_direct_run_dir() -> None:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "tests" / "fixtures" / "audit_runs" / "run_ci_fixture"
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "audit_run.py"), str(run_dir), "--strict"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_audit_run_fails_when_no_runs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "audit_run.py"), str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "nenhuma corrida encontrada" in proc.stdout

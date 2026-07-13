"""Cobertura dos caminhos de erro e modos offline da CLI (sem API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evaluation import cli


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["llm-eval", *argv])
    cli.main()


class TestArgumentValidation:
    def test_apply_hitl_requires_resume(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        csv = tmp_path / "h.csv"
        csv.write_text("id_item,rotulo\n")
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--apply-hitl", str(csv)])
        assert exc.value.code == 2

    def test_apply_hitl_missing_csv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch,
                ["--apply-hitl", str(tmp_path / "nao_existe.csv"), "--resume", str(run_dir)],
            )
        assert exc.value.code == 2

    def test_analyze_run_rejects_non_directory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--analyze-run", str(tmp_path / "nao_existe")])
        assert exc.value.code == 2

    def test_compare_runs_requires_two_dirs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        d = tmp_path / "a"
        d.mkdir()
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--compare-runs", str(d)])
        assert exc.value.code == 2

    def test_compare_runs_rejects_missing_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        d = tmp_path / "a"
        d.mkdir()
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--compare-runs", str(d), str(tmp_path / "fantasma")])
        assert exc.value.code == 2

    def test_missing_config_exits_2(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--config", str(tmp_path / "nao_existe.yaml")])
        assert exc.value.code == 2


class TestExperimentalGate:
    def test_multi_orchestration_blocked_without_experimental(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = Path("configs/smoke_amostra.yaml")
        if not config.is_file():
            pytest.skip("config smoke ausente")
        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch,
                ["--config", str(config), "--orchestration", "multi"],
            )
        assert exc.value.code == 2


class TestAnalyzeRunOffline:
    def test_analyze_run_writes_metrics_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--analyze-run sobre fixture de run existente não exige API."""
        fixture = Path("tests/fixtures/audit_runs")
        if not fixture.is_dir():
            pytest.skip("fixture de run ausente")
        run_dirs = [p for p in fixture.iterdir() if p.is_dir()]
        if not run_dirs:
            pytest.skip("fixture sem run dirs")
        src = run_dirs[0]
        dst = tmp_path / src.name
        import shutil

        shutil.copytree(src, dst)
        _run_cli(monkeypatch, ["--analyze-run", str(dst)])
        report = dst / "metrics_report.json"
        assert report.is_file()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

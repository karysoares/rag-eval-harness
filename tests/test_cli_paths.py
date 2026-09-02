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


class TestJudgeReport:
    """``--judge-report``: meta-avaliação offline do juiz a partir de predictions.jsonl."""

    @staticmethod
    def _run_dir(tmp_path: Path, *, com_juiz: bool = True) -> Path:
        run_dir = tmp_path / "run_meta"
        run_dir.mkdir()
        linhas = []
        for i in range(4):
            sinais: dict[str, object] = {
                "gold_correto": i % 2 == 0,
                "gold_incorreto": i % 2 == 1,
                "e_recusa": False,
                "embedding_max_coseno": 0.6,
                "embedding_baixo_suporte": False,
            }
            if com_juiz:
                sinais["juiz"] = {
                    "veredito": "sustentado" if i % 2 == 0 else "nao_sustentado",
                    "motivo_breve": "m",
                    "confianca": 0.8,
                }
            linhas.append(
                json.dumps(
                    {
                        "id_item": f"item-{i}",
                        "pergunta": "q?",
                        "resposta": "uma resposta de teste",
                        "gold_correto": i % 2 == 0,
                        "flag_anomalia": False,
                        "perfil_baseline": "hibrido",
                        "sinais": sinais,
                        "recuperados": [],
                        "meta": {"metricas_recuperacao": {"rank_chunk_ouro": 1}},
                    },
                    ensure_ascii=False,
                )
            )
        (run_dir / "predictions.jsonl").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        (run_dir / "summary.json").write_text(
            json.dumps({"tipo_referencia_ativo": "answer_lists"}),
            encoding="utf-8",
        )
        return run_dir

    def test_grava_judge_report_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = self._run_dir(tmp_path)
        _run_cli(monkeypatch, ["--judge-report", str(run_dir)])
        report = json.loads((run_dir / "judge_report.json").read_text(encoding="utf-8"))
        assert report["n_itens"] == 4
        assert report["n_itens_com_veredito_real"] == 4
        assert report["tipo_referencia"] == "answer_lists"
        assert report["concordancia_com_referencia"]["exatidao"] == 1.0

    def test_rejeita_diretorio_inexistente(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--judge-report", str(tmp_path / "nao_existe")])
        assert exc.value.code == 2

    def test_rejeita_run_dir_sem_predictions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        vazio = tmp_path / "vazio"
        vazio.mkdir()
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, ["--judge-report", str(vazio)])
        assert exc.value.code == 2

    def test_rejeita_jsonl_de_amostras_inexistente(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = self._run_dir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch,
                [
                    "--judge-report",
                    str(run_dir),
                    "--judge-samples",
                    str(tmp_path / "nao_existe.jsonl"),
                ],
            )
        assert exc.value.code == 2

    def test_amostras_alimentam_a_seccao_de_autoconsistencia(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        run_dir = self._run_dir(tmp_path)
        amostras = tmp_path / "samples.jsonl"
        amostras.write_text(
            "\n".join(
                json.dumps({"id_item": f"item-{i}", "vereditos": ["sustentado", "incompleto"]})
                for i in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        _run_cli(monkeypatch, ["--judge-report", str(run_dir), "--judge-samples", str(amostras)])
        report = json.loads((run_dir / "judge_report.json").read_text(encoding="utf-8"))
        assert report["autoconsistencia"]["n_itens"] == 2
        assert report["autoconsistencia"]["taxa_itens_unanimes"] == 0.0

    def test_corrida_sem_juiz_gera_relatorio_vazio_com_aviso(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = self._run_dir(tmp_path, com_juiz=False)
        _run_cli(monkeypatch, ["--judge-report", str(run_dir)])
        report = json.loads((run_dir / "judge_report.json").read_text(encoding="utf-8"))
        assert report["n_itens_com_veredito_real"] == 0
        assert "nenhum item com veredito real" in capsys.readouterr().err

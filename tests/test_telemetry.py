"""Telemetria: contrato, adaptadores e os invariantes que não pode violar.

O mais importante está em `TestInvariantes`: a telemetria é um canal lateral e
não pode alterar artefactos, derrubar a corrida, nem exportar conteúdo sem que
alguém o peça.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from llm_evaluation import pipeline
from llm_evaluation.config import load_config
from llm_evaluation.observability import LlmCallUsage
from llm_evaluation.telemetry import (
    CloudWatchEmfExporter,
    JsonlExporter,
    MultiExporter,
    NullExporter,
    build_exporter,
    telemetry_targets_from_env,
)
from llm_evaluation.telemetry.base import ItemEvent, LlmCallEvent, RunEvent, item_attributes
from llm_evaluation.telemetry.emit import build_item_event, telemetry_includes_content
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals

_GEN = json.dumps({"resposta": "Resposta.", "confianca": 0.9, "contexto_insuficiente": False})
_JUD = json.dumps(
    {
        "cadeia_de_pensamento": ["ok"],
        "veredito": "sustentado",
        "motivo_breve": "m",
        "confianca": 0.8,
    }
)


def _record(item_id: str = "i0", *, anomaly: bool = False) -> RunRecord:
    return RunRecord(
        item_id=item_id,
        question="Qual a capital?",
        answer="Brasília.",
        gold_correct=True,
        anomaly_flag=anomaly,
        signals=VerificationSignals(
            gold_correct=True,
            gold_incorrect=False,
            is_refusal=False,
            embedding_max_cosine=0.71,
            embedding_low_support=False,
            judge=JudgeResult("sustentado", "m", 0.8, raw={"veredito": "sustentado"}),
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={"metricas_recuperacao": {"score_melhor_chunk": 0.66}},
    )


def _call(role: str = "generation") -> LlmCallUsage:
    return LlmCallUsage(
        role=role,  # type: ignore[arg-type]
        model="m",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        latency_ms=250.0,
        started_at=time.time(),
        endpoint="http://localhost:11434",
    )


class TestSelecaoDeDestinos:
    def test_sem_env_nao_ha_destinos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_EVAL_TELEMETRY", raising=False)
        assert telemetry_targets_from_env() == []
        assert isinstance(build_exporter(), NullExporter)

    def test_lista_separada_por_virgulas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EVAL_TELEMETRY", " jsonl , cloudwatch ")
        assert telemetry_targets_from_env() == ["jsonl", "cloudwatch"]

    def test_varios_destinos_produzem_multi(self, tmp_path: Path) -> None:
        exp = build_exporter(run_dir=tmp_path, targets=["jsonl", "cloudwatch"])
        assert isinstance(exp, MultiExporter)
        exp.close()

    def test_destino_desconhecido_avisa_e_nao_falha(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exp = build_exporter(targets=["nao_existe"])
        assert isinstance(exp, NullExporter)
        assert "indisponível" in capsys.readouterr().err

    def test_langsmith_sem_chave_e_saltado(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert isinstance(build_exporter(targets=["langsmith"]), NullExporter)

    def test_otlp_sem_endpoint_e_saltado(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert isinstance(build_exporter(targets=["otlp"]), NullExporter)

    def test_um_destino_mau_nao_impede_o_bom(self, tmp_path: Path) -> None:
        exp = build_exporter(run_dir=tmp_path, targets=["jsonl", "nao_existe"])
        assert isinstance(exp, JsonlExporter)
        exp.close()


class TestJsonlExporter:
    def test_grava_item_e_run(self, tmp_path: Path) -> None:
        destino = tmp_path / "t.jsonl"
        exp = JsonlExporter(destino)
        exp.on_item(build_item_event(_record(), calls=[_call()], started_at=time.time()))
        exp.on_run(
            RunEvent(
                run_id="run_x",
                started_at=time.time(),
                duration_ms=10.0,
                n_items=1,
                n_anomalies=0,
                config_name="c.yaml",
            )
        )
        exp.close()
        linhas = [json.loads(x) for x in destino.read_text().splitlines()]
        assert [x["tipo"] for x in linhas] == ["item", "run"]
        assert linhas[0]["atributos"]["eval.item_id"] == "i0"
        assert linhas[0]["chamadas"][0]["gen_ai.usage.input_tokens"] == 100

    def test_cria_diretorio_em_falta(self, tmp_path: Path) -> None:
        exp = JsonlExporter(tmp_path / "novo" / "t.jsonl")
        exp.close()
        assert (tmp_path / "novo" / "t.jsonl").is_file()


class TestCloudWatchEmf:
    def test_emite_emf_valido(self) -> None:
        buffer = io.StringIO()
        exp = CloudWatchEmfExporter(namespace="Teste", stream=buffer)
        exp.on_item(
            build_item_event(_record(anomaly=True), calls=[_call()], started_at=time.time())
        )
        doc = json.loads(buffer.getvalue().splitlines()[0])
        metrica = doc["_aws"]["CloudWatchMetrics"][0]
        assert metrica["Namespace"] == "Teste"
        assert {m["Name"] for m in metrica["Metrics"]} >= {"PromptTokens", "AnomalyFlag"}
        assert doc["AnomalyFlag"] == 1
        assert doc["PromptTokens"] == 100

    def test_run_calcula_taxa_de_anomalia(self) -> None:
        buffer = io.StringIO()
        exp = CloudWatchEmfExporter(stream=buffer)
        exp.on_run(
            RunEvent(
                run_id="r",
                started_at=time.time(),
                duration_ms=1.0,
                n_items=4,
                n_anomalies=1,
                config_name="c.yaml",
            )
        )
        assert json.loads(buffer.getvalue())["AnomalyRate"] == 25.0

    def test_corrida_vazia_nao_divide_por_zero(self) -> None:
        buffer = io.StringIO()
        CloudWatchEmfExporter(stream=buffer).on_run(
            RunEvent("r", time.time(), 1.0, 0, 0, "c.yaml"),
        )
        assert json.loads(buffer.getvalue())["AnomalyRate"] == 0.0


class TestAtributos:
    def test_nomes_seguem_convencoes_reconhecidas(self) -> None:
        attrs = item_attributes(
            build_item_event(_record(), calls=[_call(), _call("judge")], started_at=time.time()),
        )
        assert attrs["llm.token_count.prompt"] == 200
        assert attrs["llm.call_count"] == 2
        assert attrs["eval.judge.verdict"] == "sustentado"
        assert attrs["eval.embedding.max_cosine"] == 0.71
        assert attrs["eval.retrieval.top_score"] == 0.66


class TestInvariantes:
    """Os três compromissos declarados em `telemetry.base`."""

    def test_conteudo_nao_e_exportado_por_omissao(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_EVAL_TELEMETRY_CONTENT", raising=False)
        assert telemetry_includes_content() is False
        evento = build_item_event(_record(), calls=[], started_at=time.time())
        assert evento.content == {}
        serializado = json.dumps(item_attributes(evento))
        assert "Brasília" not in serializado
        assert "capital" not in serializado

    def test_conteudo_exportado_apenas_com_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_EVAL_TELEMETRY_CONTENT", "1")
        assert telemetry_includes_content() is True
        evento = build_item_event(_record(), calls=[], started_at=time.time(), include_content=True)
        assert evento.content["answer"] == "Brasília."

    def test_exportador_que_rebenta_nao_derruba_a_corrida(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Explosivo:
            def on_item(self, event: ItemEvent) -> None:
                msg = "backend em baixo"
                raise RuntimeError(msg)

            def on_run(self, event: RunEvent) -> None:
                msg = "backend em baixo"
                raise RuntimeError(msg)

            def close(self) -> None:
                msg = "backend em baixo"
                raise RuntimeError(msg)

        monkeypatch.setattr(pipeline, "build_exporter", lambda **_: _Explosivo())
        records = _corre_pipeline(monkeypatch)
        assert len(records) == 2
        assert all(r.meta.get("processing_error") is None for r in records)

    def test_artefactos_identicos_com_e_sem_telemetria(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Invariante 1: a telemetria é canal lateral, não altera o que se grava."""
        from llm_evaluation.reporting import record_to_json

        sem = [record_to_json(r) for r in _corre_pipeline(monkeypatch)]
        monkeypatch.setenv("LLM_EVAL_TELEMETRY", "jsonl")
        com = [record_to_json(r) for r in _corre_pipeline(monkeypatch, run_dir=tmp_path)]

        def _limpa(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # A observabilidade traz latências reais, que variam entre corridas.
            for row in rows:
                row["meta"].pop("observabilidade", None)
            return rows

        assert _limpa(com) == _limpa(sem)
        assert (tmp_path / "telemetry.jsonl").is_file()

    def test_meta_nao_transporta_estruturas_da_telemetria(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """`meta` tem de continuar serializável: nada de dataclasses lá dentro."""
        monkeypatch.setenv("LLM_EVAL_TELEMETRY", "jsonl")
        for r in _corre_pipeline(monkeypatch, run_dir=tmp_path):
            json.dumps(r.meta, ensure_ascii=False, default=str)
            assert not any(k.startswith("_") for k in r.meta)


class _FakeLlm:
    def complete(self, system: str, user: str, **_: object) -> str:
        if "veredito" in system.lower() or "veredito" in user.lower():
            return _JUD
        return _GEN


def _corre_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_dir: Path | None = None,
) -> list[RunRecord]:
    from llm_evaluation.eval_items_load import load_eval_items

    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: _FakeLlm())
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: _FakeLlm())
    monkeypatch.delenv("LLM_EVAL_CONCURRENCY", raising=False)
    cfg = load_config(Path("configs/smoke_amostra.yaml"))
    cfg = replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )
    return pipeline.run_batch(cfg, load_eval_items(cfg), run_dir=run_dir, config_name="t.yaml")


def test_evento_de_chamada_preserva_endpoint_e_papel() -> None:
    evento = build_item_event(_record(), calls=[_call("judge")], started_at=time.time())
    assert isinstance(evento.calls[0], LlmCallEvent)
    assert evento.calls[0].role == "judge"
    assert evento.calls[0].endpoint == "http://localhost:11434"

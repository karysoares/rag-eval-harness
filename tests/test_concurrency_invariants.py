"""Invariantes de concorrência: o paralelismo não pode mudar o que se mede.

O loop de itens corre num `ThreadPoolExecutor` porque o trabalho é dominado por
latência de rede. Isso cria duas obrigações que nenhum teste anterior cobria:

1. **Estado partilhado no caminho da medição.** Um scorer reutilizado entre
   threads não rebenta — devolve um número errado atribuído ao item errado, que
   é o pior modo de falha possível num harness de avaliação.
2. **Determinismo do artefacto.** `predictions.jsonl` tem de sair idêntico
   independentemente do número de workers, ou as corridas deixam de ser
   comparáveis entre si.

Segue o padrão de `test_telemetry.py::TestInvariantes`: afirmar a propriedade,
não o sintoma.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from llm_evaluation import lexical_metrics, pipeline
from llm_evaluation.config import load_config
from llm_evaluation.types import EvalItem

if TYPE_CHECKING:
    from pathlib import Path

    from llm_evaluation.config import AppConfig


class TestEstadoPartilhadoNasMetricas:
    """`sacrebleu` e `rouge_score` não documentam segurança entre threads.

    O `BLEU` do sacrebleu mantém cache interna de referências; partilhá-lo entre
    itens que correm em paralelo é uma corrida de dados no caminho que produz os
    números publicados.
    """

    def _recolhe_por_thread(self, fabrica: Any, n: int = 8) -> list[int]:
        vistos: list[int] = []
        barreira = threading.Barrier(n)

        def _tira() -> int:
            # A barreira força as threads a existirem ao mesmo tempo: sem ela,
            # o pool poderia reutilizar uma única thread e o teste passaria por
            # acidente.
            barreira.wait(timeout=10)
            return id(fabrica())

        with ThreadPoolExecutor(max_workers=n) as pool:
            vistos = list(pool.map(lambda _: _tira(), range(n)))
        return vistos

    def test_bleu_e_por_thread(self) -> None:
        ids = self._recolhe_por_thread(lexical_metrics._bleu)
        assert len(set(ids)) == len(ids), "o mesmo objecto BLEU foi partilhado entre threads"

    def test_rouge_e_por_thread(self) -> None:
        ids = self._recolhe_por_thread(lexical_metrics._rouge)
        assert len(set(ids)) == len(ids), "o mesmo objecto ROUGE foi partilhado entre threads"

    def test_a_mesma_thread_reaproveita_a_instancia(self) -> None:
        # Por thread, não por chamada: construir um scorer por item seria caro
        # e desnecessário.
        assert lexical_metrics._bleu() is lexical_metrics._bleu()
        assert lexical_metrics._rouge() is lexical_metrics._rouge()


class _LlmDeterminista:
    """Responde em função do prompt, para que a saída dependa do item e não da ordem."""

    def complete(self, system: str, user: str) -> str:
        if "veredito" in system.lower() or "veredito" in user.lower():
            return json.dumps(
                {
                    "cadeia_de_pensamento": ["ok"],
                    "veredito": "sustentado",
                    "motivo_breve": "ok",
                    "confianca": 0.9,
                }
            )
        return json.dumps(
            {
                "resposta": f"Resposta para {hash(user) % 1000}.",
                "confianca": 0.9,
                "contexto_insuficiente": False,
            }
        )


@pytest.fixture
def cfg_offline(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    cfg = replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )
    fake = _LlmDeterminista()
    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: fake)
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: fake)
    return cfg


def _itens(n: int = 24) -> list[EvalItem]:
    return [
        EvalItem(
            id=f"conc-{i:03d}",
            question=f"Qual é o facto número {i}?",
            correct_answers=[f"O facto número {i} é este texto de referência."],
            incorrect_answers=[],
            rag_gold_chunk=f"Documento sobre o facto {i}, com contexto suficiente para responder.",
            rag_distractors=[f"Documento irrelevante {j} sem relação com {i}." for j in range(3)],
        )
        for i in range(n)
    ]


def _corre(cfg: AppConfig, itens: list[EvalItem], destino: Path) -> None:
    with destino.open("w", encoding="utf-8") as fh:

        def _escreve(rec: Any) -> None:
            fh.write(json.dumps(rec.item_id, ensure_ascii=False) + "\n")
            for chave in ("lexical", "metricas_lexicas"):
                if chave in rec.meta:
                    fh.write(json.dumps(rec.meta[chave], ensure_ascii=False, sort_keys=True) + "\n")

        pipeline.run_batch(cfg, itens, on_record=_escreve)


class TestDeterminismoSobConcorrencia:
    def test_artefacto_identico_com_1_e_com_8_workers(
        self,
        cfg_offline: AppConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concorrência é uma opção de execução, não um parâmetro da medição."""
        itens = _itens()

        monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "1")
        sequencial = tmp_path / "sequencial.jsonl"
        _corre(cfg_offline, itens, sequencial)

        monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "8")
        paralelo = tmp_path / "paralelo.jsonl"
        _corre(cfg_offline, itens, paralelo)

        assert sequencial.read_bytes() == paralelo.read_bytes()


class TestDeterminismoChegaAoArtefacto:
    """Perder o determinismo é um facto sobre a corrida, não um aviso no terminal.

    Um modelo que rejeita `temperature` corre com a temperatura por omissão. Quem
    ler os números meses depois tem de o saber a partir do `summary.json` — não do
    stderr de quem lançou a corrida.
    """

    @pytest.fixture(autouse=True)
    def _limpa(self) -> Any:
        from llm_evaluation import llm_client

        llm_client.reset_temperature_rejected()
        yield
        llm_client.reset_temperature_rejected()

    def test_registo_e_thread_safe_e_sem_duplicados(self) -> None:
        from llm_evaluation import llm_client

        n = 16
        barreira = threading.Barrier(n)

        def _regista(i: int) -> None:
            barreira.wait(timeout=10)
            llm_client.record_temperature_rejected(f"modelo-{i % 2}")

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(_regista, range(n)))

        assert llm_client.temperature_rejected_models() == ["modelo-0", "modelo-1"]

    def test_proveniencia_declara_corrida_reproduzivel_por_omissao(
        self, cfg_offline: AppConfig, tmp_path: Path
    ) -> None:
        from pathlib import Path as _P

        from llm_evaluation.run_artifacts import collect_run_metadata

        repo = _P(__file__).resolve().parents[1]
        meta = collect_run_metadata(
            cfg_offline,
            config_path=repo / "configs/smoke_amostra.yaml",
            run_dir=tmp_path / "run_x",
            n_records=2,
        )
        assert meta["determinismo"] == {
            "temperatura_fixada": True,
            "modelos_sem_temperatura": [],
            "reproduzivel": True,
        }

    def test_proveniencia_nomeia_o_modelo_que_perdeu_determinismo(
        self, cfg_offline: AppConfig, tmp_path: Path
    ) -> None:
        from pathlib import Path as _P

        from llm_evaluation import llm_client
        from llm_evaluation.run_artifacts import collect_run_metadata

        llm_client.record_temperature_rejected("gpt-5-mini")
        repo = _P(__file__).resolve().parents[1]
        meta = collect_run_metadata(
            cfg_offline,
            config_path=repo / "configs/smoke_amostra.yaml",
            run_dir=tmp_path / "run_y",
            n_records=2,
        )
        det = meta["determinismo"]
        assert det["modelos_sem_temperatura"] == ["gpt-5-mini"]
        assert det["reproduzivel"] is False


class TestAvisoUnicoSobConcorrencia:
    def test_warn_once_avisa_uma_vez_com_threads_simultaneas(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from llm_evaluation.telemetry import exporters

        estado: dict[str, bool] = {}
        n = 12
        barreira = threading.Barrier(n)

        def _avisa(_: int) -> None:
            barreira.wait(timeout=10)
            exporters._warn_once(estado, "destino", "backend em baixo")

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(_avisa, range(n)))

        linhas = [ln for ln in capsys.readouterr().err.splitlines() if "backend em baixo" in ln]
        assert len(linhas) == 1

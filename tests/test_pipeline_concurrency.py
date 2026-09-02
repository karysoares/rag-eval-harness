"""Concorrência de itens em ``run_batch``.

O contrato: aumentar ``llm.concurrency`` acelera a corrida mas não pode alterar
nem a ordem nem o conteúdo dos registos — ``predictions.jsonl`` tem de ser
byte-idêntico ao do modo sequencial.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from llm_evaluation import pipeline
from llm_evaluation.config import AppConfig, load_config
from llm_evaluation.types import EvalItem, RunRecord

_GENERATION_REPLY = json.dumps(
    {"resposta": "Resposta fixa do mock.", "confianca": 0.9, "contexto_insuficiente": False}
)
_JUDGE_REPLY = json.dumps(
    {
        "cadeia_de_pensamento": ["ok"],
        "veredito": "sustentado",
        "motivo_breve": "ok",
        "confianca": 0.9,
    }
)


class _ConcurrentLlm:
    """Mock que regista concorrência máxima observada e é seguro entre threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.n_calls = 0

    def complete(self, system: str, user: str, **_: object) -> str:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.n_calls += 1
        try:
            # Latência simulada: sem ela o pool esvazia antes de haver sobreposição.
            threading.Event().wait(0.02)
            if "veredito" in system.lower() or "veredito" in user.lower():
                return _JUDGE_REPLY
            return _GENERATION_REPLY
        finally:
            with self._lock:
                self.in_flight -= 1


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> _ConcurrentLlm:
    fake = _ConcurrentLlm()
    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: fake)
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: fake)
    monkeypatch.delenv("LLM_EVAL_CONCURRENCY", raising=False)
    return fake


def _cfg(concurrency: int) -> AppConfig:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    return replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
        llm=replace(cfg.llm, concurrency=concurrency),
    )


def _items(n: int) -> list[EvalItem]:
    return [
        EvalItem(
            id=f"item-{i:03d}",
            question=f"Qual é o facto número {i}?",
            correct_answers=[f"O facto {i}."],
            incorrect_answers=[],
            rag_gold_chunk=f"O facto número {i} é conhecido desde 1900.",
            rag_distractors=[f"Texto irrelevante {i}a.", f"Texto irrelevante {i}b."],
        )
        for i in range(n)
    ]


def _serialize(records: list[RunRecord]) -> list[tuple[str, str, bool]]:
    return [(r.item_id, r.answer, r.anomaly_flag) for r in records]


def test_concorrencia_preserva_ordem_e_conteudo(fake_llm: _ConcurrentLlm) -> None:
    items = _items(12)
    sequencial = pipeline.run_batch(_cfg(1), items)
    concorrente = pipeline.run_batch(_cfg(4), items)
    assert _serialize(concorrente) == _serialize(sequencial)
    assert [r.item_id for r in concorrente] == [i.id for i in items]


def test_concorrencia_sobrepoe_chamadas_ao_llm(fake_llm: _ConcurrentLlm) -> None:
    pipeline.run_batch(_cfg(4), _items(12))
    assert fake_llm.max_in_flight > 1


def test_modo_sequencial_nunca_sobrepoe(fake_llm: _ConcurrentLlm) -> None:
    pipeline.run_batch(_cfg(1), _items(6))
    assert fake_llm.max_in_flight == 1


def test_on_record_e_chamado_em_ordem_numa_unica_thread(fake_llm: _ConcurrentLlm) -> None:
    items = _items(10)
    vistos: list[str] = []
    threads: set[int] = set()

    def _on_record(rec: RunRecord) -> None:
        vistos.append(rec.item_id)
        threads.add(threading.get_ident())

    pipeline.run_batch(_cfg(4), items, on_record=_on_record)
    assert vistos == [i.id for i in items]
    assert len(threads) == 1


def test_observabilidade_por_item_nao_mistura_threads(fake_llm: _ConcurrentLlm) -> None:
    """Cada item regista as suas chamadas — 1 geração + 1 juiz, sem contaminação."""
    records = pipeline.run_batch(_cfg(4), _items(10))
    contagens = {r.meta["observabilidade"]["n_chamadas_llm"] for r in records}
    assert contagens == {2}


def test_resolve_concurrency_usa_config_por_omissao(fake_llm: _ConcurrentLlm) -> None:
    assert pipeline.resolve_concurrency(_cfg(3)) == 3


def test_env_sobrepoe_config(
    fake_llm: _ConcurrentLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "7")
    assert pipeline.resolve_concurrency(_cfg(1)) == 7


def test_env_invalido_cai_no_valor_da_config(
    fake_llm: _ConcurrentLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "muitos")
    assert pipeline.resolve_concurrency(_cfg(2)) == 2


def test_concorrencia_minima_e_um(
    fake_llm: _ConcurrentLlm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "0")
    assert pipeline.resolve_concurrency(_cfg(1)) == 1


def test_falha_de_item_nao_derruba_o_lote_concorrente(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Um item que falha sempre vira `_failed_record`; os restantes continuam."""

    class _FlakyLlm:
        def complete(self, system: str, user: str, **_: object) -> str:
            if "facto número 7?" in user:
                msg = "falha simulada"
                raise RuntimeError(msg)
            if "veredito" in system.lower() or "veredito" in user.lower():
                return _JUDGE_REPLY
            return _GENERATION_REPLY

    fake = _FlakyLlm()
    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: fake)
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: fake)
    monkeypatch.setenv("LLM_EVAL_ITEM_RETRIES", "1")
    monkeypatch.delenv("LLM_EVAL_CONCURRENCY", raising=False)

    records = pipeline.run_batch(_cfg(4), _items(10))
    assert len(records) == 10
    falhado = next(r for r in records if r.item_id == "item-007")
    assert falhado.meta["processing_error"]["type"] == "RuntimeError"
    assert falhado.anomaly_flag is True
    assert all(r.meta.get("processing_error") is None for r in records if r.item_id != "item-007")

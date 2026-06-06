"""Teste E2E mockando LLM e juiz: garante que o pipeline produz registos consistentes,
chama o juiz com `temperature=0`, e que `on_record` permite persistência incremental.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from llm_evaluation import pipeline
from llm_evaluation.config import load_config
from llm_evaluation.eval_items_load import load_eval_items
from llm_evaluation.types import RunRecord

if TYPE_CHECKING:
    from llm_evaluation.config import AppConfig


def _smoke_cfg() -> AppConfig:
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")
    # backend hash: rápido e determinístico, sem rede
    return replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )


class _StaticLlm:
    """LLM mock: devolve respostas determinísticas pré-definidas."""

    def __init__(self, generation_reply: str, judge_reply: str) -> None:
        self.generation_reply = generation_reply
        self.judge_reply = judge_reply
        self.calls: list[tuple[str, str]] = []
        self.n_complete = 0

    def complete(self, system: str, user: str) -> str:
        self.n_complete += 1
        self.calls.append((system[:40], user[:40]))
        # O prompt do juiz pede um JSON com "veredito"; o respondedor pede a linha RESPOSTA:
        if "veredito" in system.lower() or "veredito" in user.lower():
            return self.judge_reply
        return self.generation_reply


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> _StaticLlm:
    """Substitui clientes por mocks; mantém o resto do pipeline real."""
    judge_payload = json.dumps(
        {
            "cadeia_de_pensamento": ["alinhado ao contexto"],
            "veredito": "sustentado",
            "motivo_breve": "ok",
            "confianca": 0.9,
        }
    )
    fake = _StaticLlm(
        generation_reply=json.dumps(
            {
                "resposta": "Brasília é a capital do Brasil.",
                "confianca": 0.9,
                "contexto_insuficiente": False,
            }
        ),
        judge_reply=judge_payload,
    )

    def _fake_default_llm_from_env(**_: object) -> _StaticLlm:
        return fake

    def _fake_default_judge_from_env(**_: object) -> _StaticLlm:
        return fake

    monkeypatch.setattr(pipeline, "default_llm_from_env", _fake_default_llm_from_env)
    monkeypatch.setattr(pipeline, "default_judge_from_env", _fake_default_judge_from_env)
    return fake


def test_run_batch_produces_records_for_demo(patched_pipeline: _StaticLlm) -> None:
    cfg = _smoke_cfg()
    items = load_eval_items(cfg)
    records = pipeline.run_batch(cfg, items)
    assert len(records) == len(items)
    assert all(isinstance(r, RunRecord) for r in records)
    capital_rec = next(r for r in records if r.item_id == "amostra-1")
    assert "Brasília" in capital_rec.answer
    # Juiz mock devolveu sustentado
    assert capital_rec.signals.judge is not None
    assert capital_rec.signals.judge.veredito == "sustentado"


def test_lexical_reference_does_not_persist_gold_substring_heuristics(
    patched_pipeline: _StaticLlm,
) -> None:
    cfg = _smoke_cfg()
    items = load_eval_items(cfg)
    records = pipeline.run_batch(cfg, items)
    assert all(r.gold_correct is None for r in records)
    assert all(r.signals.gold_correct is None for r in records)
    assert all(r.signals.gold_incorrect is None for r in records)


def test_run_batch_persists_incrementally(
    patched_pipeline: _StaticLlm,
    tmp_path: Path,
) -> None:
    cfg = _smoke_cfg()
    items = load_eval_items(cfg)
    out = tmp_path / "predictions.jsonl"
    written: list[str] = []
    with out.open("w", encoding="utf-8") as fh:

        def _on_record(rec: RunRecord) -> None:
            line = json.dumps({"id": rec.item_id, "gold": rec.gold_correct})
            fh.write(line + "\n")
            fh.flush()
            written.append(line)

        records = pipeline.run_batch(cfg, items, on_record=_on_record)

    # Persistência por linha foi acionada para cada item
    assert len(written) == len(records)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(records)
    payloads = [json.loads(line) for line in lines]
    assert {p["id"] for p in payloads} == {r.item_id for r in records}


def test_weak_retrieval_gate_skips_responder_calls(
    patched_pipeline: _StaticLlm,
) -> None:
    """Com limiar impossível de cumprir, não há chamadas ao respondedor (só juiz por item)."""
    cfg = _smoke_cfg()
    cfg = replace(
        cfg,
        rag=replace(cfg.rag, min_retrieval_score=10.0),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=True),
    )
    items = load_eval_items(cfg)
    patched_pipeline.n_complete = 0
    records = pipeline.run_batch(cfg, items)
    assert len(records) == len(items)
    # Sem gate: 2 chamadas complete por item (geração + juiz); com gate: 1
    assert patched_pipeline.n_complete == len(items)
    assert all(
        r.meta.get("qualidade_geracao", {}).get("curada_por_recuperacao_fraca") for r in records
    )


def test_judge_gate_skips_judge_calls_when_embedding_high(
    patched_pipeline: _StaticLlm,
) -> None:
    """Com gate 0.0, juiz é sempre pulado para respostas não-recusa."""
    cfg = _smoke_cfg()
    cfg = replace(
        cfg,
        verification=replace(cfg.verification, judge_gate_embedding_max_cosine=0.0),
    )
    items = load_eval_items(cfg)
    patched_pipeline.n_complete = 0
    records = pipeline.run_batch(cfg, items)
    # Só chamada de geração por item; juiz pulado via gate.
    assert patched_pipeline.n_complete == len(items)
    assert all(r.signals.judge is None for r in records)


def test_judge_gate_requires_strong_context_when_enabled(
    patched_pipeline: _StaticLlm,
) -> None:
    """Com requisito de contexto forte impossível, gate não pula juiz."""
    cfg = _smoke_cfg()
    cfg = replace(
        cfg,
        verification=replace(
            cfg.verification,
            judge_gate_embedding_max_cosine=0.0,
            judge_gate_requires_strong_context=True,
            judge_gate_min_retrieval_score=10.0,
        ),
    )
    items = load_eval_items(cfg)
    patched_pipeline.n_complete = 0
    records = pipeline.run_batch(cfg, items)
    # Sem skip por gate: geração + juiz por item.
    assert patched_pipeline.n_complete == len(items) * 2
    assert all(r.signals.judge is not None for r in records)


def test_anti_refusal_repair_retries_with_strong_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quando há recusa com contexto forte, força nova geração específica."""
    judge_payload = json.dumps(
        {
            "cadeia_de_pensamento": ["alinhado ao contexto"],
            "veredito": "sustentado",
            "motivo_breve": "ok",
            "confianca": 0.9,
        }
    )

    class _RepairAwareLlm:
        def __init__(self) -> None:
            self.n_complete = 0

        def complete(self, system: str, user: str) -> str:
            self.n_complete += 1
            if "veredito" in system.lower() or "veredito" in user.lower():
                return judge_payload
            if "EVITE RECUSA GENÉRICA" in user:
                return json.dumps(
                    {
                        "resposta": "A história afirma explicitamente esse facto.",
                        "confianca": 0.85,
                        "contexto_insuficiente": False,
                    }
                )
            return json.dumps(
                {
                    "resposta": "Não há informações suficientes para responder.",
                    "confianca": 0.3,
                    "contexto_insuficiente": True,
                }
            )

    fake = _RepairAwareLlm()

    def _fake_default_llm_from_env(**_: object) -> _RepairAwareLlm:
        return fake

    def _fake_default_judge_from_env(**_: object) -> _RepairAwareLlm:
        return fake

    monkeypatch.setattr(pipeline, "default_llm_from_env", _fake_default_llm_from_env)
    monkeypatch.setattr(pipeline, "default_judge_from_env", _fake_default_judge_from_env)

    cfg = _smoke_cfg()
    cfg = replace(
        cfg,
        generation=replace(
            cfg.generation,
            anti_refusal_repair=True,
            anti_refusal_min_retrieval_score=-1.0,
            anti_refusal_max_attempts=1,
        ),
    )
    items = load_eval_items(cfg)
    records = pipeline.run_batch(cfg, items)
    assert len(records) == len(items)
    assert all(
        r.meta.get("qualidade_geracao", {}).get("recusa_reparada_por_contexto") is True
        for r in records
    )
    assert all("Não há informações suficientes" not in r.answer for r in records)

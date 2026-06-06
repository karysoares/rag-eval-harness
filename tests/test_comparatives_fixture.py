"""Contrato dos comparativos versionados."""

from __future__ import annotations

import json
from pathlib import Path


def test_comparatives_json_structure() -> None:
    path = Path(__file__).resolve().parents[1] / "assets" / "benchmarks" / "comparatives.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "1.1"
    eixos = data.get("eixos")
    assert isinstance(eixos, dict)
    assert set(eixos.keys()) >= {"interno", "externo", "calibracao_p0", "hitl"}
    comp = data.get("comparativos")
    assert isinstance(comp, dict)
    assert "interno_fairytale_evolution" in comp
    assert "referencia_tuned_n1025" in comp
    assert "calibracao_p0" in comp
    assert "hitl_amostra" in comp
    evolution = comp["interno_fairytale_evolution"]
    assert len(evolution["corridas"]) == 4
    tuned = comp["referencia_tuned_n1025"]
    assert tuned["harness"]["n_itens"] == 1025
    assert tuned["policy"]["criterio_p0_sugerido"]["passou"] is True
    p0 = comp["calibracao_p0"]
    assert len(p0["casos"]) >= 1


def test_hitl_fixture_sample() -> None:
    base = Path(__file__).resolve().parent / "fixtures" / "hitl_fairytale_sample"
    assert (base / "adjudicacoes_hitl.csv").is_file()
    lines = (base / "predictions_subset.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    meta = json.loads((base / "README.json").read_text(encoding="utf-8"))
    assert meta.get("n_itens") == 6

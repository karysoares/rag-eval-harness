"""Contrato dos comparativos versionados."""

from __future__ import annotations

import json
from pathlib import Path


def test_comparatives_json_structure() -> None:
    path = Path(__file__).resolve().parents[1] / "assets" / "benchmarks" / "comparatives.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "1.2"
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


def _entradas_com_run_id(comp: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Achata as entradas que citam uma corrida concreta."""
    out: list[tuple[str, dict[str, object]]] = []
    for chave, valor in comp.items():
        if not isinstance(valor, dict):
            continue
        if valor.get("run_id") or "proveniencia" in valor:
            out.append((chave, valor))
        for corrida in valor.get("corridas") or []:
            if isinstance(corrida, dict):
                out.append((f"{chave}.{corrida.get('label')}", corrida))
    return out


def test_cada_numero_publicado_declara_se_e_reverificavel() -> None:
    """Invariante 5: um número publicado tem de vir de uma corrida gravada.

    Quando os artefactos deixam de existir o número não se apaga — marca-se como não
    reverificável, para que ninguém o leia como reproduzível. O que não é admissível é
    publicá-lo sem dizer qual dos dois casos é.
    """
    root = Path(__file__).resolve().parents[1]
    data = json.loads(
        (root / "assets" / "benchmarks" / "comparatives.json").read_text(encoding="utf-8")
    )
    for nome, entrada in _entradas_com_run_id(data["comparativos"]):
        prov = entrada.get("proveniencia")
        assert isinstance(prov, dict), f"{nome} cita uma corrida sem bloco de proveniencia"
        presente = prov.get("artefactos_presentes")
        assert isinstance(presente, bool), f"{nome}: artefactos_presentes tem de ser booleano"
        run_id = prov.get("run_id")
        if presente:
            assert (root / "outputs" / str(run_id) / "summary.json").is_file(), (
                f"{nome} diz que os artefactos existem, mas outputs/{run_id} nao os tem"
            )
        else:
            assert prov.get("motivo"), f"{nome} nao e reverificavel e nao diz porque"


def test_meteor_nao_e_publicado_sem_denominador() -> None:
    """METEOR sem o corpus wordnet só pontua pares quase idênticos.

    A média sobre os sobreviventes é estruturalmente inflacionada, não ruidosa: publicá-la
    ao lado de um N que não é o seu denominador afirma uma cobertura que não houve.
    """
    root = Path(__file__).resolve().parents[1]
    bruto = (root / "assets" / "benchmarks" / "comparatives.json").read_text(encoding="utf-8")
    data = json.loads(bruto)
    for nome, entrada in _entradas_com_run_id(data["comparativos"]):
        for bloco in (entrada, entrada.get("harness") or {}, entrada.get("metricas") or {}):
            if not isinstance(bloco, dict) or "media_meteor" not in bloco:
                continue
            assert isinstance(bloco.get("n_meteor"), int), (
                f"{nome} publica media_meteor sem n_meteor"
            )
            assert bloco["n_meteor"] > 0

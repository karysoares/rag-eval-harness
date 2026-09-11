"""Contrato dos comparativos versionados."""

from __future__ import annotations

import json
from pathlib import Path


def test_comparatives_json_structure() -> None:
    """Só entram comparativos verificáveis numa clonagem limpa.

    O ficheiro chegou a publicar sete entradas cujas corridas já não existiam. Um
    número medido mas irreverificável dá a aparência de evidência sem a substância,
    pelo que a versão 2.0 guarda apenas o que se reproduz, e declara o que saiu.
    """
    path = Path(__file__).resolve().parents[1] / "assets" / "benchmarks" / "comparatives.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "2.0"

    removidos = data.get("removidos")
    assert isinstance(removidos, dict), "a remoção tem de ficar registada, não silenciosa"
    assert removidos.get("motivo")
    assert removidos.get("o_que_saiu")

    comp = data.get("comparativos")
    assert isinstance(comp, dict) and comp, "ficou sem nenhum comparativo verificável"
    assert "calibracao_p0" in comp
    p0 = comp["calibracao_p0"]
    assert len(p0["casos"]) >= 1
    assert all(c["run_id"] == "policy_validation_run" for c in p0["casos"])


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
        # `corridas` e `casos` aninham run_ids: sem os percorrer, uma entrada cujos
        # números vivem numa lista passava sem verificação de proveniência.
        for campo in ("corridas", "casos"):
            for item in valor.get(campo) or []:
                if isinstance(item, dict) and (item.get("run_id") or "proveniencia" in item):
                    out.append((f"{chave}.{item.get('label') or item.get('run_id')}", item))
    return out


def test_cada_numero_publicado_declara_se_e_reverificavel() -> None:
    """Invariante 5: um número publicado tem de vir de uma corrida gravada.

    Quando os artefactos deixam de existir o número não se apaga nem se mantém como
    se nada fosse — sai, e a saída fica registada. O que resta tem de apontar para
    algo que exista **nesta clonagem**: uma fixture versionada serve, um directório
    em `outputs/` (gitignored) não serve para terceiros.
    """
    root = Path(__file__).resolve().parents[1]
    data = json.loads(
        (root / "assets" / "benchmarks" / "comparatives.json").read_text(encoding="utf-8")
    )
    for nome, entrada in _entradas_com_run_id(data["comparativos"]):
        prov = entrada.get("proveniencia")
        assert isinstance(prov, dict), f"{nome} cita uma corrida sem bloco de proveniencia"
        assert prov.get("artefactos_presentes") is True, (
            f"{nome} não é verificável; entradas irreverificáveis saem do ficheiro"
        )
        caminho = prov.get("caminho")
        assert caminho, f"{nome} não diz onde estão os artefactos"
        assert (root / str(caminho)).exists(), f"{nome} aponta para {caminho}, que não existe"


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

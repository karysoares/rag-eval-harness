"""Uma corrida completa em CI, com artefactos, auditada e com gate de KPI.

O CI corria `audit_run --strict` contra uma fixture de **uma linha** cujo
`tipo_referencia_ativo` é `answer_lists` e cujas camadas de verificação estão todas
desligadas. Consequência: todos os ramos `lexical` do auditor — que são os do caso de
referência do repositório — nunca eram exercitados, e nenhum passo do CI produzia um
`summary.json` ou um `manifest.json` para auditar. Um harness de avaliação cuja
própria correcção nunca é medida ponta a ponta tem um limite de credibilidade.

Aqui a corrida é real (pipeline completo, LLM mockado, sem rede), escreve os
artefactos pelo mesmo caminho de código que o CLI usa, é auditada com `--strict` e os
seus KPIs são comparados com um golden versionado.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from llm_evaluation import pipeline
from llm_evaluation.config import AppConfig, load_config
from llm_evaluation.eval_items_load import load_eval_items
from llm_evaluation.reporting import record_to_json
from llm_evaluation.run_reprocess import reprocess_run_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_run import audit  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "ci_kpi_golden.json"

#: Tolerância por KPI. O pipeline é determinístico com o LLM mockado e o backend
#: `hash`, pelo que a tolerância existe para variação de plataforma (ordem de
#: soma em float), não para absorver mudança de comportamento.
TOLERANCIA = 1e-6


def _cfg() -> AppConfig:
    """Config do caso de referência, com as camadas léxica e de embedding ligadas.

    `reference_type: lexical` é o que faz o auditor percorrer os ramos que a fixture
    antiga saltava.
    """
    cfg = load_config(RAIZ / "configs" / "smoke_amostra.yaml")
    assert cfg.dataset.reference_type == "lexical"
    assert cfg.lexical_metrics.enabled
    return replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )


class _LlmDeterministico:
    """Respostas fixas: uma correcta e uma deliberadamente fraca.

    Duas respostas diferentes fazem o F1 léxico variar entre itens, para que os
    agregados do gate não sejam constantes triviais.
    """

    def __init__(self) -> None:
        self.n_complete = 0

    def complete(self, system: str, user: str) -> str:
        self.n_complete += 1
        if "veredito" in system.lower() or "veredito" in user.lower():
            return json.dumps(
                {
                    "cadeia_de_pensamento": ["alinhado ao contexto"],
                    "veredito": "sustentado",
                    "motivo_breve": "coerente com o contexto",
                    "confianca": 0.9,
                }
            )
        if "capital" in user.lower():
            return json.dumps(
                {
                    "resposta": "Brasília é a capital do Brasil.",
                    "confianca": 0.9,
                    "contexto_insuficiente": False,
                }
            )
        return json.dumps(
            {
                "resposta": "Não é possível determinar.",
                "confianca": 0.3,
                "contexto_insuficiente": True,
            }
        )


@pytest.fixture
def corrida(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Corre o pipeline e escreve predictions + summary + manifest."""
    fake = _LlmDeterministico()
    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: fake)
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: fake)

    cfg = _cfg()
    itens = load_eval_items(cfg)
    run_dir = tmp_path / "run_20260101T000000Z"
    run_dir.mkdir()

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        pipeline.run_batch(
            cfg,
            itens,
            on_record=lambda rec: fh.write(
                json.dumps(record_to_json(rec), ensure_ascii=False) + "\n"
            ),
            run_dir=run_dir,
            config_name="configs/smoke_amostra.yaml",
        )

    # Mesmo caminho de código do CLI: constrói summary.json, manifest.json e fila.
    reprocess_run_dir(run_dir, cfg=cfg, config_path=RAIZ / "configs" / "smoke_amostra.yaml")
    return run_dir


def test_corrida_produz_os_artefactos_auditaveis(corrida: Path) -> None:
    for nome in ("predictions.jsonl", "summary.json", "manifest.json"):
        assert (corrida / nome).is_file(), f"{nome} não foi escrito"


def test_auditoria_strict_passa_na_corrida_produzida_agora(corrida: Path) -> None:
    """O gate que o CI executava sobre uma fixture passa a correr sobre uma corrida real."""
    problemas, notas = _audita(corrida)
    assert problemas == [], f"auditoria falhou: {problemas}"
    assert isinstance(notas, list)


def test_a_corrida_exercita_os_ramos_lexicais_do_auditor(corrida: Path) -> None:
    """A fixture antiga era `answer_lists` com as camadas desligadas: estes ramos
    do auditor nunca corriam, e são os do caso de referência."""
    summary = json.loads((corrida / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("tipo_referencia_ativo") == "lexical"
    assert isinstance(summary.get("sumario_lexical"), dict)
    linhas = (corrida / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    for linha in linhas:
        registo = json.loads(linha)
        assert "f1_token" in registo["meta"]["metricas_lexicas"]


def test_kpis_nao_derivam_sem_aviso(corrida: Path) -> None:
    """Gate de regressão: um KPI que muda sem que alguém o declare reprova o CI.

    Regenerar de propósito (e revisar o diff antes de commitar):
      LLM_EVAL_REGENERAR_GOLDEN=1 uv run pytest tests/test_ci_corrida_real_offline.py
    """
    observado = _kpis(json.loads((corrida / "summary.json").read_text(encoding="utf-8")))
    if os.environ.get("LLM_EVAL_REGENERAR_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(
            json.dumps(
                {
                    "nota": (
                        "KPIs de uma corrida offline determinística (LLM mockado, backend "
                        "hash) sobre configs/smoke_amostra.yaml. Um diff aqui e uma mudanca "
                        "de comportamento do harness sao a mesma coisa: se for intencional, "
                        "regenere e explique no commit."
                    ),
                    "config": "configs/smoke_amostra.yaml",
                    "tolerancia": TOLERANCIA,
                    "kpis": observado,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        pytest.skip(f"golden regenerado: {GOLDEN}")
    if not GOLDEN.is_file():
        pytest.fail(f"golden ausente: {GOLDEN}. Regenere-o e versione-o.")
    esperado = json.loads(GOLDEN.read_text(encoding="utf-8"))["kpis"]

    assert set(observado) == set(esperado), (
        f"conjunto de KPIs mudou: só no observado={set(observado) - set(esperado)}, "
        f"só no golden={set(esperado) - set(observado)}"
    )
    desvios = {
        k: (esperado[k], observado[k]) for k in esperado if not _proximo(esperado[k], observado[k])
    }
    assert not desvios, f"KPIs derivaram além da tolerância {TOLERANCIA}: {desvios}"


def _audita(run_dir: Path) -> tuple[list[str], list[str]]:
    resultado = audit(run_dir, strict=True)
    if isinstance(resultado, tuple):
        problemas, notas = resultado[0], resultado[1] if len(resultado) > 1 else []
    else:  # pragma: no cover — assinatura antiga
        problemas, notas = resultado, []
    return list(problemas), list(notas)


def _kpis(summary: dict[str, Any]) -> dict[str, Any]:
    """Subconjunto estável e interpretável do sumário, para o gate."""
    lex = summary.get("sumario_lexical") or {}
    out: dict[str, Any] = {
        "n_itens": summary.get("n_itens"),
        "n_itens_avaliados": summary.get("n_itens_avaliados"),
        "n_itens_com_erro_execucao": summary.get("n_itens_com_erro_execucao"),
        "taxa_alerta": summary.get("taxa_alerta"),
        "tipo_referencia_ativo": summary.get("tipo_referencia_ativo"),
    }
    for chave in ("media_f1_token", "media_rouge_l_f", "n_itens_pontuados"):
        out[f"lexical.{chave}"] = lex.get(chave)
    out["lexical.n_por_metrica"] = lex.get("n_por_metrica")
    out["lexical.idioma_normalizacao"] = lex.get("idioma_normalizacao")
    return out


def _proximo(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(float(a) - float(b)) <= TOLERANCIA
    return bool(a == b)

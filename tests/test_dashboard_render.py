"""As funções de renderização do dashboard são executadas, não só importadas.

`app.py` tem ~1160 linhas e o único teste que existia afirmava que `main` é
chamável. Estava também fora do denominador da cobertura, pelo que o portão de 80%
não a via: uma superfície entregue ao utilizador em que nenhuma linha de
renderização era executada em teste nenhum.

Aqui um duplo de Streamlit registra as chamadas e as funções `_render_*` correm
sobre artefactos reais. Não valida aparência — valida que nenhuma delas rebenta com
dados válidos, incluindo os casos que quebram na prática: corrida sem juiz, sem
métricas léxicas, sem chunks, com sumário vazio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from llm_evaluation.dashboard import app as app_mod
from llm_evaluation.dashboard.data import records_to_dataframe
from llm_evaluation.reporting import record_to_json, summarize
from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals


class _Contexto:
    """Serve `with st.sidebar:` e `with st.expander(...):`."""

    def __init__(self, st: _StreamlitDuplo, nome: str) -> None:
        self._st = st
        self._nome = nome

    def __enter__(self) -> _Contexto:
        self._st.chamadas.append(f"enter:{self._nome}")
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def __getattr__(self, nome: str) -> Any:
        return getattr(self._st, nome)


class _Namespace:
    """Devolve uma fábrica permissiva para qualquer atributo.

    Serve `st.column_config.TextColumn(...)` e as figuras do plotly, sobre as quais
    o dashboard chama `add_hline`, `update_layout` e afins.
    """

    def __getattr__(self, _nome: str) -> Any:
        return lambda *a, **k: None


class _StreamlitDuplo:
    """Aceita qualquer chamada de Streamlit e devolve valores utilizáveis.

    Os widgets devolvem o primeiro valor plausível (primeira opção, mínimo do
    slider, False num checkbox) para que o caminho de renderização siga em frente
    de forma determinística.
    """

    def __init__(self) -> None:
        self.chamadas: list[str] = []
        self.sidebar = _Contexto(self, "sidebar")
        self.session_state: dict[str, Any] = {}
        # `st.column_config.TextColumn(...)` é atributo-de-atributo, não chamada
        # directa, pelo que não passa pelo __getattr__ permissivo.
        self.column_config = _Namespace()

    # -- widgets com contrato de retorno ------------------------------------
    def columns(self, spec: Any, **_: Any) -> list[_Contexto]:
        n = spec if isinstance(spec, int) else len(spec)
        self.chamadas.append(f"columns:{n}")
        return [_Contexto(self, f"col{i}") for i in range(n)]

    def tabs(self, nomes: list[str], **_: Any) -> list[_Contexto]:
        self.chamadas.append(f"tabs:{len(nomes)}")
        return [_Contexto(self, n) for n in nomes]

    def expander(self, rotulo: str, **_: Any) -> _Contexto:
        return _Contexto(self, f"expander:{rotulo}")

    def container(self, **_: Any) -> _Contexto:
        return _Contexto(self, "container")

    def form(self, *_: Any, **__: Any) -> _Contexto:
        return _Contexto(self, "form")

    def selectbox(self, _rotulo: str, opcoes: Any, **_: Any) -> Any:
        seq = list(opcoes)
        return seq[0] if seq else None

    def radio(self, _rotulo: str, opcoes: Any, **_: Any) -> Any:
        seq = list(opcoes)
        return seq[0] if seq else None

    def multiselect(self, _rotulo: str, _opcoes: Any, **_: Any) -> list[Any]:
        return []

    def slider(self, _rotulo: str, *args: Any, **kwargs: Any) -> Any:
        if "value" in kwargs:
            return kwargs["value"]
        return args[0] if args else 0

    def number_input(self, _rotulo: str, **kwargs: Any) -> Any:
        return kwargs.get("value", 0)

    def checkbox(self, _rotulo: str, **kwargs: Any) -> bool:
        return bool(kwargs.get("value", False))

    def button(self, *_: Any, **__: Any) -> bool:
        return False

    def form_submit_button(self, *_: Any, **__: Any) -> bool:
        return False

    def text_input(self, _rotulo: str, **kwargs: Any) -> str:
        return str(kwargs.get("value", ""))

    def text_area(self, _rotulo: str, **kwargs: Any) -> str:
        return str(kwargs.get("value", ""))

    def data_editor(self, dados: Any, **_: Any) -> Any:
        return dados

    # -- tudo o mais apenas registra ----------------------------------------
    def __getattr__(self, nome: str) -> Any:
        def _registra(*args: Any, **_: Any) -> None:
            primeiro = str(args[0])[:30] if args else ""
            self.chamadas.append(f"{nome}:{primeiro}")

        return _registra


@pytest.fixture
def st_duplo(monkeypatch: pytest.MonkeyPatch) -> _StreamlitDuplo:
    duplo = _StreamlitDuplo()
    monkeypatch.setattr(app_mod, "st", duplo)
    # px.* devolve figuras; o duplo de st ignora-as, mas plotly precisa de dados
    # válidos. Substituir evita depender do motor gráfico neste teste.
    for nome in ("scatter", "bar", "histogram", "line", "pie", "box"):
        if hasattr(app_mod.px, nome):
            monkeypatch.setattr(app_mod.px, nome, lambda *a, **k: _Namespace())
    return duplo


def _registo(item_id: str, *, com_juiz: bool = True, com_lexico: bool = True) -> RunRecord:
    meta: dict[str, Any] = {"referencias": ["Brasília"]}
    if com_lexico:
        meta["metricas_lexicas"] = {
            "texto_referencia": "Brasília",
            "f1_token": 0.5,
            "rouge_l_f": 0.5,
            "em_squad": False,
            "exact_match": False,
        }
    meta["metricas_recuperacao"] = {
        "rag_ativo": True,
        "n_chunks_recuperados": 1,
        "score_melhor_chunk": 0.7,
        "rank_chunk_ouro": 1,
        "chunk_ouro_no_top_k": True,
    }
    meta["diagnostico"] = {"tags": ["ok"], "padrao_primario": "nenhum", "tier_qualidade": "bom"}
    meta["qualidade_geracao"] = {"confianca": 0.9, "contexto_insuficiente": False}
    return RunRecord(
        item_id=item_id,
        question="Qual é a capital do Brasil?",
        answer="Brasília é a capital.",
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=0.8,
            embedding_low_support=False,
            judge=(
                JudgeResult(veredito="sustentado", motivo_breve="ok", confianca=0.9, raw={})
                if com_juiz
                else None
            ),
            judge_negative=False if com_juiz else None,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta,
    )


@pytest.fixture
def bundle(tmp_path: Path) -> dict[str, Any]:
    registos = [_registo("a"), _registo("b")]
    report = summarize(registos, reference_type="lexical", protocol={"verify_judge": True})
    run_dir = tmp_path / "run_20260101T000000Z"
    run_dir.mkdir()
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in registos:
            fh.write(json.dumps(record_to_json(r), ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {
        "records": registos,
        "df": records_to_dataframe(registos),
        "report": report,
        "run_dir": run_dir,
    }


RENDERIZADORES_REPORT_DF = ["_render_patterns", "_render_retrieval", "_render_reference"]


@pytest.mark.parametrize("nome", RENDERIZADORES_REPORT_DF)
def test_renderizadores_de_report_e_df(
    nome: str, bundle: dict[str, Any], st_duplo: _StreamlitDuplo
) -> None:
    getattr(app_mod, nome)(bundle["report"], bundle["df"])
    assert st_duplo.chamadas, f"{nome} não produziu output"


def _camada(report: dict[str, Any]) -> dict[str, Any]:
    camada = report.get("analise_camadas")
    return camada if isinstance(camada, dict) else {}


def test_render_overview(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_overview(bundle["report"], _camada(bundle["report"]), bundle["df"])
    assert st_duplo.chamadas


def test_render_qa_inspector(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_qa_inspector(bundle["df"], bundle["records"])  # (df, records)
    assert st_duplo.chamadas


def test_render_calibration(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_calibration(
        bundle["report"], bundle["df"], bundle["records"], bundle["run_dir"]
    )
    assert st_duplo.chamadas


def test_render_operational_kpi(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_operational_kpi(bundle["report"])
    assert st_duplo.chamadas


def test_render_signals(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_signals(_camada(bundle["report"]), bundle["report"])
    assert st_duplo.chamadas


def test_render_retrieved_chunks(bundle: dict[str, Any], st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_retrieved_chunks(bundle["records"][0])
    assert st_duplo.chamadas


def test_render_integrity_badges(st_duplo: _StreamlitDuplo) -> None:
    app_mod._render_integrity_badges({"ok": True, "score": 100}, {"avisos": []})
    assert st_duplo.chamadas


def test_renderizadores_sobrevivem_a_corrida_sem_juiz(st_duplo: _StreamlitDuplo) -> None:
    """Uma corrida com o juiz desligado é um protocolo válido, não um artefacto roto."""
    registos = [_registo("a", com_juiz=False)]
    report = summarize(registos, reference_type="lexical", protocol={"verify_judge": False})
    df = records_to_dataframe(registos)
    app_mod._render_overview(report, _camada(report), df)
    app_mod._render_patterns(report, df)
    app_mod._render_signals(_camada(report), report)
    assert st_duplo.chamadas


def test_renderizadores_sobrevivem_a_sumario_vazio(st_duplo: _StreamlitDuplo) -> None:
    vazio = pd.DataFrame()
    app_mod._render_patterns({}, vazio)
    app_mod._render_retrieval({}, vazio)
    app_mod._render_reference({}, vazio)
    app_mod._render_operational_kpi({})


def test_helpers_puros() -> None:
    assert app_mod._fmt(None) == "—" or isinstance(app_mod._fmt(None), str)
    assert isinstance(app_mod._fmt_pct(0.5), str)
    assert app_mod._active_layers_from_report({"protocolo_ativo": {"verify_judge": True}})

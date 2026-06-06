"""Smoke do módulo Streamlit (import sem subir servidor)."""

from __future__ import annotations


def test_dashboard_app_main_callable() -> None:
    from llm_evaluation.dashboard import app

    assert callable(app.main)


def test_dashboard_launch_main_callable() -> None:
    from llm_evaluation.dashboard import launch

    assert callable(launch.main)


def test_active_reference_type_prefers_summary_then_layer() -> None:
    from llm_evaluation.dashboard.app import _active_reference_type

    assert _active_reference_type({"tipo_referencia_ativo": "lexical"}, {}) == "lexical"
    assert _active_reference_type({}, {"tipo_referencia": "answer_lists"}) == "answer_lists"
    assert (
        _active_reference_type(
            {"tipo_referencia_ativo": "lexical"},
            {"tipo_referencia": "none"},
        )
        == "lexical"
    )
    assert _active_reference_type({}, {}) == ""

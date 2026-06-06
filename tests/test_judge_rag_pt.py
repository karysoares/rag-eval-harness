"""Prompts do juiz RAG em português."""

from __future__ import annotations

from llm_evaluation.verification.judge import load_prompt, run_judge


def test_rag_pt_prompt_files_exist() -> None:
    sys_txt = load_prompt("judge_rag_pt_system.txt")
    assert "veredito" in sys_txt
    assert "motivo_breve" in sys_txt
    user_tpl = load_prompt("judge_rag_pt_user_template.txt")
    assert "{question}" in user_tpl
    assert "português" in user_tpl.lower() or "portugues" in user_tpl.lower()
    assert "cadeia_de_pensamento" not in user_tpl.lower()


class _JudgePtLlm:
    def complete(self, system: str, user: str) -> str:
        assert "narrativas" in system or "histórias" in system or "historias" in system
        return (
            '{"cadeia_de_pensamento": ["p1","p2","p3","p4","p5","p6"],'
            '"veredito": "sustentado", "motivo_breve": "ok", "confianca": 0.9}'
        )


def test_run_judge_rag_pt_style() -> None:
    jr, _meta = run_judge(
        question="Por que o menino subiu?",
        context="[1] O menino subiu para falar com a princesa.",
        answer="Para falar com a princesa.",
        client=_JudgePtLlm(),
        prompt_style="rag_pt",
    )
    assert jr.veredito == "sustentado"
    assert "cadeia_de_pensamento" not in jr.raw


def test_run_judge_rag_pt_cot_when_enabled() -> None:
    jr, _meta = run_judge(
        question="q",
        context="ctx",
        answer="a",
        client=_JudgePtLlm(),
        prompt_style="rag_pt",
        return_chain_of_thought=True,
    )
    assert len(jr.raw.get("cadeia_de_pensamento", [])) >= 6

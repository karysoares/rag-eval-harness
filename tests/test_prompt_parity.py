"""Garante integridade dos prompts empacotados (fonte canónica)."""

from __future__ import annotations

from llm_evaluation.prompt_resources import source_prompts_dir

#: Ficheiros esperados, por nome. Uma contagem apenas diz que o número mudou;
#: os nomes dizem **qual** desapareceu — e um prompt em falta é uma corrida que
#: rebenta a meio, não um teste vermelho.
CANONICAL_PROMPTS = frozenset(
    {
        "critic_system.txt",
        "judge_generic_system.txt",
        "judge_generic_user_template.txt",
        "judge_rag_pt_system.txt",
        "judge_rag_pt_user_template.txt",
        "judge_system.txt",
        "judge_user_template.txt",
        "responder_generic_system.txt",
        "responder_generic_user_template.txt",
        "responder_system.txt",
        "responder_user_template.txt",
    }
)


def test_packaged_prompts_complete() -> None:
    canonical = source_prompts_dir()
    files = sorted(canonical.glob("*.txt"))
    assert files, f"Sem prompts .txt em {canonical}"
    nomes = {p.name for p in files}
    assert nomes == CANONICAL_PROMPTS, (
        f"Em falta: {sorted(CANONICAL_PROMPTS - nomes)}; "
        f"inesperados: {sorted(nomes - CANONICAL_PROMPTS)}"
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"Prompt vazio: {path.name}"


def test_todos_os_estilos_resolvem_para_ficheiros_existentes() -> None:
    """Cada estilo configurável tem de apontar para prompts que existem.

    A ligação entre estilo e ficheiro está em três sítios — gerador, juiz e
    hashing da proveniência. Um deles a divergir dá uma corrida que se declara
    reproduzível com o hash do prompt errado.
    """
    from llm_evaluation.generation import _prompt_files as ficheiros_gerador
    from llm_evaluation.prompt_resources import load_prompt_text
    from llm_evaluation.verification.judge import _prompt_files as ficheiros_juiz

    for estilo in ("rag_pt", "generic"):
        for nome in ficheiros_gerador(estilo):  # type: ignore[arg-type]
            assert load_prompt_text(nome).strip(), f"gerador/{estilo}: {nome}"
    for estilo in ("pt", "rag_pt", "generic"):
        for nome in ficheiros_juiz(estilo):  # type: ignore[arg-type]
            assert load_prompt_text(nome).strip(), f"juiz/{estilo}: {nome}"


def test_hashing_da_proveniencia_acompanha_o_estilo() -> None:
    """O `summary.json` tem de hashear os prompts que a corrida usou mesmo.

    Regressão: `prompt_files_for_config` hasheava sempre o respondedor português,
    fosse qual fosse `generation.prompt_style`.
    """
    from dataclasses import replace
    from pathlib import Path

    from llm_evaluation.config import load_config
    from llm_evaluation.run_artifacts import prompt_files_for_config

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/smoke_amostra.yaml")

    generico = replace(
        cfg,
        generation=replace(cfg.generation, prompt_style="generic"),
        verification=replace(cfg.verification, judge_prompt_style="generic"),
    )
    nomes = prompt_files_for_config(generico)
    assert "responder_generic_system.txt" in nomes
    assert "judge_generic_system.txt" in nomes
    assert "responder_system.txt" not in nomes
    assert "judge_rag_pt_system.txt" not in nomes

    portugues = replace(
        cfg,
        generation=replace(cfg.generation, prompt_style="rag_pt"),
        verification=replace(cfg.verification, judge_prompt_style="rag_pt"),
    )
    nomes_pt = prompt_files_for_config(portugues)
    assert "responder_system.txt" in nomes_pt
    assert "judge_rag_pt_system.txt" in nomes_pt
    assert not any("generic" in n for n in nomes_pt)

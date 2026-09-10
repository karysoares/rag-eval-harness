"""Contratos que os configs distribuídos têm de cumprir.

Dois defeitos que só aparecem quando se lê o config inteiro:

1. `metricas_lexicas.idioma` cai em `pt` por omissão, porque o caso de referência é
   português. Num corpus inglês isso deixa de remover `a|an|the` e parte
   `state-of-the-art` em três tokens, ou seja, a comparabilidade com resultados
   ingleses publicados — que é a razão de existir o valor `en` — fica inalcançável.
2. `meteor` ligado sem o corpus `wordnet` instalado produz uma média sobre os pares
   quase idênticos e publica-a ao lado do N total.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_evaluation.config import load_config

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))

#: Corpora cujo texto não é português. Uma entrada nova aqui obriga a declarar o idioma.
CORPORA_NAO_PORTUGUESES = {"hotpotqa_ponte", "hotpotqa", "nq_open", "beir"}


def test_ha_configs_para_testar() -> None:
    assert CONFIGS, "nenhum config encontrado em configs/"


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_idioma_da_normalizacao_corresponde_ao_corpus(caminho: Path) -> None:
    cfg = load_config(caminho)
    if not cfg.lexical_metrics.enabled:
        pytest.skip("métricas léxicas desligadas")
    nome = cfg.dataset.name.lower()
    esperado = "en" if any(c in nome for c in CORPORA_NAO_PORTUGUESES) else "pt"
    assert cfg.lexical_metrics.idioma == esperado, (
        f"{caminho.name}: corpus {cfg.dataset.name!r} normalizado como "
        f"{cfg.lexical_metrics.idioma!r}, esperado {esperado!r}"
    )


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_meteor_nao_vem_ligado_por_omissao(caminho: Path) -> None:
    """Ligá-lo é uma decisão explícita, que `validate_protocol` depois verifica."""
    assert load_config(caminho).lexical_metrics.meteor is False, (
        f"{caminho.name}: meteor ligado; sem o corpus wordnet a média sai enviesada"
    )


def test_validate_protocol_recusa_meteor_sem_wordnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falhar antes da primeira chamada paga, não no sumário depois de 200 chamadas."""
    from dataclasses import replace

    from llm_evaluation import protocol as protocol_mod
    from llm_evaluation.types import EvalItem

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke_amostra.yaml")
    cfg = replace(cfg, lexical_metrics=replace(cfg.lexical_metrics, meteor=True))
    itens = [
        EvalItem(
            id="a",
            question="p",
            correct_answers=["r"],
            incorrect_answers=[],
            rag_gold_chunk="contexto suficientemente longo para contar como corpus " * 5,
            rag_distractors=[],
        )
    ]

    monkeypatch.setattr(protocol_mod, "_count_items_with_corpus", lambda *a, **k: len(itens))
    monkeypatch.setattr("llm_evaluation.lexical_metrics.wordnet_disponivel", lambda: False)
    with pytest.raises(ValueError, match="wordnet"):
        protocol_mod.validate_protocol(cfg, itens)

    monkeypatch.setattr("llm_evaluation.lexical_metrics.wordnet_disponivel", lambda: True)
    protocol_mod.validate_protocol(cfg, itens)

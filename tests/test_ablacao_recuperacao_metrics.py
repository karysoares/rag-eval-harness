"""Regressão para as variáveis dependentes da SPEC-013.

`scripts/ablacao_recuperacao.py` não é um módulo do pacote (não há `mypy`/CI a
correr sobre `scripts/`), mas o cálculo de `respondeu` / `respondeu_e_sustentado`
/ `alucinou` é exactamente o que tornava a tabela publicada em
`docs/specs/013-retrieval-generation-bridge.md` irreproduzível a partir do
código versionado — o script só emitia `taxa_sustentado` (o KPI que o próprio
spec diz apontar ao contrário). Este teste fixa o contrato para não regredir
outra vez sem ser apanhado.

Ponto central: `respondeu` tem de vir de `qualidade_geracao.contexto_insuficiente`
(a declaração estruturada do próprio gerador), não de `signals.is_refusal`
(heurística de texto em `verification/gold.py`) — essa heurística não reconhece
recusas do prompt `generic` como «The context does not provide information
about…» e ficaria em `False` em 100% dos itens do braço degradado, escondendo
a recusa que a ablação inteira existe para medir.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from llm_evaluation.types import JudgeResult, RunRecord, VerificationSignals

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ablacao_recuperacao.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("ablacao_recuperacao", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


ablacao = _load_script()


def _registo(
    item_id: str,
    *,
    contexto_insuficiente: bool | None = False,
    veredito: str | None = "sustentado",
    fallback: bool = False,
    processing_error: bool = False,
) -> RunRecord:
    juiz = None
    if veredito is not None:
        juiz = JudgeResult(
            veredito=veredito,  # type: ignore[arg-type]
            motivo_breve="",
            confianca=0.9,
            raw={"fallback_heuristico": fallback} if fallback else {},
        )
    meta: dict[str, Any] = {"qualidade_geracao": {"contexto_insuficiente": contexto_insuficiente}}
    if processing_error:
        meta["processing_error"] = True
    return RunRecord(
        item_id=item_id,
        question="q",
        answer="a",
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,  # deliberadamente errado — a métrica não deve ler isto
            embedding_max_cosine=None,
            embedding_low_support=None,
            judge=juiz,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta=meta,
    )


def test_respondeu_le_declaracao_estruturada_nao_texto_livre() -> None:
    # is_refusal=False em todos, mas contexto_insuficiente=True: a recusa tem
    # de aparecer mesmo com a heurística de texto a dizer o contrário.
    r = _registo("i1", contexto_insuficiente=True, veredito="sustentado")
    assert ablacao._contexto_insuficiente(r) is True
    produto = ablacao._metricas_produto([r])
    assert produto["respondeu"]["i1"] is False
    assert produto["respondeu_e_sustentado"].get("i1", False) is False


def test_respondeu_e_sustentado_exige_veredito_exacto() -> None:
    # "incompleto" é resposta dada mas não é sustentada nem alucinação —
    # não pode contar para nenhum dos dois numeradores. "inseguro" é um eixo
    # diferente (política, não grounding) e também não conta como alucinação.
    respondeu_incompleto = _registo("i1", contexto_insuficiente=False, veredito="incompleto")
    respondeu_sustentado = _registo("i2", contexto_insuficiente=False, veredito="sustentado")
    respondeu_alucinado = _registo("i3", contexto_insuficiente=False, veredito="nao_sustentado")
    recusou = _registo("i4", contexto_insuficiente=True, veredito="sustentado")
    respondeu_inseguro = _registo("i5", contexto_insuficiente=False, veredito="inseguro")

    produto = ablacao._metricas_produto(
        [
            respondeu_incompleto,
            respondeu_sustentado,
            respondeu_alucinado,
            recusou,
            respondeu_inseguro,
        ]
    )

    assert produto["respondeu"] == {
        "i1": True,
        "i2": True,
        "i3": True,
        "i4": False,
        "i5": True,
    }
    assert produto["respondeu_e_sustentado"] == {
        "i1": False,
        "i2": True,
        "i3": False,
        "i4": False,
        "i5": False,
    }
    assert produto["alucinou"] == {
        "i1": False,
        "i2": False,
        "i3": True,
        "i4": False,
        "i5": False,
    }


def test_contradicao_conta_como_alucinacao() -> None:
    # judge_generic_system.txt define contradicacao como "explicit conflict
    # with context" — tão alucinação quanto nao_sustentado ("hallucination or
    # extrapolation"), só que a inverter um facto em vez de o inventar. Um
    # juiz que devolvesse isto ficaria invisível no numerador de alucinação
    # se só nao_sustentado contasse.
    respondeu_contradicao = _registo("i1", contexto_insuficiente=False, veredito="contradicacao")
    produto = ablacao._metricas_produto([respondeu_contradicao])
    assert produto["alucinou"]["i1"] is True
    assert produto["respondeu_e_sustentado"]["i1"] is False


def test_taxa_com_denominador_zero_nao_lanca_excecao() -> None:
    # Um braço inteiro sem itens medíveis (ex.: quota esgotada em 100% dos
    # itens) não pode derrubar a agregação final com ZeroDivisionError — a
    # corrida já pagou a geração; perder só o resumo é preferível a perder tudo.
    assert ablacao._taxa(0, 0) is None
    assert ablacao._taxa(3, 4) == 0.75


def test_exclusoes_nao_contaminam_numerador_nem_denominador() -> None:
    # Regra 3 e 4 do CLAUDE.md: erro de execução e fallback do juiz não são
    # resultado do sistema — têm de sair do denominador, não virar zero.
    falhou = _registo("i1", contexto_insuficiente=None, processing_error=True)
    fallback = _registo("i2", contexto_insuficiente=False, veredito="sustentado", fallback=True)
    medivel = _registo("i3", contexto_insuficiente=False, veredito="sustentado")

    assert ablacao._contexto_insuficiente(falhou) is None
    assert ablacao._veredito_medivel(fallback) is None

    produto = ablacao._metricas_produto([falhou, fallback, medivel])
    # i1: sem declaração nenhuma, fora de todas as três métricas.
    assert "i1" not in produto["respondeu"]
    # i2: respondeu é medível (vem do gerador, não do juiz), mas o par
    # respondeu_e_sustentado/alucinou fica de fora por o veredito não ser medível.
    assert produto["respondeu"]["i2"] is True
    assert "i2" not in produto["respondeu_e_sustentado"]
    assert "i2" not in produto["alucinou"]
    # i3: medível em tudo.
    assert produto["respondeu_e_sustentado"]["i3"] is True


def test_reproduz_a_tabela_da_spec_013() -> None:
    """Recria, em miniatura, a forma da SPEC-013: cobertura pior -> mais recusa,
    aprovação do juiz sobe mesmo assim porque recusa honesta conta como sustentada.
    """
    braco_bom = (
        [_registo(f"q{i}", contexto_insuficiente=False, veredito="sustentado") for i in range(7)]
        + [_registo("q7", contexto_insuficiente=False, veredito="nao_sustentado")]
        + [_registo("q8", contexto_insuficiente=True, veredito="sustentado")]
    )
    braco_degradado = [
        _registo(f"q{i}", contexto_insuficiente=True, veredito="sustentado") for i in range(8)
    ] + [_registo("q8", contexto_insuficiente=False, veredito="sustentado")]

    prod_bom = ablacao._metricas_produto(braco_bom)
    prod_deg = ablacao._metricas_produto(braco_degradado)

    taxa_respondeu_bom = sum(prod_bom["respondeu"].values()) / len(prod_bom["respondeu"])
    taxa_respondeu_deg = sum(prod_deg["respondeu"].values()) / len(prod_deg["respondeu"])
    assert taxa_respondeu_bom > taxa_respondeu_deg

    taxa_resp_sust_bom = sum(prod_bom["respondeu_e_sustentado"].values()) / len(
        prod_bom["respondeu_e_sustentado"]
    )
    taxa_resp_sust_deg = sum(prod_deg["respondeu_e_sustentado"].values()) / len(
        prod_deg["respondeu_e_sustentado"]
    )
    assert taxa_resp_sust_bom > taxa_resp_sust_deg

    # KPI ingénuo: aprovação do juiz (não-negativo) sobe no braço degradado,
    # porque recusa conta como sustentado — é o achado central da SPEC-013.
    negativos = ["nao_sustentado", "contradicacao", "inseguro"]
    aprov_bom = [ablacao._sustentado(r, negativos) for r in braco_bom]
    aprov_deg = [ablacao._sustentado(r, negativos) for r in braco_degradado]
    taxa_aprov_bom = sum(v for v in aprov_bom if v) / len([v for v in aprov_bom if v is not None])
    taxa_aprov_deg = sum(v for v in aprov_deg if v) / len([v for v in aprov_deg if v is not None])
    assert taxa_aprov_deg > taxa_aprov_bom

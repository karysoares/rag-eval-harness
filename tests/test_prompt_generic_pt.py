"""Contratos do estilo `generic_pt`: português **e** neutro quanto ao domínio.

Antes deste estilo, correr um corpus português obrigava a escolher: `rag_pt`, que diz
ao juiz que avalia contos infantis, ou `generic`, que o faz raciocinar em inglês sobre
texto português — uma variável escondida na camada cujo κ se publica.

Os testes abaixo não verificam que os ficheiros existem (isso é
`test_prompt_parity.py`). Verificam as propriedades que o estilo tem de ter para
cumprir a promessa, porque um prompt degrada por edição bem-intencionada.
"""

from __future__ import annotations

import re

import pytest

from llm_evaluation.prompt_resources import load_prompt_text

JUIZ_SYS = "judge_generic_pt_system.txt"
JUIZ_USER = "judge_generic_pt_user_template.txt"
GERADOR_SYS = "responder_generic_pt_system.txt"
GERADOR_USER = "responder_generic_pt_user_template.txt"

TODOS = (JUIZ_SYS, JUIZ_USER, GERADOR_SYS, GERADOR_USER)
SISTEMAS = (JUIZ_SYS, GERADOR_SYS)

#: Termos que amarram um prompt a um domínio. A presença de qualquer um destes
#: quebra a promessa de neutralidade — que é a única razão de o estilo existir.
TERMOS_DE_DOMINIO = (
    "conto",
    "contos",
    "fairytale",
    "história infantil",
    "histórias infantis",
    "narrativa infantil",
    "princesa",
    "rainha",
    "fada",
)

#: Ortografia pt-PT pré-1990. O corpus de referência é pt-BR e o repositório já
#: alinhou os outros prompts; um novo a divergir reintroduz a variável.
ORTOGRAFIA_PT_PT = (
    "efectiv",
    "acção",
    "exactamente",
    "correcta",
    "correcto",
    "directo",
    "objecto",
    "reflectind",
)


@pytest.mark.parametrize("nome", TODOS)
def test_nao_nomeia_dominio(nome: str) -> None:
    texto = load_prompt_text(nome).lower()
    # `judge_generic_pt_system` cita a rubrica de narrativa uma vez, para a contrastar.
    encontrados = [t for t in TERMOS_DE_DOMINIO if t in texto]
    if nome == GERADOR_SYS:
        # A frase que explica a diferença face à rubrica específica é intencional.
        assert encontrados == ["conto", "contos"], encontrados
        return
    assert not encontrados, f"{nome} nomeia domínio: {encontrados}"


@pytest.mark.parametrize("nome", TODOS)
def test_esta_em_portugues(nome: str) -> None:
    """O ponto do estilo é ser português; um prompt em inglês não o cumpre."""
    texto = load_prompt_text(nome).lower()
    marcas_pt = ("não", "resposta", "contexto", "pergunta", "trecho")
    assert sum(m in texto for m in marcas_pt) >= 4, f"{nome} não parece estar em português"


#: Linha que **ensina** uma equivalência ortográfica, na forma `facto`/`fato`.
#: Essas linhas têm de conter as duas grafias, senão não ensinam nada — logo são
#: excluídas da verificação, em vez de se abrir uma excepção por palavra.
_LINHA_DE_PARES = re.compile(r"`[^`]+`\s*/\s*`[^`]+`")


@pytest.mark.parametrize("nome", TODOS)
def test_ortografia_consistente_com_o_corpus(nome: str) -> None:
    """O corpus de referência é pt-BR e os outros prompts já foram alinhados.

    Excluem-se as linhas que ensinam pares ortográficos: aí a grafia pt-PT é o
    conteúdo. Em qualquer outra linha é uma variável não controlada na camada cujo
    κ se publica.
    """
    suspeitas: list[tuple[int, str]] = []
    for n, linha in enumerate(load_prompt_text(nome).splitlines(), 1):
        if _LINHA_DE_PARES.search(linha):
            continue
        baixa = linha.lower()
        suspeitas += [(n, o) for o in ORTOGRAFIA_PT_PT if o in baixa]
    assert not suspeitas, f"{nome} usa ortografia pt-PT fora das linhas de pares: {suspeitas}"


def test_a_deteccao_de_ortografia_apanharia_uma_regressao() -> None:
    """Um teste que exclui linhas tem de provar que ainda apanha o que importa."""
    linha_normal = "- Aplicar exactamente um valor do enum."
    linha_de_pares = "`acção`/`ação` · `objecto`/`objeto`"
    assert not _LINHA_DE_PARES.search(linha_normal)
    assert any(o in linha_normal.lower() for o in ORTOGRAFIA_PT_PT)
    assert _LINHA_DE_PARES.search(linha_de_pares)


@pytest.mark.parametrize("nome", SISTEMAS)
def test_declara_neutralidade_de_dominio(nome: str) -> None:
    assert "neutro quanto ao domínio" in load_prompt_text(nome)


@pytest.mark.parametrize("nome", TODOS)
def test_tem_fronteira_anti_injection(nome: str) -> None:
    """Sem isto, um trecho do corpus consegue ditar o veredito."""
    texto = load_prompt_text(nome).lower()
    assert "não confiáve" in texto or "não confiabil" in texto
    assert "instruç" in texto


@pytest.mark.parametrize("nome", SISTEMAS)
def test_cobre_falsificacao_de_delimitador(nome: str) -> None:
    """Um trecho que reproduz `=== FIM CONTEXTO ===` tenta terminar o bloco cedo."""
    assert "FIM CONTEXTO" in load_prompt_text(nome)


@pytest.mark.parametrize("nome", (JUIZ_USER, GERADOR_USER))
def test_template_de_utilizador_avisa_do_delimitador(nome: str) -> None:
    assert "falsificar o fim do bloco" in load_prompt_text(nome)


def test_juiz_trata_variedade_ortografica_como_a_mesma_palavra() -> None:
    """Um juiz que leia `facto` vs `fato` como discrepância factual mede ortografia.

    É a lacuna pt-PT/pt-BR que o normalizador léxico ainda não cobre; o prompt não
    pode acrescentar-lhe um segundo enviesamento na camada do juiz.
    """
    texto = load_prompt_text(JUIZ_SYS)
    for par in ("`facto`/`fato`", "`acção`/`ação`"):
        assert par in texto, f"falta o par {par}"
    assert "nunca** contam como discrepância factual" in texto


def test_juiz_tem_arvore_de_decisao_com_precedencia() -> None:
    """Sem precedência declarada, dois vereditos aplicáveis dão resultado instável."""
    texto = load_prompt_text(JUIZ_SYS)
    assert "Árvore de decisão de veredito" in texto
    assert "inseguro` > `contradicacao` > `nao_sustentado` > `incompleto` > `sustentado`" in texto
    assert "primeira** condição satisfeita decide" in texto


def test_juiz_declara_ancoras_de_confianca() -> None:
    """Medido neste harness: juízes declaram 0,91–0,98 acertando 56–61%.

    Um prompt que não ancore a confiança reproduz a sobreconfiança que torna o campo
    inútil para triagem.
    """
    texto = load_prompt_text(JUIZ_SYS)
    assert "Não use ≥0,9 por omissão" in texto
    assert "0,91–0,98" in texto
    for faixa in ("0,95–1,0", "0,80–0,94", "0,60–0,79", "0,40–0,59"):
        assert faixa in texto, f"falta a faixa {faixa}"


def test_juiz_cobre_entradas_patologicas() -> None:
    texto = load_prompt_text(JUIZ_SYS)
    assert "Entradas patológicas" in texto
    for caso in ("Contexto vazio", "Resposta vazia", "Trechos duplicados", "conflito entre si"):
        assert caso in texto, f"falta o caso {caso}"


def test_juiz_tem_regra_de_suporte_parcial() -> None:
    """Uma resposta com três afirmações precisa de três verificações."""
    texto = load_prompt_text(JUIZ_SYS)
    assert "Suporte parcial (regra de dominância)" in texto
    assert "Enumerar internamente" in texto


@pytest.mark.parametrize(
    ("nome", "campos"),
    [
        (JUIZ_SYS, ("`veredito`", "`motivo_breve`", "`confianca`")),
        (GERADOR_SYS, ("`resposta`", "`confianca`", "`contexto_insuficiente`")),
    ],
)
def test_contrato_json_esta_declarado(nome: str, campos: tuple[str, ...]) -> None:
    texto = load_prompt_text(nome)
    for campo in campos:
        assert campo in texto, f"{nome} não declara {campo}"
    assert "APENAS JSON válido" in texto


def test_juiz_lista_o_enum_completo_e_sem_extras() -> None:
    """Um valor de enum inventado quebra a desserialização e a agregação."""
    texto = load_prompt_text(JUIZ_SYS)
    esperados = {"sustentado", "nao_sustentado", "contradicacao", "incompleto", "inseguro"}
    linha = next(x for x in texto.splitlines() if x.startswith("- `veredito`: exatamente um de"))
    citados = set(re.findall(r"`([a-z_]+)`", linha)) - {"veredito"}
    assert citados == esperados, citados


def test_identificadores_de_flags_sao_partilhados_com_as_outras_rubricas() -> None:
    """Traduzir um identificador partiria um rótulo em dois na agregação."""
    pt = load_prompt_text(JUIZ_SYS)
    en = load_prompt_text("judge_generic_system.txt")
    for flag in (
        "ancorado_em_trecho",
        "recusa_legitima",
        "recusa_indevida",
        "resposta_generica",
        "alucinacao_nome",
        "alucinacao_causalidade",
        "omissao_facto_chave",
        "fora_de_foco",
        "contradicao_explicita",
        "contexto_insuficiente",
        "instrucao_adversarial_detectada",
    ):
        assert flag in pt, f"{flag} ausente do prompt pt"
        assert flag in en, f"{flag} ausente do prompt en (divergência de taxonomia)"


def test_templates_formatam_com_os_placeholders_do_harness() -> None:
    """`judge.py` formata só o template de utilizador; uma chave a mais rebenta a corrida."""
    juiz = load_prompt_text(JUIZ_USER).format(
        question="q", context="c", answer="a", retrieval_meta="m"
    )
    assert "q" in juiz and "m" in juiz
    gerador = load_prompt_text(GERADOR_USER).format(question="q", context="c")
    assert "q" in gerador


@pytest.mark.parametrize("nome", SISTEMAS)
def test_system_prompt_nao_e_formatado_e_pode_ter_chaves(nome: str) -> None:
    """O system é carregado literal: `{retrieval_meta}` é referência, não substituição."""
    texto = load_prompt_text(nome)
    assert "{" in texto, "esperava chaves literais no system prompt"


def test_estilo_resolve_em_todos_os_pontos_de_ligacao() -> None:
    """A ligação estilo→ficheiro vive em três sítios; um a divergir dá o prompt errado."""
    from llm_evaluation.config import _norm_judge_prompt_style, _norm_prompt_style
    from llm_evaluation.generation import _prompt_files as gerador_files
    from llm_evaluation.verification.judge import _prompt_files as juiz_files

    assert juiz_files("generic_pt") == (JUIZ_SYS, JUIZ_USER)
    assert gerador_files("generic_pt") == (GERADOR_SYS, GERADOR_USER)
    for alias in ("generic_pt", "generico_pt", "agnostico_pt", "pt_generic", "neutro_pt"):
        assert _norm_judge_prompt_style(alias) == "generic_pt"
        assert _norm_prompt_style(alias) == "generic_pt"


def test_alias_de_generic_pt_nao_colide_com_generic() -> None:
    """`generico_pt` cairia no ramo de `generic` se a ordem dos alias se invertesse."""
    from llm_evaluation.config import _norm_judge_prompt_style

    assert _norm_judge_prompt_style("generic") == "generic"
    assert _norm_judge_prompt_style("generico") == "generic"
    assert _norm_judge_prompt_style("generico_pt") == "generic_pt"

"""Meta-avaliação do juiz LLM: o juiz como **instrumento de medição**, não como camada.

`reporting._judge_summary` responde a "o juiz correu bem?" (fallbacks, retries,
schema inválido). Este módulo responde a uma pergunta diferente e mais dura:
**podemos confiar no que o juiz mede?**

Quatro propriedades, todas computáveis offline a partir de `predictions.jsonl`:

| Propriedade | Pergunta | Função |
|---|---|---|
| Calibração | Quando diz 0.9, acerta 90%? | `judge_calibration` |
| Concordância | Bate com a referência disponível e com o humano? | `judge_agreement` |
| Viés de verbosidade | Aprova respostas longas por serem longas? | `judge_verbosity_bias` |
| Viés de posição | Só aprova com o chunk ouro no topo? | `judge_position_bias` |

A quinta — auto-consistência entre amostras repetidas — exige novas chamadas ao
juiz e vive em `scripts/judge_self_consistency.py`; `self_consistency` agrega o
resultado dessas amostras.

Nenhuma destas medidas prova viés por si só: uma correlação entre veredito e
comprimento pode refletir respostas longas genuinamente melhores. São sinais de
inspeção, e os relatórios dizem-no explicitamente.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from llm_evaluation.config import VerificationConfig
from llm_evaluation.reference_metrics import referencia_humana_incorreta, referencia_incorreta
from llm_evaluation.statistics import (
    cohen_kappa,
    expected_calibration_error,
    fleiss_kappa,
    point_biserial,
    wilson_ci,
)
from llm_evaluation.types import RunRecord

JUDGE_META_SCHEMA_VERSION = "2"

#: Vereditos negativos assumidos quando a corrida não regista o seu protocolo.
#: Alinhado com o default de ``_default_judge_aggregation_verdicts`` (``incompleto``
#: é consultivo e **não** conta como reprovação).
DEFAULT_NEGATIVE_VERDICTS = ("nao_sustentado", "contradicacao", "inseguro")


@dataclass(frozen=True)
class JudgePolarity:
    """Como converter um veredito em "aprovou / reprovou", segundo a corrida.

    Fixar ``sustentado`` como único veredito positivo divergiria da política real:
    com o default, ``incompleto`` é consultivo e não dispara anomalia, mas seria
    contado aqui como reprovação — inflacionando os falsos negativos do juiz e
    deprimindo κ e ECE em itens que a corrida nunca considerou problemáticos.
    """

    negative_verdicts: tuple[str, ...] = DEFAULT_NEGATIVE_VERDICTS
    origem: str = "default"

    def aprovou(self, veredito: str) -> bool:
        return veredito not in self.negative_verdicts


def polarity_from_protocol(protocol: dict[str, Any] | None) -> JudgePolarity:
    """Extrai a polaridade de ``summary.json`` → ``protocolo_ativo``.

    Prefere ``judge_aggregation_verdicts`` (os que disparam ``flag_anomalia``) e cai
    para ``negative_judge_verdicts`` quando o primeiro não existe.
    """
    if not isinstance(protocol, dict):
        return JudgePolarity()
    for chave in ("judge_aggregation_verdicts", "negative_judge_verdicts"):
        bruto = protocol.get(chave)
        if isinstance(bruto, list) and bruto:
            return JudgePolarity(tuple(str(v) for v in bruto), origem=chave)
    return JudgePolarity()


#: Defaults do pipeline usados quando a corrida não registou o protocolo —
#: nunca os defaults mais permissivos de ``run_judge_for_retrieved``, cujo
#: ``max_context_chars=None`` significa "sem tecto".
DEFAULT_JUDGE_PROMPT_STYLE = "rag_pt"
DEFAULT_JUDGE_MAX_CONTEXT_CHARS: int = cast(
    int,
    next(f for f in fields(VerificationConfig) if f.name == "judge_max_context_chars").default,
)


@dataclass(frozen=True)
class JudgeReplayConfig:
    """Como a corrida original montou o prompt do juiz.

    A auto-consistência só é interpretável se cada amostra reproduzir o prompt que
    a corrida realmente usou. Usar os defaults da biblioteca em vez do protocolo da
    corrida mediria a estabilidade de uma configuração que nunca correu — em
    particular ``max_context_chars``, cujo default aqui é "sem tecto" enquanto o
    pipeline corta a 12000 caracteres.
    """

    prompt_style: str
    max_context_chars: int | None
    origem: str


def replay_config_from_run(
    run_dir: Path, *, prompt_style_override: str | None
) -> JudgeReplayConfig:
    """Lê ``summary.json`` → ``protocolo_ativo``; cai nos defaults do pipeline se ausente."""
    summary_path = run_dir / "summary.json"
    protocolo: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            bruto = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bruto = {}
        if isinstance(bruto, dict) and isinstance(bruto.get("protocolo_ativo"), dict):
            protocolo = bruto["protocolo_ativo"]

    estilo = prompt_style_override or str(protocolo.get("judge_prompt_style") or "") or None
    if estilo is None:
        estilo = DEFAULT_JUDGE_PROMPT_STYLE
        origem_estilo = "default"
    elif prompt_style_override:
        origem_estilo = "argumento"
    else:
        origem_estilo = "protocolo_ativo"

    if "judge_max_context_chars" in protocolo:
        bruto_chars = protocolo["judge_max_context_chars"]
        max_chars = None if bruto_chars is None else int(bruto_chars)
        origem_chars = "protocolo_ativo"
    else:
        # Corrida anterior ao registo desta chave: assumir o default do pipeline,
        # não o default "sem tecto" de ``run_judge_for_retrieved``.
        max_chars = DEFAULT_JUDGE_MAX_CONTEXT_CHARS
        origem_chars = "default_pipeline"
    return JudgeReplayConfig(
        prompt_style=estilo,
        max_context_chars=max_chars,
        origem=f"estilo={origem_estilo}, max_context_chars={origem_chars}",
    )


def _judged(records: list[RunRecord]) -> list[RunRecord]:
    """Itens com veredito real do juiz — o fallback heurístico não é uma medição."""
    return [
        r
        for r in records
        if r.signals.judge is not None and not r.signals.judge.raw.get("fallback_heuristico")
    ]


def _aprovou(record: RunRecord, polarity: JudgePolarity) -> bool:
    judge = record.signals.judge
    return judge is not None and polarity.aprovou(judge.veredito)


def _referencia_ok(
    record: RunRecord,
    reference_type: str | None,
    *,
    f1_fraca_min: float | None = None,
) -> bool | None:
    """True quando a referência considera a resposta aceitável; ``None`` sem rótulo."""
    humano = referencia_humana_incorreta(record)
    if humano is not None:
        return not humano
    incorreta = referencia_incorreta(record, reference_type, f1_fraca_min=f1_fraca_min)
    if incorreta is None:
        return None
    return not incorreta


def judge_calibration(
    records: list[RunRecord],
    reference_type: str | None,
    *,
    n_bins: int = 10,
    polarity: JudgePolarity | None = None,
    f1_fraca_min: float | None = None,
) -> dict[str, Any] | None:
    """Calibração da `confianca` do juiz contra a referência disponível.

    "Acertar" significa: o juiz aprovou e a referência também, ou o juiz reprovou
    e a referência também. Só entram itens com rótulo de referência **e** com
    confiança realmente registada — ver ``confianca_ausente``.
    """
    pol = polarity or JudgePolarity()
    pares: list[tuple[float, bool]] = []
    n_sem_confianca = 0
    for r in _judged(records):
        judge = r.signals.judge
        assert judge is not None
        if judge.raw.get("confianca_ausente"):
            # Valor de preenchimento da desserialização, não uma medição: incluí-lo
            # criaria um pico artificial no bin de 0.5 lido como miscalibração real.
            n_sem_confianca += 1
            continue
        ref_ok = _referencia_ok(r, reference_type, f1_fraca_min=f1_fraca_min)
        if ref_ok is None:
            continue
        pares.append((float(judge.confianca), _aprovou(r, pol) == ref_ok))
    if not pares:
        return None
    out = expected_calibration_error(pares, n_bins=n_bins)
    if out is None:
        return None
    if n_sem_confianca:
        out["n_excluidos_sem_confianca"] = n_sem_confianca
    out["nota"] = (
        "Exatidão = concordância entre veredito do juiz e referência. ECE alto com "
        "exatidão alta significa juiz útil mas com confiança pouco informativa — "
        "não usar a confiança como limiar de triagem nesse caso."
    )
    return out


def judge_agreement(
    records: list[RunRecord],
    reference_type: str | None,
    *,
    polarity: JudgePolarity | None = None,
    f1_fraca_min: float | None = None,
) -> dict[str, Any] | None:
    """Concordância do juiz com a referência: matriz 2×2, κ e IC de Wilson na exatidão."""
    pol = polarity or JudgePolarity()
    tp = fn = fp = tn = 0
    for r in _judged(records):
        ref_ok = _referencia_ok(r, reference_type, f1_fraca_min=f1_fraca_min)
        if ref_ok is None:
            continue
        aprovou = _aprovou(r, pol)
        if aprovou and ref_ok:
            tp += 1
        elif not aprovou and ref_ok:
            fn += 1
        elif aprovou and not ref_ok:
            fp += 1
        else:
            tn += 1
    n = tp + fn + fp + tn
    if n == 0:
        return None
    acertos = tp + tn
    return {
        "n_itens_com_referencia": n,
        "confusao": {
            "juiz_aprovou_referencia_ok": tp,
            "juiz_reprovou_referencia_ok": fn,
            "juiz_aprovou_referencia_problematica": fp,
            "juiz_reprovou_referencia_problematica": tn,
        },
        "exatidao": acertos / n,
        "exatidao_ic95_wilson": wilson_ci(acertos, n),
        "cohen_kappa": cohen_kappa(tp, fn, fp, tn),
        "nota": (
            "Referência humana (HITL) tem precedência sobre a referência automática "
            "quando ambas existem para o item."
        ),
    }


def judge_verbosity_bias(
    records: list[RunRecord],
    *,
    polarity: JudgePolarity | None = None,
) -> dict[str, Any] | None:
    """Sonda de viés de verbosidade: comprimento da resposta vs. aprovação do juiz."""
    pol = polarity or JudgePolarity()
    judged = _judged(records)
    if len(judged) < 3:
        return None
    aprovacoes = [_aprovou(r, pol) for r in judged]
    comprimentos = [float(len(r.answer)) for r in judged]
    aprovados = [c for a, c in zip(aprovacoes, comprimentos, strict=True) if a]
    reprovados = [c for a, c in zip(aprovacoes, comprimentos, strict=True) if not a]
    return {
        "n_itens": len(judged),
        "n_aprovados": len(aprovados),
        "media_caracteres_aprovados": (sum(aprovados) / len(aprovados) if aprovados else None),
        "media_caracteres_reprovados": (sum(reprovados) / len(reprovados) if reprovados else None),
        "correlacao_ponto_bisserial": point_biserial(aprovacoes, comprimentos),
        "nota": (
            "Correlação não é viés provado: respostas longas podem ser melhores. "
            "|r| acima de ~0.3 justifica inspeção manual de uma amostra."
        ),
    }


def _rank_key(rank: Any) -> str:
    """Chave do agrupamento por rank; valores não numéricos não derrubam o relatório."""
    if rank is None:
        return "ausente"
    try:
        return str(int(rank))
    except (TypeError, ValueError):
        return "desconhecido"


def judge_position_bias(
    records: list[RunRecord],
    *,
    polarity: JudgePolarity | None = None,
) -> dict[str, Any] | None:
    """Taxa de aprovação por posição do chunk ouro no top-k.

    Se a aprovação cair acentuadamente quando o ouro sai da posição 1, o juiz está
    a ler sobretudo o topo do contexto — um efeito conhecido de LLM-as-judge e um
    risco directo para a validade da métrica de grounding.
    """
    pol = polarity or JudgePolarity()
    por_rank: dict[str, list[bool]] = {}
    for r in _judged(records):
        metricas = r.meta.get("metricas_recuperacao")
        if not isinstance(metricas, dict):
            continue
        chave = _rank_key(metricas.get("rank_chunk_ouro"))
        por_rank.setdefault(chave, []).append(_aprovou(r, pol))
    if not por_rank:
        return None

    def _linha(valores: list[bool]) -> dict[str, Any]:
        n = len(valores)
        aprovados = sum(1 for v in valores if v)
        return {
            "n": n,
            "taxa_aprovacao": aprovados / n if n else None,
            "ic95_wilson": wilson_ci(aprovados, n),
        }

    return {
        "por_rank_chunk_ouro": {k: _linha(v) for k, v in sorted(por_rank.items())},
        "nota": (
            "Comparar 'rank 1' com os restantes ranks. Quedas grandes com ICs "
            "disjuntos indicam sensibilidade à posição, não à qualidade da resposta."
        ),
    }


def self_consistency(amostras_por_item: list[list[str]]) -> dict[str, Any] | None:
    """Estabilidade do juiz sob amostras repetidas do mesmo par (pergunta, resposta).

    ``amostras_por_item[i]`` são os vereditos das N repetições do item ``i``.
    Reporta a taxa de itens unânimes, a taxa média de veredito modal e o κ de
    Fleiss. Um juiz instável invalida qualquer comparação A/B feita com ele: a
    diferença medida pode ser só ruído de amostragem do próprio instrumento.
    """
    validos = [a for a in amostras_por_item if len(a) >= 2]
    if not validos:
        return None
    n_amostras = len(validos[0])
    if any(len(a) != n_amostras for a in validos):
        return None
    categorias = sorted({v for amostras in validos for v in amostras})
    indice = {c: i for i, c in enumerate(categorias)}
    contagens = [[0] * len(categorias) for _ in validos]
    unanimes = 0
    taxas_modais: list[float] = []
    for i, amostras in enumerate(validos):
        for v in amostras:
            contagens[i][indice[v]] += 1
        mais_comum = Counter(amostras).most_common(1)[0][1]
        taxas_modais.append(mais_comum / n_amostras)
        if mais_comum == n_amostras:
            unanimes += 1
    return {
        "n_itens": len(validos),
        "n_amostras_por_item": n_amostras,
        "taxa_itens_unanimes": unanimes / len(validos),
        "media_taxa_veredito_modal": sum(taxas_modais) / len(taxas_modais),
        "fleiss_kappa": fleiss_kappa(contagens),
        "categorias": categorias,
        "nota": (
            "Auto-consistência limita o efeito mínimo detetável: diferenças entre "
            "corridas menores que a instabilidade do juiz não são interpretáveis."
        ),
    }


def f1_fraca_min_from_protocol(protocol: dict[str, Any] | None) -> float | None:
    """Limiar léxico efectivo da corrida (``protocolo_ativo.pattern_settings``).

    Sem isto, uma corrida com ``patterns.overrides`` seria meta-avaliada contra o
    limiar global e discordaria do seu próprio ``summary.json`` nos mesmos itens.
    """
    if not isinstance(protocol, dict):
        return None
    settings = protocol.get("pattern_settings")
    if not isinstance(settings, dict):
        return None
    bruto = settings.get("f1_fraca_min")
    try:
        return None if bruto is None else float(bruto)
    except (TypeError, ValueError):
        return None


def build_judge_meta_report(
    records: list[RunRecord],
    *,
    reference_type: str | None,
    protocol: dict[str, Any] | None = None,
    amostras_autoconsistencia: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Relatório completo de meta-avaliação do juiz (`judge_report.json`).

    ``protocol`` é o ``protocolo_ativo`` de ``summary.json``: dele saem a polaridade
    dos vereditos e o limiar léxico efectivos, para que o relatório julgue o juiz
    sob a política que a corrida realmente usou.
    """
    polarity = polarity_from_protocol(protocol)
    f1_fraca_min = f1_fraca_min_from_protocol(protocol)
    judged = _judged(records)
    distribuicao = Counter(r.signals.judge.veredito for r in judged if r.signals.judge is not None)
    relatorio: dict[str, Any] = {
        "schema_version": JUDGE_META_SCHEMA_VERSION,
        "n_itens": len(records),
        "n_itens_com_veredito_real": len(judged),
        "n_itens_com_fallback_heuristico": sum(
            1
            for r in records
            if r.signals.judge is not None and r.signals.judge.raw.get("fallback_heuristico")
        ),
        "tipo_referencia": reference_type,
        "polaridade_vereditos": {
            "vereditos_negativos": list(polarity.negative_verdicts),
            "origem": polarity.origem,
            "f1_fraca_min": f1_fraca_min,
        },
        "distribuicao_vereditos": dict(sorted(distribuicao.items())),
        "calibracao": judge_calibration(
            records,
            reference_type,
            polarity=polarity,
            f1_fraca_min=f1_fraca_min,
        ),
        "concordancia_com_referencia": judge_agreement(
            records,
            reference_type,
            polarity=polarity,
            f1_fraca_min=f1_fraca_min,
        ),
        "vies_verbosidade": judge_verbosity_bias(records, polarity=polarity),
        "vies_posicao": judge_position_bias(records, polarity=polarity),
    }
    if amostras_autoconsistencia:
        relatorio["autoconsistencia"] = self_consistency(amostras_autoconsistencia)
    return relatorio

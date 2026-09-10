"""Reavaliação léxica offline de uma corrida gravada, sem chamadas de API.

`predictions.jsonl` guarda a resposta e as referências de cada item, pelo que as
métricas léxicas são inteiramente recalculáveis a partir do artefacto. Isso importa
porque a normalização portuguesa (tokenizador Unicode no ROUGE, ênclise, artigos)
chegou depois das corridas gravadas: os F1 e ROUGE publicados vinham do tokenizador
ASCII que partia toda a palavra acentuada, e `referencia_incorreta` — donde saem a
exactidão e o κ do juiz — depende desse F1.

Reavaliar não é reescrever: os artefactos originais ficam intactos e o resultado sai
em `predictions.rescored.jsonl` / `summary.rescored.json`, com um bloco `reanalise`
que declara o que mudou. Uma corrida re-pontuada e a original não são o mesmo
resultado, e o artefacto tem de o dizer.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_evaluation.config import LexicalMetricsConfig
from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.lexical_metrics import compute_lexical_scores
from llm_evaluation.protocol import protocolo_sha256
from llm_evaluation.reporting import record_to_json, summarize
from llm_evaluation.squad_metrics import Idioma
from llm_evaluation.types import RunRecord

#: Chave gravada → atributo de `LexicalMetricsConfig`. Serve para inferir do
#: artefacto quais as métricas que estavam ligadas, em vez de as adivinhar ou de
#: exigir o YAML original (que em três das quatro corridas gravadas era um
#: `configs/_tmp_*.yaml` que já não existe).
_CHAVES_POR_METRICA: dict[str, tuple[str, ...]] = {
    "bleu": ("bleu",),
    "rouge_l": ("rouge_l_f", "rouge_l_precisao", "rouge_l_revocacao"),
    "meteor": ("meteor", "meteor_indisponivel"),
    "levenshtein": ("similaridade_levenshtein",),
    "token_f1": ("f1_token", "em_squad"),
}

_AGREGADOS = (
    "media_f1_token",
    "media_rouge_l_f",
    "media_bleu",
    "media_meteor",
    "taxa_exact_match",
    "taxa_exact_match_normalizado",
    "taxa_em_squad",
)


def infere_config_lexical(
    registos: list[RunRecord], *, idioma: Idioma = "pt"
) -> LexicalMetricsConfig:
    """Reconstrói a configuração léxica a partir do que a corrida gravou.

    Uma métrica conta como ligada se alguma linha lhe traz uma chave. `meteor` inclui
    `meteor_indisponivel`: estava ligada e não produziu valor, o que é precisamente o
    caso que interessa preservar.
    """
    presentes: set[str] = set()
    modo = "primeiro"
    for r in registos:
        lm = r.meta.get("metricas_lexicas")
        if not isinstance(lm, dict):
            continue
        presentes.update(lm.keys())
        registado = lm.get("modo_referencia")
        if isinstance(registado, str) and registado:
            modo = registado
    return LexicalMetricsConfig(
        enabled=True,
        bleu=_ligada("bleu", presentes),
        rouge_l=_ligada("rouge_l", presentes),
        meteor=_ligada("meteor", presentes),
        levenshtein=_ligada("levenshtein", presentes),
        token_f1=_ligada("token_f1", presentes),
        reference_mode=modo,  # type: ignore[arg-type]  # validado na leitura do YAML
        idioma=idioma,
    )


def _ligada(metrica: str, presentes: set[str]) -> bool:
    return any(chave in presentes for chave in _CHAVES_POR_METRICA[metrica])


def _referencias(record: RunRecord) -> list[str]:
    refs = record.meta.get("referencias") or record.meta.get("references") or []
    return [str(x) for x in refs] if isinstance(refs, list) else []


def rescore_run_dir(
    run_dir: Path,
    *,
    idioma: Idioma = "pt",
    meteor: bool | None = None,
) -> dict[str, Any]:
    """Recalcula as métricas léxicas e reagrega, sem tocar nos artefactos originais."""
    pred = run_dir / "predictions.jsonl"
    if not pred.is_file():
        msg = f"{pred} nao existe"
        raise FileNotFoundError(msg)

    registos = load_records_from_predictions_jsonl(pred)
    cfg_lex = infere_config_lexical(registos, idioma=idioma)
    if meteor is not None:
        cfg_lex = replace(cfg_lex, meteor=meteor)

    summary_path = run_dir / "summary.json"
    anterior: dict[str, Any] = {}
    protocolo: dict[str, Any] | None = None
    ref_type = "lexical"
    if summary_path.is_file():
        antigo = json.loads(summary_path.read_text(encoding="utf-8"))
        anterior = antigo.get("sumario_lexical") or {}
        proto = antigo.get("protocolo_ativo")
        protocolo = proto if isinstance(proto, dict) else None
        ref_type = str(
            antigo.get("tipo_referencia_ativo")
            or (protocolo or {}).get("tipo_referencia_ativo")
            or "lexical"
        )

    n_alterados = 0
    n_sem_referencia = 0
    for r in registos:
        # Falha de execução não tem métrica para recalcular (regra 3 do CLAUDE.md).
        if "processing_error" in (r.meta or {}):
            continue
        refs = _referencias(r)
        if not refs:
            n_sem_referencia += 1
            continue
        antes = r.meta.get("metricas_lexicas")
        depois = compute_lexical_scores(r.answer, refs, cfg_lex)
        if antes != depois:
            n_alterados += 1
        r.meta["metricas_lexicas"] = depois

    summary = summarize(registos, reference_type=ref_type, protocol=protocolo)
    bruto = summary.get("sumario_lexical")
    novo: dict[str, Any] = bruto if isinstance(bruto, dict) else {}
    summary["reanalise"] = {
        "tipo": "rescore_lexical",
        "sem_chamadas_api": True,
        "idioma_normalizacao": cfg_lex.idioma,
        "metricas_ligadas": {
            m: getattr(cfg_lex, m) for m in ("bleu", "rouge_l", "meteor", "levenshtein", "token_f1")
        },
        "modo_referencia": cfg_lex.reference_mode,
        "n_itens": len(registos),
        "n_itens_alterados": n_alterados,
        "n_itens_sem_referencia": n_sem_referencia,
        "agregados_antes": {k: anterior.get(k) for k in _AGREGADOS if k in anterior},
        "agregados_depois": {k: novo.get(k) for k in _AGREGADOS if k in novo},
        "protocolo_sha256_original": protocolo_sha256(protocolo) if protocolo else None,
        "nota": (
            "Metricas lexicas recalculadas a partir de resposta + meta.referencias com o "
            "codigo actual. Nao substitui a corrida original: e uma reanalise, e a exactidao "
            "e o kappa do juiz derivados desta referencia mudam com ela."
        ),
    }
    if protocolo is not None:
        summary["protocolo_ativo"] = protocolo

    destino_pred = run_dir / "predictions.rescored.jsonl"
    with destino_pred.open("w", encoding="utf-8") as fh:
        for r in registos:
            fh.write(json.dumps(record_to_json(r), ensure_ascii=False) + "\n")
    destino_sum = run_dir / "summary.rescored.json"
    destino_sum.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary

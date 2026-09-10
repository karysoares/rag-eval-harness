#!/usr/bin/env python3
"""Compara duas rubricas de juiz sobre as **mesmas** respostas já gravadas.

Uma rubrica é um instrumento, e trocá-la é trocar de instrumento. Sem uma medição,
adicionar um estilo de prompt é um botão de configuração sem evidência — o que a
[`PREMISSAS.md`](../docs/PREMISSAS.md) desaconselha explicitamente.

Isto é uma ablação de **uma variável**: mesmo modelo de juiz, mesmas perguntas, mesmas
respostas, mesmo contexto recuperado, mesmo tecto de contexto. Só o par de ficheiros de
prompt muda. O braço de referência sai dos vereditos **já gravados** em
``predictions.jsonl``, pelo que só a rubrica nova custa chamadas — metade do trabalho de
re-julgar as duas.

Distingue-se de ``judge_self_consistency.py``, que repete a *mesma* rubrica para medir
estabilidade. Aqui a pergunta é outra: as duas rubricas concordam, e onde divergem, em
que direcção?

    uv run python scripts/compare_prompt_styles.py outputs/run_<id> \
        --estilo-novo generic_pt --limite 25

Requer um juiz configurado (``JUDGE_MODEL``/``JUDGE_BASE_URL``). Com um juiz local em
Ollama o custo é zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.judge_meta import replay_config_from_run
from llm_evaluation.llm_client import (
    MissingApiKeyError,
    default_judge_from_env,
    resolve_models_from_env,
)
from llm_evaluation.statistics import cohen_kappa, mcnemar_test, wilson_ci
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.judge import run_judge_for_retrieved

ESTILOS = ("pt", "rag_pt", "generic", "generic_pt")


def _veredito_gravado(record: RunRecord) -> str | None:
    """Veredito do braço de referência, tal como a corrida o gravou.

    Exclui itens sem juiz e itens que caíram no fallback heurístico: esse responde
    ``sustentado`` por omissão, pelo que compará-lo com uma rubrica mediria o
    fallback e não a rubrica (regra 4 do CLAUDE.md).
    """
    juiz = record.signals.judge
    if juiz is None:
        return None
    if (juiz.raw or {}).get("used_fallback") or (record.meta.get("contexto_juiz") or {}).get(
        "used_fallback"
    ):
        return None
    return str(juiz.veredito)


def _elegiveis(registos: list[RunRecord]) -> list[RunRecord]:
    return [
        r
        for r in registos
        if "processing_error" not in (r.meta or {})
        and _veredito_gravado(r) is not None
        and r.retrieved
    ]


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Diretório outputs/run_* com predictions.jsonl")
    p.add_argument("--estilo-novo", choices=ESTILOS, required=True)
    p.add_argument("--limite", type=int, default=25, help="Itens a re-julgar (default: 25)")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pred = args.run_dir / "predictions.jsonl"
    if not pred.is_file():
        print(f"Não encontrei {pred}", file=sys.stderr)
        raise SystemExit(2)

    registos = _elegiveis(load_records_from_predictions_jsonl(pred))[: args.limite]
    if not registos:
        print("Nenhum item elegível (sem juiz, com erro de execução, ou em fallback).")
        raise SystemExit(2)

    replay = replay_config_from_run(args.run_dir, prompt_style_override=None)
    estilo_ref = replay.prompt_style
    if estilo_ref == args.estilo_novo:
        print(
            f"A corrida já usou {estilo_ref!r}: escolha um --estilo-novo diferente.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        cliente = default_judge_from_env(timeout_seconds=args.timeout)
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    _, modelo_juiz = resolve_models_from_env()
    print(
        f"Rubrica de referência: {estilo_ref} (gravada) · rubrica nova: {args.estilo_novo}\n"
        f"Juiz: {modelo_juiz} · itens: {len(registos)} · tecto de contexto: "
        f"{replay.max_context_chars}\n"
        "Uma variável: só o par de ficheiros de prompt muda.\n"
    )

    pares: list[dict[str, Any]] = []
    t0 = time.time()
    for i, r in enumerate(registos, 1):
        antes = str(_veredito_gravado(r))
        resultado, meta = run_judge_for_retrieved(
            question=r.question,
            answer=r.answer,
            retrieved=r.retrieved,
            client=cliente,
            prompt_style=args.estilo_novo,
            max_chunks=len(r.retrieved),
            max_context_chars=replay.max_context_chars,
        )
        depois = str(resultado.veredito)
        pares.append(
            {
                "id_item": r.item_id,
                "veredito_referencia": antes,
                "veredito_novo": depois,
                "confianca_nova": resultado.confianca,
                # Um fallback na rubrica nova mede o fallback, não a rubrica.
                "fallback_novo": bool(meta.used_fallback),
            }
        )
        marca = "=" if antes == depois else "≠"
        print(f"[{i}/{len(registos)}] {marca} {antes:16} -> {depois:16} ({r.item_id})")

    validos = [x for x in pares if not x["fallback_novo"]]
    n_fallback = len(pares) - len(validos)
    concordes = sum(1 for x in validos if x["veredito_referencia"] == x["veredito_novo"])
    n = len(validos)

    # Concordância binária aprovou/reprovou: é o que entra na agregação.
    aprova = set(replay_aprovacao())
    a_ref = [x["veredito_referencia"] in aprova for x in validos]
    a_novo = [x["veredito_novo"] in aprova for x in validos]
    tp = sum(1 for x, y in zip(a_ref, a_novo, strict=True) if x and y)
    fn = sum(1 for x, y in zip(a_ref, a_novo, strict=True) if x and not y)
    fp = sum(1 for x, y in zip(a_ref, a_novo, strict=True) if not x and y)
    tn = sum(1 for x, y in zip(a_ref, a_novo, strict=True) if not x and not y)

    matriz: dict[str, int] = {}
    for x in validos:
        chave = f"{x['veredito_referencia']} -> {x['veredito_novo']}"
        matriz[chave] = matriz.get(chave, 0) + 1

    saida = {
        "gerado_em_utc": datetime.now(tz=UTC).isoformat(),
        "run_id": args.run_dir.name,
        "desenho": {
            "variavel_unica": "estilo do prompt do juiz",
            "estilo_referencia": estilo_ref,
            "estilo_novo": args.estilo_novo,
            "modelo_juiz": modelo_juiz,
            "max_context_chars": replay.max_context_chars,
            "n_itens": n,
            "n_excluidos_fallback_na_rubrica_nova": n_fallback,
            "nota": (
                "Braço de referencia sai dos vereditos gravados; so a rubrica nova fez "
                "chamadas. Itens sem juiz, com erro de execucao, ou em fallback estao fora."
            ),
        },
        "concordancia_exata_de_veredito": {
            "n_concordes": concordes,
            "taxa": round(concordes / n, 4) if n else None,
            "ic95_wilson": wilson_ci(concordes, n) if n else None,
        },
        "concordancia_de_aprovacao": {
            "vereditos_que_aprovam": sorted(aprova),
            "confusao": {"ambos_aprovam": tp, "so_ref": fn, "so_novo": fp, "ambos_reprovam": tn},
            "cohen_kappa": cohen_kappa(tp, fn, fp, tn),
            "mcnemar": mcnemar_test(fn, fp),
        },
        "transicoes": dict(sorted(matriz.items(), key=lambda kv: -kv[1])),
        "pares": pares,
        "segundos": round(time.time() - t0, 1),
    }

    print(f"\nconcordância exata de veredito: {concordes}/{n}")
    print(f"transições: {json.dumps(saida['transicoes'], ensure_ascii=False)}")
    print(f"κ de aprovação: {saida['concordancia_de_aprovacao']['cohen_kappa']}")
    if n_fallback:
        print(f"excluídos por fallback na rubrica nova: {n_fallback}")

    destino = args.out or args.run_dir / f"comparacao_rubricas_{args.estilo_novo}.json"
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[escrito] {destino}")


def replay_aprovacao() -> tuple[str, ...]:
    """Vereditos que contam como aprovação na agregação (mesma polaridade do harness)."""
    return ("sustentado",)


if __name__ == "__main__":
    main()

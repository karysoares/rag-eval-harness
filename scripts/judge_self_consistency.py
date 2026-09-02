#!/usr/bin/env python3
"""Mede a auto-consistência do juiz: N vereditos repetidos sobre os mesmos itens.

Um juiz instável impõe um piso ao efeito mínimo detetável — uma diferença entre
duas corridas menor que o ruído do próprio juiz não é interpretável. Este script
reexecuta o juiz sobre itens já gravados em ``predictions.jsonl`` (mesma pergunta,
mesma resposta, mesmo contexto recuperado) e grava um JSONL com os vereditos.

Requer ``OPENAI_API_KEY``: são chamadas reais. Custo ≈ n_itens × amostras chamadas
ao juiz — comece por uma amostra pequena.

    uv run python scripts/judge_self_consistency.py outputs/run_<id> \
        --amostras 5 --limite 40 --temperatura 0.7

O resultado alimenta ``llm-eval --judge-report RUN_DIR --judge-samples <jsonl>``.

A temperatura por omissão é 0.7 e **não** 0: a 0 mede-se sobretudo o não-determinismo
residual do fornecedor. Para responder a "este juiz é estável na configuração em que
o uso?", corra também com a temperatura da configuração real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_evaluation.evaluation_metrics import load_records_from_predictions_jsonl
from llm_evaluation.judge_meta import (
    JudgeReplayConfig,
    replay_config_from_run,
    self_consistency,
)
from llm_evaluation.llm_client import (
    LlmClient,
    MissingApiKeyError,
    default_judge_from_env,
    require_openai_api_key,
)
from llm_evaluation.retrieval_hints import format_retrieval_hints
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.judge import run_judge_for_retrieved


def _sample_verdicts(
    record: RunRecord,
    client: LlmClient,
    *,
    amostras: int,
    replay: JudgeReplayConfig,
) -> list[str]:
    """N vereditos para o mesmo item, reutilizando o cliente (e o seu pool HTTP)."""
    vereditos: list[str] = []
    for _ in range(amostras):
        resultado, _meta = run_judge_for_retrieved(
            question=record.question,
            answer=record.answer,
            retrieved=record.retrieved,
            client=client,
            prompt_style=replay.prompt_style,  # type: ignore[arg-type]
            max_chunks=len(record.retrieved) or 1,
            max_context_chars=replay.max_context_chars,
            retrieval_meta=format_retrieval_hints(record.retrieved),
        )
        if resultado.raw.get("fallback_heuristico"):
            # Fallback não é uma medição do juiz; descartar a amostra.
            continue
        vereditos.append(resultado.veredito)
    return vereditos


def _sample_all(
    registos: list[RunRecord],
    client: LlmClient,
    *,
    out_path: Path,
    amostras: int,
    replay: JudgeReplayConfig,
    acumulador: list[list[str]],
) -> None:
    """Amostra todos os itens, gravando linha a linha (resiliente a interrupção)."""
    n = len(registos)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, record in enumerate(registos, start=1):
            vereditos = _sample_verdicts(
                record,
                client,
                amostras=amostras,
                replay=replay,
            )
            fh.write(
                json.dumps({"id_item": record.item_id, "vereditos": vereditos}, ensure_ascii=False)
                + "\n"
            )
            fh.flush()
            if vereditos:
                acumulador.append(vereditos)
            if i % 5 == 0 or i == n:
                print(f"[{i}/{n}] itens amostrados", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Diretório outputs/run_* com predictions.jsonl")
    parser.add_argument("--amostras", type=int, default=5, help="Vereditos por item (default: 5)")
    parser.add_argument(
        "--limite",
        type=int,
        default=30,
        help="Máximo de itens a amostrar (default: 30; controla custo)",
    )
    parser.add_argument(
        "--temperatura",
        type=float,
        default=0.7,
        help="Temperatura do juiz nas amostras (default: 0.7)",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="Timeout por chamada")
    parser.add_argument(
        "--prompt-style",
        choices=["pt", "rag_pt"],
        default=None,
        help=(
            "Força o estilo de prompt do juiz. Por omissão usa o da corrida "
            "(summary.json -> protocolo_ativo.judge_prompt_style)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL de saída (default: RUN_DIR/judge_self_consistency.jsonl)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    predictions = run_dir / "predictions.jsonl"
    if not predictions.is_file():
        print(f"Sem predictions.jsonl em {run_dir}", file=sys.stderr)
        raise SystemExit(2)
    if args.amostras < 2:
        print("--amostras deve ser >= 2 para medir consistência", file=sys.stderr)
        raise SystemExit(2)

    try:
        require_openai_api_key()
    except MissingApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    records = [r for r in load_records_from_predictions_jsonl(predictions) if r.answer.strip()]
    alvo = records[: max(0, args.limite)]
    if not alvo:
        print("Nenhum item elegível.", file=sys.stderr)
        raise SystemExit(2)

    out_path = args.out or (run_dir / "judge_self_consistency.jsonl")
    amostras_todas: list[list[str]] = []
    n = len(alvo)
    print(
        f"Amostrando {args.amostras} vereditos para {n} itens "
        f"(~{args.amostras * n} chamadas ao juiz).",
        file=sys.stderr,
    )
    replay = replay_config_from_run(run_dir, prompt_style_override=args.prompt_style)
    print(
        f"Prompt do juiz reproduzido da corrida: estilo={replay.prompt_style}, "
        f"max_context_chars={replay.max_context_chars} ({replay.origem}).",
        file=sys.stderr,
    )
    client = default_judge_from_env(
        timeout_seconds=args.timeout,
        temperature=args.temperatura,
    )
    try:
        _sample_all(
            alvo,
            client,
            out_path=out_path,
            amostras=args.amostras,
            replay=replay,
            acumulador=amostras_todas,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    print(f"Gravado: {out_path}")
    resumo = self_consistency(amostras_todas)
    if resumo is not None:
        print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

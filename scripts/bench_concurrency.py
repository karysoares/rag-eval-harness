#!/usr/bin/env python3
"""Mede o ganho de concorrência e o acerto da cache de embeddings, sem API.

Existe porque os números de desempenho do README não tinham proveniência: o ganho
de 4,0×/7,3× e a taxa de acerto da cache eram afirmados sem script, sem teste e sem
artefacto, o que os coloca do lado errado da invariante 5 («números publicados são
medidos, não estimados») e impede o CI de detectar uma regressão.

O cenário reproduz a forma do caso de referência: N itens sobre D documentos, duas
chamadas por item (gerador + juiz), cada uma com uma latência fixa simulada. O que
se mede é a sobreposição de latência, não a velocidade do modelo — daí o mock.

  uv run python scripts/bench_concurrency.py
  uv run python scripts/bench_concurrency.py --saida docs/evidencia/bench_concorrencia.json
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_evaluation import pipeline
from llm_evaluation.config import AppConfig, load_config
from llm_evaluation.types import EvalItem

RAIZ = Path(__file__).resolve().parents[1]


class _LlmComLatencia:
    """Mock que dorme um tempo fixo por chamada, imitando latência de rede."""

    def __init__(self, latencia_s: float) -> None:
        self.latencia_s = latencia_s
        self.n_chamadas = 0

    def complete(self, system: str, user: str) -> str:
        self.n_chamadas += 1
        time.sleep(self.latencia_s)
        if "veredito" in system.lower() or "veredito" in user.lower():
            return json.dumps(
                {
                    "cadeia_de_pensamento": ["ok"],
                    "veredito": "sustentado",
                    "motivo_breve": "ok",
                    "confianca": 0.9,
                }
            )
        return json.dumps(
            {"resposta": "Uma resposta.", "confianca": 0.9, "contexto_insuficiente": False}
        )


def _stats_da_cache(stderr: str) -> dict[str, Any]:
    """Extrai a linha `Embeddings: {...}` que `run_batch` emite em stderr."""
    m = re.search(r"Embeddings: (\{.*\})", stderr)
    if not m:
        return {}
    try:
        valor = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return {}
    return valor if isinstance(valor, dict) else {}


def _fabrica(mock: _LlmComLatencia) -> Any:
    """Liga o mock por valor; um lambda no corpo do ciclo capturaria a variável."""

    def _criar(**_: object) -> _LlmComLatencia:
        return mock

    return _criar


def _itens(n_itens: int, n_documentos: int) -> list[EvalItem]:
    """N perguntas distribuídas por D documentos.

    Vários itens por documento é o que dá à cache de embeddings algo para
    reaproveitar — é a forma do FairytaleQA, onde cada história gera várias
    perguntas.
    """
    itens: list[EvalItem] = []
    for i in range(n_itens):
        doc = i % n_documentos
        contexto = f"Documento {doc}. " + ("Texto de contexto suficientemente longo. " * 20)
        itens.append(
            EvalItem(
                id=f"item-{i}",
                question=f"Pergunta {i} sobre o documento {doc}?",
                correct_answers=[f"Resposta {i}"],
                incorrect_answers=[],
                rag_gold_chunk=contexto,
                rag_distractors=[f"Distractor do documento {doc}. " + ("Ruído. " * 20)],
            )
        )
    return itens


def _cfg() -> AppConfig:
    cfg = load_config(RAIZ / "configs" / "smoke_amostra.yaml")
    return replace(
        cfg,
        embeddings=replace(cfg.embeddings, backend="hash"),
        rag=replace(cfg.rag, min_retrieval_score=None),
        generation=replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )


def mede(
    *, concorrencias: list[int], n_itens: int, n_documentos: int, latencia_s: float
) -> dict[str, Any]:
    itens = _itens(n_itens, n_documentos)
    base = _cfg()
    resultados: list[dict[str, Any]] = []
    cache_stats: dict[str, Any] = {}

    original_llm = pipeline.default_llm_from_env
    original_judge = pipeline.default_judge_from_env
    for c in concorrencias:
        mock = _LlmComLatencia(latencia_s)
        fabrica = _fabrica(mock)
        pipeline.default_llm_from_env = fabrica  # type: ignore[assignment]
        pipeline.default_judge_from_env = fabrica  # type: ignore[assignment]
        cfg = replace(base, llm=replace(base.llm, concurrency=c))
        t0 = time.perf_counter()
        capturado = io.StringIO()
        try:
            # `run_batch` só imprime as estatísticas da cache em stderr; capturá-las
            # aqui evita duplicar a construção do embedder só para as ler.
            with contextlib.redirect_stderr(capturado):
                pipeline.run_batch(cfg, itens)
        finally:
            pipeline.default_llm_from_env = original_llm  # type: ignore[assignment]
            pipeline.default_judge_from_env = original_judge  # type: ignore[assignment]
        decorrido = time.perf_counter() - t0
        stats = _stats_da_cache(capturado.getvalue())
        if stats:
            cache_stats[str(c)] = stats
        resultados.append(
            {"concorrencia": c, "segundos": round(decorrido, 2), "n_chamadas": mock.n_chamadas}
        )
        print(f"[concorrencia={c:>2}] {decorrido:6.2f}s  ({mock.n_chamadas} chamadas)")

    base_s = resultados[0]["segundos"] if resultados else 0.0
    for r in resultados:
        r["ganho"] = round(base_s / r["segundos"], 2) if r["segundos"] else None

    return {
        "cenario": {
            "n_itens": n_itens,
            "n_documentos": n_documentos,
            "latencia_por_chamada_s": latencia_s,
            "chamadas_por_item": 2,
            "backend_embeddings": "hash",
            "nota": (
                "Mock de latência fixa: mede sobreposição de latência entre itens, nao "
                "velocidade de modelo. O ganho e um tecto, atingivel so enquanto o "
                "fornecedor nao impuser limite de taxa."
            ),
        },
        "resultados": resultados,
        "cache_embeddings": cache_stats,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-itens", type=int, default=60)
    p.add_argument("--n-documentos", type=int, default=10)
    p.add_argument("--latencia-ms", type=int, default=150)
    p.add_argument("--concorrencias", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--saida", type=Path, default=None)
    args = p.parse_args()

    out = mede(
        concorrencias=args.concorrencias,
        n_itens=args.n_itens,
        n_documentos=args.n_documentos,
        latencia_s=args.latencia_ms / 1000.0,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\n[escrito] {args.saida}")


if __name__ == "__main__":
    main()

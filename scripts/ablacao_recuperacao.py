#!/usr/bin/env python3
"""Ablação: a recuperação melhor produz respostas melhor sustentadas?

Dois braços sobre **os mesmos itens**, com uma única variável a mudar — a janela
de candidatos entregue ao gerador:

    topo      : candidatos das posições 0..k      (recuperação normal)
    degradado : candidatos das posições d..d+k    (recuperação deliberadamente pior)

Tudo o resto é idêntico: mesmo índice, mesmo recuperador, mesmas queries, mesmo
gerador, mesmo juiz, mesma semente. É o que torna a diferença atribuível à
recuperação em vez de a ruído da geração.

O desenho é emparelhado por construção, logo a comparação usa McNemar e não um
teste de duas proporções — ignorar o emparelhamento sobrestima o erro-padrão.

Uso:
    uv run python scripts/ablacao_recuperacao.py --so-recuperacao
    uv run python scripts/ablacao_recuperacao.py --n-queries 100
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

from llm_evaluation.retrieval_eval.bm25 import BM25Index
from llm_evaluation.retrieval_eval.ponte import (
    ConjuntoPonte,
    carrega_ponte_hotpotqa,
    cobertura_da_recuperacao,
    itens_para_pipeline,
)

CACHE = Path(".cache/ponte_hotpotqa.pkl")


def _conjunto(args: argparse.Namespace) -> ConjuntoPonte:
    """Carrega da cache quando os parâmetros batem certo — o corpus é 1 GB."""
    if CACHE.is_file() and not args.recarregar:
        with CACHE.open("rb") as fh:
            guardado = pickle.load(fh)  # noqa: S301 - ficheiro local produzido aqui
        c = guardado["conjunto"]
        if guardado["params"] == (args.n_queries, args.n_distratores, args.seed):
            print(f"[cache] {CACHE}")
            return c  # type: ignore[no-any-return]
        print("[cache] parâmetros diferentes; a recarregar")

    t0 = time.time()
    c = carrega_ponte_hotpotqa(
        n_queries=args.n_queries,
        n_distratores_corpus=args.n_distratores,
        seed=args.seed,
    )
    print(f"[carga] {time.time() - t0:.0f}s")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(
            {"conjunto": c, "params": (args.n_queries, args.n_distratores, args.seed)},
            fh,
        )
    return c


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-distratores", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recarregar", action="store_true", help="ignora a cache local")
    p.add_argument("--top-k", type=int, default=4, help="passagens entregues ao gerador")
    p.add_argument("--desvio", type=int, default=50, help="início da janela no braço degradado")
    p.add_argument("--so-recuperacao", action="store_true", help="pára antes da geração")
    p.add_argument("--saida", type=Path, default=Path("outputs/ablacao"))
    args = p.parse_args()

    conjunto = _conjunto(args)
    print(json.dumps(conjunto.resumo(), indent=2, ensure_ascii=False))

    t0 = time.time()
    indice = BM25Index().build(conjunto.doc_ids, conjunto.textos)
    t_indice = time.time() - t0

    t0 = time.time()
    corrida = {
        qid: [d for d, _ in indice.search(conjunto.queries[qid], 200)] for qid in conjunto.qrels
    }
    t_consulta = time.time() - t0
    print(f"[bm25] indexação {t_indice:.0f}s · {len(corrida)} queries em {t_consulta:.0f}s")

    bracos: dict[str, dict[str, Any]] = {}
    for nome, desvio in (("topo", 0), ("degradado", args.desvio)):
        cob = cobertura_da_recuperacao(conjunto, corrida, top_k=args.top_k, desvio=desvio)
        itens = itens_para_pipeline(conjunto, corrida, top_k=args.top_k, desvio=desvio)
        bracos[nome] = {"cobertura": cob, "n_itens": len(itens)}
        acertos = f"{cob['n_com_relevante_na_janela']}/{cob['n_queries']}"
        print(f"[{nome:<10}] cobertura={cob['cobertura']}  ({acertos})")

    args.saida.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "conjunto": conjunto.resumo(),
        "parametros": {
            "top_k": args.top_k,
            "desvio": args.desvio,
            "seed": args.seed,
            "n_distratores_corpus": args.n_distratores,
        },
        "bm25": {
            "segundos_indexacao": round(t_indice, 1),
            "segundos_consulta": round(t_consulta, 1),
        },
        "bracos": bracos,
    }
    destino = args.saida / "ablacao_recuperacao.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[escrito] {destino}")

    if args.so_recuperacao:
        # Sem geração não há conclusão sobre grounding — só a variável independente.
        print("\n--so-recuperacao: parado antes da geração.")
        return

    print("\nGeração ainda não ligada neste script; corra com --so-recuperacao.")


if __name__ == "__main__":
    main()

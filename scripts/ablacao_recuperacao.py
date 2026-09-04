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

from dotenv import load_dotenv

from llm_evaluation.config import load_config
from llm_evaluation.pipeline import run_batch
from llm_evaluation.reporting import record_to_json
from llm_evaluation.retrieval_eval.bm25 import BM25Index
from llm_evaluation.retrieval_eval.ponte import (
    ConjuntoPonte,
    carrega_ponte_hotpotqa,
    cobertura_da_recuperacao,
    contexto_entregue_tem_relevante,
    itens_para_pipeline,
    verifica_manipulacao,
)
from llm_evaluation.statistics import mcnemar_test, paired_bootstrap_diff_ci
from llm_evaluation.types import RunRecord
from llm_evaluation.verification.aggregate import judge_negative_for_aggregation

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


def _sustentado(registo: RunRecord, vereditos_negativos: list[str]) -> bool | None:
    """True se o juiz considerou a resposta sustentada; None quando não é medível.

    None em dois casos que **não** são resultados do sistema: o item falhou por
    erro de execução, ou o juiz caiu no fallback heurístico — que responde
    sempre `sustentado` e tornaria um juiz avariado indistinguível de um juiz
    permissivo.
    """
    if registo.meta.get("processing_error"):
        return None
    juiz = registo.signals.judge
    if juiz is None or juiz.raw.get("fallback_heuristico"):
        return None
    return not judge_negative_for_aggregation(registo.signals, vereditos_negativos)


def _compara(
    a: dict[str, bool],
    b: dict[str, bool],
    *,
    nome_a: str,
    nome_b: str,
) -> dict[str, Any]:
    """McNemar mais bootstrap sobre os itens comuns aos dois braços.

    Emparelhado por construção: os braços correm sobre os mesmos ids. Um teste
    de duas proporções sobrestimaria o erro-padrão e perderia poder.
    """
    comuns = sorted(set(a) & set(b))
    va = [a[i] for i in comuns]
    vb = [b[i] for i in comuns]
    # b = sustentado só em A; c = sustentado só em B.
    disc_b = sum(1 for x, y in zip(va, vb, strict=True) if x and not y)
    disc_c = sum(1 for x, y in zip(va, vb, strict=True) if y and not x)
    return {
        "par": [nome_a, nome_b],
        "n_comuns": len(comuns),
        "n_excluidos_a": len(a) - len(comuns),
        "n_excluidos_b": len(b) - len(comuns),
        "taxa_a": round(sum(va) / len(va), 4) if va else None,
        "taxa_b": round(sum(vb) / len(vb), 4) if vb else None,
        "mcnemar": mcnemar_test(disc_b, disc_c),
        "bootstrap": paired_bootstrap_diff_ci(va, vb),
    }


def main() -> None:
    # Igual ao `llm-eval`: sem isto o script pede a chave que já está no .env.
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-distratores", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recarregar", action="store_true", help="ignora a cache local")
    p.add_argument("--top-k", type=int, default=4, help="passagens entregues ao gerador")
    p.add_argument(
        "--desvios",
        type=int,
        nargs="+",
        default=[0, 2, 50],
        help="início da janela em cada braço; 0 = recuperação normal",
    )
    p.add_argument("--config", type=Path, default=Path("configs/hotpotqa_ponte.yaml"))
    p.add_argument("--chunk-max-chars", type=int, default=1200, help="igual ao rag.chunk_max_chars")
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
    itens_por_braco: dict[str, list[Any]] = {}
    for desvio in args.desvios:
        nome = f"desvio_{desvio}"
        cob = cobertura_da_recuperacao(conjunto, corrida, top_k=args.top_k, desvio=desvio)
        itens = itens_para_pipeline(conjunto, corrida, top_k=args.top_k, desvio=desvio)
        itens_por_braco[nome] = itens
        # O que importa é o contexto que o gerador recebe, não o ranking: as duas
        # coisas já divergiram e custaram uma corrida inteira.
        entregue = contexto_entregue_tem_relevante(
            conjunto, itens, chunk_max_chars=args.chunk_max_chars
        )
        bracos[nome] = {"cobertura": cob, "contexto_entregue": entregue, "n_itens": len(itens)}
        acertos = f"{cob['n_com_relevante_na_janela']}/{cob['n_queries']}"
        entregues = f"{entregue['n_com_relevante_no_contexto']}/{entregue['n_itens']}"
        print(f"[{nome:<12}] cobertura={cob['cobertura']} ({acertos})  contexto={entregues}")

    verifica_manipulacao({n: v["contexto_entregue"] for n, v in bracos.items()})

    args.saida.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "conjunto": conjunto.resumo(),
        "parametros": {
            "top_k": args.top_k,
            "desvios": args.desvios,
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

    cfg = load_config(args.config)
    negativos = list(cfg.verification.judge_aggregation_verdicts)
    sustentados: dict[str, dict[str, bool]] = {}
    for nome, itens in itens_por_braco.items():
        print(f"\n=== geração: {nome} ({len(itens)} itens) ===")
        t0 = time.time()
        dir_braco = args.saida / nome
        dir_braco.mkdir(parents=True, exist_ok=True)
        # `run_batch` não escreve artefactos — isso é do CLI. Sem `predictions.jsonl`
        # a corrida não é auditável e o resultado não é reconferível item a item.
        with (dir_braco / "predictions.jsonl").open("w", encoding="utf-8") as fh:

            def _escreve(rec: RunRecord, _fh: Any = fh) -> None:
                _fh.write(json.dumps(record_to_json(rec), ensure_ascii=False) + "\n")
                _fh.flush()

            registos = run_batch(
                cfg,
                itens,
                on_record=_escreve,
                run_dir=dir_braco,
                config_name=str(args.config),
            )
        medidos = {r.item_id: _sustentado(r, negativos) for r in registos}
        sustentados[nome] = {k: v for k, v in medidos.items() if v is not None}
        excluidos = len(medidos) - len(sustentados[nome])
        taxa = sum(sustentados[nome].values()) / len(sustentados[nome])
        bracos[nome]["geracao"] = {
            "n_medidos": len(sustentados[nome]),
            "n_excluidos": excluidos,
            "taxa_sustentado": round(taxa, 4),
            "segundos": round(time.time() - t0, 1),
        }
        print(f"  sustentado={taxa:.3f}  medidos={len(sustentados[nome])}  excluídos={excluidos}")

    nomes = list(sustentados)
    comparacoes = [
        _compara(sustentados[a], sustentados[b], nome_a=a, nome_b=b)
        for i, a in enumerate(nomes)
        for b in nomes[i + 1 :]
    ]
    relatorio["bracos"] = bracos
    relatorio["comparacoes"] = comparacoes
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== COMPARAÇÕES EMPARELHADAS ===")
    for c in comparacoes:
        boot = c["bootstrap"] or {}
        mac = c["mcnemar"] or {}
        exclui_zero = boot and (boot["ic_inferior"] > 0 or boot["ic_superior"] < 0)
        marca = "SIGNIFICATIVO" if exclui_zero else "não distinguível de ruído"
        print(
            f"  {c['par'][0]} vs {c['par'][1]}: "
            f"{c['taxa_a']} vs {c['taxa_b']}  "
            f"dif={boot.get('diferenca_observada', float('nan')):+.4f} "
            f"IC95=[{boot.get('ic_inferior', float('nan')):+.4f},"
            f"{boot.get('ic_superior', float('nan')):+.4f}]  "
            f"p={mac.get('p_valor', float('nan')):.4g}  {marca}"
        )
    print(f"\n[escrito] {destino}")


if __name__ == "__main__":
    main()

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
    """True se o juiz não classificou a resposta como negativa; None quando não é medível.

    Esta é a taxa de aprovação do juiz — o KPI que sairia naturalmente do
    `summary.json`. Ela é **enganosa como métrica de produto** (ver
    SPEC-013): a árvore de decisão do juiz classifica uma recusa honesta com
    contexto insuficiente como `sustentado`, e está correcta ao fazê-lo — mas
    isso faz a taxa subir à medida que a recuperação piora, porque mistura
    «respondeu bem» com «recusou bem». Fica no relatório só como contraste
    com `respondeu_e_sustentado`, que é a variável dependente correcta.

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


def _contexto_insuficiente(registo: RunRecord) -> bool | None:
    """O que o próprio gerador declarou sobre o contexto, na sua saída estruturada.

    Isto — e não `verification.gold.is_refusal` sobre o texto da resposta — é
    o sinal correcto de recusa aqui. `is_refusal` procura frases como «cannot»,
    «não sei»; o prompt `generic` produz recusas como «The context does not
    provide information about…», que essa heurística não reconhece, e o
    braço degradado ficaria com `e_recusa=False` em 100% dos itens apesar de
    recusar na prática. `contexto_insuficiente` vem do próprio JSON validado
    do gerador (`responder_schema.py`) — é uma declaração, não um palpite
    sobre texto livre.

    None quando não há declaração para ler: erro de execução, ou saída que
    falhou a validação de schema (`generation.py` grava `contexto_insuficiente:
    None` nesse caso).
    """
    if registo.meta.get("processing_error"):
        return None
    qg = registo.meta.get("qualidade_geracao") or {}
    valor = qg.get("contexto_insuficiente")
    return valor if isinstance(valor, bool) else None


def _veredito_medivel(registo: RunRecord) -> str | None:
    """Veredito do juiz, ou None quando não é medível (mesmas exclusões de `_sustentado`)."""
    if registo.meta.get("processing_error"):
        return None
    juiz = registo.signals.judge
    if juiz is None or juiz.raw.get("fallback_heuristico"):
        return None
    return str(juiz.veredito)


def _metricas_produto(registos: list[RunRecord]) -> dict[str, dict[str, bool]]:
    """As variáveis dependentes que a SPEC-013 usa, uma por item, só onde medíveis.

    `respondeu` e `respondeu_e_sustentado` separam «tentou responder» de
    «tentou e o juiz sustentou completamente» (regra 8 do CLAUDE.md — planos
    métricos não se misturam). `alucinou` é respondeu com veredito
    `nao_sustentado` — resposta dada, contradita ou sem suporte, distinta de
    recusa (não deu resposta) e de `incompleto` (respondeu, parcialmente
    sustentado, nem aprovado nem contradito).

    Cada dicionário só contém itens onde a variável é medível; a exclusão é
    contada por braço em `bracos[nome]["geracao"]`.
    """
    respondeu: dict[str, bool] = {}
    respondeu_e_sustentado: dict[str, bool] = {}
    alucinou: dict[str, bool] = {}
    for r in registos:
        ci = _contexto_insuficiente(r)
        if ci is None:
            continue
        resp = not ci
        respondeu[r.item_id] = resp
        veredito = _veredito_medivel(r)
        if veredito is None:
            continue
        respondeu_e_sustentado[r.item_id] = resp and veredito == "sustentado"
        alucinou[r.item_id] = resp and veredito == "nao_sustentado"
    return {
        "respondeu": respondeu,
        "respondeu_e_sustentado": respondeu_e_sustentado,
        "alucinou": alucinou,
    }


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
    respondeu_por_braco: dict[str, dict[str, bool]] = {}
    resp_sust_por_braco: dict[str, dict[str, bool]] = {}
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
        taxa_sustentado = sum(sustentados[nome].values()) / len(sustentados[nome])

        produto = _metricas_produto(registos)
        respondeu_por_braco[nome] = produto["respondeu"]
        resp_sust_por_braco[nome] = produto["respondeu_e_sustentado"]
        n_medidos_resp = len(produto["respondeu"])
        taxa_respondeu = sum(produto["respondeu"].values()) / n_medidos_resp
        # respondeu_e_sustentado e alucinou têm denominador próprio: excluem
        # também os itens sem veredito medível (regra 4 do CLAUDE.md), que é
        # um conjunto de exclusão diferente do de `respondeu` sozinho.
        n_medidos_resp_sust = len(produto["respondeu_e_sustentado"])
        taxa_resp_sust = sum(produto["respondeu_e_sustentado"].values()) / n_medidos_resp_sust
        taxa_alucinou = sum(produto["alucinou"].values()) / n_medidos_resp_sust

        bracos[nome]["geracao"] = {
            "n_medidos": len(sustentados[nome]),
            "n_excluidos": excluidos,
            # KPI ingénuo (ver docstring de `_sustentado`): sobe quando a
            # recuperação piora. Fica aqui só para o contraste ficar no artefacto.
            "taxa_aprovacao_juiz": round(taxa_sustentado, 4),
            # Variáveis dependentes correctas (SPEC-013): separam «tentou
            # responder» de «tentou e o juiz sustentou», em vez de misturar
            # resposta sustentada com recusa honesta num só número.
            "n_medidos_respondeu": n_medidos_resp,
            "taxa_respondeu": round(taxa_respondeu, 4),
            "taxa_recusou": round(1 - taxa_respondeu, 4),
            "n_medidos_respondeu_e_sustentado": n_medidos_resp_sust,
            "taxa_respondeu_e_sustentado": round(taxa_resp_sust, 4),
            "taxa_alucinou": round(taxa_alucinou, 4),
            "segundos": round(time.time() - t0, 1),
        }
        print(
            f"  respondeu={taxa_respondeu:.3f}  respondeu_e_sustentado={taxa_resp_sust:.3f}  "
            f"alucinou={taxa_alucinou:.3f}  (aprovação_juiz={taxa_sustentado:.3f})"
        )

    nomes = list(sustentados)
    pares = [(a, b) for i, a in enumerate(nomes) for b in nomes[i + 1 :]]
    comparacoes_respondeu = [
        _compara(respondeu_por_braco[a], respondeu_por_braco[b], nome_a=a, nome_b=b)
        for a, b in pares
    ]
    comparacoes_resp_sust = [
        _compara(resp_sust_por_braco[a], resp_sust_por_braco[b], nome_a=a, nome_b=b)
        for a, b in pares
    ]
    comparacoes_aprovacao_juiz = [
        _compara(sustentados[a], sustentados[b], nome_a=a, nome_b=b) for a, b in pares
    ]
    relatorio["bracos"] = bracos
    # `respondeu_e_sustentado` é a comparação que sustenta a conclusão do
    # relatório; `respondeu` isolada mostra se a propensão a tentar responder
    # já muda sozinha; `taxa_aprovacao_juiz` fica ao lado só para mostrar que
    # a métrica óbvia aponta ao contrário (SPEC-013).
    relatorio["comparacoes"] = {
        "respondeu": comparacoes_respondeu,
        "respondeu_e_sustentado": comparacoes_resp_sust,
        "taxa_aprovacao_juiz": comparacoes_aprovacao_juiz,
    }
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    def _imprime(titulo: str, comparacoes: list[dict[str, Any]]) -> None:
        print(f"\n=== {titulo} ===")
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

    _imprime("COMPARAÇÕES EMPARELHADAS — respondeu_e_sustentado", comparacoes_resp_sust)
    _imprime(
        "COMPARAÇÕES EMPARELHADAS — taxa_aprovacao_juiz (KPI ingénuo)",
        comparacoes_aprovacao_juiz,
    )
    print(f"\n[escrito] {destino}")


if __name__ == "__main__":
    main()

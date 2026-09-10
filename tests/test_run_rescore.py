"""Reavaliação léxica offline de uma corrida gravada.

A normalização portuguesa chegou depois das corridas gravadas, e `referencia_incorreta`
— donde saem a exactidão e o κ do juiz — depende do F1 léxico. Sem um caminho de
reavaliação, uma correcção de medição fica invisível na evidência publicada.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_evaluation.run_rescore import infere_config_lexical, rescore_run_dir
from llm_evaluation.types import RunRecord, VerificationSignals


def _registo(item_id: str, resposta: str, referencia: str, **lexical: object) -> RunRecord:
    return RunRecord(
        item_id=item_id,
        question="pergunta?",
        answer=resposta,
        gold_correct=None,
        anomaly_flag=False,
        signals=VerificationSignals(
            gold_correct=None,
            gold_incorrect=None,
            is_refusal=False,
            embedding_max_cosine=0.9,
            embedding_low_support=False,
        ),
        retrieved=[],
        baseline_profile="hibrido",
        meta={
            "referencias": [referencia],
            "metricas_lexicas": {"modo_referencia": "primeiro", **lexical},
        },
    )


def _escreve_corrida(tmp_path: Path, registos: list[RunRecord]) -> Path:
    from llm_evaluation.reporting import record_to_json

    run_dir = tmp_path / "run_teste"
    run_dir.mkdir()
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in registos:
            fh.write(json.dumps(record_to_json(r), ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"tipo_referencia_ativo": "lexical", "sumario_lexical": {"media_f1_token": 0.0}}
        ),
        encoding="utf-8",
    )
    return run_dir


def test_infere_apenas_as_metricas_que_a_corrida_gravou() -> None:
    """O YAML original pode já não existir: três das quatro corridas gravadas
    apontam a um `configs/_tmp_*.yaml` apagado. O artefacto diz o que estava ligado."""
    registos = [_registo("a", "resposta", "ref", f1_token=0.5, bleu=0.2)]
    cfg = infere_config_lexical(registos)
    assert cfg.token_f1 is True
    assert cfg.bleu is True
    assert cfg.rouge_l is False
    assert cfg.meteor is False
    assert cfg.levenshtein is False


def test_meteor_indisponivel_conta_como_metrica_ligada() -> None:
    """Estava ligada e não produziu valor — é justamente o caso a preservar."""
    registos = [_registo("a", "r", "ref", meteor_indisponivel="recurso_nltk_ausente")]
    assert infere_config_lexical(registos).meteor is True


def test_nao_toca_nos_artefactos_originais(tmp_path: Path) -> None:
    registos = [_registo("a", "A ação começou à noite", "A ação começou à noite", f1_token=0.1)]
    run_dir = _escreve_corrida(tmp_path, registos)
    antes_pred = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")
    antes_sum = (run_dir / "summary.json").read_text(encoding="utf-8")

    rescore_run_dir(run_dir)

    assert (run_dir / "predictions.jsonl").read_text(encoding="utf-8") == antes_pred
    assert (run_dir / "summary.json").read_text(encoding="utf-8") == antes_sum
    assert (run_dir / "predictions.rescored.jsonl").is_file()
    assert (run_dir / "summary.rescored.json").is_file()


def test_declara_o_que_mudou(tmp_path: Path) -> None:
    """Uma corrida re-pontuada não é o mesmo resultado que a original."""
    registos = [_registo("a", "A ação começou", "A ação começou", f1_token=0.1)]
    run_dir = _escreve_corrida(tmp_path, registos)
    summary = rescore_run_dir(run_dir)
    rean = summary["reanalise"]
    assert rean["tipo"] == "rescore_lexical"
    assert rean["sem_chamadas_api"] is True
    assert rean["idioma_normalizacao"] == "pt"
    assert rean["n_itens_alterados"] == 1
    assert "agregados_antes" in rean
    assert "agregados_depois" in rean


def test_falha_de_execucao_nao_e_repontuada(tmp_path: Path) -> None:
    """Um item perdido por quota não tem métrica para recalcular (regra 3)."""
    bom = _registo("a", "resposta", "resposta", f1_token=0.9)
    mau = _registo("b", "", "resposta", f1_token=0.0)
    mau.meta["processing_error"] = {"type": "PermanentApiError", "message": "429"}
    run_dir = _escreve_corrida(tmp_path, [bom, mau])

    summary = rescore_run_dir(run_dir)

    linhas = [
        json.loads(x)
        for x in (run_dir / "predictions.rescored.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    falhado = next(x for x in linhas if x["id_item"] == "b")
    assert falhado["meta"]["metricas_lexicas"]["f1_token"] == 0.0
    assert summary["n_itens_avaliados"] == 1
    assert summary["n_itens_com_erro_execucao"] == 1


def test_item_sem_referencia_e_contado_nao_pontuado(tmp_path: Path) -> None:
    registo = _registo("a", "resposta", "ref", f1_token=0.5)
    registo.meta["referencias"] = []
    run_dir = _escreve_corrida(tmp_path, [registo])
    rean = rescore_run_dir(run_dir)["reanalise"]
    assert rean["n_itens_sem_referencia"] == 1
    assert rean["n_itens_alterados"] == 0


def test_idioma_en_reproduz_o_protocolo_squad(tmp_path: Path) -> None:
    """`en` existe para que resultados ingleses publicados continuem comparáveis."""
    registos = [_registo("a", "the state-of-the-art model", "state of the art model", f1_token=0.0)]
    run_dir = _escreve_corrida(tmp_path, registos)
    pt = rescore_run_dir(run_dir)["reanalise"]["agregados_depois"]["media_f1_token"]
    en = rescore_run_dir(run_dir, idioma="en")["reanalise"]["agregados_depois"]["media_f1_token"]
    assert pt != en

"""Regressões dos achados da revisão pré-MR.

Cada teste fixa um comportamento que a suite anterior não cobria e que, por isso,
falhava em silêncio: prompt do juiz reproduzido do protocolo, cancelamento sob
concorrência, respeito por ``Retry-After``, rótulos únicos, polaridade de vereditos
derivada da corrida e confiança de preenchimento fora da calibração.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_evaluation import cli, pipeline
from llm_evaluation.config import VerificationConfig
from llm_evaluation.evaluation_metrics import prediction_row_to_run_record
from llm_evaluation.judge_meta import (
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_NEGATIVE_VERDICTS,
    build_judge_meta_report,
    f1_fraca_min_from_protocol,
    judge_calibration,
    polarity_from_protocol,
    replay_config_from_run,
)
from llm_evaluation.llm_client import DEFAULT_MAX_CONNECTIONS, pool_size_for_concurrency
from llm_evaluation.protocol import build_protocolo_ativo

# --- Achado 1: o replay do juiz reproduz o prompt da corrida ------------------


def _write_run(tmp_path: Path, protocolo: dict[str, Any] | None) -> Path:
    run_dir = tmp_path / "run_replay"
    run_dir.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"tipo_referencia_ativo": "answer_lists"}
    if protocolo is not None:
        summary["protocolo_ativo"] = protocolo
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_replay_usa_o_tecto_de_contexto_da_corrida(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, {"judge_prompt_style": "pt", "judge_max_context_chars": 4000})
    replay = replay_config_from_run(run_dir, prompt_style_override=None)
    assert replay.prompt_style == "pt"
    assert replay.max_context_chars == 4000


def test_replay_sem_protocolo_usa_default_do_pipeline_nao_sem_tecto(tmp_path: Path) -> None:
    """Regressão: o default de ``run_judge_for_retrieved`` é ``None`` (sem tecto)."""
    replay = replay_config_from_run(_write_run(tmp_path, None), prompt_style_override=None)
    assert replay.max_context_chars == DEFAULT_JUDGE_MAX_CONTEXT_CHARS
    assert replay.max_context_chars is not None


def test_default_do_script_acompanha_o_default_da_config() -> None:
    """Se ``VerificationConfig`` mudar o tecto, o replay tem de mudar com ele."""
    campo = next(f for f in fields(VerificationConfig) if f.name == "judge_max_context_chars")
    assert campo.default == DEFAULT_JUDGE_MAX_CONTEXT_CHARS


def test_argumento_explicito_sobrepoe_o_protocolo(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, {"judge_prompt_style": "pt"})
    replay = replay_config_from_run(run_dir, prompt_style_override="rag_pt")
    assert replay.prompt_style == "rag_pt"
    assert "argumento" in replay.origem


def test_protocolo_ativo_regista_o_tecto_de_contexto() -> None:
    from pathlib import Path as P

    from llm_evaluation.config import load_config

    cfg = load_config(P("configs/smoke_amostra.yaml"))
    protocolo = build_protocolo_ativo(cfg)
    assert "judge_max_context_chars" in protocolo


# --- Achado 2: concorrência cancelável e submissão limitada -------------------


def test_submissao_e_limitada_a_uma_janela(monkeypatch: pytest.MonkeyPatch) -> None:
    """Não enfileira a corrida inteira: só assim o Ctrl+C consegue cancelar."""
    submetidos = 0
    lock = threading.Lock()
    libertar = threading.Event()

    def process(position: int, item: object) -> object:
        nonlocal submetidos
        with lock:
            submetidos += 1
        libertar.wait(timeout=5)
        return position

    itens = list(range(200))
    resultado: list[object] = []

    def consumir() -> None:
        resultado.extend(
            pipeline._run_batch_concurrent(  # noqa: SLF001
                itens,  # type: ignore[arg-type]
                process,  # type: ignore[arg-type]
                on_record=None,
                workers=4,
                t0=time.time(),
            )
        )

    t = threading.Thread(target=consumir)
    t.start()
    time.sleep(0.3)
    with lock:
        submetidos_em_espera = submetidos
    libertar.set()
    t.join(timeout=30)

    # Janela = max(2*workers, workers+1) = 8; nunca as 200.
    assert submetidos_em_espera <= 8
    assert len(resultado) == 200


def test_excecao_nao_espera_pelos_itens_restantes() -> None:
    """Regressão: ``shutdown(wait=True)`` fazia o Ctrl+C esperar por toda a corrida."""
    processados = 0
    lock = threading.Lock()

    def process(position: int, item: object) -> object:
        nonlocal processados
        with lock:
            processados += 1
        if position == 3:
            msg = "falha simulada"
            raise RuntimeError(msg)
        time.sleep(0.05)
        return position

    inicio = time.monotonic()
    with pytest.raises(RuntimeError, match="falha simulada"):
        pipeline._run_batch_concurrent(  # noqa: SLF001
            list(range(500)),  # type: ignore[arg-type]
            process,  # type: ignore[arg-type]
            on_record=None,
            workers=2,
            t0=time.time(),
        )
    assert time.monotonic() - inicio < 5.0
    assert processados < 500


# --- Achado 3: Retry-After nunca é encurtado ---------------------------------


def _client(**overrides: Any) -> Any:
    from llm_evaluation.llm_client import OpenAiCompatibleClient

    base = {
        "api_key": "k",
        "base_url": "https://api.example.com",
        "model": "m",
        "timeout_seconds": 1.0,
    }
    base.update(overrides)
    return OpenAiCompatibleClient(**base)


def test_retry_after_do_servidor_e_marcado_como_directiva() -> None:
    c = _client()
    request = httpx.Request("POST", "https://api.example.com")
    resposta = httpx.Response(429, request=request, headers={"Retry-After": "20"})
    atraso, do_servidor = c._retry_delay_seconds(0, status_code=429, response=resposta)  # noqa: SLF001
    assert atraso == 20.0
    assert do_servidor is True


def test_backoff_interno_nao_e_directiva() -> None:
    c = _client(backoff_seconds=(2.0,))
    atraso, do_servidor = c._retry_delay_seconds(0)  # noqa: SLF001
    assert atraso == 2.0
    assert do_servidor is False


def test_jitter_nunca_dorme_menos_que_o_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regressão: jitter simétrico podia retomar 20% antes do prazo do servidor."""
    dormidas: list[float] = []
    monkeypatch.setattr("llm_evaluation.llm_client.time.sleep", dormidas.append)
    c = _client()
    request = httpx.Request("POST", "https://api.example.com")
    resposta = httpx.Response(429, request=request, headers={"Retry-After": "20"})
    for _ in range(200):
        c._sleep_backoff(0, status_code=429, response=resposta)  # noqa: SLF001
    assert min(dormidas) >= 20.0
    assert max(dormidas) <= 24.0


def test_backoff_interno_mantem_jitter_simetrico(monkeypatch: pytest.MonkeyPatch) -> None:
    dormidas: list[float] = []
    monkeypatch.setattr("llm_evaluation.llm_client.time.sleep", dormidas.append)
    c = _client(backoff_seconds=(10.0,))
    for _ in range(200):
        c._sleep_backoff(0)  # noqa: SLF001
    assert min(dormidas) < 10.0
    assert max(dormidas) > 10.0


# --- Achado 9: pool dimensionado pela concorrência ---------------------------


def test_pool_cresce_com_a_concorrencia() -> None:
    assert pool_size_for_concurrency(1) == DEFAULT_MAX_CONNECTIONS
    assert pool_size_for_concurrency(64) == 68
    assert pool_size_for_concurrency(64) > 64


# --- Achado 4: rótulos únicos em --compare-runs ------------------------------


def test_basenames_iguais_sao_desambiguados(tmp_path: Path) -> None:
    """Regressão: dois run dirs homónimos colapsavam e a análise emparelhada sumia."""
    a = tmp_path / "A" / "run_20260101T00Z"
    b = tmp_path / "B" / "run_20260101T00Z"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    labels = cli._unique_run_labels([a, b])  # noqa: SLF001
    assert len(set(labels)) == 2
    assert labels == ["A/run_20260101T00Z", "B/run_20260101T00Z"]


def test_basenames_distintos_ficam_curtos(tmp_path: Path) -> None:
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    assert cli._unique_run_labels([a, b]) == ["run_a", "run_b"]  # noqa: SLF001


def test_colisao_ate_no_pai_cai_no_caminho_completo(tmp_path: Path) -> None:
    a = tmp_path / "x" / "p" / "run"
    b = tmp_path / "y" / "p" / "run"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    labels = cli._unique_run_labels([a, b])  # noqa: SLF001
    assert len(set(labels)) == 2
    assert labels == [str(a), str(b)]


# --- Achados 5 e 10: polaridade e limiar vêm do protocolo --------------------


def test_polaridade_default_trata_incompleto_como_aprovado() -> None:
    pol = polarity_from_protocol(None)
    assert pol.negative_verdicts == DEFAULT_NEGATIVE_VERDICTS
    assert pol.aprovou("incompleto") is True
    assert pol.aprovou("sustentado") is True
    assert pol.aprovou("nao_sustentado") is False


def test_polaridade_prefere_judge_aggregation_verdicts() -> None:
    pol = polarity_from_protocol(
        {
            "judge_aggregation_verdicts": ["nao_sustentado"],
            "negative_judge_verdicts": ["nao_sustentado", "incompleto"],
        }
    )
    assert pol.negative_verdicts == ("nao_sustentado",)
    assert pol.origem == "judge_aggregation_verdicts"
    assert pol.aprovou("incompleto") is True


def test_polaridade_cai_para_negative_judge_verdicts() -> None:
    pol = polarity_from_protocol({"negative_judge_verdicts": ["incompleto"]})
    assert pol.aprovou("incompleto") is False
    assert pol.origem == "negative_judge_verdicts"


def test_limiar_lexico_vem_do_protocolo() -> None:
    assert f1_fraca_min_from_protocol({"pattern_settings": {"f1_fraca_min": 0.45}}) == 0.45
    assert f1_fraca_min_from_protocol(None) is None
    assert f1_fraca_min_from_protocol({"pattern_settings": {"f1_fraca_min": "x"}}) is None


def _row(item_id: str, *, veredito: str, gold: bool, confianca: float | None) -> dict[str, Any]:
    juiz: dict[str, Any] = {"veredito": veredito, "motivo_breve": "m"}
    if confianca is not None:
        juiz["confianca"] = confianca
    return {
        "id_item": item_id,
        "pergunta": "q?",
        "resposta": "uma resposta",
        "gold_correto": gold,
        "flag_anomalia": False,
        "sinais": {"gold_correto": gold, "gold_incorreto": not gold, "juiz": juiz},
        "recuperados": [],
        "meta": {"metricas_recuperacao": {"rank_chunk_ouro": 1}},
    }


def test_incompleto_nao_conta_como_falso_negativo_por_omissao() -> None:
    """Regressão: ``incompleto`` sobre resposta correta deprimia κ sem motivo."""
    records = [
        prediction_row_to_run_record(_row(f"i{i}", veredito="incompleto", gold=True, confianca=0.8))
        for i in range(4)
    ]
    report = build_judge_meta_report(records, reference_type="answer_lists")
    concordancia = report["concordancia_com_referencia"]
    assert concordancia["confusao"]["juiz_aprovou_referencia_ok"] == 4
    assert concordancia["exatidao"] == 1.0


def test_protocolo_pode_tornar_incompleto_negativo() -> None:
    records = [
        prediction_row_to_run_record(_row(f"i{i}", veredito="incompleto", gold=True, confianca=0.8))
        for i in range(4)
    ]
    report = build_judge_meta_report(
        records,
        reference_type="answer_lists",
        protocol={"judge_aggregation_verdicts": ["nao_sustentado", "incompleto"]},
    )
    concordancia = report["concordancia_com_referencia"]
    assert concordancia["confusao"]["juiz_reprovou_referencia_ok"] == 4
    assert report["polaridade_vereditos"]["origem"] == "judge_aggregation_verdicts"


# --- Achado 6: confiança de preenchimento fora da calibração -----------------


def test_confianca_ausente_e_marcada_na_desserializacao() -> None:
    rec = prediction_row_to_run_record(_row("i0", veredito="sustentado", gold=True, confianca=None))
    assert rec.signals.judge is not None
    assert rec.signals.judge.raw["confianca_ausente"] is True
    assert rec.signals.judge.confianca == 0.5


def test_calibracao_ignora_confianca_de_preenchimento() -> None:
    """Regressão: 0.5 sintético criava um pico artificial no bin central do ECE."""
    records = [
        prediction_row_to_run_record(
            _row(f"i{i}", veredito="sustentado", gold=True, confianca=None)
        )
        for i in range(10)
    ]
    assert judge_calibration(records, "answer_lists") is None


def test_calibracao_reporta_quantos_excluiu() -> None:
    records = [
        prediction_row_to_run_record(_row("real", veredito="sustentado", gold=True, confianca=0.9)),
        prediction_row_to_run_record(
            _row("falta", veredito="sustentado", gold=True, confianca=None)
        ),
    ]
    out = judge_calibration(records, "answer_lists")
    assert out is not None
    assert out["n"] == 1
    assert out["n_excluidos_sem_confianca"] == 1


# --- Achado 7: pacing honrado sob concorrência -------------------------------


def test_pacer_espaca_itens_entre_threads() -> None:
    pacer = pipeline._Pacer(0.05)  # noqa: SLF001
    marcas: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        pacer.wait()
        with lock:
            marcas.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    inicio = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4 itens a 50 ms de espaçamento => o último não arranca antes de ~150 ms.
    assert max(marcas) - inicio >= 0.14


def test_pacer_desligado_nao_espera() -> None:
    pacer = pipeline._Pacer(0.0)  # noqa: SLF001
    inicio = time.monotonic()
    for _ in range(50):
        pacer.wait()
    assert time.monotonic() - inicio < 0.1


def test_pausa_invalida_e_tratada_como_desligada(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EVAL_INTER_ITEM_SLEEP", "muito")
    assert pipeline.inter_item_pause_seconds() == 0.0
    monkeypatch.setenv("LLM_EVAL_INTER_ITEM_SLEEP", "-3")
    assert pipeline.inter_item_pause_seconds() == 0.0


# --- Achado 11: JSONL de amostras malformado --------------------------------


def test_jsonl_truncado_sai_com_codigo_2(tmp_path: Path) -> None:
    """O script grava linha a linha: uma interrupção deixa a última linha parcial."""
    caminho = tmp_path / "s.jsonl"
    caminho.write_text('{"id_item": "a", "vereditos": ["x", "y"]}\n{"id_item": "b", "vered\n')
    with pytest.raises(SystemExit) as exc:
        cli._load_judge_samples(caminho)  # noqa: SLF001
    assert exc.value.code == 2


def test_jsonl_com_linha_nao_objeto_sai_com_codigo_2(tmp_path: Path) -> None:
    caminho = tmp_path / "s.jsonl"
    caminho.write_text('["nao", "e", "objeto"]\n')
    with pytest.raises(SystemExit) as exc:
        cli._load_judge_samples(caminho)  # noqa: SLF001
    assert exc.value.code == 2


def test_jsonl_valido_ignora_linhas_vazias(tmp_path: Path) -> None:
    caminho = tmp_path / "s.jsonl"
    caminho.write_text(
        '{"id_item": "a", "vereditos": ["x"]}\n\n{"id_item": "b", "vereditos": []}\n'
    )
    assert cli._load_judge_samples(caminho) == [["x"]]  # noqa: SLF001


# --- Fornecedores livres: erro de configuração falha já, com a causa ---------


def test_erro_permanente_nao_e_retentado_ao_nivel_do_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Um modelo inexistente não melhora com 3 tentativas — falha já e explica."""
    from llm_evaluation.llm_client import PermanentApiError

    tentativas = 0

    class _Broken:
        def complete(self, system: str, user: str, **_: object) -> str:
            nonlocal tentativas
            tentativas += 1
            msg = "HTTP 404 de http://localhost:11434/v1/chat/completions: model 'x' not found"
            raise PermanentApiError(msg)

    monkeypatch.setattr(pipeline, "default_llm_from_env", lambda **_: _Broken())
    monkeypatch.setattr(pipeline, "default_judge_from_env", lambda **_: _Broken())
    monkeypatch.delenv("LLM_EVAL_CONCURRENCY", raising=False)
    monkeypatch.setenv("LLM_EVAL_ITEM_RETRIES", "3")

    from llm_evaluation.config import load_config

    cfg = load_config(Path("configs/smoke_amostra.yaml"))
    from dataclasses import replace as dc_replace

    from llm_evaluation.types import EvalItem

    cfg = dc_replace(
        cfg,
        embeddings=dc_replace(cfg.embeddings, backend="hash"),
        rag=dc_replace(cfg.rag, min_retrieval_score=None),
        generation=dc_replace(cfg.generation, skip_llm_on_weak_retrieval=False),
    )
    item = EvalItem(
        id="i0",
        question="q?",
        correct_answers=["a"],
        incorrect_answers=[],
        rag_gold_chunk="contexto",
    )
    records = pipeline.run_batch(cfg, [item])
    assert tentativas == 1
    erro = records[0].meta["processing_error"]
    assert erro["type"] == "PermanentApiError"
    assert "not found" in erro["message"]

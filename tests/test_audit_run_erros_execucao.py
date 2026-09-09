"""Um item perdido por falha de execução não reprova a auditoria da corrida.

`_failed_record` marca o item para revisão, mas ele não tem métricas por medir:
tratá-lo como artefacto incompleto fazia `audit_run --strict` sair com erro por
uma propriedade da infraestrutura — quota, rede, 4xx do fornecedor — em vez de
por uma propriedade do sistema avaliado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_run import audit  # noqa: E402


def _corrida(tmp_path: Path, *, com_erro: bool) -> Path:
    run = tmp_path / "run_20260101T000000Z"
    run.mkdir()
    registo: dict[str, object] = {
        "id_item": "item-1",
        "pergunta": "p",
        "resposta": "" if com_erro else "uma resposta",
        "contextos_recuperados": [],
        "sinais": {},
        "flag_anomalia": True,
        "meta": {},
    }
    if com_erro:
        registo["meta"] = {
            "processing_error": {
                "type": "PermanentApiError",
                "message": "HTTP 429 de https://host",
            },
        }
    else:
        registo["meta"] = {"metricas_lexicas": {"f1_token": 0.5}}
    (run / "predictions.jsonl").write_text(json.dumps(registo) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "tipo_referencia_ativo": "lexical",
                "kpi_primario": "sumario_lexical",
                "sumario_lexical": {"media_f1_token": 0.5},
            }
        ),
        encoding="utf-8",
    )
    return run


def test_item_com_erro_de_execucao_nao_gera_problema(tmp_path: Path) -> None:
    problemas = audit(_corrida(tmp_path, com_erro=True))
    assert not [p for p in problemas if "f1_token" in p or "resposta vazia" in p]


def test_item_sem_erro_continua_a_ser_verificado(tmp_path: Path) -> None:
    run = _corrida(tmp_path, com_erro=False)
    linha = json.loads((run / "predictions.jsonl").read_text(encoding="utf-8"))
    linha["meta"] = {}
    (run / "predictions.jsonl").write_text(json.dumps(linha) + "\n", encoding="utf-8")
    problemas = audit(run)
    assert any("falta f1_token" in p for p in problemas)

"""O identificador de reprodutibilidade tem de cobrir a variável independente.

Incidente: num A/B de quatro juízes, dois braços partilhavam `config_hash_sha256`
porque o YAML não declara o modelo — `LLM_MODEL` e `JUDGE_MODEL` vêm do ambiente.
O hash do ficheiro não distinguia as corridas que a experiência existia para comparar.
"""

from __future__ import annotations

from typing import Any

from llm_evaluation.protocol import protocolo_sha256
from llm_evaluation.run_reprocess import provenance_block


def _protocolo(judge_model: str) -> dict[str, Any]:
    return {
        "aggregation_policy": "embedding_e_juiz",
        "embedding_min_cosine": 0.24,
        "models": {
            "llm_model": "gpt-4o-mini",
            "judge_model": judge_model,
            "llm_endpoint": "https://api.openai.com",
            "judge_endpoint": "https://api.openai.com",
        },
    }


def test_bracos_que_so_diferem_no_juiz_tem_hashes_distintos() -> None:
    a = protocolo_sha256(_protocolo("gpt-4o"))
    b = protocolo_sha256(_protocolo("gpt-4o-mini"))
    assert a != b


def test_hash_e_estavel_para_o_mesmo_protocolo() -> None:
    assert protocolo_sha256(_protocolo("gpt-4o")) == protocolo_sha256(_protocolo("gpt-4o"))


def test_hash_ignora_a_ordem_das_chaves() -> None:
    base = _protocolo("gpt-4o")
    invertido = dict(reversed(list(base.items())))
    assert protocolo_sha256(base) == protocolo_sha256(invertido)


def test_proveniencia_expoe_os_dois_hashes() -> None:
    """O hash do ficheiro fica: mudá-lo quebraria `--resume` de corridas em curso."""
    prov = provenance_block({"config_hash_sha256": "abc"}, _protocolo("gpt-4o"))
    assert prov["config_hash_sha256"] == "abc"
    assert isinstance(prov["protocolo_sha256"], str)
    assert len(str(prov["protocolo_sha256"])) == 64


def test_proveniencia_sem_protocolo_nao_inventa_hash() -> None:
    prov = provenance_block({"config_hash_sha256": "abc"})
    assert prov["protocolo_sha256"] is None

"""Credenciais não podem sobreviver à serialização de artefactos.

`meta.processing_error.message` acaba em `predictions.jsonl`, que é um artefacto
que se publica. Uma base URL com `https://utilizador:senha@host` — forma aceite
por proxies e gateways — chegava lá em claro.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_evaluation.llm_client import _permanent_http_error, redact_secrets
from llm_evaluation.pipeline import _failed_record
from llm_evaluation.reporting import record_to_json
from llm_evaluation.types import EvalItem


class TestRedactSecrets:
    def test_userinfo_na_url(self) -> None:
        assert (
            redact_secrets("falhou em https://utilizador:senha@api.x.com/v1")
            == "falhou em https://***@api.x.com/v1"
        )

    def test_chave_estilo_openai(self) -> None:
        saida = redact_secrets("chave sk-proj-AbCdEf0123456789XyZ rejeitada")
        assert "sk-proj-AbCdEf" not in saida
        assert "***" in saida

    def test_cabecalho_bearer(self) -> None:
        assert "eyJhbGci" not in redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef")

    def test_texto_sem_segredos_fica_intacto(self) -> None:
        original = "HTTP 404: model 'qwen2.5:7b' not found em http://localhost:11434/v1"
        assert redact_secrets(original) == original

    def test_url_sem_credenciais_nao_e_alterada(self) -> None:
        original = "HTTP 500 de https://api.openai.com/v1/chat/completions"
        assert redact_secrets(original) == original

    def test_token_curto_nao_e_confundido_com_chave(self) -> None:
        # 'sk-abc' é curto demais para ser uma chave; não deve virar ***
        assert redact_secrets("prefixo sk-abc") == "prefixo sk-abc"


class TestNaoVazaParaArtefactos:
    @staticmethod
    def _linha(err: Exception) -> str:
        rec = _failed_record(
            EvalItem("i", "q", [], []),
            baseline_profile="h",
            orchestration="unico",
            err=err,
            attempt=1,
        )
        return json.dumps(record_to_json(rec), ensure_ascii=False)

    def test_credencial_na_url_nao_chega_a_predictions(self) -> None:
        req = httpx.Request("POST", "https://u:SENHA_SECRETA@api.x.com/v1/chat/completions")
        err = _permanent_http_error(
            httpx.Response(404, request=req, json={"error": {"message": "x"}}),
        )
        assert "SENHA_SECRETA" not in self._linha(err)

    def test_chave_ecoada_pelo_fornecedor_nao_chega_a_predictions(self) -> None:
        """Alguns fornecedores devolvem o cabeçalho recebido na mensagem de erro."""
        req = httpx.Request("POST", "https://api.x.com/v1/chat/completions")
        corpo = {"error": {"message": "invalid key sk-proj-AbCdEf0123456789XyZabc"}}
        err = _permanent_http_error(httpx.Response(401, request=req, json=corpo))
        assert "sk-proj-AbCdEf" not in self._linha(err)

    def test_excecao_arbitraria_tambem_e_limpa(self) -> None:
        """A defesa não depende de a excepção ser nossa."""
        assert "OUTRA" not in self._linha(ValueError("falhou em https://u:OUTRA@h/v1"))

    @pytest.mark.parametrize("texto", ["erro simples", "timeout após 120s"])
    def test_mensagens_normais_sobrevivem(self, texto: str) -> None:
        assert texto in self._linha(RuntimeError(texto))

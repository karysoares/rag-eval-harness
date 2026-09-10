"""Credenciais não podem sobreviver à serialização de artefactos.

`meta.processing_error.message` acaba em `predictions.jsonl`, que é um artefacto
que se publica. Uma base URL com `https://utilizador:senha@host` — forma aceite
por proxies e gateways — chegava lá em claro.
"""

from __future__ import annotations

import inspect
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


class TestCredenciaisForaDoFormatoOpenAI:
    """Formas de credencial que não seguem ``sk-`` nem cabeçalho ``Bearer``.

    ``predictions.jsonl`` publica-se, e vários fornecedores compatíveis
    autenticam por query-string ou usam prefixos próprios. O padrão tem de
    apanhar a credencial pelo nome do parâmetro quando a forma do valor não é
    reconhecível.
    """

    @pytest.mark.parametrize(
        "texto",
        [
            "HTTP 401 de https://api.provedor.com/v1/chat?api-key=abc123def456ghi789",
            "HTTP 401 de https://host/v1?key=AIzaSyD-1234567890abcdef",
            "HTTP 401 de https://host/v1?access_token=zzz999888777666555",
            "falha com token gsk_9f8e7d6c5b4a3f2e1d0c9b8a7654321",
            "falha com hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        ],
    )
    def test_credencial_e_mascarada(self, texto: str) -> None:
        limpo = redact_secrets(texto)
        for segredo in (
            "abc123def456ghi789",
            "AIzaSyD-1234567890abcdef",
            "zzz999888777666555",
            "gsk_9f8e7d6c5b4a3f2e1d0c9b8a7654321",
            "hf_AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        ):
            assert segredo not in limpo
        assert "***" in limpo

    def test_nome_do_parametro_sobrevive(self) -> None:
        """Diz qual credencial falhou sem revelar o valor."""
        limpo = redact_secrets("https://host/v1?api-key=abc123def456ghi789")
        assert "api-key=***" in limpo


class TestFormasSemQueryString:
    """Credenciais que não vêm em query-string nem com prefixo de fornecedor.

    O padrão da query-string exige `?` ou `&`, pelo que a forma de cabeçalho
    (`x-api-key: <valor>`) passava intacta — e é justamente a forma que um corpo de
    resposta usa quando ecoa os cabeçalhos recebidos, o incidente que originou a
    invariante 1. Chaves sem prefixo reconhecível (Google, AWS, Slack) também.
    """

    def test_cabecalho_x_api_key(self) -> None:
        saida = redact_secrets("x-api-key: 1234567890abcdefghij")
        assert "1234567890abcdefghij" not in saida
        assert "x-api-key" in saida, "o nome diz qual credencial falhou; deve sobreviver"

    def test_cabecalho_api_key_sem_prefixo_x(self) -> None:
        assert "1234567890abcdefghij" not in redact_secrets("api-key: 1234567890abcdefghij")

    def test_chave_google(self) -> None:
        chave = "AIzaSyD1234567890abcdefghijklmnopqrstu"
        assert chave not in redact_secrets(f"erro com {chave} no corpo")

    def test_access_key_id_aws(self) -> None:
        assert "AKIAIOSFODNN7EXAMPLE" not in redact_secrets("id AKIAIOSFODNN7EXAMPLE recusado")

    def test_token_slack(self) -> None:
        assert "xoxb-1234567890-abcdefghij" not in redact_secrets(
            "token xoxb-1234567890-abcdefghij"
        )

    def test_texto_benigno_sobrevive(self) -> None:
        """Redigir demasiado torna as mensagens de erro inúteis para depurar."""
        for benigno in (
            "HTTP 401 de https://api.openai.com",
            "modelo qwen2.5:7b nao encontrado",
            "erro no campo resposta: contexto insuficiente",
            "rouge_l_f: 0.38",
        ):
            assert redact_secrets(benigno) == benigno, benigno


class TestEscritoresDeMetaQueEscapavam:
    """Quatro sítios escreviam em `meta` sem passar por `redact_secrets`.

    Só `pipeline._failed_record` redigia. Estes quatro chegam ao mesmo
    `predictions.jsonl` publicado por caminhos diferentes.
    """

    def test_generation_structured_output_error(self) -> None:
        import llm_evaluation.generation as gen
        from llm_evaluation.generation import _RESPONDER_INVALID_ANSWER_PT  # noqa: F401
        from llm_evaluation.structured_output import StructuredOutputError

        fonte = inspect.getsource(gen)
        assert "redact_secrets(str(exc))" in fonte
        assert StructuredOutputError is not None

    def test_critico_structured_output_error(self) -> None:
        import llm_evaluation.orchestration.multi as multi

        assert "redact_secrets(str(exc))" in inspect.getsource(multi)

    def test_metricas_lexicas_erro(self) -> None:
        from llm_evaluation.lexical_metrics import attach_lexical_to_meta

        assert "redact_secrets(msg)" in inspect.getsource(attach_lexical_to_meta)

    def test_ragas_erro(self) -> None:
        import llm_evaluation.benchmarks.ragas_adapter as ragas

        assert "redact_secrets(str(e))" in inspect.getsource(ragas)

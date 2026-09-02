import pytest

from llm_evaluation.llm_client import (
    OpenAiCompatibleClient,
    chat_completions_url,
    default_judge_from_env,
    default_llm_from_env,
    endpoint_host,
    judge_api_key_from_env,
    judge_base_url_from_env,
    openai_base_url_from_env,
)
from llm_evaluation.protocol import judge_generator_same_model_warning


def test_openai_base_url_empty_means_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert openai_base_url_from_env() == "https://api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    assert openai_base_url_from_env() == "https://api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    assert openai_base_url_from_env() == "https://api.openai.com"


def test_openai_base_url_adds_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "api.example.com/v1")
    assert openai_base_url_from_env() == "https://api.example.com/v1"


def test_openai_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1/")
    assert openai_base_url_from_env() == "https://api.example.com/v1"


class TestChatCompletionsUrl:
    """Regressão: fornecedores compatíveis publicam a base já com ``/v1``.

    Concatenar ``/v1/chat/completions`` às cegas dava ``/v1/v1/...`` e um 404 que,
    não sendo transitório, falhava sem explicar a causa — bloqueando na prática
    todo o endpoint não-OpenAI (Ollama, vLLM, OpenRouter, DashScope).
    """

    def test_base_sem_sufixo_recebe_v1(self) -> None:
        assert (
            chat_completions_url("https://api.openai.com")
            == "https://api.openai.com/v1/chat/completions"
        )

    def test_base_terminada_em_v1_nao_duplica(self) -> None:
        assert (
            chat_completions_url("https://openrouter.ai/api/v1")
            == "https://openrouter.ai/api/v1/chat/completions"
        )

    def test_ollama_local(self) -> None:
        assert (
            chat_completions_url("http://localhost:11434/v1")
            == "http://localhost:11434/v1/chat/completions"
        )
        assert (
            chat_completions_url("http://localhost:11434")
            == "http://localhost:11434/v1/chat/completions"
        )

    def test_dashscope_compatible_mode(self) -> None:
        base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        assert chat_completions_url(base) == base + "/chat/completions"

    def test_endpoint_completo_fica_intacto(self) -> None:
        url = "http://localhost:8000/v1/chat/completions"
        assert chat_completions_url(url) == url

    def test_barra_final_e_ignorada(self) -> None:
        assert (
            chat_completions_url("http://localhost:11434/v1/")
            == "http://localhost:11434/v1/chat/completions"
        )


class TestJudgeProvider:
    """Juiz num fornecedor distinto do gerador (ex.: API paga + Ollama local)."""

    def test_juiz_herda_base_do_gerador(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com")
        monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
        assert judge_base_url_from_env() == "https://api.example.com"

    def test_judge_base_url_sobrepoe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:11434")
        assert judge_base_url_from_env() == "http://localhost:11434"
        assert openai_base_url_from_env() == "https://api.example.com"

    def test_juiz_herda_chave_do_gerador(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-gerador")
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        assert judge_api_key_from_env() == "sk-gerador"

    def test_judge_api_key_sobrepoe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-gerador")
        monkeypatch.setenv("JUDGE_API_KEY", "ollama")
        assert judge_api_key_from_env() == "ollama"

    def test_endpoint_host_descarta_caminho_e_credenciais(self) -> None:
        assert endpoint_host("https://user:token@api.example.com/v1") == "https://api.example.com"
        assert endpoint_host("http://localhost:11434") == "http://localhost:11434"

    def test_default_judge_usa_o_endpoint_do_juiz(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-gerador")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("JUDGE_API_KEY", "ollama")
        monkeypatch.setenv("JUDGE_MODEL", "qwen2.5:7b")
        judge = default_judge_from_env(timeout_seconds=30)
        gerador = default_llm_from_env(timeout_seconds=30)
        assert isinstance(judge, OpenAiCompatibleClient)
        assert isinstance(gerador, OpenAiCompatibleClient)
        assert judge.base_url == "http://localhost:11434"
        assert judge.api_key == "ollama"
        assert judge.model == "qwen2.5:7b"
        assert gerador.base_url == "https://api.example.com"
        assert gerador.api_key == "sk-gerador"


class TestSelfReferenceWarning:
    """O aviso de auto-referência tem de considerar o endpoint, não só o nome."""

    def test_avisa_com_mesmo_modelo_e_mesmo_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-mini")
        monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
        aviso = judge_generator_same_model_warning()
        assert aviso is not None
        assert "mesmo endpoint" in aviso

    def test_nao_avisa_com_modelos_distintos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("JUDGE_MODEL", "qwen2.5:7b")
        assert judge_generator_same_model_warning() is None

    def test_nao_avisa_com_mesmo_nome_em_fornecedores_distintos(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``qwen2.5`` local e hospedado são modelos distintos na prática."""
        monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("JUDGE_MODEL", "qwen2.5:7b")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:11434")
        assert judge_generator_same_model_warning() is None

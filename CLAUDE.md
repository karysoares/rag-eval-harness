# CLAUDE.md — contexto e invariantes deste repositório

Harness de avaliação RAG + LLM. O núcleo mede recuperação, geração e verificação em
camadas independentes; cada dataset é um adaptador. Ver [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
e [`docs/PREMISSAS.md`](docs/PREMISSAS.md) antes de mudar o desenho.

## Comandos

```bash
uv sync --extra dev --extra dashboard
uv run pytest -q                              # suite completa
uv run ruff check . && uv run ruff format --check .
uv run mypy src                               # strict
uv run pytest tests/test_pipeline_e2e_mock.py -q   # smoke offline, sem API
uv run python scripts/audit_run.py outputs --strict
```

Uma alteração só está pronta com os quatro verdes: ruff, format, mypy strict, pytest.

---

## Verificações obrigatórias em code review

Cada uma existe porque falhou de verdade. O incidente está citado para que a regra
não seja lida como zelo abstrato.

### 1. Nada de credenciais em artefactos

`predictions.jsonl` e `summary.json` são publicados. Antes de aprovar qualquer código
que escreva nesses ficheiros, seguir o caminho de cada string até à serialização.

- Mensagens de excepção entram em `meta.processing_error.message`. Passar por
  `llm_client.redact_secrets`.
- URLs entram em artefactos só como `endpoint_host()` (`scheme://host`), nunca
  completas: `https://utilizador:senha@host` é forma válida e vazava em claro.
- Corpos de resposta de fornecedores podem ecoar a chave enviada.

> **Incidente:** uma base URL com userinfo e uma chave ecoada pelo fornecedor
> chegavam ambas ao `predictions.jsonl`. Corrigido em [#11](https://github.com/karysoares/rag-eval-harness/pull/11);
> regressões em `tests/test_secret_redaction.py`.

### 2. Canais laterais não alteram artefactos

Telemetria, logs e métricas são observação, não resultado. Um teste tem de afirmar
que os artefactos são idênticos com e sem o canal ligado — ver
`tests/test_telemetry.py::TestInvariantes`. Nunca guardar estruturas do canal em
`meta`: além de contaminar o artefacto, parte a serialização.

Corolário: telemetria não exporta conteúdo (pergunta, resposta, contexto) sem
opt-in explícito. Um endpoint de observabilidade é mais um sítio onde o corpus
passa a existir.

### 3. Falha de execução não é anomalia do sistema

`_failed_record` marca `flag_anomalia=True` para o item ir à revisão — correto para
a fila operacional, veneno para estatística. Qualquer análise sobre `flag_anomalia`
tem de excluir itens com `processing_error`, e declarar a exclusão.

> **Incidente:** 9 itens perdidos por quota esgotada produziram um McNemar com
> p=0,004 que media propagação de faturação. Excluídos, todos os pares dão p=1.

### 4. Fallbacks não podem falhar para "aprovado"

`heuristic_judge_json` devolve `sustentado` quando o juiz falha. Um juiz que nunca
responde no schema fica indistinguível de um juiz permissivo. Ao adicionar qualquer
fallback: contá-lo, excluí-lo das métricas de qualidade, e expor a contagem.

> **Incidente:** `mistral:7b` caiu 100% no fallback e apareceria com 100% de
> aprovação; `gpt-5-mini` fazia o mesmo por rejeitar `temperature=0`.

### 5. Números publicados são medidos, não estimados

Se um número aparece no README ou em `docs/evidencia/`, tem de vir de uma corrida
gravada. Estimativas são permitidas mas **rotuladas como tal**, com os parâmetros
de que derivam.

> **Incidente:** o custo por corrida foi publicado como facto vindo de uma conta de
> tokens. A medição real deu 9,7× mais, porque um preço único era aplicado a
> gerador e juiz em modelos diferentes.

### 6. Custo e tokens são por modelo

Gerador e juiz em modelos distintos é a configuração recomendada. Qualquer
agregação de custo que use um par único de preços está errada por construção.

### 7. Documentação publicada não referencia o que não existe

Antes de publicar `docs/`, resolver todos os links relativos. Configs, scripts e
ficheiros citados têm de existir no repositório — ou a menção deixa de ser link e
ganha nota a dizer que não é distribuído.

### 8. Planos métricos não se misturam

Recuperação é diagnóstica. Embedding, juiz e referência léxica ficam separados até
à política de agregação no YAML. HITL é o plano C e tem precedência sobre referência
automática quando ambos existem — mas nunca somados no mesmo denominador.

Um κ baixo entre juiz e referência léxica significa **sinais independentes**, não
juiz incompetente: medem perguntas diferentes. Escrever isso ao lado do número.

---

## Ao implementar

- Novo dataset → spec em `docs/specs/adapters/` + config de exemplo que exista.
- Nova saída em `summary.json` → verificar `dashboard/schema_validation.py`.
- Alterar `configs/*.yaml` versionados quebra `--resume` de corridas em curso
  (`config_hash` é SHA256 do ficheiro). Mesmo mudar só um comentário.
- Concorrência: estado partilhado entre itens tem de ser thread-local ou protegido.
  `UsageAccumulator` e `last_usage` já foram corridas de dados.
- Comentários explicam **porquê**, não o quê. Preferir a razão de desenho ao
  resumo da linha seguinte.

## Estilo

Português europeu nos comentários, docstrings e documentação; inglês nas mensagens
de commit e no `README.md` (o `README.pt-BR.md` é o espelho). Linha de 100 colunas,
ruff format. Sem `# type: ignore` sem justificação na mesma linha.

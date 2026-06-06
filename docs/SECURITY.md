# Segurança

## O que não fazer

- Não commite `/.env` nem tokens (OpenAI, Hugging Face, etc.).
- Não inclua chaves em issues, PRs ou logs públicos.
- Não desative verificação TLS em cliente HTTP deste projeto.

## Segredos e configuração

- Copie `.env.example` para `.env` local (ignorado pelo Git).
- Variáveis suportadas (ver `.env.example`): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`, `JUDGE_MODEL`, `HF_TOKEN`, `RUN_INTEGRATION` (só testes locais).

## Artefatos e dados

- `outputs/` contém resultados de execução: por defeito **não** versionado. Partilhe apenas agregados anonimizados quando publicar resultados.
- Respeite as políticas de retenção dos fornecedores de API ao publicar exemplos.

## Dependências

- Use lockfile (`uv.lock`) e, para manutenção, `uv pip audit` ou `pip-audit` no ambiente virtual.

## Reportar problemas

- Abra uma issue privada ou contacte o mantenedor se encontrar vulnerabilidade; não divulgue exploit público antes de mitigação acordada.

## Âmbito

Este repositório é uma **ferramenta de pesquisa e avaliação offline**, não um serviço exposto à Internet sem hardening (auth, rate limit, WAF).

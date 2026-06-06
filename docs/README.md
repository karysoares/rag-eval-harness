# Documentação — índice e trilha de leitura

Documentação técnica do harness de avaliação de sistemas de linguagem com recuperação aumentada, cobrindo arquitetura, especificações verificáveis, definições de métricas e decisões de desenho.

**Como correr a pipeline:** ver o guia rápido no [`README.md`](../README.md) na raiz (inclui `configs/smoke_amostra.yaml` e referência da CLI `llm-eval --help`).

## Ordem sugerida (estudo + implementação)

1. [`PREMISSAS.md`](PREMISSAS.md) — objetivo do projeto, o que priorizar e o que evitar (dataset-agnóstico).
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — fluxo retrieval → geração → grounding / juiz / referência.
3. [`specs/README.md`](specs/README.md) — especificações verificáveis (spec-driven).
4. [`CONTRIBUTING.md`](CONTRIBUTING.md) — fluxo de PR e comandos locais.
5. [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) — critérios e validadores antes de codificar.
6. [`metrics.md`](metrics.md) — resumo operacional (derivado das specs).
7. [`references.md`](references.md) — mapa técnica ↔ literatura.
8. [`SECURITY.md`](SECURITY.md) — segredos e artefatos.
9. Fichas em [`techniques/`](techniques/) — aprofundamento por tema (ordem livre; começar por RAG + juiz LLM).

## Ficheiros de apoio

- [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) — qualidade antes de considerar um marco fechado.
- [`glossary.md`](glossary.md) — termos usados nos relatórios.
- [`decisions/README.md`](decisions/README.md) — decisões arquiteturais (ADRs).

## Política de corridas

- Resultados brutos em `outputs/` (local, normalmente fora do Git). Sumários agregados podem ser copiados para `docs/runs/` **sem** dados sensíveis ou chaves de API, quando for necessário publicar resultados.

## Ligação ao código

Os módulos em `src/llm_evaluation/` referenciam a ficha `docs/techniques/` relevante nas docstrings quando aplicável.

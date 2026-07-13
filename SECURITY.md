# Security Policy

## Scope & threat model

`rag-eval-harness` is an **offline-first evaluation harness**. It does not run a
public service. The relevant surfaces are:

| Surface | Risk | Mitigation |
|---------|------|------------|
| API keys (`OPENAI_API_KEY`) | Leakage via commits/artifacts | Keys live only in `.env` (gitignored); `.env.example` documents shape; secret scanning in CI |
| Run artifacts (`outputs/run_*`) | Accidental PII from datasets | Reference corpus (FairytaleQA pt-BR) is public fiction; adapters must not ingest private data without review |
| LLM judge calls | Prompt injection from corpus text | Judge prompts are versioned and hashed in the run manifest; heuristic fallback is excluded from aggregation |
| Dependencies | Supply chain | `uv.lock` committed; Dependabot + `pip-audit` in CI |
| Dashboard (Streamlit) | Local only | Never intended for public deployment; no auth by design |

## Reporting a vulnerability

Open a GitHub issue with the label `security`, or contact the maintainer via the
profile listed on the repository. Please include reproduction steps. You should
receive a response within 7 days.

## Hard rules

- No real customer data in `configs/`, `tests/fixtures/` or committed `outputs/`.
- Secrets never in code, YAML, or notebooks — env vars only.
- New adapters that touch non-public corpora require a data note in `docs/`.

# Artifacts, secrets and operations

Reference for the operational half of an evaluation review. Read the section you
need.

- [Artifact integrity](#artifact-integrity)
- [Secret leakage paths](#secret-leakage-paths)
- [Side channels](#side-channels)
- [Provider portability](#provider-portability)
- [Cost accounting](#cost-accounting)
- [Reproducibility](#reproducibility)
- [CI for evaluation repositories](#ci-for-evaluation-repositories)
- [Published claims](#published-claims)

## Artifact integrity

Run artifacts — per-item predictions, run summaries, manifests — are the product.
They get committed, shared, and cited. Three properties are worth defending in
review:

**Determinism.** The same inputs and config produce the same artifact, including
ordering. Concurrency must not change row order. Where true determinism is
impossible (non-zero temperature, provider variability), the config that caused it
should be recorded so a reader knows not to expect it.

**Serializability.** Anything placed in a metadata dict eventually hits JSON. A
dataclass or a client object stashed there breaks the write, or worse, gets
coerced to a string that leaks internals. Check what a change adds to metadata.

**Schema evolution.** New fields in a summary need the validators updated and the
version considered. A validator that rejects unknown fields will fail on the new
artifact; one that ignores them will not catch typos. Know which behaviour the
repo has.

## Secret leakage paths

Trace every string that reaches an artifact or a log. The recurring paths:

**Exception messages quoting a URL.** Error text usually names the endpoint that
failed. A base URL may carry credentials as `https://user:password@host` — a form
proxies and gateways accept — and that goes straight into the message, into the
per-item error field, into the committed artifact.

**Provider error bodies.** Surfacing the provider's own message is good practice
for diagnosis, and some providers echo the rejected key or the received
`Authorization` header inside it.

**Config or environment dumps.** Metadata collectors that record "the environment
for reproducibility" and sweep up a token.

**URLs recorded for provenance.** Record scheme and host, never the full URL, so
userinfo cannot survive.

Defend at two layers: redact at the source where the message is built, and again
where the artifact is written. Two layers because the second does not depend on
which exception type arrived — a `ValueError` from anywhere gets the same
treatment as the client's own error.

A redaction helper should cover URL userinfo, common key prefixes, and bearer
tokens, and must leave ordinary text alone — over-redaction that mangles
`http://localhost:11434/v1` makes debugging worse.

## Side channels

Telemetry, tracing, and metrics observe a run; they are not part of its output.

- Artifacts must be byte-identical with the channel on and off. This deserves an
  explicit test, because the failure is silent.
- A failing backend produces a warning, not a failed run. Failing an evaluation
  because its instrumentation is down trades the goal for the instrument. Warn once
  per destination, not once per item.
- Content export — prompts, answers, retrieved context — defaults off. An
  observability endpoint is a second copy of the corpus in a system the operator
  may not control. Metrics and verdicts are safe defaults; text is a decision.
- Prefer one internal event model with adapters per destination over separate
  integrations. Most destinations speak OTLP; the mapping is written once.

## Provider portability

"OpenAI-compatible" covers a lot of shapes. What breaks:

**URL construction.** Most providers publish a base that already ends in `/v1`.
Appending `/v1/chat/completions` unconditionally yields `/v1/v1/...` and a 404 —
which, not being transient, fails without explaining itself. Accept the base with
and without the suffix, and the full endpoint too.

**Error classification.** Quota exhaustion commonly arrives as a rate-limit status
but never recovers. Retrying with backoff wastes the run and buries the provider's
message. Classify on the error body, not only the status.

**Rejected parameters.** Some models accept only a default temperature and reject
an explicit value with a 400. Code that pins temperature for determinism then fails
every call — and if there is a fallback, the failure looks like a permissive
result. Degrade with a warning and record that determinism was lost.

**Separate roles, separate providers.** Generator and judge should be configurable
independently. Beyond cost, a judge from a different family is methodologically
stronger, since a model grading its own output tends to prefer it. A single base
URL variable serving both prevents this entirely.

## Cost accounting

Generator and judge routinely use different models at very different prices. A
single price pair applied to all calls is wrong by construction, and the error runs
in whichever direction the expensive model sits.

Requirements:
- Token usage tracked per model, not only in aggregate.
- Prices configurable per model. Model identifiers may contain colons (local
  runtimes use `model:tag`), so a `name:prompt:completion` format must split from
  the right.
- Models with no configured price listed explicitly rather than dropped — a total
  that silently omits a model is worse than no total.
- Local models entered at zero so they appear in the total rather than as unknowns.

## Reproducibility

**Config hashing.** If resumption compares a hash of the config file, any edit to a
tracked config — including a comment — invalidates in-flight runs. Flag config
edits bundled with behaviour changes; they are often unintentional.

**Protocol snapshots.** The effective configuration should be recorded in the run
output, not just the config path, so a reader can reconstruct what produced the
numbers. Anything a replay tool needs — prompt style, context limits, verdict
polarity, thresholds — belongs there. A replay that falls back to library defaults
instead of the run's own settings measures a configuration that never ran.

**Seeds and ordering.** Sampling, shuffling, and bootstrap resampling should be
seeded and the seed recorded.

## CI for evaluation repositories

- An offline smoke path that exercises the full pipeline with a mock model, so CI
  does not depend on a provider or spend money.
- Artifact validation on a committed fixture run — schema and invariants — which
  catches contract drift that unit tests miss.
- Coverage gates are fine; treat them as a floor, not a goal.
- Dependency scanning: decide deliberately whether it blocks. When findings are
  transitive and their fixes sit outside the project's version bounds, a blocking
  gate that can never pass trains people to ignore it. Non-blocking with the
  reasoning written inline is more honest — and the reasoning belongs in the
  workflow file, where the next person will read it.
- Anything that publishes documentation should verify relative links resolve.

## Published claims

The line between a portfolio and a brochure is whether the numbers were measured.

- Every figure in a README or evidence file comes from a recorded run, or is
  labelled an estimate with the parameters it derives from.
- Aggregate evidence, not raw corpus content — respect dataset licences and avoid
  redistributing text.
- State the limitations beside the result: sample size, whether the reference was
  automatic or human, what was not tested.
- When a published number turns out to be wrong, correcting it is the work. An
  evidence file with a correction history is more credible than one without.

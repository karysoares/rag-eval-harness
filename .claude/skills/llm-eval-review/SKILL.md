---
name: llm-eval-review
description: Deep code review for LLM evaluation and LLMOps codebases — harnesses, judges, RAG pipelines, benchmark runners. Checks the failure modes that generic review misses: metrics that silently measure the wrong thing, fallbacks that default to "pass", execution errors contaminating statistics, credentials reaching published artifacts, per-model cost accounting, concurrency in run loops, and provider portability. Use this whenever reviewing changes to an evaluation harness, an LLM-as-judge, a scoring or metrics module, a benchmark or experiment runner, run artifacts like predictions.jsonl or summary.json, telemetry and observability for LLM calls, or CI that gates on evaluation results — and also when someone asks whether a measurement, benchmark or judge comparison is trustworthy, even if they do not use the word "review".
---

# Reviewing LLM evaluation code

Generic code review hunts for crashes. Evaluation code has a worse failure mode: it
runs clean, produces a number, and the number is wrong. Nobody notices, because the
output looks exactly like a correct one. Decisions get made on it.

This review targets that class. Work through the passes below in order — each later
pass assumes the earlier one held.

## How to review

Start by finding what kind of change this is, because it determines which passes
matter:

```bash
git diff --stat main...HEAD          # or the target range
```

- Touches metrics, judges, aggregation, statistics → passes 1–3 are the point.
- Touches the run loop, clients, or concurrency → passes 4–5.
- Touches artifacts, telemetry, or anything serialized → pass 6.
- Touches CI, configs, or docs with numbers in them → pass 7.

Report findings ranked by whether they corrupt a measurement, in which case they
block; degrade robustness, in which case they are worth fixing now; or are
stylistic, in which case say so and move on. A finding needs a concrete scenario:
*with these inputs, this number comes out wrong.* If you cannot construct one, you
have a suspicion, not a finding — label it as such.

---

## Pass 1 — Is the measurement measuring what it claims?

**Metric planes must not be mixed.** Evaluation systems accumulate signals that
answer different questions: retrieval quality, grounding, a judge verdict, lexical
overlap with a reference, human adjudication. Collapsing them into one score
destroys the information that made them worth collecting separately. Look for a
new aggregate that silently absorbs a diagnostic signal into a decision signal.

When a change reports agreement between two signals, check whether they actually
ask the same question. A judge scoring *grounding in retrieved context* against a
reference scoring *token overlap with a gold answer* will show κ near zero — that
is independence, not incompetence, and the report must say so next to the number.
A reviewer who lets κ≈0 be written up as "the judge is bad" has approved a wrong
conclusion.

**Denominators must be explicit and honest.** Any rate needs its denominator
visible. Watch for silent exclusions: items skipped by a gate, items with no
reference label, items where the signal was absent. Each is defensible; each must
be counted and reported. A metric that quietly drops a third of the corpus is worse
than no metric.

**Placeholder values must not enter statistics.** Deserialization often fills
missing fields with a default (confidence 0.5, score 0.0). If that default flows
into a calibration curve or a mean, it fabricates data. The fix is to mark the
substitution at the point it happens and filter on the mark, not to guess whether
0.5 was real.

## Pass 2 — Does the statistical design match the data?

**Paired data needs paired tests.** Comparing two configurations over the same
items is a paired design. An unpaired two-proportion test on paired data
overestimates the standard error and loses power — it is not conservative, it is
wrong in a way that hides real effects. McNemar for binary outcomes over shared
items; a paired bootstrap when an effect size with uncertainty is wanted.

Check the alignment key. Runs are usually joined on an item id; if labels or keys
can collide (directory basenames derived from timestamps are a classic), the join
silently drops or mismatches rows.

**Execution failures are not system outcomes.** This one is subtle and costly.
Harnesses commonly mark a failed item as an anomaly so it reaches a review queue —
correct for operations, poison for statistics. A run that lost items to a rate
limit, a quota, or a timeout then shows "anomalies" that measure infrastructure,
not the system under evaluation. Any analysis over an anomaly flag must exclude
items carrying an execution error, and must say how many it excluded on each side.
Asymmetric exclusion is itself a finding: it usually means one arm ran under
different conditions than the other.

**Degenerate results must be distinguishable from good ones.** A judge that
answers the same verdict to everything scores well on a small test and has zero
discriminative power. Check that the report exposes the outcome distribution, not
only accuracy — a collapsed distribution is the tell.

## Pass 3 — Is the judge characterised, or just used?

An LLM judge is an instrument. Review whether the change treats it as one.

- **Calibration**: does stated confidence track accuracy? Expected Calibration
  Error over reliability bins. A judge that says 0.98 while being right 60% of the
  time cannot be used for threshold-based triage, and any code that thresholds on
  its confidence is unsound.
- **Stability**: does it return the same verdict twice? Repeated sampling with
  Fleiss' κ. This sets a floor on the minimum detectable effect — a difference
  between two runs smaller than the judge's own sampling noise is not
  interpretable, whatever the p-value says.
- **Bias probes**: approval correlated with answer length, or with the position of
  the supporting evidence. These do not prove bias — longer answers may genuinely
  be better — but a strong correlation obliges inspection, and the report should
  say that rather than implying proof.
- **Policy inheritance**: a meta-evaluation must read the verdict polarity and
  thresholds the run actually used, not hardcode its own. Otherwise an advisory
  verdict gets scored as a failure and the judge looks worse than it is.

Fallback verdicts — the heuristic used when the judge call fails — must be excluded
from every quality metric. A fallback is a measurement of the fallback, not of the
judge.

## Pass 4 — Fallbacks, defaults, and the direction of failure

**Which way does a failure fail?** This is the highest-yield question in the whole
review. A fallback that defaults to "approved", "passed", or "no issue" makes a
broken component indistinguishable from a permissive one. The evaluation reports
success and nobody investigates.

Trace every default and every exception handler in the scoring path and ask what
the system concludes when it fires. If the answer is "everything is fine", the
change needs a counter, an exclusion from quality metrics, and the count surfaced
in the report.

Related: a broad `except` around a measurement that returns a neutral value is the
same bug wearing a different hat.

## Pass 5 — Concurrency and the run loop

Evaluation runs are I/O-bound and get parallelised. Review the consequences:

- **Shared mutable state across items.** Accumulators, "last result" attributes,
  caches. Under a thread pool these interleave and misattribute measurements to the
  wrong item. Thread-local storage or a lock; a comment saying "thread-safe" is not
  evidence.
- **Output determinism.** Results must land in a deterministic order regardless of
  worker count, or artifacts stop being comparable between runs. Consuming futures
  in submission order achieves this; consuming as-completed does not.
- **Interruptibility.** Submitting every item up front means a cancellation waits
  for the whole queue. A bounded window plus `cancel_futures` keeps Ctrl+C useful
  on a long run.
- **Throttling that survives parallelism.** Pacing applied per-item in a sequential
  loop disappears when concurrency is introduced — exactly when it is most needed.
- **Resource reuse.** A new HTTP client per call means a TLS handshake per call.
  Over thousands of calls this dominates. Check the pool is sized against the
  configured concurrency, or workers block until timeout and fail in a way that
  looks like provider flakiness.

## Pass 6 — Artifacts, secrets, and side channels

**Nothing secret may reach a published artifact.** Run outputs get committed,
shared, and attached to reports. Trace every string that reaches them:

```bash
grep -rn "processing_error\|str(err)\|str(exc)\|request.url\|base_url" src/
```

Exception messages are the usual culprit, because they quote the URL to say what
failed — and a base URL may carry `https://user:password@host`, a form proxies
accept. Provider error bodies can echo the key that was rejected. Redact at the
source and again where the artifact is written, so the guarantee does not depend on
which exception type arrived.

**Side channels must not alter the primary output.** Telemetry, tracing, and
metrics are observation. There should be a test asserting the artifacts are
identical with and without them enabled. Watch for instrumentation data stashed in
a metadata dict that later gets serialized — it corrupts the artifact and often
breaks serialization outright.

**Content export needs opt-in.** Shipping prompts and answers to an observability
backend creates a second copy of the corpus somewhere the operator may not control.
Default to metrics only.

## Pass 7 — Reproducibility, cost, and claims

- **Config hashing.** If run resumption compares a hash of a config file, then
  editing that file — even only a comment — breaks resumption for in-flight runs.
  Flag config edits in a change that also ships behaviour.
- **Cost per model.** Generator and judge routinely differ, and often by an order
  of magnitude in price. A single price pair applied to all calls is wrong by
  construction. Token accounting must be per model, and models without a configured
  price must be surfaced rather than dropped from the total.
- **Provider portability.** Endpoint URL construction that assumes one vendor's
  path shape breaks every compatible provider. Error classification matters too:
  quota exhaustion often arrives with a rate-limit status but never recovers, so
  retrying wastes the run and hides the cause. Parameters a model rejects — a fixed
  temperature, for instance — should degrade with a warning rather than failing
  every call into a fallback.
- **Published numbers are measured or labelled.** Any figure in a README, report,
  or evidence file must come from a recorded run, or be explicitly marked as an
  estimate with the parameters it derives from. This is the difference between a
  portfolio and a brochure.
- **Docs must not reference what does not exist.** Before publishing documentation,
  resolve relative links and check that referenced configs and scripts are actually
  shipped.

---

## Reporting

Lead with what would produce a wrong number, and give the scenario. Then robustness.
Then, briefly, style.

Distinguish confidence honestly: a finding you traced end to end reads differently
from a pattern that looks risky. Say which is which — an inflated finding costs the
reviewer's credibility on the ones that matter.

When the change is sound, say so plainly and name what convinced you. "Passes 1–3
hold: the new metric reports its denominator, excludes fallbacks, and the paired
test aligns on item id" is a more useful review than silence.

## Deeper reference

Read these when a pass turns up something you want to verify properly:

- `references/statistics.md` — choosing and validating tests for evaluation data:
  paired designs, exclusion rules, calibration error, agreement measures, and the
  degenerate cases each one hides.
- `references/artifacts-and-ops.md` — artifact integrity, secret redaction, CI
  gating for evaluation repos, and provider portability specifics.

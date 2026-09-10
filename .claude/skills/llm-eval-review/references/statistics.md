# Statistics for evaluation code review

Reference for verifying a statistical claim in an evaluation codebase. Read the
section you need; this is not meant to be read start to finish.

- [Choosing the test](#choosing-the-test)
- [Paired designs](#paired-designs)
- [What to exclude, and how to say so](#what-to-exclude-and-how-to-say-so)
- [Uncertainty on a proportion](#uncertainty-on-a-proportion)
- [Agreement](#agreement)
- [Calibration](#calibration)
- [Correlation probes](#correlation-probes)
- [Degenerate cases worth naming](#degenerate-cases-worth-naming)

## Choosing the test

| Question | Design | Test |
|---|---|---|
| Do two configurations differ, same items? | paired, binary outcome | McNemar |
| Same, and an effect size is wanted | paired | bootstrap over items |
| Two runs over *different* samples | unpaired | two-proportion z |
| How uncertain is this rate? | — | Wilson interval |
| Do two signals agree beyond chance? | — | Cohen's κ |
| Do N repeated samples agree? | — | Fleiss' κ |
| Does stated confidence track accuracy? | — | ECE / MCE over bins |

The most common review finding is the first row done with the third row's test.

## Paired designs

When both systems saw the same items, the correct statistic uses only the
**discordant pairs** — items where they disagree. Concordant pairs carry no
information about the difference, and including them inflates the denominator.

McNemar with `b` = marked by A only, `c` = marked by B only:

- Few discordant pairs (roughly `b + c ≤ 25`): exact binomial, `p = 0.5`.
- More: χ² with continuity correction, `(|b − c| − 1)² / (b + c)`, one degree of
  freedom. Survival function for 1 df is `erfc(sqrt(x/2))` — no SciPy needed.
- `b + c = 0`: the difference is **undefined**. Returning a p-value here is a bug;
  return nothing and say why.

A paired bootstrap resamples *items*, preserving the pairing, and gives a
confidence interval on the difference. It answers "how big, and how sure" where
McNemar answers only "is it distinguishable from noise".

Review checklist:
- What is the join key, and can it collide?
- Is coverage reported — how many items each side contributed, and how many were
  common?
- Is the unpaired test still used anywhere it shouldn't be?

## What to exclude, and how to say so

Exclusions are legitimate and must be visible. Three that recur:

**Execution failures.** Items that errored (rate limit, quota, timeout, crash)
often get flagged as anomalies so they reach a review queue. They must not enter a
comparison — they measure infrastructure. Report the count per side; asymmetric
exclusion is a finding, because it usually means the arms ran under different
conditions.

**Fallback outputs.** A heuristic substituted for a failed model call measures the
heuristic. Exclude from quality metrics; count separately.

**Fabricated defaults.** A field filled in during deserialization (confidence 0.5,
score 0.0) is not an observation. Mark it where the substitution happens and filter
on the mark.

The general rule: a total that silently omits cases is worse than one that reports
"total over N of M, excluding K for reason R".

## Uncertainty on a proportion

Use the Wilson interval, not the normal approximation — the normal interval is
badly behaved for small `n` or proportions near 0 and 1, which is most of what
evaluation produces.

Two rates whose Wilson intervals overlap are not distinguishable at that level.
A review finding: any comparison of two rates presented without intervals.

## Agreement

**Cohen's κ** for two raters over the same items, 2×2. Rough reading: `>0.6`
substantial, `0.2–0.6` moderate, `<0.2` weak. It is undefined when expected
agreement is 1 — every item in the same category — and returning 0 there hides a
degenerate rater.

**Fleiss' κ** for N repeated ratings per item. All items need the same number of
ratings. Used for self-consistency: rate the same input N times and measure
stability.

Interpretation trap worth catching in review: κ near zero between two *different*
constructs means the signals are independent, which is expected and often the
reason both are collected. It is only bad news when both claim to measure the same
thing.

## Calibration

Expected Calibration Error bins predictions by stated confidence and compares the
mean confidence in each bin to the observed accuracy:

`ECE = Σ (n_b / N) · |accuracy_b − confidence_b|`

MCE is the worst bin. Report the reliability table too — the aggregate hides
whether the model is overconfident everywhere or only in one range.

High ECE with high accuracy means a useful model whose confidence is not
informative. Code that thresholds on that confidence is unsound and should be
flagged.

Edge cases to check in an implementation:
- Confidence exactly `1.0` must land in the last bin, not fall outside.
- Values outside `[0, 1]` and NaN filtered before binning.
- Bin occupancy reported; an ECE over three items is noise.

## Correlation probes

**Point-biserial** relates a binary outcome to a continuous variable — approval
versus answer length, for a verbosity probe:

`r = ((M₁ − M₀) / σ) · sqrt(p·q)` with population σ.

Undefined with no variance or an empty group. And correlation here is a prompt to
inspect, never proof of bias: in many corpora complete answers are genuinely
longer. A report that states the correlation as bias is a finding.

## Degenerate cases worth naming

These pass tests and produce plausible numbers:

- **The constant rater.** Answers the same thing to everything. Accuracy looks fine
  on an unbalanced set; κ is ~0; discriminative power is nil. Catch it by reporting
  the outcome distribution.
- **The empty denominator.** Every item excluded for some reason, and the code
  divides anyway or returns 0.
- **The single-bin calibration.** All confidence in one bin, ECE computed and
  reported as if the curve were measured.
- **The unpaired paired test.** Discussed above; the most consequential of these.
- **The one-sided failure.** One arm lost items and the other did not, so the
  comparison silently measures the loss.

# Interval-coverage diagnostics

**The question:** when the model reports a 90% credible interval, does the true
value actually land inside it 90% of the time? Users implicitly assume yes;
this document describes the tools that check it and the ways it fails —
including the commonly observed "my 90% HDI contained the truth only ~50% of
the time in simulation".

Code: `src/mmm_framework/diagnostics/coverage.py` (+ `diagnostics/sbc.py`).
Interval convention everywhere: central equal-tailed percentile intervals,
matching `compute_hdi_bounds` (percentile-based despite the name) — so the
coverage measured is the coverage of the intervals the framework actually
reports.

## Three notions of coverage (don't conflate them)

| Notion | Question | Tool |
|---|---|---|
| **Predictive coverage** | Do posterior-predictive intervals contain the *observed data* at the nominal rate? | PPC checks, report calibration curve (`run_posterior_predictive_checks`) |
| **Engine calibration (SBC)** | Averaged over the model's own prior, are posterior ranks uniform (⇔ every interval has nominal coverage *on average*)? | `run_calibration_check` / `diagnostics.sbc.run_mmm_sbc` |
| **Recovery coverage at a fixed truth** | If the world were exactly θ\*, would repeated fits' intervals contain θ\* at the nominal rate? | `run_coverage_check` / `diagnostics.coverage.run_recovery_coverage` |

They can disagree, and the disagreement is diagnostic:

- SBC uniform but fixed-truth recovery under-covers → θ\* sits where the prior
  fights the data (prior–data conflict) or the point chosen is atypical.
- Both pass but intervals missed an **external** answer key (synthetic worlds,
  lift tests) → structural failure: misspecification, confounding, or you
  compared the wrong estimand.
- Predictive coverage fine but parameter coverage broken → the *fit* looks
  right while the *decomposition* is wrong (classic under weak identification
  plus approximate inference).

## The easy tools

### 1. Chat / agent: `run_coverage_check`

Fixes every free parameter at a known truth θ\* (fitted posterior mean, or a
prior draw pre-fit), simulates `n_sims` datasets from the model at θ\*
(`pm.do` + prior-predictive on the do-graph), refits each on the same graph
(`pm.observe`), and reports, per target (scalar parameters **and** per-channel
total contributions):

- empirical coverage at 50/80/90/95% with a Jeffreys Monte-Carlo interval —
  "90% intervals covered 63% [51–74%]" is a finding; "covered 83% [70–92%]"
  with 16 sims is noise;
- a **bias vs width decomposition** from z = (posterior mean − θ\*)/posterior
  sd across refits: `bias_z` (≈N(0,1) under health; large ⇒ location off) and
  `z_spread` (≈1 under health; ≫1 ⇒ intervals too narrow = overconfident,
  ≪1 ⇒ conservative);
- the failure-mode guide below.

Also available: the Validation tab's **Coverage** button (background job,
`n_sims=40`), the model op `recovery_coverage_check`, and the Python API:

```python
from mmm_framework.diagnostics import run_recovery_coverage

cov = run_recovery_coverage(model, truth="posterior_mean", n_sims=40)
print(cov.summary())          # per-target coverage + caveats
cov.worst()                   # the target with the lowest 90% coverage
cov.to_dashboard()            # JSON-safe payload
```

### 2. Free coverage numbers from SBC

An SBC run already contains every coverage number: the truth is inside the
central `level` interval exactly when its normalized rank lands in the central
`level` mass. `SBCParamStat.coverage` (via `coverage_from_ranks`) now states
this in user language — `run_calibration_check` output and `SBCResult.summary()`
include "90% interval covers X% [lo–hi]" per parameter.

### 3. External answer keys (the misspecification probe)

Recovery coverage simulates data *from the model*, so it can never indict the
model's structure. For that, fit on synthetic worlds with known causal truth
(`mmm_framework.synth`, `generate_mff()` writes `synthetic_truth.json`) and
compare intervals to the key across seeds/scenarios. Note the `Scenario.violates`
and `representable` fields: many scenarios *deliberately* put the truth outside
the model's hypothesis space — under-coverage there is the expected finding,
not a bug. Make sure the compared estimand matches (see failure mode 7).

## When coverage fails — the failure modes

1. **Approximate fit (MAP / ADVI / Pathfinder).** The most common cause of
   "90% ≈ 50%". These posteriors are fast approximations; MAP has no real
   spread and mean-field ADVI systematically underestimates variance. The
   framework flags this end-to-end (`results.approximate`, report banners) —
   re-fit with NUTS before trusting any interval. Signature: `z_spread ≫ 1`
   with small `bias_z`.
2. **Sampler not converged.** High R-hat / low ESS / divergences mean the
   "posterior" isn't the posterior. Check convergence diagnostics first;
   coverage of a broken sample is meaningless.
3. **Priors too tight (prior–data conflict).** An informative prior centered
   away from the truth drags every refit's posterior toward itself: biased
   location + confident width ⇒ under-coverage. SBC will NOT catch this (it
   draws truths *from* the prior); fixed-truth recovery and the
   parameter-learning diagnostic (prior-dominated posteriors) will. Fix by
   widening or re-justifying priors (`priors.media.<ch>.roi`, prior predictive
   checks).
4. **Inference-engine miscalibration.** Rare with NUTS, real with hard
   geometry (funnels, ridges: adstock↔saturation trade-offs, AR states). SBC
   is the detector; ∪-shaped ranks = overconfident intervals.
5. **Model misspecification.** Wrong adstock/saturation family, missing
   seasonality/trend break, time-varying effects, ignored synergy, wrong
   likelihood (heavy tails). The posterior concentrates on a pseudo-truth;
   with more data the intervals *shrink around the wrong value*, so coverage
   **worsens** as data grows. Invisible to self-simulation; probe with
   PPC/residual checks, spec-curve sensitivity, and synthetic violation worlds.
6. **Unobserved confounding / endogeneity.** Demand-chasing spend makes the
   estimand itself biased — tight intervals around the wrong ROI. Detect with
   the refutation suite / endogeneity check; fix with confounder controls or
   experiment calibration (lift tests), not wider priors.
7. **Estimand mismatch.** `contribution_roi` (in-graph decomposition) and
   `counterfactual_roi` (zero-out predict) are *different numbers by
   construction*; grading one's interval against the other's truth
   manufactures under-coverage. Same for direct-vs-total effects in mediation
   models. Compare like-for-like (the estimands registry documents each).
8. **Monte-Carlo noise.** Coverage estimated from few sims is itself
   uncertain: at true 90% coverage, 10 sims produce ≤7 hits ~7% of the time.
   Always read the binomial interval. (50% at n≥20 is *not* noise —
   P(≤10 hits | n=20, p=0.9) ≈ 1e-8.)
9. **Bayesian ≠ frequentist guarantee (the honest footnote).** A perfectly
   calibrated Bayesian model guarantees nominal coverage *averaged over the
   prior*, not at every fixed θ\*. At a θ\* in the prior's tail, informative
   priors legitimately produce under-coverage; near the prior center,
   over-coverage. Deviations at a well-supported point (the posterior mean)
   or large deviations anywhere are still genuine red flags.

## Reading the output

| Pattern | Diagnosis |
|---|---|
| Coverage low, `z_spread ≫ 1`, `bias_z ≈ 0` | Intervals too narrow → approximate fit, engine miscalibration, or likelihood too confident (e.g. Normal on heavy-tailed data) |
| Coverage low, `bias_z` large, `z_spread ≈ 1` | Location off → prior–data conflict at this θ\*; against external truth: confounding/misspecification |
| Coverage fine here, failed vs answer key | Structural: misspecification / confounding / estimand mismatch |
| Coverage ≈ 100%, `z_spread ≪ 1` | Conservative — intervals wider than needed (weak identification is the usual cause; not dangerous, but decisions get vague) |

## Cost

One refit per simulation, like SBC. Chat default `n_sims=16` (gross failures
only), Validation-tab job 40. Refits use `numpyro` NUTS by default so the
verdict applies to the production posterior; `sampler="advi"` is faster but
measures ADVI's (known-too-narrow) intervals.

Tests: `tests/test_coverage_diagnostics.py`.

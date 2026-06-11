# Structural-violation stress tests

> *"This is only tested on very simple synthetic data."*

That critique is correct, and it matters. The framework's existing synthetic
fixtures draw data from (essentially) the **same generative family the model
assumes** — geometric adstock, concave saturation, additive channels, Gaussian
noise, constant coefficients, exogenous spend. A passing recovery test on that
data proves the *sampler* works when reality matches the model. It says nothing
about what happens on real marketing data, which violates that structure in ways
that bias the answer **silently**: the fit converges, the posterior-predictive
checks pass, the refutation suite is quiet — and the ROI is wrong.

This package generates data that deliberately breaks each structural assumption,
one at a time, with **known ground truth**, then fits the model and scores:

1. **recovery error** — estimated vs. true total contribution;
2. **coverage** — does the true (causal) effect fall inside the model's 90%
   credible interval? (Overconfidence under misspecification — a tight interval
   that *excludes* the truth — is the failure mode "easy data" hides.)
3. **which diagnostics fired** — convergence, PPC, the unobserved-confounding
   robustness value, the causal refutation suite, and prior→posterior learning.

The headline is the set of **silent failures**: scenarios where the answer is
badly wrong yet every routine diagnostic is green.

```
tests/synth/
  dgp.py                 # the data-generating processes (one violation each)
  harness.py             # fit + score: recovery, coverage, diagnostics, matrix
  run_stress_matrix.py   # CLI runner -> results/stress_matrix.{md,csv,json}
  test_dgp.py            # fast (no-MCMC) checks the DGPs break what they claim
  results/               # generated matrix + run log
```

## Running it

```bash
# fast (no fit) — proves each scenario truly violates its assumption
uv run pytest tests/synth/test_dgp.py -q

# the stress matrix (real MCMC; minutes)
uv run python -m tests.synth.run_stress_matrix --quick    # control + top 4
uv run python -m tests.synth.run_stress_matrix --all      # all 13 scenarios
uv run python -m tests.synth.run_stress_matrix clean synergy   # pick scenarios

# fit-time / convergence benchmark
uv run python -m tests.synth.bench

# self-contained HTML report (data + results) -> results/robustness_report.html
uv run python -m tests.synth.make_report_html

# knobs
  --draws/--tune/--chains    main-fit MCMC config (default 600/600/4)
  --ref-draws                refutation refit draws (default 150)
  --no-refutation            skip the (slow) refit-based causal suite
  --legacy-adstock           use the model's default 2-point adstock blend
```

Output lands in `tests/synth/results/stress_matrix.md` (human-readable matrix +
per-channel detail for the failures), plus `.csv` / `.json`.

## How ground truth is defined (and why it's fair)

For every scenario, "true contribution" of a channel is the **counterfactual
zero-out** of its spend, evaluated on the *noiseless structural mean* — exactly
the estimand `BayesianMMM.compute_counterfactual_contributions` reports. Truth
and estimate are therefore the **same quantity on the same (KPI) scale**, so the
gap is a real recovery error, not a units mismatch.

For confounded / endogenous worlds, truth is the **causal** media effect (zeroing
spend does not move the demand-driven baseline). The gap to the estimate *is* the
confounding bias — that is the point.

A scenario is flagged `representable = False` when the truth lies outside the
model's hypothesis space (e.g. a genuinely negative effect under a positive-only
prior). There a large error is expected by construction, and the row is reported
as an *expected* failure, not a silent one.

The **positive control** (`clean`) is data drawn from the model's exact
assumptions. The model recovers it to within ~7% with full 90%-CI coverage — that
is the floor every violation is measured against. Without it the matrix would be
uninterpretable (you couldn't tell a broken assumption from a scoring bug).

## The catalog

Each scenario isolates **one** violation that real marketing data exhibits.

| scenario | real-world mechanism | model assumption broken | expected symptom |
|---|---|---|---|
| `clean` | — (positive control) | none | ~0 error, full coverage, all green |
| `unobserved_confounding` | latent demand drives **both** spend (teams bid harder when demand is high) and sales | exogeneity / no unobserved confounding | demand-chasing channels **over**-attributed; truth outside CI |
| `confounding_controlled` | same world, but a noisy demand proxy is included as a control | — (shows the back-door *closed*) | bias shrinks vs. `unobserved_confounding` |
| `reverse_causality` | budgets paced to recent revenue (spend chases sales) | exogeneity of spend (simultaneity) | spend↔sales feedback inflates effects |
| `multicollinearity` | always-on, synchronized flighting (all channels ramp together) | identifiability (independent spend variation) | per-channel attribution scrambled; wide/unstable CIs |
| `adstock_misspec` | long, delayed carryover (brand TV pays out over months) | geometric carryover truncated at 8 weeks | mis-timed/under-counted carryover |
| `saturation_misspec` | S-shaped response with a low-spend threshold | strictly concave `1-exp` saturation | curvature error, worst at low/high spend |
| `time_varying_beta` | creative fatigue + a mid-series structural break (algo change, COVID) | time-invariant coefficients (stationarity) | one averaged β; sub-period misfit, coverage miss |
| `heavy_tailed_noise` | promo spikes, stockouts, data errors; variance grows with level | Gaussian, homoscedastic likelihood | inflated σ; outliers distort fit (PPC should catch) |
| `synergy` | TV primes search (cross-channel interaction) | additive separability of channels | misattribution; per-channel truths sum to > total media |
| `spend_outliers` | a data-entry spike inflates one channel's max | robustness of per-channel max-normalization | real weeks collapse toward zero on the curve; attenuated effects |
| `negative_effect` | a deep-discount channel cannibalizes / pulls demand forward | positivity of effects (Gamma prior, β ≥ 0) | **unrepresentable** — model cannot reach a negative coefficient |
| `aurora_kitchen_sink` | the showcase world: confounding + mediation + cannibalization + Hill saturation, fit demand-blind | several at once | realistic combined bias |

## Reading the matrix

* `med|err|` / `max|err|` — median / worst per-channel |relative error| on total
  contribution.
* `cover` — fraction of channels whose true contribution lands in the 90% CI.
* `rhat`, `div`, `ppc`, `refut`, `fragile` — the diagnostics. A 🔴 **silent
  failure** is a *representable* scenario with bad recovery (median err > 25%,
  worst channel > 50%, or coverage < 75%) where **the checks an analyst acts on
  are green**: MCMC convergence (`rhat` < 1.05, `div` = 0) and the
  unobserved-confounding robustness value (no `fragile` channel).

`ppc` and `refut` are reported in the matrix but **excluded from the
silent-failure gate**. Both refit / resample at low fidelity and fire on the
*clean* control (false positives — see the `clean` row), so neither is a reliable
all-clear. That they can be simultaneously over-sensitive (firing on clean) and
under-sensitive (missing real confounding) is itself a finding. Where a flagged
silent-failure row shows `ppc`=✗, the failure is silent only to an analyst who
has — reasonably — learned to discount PPC's crying-wolf.

## What the current run found

A full sweep (156 weeks, 600/600/4 MCMC, parametric-geometric adstock, seed 0;
see `results/stress_matrix.md`) puts the 13 scenarios into four buckets. The
positive control recovers to **7%** with **100% coverage** — that is the floor.

**🔴 Silent failures (5) — wrong answer, convergence + robustness value green:**

| scenario | median err | worst channel | coverage | what the model does |
|---|--:|--:|--:|---|
| `unobserved_confounding` | 41% | Search **+153%** | 50% | over-credits the demand-chasing channels (Search/Social), exactly as confounding theory predicts |
| `multicollinearity` | 29% | Display +80% | 100% | scrambles attribution across synchronized channels — total media is ~right (−2%) but the per-channel split is wrong, and nothing flags it |
| `saturation_misspec` | 46% | +63% | **25%** | the S-curve fools the concave model; 3 of 4 truths fall outside the CI — and PPC *and* the refutation suite both pass |
| `spend_outliers` | 46% | −75% | **0%** | one data-entry spike per channel inflates the max-normalizer; every real week collapses toward zero and every channel is under-credited |
| `confounding_controlled` | 11% | Search +79% | 75% | adding the proxy as a `CONFOUNDER` recovers 3 of 4 channels (median 41%→11%), but the noisy proxy leaves residual confounding on the most-targeted channel |

These answer the critique directly: the fit converges (r̂ ≈ 1.01, 0 divergences),
the robustness value looks fine, and the ROI is wrong — quietly.
`saturation_misspec` is the starkest: it passes **even PPC and the refutation
suite** (`ppc`=✓, `refut`=✓), so *every* implemented check is green while 3 of 4
channels fall outside their CI. On `spend_outliers` PPC and a refuter do fire —
but both also fire on the clean control (PPC=✗ on `clean` too), so an analyst has
no reliable way to tell the true alarm from the false one. Only the
experiment-anchored calibration (`calibration/`, off by default) catches these.

**🟡 Bad but *caught* — a diagnostic fired, so not silent:**

| scenario | median err | caught by |
|---|--:|---|
| `adstock_misspec` | 77% | robustness value flags `Search` as fragile |
| `aurora_kitchen_sink` | 88% | r̂ = 1.11 + 108 divergences (also unrepresentable) |

**🟢 Robust at the window-total level:**

`reverse_causality` (14%), `time_varying_beta` (11%), `synergy` (6%),
`heavy_tailed_noise` (3%). Honest nuances (these are estimand-insensitivity, not
proof of robustness):

* `time_varying_beta` / `reverse_causality`: the *total-over-window* contribution
  is recovered even though sub-period effects (and the simultaneity) are not —
  the estimand integrates the drift out. A time-resolved estimand would expose
  these; the contribution total does not.
* `synergy`: the model misattributes the interaction, but the **leave-one-out**
  contribution estimand is self-consistent (zeroing a channel removes its share
  of the synergy in both truth and estimate), so the totals still line up.
* `heavy_tailed_noise`: the Gaussian likelihood recovers the *mean* well; the
  tails inflate σ rather than bias the point estimate.

**⚪ Unrepresentable (expected failures):** `negative_effect` (the positive-only
prior cannot reach a −7,281 contribution; estimates it as ≈0) and
`aurora_kitchen_sink`.

**Takeaway.** The model is well-behaved on data that matches its assumptions
(7% floor, full coverage). Its routine diagnostics catch only the *grossest*
problems — `adstock_misspec` (via the robustness value) and the non-converging
`aurora`. They are blind to **five** realistic pathologies that each produce a
confident, wrong ROI: unobserved confounding, multicollinearity, wrong
saturation shape, spend outliers distorting the normalizer, and even
*imperfectly-controlled* confounding. The standing recommendation in the codebase
— anchor effects with randomized geo-lift / incrementality experiments (see
`calibration/`) — is the right mitigation: a measured lift is the one signal
these failures cannot fake.

## Caveats

* **Fast config.** The default 600/600/4 MCMC is "fast but converged," not a
  production run. Errors are recovery *floors*; a longer run tightens intervals
  (which generally makes coverage misses *worse*, not better).
* **One model config.** All scenarios use the same parametric-geometric-adstock
  model so the matrix is comparable. `--legacy-adstock` runs the framework's
  *default* 2-point blend, which is strictly less flexible (its `clean` floor is
  higher — even well-specified data isn't perfectly recoverable by it).
* **Seeds.** Each factory has a fixed default seed; pass a seed to resample. The
  qualitative pattern (which assumptions cause silent failures) is stable across
  seeds; exact numbers move. The marginal rows (those that trip the gate only via
  one channel's >50% error, e.g. `confounding_controlled`) are the most
  seed-sensitive — re-run them across several seeds before quoting a number.
* **`aurora_kitchen_sink` truth is illustrative, not the zero-out estimand.** Its
  ground truth is the showcase generator's own structural decomposition (a Hill
  curve plus a mediation path), *not* the counterfactual zero-out the other
  scenarios use, so its per-channel errors are not strictly comparable to the
  rest of the matrix. It is `representable=False` and does not converge (r-hat >
  1.05) — it is included as a realistic "everything at once" reference, not a
  clean single-violation measurement.

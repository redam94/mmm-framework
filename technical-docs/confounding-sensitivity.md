# Sensitivity to unmeasured confounding

**Status:** shipped. **Modules:** `diagnostics/bias_sensitivity.py`,
`validation/confounding_sensitivity.py`, `planning/experiment_sensitivity.py`,
`reporting/charts/sensitivity_bias.py`.

## The problem

Every causal number this framework produces rests on an assumption no dataset can check:
that no unmeasured common cause drives both the treatment and the outcome. For an MMM that
assumption is **unobserved demand** — budgets rise when demand is expected to rise, so spend
correlates with a latent driver of the KPI that no adjustment set removes. For a
matched-market geo readout it is **parallel trends**. A good fit, tight intervals and passing
posterior-predictive checks are all perfectly compatible with the assumption being false.

Before this work the framework quantified that exposure one way and on one side:
`validation/sensitivity_unobserved.py` reports a Cinelli–Hazlett **robustness value** per
channel, on the *coefficient* scale, with no way to say whether the confounder strength it
demands is plausible. Experiments had nothing at all — a matched-market DiD readout anchored
the model at its stated standard error exactly as if it had been randomized.

## The device

Following [the PyMC Labs treatment][pymc], decompose the observed effect into a causal part
and a bias part, put a **prior** on the bias, and report which conclusions survive which
priors:

```
observed = tau + beta,    beta ~ N(mu, sigma)
```

Three outputs follow.

**A tipping point.** The bias commitment at which `P(tau > reference)` falls through the
decision threshold: *"TV's ROI would have to be overstated by more than 24% of its own size
before it stops clearing break-even."*

**A sensitivity surface.** The same probability over a grid of `(mu, sigma)` commitments — a
partition of the analyst positions that do and do not support the conclusion, rather than a
single verdict.

**A benchmark.** What makes the tipping point an argument rather than a slider: Cinelli–Hazlett
bounding prices a hypothetical confounder against the covariates that *were* measured. *"A
confounder as strong as Price implies 9% — well inside the 24% it would take."*

[pymc]: https://www.pymc.io/projects/examples/en/latest/causal_inference/sensitivity_unmeasured_confounding.html

## Layer 1 — the engine (`diagnostics/bias_sensitivity.py`)

Model-free, numpy + scipy only, no PyMC.

### Exact, not Monte-Carlo

The de-biased posterior is a Gaussian mixture over draws that already exist, so nothing
samples. For draws `x_d`, `tau | x_d ~ N(x_d - mu, sigma^2)` and therefore

```
P(tau > r) = mean_d Phi((x_d - mu - r) / sigma)
```

in closed form. Three consequences, all load-bearing: results are **deterministic** (no seed
to reproduce), they are **exact** rather than noisy at small draw counts — which is what makes
a bisected tipping point trustworthy — and the identical arithmetic **runs in JavaScript**, so
a report can recompute a tipping point in the browser without another posterior pass.
Interval endpoints come from bisecting the same mixture CDF.

### Two scales, and why there is no third

`scale="absolute"` puts the bias in the estimand's units. `scale="fraction_of_mean"` uses
`tau_d = x_d - b * |mean(x)|`, which makes a commitment comparable across channels and lines
up with the Cinelli–Hazlett bound (a bias expressed as a *fraction* of the estimate). Both are
affine in the draws, so the de-biased quantity stays exactly Gaussian per draw.

A **per-draw multiplicative** form `tau_d = x_d * (1 - b_d)` is the obvious way to say
"relative" and is deliberately **not offered**:

* An efficiency channel's break-even reference is **0**, and every media coefficient here has
  positive support by construction (`Gamma`, `LogNormal` on ROI, or `exp(...)`), so `x_d > 0`
  for every draw and `P(tau > 0) = P(b < 1)` **exactly** — the same number for every channel
  in the portfolio, with no data in it at all.
* On draws that cross zero (`counterfactual_roi` and `marginal_roas` numerators routinely do)
  the bias direction flips with the sign of the draw.
* The product of two normals is skewed with heavy tails, so the closed forms stop being exact.

`fraction_of_mean` has none of these. At reference 0 it is also scale-invariant — two
posteriors differing only in units agree, and what separates them is the shape of the
posterior relative to its own mean. Gated by
`tests/test_bias_sensitivity.py::TestReferenceZeroIsNotDataFree`.

### The flat-tau condition

Subtracting an independent bias is exactly right when the input draws are a **flat-prior**
posterior of `tau + beta`: then `∫ exp(-(d - tau - beta)^2 / 2se^2) dtau` is constant in
`beta`, so `p(beta | d) = pi(beta)` and the data are provably uninformative about the bias.
`bias_adjusted_moments(..., tau_prior_sd=None)` takes that limit and reduces exactly to
`N(d - mu, se^2 + sigma^2)` — pinned against the finite-prior conjugate solve at
`tau_prior_sd=1e6`.

An MMM posterior is **not** flat: it carries a `LogNormal(0,1)` ROI or `Gamma(mu=1.5,sigma=1)`
coefficient prior. The effective prior on `tau` is therefore the media prior convolved with
the bias prior — coherent, but *undeclared*. So `bias_sensitivity_report` accepts
`prior_draws` and, when given them, reports what the commitment implied **before any data**
(`implied_prior_prob`). A conclusion whose prior already clears the threshold was not
established by this analysis, and the report says so.

### Guards that ship

| Failure it prevents | Guard |
|---|---|
| Reporting "robust" for a conclusion that was never scanned past a narrow range | `TippingPoint.max_scanned` on every instance; `describe()` says "still supported at the widest bias scanned (X)" |
| Confusing "not supported even at zero bias" with "survived the whole scan" | `already_below` separates them; both give `value=None`-adjacent states that must be read differently |
| Bisecting a non-monotone probability | Grid **scan** for the first crossing, then bisect inside that bracket; realized monotonicity reported on `TippingPoint.monotone` |
| A tight prior manufacturing a high tipping point | `prior_dominance_caveat` refuses to quote resilience below 0.20 prior→posterior contraction |
| A positive-support prior making `P(above)` free | `positivity_constrained` flag when `P = 1.0` exactly, plus a caveat naming the mechanism |
| A ladder of guesses read as measurements | `BiasPrior.source` on every prior; `is_measured` property; a caveat when nothing was measured |
| An E-value on a quantity it is not defined for | `evalue` accepts risk/rate/prevalence ratios only and **refuses** anything else with a reason pointing at the tipping point. An ROI is a ratio of continuous quantities, not a risk ratio |

**Verdict vocabulary** is `overturned | fragile | resilient | not_assessable`. `resilient`
deliberately avoids the word "robust": it claims only that the conclusion survived the range
actually scanned.

## Layer 2 — MMM results (`validation/confounding_sensitivity.py`)

### Decision scale, no refit

`run_confounding_sensitivity(model)` applies the engine per channel to the contribution-ROI
posterior (`validation.spec_curve.channel_roi_draws` — the same numerator and denominator as
the `contribution_roi` estimand) and judges it against the measurement-aware break-even
reference from `reporting.helpers.measurement.resolve_channel_divisor`, so an
impression-measured channel is compared to 0 rather than to 1.

### The benchmark, and its exact identity

`benchmark_bias_priors(model)` builds the linear design at the model's posterior-mean transform
point via `frequentist/design.py::build_design_matrix` — whose documented invariant is that
`X @ theta` reproduces the graph's `mu` to 1e-12 with adstock and saturation fixed — then fits
ordinary least squares and converts each observed covariate's partial `R^2` into a bound.

```
r2dz_x   = kd * r2dxj_x / (1 - r2dxj_x)
r2zxj_xd = kd * r2dxj_x^2 / ((1 - kd*r2dxj_x) * (1 - r2dxj_x))
r2yz_dx  = ((sqrt(ky) + sqrt(r2zxj_xd)) / sqrt(1 - r2zxj_xd))^2 * r2yxj_dx / (1 - r2yxj_dx)
|bias|   = se * sqrt(df) * sqrt(r2yz_dx * r2dz_x / (1 - r2dz_x))
adj_se   = se * sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * sqrt(df / (df - 1))
```

When `Z` really *is* the omitted confounder the bias formula holds **with equality**, which is
how the algebra is validated: fit the same data with and without `Z`, and the predicted bias
must equal the realized one. It does, to ~1e-15
(`tests/test_confounding_sensitivity.py::TestOmittedVariableBiasIdentity`). That is a stronger
gate than any remembered reference number.

**The binding validity condition is `r2dxj_x < 1 / (1 + kd)`** — at `kd = 1` that is
`r2dxj_x < 0.5`, *not* `< 1`. The naive `kd * r2dxj_x < 1` is the non-binding one. Past the
real threshold the published formula square-roots a negative number and returns `NaN`, and a
`NaN` compares `False` against every fragility threshold — i.e. it would be read as "not
fragile". So a breach is an explicit refusal naming the largest admissible `kd`, never a clip.
Only `r2yz_dx > 1` is clipped, and then `saturated=True` marks the bound as degenerate.

### Why OLS and not the posterior sd

The identity is calibrated on the OLS fact that `se * sqrt(df) = ||y_res|| / ||d_res||`.
Substituting a Bayesian posterior sd breaks it by whatever factor the prior contributed — and
in the dangerous direction, because this framework's media priors are informative with
positive support, so the posterior sd is *smaller*, the implied bias comes out *too small*, and
the model reports robustness the data did not supply.

Note this is the **mirror image** of the robustness value's caveat, which is *inflated* by a
tight prior because `RV` rises with `|mean| / sd`. Two opposite mechanisms, one direction: a
tighter prior makes both numbers look more reassuring. A reader who has internalised one
caveat must not assume the other path inherits it.

### Rank deficiency is not optional

Two independent things make the raw design singular: a full geo/product dummy set beside an
unpenalized intercept (the frequentist path relies on the ridge penalty for a unique split —
OLS has no tie-break), and a control that happens to be an exact function of the model's own
basis. `make_unobserved_confounding`'s `Price` is literally `12 + 0.5*cos(2*pi*t/52)`, an exact
yearly Fourier term. `pinv` would return the minimum-norm solution and every partial `R^2`
would depend on that arbitrary choice.

`_select_independent_columns` runs modified Gram–Schmidt in a **priority order** —
intercept, trend, seasonality, geo, product, media, controls. Dropping a redundant column
leaves the column *space* untouched, so no partial `R^2` changes; what the order buys is
*which* member of a redundant pair is dropped, so a control that merely duplicates the
seasonal basis is reported as uninformative rather than allowed to stand in for it. A dropped
**media** column is a refusal: that channel is not separately identified.

**Benchmark covariates default to `CausalControlRole.CONFOUNDER` controls.** Benchmarking
against a mediator or collider produces a bound with no causal meaning, and the framework
already knows which is which.

## Layer 3 — experiments (`planning/experiment_sensitivity.py`)

### The threat follows assignment, not the estimator

| Design | Threat | Floor | Why |
|---|---|---|---|
| Matched-market DiD | unmeasured confounding | 0.25 | Nothing randomized; parallel trends is an assumption |
| Randomized geo lift | **interference** | 0.10 | Assignment is exogenous — confounding is *not* the threat. Spillover attenuates, so the correction is toward a larger effect |
| Synthetic control | unmeasured confounding | 0.20 | Placebos bound extrapolation, not confounding |
| TBR | unmeasured confounding | 0.20 | The interval prices sampling noise, not a regime change |
| GBR / regression-adjusted geo | unmeasured confounding | 0.15 | Conditions on the pre-period only |
| Switchback / flighting | interference | 0.15 / 0.20 | Carryover across block boundaries |
| Ghost ads | **external validity** | 0.10 | Individual randomization — there is no unmeasured confounder. The gap is platform-addressable users vs an aggregate panel |

`threat_for` re-points a design flagged `randomized=True` at the interference threat even when
its estimator key is a DiD one, and widens the floor when the design's own
`parallel_trends_warning` or `carryover_warning` is set.

### The measured width, and the double-counting trap

`planning.design._placebo_did` scores every historical window as though it were the
experiment. That spread contains sampling noise *and* whatever differential drift the pair
structure carries, and the analytic DiD standard error models only the first — so the
**excess** is the defensible prior:

```
bias_sigma = sqrt(max(placebo_sd^2 - analytic_sd^2, 0))
```

**But `geo_lift_design` already multiplies its own `se_roas` by
`calibration_factor = placebo_sd / analytic_sd`** whenever it has ≥12 placebo windows. If the
readout's standard error came from that design, the inflation is already inside it and adding
the excess again inflates every calibrated interval for no reason.
`derive_bias_prior` compares the readout's SE against the design's and reports which regime it
is in (`DerivedBiasPrior.absorbed`) rather than guessing. The floor still applies either way —
absorbing sampling inflation does not absorb spillover.

### Design-derived widths are ROAS-only

A design's evidence lives on its own scale and moving it is where the silent errors are: the
placebo runs on the *average matched pair* while most estimators return the *whole treated
cell*; `weekly_spend_delta` is stored unsigned even though a holdout's delta is negative; and
planned spend is not realized spend. Since `geo_lift_design` already reports `se_roas`,
deriving a ROAS-scale width needs **no conversion at all**. So automatic derivation is offered
for `roas` and refused for `contribution` and `mroas`, which need `explicit_sigma`. The floor,
being relative to the measured value, still applies to every estimand.

The default `mu` is **0**. A placebo spread is a width, not a signed offset, and inferring a
direction would mean dividing by the unsigned `weekly_spend_delta`.

### Propagation into the model

`ExperimentMeasurement` gains `bias_mu`, `bias_sigma`, `bias_scale`, `bias_source`.
`attach_experiment_likelihood` observes `value - bias_mu` at `hypot(se, bias_sigma)`, so a
quasi-experimental readout anchors the model less firmly than a randomized one.

* **Defaults are byte-identical.** The un-biased branch keeps the original expressions rather
  than `hypot(se, 0.0)` and `value - 0.0`, neither of which is guaranteed to round-trip.
  Gated by a log-probability comparison against a hand-built reference graph.
* **Lognormal bias lives on the log scale.** An additive shift under a multiplicative error
  model is a category error and can drive the observed value non-positive; `__post_init__`
  refuses the combination.
* **No `pm.Deterministic` for the bias.** It is parameter-independent, and a constant
  Deterministic breaks Pathfinder's trace conversion (see `model/base.py::_anchored_det`).
* **The registry keeps the raw readout.** Only the staged spec entry carries the assumption, so
  the same measurement can be staged under several assumed biases, and the information-decay
  and EIG maths that read `se` are not silently double-counting.
  `apply_experiment_calibration(bias_mode="stored"|"off")`.

## Layer 4 — the in-graph sweep (opt-in, exact)

Layers 1–3 re-weight a fitted posterior, which shifts and widens an estimand but **cannot move a
coefficient relative to the controls**. `run_confounder_sweep` puts a hypothetical confounder
*into the graph* and re-fits at each assumed strength, so the media coefficients genuinely move
— through the nonlinear adstock and saturation, and including being partly absorbed by the
trend, which is itself worth seeing. Measured on `unobserved_confounding`: at an assumed partial
`R^2` of 0.25, TV falls 2.83 → 2.18 while Display *rises* 1.95 → 2.19. A convolution cannot
produce that; it moves every channel the same way.

### The confounder is fixed data, not a sampled latent

`BayesianMMM.add_latent_confounder(u_scaled)` attaches a period-axis vector behind a `pm.Data`
container; `_build_latent_confounder` returns `None` when none is attached, so the default graph
is **byte-identical** (the term is absent, not zero — the discipline the event/interaction/lever
blocks already follow, gated by a named-variable comparison).

Sampling the confounder's loading instead would be a *model of* confounding, not a sensitivity
analysis: the likelihood would choose how much to admit, and any spend-predictable structure
could be relabelled as `U` and shrink the media coefficients for free. Fixing both associations
at the grid point — Imbens-style — is what makes the sweep answer *"if the world were like
this, what would I have concluded?"*.

### Construction

`build_confounder` residualizes the KPI and each targeted channel's spend against the
**adjustment basis** (intercept, trend, seasonality, controls) on the period axis, builds an
AR(1) series with persistence matched to the KPI residual — a white-noise confounder is
trivially absorbed and would understate the exposure — mixes it toward the spend residuals,
orthogonalizes against the basis and standardizes **in numpy**. Orthogonalization is
load-bearing twice: a confounder that is a linear function of an observed control is not
*unobserved*, and without it the trend and seasonality simply re-fit to absorb the injected term
and silently nullify the sweep.

**`strength` has to mean what it says.** Mixing at `w = sqrt(strength)` looks right and is not:
the driver averages the targeted channels, so its correlation with any one of them is diluted.
Measured before the fix: 0.011–0.049 delivered against an assumed 0.15. The mix weight is now
bisected until the *mean* delivered partial `R^2` matches the assumption (0.050 / 0.150 / 0.300
against 0.05 / 0.15 / 0.30), and per-channel spread is reported in `delivered_r2_t` so
absorption stays visible instead of silent.

The confounder is kept **out of** `channel_contributions` / `media_total` (it is a baseline
demand driver, not a media stream — folding it in would double-count it in every ROI and
estimand) and out of `X_controls` (which would change `n_controls`, the `control` coord and the
shape of `control_contributions`). It gets its own deterministic beside `controls_total`.

### Refusals

Named, in this repo's established style: extension models (no single "the" media coefficient),
the frequentist paradigm (a ridge fit is not a posterior), a multiplicative specification (`mu`
is a log there, so an additive term is a multiplicative confounder and the assumed partial `R^2`
is not the quantity computed), and a Gaussian-process trend (flexible enough to absorb any
smooth confounder and report robustness that is really the trend re-fitting).

The sweep is **national** — one series over time. A geo-varying confounder is a strictly
stronger threat and is not covered.

## Surfaces

| Surface | Entry point |
|---|---|
| Library | `run_confounding_sensitivity(model)`, `experiment_sensitivity(...)` |
| Agent op | `confounding_sensitivity` (`agents/model_ops.py::OPS`) |
| Agent tool | `run_sensitivity_analysis` (`agents/causal_tools.py`), `persist_check="sensitivity"` |
| REST | `POST /projects/{id}/validate` with `check="sensitivity"` |
| UI | Oracle → Validation tab → **Sensitivity** |
| Report | `CausalAssumptionsSection` — tipping-point table beneath the robustness values |
| Charts | `create_tipping_point_chart`, `create_bias_curve_chart`, `create_bias_contour_chart` |

## Tests

| File | Covers |
|---|---|
| `tests/test_bias_sensitivity.py` | Engine: flat-tau limit vs conjugate solve, mixture arithmetic vs brute-force MC, tipping-point accuracy, E-value against the published worked example, and the reference-0 regression gate |
| `tests/test_confounding_sensitivity.py` | The OVB identity to ~1e-15, sensemakr validity conditions, column-selection priority and span preservation, planted-truth benchmarks on `unobserved_confounding` / `confounding_controlled`, report section and charts |
| `tests/test_experiment_sensitivity.py` | Threat model, double-counting guard, unit refusals, byte-identical default graph, lognormal log-scale bias, and that a biased measurement pulls the posterior less |

## What this cannot do

Layers 1–3 re-weight a fitted posterior: they shift and widen an estimand but cannot move a
coefficient relative to the controls. Layer 4 can, at the cost of one re-fit per grid point —
and is national, so a geo-varying confounder remains out of reach.

And the standing caveat, which every surface repeats: a tipping point is an argument that a
confounder large enough to matter is implausible. It is never evidence that an effect is
causal. Only a randomized experiment gives that.

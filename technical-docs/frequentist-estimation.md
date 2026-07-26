# Frequentist estimation — ridge, constrained QP, and bootstrap intervals

Design spec for epic [#180](https://github.com/redam94/mmm-framework/issues/180).
**Written before implementation** ([#182](https://github.com/redam94/mmm-framework/issues/182)),
because the design questions here are load-bearing: get them wrong and the result
is a second inference path that quietly lies about its own uncertainty.

`InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY` have existed as enum
values since early in the project and were never implemented. Until v1.2.0 they
silently fitted Bayesian NUTS; since v1.2.0 they refuse
([#181](https://github.com/redam94/mmm-framework/issues/181)). This document
specifies what they will do instead.

Every empirical number below was measured on this codebase, not assumed. The
measurement scripts are described in [§10](#10-how-the-numbers-here-were-measured).

---

## 0. Assumed semantics (read before use)

Matching the convention in `calibration/likelihood.py`, these are the
load-bearing assumptions. If one is false for your data, the output is wrong in
a way no diagnostic here will catch.

* **Transforms are fixed, not estimated.** The frequentist path holds each
  channel's adstock and saturation parameters at a searched point `(α̂, λ̂)` and
  estimates the remaining coefficients conditional on them. Every interval it
  reports is therefore **conditional on the selected transforms** unless
  `refit_search=True`. The Bayesian path integrates over them; this one does not.
* **The intervals are confidence intervals, not credible intervals.** They come
  from resampling the data, not from a posterior. "There is a 90% probability the
  ROI lies in this range" is **false** for them. See [§5](#5-uncertainty).
* **Ridge is biased by construction.** A percentile bootstrap interval around a
  shrunk estimator covers the *estimator's* sampling distribution, not the true
  parameter. Shrinkage that helps prediction hurts interval coverage for β.
* **Least squares assumes a Gaussian-scale outcome.** Only likelihood families
  that z-score `y` are in scope. Count/bounded families are refused, not
  approximated.
* **Selection on predictive error is not selection on the causal estimand.** The
  transform search ([#184](https://github.com/redam94/mmm-framework/issues/184))
  optimizes out-of-sample prediction. A specification that predicts better can
  attribute worse. This is the same objection raised against LOO-stacking in the
  v1.1.0 `spec_curve` fix, and it applies here with full force.

---

## 1. What this buys over `fit(method="map")`

The epic opens with a fair challenge: **ridge regression is MAP estimation under
Gaussian priors**, so `fit(method="map")` should already be ridge, and adding
`frequentist_ridge` risks shipping a synonym.

That equivalence is a real theorem. **It does not describe this framework's
default configuration**, and the difference is the epic's justification.

Enumerating the actual prior family of every free parameter in a default core
model:

| block | prior as built | ridge-equivalent? |
|---|---|---|
| `intercept` | `Normal(0, 0.5)` (`base.py:2167`) | **yes** — L2 |
| `trend_slope` | `Normal(growth_prior_mu, growth_prior_sigma)` (`base.py:1335`) | yes, but toward a **non-zero centre** ⇒ generalized Tikhonov `‖θ−θ₀‖²`, not plain ridge |
| `season_<name>` | `Normal(0, prior_sigma)` (`base.py:2192`) | **yes** — L2 |
| `beta_controls` | `Normal(0, role_widths)` (`base.py:962`) | **yes** — L2 with a **per-column** width; confounders get `_CONFOUNDER_PRIOR_SIGMA` (wide ⇒ weak penalty), precision controls the historical narrow one |
| `beta_<ch>` — `media_prior_mode="coefficient"` (library default) | **`Gamma(μ=1.5, σ=1)`** (`base.py:2503`) | **no** — positivity-constrained, not Gaussian |
| `roi_<ch>` — `media_prior_mode="roi"` (**the agent/spec default**) | **`LogNormal(0, 1)`** (`base.py:2473`) | **no** |
| `trend_delta` (piecewise) | **`Laplace(0, b)`** (`base.py:1367`) | **no** — that is an L1/lasso penalty |
| `geo_offset`/`product_offset`, `spline_coef` | `HalfNormal(σ) × Normal(0,1)` non-centered | **no** — a scale mixture, i.e. a *learned* penalty strength |

So on the block that anyone actually cares about — media coefficients —
`fit(method="map")` is **not** ridge. It is MAP under a Gamma or LogNormal prior:
a positivity-constrained, log-scale-penalized point estimate. A genuine ridge
path is a different estimator, not a rename.

With that settled, the three things this path adds:

1. **Transform hyperparameters by search, not by prior.** The Bayesian path
   estimates adstock α and saturation λ *jointly* with β, letting the prior and
   the likelihood trade off. The frequentist path **fixes** them per candidate
   and solves the resulting linear problem in closed form. Different estimator,
   different failure modes, and fast — the inner solve is one `scipy` call.
2. **Frequentist intervals.** A bootstrap sampling distribution, which is not a
   posterior and cannot reuse its semantics or its labels.
3. **Hard constraints.** A prior makes a negative coefficient *unlikely*
   (`HalfNormal`); it cannot make it *impossible*, and it cannot say "these three
   channels' contributions sum to the number finance already booked". A convex
   program says both exactly. This is the one capability with no Bayesian
   equivalent, and it is the strongest single justification in the epic.

**Non-goal.** This does not replace the Bayesian path as the recommended default.
It is a fast, constrained alternative and a triangulation tool.

---

## 2. The premise, verified: `mu` is `X @ θ`

Given per-channel transforms held fixed, the core model's mean is **exactly** a
linear function of the remaining parameters. This was measured, not assumed:
a numpy design matrix was built out of the graph, the graph's `mu` was evaluated
at a fixed parameter point, and the two compared.

| case | design cols | `max |mu_graph − X@θ|` |
|---|---|---|
| national · geometric · logistic · linear trend | 9 | 2.77e-13 |
| national · weibull · hill · spline trend | 22 | 6.33e-13 |
| national · delayed · root · piecewise trend | 20 | 3.54e-13 |
| national · geometric · tanh · no trend | 8 | 3.30e-13 |
| national · geometric · michaelis_menten · linear | 9 | 1.88e-13 |
| **geo (3 geos × 104 wk) · geometric · logistic · linear** | 12 | 2.77e-13 |

Panels were 104 weeks, 2 channels with one 25%-flighted (so zero-spend rows are
exercised), 1 control, yearly Fourier seasonality.

Two consequences for [#183](https://github.com/redam94/mmm-framework/issues/183):

* **The achievable tolerance is ~1e-12, not the 1e-10 the issue proposed.** The
  equivalence test should assert `1e-12`; anything looser would let a real
  regression through.
* **Geo panels are in scope.** The issue offered "handled, or explicitly rejected
  in v1". They are handled — see [§4](#4-the-design-matrix).

`mu` itself is not a named node; it is reconstructed as the sum of the component
Deterministics (`intercept_component + trend_component + seasonality_component +
geo_component + product_component + media_total + controls_total`), which is `mu`
by construction — compare `base.py:2588`.

---

## 3. What is linear, and what is refused

### Linear given fixed transforms — in scope

| term | design contribution |
|---|---|
| `intercept` | a constant column |
| trend `none` / `linear` | none / `t_scaled[time_idx]` |
| trend `piecewise` | expand `(k + A·δ)t + m + A·(−s∘δ)` into `k·[t] + m·[1] + Σⱼ δⱼ·[Aᵢⱼ(tᵢ − sⱼ)]` |
| trend `spline` | the **mean-centred** B-spline basis; linear in the composite `spline_coef = spline_scale · cumsum(raw)` |
| seasonality | `fourier_features[time_idx]`, one column per Fourier term |
| geo / product | dummy columns; linear in the **composite** `geo_sigma · geo_offset[g]` |
| media | one column per channel, `sat(adstock(x_c))`, constant once transforms are fixed — **scaled into ROI space**, below |
| controls | `X_controls` as standardized |
| events (#143) | `event_features[time_idx]` |
| channel interactions (#142) | `sat_i · sat_j` — a product of two *constant* columns, so still one linear column |
| price lever (#138) | `log(max(price, 1e-9) / reference)` |
| promo lever (#138) | the normalized promo series — **only when `adstock_lmax <= 1`**; above that `promo_alpha_<v> ~ Beta(1,3)` is a free carryover parameter and the term leaves the linear family |

Four things inside that table need stating:

* **`trend_m` is exactly collinear with `intercept`.** Both are constant columns.
  The Bayesian model tolerates this because independent priors regularize the
  split; a design matrix does not. The two must be merged into a single intercept
  column, and the reported `trend_m` reconstructed as 0.
* **The faithful penalty is not always L2.** Piecewise `trend_delta` has a Laplace
  prior (L1) and the spline coefficients are a random walk (a first-difference
  P-spline penalty). Applying plain ridge to those blocks is a *different*
  regularizer than the Bayesian path uses. v1 penalizes them with L2 and
  **says so** rather than silently implying equivalence.
* **Fit media in ROI space.** Under `media_prior_mode="roi"` (the agent default)
  `beta_<ch>` is a Deterministic `roi · divisor / (y_std · Σsat_ref + 1e-9)`. With
  transforms fixed `Σsat_ref` is a **known scalar** — it is built from a frozen
  `pt.constant` copy of the training media — so `beta_c = c_c · roi_c` for a known
  `c_c`, a pure column rescale. Set `X[:, c] = c_c · sat_c` so the solved
  coefficient *is* `roi_c`, and emit both `roi_<ch>` and `beta_<ch>`. This is not
  cosmetic: `config/model.py:172` documents that the ROI parameterization exists
  because a coefficient-scale prior implies **arbitrary, spend-dependent** prior
  ROIs across channels — and a single ridge penalty on `β` has exactly that defect,
  penalizing each channel's ROI in proportion to its spend. Under
  `media_prior_mode="coefficient"`, `c_c = 1` and the columns are the raw saturated
  series.
* **Geo dummies are identified only by the penalty.** The graph has **no per-geo
  intercept** — `intercept` is one pooled scalar and geo enters purely as a variance
  component with no sum-to-zero constraint, so the dummies are exactly collinear
  with the intercept in an *unpenalized* design. Leaving the intercept unpenalized
  while penalizing the dummies makes the split unique and shrinks geo effects toward
  zero as the hierarchy intends — the standard ridge↔BLUP correspondence. The honest
  difference: the Bayesian path **estimates** the pooling strength (`geo_sigma`);
  ridge fixes it via the penalty. Report that rather than implying equivalence.

### Not linear — refused loudly in v1

Each of these raises a `NotImplementedError` naming the feature and pointing at
the Bayesian path, rather than silently dropping the term:

| feature | why it is not linear |
|---|---|
| **GP trend** (`TrendType.GAUSSIAN_PROCESS`) | the HSGP spectral weights depend on `gp_lengthscale` / `gp_amplitude`, which are estimated and are not in the (adstock, saturation) search space |
| **grouped media betas** (DF-2 `parent_channel`) | `pt.exp(grouped_log_beta)` — exponential of a parameter |
| **legacy (non-parametric) adstock** | `adstock_mix ~ Beta(2,2)` blends two precomputed IIR series and normalizes by `_media_max` (the *adstocked* max) rather than `_media_raw_max` — a different model family, not a fixable parameterization |
| **multiplicative specification** | linear on standardized `log(y)`, so it is tractable; deferred because the back-transform makes intervals asymmetric and `predict` semantics diverge — scope, not mathematics |
| **reach / frequency channels** (#141) | `_channel_media_input` multiplies the media column by an in-graph frequency gain `g(freq)` carrying its own RVs |
| **non-Gaussian likelihoods** | least squares is the wrong objective; `_prepare_data` does not even standardize `y` for count/bounded families |
| **time-varying coefficients** (#137) | linear in `beta_t`, but the random-walk prior is what identifies it; plain ridge would not |
| **V3 per-geo betas** (`vary_media_by_geo`) | linear, but expands to `n_geos × n_channels` columns and discards the partial pooling that makes it work |
| **experiment-calibration likelihoods** | the three estimands (contribution / ROAS / mROAS) *are* linear in β given fixed transforms and could be folded in as weighted extra rows — but the `1/se²` weighting interacts with the ridge penalty scale with no defensible default. **Refuse loudly; never drop silently**, since dropping discards the randomized incrementality evidence the Bayesian fit used |
| **control selection** (horseshoe / spike-slab) | the design stays linear but the penalty is adaptive, not L2 |
| **extension models** (Nested / MV / Combined / Structural) | bespoke graphs that are not linear given fixed transforms — stated in the epic and confirmed |

The last three are *deferrals*, not impossibilities; the first five are
structural. TVP and per-geo betas become tractable with a difference penalty and
a group penalty respectively, and that is the natural v2 extension.

---

## 4. The design matrix

`src/mmm_framework/frequentist/design.py`

```python
build_design_matrix(
    panel, alpha: dict[str, dict], lam: dict[str, dict], *,
    model_config, trend_config,
) -> DesignMatrix
```

`DesignMatrix` carries `X`, the column→parameter mapping, the standardization
used, and the `penalize` mask ([§6](#6-ridge)).

### Reuse adstock; do **not** reuse `transforms/saturation.py`

This is the single most important implementation instruction in the spec.

**Adstock is clean.** The numpy primitives agree with the graph:

| check | max abs error |
|---|---|
| `parametric_adstock` vs `parametric_adstock_pt`, 3 kinds × `normalize ∈ {True, False}` | ≤ 7.7e-13 |
| `adstock_weights` vs `adstock_weights_pt` | ≤ 3.0e-13 |
| per-cell numpy loop vs `parametric_adstock_panel_pt` | 3.06e-13 |

**Saturation is not.** `_apply_saturation_pt` (`base.py:194`) is the graph's single
source of truth, and the public numpy twins in `transforms/saturation.py` are
subtly different:

| family | numpy vs graph | cause |
|---|---|---|
| `logistic` | **2.06e-09** once `λ·x > 20` | the graph clips the **exponent** at −20 (`pt.clip(-sat_lam*x, -20, 0)`); numpy clips **x** at 0 and never saturates |
| `root` | **3.16e-05** at *every zero-spend row* | the graph does `pt.maximum(x, 1e-9)**k` → `1e-9**0.5 = 3.16e-5`; numpy does `clip(x,0,None)**k` → `0` |
| `hill`, `michaelis_menten`, `tanh` | no twin in `transforms/saturation.py` | — |

Both drifts are fatal to a 1e-12 test, and the `logistic` one is not exotic: the
search in #184 ranges over λ on media normalized to ~[0,1], so `λ > 20` is inside
any sane search space.

**But a graph-faithful numpy implementation already exists elsewhere.**
`validation/backtest.py::PosteriorForecaster._saturate` (`:373-393`) transcribes
logistic / hill / michaelis_menten / tanh **with the graph's clips intact**, and
`:339-371` already performs exactly the normalize-then-window adstock construction
#183 needs. `PosteriorForecaster` is, in effect, a partial out-of-graph design
matrix that has been in the repo since the backtest work.

⇒ **extract, do not duplicate.** `frequentist/_transforms.py` becomes the single
numpy mirror of `_apply_saturation_pt`, `PosteriorForecaster` is refactored to
import it, and the fidelity test guards both callers at once. Writing a third
implementation would give the codebase three saturation definitions to keep in
sync, which is how the two drifts above happened in the first place.

Note `_saturate` covers four families — **`root` is missing there too**, so the
extracted module adds it (with `pt.maximum(x, 1e-9)`, per the graph).

`transforms/saturation.py` is **left alone**: it is public API with its own
documented semantics, and "fixing" it would be a silent behavior change to every
existing caller.

### Panel handling

Adstock must be applied **per cell**, looping over
`cell_idx = geo_idx * n_products + product_idx` and sorting by `time_idx` within
each cell. Adstocking the flat stacked series instead is wrong by **0.256** on a
3-geo panel — carryover bleeds across geographies.

### Standardization

Matching `_prepare_data` exactly, because a penalty is not scale-invariant:

* `y` — z-scored, `y_std = y.std() + 1e-8` (**population** std, `ddof=0`). The
  multiplicative specification z-scores `log(y)` instead.
* media — per-channel `X / (raw_max + 1e-8)`, applied to the **raw** series
  *before* adstock (the parametric path feeds normalized raw media and adstocks
  in-graph).
* controls — z-scored, `(X − control_mean) / (control_std + 1e-8)`.

Because media columns land in `[0,1]` and controls at unit variance, **a single
scalar penalty is already comparable across the media block**. That is what makes
the scale-invariance requirement in #185 testable: changing a channel's units
changes `raw_max` proportionally and must leave the fitted contribution unchanged.

### The equivalence test is the deliverable

`tests/frequentist/test_design_equivalence.py` parameterizes over
{adstock family} × {saturation family} × {trend family} × {national, geo},
builds the PyMC graph, substitutes a fixed parameter point for every free RV,
evaluates the component Deterministics, and asserts
`max|mu_graph − X @ θ| < 1e-12`.

If this test drifts, a frequentist fit and a Bayesian fit stop being comparable
and every benchmark between them becomes meaningless. It is the load-bearing
test of the whole epic.

---

## 5. Uncertainty

A ridge fit is a point. Everything downstream in this framework consumes draws,
so the bootstrap must produce them — honestly.

### The estimator

`src/mmm_framework/frequentist/bootstrap.py`

```python
bootstrap_fit(panel, *, n_boot=500, block_length=None,
              refit_search=False, seed=None) -> tuple[az.InferenceData, dict]
```

**Moving-block residual bootstrap.** The design matrix is deterministic once
transforms are fixed, so the resampling target is the residual series: draw
overlapping blocks of residuals, lay them over the fitted values, re-solve. A
pairs bootstrap would destroy the time ordering that the whole exercise is about.

**Block length is estimated, not constant.** MMM residuals are serially
correlated; an iid residual bootstrap treats each week as exchangeable and
produces intervals that are too narrow — the same error class as the AR(1)
design-effect fix in the docs work and the autocorrelation-inflated false-positive
rate in `planning/simulation.py`. Default `block_length=None` estimates the
residual AR(1) coefficient ρ̂ and applies the standard rule
`b = clip(ceil(n^(1/3) · (2ρ̂/(1−ρ̂²))^(2/3)), 1, n//4)`, recorded in the result.

**Panels resample time blocks jointly across cells**, not independently per geo —
resampling geos independently destroys the contemporaneous cross-sectional
correlation that makes a panel more informative than one national series.

### Post-selection inference

If `(α̂, λ̂)` and the penalty are chosen once by search and every replicate
conditions on that choice, the intervals ignore selection uncertainty and are
again too narrow. Re-running the search inside each replicate is correct and
costs `n_boot × search`.

**Decision: `refit_search=False` by default, and the cheap path is labelled, not
silently shipped.** The label rides in three places so it cannot be lost:
`diagnostics["interval_semantics"]`, the `InferenceData` attrs, and every rendered
surface ([§8](#8-the-gating-checklist)). `refit_search=True` is the documented
requirement for any interval that will be published, and the coverage table in
[#186](https://github.com/redam94/mmm-framework/issues/186) must report **both**
so the size of the gap is visible rather than asserted.

### Ridge is biased, and no interval method fixes that

A percentile bootstrap interval around a shrunk estimator covers the
*estimator's* sampling distribution. Bias-correction (BC, via the fraction of
replicates below the point estimate) and acceleration (BCa) correct the bootstrap
distribution's **median-bias and skewness** — they do **not** remove shrinkage bias
relative to the true parameter. Coverage for the truth therefore falls below
nominal exactly when the penalty is doing real work.

The honest instrument for "how much work is it doing" is the **effective degrees
of freedom** `tr(X(XᵀX+λP)⁻¹Xᵀ)` reported by the ridge fit ([§6](#6-ridge)).
v1 ships BC intervals by default, BCa optional (affordable, since each replicate
is one linear solve), and states this limitation wherever an interval is rendered.

### Validate, do not assume

`diagnostics/coverage.py::run_recovery_coverage` exists to answer exactly "does my
90% interval cover 90% of the time?" on simulated truth. An empirical coverage
table at 50/80/90/95% — **including the iid-bootstrap comparison that shows the
under-coverage the block version fixes** — is the acceptance evidence for #186. A
path that under-covers ships with that stated, or does not ship.

Pointing it at this path is **not free**, and #186 must budget for three things
the issue did not anticipate:

* `run_recovery_coverage` is **hard-wired to PyMC** (`coverage.py:573-630`):
  `pm.do(model.model, θ)` → `sample_prior_predictive` → `pm.observe` →
  `_sample_swapped`, with **no refit-injection hook**. A refit callback has to be
  added before a frequentist estimator can be graded. The simulate-from-the-model
  half is reusable as-is; the refit half is not.
* it grades **central equal-tailed percentile** intervals, deliberately matching
  `compute_hdi_bounds` (`coverage.py:41`). A BC/BCa interval is therefore **not
  graded correctly** by default. v1 ships **percentile intervals as the graded
  default**, with BC/BCa optional and separately validated — which also keeps the
  headline coverage number comparable to the Bayesian path's.
* `coverage_from_draws` **silently drops** any simulation with fewer than 4 finite
  draws (`coverage.py:159,366`), shrinking the denominator without an error, and
  `build_recovery_result` only auto-attaches an uncertainty caveat for
  `advi`/`fullrank_advi` (`:405`). The shrinkage-bias and conditional-on-selection
  caveats must be passed **explicitly**.

### The container

`MMMResults` requires `trace: az.InferenceData`, `model: pm.Model` and
`panel: PanelDataset` (`model/results.py:24`), so the answer to "does the PyMC
graph still get built?" is **yes**. Building it is cheap, `predict` /
`sample_channel_contributions` / the estimand engine all read it, and serialization
already knows how to handle it. The frequentist path builds the graph for
structure and estimates *out* of it.

Replicates are packaged as `(chain=1, draw=n_boot)`, following the
`arviz_compat.point_to_idata` precedent that wraps a MAP point as
`(chain=1, draw=1)`. Two traps the shim documents and this path must respect:

* `posterior_from_dict` requires every value **already** shaped
  `(chain, draw, *shape)`, and the wrong convention **fails silently** — it wraps
  everything as one variable literally named `"posterior"`. The shim's own
  `expected.issubset(idata.posterior.data_vars)` guard is what catches it.
* names ending in `__` (PyTensor transformed duplicates) are dropped.

Every Deterministic downstream code reads must be synthesized per replicate —
`channel_contributions`, `media_total`, `controls_total`, `y_obs_scaled`,
`beta_<ch>`, the component Deterministics — which is done by evaluating the graph's
Deterministics at each replicate's parameter vector, so they cannot drift from the
Bayesian definitions.

`run_smc_fit` (`base.py:370`), not `run_approximate_fit`, is the structural
precedent: an estimator that is **not approximate** (`approximate` stays `False`)
but is also **not NUTS**, whose estimator-specific numbers ride in the returned
extra-diagnostics dict. A frequentist fit is the same shape — see [§8](#8-the-gating-checklist)
for why `approximate` is the wrong flag to reuse.

---

## 6. Ridge

`src/mmm_framework/frequentist/ridge.py`

```python
fit_ridge(design: DesignMatrix, y, *, penalty, penalize=None,
          nonneg=False) -> RidgeFit
```

Closed form `θ̂ = (XᵀX + λ_r P)⁻¹ Xᵀy`, solved via `scipy.linalg.lstsq` on the
**augmented system** `[X; √(λ_r P)] θ ≈ [y; 0]` rather than forming an explicit
inverse. **numpy/scipy only — `scikit-learn` is not needed and must not be added.**

* **The intercept is not penalized.** Nor are the trend and seasonality basis
  coefficients by default — they are structural, not effects to be shrunk. This is
  the `penalize` mask, and its default is derived from the column mapping.
* **Per-column penalties are already implied by the model.** `beta_controls` has
  role-dependent prior widths — confounders wide (weak penalty), precision
  controls narrow — so `P` is diagonal with `λ_j ∝ 1/σ_j²`, not a scalar. Shrinking
  a confounder re-opens the back-door, and the Bayesian path already knows that;
  the ridge path must inherit it rather than rediscover it.
* **Penalty selection** by the same out-of-sample criterion as the transforms
  (#184), never an in-sample rule. An explicit user value is honored.
* **Non-negativity** via `scipy.optimize.nnls` on the augmented system — the one
  constraint worth having without pulling in `cvxpy`.
* **Effective degrees of freedom** `tr(X(XᵀX+λP)⁻¹Xᵀ)` is reported. It is the
  honest "how much did the penalty shrink this" number and it is what makes the
  coverage caveat in [§5](#5-uncertainty) actionable rather than rhetorical.

### The ridge ≡ MAP test

#185 asks for an executable proof of the equivalence claim. Per [§1](#1-what-this-buys-over-fitmethodmap),
it holds **only for the Gaussian blocks**. The test therefore fits a model with
*explicitly configured* Normal coefficient priors (which `_explicit_prior` honors,
`base.py:131`) at fixed `(α, λ)`, and asserts `fit_ridge` with
`λ_j = σ̂²/τ_j²` reproduces `fit(method="map")` to optimizer tolerance.

A companion test asserts the **converse** on the default config: with the shipped
`Gamma` / `LogNormal` media prior, ridge and MAP **do not** agree — which is the
executable form of this spec's central claim and stops a future reader from
"simplifying" the frequentist path into an alias for `method="map"`.

---

## 7. Constrained estimation (CVXPY)

`src/mmm_framework/frequentist/constrained.py`

```python
fit_constrained(design, y, *, constraints, penalty, solver=None) -> ConstrainedFit
```

With transforms fixed the objective is a **convex QP** — minimize
`‖y − Xθ‖² + λ_r‖Pθ‖²` subject to linear constraints — so it solves to global
optimality with no local-minimum caveat. Four constraint families:

| family | example |
|---|---|
| non-negativity | `β_c ≥ 0` on media (also available without cvxpy via NNLS, §6) |
| linear equality / inequality on contributions | "these three channels sum to the booked number"; the natural home for reconciling an MMM to an experiment readout as a **hard** constraint rather than a soft calibration likelihood |
| monotonicity / ordering | `β_TV ≥ β_Display` |
| sum constraints | total media contribution ≤ a share of KPI |

**Dependency posture.** `cvxpy` goes in an optional `[frequentist]` extra, never
core. Imported lazily *inside* the function with an actionable `ImportError`
naming the extra. `tests/test_lean_imports.py` must stay green with the extra
absent, and ridge (§6) must keep working without it.

**Infeasibility raises**, naming which constraint failed — never a silent fall
back to the unconstrained solution.

**Boundary solutions are flagged.** A coefficient pinned at its constraint has no
meaningful two-sided interval, and the bootstrap will misreport it if treated as
interior: replicates pile up at the boundary and the percentile interval collapses
to a point. `ConstrainedFit` marks active constraints, and §8 renders them as
"at constraint" rather than as an estimate with a CI.

**A hard constraint is an assumption with no uncertainty.** Every interval from a
constrained fit conditions on the constraint being true. Documented, and stated in
the report banner.

---

## 8. The gating checklist

**This is the integration risk of the epic**, and the audit behind it was run
against the real code rather than reasoned about.

### The root cause, empirically verified

A `(chain=1, draw=B)` trace **passes every convergence gate as converged**:

```
compute_convergence  -> {'divergences': None, 'rhat_max': None, 'ess_bulk_min': 966.5}
convergence_flags    -> []
is_converged         -> True
convergence_warning_message -> None
```

`az.rhat` on one chain returns NaN → filtered to `None`; `convergence_flags` only
flags when a metric is **not** None (`convergence.py:88`), so a missing metric
reads as a pass; and `az.ess` returns ≈B because bootstrap replicates are iid.
`is_converged` returns `None` only when `diagnostics["approximate"]` is truthy or
when *all* metrics are None (`convergence.py:103`) — neither holds.

So a frequentist fit is **silently green everywhere the verdict is consumed**, and
the Augur client deck — whose only stop sign fires on `approximate` or
`is_converged is False` (`augur_sections.py:173`) — renders with **zero caveat**.

`approximate=True` is the only existing escape hatch and it is the **wrong flag**:
an approximate fit is a badly-estimated posterior; a ridge fit is not a posterior
at all, and its point estimate may be excellent. A distinct provenance field is
required.

### The provenance contract

Every frequentist fit stamps, in `results.diagnostics` and the `InferenceData`
attrs:

| key | value |
|---|---|
| `inference_family` | `"frequentist"` (Bayesian fits stamp `"bayesian"`; absence must be read as Bayesian for back-compat) |
| `estimator` | `"ridge"` / `"constrained"` |
| `interval_kind` | `"bootstrap_confidence"` |
| `interval_semantics` | `"conditional_on_selection"` or `"selection_resampled"` |
| `selection_criterion` | e.g. `"rolling_origin_mape"` |
| `block_length`, `n_boot`, `effective_dof`, `penalty` | the numbers behind the interval |
| `approximate` | **stays `False`** |
| `fit_method` | **`None`** — `FitMethod` has no frequentist member; the selector lives on `InferenceMethod`, and defaulting it is what makes the interactive report print "NUTS" |

`diagnostics/convergence.py::is_converged` gains a paradigm branch: `None` (not
assessable) whenever `inference_family == "frequentist"`, and
`_extract_diagnostics`'s NaN R-hat is nulled the same way.

### Ranked checklist

Severity is "what a reader would wrongly believe".

| # | surface | anchor | current behavior | required |
|---|---|---|---|---|
| 1 | `is_converged` / `convergence_flags` | `diagnostics/convergence.py:88,103` | returns **`True`** for a bootstrap trace | `None` on `inference_family == "frequentist"`; `None` metrics must not read as a pass |
| 2 | Augur client deck caveat | `augur_sections.py:173` | **no banner at all** | third banner: bootstrap CIs, conditional on selection, ridge is biased |
| 3 | `_merge_fit_provenance` | `extractors/base.py:209` | gated on `approximate` only; a frequentist fit passes through untouched | branch on `inference_family`; null R-hat/ESS/divergences; re-derive `converged` |
| 4 | `DiagnosticsSection` | `sections.py:1976` | extractor supplies **NaN** (not None) so the `is None` branch misses → renders literal `'nan'` and `'⚠️ nan'`, with ESS ≈B marked `'✅ Pass'` | gate the whole section off with an explanation |
| 5 | interactive Inference card | `interactive/generator.py:866` | defaults to **`'NUTS'` / "full MCMC posterior"** when `fit_method` is missing | read `inference_family`; never default to NUTS |
| 6 | `PosteriorPredictiveSection`, `AugurPPCSection`, interactive predictive-checks | `sections.py:2753`, `augur_sections.py:1195`, `generator.py:402` | compute and render **Bayesian p-values** | gate off — a posterior-predictive p-value is undefined without a posterior |
| 7 | prior→posterior, prior-predictive, evidence tiers, SBC, `diagnostics/learning.py` | `generator.py:717,788`, `script.py:383`, `interactive/script.py:1399` | render prior-vs-posterior densities and "prior-dominated" chips | gate off — a ridge fit has no prior |
| 8 | every "credible interval" string | `sections.py:93,391,552,1718,2065,2161,2296,2607`; `augur_sections.py:402,470,1138`; `interactive/{facts,insights,script,generator}.py` | asserts posterior probability semantics | family-aware wording; reuse the one already-neutral phrase, `"{ci}% range"` (`augur_sections.py:211`) |
| 9 | `EstimandsSection` CI header | `sections.py:2663` | derives the header from the wire field literally named **`hdi_prob`** | percentile intervals for frequentist fits; record which in `Realization` |
| 10 | `ExecutiveSummarySection` uncertainty callout | `sections.py:93` | "credible intervals reflecting genuine uncertainty" | family-aware |
| 11 | `ModelFitSection` band prose | `sections.py:391` | "credible interval, capturing both parameter uncertainty **and residual variance**" | doubly wrong for a sampling-distribution band |
| 12 | interactive facts contract | `interactive/facts.py:14,1461` | docstring pins "central credible intervals"; `meta` has **no slot** for an inference family | add the slot; the JS inherits the claim from here |
| 13 | serialization + `planning/history` run metrics | see §9 | records `fit_method`/`approximate` only | record the provenance contract above; a reloaded frequentist fit must not read as Bayesian |

**Acceptance test.** An end-to-end ridge fit → report render, asserting the output
contains no occurrence of "credible", "posterior mean", "R-hat" or "Bayes p", and
that the banner is present. A grep-based assertion is crude and exactly right
here: the failure mode is a string, so the test is a string test.

---

## 9. File plan

### New — `src/mmm_framework/frequentist/`

| module | public surface |
|---|---|
| `__init__.py` | re-exports; **no `cvxpy` import at module level** |
| `_saturation.py` | graph-faithful numpy saturation mirroring `_apply_saturation_pt` guard-for-guard (§4) |
| `design.py` | `build_design_matrix(panel, alpha, lam, *, model_config, trend_config) -> DesignMatrix` |
| `ridge.py` | `fit_ridge(design, y, *, penalty, penalize=None, nonneg=False) -> RidgeFit` |
| `search.py` | `search_transforms(panel, *, objective, budget, strategy="random", seed) -> SearchResult` |
| `bootstrap.py` | `bootstrap_fit(panel, *, n_boot, block_length, refit_search, seed) -> (az.InferenceData, dict)` |
| `constrained.py` | `fit_constrained(design, y, *, constraints, penalty, solver=None) -> ConstrainedFit` — lazy `cvxpy` |

**What #184 can actually reuse from `validation/backtest.py`.** `run_backtest` is
**not** pluggable: `_clone_for_prefix` (`:729`) hard-imports `BayesianMMM` and
calls `.fit(draws=, tune=, chains=)`. The reusable pieces are the pure ones —
`rolling_origins` (`:138`, int-in/list-out, no model dependency),
`_slice_panel_prefix` (`:696`, geo-correct: obs are period-major/cell-minor), and
`_point_metrics` (`:559`, returning `mape`/`smape`/`rmse`/`mae`/`bias`). #184
composes those into its own CV loop rather than driving `run_backtest`.

(Unrelated but worth fixing while in there: `backtest.py`'s module docstring
claims "national data only" and "trend NONE or LINEAR"; both are stale — the code
dispatches to `_forecast_geo` when `n_cells > 1`.)

### Modified

| file | change |
|---|---|
| `config/model.py:113` | drop `FREQUENTIST_*` from `_UNIMPLEMENTED_METHODS`; make `ridge_alpha` / `bootstrap_samples` / `optim_maxiter` live (they round-trip through `configs.json` for free) |
| `model/base.py:3087` | replace the `is_implemented` refusal with a dispatch on `model_config.is_bayesian`; add `run_frequentist_fit(model, ...)` alongside `run_approximate_fit` / `run_smc_fit`, returning `(idata, diagnostics)` |
| `diagnostics/convergence.py:88,103` | paradigm branch — `None` for `inference_family == "frequentist"`; a `None` metric must not read as a pass |
| `reporting/extractors/base.py:209` | extend `_merge_fit_provenance` to branch on `inference_family` as well as `approximate` |
| `reporting/{sections,augur_sections}.py`, `reporting/interactive/*` | §8 checklist rows 2, 4–12 |
| `serialization.py:649` | stamp the §8 provenance contract. `_FORMAT_VERSION` stays `"1.1"` — `load()` gates on **major** only, so added keys need no bump. Note the existing stamp derives `approximate` **solely** from `model_config.fit_method`, so a `None` fit_method stamps nothing and a reloaded frequentist fit would read as Bayesian by omission |
| `agents/fitting.py` | `_INFERENCE_METHODS` registry so `update_model_setting` validates the value; thread the spec key |
| `validation/backtest.py` | refactor `PosteriorForecaster._saturate` / the adstock windowing to import `frequentist/_transforms.py`, so there is one numpy mirror rather than two |
| `diagnostics/coverage.py:573` | add a refit-injection hook so `run_recovery_coverage` can grade a non-PyMC estimator; pass the frequentist caveats explicitly (`:405` only auto-caveats ADVI) |
| `pyproject.toml` | `[project.optional-dependencies] frequentist = ["cvxpy>=1.5"]` |
| `CLAUDE.md` | troubleshooting row: "which estimator do I want" |

### New tests

| module | pins |
|---|---|
| `tests/frequentist/test_design_equivalence.py` | **the load-bearing test** — `max|mu_graph − X@θ| < 1e-12` across {adstock} × {saturation} × {trend} × {national, geo} |
| `tests/frequentist/test_saturation_fidelity.py` | the numpy saturation matches `_apply_saturation_pt` at `λ·x > 20` and at zero-spend rows (the two measured drifts, §4) |
| `tests/frequentist/test_ridge.py` | OLS limit at `λ_r→0`; scale invariance of fitted contributions; effective dof; **ridge ≡ MAP under explicit Normal priors**, and **ridge ≠ MAP under the shipped Gamma/LogNormal default** (§6) |
| `tests/frequentist/test_search.py` | recovers planted `(α, λ)` from `synth.dgp` |
| `tests/frequentist/test_bootstrap_coverage.py` | empirical coverage at 50/80/90/95 via `run_recovery_coverage`, **block vs iid** |
| `tests/frequentist/test_frequentist_gating.py` | `converged is None`; rendered report contains no "credible"/"Bayes p"/"R-hat"; banner present |
| `tests/test_lean_imports.py` | unchanged and still green with `cvxpy` absent |

---

## 10. How the numbers here were measured

Every empirical claim above came from a probe run against this branch, not from
reasoning:

* **`mu == X@θ`** (§2) — build the graph, substitute a fixed constant for every
  free RV via `pytensor.graph.replace.graph_replace(..., strict=False)`, evaluate
  the component Deterministics, compare to a hand-built numpy design matrix.
  Parameterized over the family matrix in §2.
* **saturation drift** (§4) — evaluate `_apply_saturation_pt` on `pt.constant`
  input against the `transforms/saturation.py` twin across λ ∈ {2, 10, 25, 60}
  and at `x = 0`.
* **adstock fidelity** (§4) — `parametric_adstock` vs `parametric_adstock_pt` over
  3 kinds × `normalize ∈ {True, False}`; panel path against an explicit per-cell
  numpy loop, plus the flat-stacking control that shows the 0.256 bleed.
* **prior families** (§1) — enumerate `model.free_RVs` and read
  `rv.owner.op.__class__.__name__` under both `media_prior_mode` values.
* **the convergence gate** (§8) — run `compute_convergence` / `convergence_flags`
  / `is_converged` on a synthetic `(chain=1, draw=1000)` trace built with
  `arviz_compat.posterior_from_dict`.

These probes become the tests in §9; the spec's claims and the test suite are the
same assertions.

---

## Deferred

Not impossibilities — the natural v2, in rough value order:

* **Time-varying coefficients** (#137) under a first-difference penalty.
* **V3 per-geo betas** under a group penalty (the frequentist analogue of partial
  pooling).
* **Faithful non-L2 penalties** where the Bayesian path uses them: L1 for
  piecewise changepoints, a P-spline difference penalty for spline trends.
* **Multiplicative specification** — linear on standardized `log(y)`, so it is
  mostly a `_prepare_data` branch.
* **Experiment calibration** as hard constraints rather than likelihood terms —
  the most interesting item here, and the one CVXPY makes natural.
* **Extension models** — out of scope structurally, not just by effort.

## Key files

| concern | file |
|---|---|
| the graph's saturation (source of truth) | `model/base.py:194` (`_apply_saturation_pt`) |
| the graph's `mu` assembly | `model/base.py:2588` |
| standardization | `model/base.py:656` (`_prepare_data`) |
| numpy adstock (safe to reuse) | `transforms/adstock.py:133,219,252` |
| the refusal to replace | `model/base.py:3087`, `config/model.py:113` |
| non-NUTS packaging precedent | `model/base.py:281` (`run_approximate_fit`), `:370` (`run_smc_fit`) |
| trace assembly shims | `utils/arviz_compat.py:127` (`posterior_from_dict`), `:238` (`point_to_idata`) |
| the convergence gate to branch | `diagnostics/convergence.py:88,103` |
| the provenance hook | `reporting/extractors/base.py:209` (`_merge_fit_provenance`) |
| coverage validation | `diagnostics/coverage.py::run_recovery_coverage` |
| rolling-origin objective | `validation/backtest.py` |

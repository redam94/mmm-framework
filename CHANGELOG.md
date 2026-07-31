# Changelog

All notable changes to `mmm-framework` are documented here. The project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html): the major version changes when a
frozen public contract breaks, and the contract itself is pinned by
`tests/test_api_contracts.py` and `tests/test_lean_imports.py`.

> **Maintainers:** every entry added here must also be mirrored into
> [`docs/changelog.html`](docs/changelog.html) — the public documentation site — in the same PR
> as the version bump, including moving the `Current` chip onto the new release. That page is
> hand-authored HTML; nothing generates it from this file and no test gates it, so it rots
> silently (1.1.0 and 1.2.0 both shipped while the site still announced 1.0.0). The full
> checklist is in `CLAUDE.md` § "Release checklist".

## [Unreleased]

### Fixed

- **Rolling-window cross-validation on a geo panel forecast out of phase with its own training
  window** ([#273]). `validator.py` passed `min(train_indices)` — an *observation* index — into
  `PosteriorForecaster.forecast`'s `train_offset`, which is a *period* offset. The two axes
  coincide on a national panel and differ by `n_cells` on a geo/product one, so a 4-cell window
  starting at observation 80 (period 20) shifted the Fourier basis by 80 periods and drove the
  linear trend far negative. Measured on a 4-cell panel: the forecast mean was **175.2 against an
  actual of 232.3** (25% low), a 139.6-unit error against a 35.3-unit noise tolerance. A
  mis-phased seasonality moves the forecast *level* while leaving the interval width untouched, so
  the miss reads as an effectiveness change rather than as an index.

  `forecast()` now takes `train_positions` — observation positions, the same units as `positions`
  — and derives the period offset itself; `train_offset` remains, documented as a period offset.
  A window that does not start on a period boundary is refused by name rather than rounded.

  Three structural fixes ship with it.

  **CV splits are now generated on the period axis** and expanded to observations, so every
  window covers whole periods. `CrossValidationConfig`'s `min_train_size`, `test_size` and `gap`
  therefore count **periods** — unchanged on a national panel, where `n_cells == 1`. Because the
  shipped default of 52 is a full year of weekly history, a one-year *geo* panel can no longer
  spare it, so that case now raises a message naming both numbers instead of returning zero folds
  and surfacing as a generic warning with no CV section and no stated reason.

  **`_slice_panel_data` refuses a slice that does not cover whole periods** instead of rebuilding
  coordinates from whatever index values survive: 101 observations of a 4-cell panel previously
  produced a clone claiming `n_periods=26` — 104 observations' worth — after which every
  downstream reshape read the wrong cell. The check is exact rather than a divisibility test,
  because `[0,1,2,3,5,6,7,8]` is eight observations starting on a period boundary and still
  straddles three periods.

  **The causal-refutation `data_subset` test samples whole periods on a geo panel.** It drew a
  random subset of raw observations, so the refit ran against a fabricated period axis and the
  "effects should be stable" verdict was partly measuring that. The national path — including its
  RNG draw — is byte-identical.

- **The HTML report and the slide deck gave opposite recommendations for the same channel**
  ([#221]). `deck/engine.py` computed a margin-adjusted break-even (`1/margin`) while the Augur HTML
  took `channel_rows`' default of `1.0`. At a 40% gross margin a channel with ROI 1.8 was tiered
  **Scale** in the report and **Reduce** in the deck, with no cross-reference. Both now resolve one
  break-even through `reporting.helpers.measurement.resolve_break_even`.

  The convention is **move the reference, not the number**: a margin-scaled figure would be a profit
  number wearing a revenue label. `MetricMeta` gains `basis` (`"revenue"` by default, so every
  existing number is byte-identical), `value_per_kpi` and `value_source`, so a metric's definition
  travels with it. Any artifact judging channels on a profit basis now carries a banner naming the
  margin, its source, and the constant-margin assumption — triggered by the data, not a config flag.
  A margin passed as a percentage (`40` rather than `0.40`) is refused rather than silently
  producing a 0.025 break-even that tiers everything Scale.

- **`_masked_sum` silently returned an unwindowed divisor** ([#221]). On a dtype or length mismatch
  it fell back to the FULL-series sum, so a windowed ROI divided by every period's spend instead of
  the window's — understating the metric with no error, across windowed marginal ROAS, the
  interactive per-period divisor and `analysis.py`. It now raises, naming the channel and both
  shapes. The geo-panel case its comment cited as justification does not reach it: measured on a geo
  panel through `build_and_fit`, it ran 12 times and took the masked branch every time.

- **The CFO baseline is the model's fitted non-marketing outcome, with the residual named**
  ([#220]). It was `observed_total − modelled_media`, so model error landed inside a number read as
  base demand. `cfo_facts` now reports `fitted_total`, `unexplained` and `baseline_basis`, the
  rollup reconciles as `base + marketing + unexplained == observed`, and both surfaces state which
  basis they are on rather than leaving it to be assumed.

- **Models that consume a control column could be saved but never reloaded** ([#237], [#222]).
  A reach/frequency `frequency_column`, and price/promo levers, are stripped out of
  `model.control_names` because they are consumed as model terms rather than linear controls. The
  serializer recorded that **post-strip** list and compared it against the panel's **pre-strip**
  columns, so `MMMSerializer.load` raised
  `ValueError: Panel controls ['Price'] don't match saved model controls []` every time. Save
  succeeded; load always failed. Metadata now carries `panel_controls` — mirroring the
  `panel_channels` field that already existed for the same reason on the media axis — and the gate
  compares against that, falling back to `control_names` for saves written before it existed. Scoped
  by the strip mechanism rather than by any one feature, so it covers both.

- **`predict(X_controls=...)` silently returned the baseline** when levers or reach/frequency had
  consumed every control ([#222]). The guard was `if X_controls is not None and self.n_controls > 0`,
  so the argument was dropped without an error. A narrower case was also wrong: with one surviving
  control, a mismatched-width array was applied to whichever control happened to survive. Both now
  raise, naming the consumed columns and the accepted order. The legitimate partial swap — passing
  the surviving controls — still works and is not blanket-refused.

- **`sample_channel_contributions()` now refuses a multiplicative specification** instead of
  returning log-scale numbers as though they were original-scale contributions ([#220]). Its
  docstring rests on "the model is additive in channels" and it ends in `contrib * y_std`, both of
  which hold only on the log scale for a multiplicative fit. It was wrong *quietly*: measured on a
  MAP fit, the CFO one-pager reported marketing at **0.005%** of KPI where the additive equivalent
  of the same world reported **10.7%**. Its sibling `compute_marginal_contributions()` has refused
  this since it shipped; this closes the asymmetry.

  **Behaviour change.** Roughly twenty reporting and planning call sites reach this method, so a
  multiplicative model now loses those surfaces rather than showing wrong numbers — reports still
  render, with the affected sections absent. `garden/compat.py::_ops_smoke_tier` executes real ops,
  so a multiplicative garden model's ops-smoke tier turns **red**; red-because-refused is the
  intended outcome. Use `compute_component_decomposition()` (exact LMDI, original scale) or
  `compute_counterfactual_contributions()` / `compute_channel_roi()`.

- **Component shares are computed against the signed total, not a sum of magnitudes** ([#220]).
  Components sum to the fitted outcome, so a magnitude denominator made "% of total" wrong whenever
  any component was negative — a declining trend or a negative control was enough, and shares did
  not add to 1. A stability rule falls back to the magnitude denominator, flagged, when the signed
  total is near zero (a naive signed share there renders Baseline −1105.8% and Trend +1613.0%).

- **The fallback decomposition's Baseline was short by `y_mean × n_obs`** ([#220]).
  `_compute_decomposition_from_trace` computed `intercept * y_std` with no `+ y_mean`, and that
  path is taken exactly when `compute_component_decomposition()` raised — when a trustworthy number
  matters most.

[#220]: https://github.com/redam94/mmm-framework/issues/220
[#221]: https://github.com/redam94/mmm-framework/issues/221
[#222]: https://github.com/redam94/mmm-framework/issues/222
[#237]: https://github.com/redam94/mmm-framework/issues/237
[#273]: https://github.com/redam94/mmm-framework/issues/273

## [1.3.3] — 2026-07-27

A security release. Cut from the `v1.3.2` tag with only the fix below, so 1.3.x users can take it
without adopting in-progress v1.4 work.

### Fixed

- **Generated HTML reports did not escape `</` in embedded chart JSON**
  ([GHSA-7q6v-xpwm-4937]). `reporting/charts/base.py::_to_json` serialised Plotly payloads with a
  bare `json.dumps`, which escapes neither `<` nor `/`. Every caller interpolates the result
  straight into an inline `<script>` block, so a payload string containing `</script>` closed the
  block early and the browser parsed the remainder as **HTML**.

  Affects the **classic** (`ReportConfig.full()` / `.minimal()` / `.presentation()`) and **augur
  readout** shells. `InteractiveReportGenerator` was never affected — it already applied the same
  guard.

  Channel names could not carry the payload, because PyMC rejects `/` in random-variable names and
  a hostile channel name therefore fails at fit time. **Control-variable and geography names are
  xarray coords / DataFrame columns, not RV names**, and they reach chart traces, hovertemplates
  and axis titles — so the vector is reachable from any untrusted modeled dataset (a client CSV, an
  upload, a Data Studio import). HTML-body interpolation of the same names was already correctly
  `html.escape`d; only the chart-JSON path was affected.

  `_to_json` now appends `.replace("</", "<\/")`. `\/` is a legal JSON string escape and legal
  JavaScript, so Plotly decodes a byte-identical string while the literal `</script>` never
  appears. Because `_to_json` is the single choke point for those shells, one change covers every
  chart in them.

  The premature tag also corrupted the surrounding chart payloads, so this is a rendering fix as
  well: in the reproduction, charts rendered went from 26/28 to 28/28 and JS errors from 2 to 0.

  Found by pressure-testing the report generators against a nine-model matrix that included
  deliberately hostile channel, control and geography names.

[GHSA-7q6v-xpwm-4937]: https://github.com/redam94/mmm-framework/security/advisories/GHSA-7q6v-xpwm-4937

## [1.3.2] — 2026-07-27

### Fixed

- **Three names v1.3.1 documented as public were not actually exported.** Its changelog stated
  that `rebuild_like()`, `audit_forward_pass()` and `audit_refit()` were "exported from
  `mmm_framework.validation`". They were reachable only from
  `mmm_framework.validation.backtest`. The claim is now true.

  Found by importing the published wheel and checking it against its own release notes rather
  than against the working tree. Nothing pinned the export set — `PUBLIC_SURFACE` in
  `tests/test_api_contracts.py` did not cover `mmm_framework.validation`, and the docs-snippet
  gate reads code fences, so a *prose* claim about the API surface had nothing checking it.
  `mmm_framework.validation` is now in `PUBLIC_SURFACE`, which makes removing or renaming any of
  these a declared breaking change.

## [1.3.1] — 2026-07-26

A correctness release for the out-of-time forecast path. One bug, of the class this project
treats as most serious: **a code path that reads as complete and silently does nothing for one
configuration** — the same shape as the v1.2.0 sampler fixes (#169/#171) and the root-saturation
fix in 1.3.0 (#202).

### Fixed

- **`PosteriorForecaster` summed five of the fitted mean's ten terms** ([#219]). The model's
  `mu` is `intercept + trend + seasonality + geo + product + media + controls`, plus conditional
  `event` (#143), cross-channel `interaction` (#142) and price/promo `lever` (#138) blocks. The
  forward pass replayed only intercept, trend, seasonality, media and controls — and raised
  nothing. It also applied a **time-averaged** coefficient to a time-varying channel
  (`beta_<ch>` is `pt.mean(beta_t)`), and convolved **raw reach** for a reach/frequency channel,
  omitting the frequency gain.

  Every metric computed from that forward pass was affected on such a model — MAPE, sMAPE,
  RMSE, MAE, bias, MASE, the naive-baseline comparison and 50/80/95% interval coverage — and the
  failure was silent: the backtest completed and reported plausible accuracy. On a
  geo × product panel the dropped product offset alone moved the forecast by up to **113 KPI
  units**, on every observation.

  These numbers ship through `run_backtest`, the `cross_validation` agent tool, the
  Validation-tab REST job and a client artifact.

  The fix follows the frequentist path's refusal convention (#183): the **product level offset is
  now replayed**, and every term the forward pass genuinely cannot reproduce raises the new
  `ForecastUnsupportedError` — naming the feature, listing every blocker at once, and doing so at
  **construction time**, so a caller cannot hold a forecaster whose output it is not allowed to
  trust. Refused: price and promotion levers, events, cross-channel interactions,
  reach/frequency, time-varying coefficients, and the multiplicative specification.

  **This turns some previously "working" backtests into hard refusals.** That is the point: the
  numbers they produced were wrong. The `cross_validation` agent tool reports it as a stated
  "not assessable for this model" reason rather than a failure.

- **`_clone_for_prefix` downcast custom model classes** ([#219]). It hard-constructed
  `BayesianMMM(...)`, so backtesting any garden or custom model — a `CustomMMM` subclass such as
  `LatentFactorMMM` or the awareness model — silently fit and graded a **plain additive MMM** and
  reported its accuracy under the custom model's name. It now reconstructs `type(model)` and
  forwards `model_params`, refusing when the class cannot be rebuilt rather than falling back.

- **`run_backtest` now refuses an experiment-calibrated model.** The refit drops calibration
  likelihoods (their estimands reference full-period spend), so the reported accuracy was an
  *uncalibrated* model's, carrying the calibrated model's name.

- **A geo/product panel with a spline, GP or piecewise trend replayed the trend at the wrong
  time index** ([#219]). `trend_component` is registered per-*observation*, but the forward pass
  indexed it with *period* positions — so on an `n_cells`-wide panel, period `p` read
  observation `p`, which belongs to period `p // n_cells`. The trend was stretched by a factor of
  `n_cells`, and the documented hold-last-flat clamp never fired because positions never reached
  the end of the obs axis. Measured at **up to 26% of KPI level** on a 6-cell spline panel. A
  national panel was unaffected (`n_obs == n_periods` makes the two indexings identical), which
  is why it went unseen.

- **Student-t fits drew Gaussian observation noise.** `LikelihoodFamily.STUDENT_T` is a supported
  additive family (default `nu = 4.0`, genuinely heavy-tailed), but `include_noise=True` always
  drew `Normal(0, 1) * sigma`. Since the whole purpose of that flag is the predictive
  distribution that `BacktestResult.coverage()` grades, the reported interval coverage was not
  the model's. The noise draw now dispatches on the fitted family.

- **The rolling-window cross-validation forecast seasonality out of phase.** `_trend_at` honoured
  `train_offset` but `_seasonality_at` did not, so a clone whose window starts at absolute period
  `s` evaluated the Fourier basis `s` periods out of phase. `run_backtest` always passes
  `train_offset=0` and was unaffected; `ModelValidator.cross_validate(strategy="rolling")` was not.

- **`ModelValidator.cross_validate` swallowed the new refusal into a per-fold warning.** Its broad
  `except Exception` downgraded a configuration-level refusal to a log line — and not uniformly,
  since event regressors are rebuilt per fold, so a fold containing no configured holiday passed
  while one containing a holiday was silently dropped. It now propagates, and the audit is hoisted
  above the fold loop so the refusal costs zero refits instead of one per fold. Cross-validation
  also now applies `audit_refit`: it drops `experiments` exactly as the backtest refit does, so it
  was silently grading an uncalibrated model while the backtest harness refused to.

- **Four more clone sites in `validation/validator.py` downcast custom model classes** — the
  cross-validation, prior-sensitivity and stability clones, and `_fit_clone`, which powers the
  causal refutation suite behind the **Model Defense** document. A garden model's defense report
  was showing a plain additive MMM's refutation results under the garden model's name. All five
  clone sites now share one `rebuild_like()` helper.

### Added

- `ForecastUnsupportedError` (with `.feature`, `.reason`, `.all_unsupported`),
  `TrendExtrapolation`, `rebuild_like()`, and the public `audit_forward_pass()` /
  `audit_refit()` helpers, all exported from `mmm_framework.validation`.
- **A subclass that overrides `_build_model` is now refused**, because it defines its own mean
  and this forward pass reproduces only the base additive one. A class whose mean *is* the base
  mean opts in with `__forecast_forward_pass__ = "base"`. This is the necessary counterpart to
  class-preserving cloning: fitting the right class while replaying the base forward pass would
  have traded one silent-drop path for another.
- **`PosteriorForecaster.trend_extrapolation`** states how the trend is continued past the
  training window instead of leaving it implicit: `linear` extrapolates in closed form, while
  spline/GP/piecewise **hold the last fitted level flat** — so those forecast intervals do not
  widen with horizon. `is_model_defined` distinguishes the model's own continuation from the
  heuristic. It also records the training length, because a linear trend's slope *per period* is
  a function of it.
- `PosteriorForecaster(..., strict=False)` downgrades a refusal to a warning for diagnostic use.
  It omits a term the model estimated and must not be used for planning.

### Notes

The regression test (`tests/test_backtest_completeness.py::TestComponentSumIdentity`) asserts
that `forecast(include_noise=False)` over training positions equals the sum of the model's
registered component Deterministics to `atol=1e-9`. **That test is the audit**: any term the
forward pass fails to sum appears as a residual rather than as a plausible number. It
deliberately does not compare against `predict()`, which draws from the observation likelihood
and carries MC noise no seed removes.

The module and `validator.py` docstrings claimed "national data only; trend NONE or LINEAR"
while geo panels and every trend family in fact ran. They now state the actual scope.

[#219]: https://github.com/redam94/mmm-framework/issues/219

## [1.3.0] — 2026-07-26

### Changed

- **The default logistic saturation prior is now stated in units of observed spend.**
  Media reaches saturation normalized by the channel's training maximum, so `sat_lam` *is*
  the curve's elbow — half-saturation sits at `ln(2)/lam` in units of that maximum. The
  previous default, `Exponential(lam=0.5)`, was never reparameterized after that
  normalization: its **mode was `lam = 0`** (a channel that never saturates) and **29.3% of
  its mass placed the elbow beyond maximum observed spend**, where no observational data can
  move it.

  That matters more than an ordinary prior choice because saturation is **not identified from
  the sales likelihood** — Jin et al. (Google, 2017) call the Hill parameters "essentially
  unidentifiable"; Dew et al. (2024) show predictive fit cannot arbitrate between
  observationally equivalent response curves. When a parameter is unidentified the prior is
  largely the answer, so it has to be defensible.

  The new default places the median elbow at half of maximum observed spend, with a 90%
  interval of roughly 26–97% of maximum and ~4% of mass beyond it — the same move Robyn
  (`inflexion = max(x)·γ`) and Meridian (`ec` scaled to median spend) make.

  **This changes fitted results for any model that did not set an explicit `lam_prior`.**
  To restore the old prior, pass
  `SaturationConfig.logistic(lam_prior=PriorConfig(distribution="Gamma", params={"alpha": 1.0, "beta": 0.5}))`
  — `Gamma(1, rate)` *is* the Exponential, and `PriorType` has no Exponential member.
  ([#207](https://github.com/redam94/mmm-framework/issues/207))

### Added

- **`SaturationConfig.anchor_kappa_to_data`** — confines the Hill half-saturation prior to the
  channel's own observed spend percentiles rather than the whole normalized `[0, 1]` range,
  via the existing `compute_kappa_bounds_from_data` (previously reachable only from the
  extension priors). Opt-in, because it changes fitted results; an explicit `kappa_prior`
  still wins. ([#207](https://github.com/redam94/mmm-framework/issues/207))

- **`mmm_framework.diagnostics.saturation`** — the pair of checks for a parameter whose value
  is mostly its prior. `saturation_prior_report` asks, *before* fitting, where the prior puts
  the curve's elbow relative to spend you have actually observed, and grades it
  `anchored` / `diffuse` / `unanchored`; `warn_if_saturation_prior_is_unanchored` warns on the
  last of those and is silent otherwise; `saturation_learning` reports post-fit
  prior→posterior contraction for the saturation block, where a `"prior-dominated"` verdict
  means the fitted curve is the prior you chose. Neither makes the parameter identifiable —
  only dose spread does that. ([#207](https://github.com/redam94/mmm-framework/issues/207))

- **`mmm_framework.frequentist`** — the estimation layer for epic
  [#180](https://github.com/redam94/mmm-framework/issues/180). With per-channel adstock and
  saturation held fixed the core model is linear in its remaining parameters, so its mean is
  exactly `X @ theta`. `build_design_matrix` produces that design in pure NumPy and an
  equivalence test pins it against the PyTensor graph to 1e-12 across every supported
  transform, trend and panel shape; `fit_ridge` solves it in closed form (NumPy/SciPy only,
  diagonal penalty, effective degrees of freedom, optional non-negativity); and
  `search_transforms` chooses the transforms by rolling-origin out-of-sample error.

  Note what the search does **not** do. Graded against the planted truth in
  `synth.dgp.make_clean`, carryover is recovered better than chance (mean absolute error
  ~0.17 on α against ~0.26 for an uninformed draw) but **saturation is not identified** —
  among candidates scoring within 10% of the best, one channel's `sat_lam` ranges over
  ~0.16 to 7.8 while the planted value is 1.6. At a large budget the winner *beats the true
  parameters* out of sample while sitting further from them. Read `SearchResult.spread()`,
  not the winner, and render `SearchResult.caveat` wherever its ROI is shown.

  No estimator is wired to `fit()` yet — `frequentist_ridge` and `frequentist_cvxpy` still
  refuse. Design spec: `technical-docs/frequentist-estimation.md`.

- **`mmm_framework.frequentist.bootstrap_fit`** — moving-block residual bootstrap turning the
  ridge point estimate into `(chain=1, draw=n_boot)` `InferenceData`, so `predict`, the
  estimand engine and reporting all work unchanged. Every deterministic in the container is
  evaluated out of the model's own PyTensor graph at each replicate's parameter vector, so a
  bootstrap `channel_contributions` cannot drift from the Bayesian definition of one.
  ([#186](https://github.com/redam94/mmm-framework/issues/186))

  Three deliberate choices, each addressing a way a plausible bootstrap ships intervals that
  do not cover:

  - **Blocks, with a data-driven length.** MMM residuals are serially correlated, and an iid
    residual bootstrap treats each week as exchangeable. Measured over 60 simulations × 300
    replicates of the `make_clean` world with AR(1) errors at ρ = 0.6, per-channel
    contribution coverage of 90% intervals is **79.6%** for the iid bootstrap and **90.4%**
    for the block version (median block length 7). At ρ = 0 the estimated block length
    collapses to 1 and the two agree (92.9% vs 93.3%), so nothing pays a width penalty for a
    dependence that is not there. Panel cells resample the *same* period sequence, because
    resampling geographies independently destroys the contemporaneous correlation that makes
    a panel more informative than one national series.
  - **The cheap interval is labelled, not silently shipped.** The transforms and penalty are
    chosen once by search, and every replicate conditioning on that choice omits selection
    uncertainty — which matters more than usual here, because saturation is not identified by
    the criterion at all. `refit_search=True` re-runs the search inside each replicate and is
    correct; the default is `False` and stamps
    `diagnostics["interval_semantics"] = "conditional_on_selection"` plus a plain-language
    caveat, on the reasoning that an unaffordable honest default gets switched off and takes
    its own label with it.
  - **The bias is stated, not papered over.** Percentile intervals around a shrunk estimator
    cover the estimator's sampling distribution, not the true parameter, and no interval
    method fixes that. `bc_interval` / `bca_interval` correct the bootstrap distribution's
    median-bias and skewness for a named scalar; the caveat carries the fit's effective
    degrees of freedom, which is the honest measure of how much work the penalty is doing.

  Every fit stamps `inference_family="frequentist"`, `interval_kind="bootstrap_percentile"`
  and `approximate=False` — `approximate` is the wrong flag to reuse, since an approximate fit
  is a badly-estimated posterior and a ridge fit is not a posterior at all. Nothing renders
  these yet; that is [#188](https://github.com/redam94/mmm-framework/issues/188).

- **`fit()` dispatches the frequentist path, and every surface tells the truth about it.**
  `InferenceMethod.FREQUENTIST_RIDGE` / `FREQUENTIST_CVXPY` now select a real estimator instead
  of raising. `ModelConfigBuilder().frequentist_ridge()` / `.frequentist_cvxpy()` are live, as is
  the generic `.with_inference_method(...)`; `ridge_alpha`, `bootstrap_samples` and
  `optim_maxiter` stop being inert and become the penalty fallback, replicate count and search
  budget. The agent spec accepts `inference.method = "frequentist_ridge"`.
  ([#188](https://github.com/redam94/mmm-framework/issues/188))

  **`inference_method` selects the paradigm; `fit_method` selects among the Bayesian
  estimators.** A frequentist fit leaves `fit_method` as `None` — `FitMethod` has no frequentist
  member, and a stray `"nuts"` there is precisely what made downstream surfaces announce a full
  MCMC posterior for a ridge fit.

  The integration half is the larger one, because every report section, estimand label and
  diagnostic in this codebase was written assuming a posterior:

  - **`MMMResults.converged` is `None`** for a frequentist fit, and the sampler metrics are
    nulled rather than computed. This is not defensive: a `(chain=1, draw=B)` bootstrap trace
    passed *every* convergence gate as `True` — `az.rhat` on one chain is NaN and a `None` metric
    raises no flag, while `az.ess` returns ≈`B` because bootstrap replicates are iid. The
    estimator was silently green everywhere the verdict is consumed.
  - **Interval wording is family-aware.** A bootstrap percentile interval is a **confidence**
    interval; "there is a 90% probability the ROI is in this range" is true of a credible
    interval and false of this one. `mmm_framework.diagnostics.provenance` is the single source
    of that vocabulary, and the classic report, the Augur client deck and the interactive
    report's JavaScript all read it instead of hard-coding "credible".
  - **Posterior-only views are gated off with a stated reason**, not blanked: the convergence
    table, posterior-predictive checks and Bayesian p-values, prior-predictive and
    prior-vs-posterior contraction. A missing table reads as an oversight; "not applicable
    because there is no chain" reads as a property of the method.
  - **A banner names the estimator, the selection criterion and the interval semantics** in all
    three shells — including the Augur client deck, which previously rendered a frequentist fit
    with *zero* caveat, since its only stop sign fires on `approximate` (False here) or
    `is_converged is False` (`None` here).
  - **`inference_family` is a distinct provenance field**, carried into report bundles, the
    serializer's `metadata.json` and `planning/history` run metrics. `approximate` is the wrong
    flag to reuse: an approximate fit is a badly-estimated posterior, a ridge fit is not a
    posterior at all and its point estimate may be excellent. Absence of the field reads as
    Bayesian, so every fit produced before this release is unaffected.

  Selecting `frequentist_cvxpy` with no explicit `constraints=` applies non-negative media — the
  one restriction every MMM wants and the one a prior can only express softly. It needs the
  optional extra (`pip install 'mmm-framework[frequentist]'`).

  The Excel template's `Inference Method` cell accepts both values too — it had its own hard
  refusal, added while they were unimplemented, which would have outlived the thing it gated.

- **The frequentist path, graded — and the verdict written down.** Epic
  [#180](https://github.com/redam94/mmm-framework/issues/180) opened by challenging itself
  ("ridge is MAP with Gaussian priors, so this risks shipping a synonym").
  [#189](https://github.com/redam94/mmm-framework/issues/189) answers it with numbers on five
  synthetic worlds with answer keys, and records the answer where it is unflattering.

  Mean |relative error| on per-channel contribution (NUTS at 4 × 1000; full table and signed
  bias in `tests/frequentist/test_recovery_comparison.py`):

  | world | ridge | map | nuts |
  |---|---|---|---|
  | `clean` (control) | **0.052** | 0.073 | 0.076 |
  | `realistic` | 0.317 | 0.216 | **0.182** |
  | `unobserved_confounding` | 0.670 | **0.297** | 0.302 |
  | `adstock_misspec` | 0.959 | 0.955 | **0.809** |
  | `saturation_misspec` | 0.688 | 0.399 | **0.389** |

  Runtime: ridge 1.7–2.9 s, MAP 2.6–5.6 s, NUTS 8.9–19.6 s.

  **Ridge is not a synonym for MAP** — the two differ by roughly 2× on three of five worlds
  and in opposite directions on `adstock_misspec`. **But it is not more accurate.** It wins
  only on the positive control, and under unobserved confounding it over-credits media by
  **+41.6%** against MAP's +5.9% — because the shipped `Gamma` / `LogNormal` media priors
  shrink media effects in a way that happens to counteract back-door bias, and a penalty
  selected by out-of-sample error carries no such opinion.

  **So the honest recommendation is: use the Bayesian path.** Reach for the frequentist one
  when you need a **hard constraint** a prior cannot express (β ≥ 0, an ordering, a
  contribution total that must match a booked number — the epic's strongest justification),
  a **fast second opinion** for triangulation, or iteration speed while a specification is
  still moving. Do not reach for it to publish intervals, to encode knowledge, or on data you
  suspect of confounding.

  Ships with `nbs/demos/frequentist_vs_bayesian.ipynb` — the **paradigm** axis, sibling to
  `approximate_posteriors.ipynb` (method) and `nuts_backends.ipynb` (backend) — which measures
  all of the above at bake time rather than quoting it.

- **`run_recovery_coverage(refit=...)`** — estimator injection for
  `diagnostics/coverage.py`, which was hard-wired to `pm.observe` + NUTS. Supplying a callable
  grades a non-PyMC estimator against the same θ*, the same simulated datasets and the same
  central equal-tailed intervals as the Bayesian path, which is what made the #186 coverage
  table possible. `extra_caveats=` was added alongside it: the function auto-attaches an
  uncertainty caveat for ADVI only, so ridge shrinkage bias and conditional-on-selection
  intervals have to be passed explicitly or they go unstated.

### Fixed

- **`fit(method="map")` no longer fails with a bare `ZeroDivisionError`.** `find_MAP`
  optimizes in the unconstrained space, so a `[0, 1]`-bounded parameter — every `Beta`-prior
  parameter in the model: `adstock_alpha`, `sat_half`, `sat_exponent` — reaches the optimizer
  as a logit, and float64 `sigmoid` returns *exactly* `1.0` past about 37. `1 - alpha` then
  becomes exactly zero and the `Beta` log-density's gradient divides by zero inside the
  compiled graph, surfacing as a bare `ZeroDivisionError` naming a `Composite` op and nothing
  actionable.

  The error now names the mechanism, lists the bounded parameters in your model, and gives the
  ways out (ADVI or NUTS; a prior with less mass at the edge; or checking the channel has
  enough spend variation to inform its carryover at all). This does not prevent the crash —
  preventing it means changing priors, which is the caller's decision. Note the `#207` prior
  change above happens to stop the originally-reported configuration from crashing, but the
  failure mode is unchanged. ([#203](https://github.com/redam94/mmm-framework/issues/203))

- **Backtests on a root-saturation model were computed from unsaturated forecasts.**
  `PosteriorForecaster._saturate` — the NumPy forward pass behind `run_backtest` —
  dispatched on four saturation families and returned its input unchanged for anything
  else, so `SaturationType.ROOT` fell through to the no-saturation return. Every metric
  the backtest reported for such a model was therefore wrong: MAPE, sMAPE, RMSE, MAE,
  bias, MASE, the naive-baseline comparison and the 50/80/95% interval coverage. Nothing
  raised; the backtest completed and reported plausible numbers. Measured on a real MAP
  fit, the saturated response was off by 0.27 on a transform whose range is `[0, 1]`.

  Saturation now has **one** NumPy definition — `mmm_framework.frequentist._transforms`,
  a bit-for-bit mirror of the in-graph `_apply_saturation_pt` — which both the forecaster
  and the new frequentist design matrix import, and an unhandled family raises rather than
  degrading to identity. If you have run a backtest on a model with `saturation="root"`,
  re-run it. ([#202](https://github.com/redam94/mmm-framework/issues/202))

## [1.2.0] — 2026-07-25

Sampler selection worked in one place and silently failed in four. Fixing that required new
public API — a third NUTS backend that was a declared dependency no code path could reach —
so this is a minor rather than a patch bump. It also raises an install floor and marks two
inference methods as deprecated, neither of which is patch-safe.

Plus a reporting correctness fix: cross-effect tables in multi-outcome reports listed pairs
the model never estimated.

The theme is knobs that read as configuration and were no-ops. If you set `target_accept`,
selected a frequentist inference method, or read a multi-outcome cross-effect table, your
results change.

### Fixed

- **`ModelConfig.target_accept` is now honored by `fit()`.** The fallback went straight to a
  literal `0.9`, so `ModelConfigBuilder().with_target_accept(0.95)` was a silent no-op — the
  first knob the sampling-failure playbook tells you to reach for, and the one that does
  nothing about divergences if it never reaches `pm.sample`. Precedence is now explicit
  argument → config → `0.9`. The field's own default is `0.9`, so an untouched config
  samples byte-identically to 1.1.0; if you *had* set it, your fits change.

- **`fit(nuts_sampler=...)` no longer raises on the core model.** `BayesianMMM.fit()` had no
  such parameter, so the keyword — which `BaseExtendedMMM.fit()` does accept, and which four
  places in the docs recommended — fell into `**kwargs` and collided with the explicit
  argument already passed to `pm.sample`, raising a `TypeError` that named `pm.sample` rather
  than anything the caller wrote. The same line worked on an extension model and failed on a
  plain one.

- **`frequentist_ridge` / `frequentist_cvxpy` no longer silently fit Bayesian NUTS.**
  Both are declared `InferenceMethod` values with builder methods advertising "Ridge
  regression (fast, frequentist)" and "CVXPY for constrained optimization". Neither has ever
  been implemented: `fit()` dispatches on `FitMethod`, never on `InferenceMethod`, and the
  package depends on neither `scikit-learn` nor `cvxpy`. Selecting one did not raise — it
  fitted a **full Bayesian posterior** via the `"pymc"` fallback in `ModelConfig.nuts_sampler`,
  so you asked for a fast frequentist point estimate, waited out MCMC, and got a posterior
  with no indication the request had been ignored.

  `fit()` now raises `NotImplementedError` naming the supported alternative, and constructing
  such a config emits a `DeprecationWarning`. **The alternative is `fit(method="map")`**: under
  Gaussian coefficient priors, maximum a posteriori estimation *is* ridge regression, so the
  capability the name promised has effectively been available all along.

  The enum values are **retained** — stored configs still parse and the frozen-enum contract
  stays green. The Excel template parser rejects the strings at parse time. The config fields
  that exist only for this path (`ridge_alpha`, `bootstrap_samples`, `optim_maxiter`) are
  likewise inert and now marked as such; they are the reserved surface for the real
  implementation, tracked in
  [#180](https://github.com/redam94/mmm-framework/issues/180).

- **Cross-effect summaries no longer report structurally-zero outcome pairs as estimated.**
  `reporting/helpers/mediated.py::compute_cross_effects` probed for `get_cross_effect_summary`
  (singular); both `MultivariateMMM` and `CombinedMMM` spell it `get_cross_effects_summary`,
  so the `hasattr` check never matched and every report fell through to the manual branch.
  That branch walks every off-diagonal `psi` entry, and the matrix starts from zeros with
  only declared specs filled — so undeclared pairs appeared alongside the real ones, and
  `effect_type` was dropped. If you have a multi-outcome report with cross-effect rows you
  did not declare, that is this bug; re-run to get the declared set.

### Added

- `InferenceMethod.BAYESIAN_NUTPIE` — the `nutpie` Rust NUTS sampler has been a core
  dependency since 1.0.0, but sampler choice was the binary `use_numpyro` with no
  representable value for it. Existing enum values are unchanged.
- `ModelConfigBuilder.bayesian_nutpie()` and `DAGModelBuilder.bayesian_nutpie()`, alongside
  the existing `.bayesian_pymc()` / `.bayesian_numpyro()`. The three sample the same graph;
  only the NUTS implementation differs. Verified end-to-end — a real MMM fit through nutpie
  agrees with the pymc backend.
- `ModelConfig.nuts_sampler` — the single resolver from inference method to the
  `pm.sample(nuts_sampler=...)` string. Non-Bayesian methods report `"pymc"` rather than
  raising, so a caller reading it on a frequentist config still gets the historical default.
- `BayesianMMM.fit(nuts_sampler=...)` — explicit argument, defaulting to the config. Accepts
  `"pymc"`, `"numpyro"`, `"nutpie"` or `"blackjax"`; ignored by the non-NUTS fit methods.
- `"bayesian_nutpie"` is accepted by the Excel config parser's inference-method mapping.

### Changed

- **`nutpie>=0.16.4` → `>=0.16.10`.** pymc 6.0 raises `ImportError` below `0.16.10`, so the
  old pin shipped a sampler that could not have started even if it had been reachable. A
  locked environment resolving the old floor needs to update.

### Deprecated

- `InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY`, and the corresponding
  `ModelConfigBuilder.frequentist_ridge()` / `.frequentist_cvxpy()`. They are unimplemented
  and now refuse rather than falling through to NUTS (see Fixed). They are **not** removed:
  removal would break the frozen-enum contract and is reserved for a major version. They will
  become live — not removed — when [#180](https://github.com/redam94/mmm-framework/issues/180)
  lands.

### Notes

- Extension models (`BaseExtendedMMM` — Nested / MV / Combined / Structural) keep their
  `nuts_sampler="pymc"` default deliberately. Their bespoke graphs are not all JAX-traceable,
  so inheriting a numpyro config would break fits rather than speed them up.
- If a pipeline of yours currently runs with a frequentist inference method, it will now fail
  loudly. That is the point: it was returning a Bayesian posterior for a frequentist request.
  Switch to `fit(method="map")` for the fast penalized estimate, or to an explicit Bayesian
  method to keep exactly what you were already getting.
- Internal only, no package surface: the docs code-snippet gate now covers Markdown
  (`README.md`, `CLAUDE.md`, `technical-docs/*.md`) as well as HTML, docs navigation
  registration is gated, and `docs/tools/build_seo.py` is idempotent.

## [1.1.0] — 2026-07-25

Two methodological fixes in `validation/`. Both corrected numbers that read as **more
trustworthy than they were**, so both change output you may have quoted.

A minor rather than a patch bump: the fixes are accompanied by new public API (a `weighting`
argument, new result fields, two new exported names), and one of them changes a default so
that a re-run returns a different number than 1.0.0 did.

### Fixed

- **Spec-curve model averaging no longer weights a causal estimand by predictive skill.**
  `run_spec_curve` applied LOO-stacking weights to per-channel ROI. Stacking
  (Yao et al. 2018) maximizes expected *predictive* utility — it answers "which mixture
  forecasts held-out `y` best?" — while a spec curve averages a *causal* estimand. The two
  objectives come apart exactly where MMM specs differ, because two specs can predict the
  KPI equally well while splitting that same fitted mean very differently between media and
  baseline. Worse, the direction inverts: a spec that overfits the confounder block often
  predicts *better* while being *less* trustworthy for the causal contrast, so stacking
  systematically upweighted the specs a causal analyst should trust least.

  The default is now **equal weights** over the pre-registered set, on the reasoning that
  pre-registration already asserted every variant is defensible, so there is no post-hoc
  predictive ground to promote one.

- **The unobserved-confounding robustness value is no longer inflated by tight priors.**
  `robustness_value` is strictly increasing in `|t|`, and `t = posterior_mean / posterior_sd`.
  Tightening a prior shrinks the posterior sd, which *raised* the reported robustness with no
  new evidence — so the most prior-dominated channel could report the most robust value, and
  robustness values were not comparable across channels with differently tight priors.

  Prior contraction is now computed per channel where a prior group exists. A channel below
  the `PRIOR_DOMINATED_CONTRACTION` threshold (0.20) renders as **"Not assessable
  (prior-driven)"** rather than a green "Robust", with a footnote explaining the inversion.
  "Could not check" is now reported distinctly from "checked and passed".

### Added

- `run_spec_curve(..., weighting=...)` — `"equal"` (default) or `"stacking"`. Requesting
  stacking opts back into the old behavior and logs a warning; it is defensible only when
  every spec in the set is identified for the same estimand and you genuinely want a
  predictive mixture. An unrecognized value raises `ValueError`.
- `SpecCurveResult.weighting` and `.predictive_weights` — the stacking weights are still
  computed and reported, deliberately, as a *diagnostic* rather than an input: divergence
  from uniform says predictive fit discriminates between your specifications, which is worth
  seeing and not worth acting on. `to_dict()` also emits `weighting_caveat`.
- `validation.spec_curve.WEIGHTING_CAVEAT` — the caveat text keyed by weighting mode, so
  reports and payloads state which weighting produced a number.
- `ChannelRobustness.prior_contraction`, `.is_prior_dominated` and `.rv_is_quotable`, plus
  `validation.sensitivity_unobserved.prior_inflation_warning()` and
  `PRIOR_DOMINATED_CONTRACTION`.

### Changed

- `_stacking_weights` returns `{}` on every fallback path instead of fabricating a uniform
  vector, so `predictive_weights` honestly distinguishes "unavailable" from "uniform".
- Report surfaces relabelled to match: the spec-curve blend prose is now derived from the
  weighting actually used, the weights table separates the applied weight from the
  (unapplied) predictive weight, and a prior-driven channel can no longer render green.

### Notes

If you have a stored `SpecCurveResult` or a report produced by 1.0.0, its model-averaged ROI
was stacking-weighted. Re-run to get the equal-weight number, or read the new
`predictive_weights` field to see how far the two diverge on your set.

## [1.0.0] — 2026-07-24

First stable release: the package was split into a lean modeling core and optional
application layers, the public contracts were audited and frozen, and the project adopted
strict semantic versioning. See the
[v1.0.0 release notes](https://github.com/redam94/mmm-framework/releases/tag/v1.0.0) for the
full list of breaking packaging changes, the contract freeze, and the audit fixes.

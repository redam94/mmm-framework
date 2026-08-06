# Forward forecast under a spend plan

`src/mmm_framework/planning/forecast.py` turns a *future* spend plan into a KPI forecast. Before
it existed the framework could replay a fitted model forward — `validation/backtest.py`'s
`PosteriorForecaster` does exactly that — but only backwards, to grade itself. The module's code
is small; the thing it has to get right is the caveat block, because a single mean line with a
tight band reads like a measurement, and a forward forecast is a counterfactual under a plan the
model has never observed. The governing rule is structural, not editorial: **a `ForecastResult`
without caveats refuses to render its headline.**

## Design

### Data flow

```
future_media (dict / flighting schedule / array)
  └─ _normalize_plan → (n_future, n_channels) raw spend, model channel order
model history (X_media_raw, X_controls_raw)
  └─ concatenated with the plan so ADSTOCK CARRYOVER into the window is correct
       └─ PosteriorForecaster.forecast(..., include_noise) → (n_samples, n_pos) draws
            └─ geo: reshape (draws, n_future, n_cells).sum(cells) → national total
                 └─ thin → summarize → ForecastResult(mean, lower, upper, by_channel,
                                                      baseline, draws_b64, caveats)
```

Key types, all in `planning/forecast.py`:

- **`forecast_under_plan(model, future_media, *, calendar, future_controls, interval,
  include_noise, max_draws, random_seed, strict)`** — the only constructor that matters. It
  always computes the caveats.
- **`ForecastResult`** — per-period `mean`/`lower`/`upper`, per-channel decomposition
  `by_channel` plus `baseline` (everything non-media), the thinned draws as `draws_b64`, and
  `caveats`. Methods: `headline()`, `draws()`, `window_total_interval()`.
- **`ForecastCaveats`** — every field is *computed* from the fitted model or the plan, never a
  template sentence, because a template sentence is equally true of a good forecast and a bad
  one. `statements()` renders them as prose.

### One signature, national and geo

The forward pass addresses national models on the PERIOD axis and geo models on the OBS axis
(period-major, cell-minor). `forecast_under_plan` resolves that asymmetry so callers never see
it. A geo model gets the national plan split across cells **in proportion to each cell's share of
training spend**, and the per-cell draws are summed back to the national total — the number a
plan is judged on. A per-cell figure would be a different quantity under the same name.

### The interval is predictive by default

`include_noise=True` includes observation noise, so the band is a PREDICTIVE interval — the
interval actuals will be compared against — rather than an interval on the mean. The mean band is
about a third the width (measured 0.342 in `tests/test_forecast.py`); grading a predictive
interval against a noiseless planted mean over-covers by construction, which is why the tests
grade both pairings and report the unflattering one.

### Why per-period draws ride along (`draws_b64`)

The window total's interval **cannot be recovered from per-period bounds**: periods are
correlated under the posterior and their errors partly cancel, so summing per-period bounds gives
a wider, wrong number. `window_total_interval()` therefore sums the stored draws per draw and
takes quantiles. Draws are thinned to `DEFAULT_MAX_DRAWS` (200) and stored base64 float32
little-endian (the `reporting/interactive/facts.py` encoding) — 4000×52 float64 per plan version
is how the sessions.db bloat incident started. Thinning happens **before** summarizing, via one
`_thin_index` subset applied to both the stored draws and the decomposition, so recomputing from
the artifact reproduces the reported numbers exactly (#225 compares actuals against the stored
draws, not the summary).

### The caveat fields

- `trend_extrapolation` (`{policy, trend_type, n_train_periods}`) + `interval_widens_with_horizon`
  — spline/GP/piecewise trends have no out-of-time continuation, so the forecaster holds the last
  level flat (`held_flat`) and the band does not widen with horizon; only a `linear` policy
  widens. A 26-week-out band the same width as a 1-week-out band is not a forecast of week 26,
  and the field says so.
- `extrapolated_channels` — future media is normalized by the *training* max, so a channel
  planned above observed spend asks the saturation curve about a region with no data. Flagged per
  channel with the multiple, reusing the budget optimizer's own test.
- `residual_autocorrelation` (`{ljung_box_p, lag, autocorrelated}`) — measured on training
  residuals with a numpy-only Ljung-Box. Deliberately NOT corrected by a fudge factor: an AR(1)
  predictive variance is a modelling change, and a widened band with no stated reason is worse
  than a narrow one with a stated reason.
- `approximate` / `fit_method` / `interval_noun` / `inference_family` — routed through
  `diagnostics.provenance`. A bootstrap trace carries the same variable names as a posterior and
  would otherwise render a CONFIDENCE interval as a credible one.
- `interval_available` / `n_posterior_draws` — a MAP posterior has ONE draw; below
  `MIN_DRAWS_FOR_INTERVAL` (10) the bounds are NaN and the headline carries `total_lower=None`.
  Per #249, a zero-width band is the visual language of extreme precision — the opposite of what
  a single-draw posterior means — so the interval is reported as *absent*, honestly, rather than
  collapsed. Frequentist ridge fits with bootstrap replicates keep a real interval and are not
  stamped approximate (#188: a penalized point estimate with bootstrap intervals is not an
  approximation of a posterior).

## Refusals

Each refusal exists because the silent alternative is a forecast of something other than what the
caller asked for.

- **`headline()` on a result without caveats raises `ValueError`.** The caveats travel with the
  number or the number does not travel; `forecast_under_plan` always computes them, so only a
  hand-built `ForecastResult` can hit this.
- **Missing `future_controls` when the model has controls.** Future control values are a planning
  assumption with no defensible default (zero, last-observed, and mean each encode a different
  world). Refused by name of the controls involved.
- **A plan channel the model does not know.** Refused by name rather than dropped — a silently
  ignored plan line is a forecast of a different plan.
- **Ragged period counts** across channels, an **empty plan**, `future_controls` covering a
  different number of periods than `future_media`.
- **An unfitted model** (`_trace is None`) and an **`interval` outside (0, 1)**.
- A caveat computation, by contrast, must never fail the forecast — `_residual_autocorrelation`
  swallows its own errors and reports all-`None` (see gotchas).

## Interactions

- **`validation/backtest.py`** — `PosteriorForecaster` does the graph-faithful forward math
  (`forecast`, `media_by_channel_at`, `trend_extrapolation`, `unsupported`). This module owns
  plan handling, geo summation, thinning/encoding, and the caveats.
- **`planning/flighting.py` / `planning/calendar.py`** — a flighting schedule dict is an accepted
  `future_media` shape; a `PlanningCalendar` supplies dated period labels (otherwise `t+1…`).
- **`platform/plan_of_record.py`** — the commit gate stores the forecast snapshot and, since
  #227, the reproduction inputs `plan_media` / `plan_controls` / `random_seed` *inside* the
  snapshot (emitted by the `forecast_plan` op in `agents/model_ops.py`). Reproduction re-runs
  `forecast_under_plan` with exactly those inputs; a snapshot without a plan refuses to
  reproduce. Reading them only from the payload top level was a live defect — every real
  commitment refused with "records no per-period spend plan".
- **`planning/variance.py` / `platform/variance.py`** — variance-to-plan runs the same function
  on plan and actuals and diffs the results.
- **`agents/tools.py`** — the oracle's `forecast_under_plan` tool wraps this function.
- **`diagnostics/provenance.py`** — interval vocabulary and the approximate stamp.

```python
from mmm_framework.planning import forecast_under_plan

fc = forecast_under_plan(
    model,                       # a fitted BayesianMMM (national or geo)
    {"TV": [50_000.0] * 26, "Search": [20_000.0] * 26},
    future_controls={"price": [9.99] * 26},   # REQUIRED when the model has controls
    interval=0.9,
    include_noise=True,          # predictive interval — what actuals are judged against
    random_seed=42,
)
headline = fc.headline()         # raises if caveats were absent; here they never are
lo, hi = fc.window_total_interval()
```

## Test anchors

`tests/test_forecast.py`: `TestHeadlineRequiresCaveats`, `TestCaveatsAreComputed`,
`TestDecompositionAndShapes` (parts-equal-whole, draws round-trip, window-total vs summed
bounds), `TestPlanNormalization` (the refusals), `TestForecastAccuracyAndCoverage` (slow; both
coverage pairings, seasonal-naive baseline), `TestGeoPanel`, `TestFrequentistForecast`.

## Gotchas

- `_residual_autocorrelation` has a blanket `except` by design (a caveat must never fail the
  forecast). That same blanket once swallowed an `AttributeError` on a method that did not exist,
  so for every core model the autocorrelation caveat silently never fired. If you touch the
  fitted-mean reconstruction, verify `ljung_box_p` is non-`None` on a core fit, not just that the
  code runs.
- `interval_widens_with_horizon` is derived from the policy string (`policy == "linear"`), not
  measured; `held_flat` and `none` both mean the band width is driven only by parameter + noise
  uncertainty. The test pins the band at horizon 26 to within 15% of horizon 1 under a spline
  trend so the field cannot describe the wrong thing.
- `by_channel` values are posterior means over the *thinned* subset, scaled by `y_std` to the
  original KPI scale, and `baseline = mean − Σ by_channel` — parts equal whole to 1e-9 by
  construction. Compute the baseline any other way and geo models will drift.
- The equal-tailed percentile interval matches `compute_hdi_bounds`; do not swap in an HDI here
  without changing the reproduction path in `plan_of_record.py`, which recomputes and diffs
  `lower`/`upper` elementwise.

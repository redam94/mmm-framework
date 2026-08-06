# Lever optimization — decision arms, promo ROI, and the price refusal (#226)

The budget optimizer's decision vector was dollars of media spend, `Σs = B`. A promotion does
not fit that vector: its cost is margin given away (depth × price × units), not a spend line,
and a price change has no "budget" at all. This spec covers the planning-side machinery that
lets promo depth compete with media for the same money, and the refusals that keep price and
unpriceable promos out of the allocator. The model-side lever mechanism — how `with_price()` /
`with_promotions()` reach the graph, the sign guards, the elasticity interpretation — is
[`price-promotion-levers.md`](price-promotion-levers.md); read that first, this builds on it.

## The governing reduction: per-arm cost bases

Everything here is one idea: **re-parameterize every decision by its realized cost**, then run
the existing dollar allocator untouched.

- A **media arm**'s cost is its spend — `cost_fn=None`, the identity. The media path is
  bit-identical to the pre-arm allocator (pinned by test).
- A **promo arm**'s decision is *average weekly depth*; its cost is `depth × unit_cost`,
  linear, so the arm shares the media multiplier grid exactly (`cost = base_cost × m`).

Because the joint decision space is homogeneous dollars, the KKT water level stays a single
number and no per-group shadow-price correction is needed — the correction the issue
anticipated for a level-space decision vector is made unnecessary by construction. The
per-arm `marginal_roas` labels change meaning to "per dollar of realized cost", which is
exactly what makes a media dollar and a promo margin dollar comparable.

### Key types (`src/mmm_framework/planning/decision_arms.py`)

- `DecisionArm` (frozen dataclass) — `name`, `kind` (`"media"`/`"promo"`), `levels` (the
  arm's grid in its OWN units: dollars or depth fraction), `level_units`, `base_level`,
  `obs_min`/`obs_max` (support bounds in level units), `cost_fn`. `arm.cost(level)` prices a
  level; identity for media.
- `ArmCurves` — **subclasses** `ResponseCurves` rather than wrapping it, so every existing
  consumer (`optimize_budget`, `budget_frontier`, the per-draw loop, `within_observed_range`)
  runs unchanged; the arm metadata rides along. `level_for_cost(arm_name, cost)` interpolates
  a recommended cost back to the arm's own level. `is_mixed` is true when kinds differ.
- `PromoArmResult` — a promo ROI "with everything it cannot travel without": the interval
  and its mass, the frozen cost basis (`realized_cost`, `unit_cost`, `avg_depth`), the
  valuation and its source, and `caveats` stating that the economics are inputs, not
  measurements.

### Data flow

`build_arm_curves(model, promo_var=..., unit_cost=...)`:

1. Media curves come from `budget.compute_response_curves` unchanged.
2. The promo curve is one paired `model.sample_lever_contributions(...)` pass per grid point
   — promo column set to `full(n, base_depth × m)`, differenced against a zero-depth pass,
   shared seed so the draws difference cleanly and the curve anchors at 0.
3. Both are stacked on the shared multiplier grid; the promo arm's `obs_max_spend` entry is
   the cost of the deepest single observed week, so `within_observed_range` flags an
   out-of-support depth exactly as it flags spend.

`optimize_arms(...)` is a thin orchestration over `budget.optimize_budget` (which owns the
risk objectives, constraints, per-draw uncertainty, and the concavity gate), then re-labels
result rows with `arm_kind`, `level_units`, and `optimal_level` — a depth, not a dollar
figure, for the promo arm. Profit is `mode="free"` (maximizes `value·KPI − Σcost`, refuses
without a valuation via `finance.UnresolvedValueError`); under `mode="fixed"` the valuation
is optional (scale-free argmax).

`model.sample_lever_contributions` (in `model/base.py`) is the substrate: posterior draws of
the `price_component` / `promo_component` Deterministics under a swapped RAW `X_levers`
matrix, returned in **original KPI units** (the `* y_std` is load-bearing — the graph
registers lever components on the standardized scale). Components not configured are absent,
never zero-filled. It raises on unfitted models, lever-free models, and multiplicative
specs (components are additive on the log scale there; use the LMDI decomposition).

```python
from mmm_framework.planning import promo_roi, optimize_arms, price_whatif

res = promo_roi(model, "promo_depth", unit_cost=180_000.0, value_per_kpi=42.0)
plan = optimize_arms(
    model, promo_var="promo_depth", unit_cost=180_000.0, total_budget=500_000.0
)
scenario = price_whatif(model, 0.95)  # scenario["recommendation"] is always None
```

## Refusals

Each of these fails loudly, by name, with the fix in the message — never a silent default.

- **Flag-valued promo** (`_refuse_flag_or_unknown_units`, hit by both `promo_roi` and
  `build_arm_curves`): a 0/1 event flag has no depth, so no ΔP×Q cost exists; dividing by a
  normalized flag prints a ratio with no units. Supply the actual discount-depth column.
- **Unknown-unit promo** (same gate): values outside [0, 1] are not a discount fraction. The
  model normalizes internally, but a COST cannot be priced on a normalized column of unknown
  units. Supply depth as a fraction (0.25 = 25% off). All-zero depth also refuses — there is
  no observed depth to curve.
- **Missing valuation**: `promo_roi` with `value_per_kpi=None` raises
  `finance.UnresolvedValueError`. A silent `value_per_kpi=1.0` is the exact defect
  `mmm_framework.finance` exists to prevent. Zero/negative realized cost also refuses
  (nothing was given away, so no ROI exists), as does a non-positive `unit_cost`.
- **`price_whatif` never recommends**: `recommendation` is always `None` with
  `refusal_reason` stating why — the repo's own published measurement
  (`docs/blog-modelled-one-p.html`) recovers ~39% of a planted price elasticity under the
  shipped mechanism (9% with no designed variation), so an optimizer pointed at that
  coefficient would recommend moves whose real P&L consequence is ~2.5× what the model
  believes. The refusal holds *regardless of the endogeneity screen*: a lead/lag flag not
  firing is weak evidence of exogeneity, not a licence. It evaluates the stated scenario,
  labels the elasticity as attenuated, and tells you to run a pricing experiment.
- **Non-monetary channel divisor** (`budget.compute_response_curves`): a channel whose
  resolved divisor is not monetary (impressions/clicks with no spend column or CPM/CPC
  basis) refuses — its "base spend" is a volume wearing a dollar label, and it would be
  summed into `total_budget` and traded against genuine spend at one shadow price. Refusal,
  not warning: the number is wrong, not questionable.
- **Concavity gate** (`budget.check_concavity` + the gate in `optimize_budget`): the greedy
  water-fill is exact ONLY for concave curves, and a promo depth-response over a realistic
  range need not be concave. This one degrades rather than refuses: a non-concave arm routes
  the whole solve through the multi-start SLSQP path (seeded from greedy, so it can only
  match or beat it) and appends a note naming the arm and the "best found, not proven
  global" caveat.
- **`goal_seek` refuses a mixed-arm portfolio** (`planning/frontier.py`): the bisection's
  monotone-concave-frontier premise is proven for concave spend-response curves only; one
  non-concave promo arm voids it, and the "minimum budget reaching the target" becomes a
  local artifact presented as an inverse solve. Optimize at candidate budgets with
  `optimize_arms` and read the frontier directly instead.

## The endogeneity screen and lever kinds

`diagnostics/endogeneity.py` is a pre-fit Granger-style lead/lag asymmetry on **differenced**
series. The direction matters: a *lagged* KPI change predicting a *current* lever change
(demand → setting) is the endogeneity signature — planners cutting price into weak demand,
clearance behaviour. Contemporaneous correlation between a lever and demand is the response
the model is there to measure, not endogeneity evidence, and differencing removes the shared
trend/seasonality that would spuriously correlate everything. Since #226 the screen covers
levers as well as media: each row carries a `kind` (`"media"`/`"price"`/`"promo"`, price
identified via `model._price_lever`), flagged levers land in `flagged_levers`, and the note
states the consequence precisely — an endogenous lever's fitted elasticity is *attenuated or
wrong-signed by construction, not merely uncertain*. A NOT-flagged lever remains weak
evidence of exogeneity; the price refusal above does not lift on a clean screen.

## Grading: the joint solve is scored on planted profit

`synth/dgp.py::make_promo_and_media` (registered as `"promo_and_media"`) plants a world where
the true profit optimum wants promo money, with the answer key frozen *before* the optimizer
runs and round-tripped through JSON. `tests/test_decision_arms.py` fits the world, runs
`optimize_arms`, and grades the recommendation on **true planted profit**, not model-believed
profit — the milestone test is that the joint (media + promo) solve beats the media-only
solve on true profit, and a discrimination test shows modeling promo as a mere control misses
the split by ~15pp. Recovery tests pin the exogenous world (lift and depth-curve alpha
recovered) against the endogenous world (attenuation, with a floor).

## Interactions

- **`planning/budget.py`** owns the allocator, objectives, constraints, and the concavity
  gate; `decision_arms` only builds curves and re-labels. **`planning/frontier.py`** carries
  the mixed-portfolio `goal_seek` refusal. **`finance/`** supplies `UnresolvedValueError`
  and the valuation-source discipline. **`model/base.py`** supplies
  `sample_lever_contributions`; **`reporting/helpers/measurement.py`** supplies
  `resolve_channel_divisor` for the monetary-divisor refusal. Model-side lever config and
  identification caveats: [`price-promotion-levers.md`](price-promotion-levers.md).

## Test anchors

- `tests/test_decision_arms.py` — concavity gate, analytic equal-marginal-profit checks,
  every refusal by name, promo-world registration + frozen answer key, joint-vs-media-only
  on true profit, exogenous/endogenous recovery, the lever endogeneity screen, price what-if.
- Demo notebook: `nbs/demos/promo_depth_optimization.ipynb`.

## Gotchas

- `promo_roi`'s cost basis is *average* depth × `unit_cost` — `unit_cost` prices one unit of
  average weekly depth over the window, not one promoted unit. Get the basis wrong and the
  ROI is off by exactly that factor; the result's `caveats` restate the basis for this reason.
- The promo arm's per-observation cost divides by `n_obs` (average-year basis), so its
  max-observed multiplier works out to `max(depth)/mean(depth)` — a spiky calendar makes the
  support bound much wider than the mean suggests.
- `ArmCurves` truncates to the smaller draw count when stacking media and promo draws
  (`D = min(...)`); use the same `max_draws` for comparable interval mass.
- Duck-typed fakes without measurement config stay allocatable: the divisor refusal
  deliberately swallows resolver errors so test stubs pass — it only refuses a channel that
  *resolves* to a non-monetary divisor.

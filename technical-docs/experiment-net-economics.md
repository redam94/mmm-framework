# Net Experiment Economics — Test Revenue Loss vs Reallocation Gain (Phase 3)

> Status: implementation spec. Companion to `experiment-economics.md` (the
> existing opportunity-cost + EVOI machinery) and `experiment-methods-framework.md`.
> Delivers the single decision figure the client asked for: **expected revenue
> lost while running the test** netted against the **expected reallocation upside
> the learning unlocks**, with honest uncertainty — so "run it or not" has one
> number, not two half-answers.

## 0. The gap this closes

Both halves already exist but are never netted into one economic figure
(confirmed in the planning inventory):

- **Loss side** — `opportunity_cost.compute_opportunity_cost` (`opportunity_cost.py:333`)
  → `net_profit_impact` (signed), `opportunity_cost_dollar = max(0, −net)`,
  `forgone_kpi`, `spend_at_risk`. This is the short-term $ cost of deviating from
  BAU during the test window.
- **Gain side** — `evoi.compute_evoi_for_channel` (`evoi.py:138`) → EVOI in **KPI
  contribution units** (the value of the sharper posterior to the budget decision),
  and `budget.optimize_budget` / `default_reallocation` (`budget.py:434, 735`) →
  the reallocation the learning would justify.

They are bridged today only by a unitless `learning_to_cost_ratio`
(`opportunity_cost.py:459-480`) and the optimizer's tradeoff axis
(`experiment_optimizer.py:_tradeoff:393`). There is **no function that returns
`E[reallocation gain] − E[test loss]` in dollars with a distribution.**

## 1. Verified facts

| # | Fact | Source |
|---|---|---|
| E1 | EVOI is in **KPI contribution units** (paired preposterior MC over reweighted posterior, re-optimized allocation). Cap at EVPI. | `evoi.py:73-160`, OC-5 |
| E2 | `compute_opportunity_cost` returns signed `net_profit_impact = margin·kpi_delta − spend_delta` and `evoi_kpi_units` is already an input — so the loss module already *sees* an EVOI in KPI units. | `opportunity_cost.py:54, 333, 459-480` |
| E3 | KPI→$ conversion: `margin_per_kpi` (revenue KPI) or `price` (units KPI); `kpi_kind` selects. The measurement resolver (`reporting/helpers/measurement.py`) gives per-channel monetary vs efficiency reference. | `opportunity_cost.py:277`, CLAUDE.md measurement row |
| E4 | EVOI is a **one-shot** value of the decision *now*; a realized experiment's value persists over a horizon and decays (`eig.channel_half_life`, `reexperiment_due`). `response_horizon_weeks` already parameterizes the loss module. | `eig.py:156-178`, `opportunity_cost.py:333` |
| E5 | `experiment_economics` op assembles design + anchor + opportunity_cost + simulation into `dashboard["experiment_economics"]`; loose JSON. | `model_ops.py:2051`, F12 |

## 2. The net figure — `planning/experiment_value.py` (new)

One module, one headline dataclass. All in **dollars**, all with a distribution
(reusing the draw-paired posterior passes the two halves already compute — no new
sampling).

```python
@dataclass(frozen=True)
class ExperimentNetValue:
    # loss side ($, signed downside is positive-cost)
    test_loss_dollar: float            # E[opportunity_cost_dollar] over the test
    test_loss_hdi: tuple[float, float]
    # gain side ($)
    reallocation_gain_dollar: float    # E[realized reallocation upside] over horizon
    reallocation_gain_hdi: tuple[float, float]
    # net
    net_value_dollar: float            # gain − loss (the headline)
    net_value_hdi: tuple[float, float]
    prob_net_positive: float           # P(gain − loss > 0) over draws
    breakeven_horizon_weeks: float | None   # weeks of gain to repay the test loss
    # provenance
    horizon_weeks: float
    evpi_dollar: float                 # perfect-info ceiling (gain can't exceed this)
    assumptions: dict                  # margin, discount, decay half-life, ...
    basis: str                         # 'model_anchored' | 'evoi_bounded' | 'insufficient'
```

### 2.1 Loss → dollars

Reuse `compute_opportunity_cost`. Headline loss = `opportunity_cost_dollar`
(the non-negative downside, OC-3). Carry its per-draw distribution (the paired
BAU-vs-experiment contribution draws) → `test_loss_hdi` via
`utils/arviz_compat.hdi_bounds`. Uses the **with-carryover** cost (OC-7) since the
deviation's KPI effect outlives the window.

### 2.2 Gain → dollars

EVOI is the value of the learning to the *current* one-shot decision, in KPI
units (E1). Convert and extend to a realized experiment's value:

1. **KPI→$**: `evoi_dollar_now = kpi_to_dollars(EVOI, margin_per_kpi | price, kpi_kind)`
   (E3). EVPI likewise → `evpi_dollar` (the ceiling).
2. **Realized precision, not prior EVOI**: EVOI as computed uses the *design's*
   `sigma_exp`. When a model anchor is available, use the **realized**
   `sigma_exp` from `design_anchor.realized_sigma_exp_for_anchor` (ANCHOR-1) so
   the gain reflects the precision this *specific* method+duration actually buys
   (a weak design learns less → smaller gain). This is the same override already
   plumbed into `priority.compute_experiment_priorities(roi_draws_overrides=...)`.
3. **Horizon + decay**: the learning informs allocation for
   `response_horizon_weeks`, decaying at the channel's info half-life (E4). Value
   over the horizon = `evoi_dollar_now · Σ_{w<H} discount^w` where
   `discount = decayed_sigma`-consistent geometric weight (reuse
   `eig.channel_half_life`). Optional financial `discount_rate` on top.
4. `reallocation_gain_dollar = min(evpi_dollar_horizon, evoi_dollar_horizon)` —
   EVOI can never exceed EVPI (E1).

### 2.3 Net + break-even

- `net_value_dollar = reallocation_gain_dollar − test_loss_dollar`.
- Distribution: pair the loss draws and the gain draws (both derive from the same
  posterior; where a clean pairing isn't available, convolve independently and
  flag `basis='evoi_bounded'`). `prob_net_positive = mean(gain_d − loss_d > 0)`.
- `breakeven_horizon_weeks` = smallest `H` where cumulative discounted gain ≥
  test loss (None if never, or 0 for a net-positive money-saving holdout — recall
  a go-dark holdout can have negative test loss, OC-3).

### 2.4 Degradation

No model → `basis='insufficient'`, return the loss side + a *bounded* gain (EVOI
with the design's prior `sigma_exp`), clearly labeled as not model-anchored.
Budget-neutral flighting (≈0 test cost) → net ≈ gain; headline `prob_net_positive`
near 1 unless the channel is already precise (`channel_already_precise`).

## 3. Public API

```python
def compute_experiment_net_value(
    mmm, design, *, curves, optimization, roi_draws,
    margin_per_kpi=None, price=None, kpi_kind="revenue",
    response_horizon_weeks=26, discount_rate=0.0,
    evoi_result=None, opportunity_cost_result=None,   # reuse if already computed
    max_draws=200, random_seed=0,
) -> ExperimentNetValue
```

Accepts already-computed `evoi_result` / `opportunity_cost_result` so the
optimizer's shared BAU pass (`experiment_optimizer.py:499-502`) isn't duplicated.

## 4. Wiring

- **`planning/experiment_value.py`** — new (§2–§3).
- **`planning/experiment_optimizer.py`** — SHIPPED (2026-07-19): the Pareto
  cost axis IS `−net_value` when a margin is known. The per-candidate EVOI
  that made this prohibitive is priced by a **calibrated Gaussian surrogate**
  (`evoi.fit_evoi_surrogate` / `EvoiSurrogate`): the Raiffa–Schlaifer
  preposterior form `EVOI(σ) ≈ k·s(σ)·Ψ(δ/s(σ))` with
  `s(σ) = τ·sqrt(τ²/(τ²+σ²))` (sd of the preposterior mean), where `(k, δ)`
  are fitted to TWO anchored preposterior-MC EVOIs placed at the extremes of
  the grid's design precisions (`_evoi_anchor`, common random numbers across
  the two anchors) — every candidate interpolates inside the calibrated
  bracket, where the surrogate tracks the exact MC to ~±15% even on a bimodal
  ROI posterior (extrapolation beyond the weak anchor under-estimates, which
  is why anchors sit at the extremes). Each candidate's `sigma_exp = MDE/2.8`;
  its surrogate EVOI is decayed/EVPI-capped through
  `compute_experiment_net_value` (summary-only shim over the candidate's
  already-computed opportunity-cost medians — zero extra posterior passes)
  and the axis swap is ALL-OR-NOTHING (`_apply_net_value_axis`; any
  unpriceable candidate leaves the whole grid on the legacy cost axis).
  Payload: `net_value_axis`, `evoi_anchor` (anchors + k/δ/τ/EVPI provenance),
  per-candidate `net_value`/`evoi_kpi`/`reallocation_gain`/`sigma_exp`,
  `tradeoff_basis="net_value"` (tradeoff = −net_value, lower-better Pareto
  convention). Knobs: `net_value_axis=True`, `response_horizon_weeks=26` on
  grid/suggest/op/endpoint. FE DesignStudio plots the net value itself
  (up = better), table/detail cards show signed $ net value.
- **`agents/model_ops.py`** — `experiment_economics` op adds a
  `net_value: {...}` block to `dashboard["experiment_economics"]`. Also expose a
  focused `experiment_net_value` op (`allow_unfitted=False`).
- **`agents/tools.py`** — `design_experiment_plan` / `simulate_experiment` /
  `suggest_experiment` surface the net figure as the **headline decision line**
  ("Expected net value of this test: +$X (78% chance net-positive); break-even in
  N weeks").
- **`api/main.py`** — rides the existing `experiment_economics` payload
  (`experiment-design/simulate`); no new endpoint required.
- **Frontend** — `DesignStudio.tsx` `OpportunityCostPanel` gains a **net-value
  headline card** (gain vs loss bar + P(net>0) + break-even) above the existing
  forgone-KPI detail. The Experiments home "next action" (`main.py:2938-2952`)
  can rank channels by net value.

## 5. Build order

1. `experiment_value.py` core (KPI→$ + horizon/decay + net + distribution).
2. Reuse-path wiring into `experiment_economics` op (accept precomputed halves).
3. Optimizer `_tradeoff` net-value option.
4. Agent headline line + DesignStudio net-value card.
5. Tests + doc.

## 6. Tests — `tests/test_experiment_net_value.py`

- KPI→$ conversion correct for revenue (`margin_per_kpi`) and units (`price`).
- Gain ≤ EVPI ceiling always; horizon extension monotone in `response_horizon_weeks`;
  decay reduces gain vs no-decay.
- Realized-precision path: a weaker design (larger `sigma_exp`) yields smaller
  gain than a strong one on the same channel.
- Net + break-even: a cheap high-learning test is net-positive with high
  `prob_net_positive`; an expensive low-learning test is net-negative; a
  budget-neutral money-saving holdout has ≤0 loss → net ≈ gain, break-even 0.
- Degradation: no-model path returns loss + bounded gain with `basis` set.
- Distribution: `net_value_hdi` finite, `prob_net_positive ∈ [0,1]`, no NaN leak
  (finite-guard like the PPC extractors).

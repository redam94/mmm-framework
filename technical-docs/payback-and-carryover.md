# Payback horizons and the carryover kernel

"TV pays back in 3 weeks" is the most CFO-legible number an MMM can emit, and it rests on
the model's *least identified* parameter. The adstock decay sits on an equifinality ridge
with saturation and the coefficient, its default prior differs 2x in implied half-life
depending on which constructor built the config, and the fitted kernel is truncated at
`l_max` and renormalized — which makes every horizon read off it structurally optimistic.
This subsystem exists so that number ships with its epistemics attached, or refuses. It
covers `planning/payback.py` (issue #224), `planning/discount.py`, the kernel reader in
`transforms/carryover.py` (issue #218), and `reporting/evidence.py::carryover_learning`.

## The word "payback" means two different things — so there are two functions

- **`channel_payback(model)` — response timing.** When does the *effect* land: the
  per-draw interpolated lag where the cumulative carryover kernel crosses a threshold.
  `DEFAULT_THRESHOLDS = (0.5, 0.9)` → `t50` / `t90`. This is NOT cash break-even; it is a
  statement about the shape of the kernel, and it needs no valuation.
- **`payback_breakeven(model, value_per_kpi=...)` — the finance sense.** The lag at which
  cumulative *discounted dollar* return on a dollar of spend reaches 1. Composition:
  per-draw ROI × `value_per_kpi` gives dollars back; the kernel times when they land;
  `discount_weights` prices the delay. Draws that never reach 1 within the window become
  `prob_never`, not dropped draws.

Do not add a third function that blends them, and do not present a `t50` as break-even —
splitting the word was the point of #224.

## Data flow

```
model._trace ──▶ posterior_carryover_kernels(model)         # transforms/carryover.py
                   │  per-draw (n_draws, l_max) kernels, family-aware,
                   │  delegates to transforms.adstock.adstock_weights —
                   │  the SAME function the model graph uses
                   ▼
                 carryover_crossing_lags(kernel, share)     # per-draw crossing, interpolated
                   ▼
channel_payback ──▶ PaybackResult{channels: {ch: ChannelPayback}}
   gates: carryover_learning(learning, ch)   # reporting/evidence.py
          Ljung-Box + PPC lag-1 autocorrelation (_ppc_acf1)
          truncated_tail_mass (from CarryoverKernel)
payback_breakeven ──▶ {ch: BreakevenResult}  # + resolve_channel_divisor + discount_weights
```

Key types: `CarryoverKernel` (channel, family, l_max, per-draw `kernel`, `status`,
`truncated_tail_mass`), `ChannelPayback` / `PaybackResult`, `BreakevenResult`. All have
`to_dict()`; consumers are `planning/history.py` (run metrics), `reporting/extractors/
mixins.py`, `reporting/interactive/facts.py`, and `agents/model_ops.py`.

```python
from mmm_framework.planning.payback import channel_payback, payback_breakeven

res = channel_payback(model)                       # PaybackResult
tv = res.channels["TV"]                            # status: ok | downgraded | refused
be = payback_breakeven(model, value_per_kpi=42.0)  # refuses without a valuation
```

## Invariants

1. **Every kernel comes from `posterior_carryover_kernels`.** It delegates to
   `adstock_weights`, the function the graph itself uses, so "agrees with the model" holds
   by construction. Five prior readers disagreed (family-blind `alpha**lags`, dropped
   Weibull channels, a legacy Beta *mixture weight* rendered as a decay rate); do not
   write a sixth.
2. **Per-draw, then summarize.** The crossing is convex in `alpha`, so the ETI of
   per-draw crossings ≠ the crossing of `mean(alpha)` (a real posterior understated the
   lag-5 weight 7x that way). No public function here derives a horizon from a mean
   parameter.
3. **Every requested channel appears in the output** — as a horizon, a downgrade, or a
   named refusal. Nothing is silently dropped (the old reader dropped Weibull channels
   with a log line).
4. **Truncation bias is disclosed, and it points SHORT.** The kernel is cut at `l_max`
   and renormalized inside the window, so every horizon understates the untruncated one
   (geometric `alpha=0.8` at `l_max=8` reads t90≈5.8 against a true 9.3). Above
   `TAIL_MASS_CAVEAT_MIN = 0.10` untruncated tail mass, the caveat is promoted from a
   field to a sentence and the status to `"downgraded"`.
5. **The carryover-learning gate.** The evidence tier attributes only effect-size
   parameters (`beta_`/`roi_`) to a channel; a horizon depends on
   `adstock_alpha_/theta_/shape_/scale_` instead. `reporting.evidence.carryover_learning`
   applies the same worst-case attribution rule to that parameter set. Contraction below
   0.10 (mirroring `DEFAULT_CONTRACTION_MIN`, unless the verdict is
   strong/moderate/relocated) marks the horizon `prior_dominated`: it is a restatement of
   the assumed prior, and two defensible priors ship with half-lives differing 2x.
6. **The autocorrelation gate.** Residual Ljung-Box plus a posterior-predictive lag-1
   autocorrelation check (the PPC catches what whitened residuals hide: on
   `adstock_misspec` Ljung-Box reads p=0.16 while the PPC is extreme at p=0.004). Firing
   means the carryover window is misspecified and the horizon is a *lower bound*. A gate
   that could not run reports UNTESTED — never a pass.
7. **Interval provenance rides the output.** Frequentist fits report
   `interval_kind="confidence"`; a collapsed interval (MAP/ADVI, or a pair identical at
   3 decimals, #249) renders `lower/upper = None` with a caveat — an absent interval is
   not evidence of precision.
8. **Discounting is one function.** `discount_weights(horizon_weeks, rate_annual=...)`
   uses exact geometric compounding `(1+r)^(-w/52)`; week 0 is never discounted. The
   default rate is 0.0 *deliberately*: at MMM horizons a 10%/yr rate moves the mean
   weight by under 5% (measured 0.33%–2.4% across repo surfaces at 26–52 weeks) — small
   against kernel and effect-size uncertainty, so a rate is a disclosed input, never a
   silent assumption. `mid_horizon_discount_factor` names the CLV MVP approximation.

## Refusals

A payback that cannot be computed honestly is a refusal, not a number. Family refusals
(`_refusal`, checked specific → generic) apply to both functions:

- **Dual-stock brand models** — detected by posterior *variables* (`brand_retention` /
  `long_term_fraction`), not class name, so user-authored copies of `LongTermBrandMMM`
  still refuse. `adstock_alpha_<ch>` there is only the FAST stock; a kernel horizon is
  dramatically too short while the model's own long-term split says the opposite.
- **StructuralNestedMMM with AR(1) mediators/factors** — persistence lives in the state's
  ρ, on a stated ridge with the per-channel α; a kernel-only horizon ignores the slow
  mediated path. The message names the AR(1) spec.
- **Extension models** (Nested / Multivariate / Combined) — one hardcoded geometric
  family at fixed `l_max=8` plus mediated paths the media kernel does not time. Detected
  by `BaseExtendedMMM` in the MRO *by name*, so this module never imports the extension
  stack.

Per-channel and call-level refusals:

- **Unreadable kernel** (`unsupported` / `missing_params` status) — reported per channel
  with the reason; `family="none"` is NOT a refusal (all effect lands at lag 0, and that
  is the answer).
- **`payback_breakeven` without a valuation** raises
  `mmm_framework.finance.UnresolvedValueError` — a silent `value_per_kpi=1.0` is exactly
  the defect `finance/` exists to prevent.
- **Efficiency-measured channels** (`MetricMeta.is_monetary` False via
  `resolve_channel_divisor`) — an impressions-denominated contribution has no dollar-in
  to pay back (#221). Also refused: no positive spend.
- **Unknown `basis`** and **unfitted model** raise `ValueError` immediately.

## basis="counterfactual" cross-check

The graph is `beta·sat(adstock(x))`, so the kernel basis is exact only under a linear
response. `basis="counterfactual"` measures the true incremental profile
`beta·sat'(a_t)·w_lag` by differencing two shared-seed `sample_channel_contributions`
passes around a one-period mean-spend pulse, and reports `kernel_t50_mean` alongside so
the disagreement between the two bases is measured once. If the pulse pass fails, the
result falls back to kernel basis *and says so* — never a silent basis switch.

## Interactions

- `transforms/carryover.py` also feeds the reporting adstock surfaces; the legacy-blend
  branch (`status="legacy_blend"`) reconstructs the two-alpha IIR mixture at a stated
  26-lag truncation.
- `planning/forecast.py` supplies `_residual_autocorrelation`; a forecast's
  `caveat_fields` dict can be passed in to skip recomputation.
- `planning/history.py` persists payback per run; the serializer records basis / family /
  l_max / tail mass (see `TestPersistenceProvenance`).
- `discount_weights` is also the one implementation behind
  `planning.experiment_value.compute_experiment_net_value` and the CLV garden model —
  two sites previously disagreed on the same convention.

## Test anchors

`tests/test_planning_payback.py` — analytic closed-form crossings and the
ETI-of-transform proof (`TestCrossingLagsAnalytic`), discount legacy-compat and measured
smallness (`TestDiscount`), planted recovery on `synth.dgp.make_clean`
(`TestPlantedRecoveryClean`), the biased-SHORT direction and PPC-vs-Ljung-Box split on
the misspec world (`TestNegativeControlMisspec`), prior sensitivity
(`TestPriorSensitivity`), all three family refusals by name (`TestRefusalsByName`),
MAP-collapse and frequentist provenance (`TestProvenanceAndCollapse`). Walkthrough:
`nbs/demos/payback_horizon.ipynb`.

## Gotchas

- `carryover_crossing_lags` deliberately uses one branch for every `j` — the shipped
  special-case at `j == 0` made half-life discontinuous and *non-monotone in alpha*
  (more carryover reported as a shorter half-life). Keep it branch-uniform.
- "Half-life" is the cumulative-50% crossing. Four different formulas shipped under that
  word (they give 6.58 / 2.19 / 7 / 8 at geometric `alpha=0.9`); this one is canonical
  because it means the same thing for humped (delayed/Weibull) kernels.
- `learning=False` and `autocorrelation={}` are explicit *skip* switches for cheap paths;
  both render as UNTESTED downstream. Passing `None` triggers best-effort computation
  (a `compute_parameter_learning` call, one `predict()` pass).
- `CarryoverKernel.alpha_mean` is `None` for Weibull, none, and legacy blends — for
  legacy the stored `adstock_<ch>` is a mixture weight, and surfacing it as a decay rate
  was the original bug.

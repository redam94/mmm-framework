# Declarative Estimands — the counterfactual causal lens

`mmm_framework.estimands` is a first-class, **named, serializable** estimand
subsystem. An `Estimand` is a counterfactual contrast

```
reduce_over( op( quantity | intervention,  quantity | baseline ) ) / normalizer
```

evaluated from a fitted posterior as **mean + HDI** (+ tail-probability
summaries). It is the single registry that subsumes the framework's previously
scattered estimand logic while keeping every existing number **bit-stable**, and
it is model-agnostic by design: a future non-MMM family (CFA / LCA / EFA)
declares its own estimands over its own quantities and gates them with
capability flags.

## Why a registry — the four notions that already existed

The framework computed four *distinct* numbers in four places. A naive
unification silently changes the dashboard, so the registry keeps them separate
and the equivalence tests (`tests/test_estimands.py`) pin each to the bit:

| Built-in | Legacy source | Notes |
|---|---|---|
| `contribution_roi` | `reporting/helpers/roi.py::compute_roi_with_uncertainty` | **Dashboard** decomposition ROI — reads the in-graph `channel_contributions` Deterministic; `point = mean(roi_samples)`; `az.hdi`; carries `prob_positive`/`prob_profitable`; drops `spend<=0` channels. **This is the UI number.** |
| `counterfactual_roi` | `analysis.py::MMMAnalyzer.compute_channel_roi` | Zero-out predict; `point = Σ(predict(obs)−predict(ch→0))` (difference of per-call means); percentile HDI on the contribution; **unpaired** seeds. A *different* number from the dashboard ROI in general. |
| `marginal_roas` | `model/base.py::compute_marginal_contributions` | +10% perturbation; `_hdi_finite`; **paired** seed (synthesized when none given). |
| `contribution` | `model/base.py::compute_counterfactual_contributions` | The counterfactual contribution total (no denominator). |

In-graph (likelihood-time) estimands — `ExperimentEstimand{CONTRIBUTION,ROAS,
MROAS}` — are the same algebra in PyTensor.

## The spec (`spec.py`, pure Pydantic, no numpy/pytensor)

- **`Intervention`** (discriminated on `type`): `Observed`, `ZeroInput(target)`,
  `ScaleInput(target,factor)`, `SetInput(target,value,sustained,carryover_state)`,
  `CustomIntervention(ref,params)`.
- **`Quantity`**: `Outcome`, `Contribution(target,source=counterfactual|in_graph_deterministic)`,
  `ObservedInput(target,source=raw|panel)`, `MarginalSpend(target,intervention_ref)`,
  `LatentVar(name)`, `Constant(value)`.
- **`Contrast`** = `{quantity, intervention, baseline, op, reduce, over, paired_seed}`.
- **`Estimand`** = `{name, kind, numerator, denominator, op_ratio_zero_denominator,
  window, hdi_prob, summaries, realization, required_capabilities, units,
  causal_assumptions, schema_version}`. The `window` lives at the estimand level
  so numerator and denominator share one time mask; the marginal denominator
  references the numerator's factor via `MarginalSpend(intervention_ref="numerator")`
  (never a second literal percentage — a desync guard).
- **`Realization`** carries the bit-stability knobs (see below).
- **`EstimandResult`** = `{name, status, mean, hdi_low, hdi_high, hdi_prob, units,
  extra, reason}`. `status="unsupported"` (with `reason`) is *returned*, never
  raised.
- **`SupportsEstimands`** is the `Protocol` marking the future non-MMM seam.

`spec.py` imports neither numpy nor pytensor, so it loads anywhere (host/kernel)
for serialization/validation — mirroring `garden/contract.py`.

## Two realization engines (they share only `spec.py`)

- **`evaluate.py` — post-hoc** (`EstimandEvaluator`): numpy; calls
  `model.predict_under(intervention, random_seed=…)` for per-draw arrays, masks
  the window, applies the per-estimand point/HDI/summary rules. Reuses the
  *exact* legacy sample-extraction helpers (`_get_contribution_samples`,
  `_extract_spend_from_model`, `_compute_hdi`, `_hdi_finite`,
  `compute_hdi_bounds`) so the decomposition path is bit-identical. Caches
  predictions by `(intervention, seed)` so the shared baseline is computed once
  (matching the legacy functions). Refuses off-panel (`IN_GRAPH_RESPONSE_CURVE`)
  specs — those are graph-only.
- **`graph.py` — in-graph** (`build_estimand_expr`): pytensor; the canonical home
  of the contribution/ROAS/mROAS algebra (`delta=(pert−base)*scale`).
  `calibration/likelihood.build_estimand_expr` now re-exports it.

The two never call into each other (numpy vs pytensor, paired-seed vs
deterministic, bool-mask vs int-index would fight).

### Bit-stability knobs (`Realization`)

- `point_rule`: `diff_of_means` (counterfactual/marginal — point from reduced
  per-call means, **not** `draws.mean()`) vs `mean_of_samples` (dashboard ROI).
- `hdi_method`: `percentile` (`compute_hdi_bounds`) / `az_hdi` (`_compute_hdi`) /
  `finite_percentile` (`_hdi_finite`, filters non-finite draws).
- Seeds: `paired_seed=True` synthesizes one shared seed (marginal); `False`
  passes the user seed through (unpaired counterfactual when seed is `None`).
- `op_ratio_zero_denominator`: `zero` (counterfactual/marginal → 0 point) /
  `skip` (dashboard → drop channel) / `nan`.
- Spend source: `ObservedInput(source="raw")` = `X_media_raw[window,idx].sum()`;
  `source="panel"` = `_extract_spend_from_model` (dashboard).
- `contribution_pct` is a cross-channel post-reduction step (needs the full
  channel set), not a per-estimand field.

## Capabilities + registry

`capabilities.model_capabilities(model)` returns a `set[str]` by cheap
duck-typing: `HAS_CHANNELS`, `HAS_CONTRIBUTIONS`,
`HAS_CONTRIBUTION_DETERMINISTIC`, `IN_GRAPH_RESPONSE_CURVE`, and parameterized
`HAS_LATENT:<name>`. `registry.defaults_for(capabilities)` returns the MMM
defaults (`contribution_roi`, `marginal_roas`, `contribution`) **filtered by
capability, not class name**, so a non-MMM model auto-drops incompatible
estimands. Two demonstrators prove generality: `awareness_lift` (mean lift, no
denominator) and `cost_per_conversion` (inverted ratio: spend / incremental
conversions).

Estimands with a wildcard `target="*"` expand to one result per channel, keyed
`"{name}:{channel}"`.

## Model surface & threading

On `BayesianMMM`:

- `predict_under(intervention, time_period=None, random_seed=None)` — thin
  wrapper over `predict(X_media=<transform of X_media_raw>)`.
- `model_capabilities() -> set[str]`.
- `declared_estimands: list[Estimand]` — set from a spec / a reloaded model.
- `evaluate_estimands(estimands=None, *, random_seed=None)` — uses
  `declared_estimands` if any, else a class-level `DEFAULT_ESTIMANDS`, else
  `registry.defaults_for(capabilities)`.

Wiring:

- `MMMResults.estimands` is best-effort populated at fit time **only when the
  model declares estimands** (a default fit pays no extra posterior-predictive
  passes; an evaluation failure never breaks a fit).
- `agents/fitting.build_model` parses `spec["estimands"]` → `declared_estimands`
  (same `from_dict`+try/except shape as the experiments block).
- `serialization.py` round-trips `declared_estimands` (with `schema_version`),
  mirroring the experiments round-trip.
- Agent op `compute_estimands` (`agents/model_ops.py`) is the registry-driven
  surface; `roi_metrics` is left untouched.
- The garden manifest can carry advisory `default_estimands`/`capabilities`
  (additive); a garden model declares defaults directly via a class-level
  `DEFAULT_ESTIMANDS`.

### Scoping note (what was deliberately *not* changed)

The four reference analysis functions (`compute_channel_roi`,
`compute_marginal_contributions`, `compute_counterfactual_contributions`,
`compute_roi_with_uncertainty`) and `roi_metrics` are **left as-is**. The engine
is built on their primitives and the equivalence tests prove bit-for-bit
reproduction, so a physical rewrite would be churn with real regression risk —
notably, the engine's capability gating for `contribution_roi` is *stricter* than
`compute_roi_with_uncertainty`'s `beta*media` fallback, so flipping `roi_metrics`
would regress models exposing only `beta_<channel>`. The registry is the forward
path; `compute_estimands` is the registry-driven agent surface.

## Deferred (the "then layer")

- `LikelihoodConfig` (binomial/Poisson/Beta + link).
- Per-family config schema with own defaults/validators.
- Capability-gating of `garden/contract.py` REQUIRED_ATTRS + a real non-MMM
  family.
- Frontend DAG adapter / estimand-declaration UI.

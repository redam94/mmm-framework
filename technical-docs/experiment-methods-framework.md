# Experiment-Method Framework + Geo Estimators — Build Spec (Phase 1)

> Status: implementation spec (grounded against the current codebase). This is the
> single source of truth for the new `planning/methods/` package, the first-class
> geo estimators (synthetic control, TBR, GBR, DiD-MMT), and their wiring into the
> existing design → power → simulation → economics → lifecycle loop.
>
> This is **Phase 1 of 5**. Companion specs: `experiment-methods-nongeo.md`
> (ghost ads + switchback), `experiment-net-economics.md` (net test-loss vs
> reallocation-gain figure), `ltv-clv-modeling.md` (LTV phases 4–5).

## 0. Motivation

Today the estimator zoo lives as loose module-level functions in
`planning/simulation.py` (`pooled_did_estimator`, `per_pair_did_estimator`,
`regadj_geo_estimator`, `national_onoff_estimator`), registered in two private
dicts (`_GEO_ESTIMATORS`, `_NATIONAL_ESTIMATORS`). There is no named-method
concept a user (or the agent, or the DesignStudio) can enumerate, no per-method
data-requirement gate, and no first-class synthetic control / TBR / GBR. This
phase introduces a **method registry** — mirroring the existing `estimands/`
and `garden/` registry patterns — so every experiment method declares four
capabilities uniformly:

1. **design** — how to build a runnable `Assignment` (treatment/control/schedule),
2. **estimate** — an `EstimatorResult` from a panel + assignment + window,
3. **power** — an analytic MDE model (calibrated + simulated on top by the
   existing A/A·A/B harness),
4. **data requirement** — what the method needs (≥N geos / national series /
   user-level counts), so `design_options` gates cleanly.

Ghost ads and switchback (Phase 2) plug into the *same* registry with a
`family="user"` / `family="switchback"` data requirement, so "the whole loop"
works across every method without special-casing.

## 1. Verified codebase facts that pin the design

| # | Fact | Source |
|---|---|---|
| M1 | Geo/national estimators share the contract `fn(panel: SimPanel, assignment: Assignment, window: Window) -> EstimatorResult`. Pure numpy/pandas; read only assignment-named rows. | `simulation.py:161-248` |
| M2 | `EstimatorResult(estimate, se, spend_delta, n_eff)`; `se=None` means "no analytic SE → null is bootstrap/permutation" and the A/A path supplies the critical value empirically. | `simulation.py:80-86, 341` |
| M3 | `Assignment(kind, treatment_geos, control_geos, pairs, schedule_mult, seed)` is **frozen** and built ONCE per family (`build_geo_assignment` `:254`), reused across estimators (SIM-6). | `simulation.py:67-78, 254` |
| M4 | `Window(pre_slice, test_slice, t_pre, t_test)`; A/A slides windows over untreated history, A/B injects a known lift on treated×test cells only. | `simulation.py:59-65, 341, 615` |
| M5 | `SimPanel(kpi_wide, spend_wide, kpi_national, spend_national, residuals, periods, geos)`; `residuals` = geo KPI residualized on trend+Fourier+spend (`residualize_geo_panel`). `kpi_wide` cols are geo names from the raw-csv pivot (a **different** name space than `mmm.geo_names`). | `simulation.py:119-145`, F6 in `experiment-economics.md` |
| M6 | `_GEO_ESTIMATORS`/`_NATIONAL_ESTIMATORS` dicts drive `methodology_leaderboard` (`:781`); registering a new geo estimator there makes A/A·A/B run it automatically. | `simulation.py:758-763` |
| M7 | `design.design_experiment(dataset_path, kpi, channel, design_key=None, **kwargs)` dispatches to `geo_lift_design` / `flighting_design`; `design_options(...)` reports which families the data supports (geo needs ≥4 geos). Pure pandas, no fitted model. | `design.py:846, 828, 348, 640` |
| M8 | `geo_lift_design(randomize=True)` = randomized geo lift; `randomize=False` = matched-market DiD (`design_key="matched_market_did"`). Returns `treatment_geos`/`control_geos`/`pairs`, `weekly_spend_delta=abs(...)` (unsigned), `intensity_pct` (signed). Placebo-calibrated SE via `_placebo_did`. | `design.py:348, 410, 450-456` |
| M9 | `matched_pairs(kpi_wide, n_pairs, spend_wide)` residualizes each geo on trend+yearly Fourier(2)+spend, matches on residual co-movement + covariate distance (blossom via networkx, greedy fallback). Balance table `_balance_table`. | `design.py:219, 191, 254` |
| M10 | MDE convention throughout: `MDE = MDE_FACTOR(=2.8) * SE`, `2.8 = z.975 + z.80`. | `design.py:35` |
| M11 | The design endpoint `POST /projects/{id}/experiment-design` calls `planning.design.design_experiment`/`design_options`; returns loosely-typed JSON (extra keys can't 422). Options endpoint `GET .../experiment-design/options`. | `main.py:3388, 3356` |
| M12 | `design_experiment_plan` (agent tool, `tools.py:5910`) writes `dashboard_data["experiment_design_plan"]`; enriches with model anchor + opportunity cost when a model is fitted. | `tools.py:5910` |
| M13 | Geo synthetic worlds with known causal truth live in `synth/dgp_geo.py` (panel scenarios) + `synth/mff.py` (scenario→MFF + JSON answer key). | CLAUDE.md, `synth/` |

## 2. The method registry — `planning/methods/`

New package. All numpy/pandas only (kernel-safe); no PyMC at import (TBR uses a
closed-form conjugate Bayesian regression, not MCMC — see §4).

```
planning/methods/
├── __init__.py            # registry, list_methods, get_method, method_for_design_key, re-exports
├── base.py                # MethodSpec, DataRequirement, MethodResult, register()
├── synthetic_control.py   # donor-pool convex-weight SCM + placebo permutation inference
├── tbr.py                 # time-based regression (Bayesian counterfactual, conjugate)
├── gbr.py                 # geo-based regression (cross-sectional post-on-pre)
└── did_mmt.py             # matched-market DiD (thin wrapper over the existing path)
```

### 2.1 `base.py` — the descriptor

```python
@dataclass(frozen=True)
class DataRequirement:
    family: str                    # 'geo' | 'national' | 'user' | 'switchback'
    min_geos: int = 0              # geo methods
    needs_panel: bool = False      # requires kpi_wide (geo × week)
    needs_pre_period: bool = True  # requires a clean pre-window
    min_pre_weeks: int = 8
    notes: str = ""

@dataclass(frozen=True)
class MethodSpec:
    key: str                       # 'synthetic_control' | 'tbr' | 'gbr' | 'did_mmt' | ...
    name: str                      # human label
    requirement: DataRequirement
    # design: build a runnable Assignment (+ a design dict for the UI).
    design_fn: Callable[..., dict]         # (dataset_path, kpi, channel, **kw) -> design dict
    # analysis: EstimatorResult on a SimPanel (geo/national families).
    estimator_fn: Callable | None          # (panel, assignment, window, **kw) -> EstimatorResult
    # analytic power: method-specific SE→MDE (before placebo/sim calibration).
    power_fn: Callable | None              # (design, **kw) -> {mde_roas, se, se_source}
    references: tuple[str, ...] = ()
    description: str = ""

_METHODS: dict[str, MethodSpec] = {}

def register(spec: MethodSpec) -> MethodSpec: ...
def get_method(key: str) -> MethodSpec: ...
def list_methods(*, family: str | None = None) -> list[MethodSpec]: ...
def methods_for_data(*, n_geos: int, n_weeks: int, has_user_counts: bool) -> list[MethodSpec]:
    """Filter by DataRequirement — the gate design_options consumes."""
```

`MethodResult` is a thin, serializable envelope the agent/endpoint return:
`{key, name, family, estimate, se, mde_roas, power, null_method, ...}` — all
scalars `float()`-cast, non-finite → `None` (the F12 loose-JSON convention).

### 2.2 `__init__.py` — registration

Import the geo methods and register them at module import (like
`estimands/registry.py` registers built-ins). Also **register their estimators
into `simulation._GEO_ESTIMATORS`** (M6) so `methodology_leaderboard` and the
existing A/A·A/B simulator pick them up with zero further wiring. Keep the four
legacy estimator functions as-is (back-compat).

**Decision: SCM and ridge are BOTH surfaced as distinct named methods.** The
registry exposes `synthetic_control` (convex weights + placebo inference, §3) and
`regadj_geo` (the existing unconstrained ridge) as *separate* `MethodSpec`s, and
BOTH stay in `_GEO_ESTIMATORS` so the leaderboard runs and compares them
side-by-side. `did_mmt` → the pooled/per-pair DiD; `tbr` → the fast Kalman path
(§4); `gbr` → the cross-sectional estimator (§5). So the geo method set is:
`{synthetic_control, regadj_geo, tbr, gbr, did_mmt}` (+ `pooled_did` retained).

## 3. Synthetic control — `synthetic_control.py`

The current `regadj_geo_estimator` (`simulation.py:193`) is an **unconstrained
ridge** with an intercept — it can extrapolate outside the donor convex hull and
has no placebo inference. A real SCM (Abadie–Diamond–Hainmueller) is:

**Weights.** Solve for convex donor weights `w ≥ 0, Σw = 1` minimizing pre-period
fit `‖y_pre_treated − X_pre_donor w‖²`. Implement as projected-gradient / a small
simplex-constrained least squares (numpy only; SLSQP from scipy is already a dep
via the planner — reuse it). Optionally match on pre-period covariates too, but
KPI-path matching is the MVP.

**Effect.** Counterfactual `ŷ_test = X_test_donor w`; per-week gap
`y_test − ŷ_test`; total incremental KPI = `Σ_test gap`. `spend_delta` from the
design.

**Inference — placebo permutation (the piece that's missing).** Re-run the whole
SCM treating each *donor* geo as a fake treated unit (its own donor pool = the
rest). The distribution of placebo total-gaps is the null. Report:
- a permutation p-value `= (1 + #{|placebo| ≥ |actual|}) / (1 + n_placebo)`,
- a null SD used as the `se` surrogate, but return `EstimatorResult.se=None` so
  the A/A path treats the null as empirical (M2) — and additionally attach the
  permutation p-value in the `MethodResult`.
- Optionally filter placebos with poor pre-period fit (RMSPE ratio), the standard
  ADH robustness screen.

```python
def synthetic_control_estimator(panel, assignment, window, *, ridge=0.0) -> EstimatorResult
def synthetic_control_weights(x_pre_donors, y_pre_treated) -> np.ndarray  # convex
def placebo_distribution(panel, assignment, window) -> np.ndarray         # per-donor gaps
```

Register with `power_fn` = placebo-calibrated MDE (reuse the A/A machinery in
`simulation.py`; a `null_method="placebo_permutation"` path).

## 4. Time-based regression (TBR) — `tbr.py`

Google's TBR (Kerman et al. 2017) is a **Bayesian regression of the test-region
time series on the control-region time series**, fit on the *pre* period, then
used to *project the counterfactual* through the test period; the causal effect
is the cumulative `Σ_test (observed − predicted)` with a full posterior.

**Decision: full Bayesian structural time series (BSTS), CausalImpact-style.**
The counterfactual is a state-space model of the treated series regressed on the
control series with a **local level (+ optional local linear trend and weekly
seasonality)** state component, so trending/seasonal control series are handled
properly (not just a static linear fit). Fit via PyMC.

**Two-tier engine (this is the load-bearing design nuance — a full MCMC BSTS is
too slow to run thousands of times inside the A/A·A/B sliding-window loop):**

1. **Headline analysis** (the *actual* experiment — one call): full BSTS via
   PyMC. Local-level state + control regressors + optional trend/seasonality;
   NUTS (or a fast `method` — MAP/ADVI/Laplace, reusing `run_approximate_fit`).
   Fit on the pre-period, forecast the counterfactual through the test window,
   cumulative effect posterior = `Σ_test (observed − predicted_draws)` → point =
   posterior mean, `se` = posterior SD, plus a full pointwise + cumulative
   credible band. This is what the DesignStudio/report show.
2. **Power/simulation** (the A/A·A/B harness — many calls): a fast
   **Kalman-filter / conjugate approximation** of the same state-space model
   (closed-form Gaussian filtering, numpy only — no MCMC per window), so
   `run_aa_simulation`/`run_ab_simulation` stay tractable. The fast path is the
   registered `estimator_fn` (M1 contract, `se` set); the full BSTS is invoked
   only for the single headline readout via a separate `tbr_causal_impact(...)`
   entry point.

Because PyMC is imported, `tbr.py` uses a **lazy import** (like
`mmm_extensions`): the numpy Kalman path has no PyMC dependency, and
`import planning.methods` stays PyMC-free; only `tbr_causal_impact` pulls PyMC in.

```python
# fast Gaussian-filter path — the registered estimator (kernel-safe, numpy)
def tbr_estimator(panel, assignment, window, *, n_draws=400, seed=0) -> EstimatorResult
def tbr_kalman_counterfactual(x_pre, y_pre, x_test, *, seed) -> dict
    # local-level + regression via closed-form filtering; {pred_mean, pred_sd, cum_sd}
# full BSTS — headline readout only (lazy PyMC import)
def tbr_causal_impact(x_pre, y_pre, x_test, y_test=None, *, method="nuts",
                      trend=False, seasonality=None, draws=1000) -> dict
    # {pred_mean, pred_band, cumulative_effect, cumulative_band, posterior}
```

`se` from the fast path is set (not None) so the A/A path can still calibrate it.
Reference: matches the existing `docs/blog-geo-experiments-tbr.html` narrative and
the CausalImpact (Brodersen et al. 2015) BSTS formulation.

## 5. Geo-based regression (GBR) — `gbr.py`

Google's GBR is **cross-sectional**: one row per geo, regress the test-period
response on the pre-period response and the treatment's spend change; the
incremental effect is the coefficient on the treatment spend delta (an
iROAS estimate). Classic `GeoexperimentsResearch` GBR.

- Row per geo `g`: `y_post_g = α + β·y_pre_g + θ·Δspend_g + ε_g`, weighted least
  squares (weights ∝ geo size to stabilize heteroskedasticity). `Δspend_g` is the
  design's per-geo spend change (treatment geos non-zero, controls zero).
- `θ` is the incremental-response-per-dollar; total incremental KPI =
  `θ · Σ_treated Δspend`. `se` = the WLS SE of `θ` scaled to total-KPI units.

```python
def gbr_estimator(panel, assignment, window, *, weighted=True) -> EstimatorResult
```

Register `power_fn` from the WLS SE of `θ`.

## 6. DiD-MMT — `did_mmt.py`

Matched-market DiD already exists (`geo_lift_design(randomize=False)`,
`design_key="matched_market_did"`, pooled/per-pair DiD estimators). This module
is a **thin wrapper** that (a) registers it as a named `MethodSpec` with the
`geo` requirement and (b) points `estimator_fn` at `per_pair_did_estimator`
(cluster-robust) as the default MMT analysis. No new math — this closes the named
set the client asked for and gives the UI a consistent method card.

## 7. Design + power wiring (reuse, don't rebuild)

The existing pipeline already does the heavy lifting; the registry threads
methods through it:

- **`design_options`** (`design.py:828`) → call `methods.methods_for_data(...)`
  and return the enumerated methods (key, name, requirement, supported?). The
  endpoint `GET .../experiment-design/options` (M11) then lists real named
  methods with data-gating reasons instead of coarse `design_key` strings.
- **`design_experiment`** (`design.py:846`) → accept `method=<key>`; when a
  method supplies its own `design_fn`, use it; otherwise fall back to the current
  `geo_lift_design`/`flighting_design` dispatch. The returned design dict gains
  `method`, `method_name`, `references`.
- **Power** → each design carries the method's analytic MDE (`power_fn`), and the
  existing `methodology_leaderboard`/`run_aa_simulation`/`run_ab_simulation`
  (M6) automatically produce the placebo-calibrated + empirical MDE for every
  registered estimator (SCM, TBR, GBR, DiD-MMT) on the same fixed assignment.

## 8. Model anchor + economics (already there — just carries the method through)

No change to `design_anchor.py` / `opportunity_cost.py` math — the model anchor
perturbs treated×window rows independent of the estimator, and opportunity cost
is design-driven. The `method` key just rides along on the design dict so the
`experiment_economics` op (`model_ops.py:2051`) and the DesignStudio panels label
which method the numbers belong to. Phase 3 (`experiment-net-economics.md`) adds
the single net figure on top.

## 9. Synthetic ground truth — `synth/dgp_geo.py`

Add a geo-experiment DGP with a **known injected lift** in a known test window on
known treatment geos (extend the existing panel scenarios). Emits the panel + a
JSON answer key (true incremental KPI, true iROAS). Recovery tests assert each
estimator recovers the planted effect within its CI and that SCM's placebo
p-value is small on a real effect / ~uniform on a null.

## 10. Wiring summary

- **`planning/methods/`** — new package (§2–§6).
- **`planning/simulation.py`** — register new estimators into `_GEO_ESTIMATORS`;
  add `null_method="placebo_permutation"` support to the A/A path for SCM.
- **`planning/design.py`** — `design_options`/`design_experiment` accept + return
  `method`; enumerate via the registry.
- **`agents/tools.py`** — `design_experiment_plan` surfaces the chosen method +
  the available-methods list; new lightweight `list_experiment_methods` tool.
- **`agents/model_ops.py`** — `experiment_economics` design dict carries `method`.
- **`src/mmm_framework/api/main.py`** — `GET .../experiment-design/options` returns named methods;
  `POST .../experiment-design` accepts `method`. Loose JSON (M11) → no schema
  break.
- **Frontend** — `DesignStudio.tsx`: a method picker (with data-gating
  disabled-reasons) feeding the existing design/power/simulation panels;
  method label on the result cards. `ParetoTable`/`MethodologyPanel` already
  iterate estimators — they now show the named methods.
- **`synth/dgp_geo.py`** — known-lift geo DGP + answer key (§9).
- **Docs** — the existing `blog-synthetic-control.html`, `blog-geo-experiments-tbr.html`,
  `blog-staggered-did.html` become linkable from the method cards (`references`).

## 11. Build order

1. `methods/base.py` (registry + descriptors) — the contract everything imports.
2. `methods/synthetic_control.py` (convex weights + placebo inference) — the
   highest-value new estimator; verify recovery on §9 DGP.
3. `methods/tbr.py` (conjugate Bayesian counterfactual) — verify counterfactual
   projection + cumulative CI on the DGP.
4. `methods/gbr.py` + `methods/did_mmt.py`.
5. `methods/__init__.py` registration + `simulation._GEO_ESTIMATORS` wiring.
6. `design.py` `design_options`/`design_experiment` method threading.
7. `synth/dgp_geo.py` known-lift DGP + recovery tests
   (`tests/test_experiment_methods.py`).
8. Agent tool + endpoint + DesignStudio method picker.
9. Docs cross-links.

## 12. Tests — `tests/test_experiment_methods.py`

- Registry: every registered method round-trips through `get_method`/`list_methods`;
  `methods_for_data` gates correctly (e.g. SCM needs ≥ a donor pool; ghost ads
  hidden without user counts once Phase 2 lands).
- SCM: convex weights sum to 1, non-negative; recovers planted lift on the §9 DGP
  within CI; placebo p-value small on effect, ~uniform under null; refuses to
  extrapolate (weights stay in the donor hull).
- TBR: counterfactual matches the pre-period fit; cumulative-effect CI covers the
  planted effect; degrades gracefully with <`min_pre_weeks`.
- GBR: `θ` recovers planted iROAS; WLS SE finite.
- DiD-MMT: wrapper returns the same numbers as the direct `per_pair_did_estimator`
  path (no drift).
- Leaderboard: all four methods run on one fixed assignment; A/A FPR + empirical
  MDE produced for each; SCM uses the permutation null.

# Extension-model priors + trend / seasonality / likelihood

*Shipped 2026-07-08.* Makes the DAG-routed extension models
(`NestedMMM` / `MultivariateMMM` / `CombinedMMM`) honor the spec's baseline
dynamics and outcome likelihood, and makes their mediator / outcome /
cross-effect priors **settable and inspectable** through the spec — closing the
gap where an extension model silently ignored everything the user configured
except the causal-DAG structure.

## Why

Before this change the extension models were heavily hard-coded:

- `DAGModelBuilder.build()` forwarded `model_config` / `trend_config` only to the
  plain `BayesianMMM` branch; the three extension branches dropped them.
- No extension model built a trend or seasonality term, so a real drift or
  seasonal pattern leaked into the media coefficients.
- The outcome likelihood was hard-coded (`Normal` for Nested, `MvNormal`+LKJ for
  MV/Combined) — `spec["likelihood"]` was never consulted.
- Several config fields were *populated but read nowhere*: `MediatorConfig.
  direct_effect`, `OutcomeConfig.intercept_prior_sigma` / `.media_effect`,
  `include_trend` / `include_seasonality`.
- The consumed-paths registry rejected **all** `priors.*` for an extension spec,
  so a user had no spec-level way to set an extension prior at all.

## Design

### 1. Threading the core config in

`BaseExtendedMMM.__init__` gained `model_config` and `trend_config` parameters
(default `None`). When both are `None` the graph is **byte-identical** to before,
so the array-level construction path and its whole test suite are unchanged. The
subclass ctors forward them to `super().__init__`; `DAGModelBuilder.build()`
passes `model_config=model_config, trend_config=self._trend_config` to all three
extension branches. `agents/fitting._build_extension_model` builds the SAME
`ModelConfig` the plain path builds — the config construction was extracted into
the shared `_model_config_from_spec(spec)` helper — and attaches it with
`.with_model_config(...)`.

### 2. Trend + seasonality (`mmm_extensions/components/temporal.py`)

Both are additive terms on the **standardized-outcome scale** (the scale the
extension likelihoods fit on):

- `build_trend_contribution` — the **full core trend family**: `linear`
  (`trend_slope`), `piecewise` (Prophet changepoints — `trend_k`/`trend_delta`/
  `trend_m`), `spline` (random-walk-smoothed B-spline — `spline_coef_raw`/
  `spline_scale`), and `gaussian_process` (HSGP with a Matérn-3/2 kernel —
  `gp_lengthscale`/`gp_amplitude`/`trend_gp`). It reuses the SAME basis builders
  (`create_bspline_basis`, `create_piecewise_trend_matrix`) and `pymc.gp.HSGP`
  the core `BayesianMMM` uses, and the `TrendConfig` priors, so an extension
  trend means the same thing as a plain MMM's. Every **non-linear** family is
  **zero-centered** (`trend - trend.mean()`) so its level is absorbed by the
  model intercept rather than fighting it (the identifiability guard); GP warns
  that its lengthscale is weakly identified under MAP → prefer NUTS. (The
  `model/components/trend.py::TrendBuilder` strategy classes are vestigial dead
  code with an incompatible config shape; the live path this mirrors is the
  inline `model/base.py::_build_trend_component`.)
- `build_seasonality_contribution` — a Fourier design matrix (reusing the core
  `create_fourier_features`) times a coefficient vector
  (`seasonality_coefs ~ Normal(0, prior_sigma)`). The period is derived from the
  datetime-index spacing (weekly data → yearly period ≈ 52), Nyquist-clamped, and
  honors `SeasonalityConfig.prior_sigma`. A non-datetime index → skipped with a
  warning.

Both return `None` when unconfigured; the design matrices are data-fixed
constants so the RVs enter `mu` and are **never dead** (the Pathfinder
`_anchored_det` invariant is unaffected). `NestedMMM` registers
`trend_component` / `seasonality_component` Deterministics in original units.
MV/Combined build per-outcome (or shared, per `share_trend` /
`share_seasonality`) terms via `BaseExtendedMMM._build_baseline_dynamics`.

### 3. Outcome likelihood (`mmm_extensions/components/outcome.py`)

`build_outcome_likelihood` dispatches on `model_config.likelihood.family`:
`normal` (byte-identical — same `sigma_y`), `student_t` (adds `nu_y ~
Gamma(2, 0.1)`). Other families raise. **Only NestedMMM** uses it — MV/Combined
keep their joint `MvNormal`+LKJ, because a per-outcome non-Gaussian family cannot
join the cross-outcome correlation matrix (a genuine modeling constraint).

### 4. Spec-settable priors

Spec keys (under the normal `priors.*` namespace):

| Path | Keys |
|------|------|
| `priors.mediator.<name>` | `media_effect_sigma`, `media_effect_constraint` (`none`/`positive`/`negative`), `outcome_effect_sigma`, `direct_effect_sigma`, `observation_noise_sigma`, `allow_direct_effect` |
| `priors.outcome.<name>` | `intercept_prior_sigma`, `media_effect_sigma`, `include_trend`, `include_seasonality` |
| `priors.cross_effect.<src>__<tgt>` | `effect_type`, `prior_sigma` |

`agents/fitting._inject_extension_priors(dag_dict, spec)` folds these into the
DAG node `config` dicts (mediator/outcome nodes) and edge metadata (cross-effect)
**before** `dag_to_*_config` translates the DAG — so an override becomes a
`MediatorNodeConfig` / `OutcomeNodeConfig` field, which the translator already
maps into the graph-consumed `MediatorConfig` / `OutcomeConfig`. The injection
deepcopies (never mutates the caller's DAG), and writes only registry-validated
keys (a stray key would crash the `extra='forbid'` node config). Because
`prior_predictive_check` builds the *real* extended model, the plausibility check
reflects the settings automatically.

### 5. Dead-wire fixes

Making the model *read* config it already received:

- NestedMMM `delta_direct_<ch>` now reads `MediatorConfig.direct_effect` (was a
  literal `Normal(0, 0.5)` — the default is byte-identical).
- MV/Combined intercept + `beta_media`/`beta_direct` read per-outcome
  `intercept_prior_sigma` / `media_effect.sigma` as a vector sigma (defaults
  2.0 / 0.5 → byte-identical).
- Combined `gamma` reads per-mediator `outcome_effect.sigma` — the one
  **intended** default change (0.5 → 1.0, matching NestedMMM).

### 6. Registry (DAG-aware)

`unconsumed_prior_path` for an extension spec ACCEPTS
`priors.mediator/outcome/cross_effect.<name>.<validkey>` and `priors.seasonality.*`,
and REJECTS the plain-model groups (`priors.media/controls/media_default/
intercept`) and unknown keys with a clear message — so `update_model_setting`
tells the user what will and won't apply. Plain-model validation is unchanged;
`unconsumed_spec_path` still rejects `media_prior_mode` for extensions.

## Scope

`NestedMMM` (the model the agent routes to for a mediator DAG) is fully wired:
trend + seasonality + likelihood family + all mediator/direct/intercept priors,
spec-settable, FE-editable, reflected by the prior-predictive check. MV/Combined
gain trend/seasonality + live per-outcome priors but keep the MvNormal-LKJ
likelihood.

## Tests

- `tests/mmm_extensions/test_extension_priors_wiring.py` — byte-compat baseline,
  trend/seasonality/likelihood RVs, spec-settable priors scale the RVs (prior
  draws), the DAG-node injection, and the registry gate.
- `tests/test_extension_fit_path.py::test_spec_trend_seasonality_likelihood_and_priors_reach_nested_graph`
  — the spec → `build_model` → graph integration path.

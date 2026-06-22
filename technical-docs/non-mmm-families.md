# Non-MMM model families (CFA / LCA / EFA …)

The final axis of the modeling-system rework. After estimands ([estimands.md](./estimands.md))
and per-model config + likelihood ([custom-model-config.md](./custom-model-config.md)) made the
spine pluggable, this removes the last "everything is an MMM" assumption: a genuinely **non-MMM
Bayesian family** can be authored as a Model Garden model and ride the existing build → fit →
serialize → estimand → report pipeline instead of forking it. The first concrete family is a
**Bayesian confirmatory factor analysis (CFA)**; `BayesianMMM` is unchanged (byte-identical).

## The contract: `__garden_model_kind__` + `is_mmm_model`

A model declares its family with the class attribute **`__garden_model_kind__`** (default `"mmm"`).
A non-MMM family sets it to its own kind (e.g. `"cfa"`, `"latent_class"`). Two helpers in
`garden/contract.py` drive the gating:

- `model_kind(obj)` — the declared kind, else `"mmm"` for a `BayesianMMM` subclass, else `"unknown"`.
- `is_mmm_model(obj)` — **True unless the model explicitly declares a non-`"mmm"` kind.** A
  duck-typed / unknown model is treated as MMM (the historical default), so only a *declared*
  non-MMM family opts out.

The MMM-specific gates apply only when `is_mmm_model` is true:

| Gate | Location | MMM-only check that a non-MMM family skips |
|---|---|---|
| `validate_class` | `garden/contract.py` | the `predict` + `sample_channel_contributions` read surface (a declared non-MMM class needs only `fit()`) |
| `validate_instance` | `garden/contract.py` | `channel_names` non-empty + `_media_raw_max` (split: `REQUIRED_ATTRS_BASE` always, `REQUIRED_ATTRS_MMM` only for MMM) |
| `validate_fitted` | `garden/contract.py` | the `beta_<channel>` posterior convention (the `chain`/`draw` trace check still runs) |
| compat tiers | `garden/compat.py` | `scaling`, `ops_smoke`, `accuracy` self-skip; `static`/`build`/`fit`/`instance`/`trace` still run |
| serializer | `serialization.py` | channel/control panel-compat match + the y/media re-standardization on load (records `model_kind` in metadata) |

The manifest gains an advisory `model_kind` (AST-detected from the source by
`garden_registry.static_model_kind`). The channel read-ops (`roi_metrics`, `adstock_weights`, …)
already degrade to empty for `channel_names == []`.

## Estimands for non-MMM families — bare `LatentVar`

The estimand engine is model-agnostic: `model_capabilities` auto-exposes every posterior variable
as `HAS_LATENT:<var>`. The one extension this phase adds is realizing a **bare `LatentVar`
quantity** (`estimands/evaluate.py::_latent_quantity`): read `posterior[name]`, collapse chain×draw
to the sample axis, and

- per-draw **scalar** (a fit index `cfi`/`srmr`, a named scalar loading, a class size) → mean + HDI;
- **obs-indexed** latent → mean over the window;
- **vector/matrix** latent (e.g. a full loadings matrix) → `status="unsupported"` (surfaced as a
  table instead).

`registry.latent_scalar(name)` / `fit_index(name)` / `factor_loading(name)` build these estimands,
gated by `HAS_LATENT:<var>`, so they cleanly return `unsupported` on a model that doesn't carry the
variable. A latent used in a **contrast** (intervention vs baseline) is realized by
`_eval_latent_contrast` via the model's `sample_latent_under(var, intervention)` (e.g. an MMM's
goodwill stock under media-on vs a channel off) — see [estimands.md](./estimands.md).

## Authoring a non-MMM family (the CFA example)

`examples/garden_models/bayesian_cfa.py` is the worked example. The pattern:

1. **Subclass `CustomMMM`**, set `__garden_model_kind__ = "<kind>"`, and a `CONFIG_SCHEMA`
   (bespoke params with defaults — the CFA's `n_factors`, `factor_assignment`, priors).
2. **Override `_prepare_data`** to assemble your observed data and set the model-agnostic
   attributes the inherited `fit`/serializer read: `channel_names = []`, `n_obs`, `y_mean=0`,
   `y_std=1`, `_media_raw_max={}`, `_media_max={}`, `X_controls_raw=None`, `time_idx`,
   `n_periods`, `trend_features={}`, `seasonality_features={}`, `has_geo`/`has_product`.
3. **Override `_build_model`** with your graph, registering per-draw deterministics for your
   quantities of interest (fit indices, named loadings, a loadings matrix). The CFA uses the
   **marginal** multivariate-normal likelihood (factor scores integrated out): `yᵢ ~ MvNormal(0,
   ΛΛᵀ + Ψ)`, with positive (HalfNormal) loadings on a fixed simple-structure pattern for
   identification, and per-draw `srmr` / `cov_fit` from the implied vs observed covariance (no
   `pytensor.scan`).
4. **Declare `DEFAULT_ESTIMANDS`** (e.g. `[fit_index("srmr"), fit_index("cov_fit")]`); add a
   `factor_loadings_summary()` for the matrix-valued table. `fit` auto-attaches declared estimands.

**Data path:** the CFA reuses `PanelDataset` — list the indicators as the spec's `kpi` +
`media_channels` (the MFF requires a kpi + ≥1 "media" column); `_prepare_data` reads every observed
column (kpi + media + controls) as an indicator uniformly, in that order (so `factor_assignment`
follows it). No new data container.

### Verified end-to-end

`tests/test_cfa_garden_model.py` fits the CFA (MAP) on synthetic data with a planted 2-factor
structure and asserts it **recovers the loadings** (≈ the planted value), the `srmr`/`cov_fit`
**estimands evaluate** via the bare-`LatentVar` engine, it **serializes/round-trips**, and it
**passes the capability-gated compat suite** (scaling/ops_smoke skipped; static/build/fit/instance/
trace pass). `tests/test_non_mmm_families.py` covers the contract/compat gating + the estimand
extension in isolation.

## Shipped follow-ons

- **Latent-variable contrasts** — a `LatentVar` in a counterfactual `Contrast` is realized via
  `BayesianMMM.sample_latent_under(var, intervention)` + `evaluate.py::_eval_latent_contrast` (e.g.
  an MMM's goodwill stock under media-on vs a channel off). `tests/test_latent_contrasts.py`.
- **CFA HTML report section** — `FactorAnalysisSection` (loadings table + fit-index cards) +
  `FactorAnalysisExtractor`; `MMMReportGenerator` gates the channel/ROI sections off and the
  factor-analysis section on via `bundle.model_kind`. `tests/test_cfa_report.py`.

## Out of scope (deferred)

- **LCA** (label-switching ordering constraint) and **EFA** (rotation indeterminacy) — CFA proves
  the path first; both are now slot-ins on the same contract.
- A dedicated non-MMM data container (the CFA reuses `PanelDataset`).

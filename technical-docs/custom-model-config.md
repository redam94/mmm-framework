# Per-model configuration + pluggable likelihood

This is the second axis of the modeling-system rework (the first was the
declarative [Estimand subsystem](./estimands.md)). It lets a model — especially a
bespoke **Model Garden** model — declare **its own configuration fields with
defaults** and choose a **non-default observation likelihood**, so a model like a
binomial *awareness* model (KPI = aware-count out of a survey, needing a
`number_of_trials`) is fully spec-driven instead of hard-coding tuning as
invisible class attributes.

It is built on four independent axes; this phase delivers **config schema** and
**likelihood**, designed so the others (capability-gated non-MMM families) slot in
later. `BayesianMMM` with the default config is **byte-identical** to before.

## 1. Per-model config schema (`CONFIG_SCHEMA` + `model_params`)

A model declares a class attribute `CONFIG_SCHEMA` — a `pydantic.BaseModel`
subclass — giving it **settable, defaulted, validated** parameters:

```python
from pydantic import BaseModel, Field
from mmm_framework.garden import CustomMMM

class AwarenessParams(BaseModel):
    number_of_trials: int = Field(default=1000, gt=0)
    retention_prior_alpha: float = Field(default=6.0, gt=0)
    model_config = {"extra": "forbid"}

class AwarenessMMM(CustomMMM):
    CONFIG_SCHEMA = AwarenessParams
    def _build_model(self):
        n = self.model_params.number_of_trials   # validated + defaulted
        ...
```

- **Carrier:** the optional keyword constructor arg `model_params`
  (`BaseModel | dict | None`) on `BayesianMMM.__init__`, stored as
  `self.model_params`. The 3-positional-arg garden call
  (`resolved_cls(panel, model_config, trend_config)`) is unchanged — the garden
  contract permits trailing kwargs.
- **Coercion** (`_coerce_model_params`): with a schema declared, a dict / other
  `BaseModel` / `None` is validated *through* it so defaults + validators always
  apply (`self.model_params.<field>` is always present and typed). With no schema
  (the base model), the value is passed through untouched.
- **Spec threading:** `agents.fitting.build_model` validates
  `spec["model_params"]` against the resolved class's `CONFIG_SCHEMA` (clear,
  build-context error on failure) and hands it to the constructor.
- **Serialization:** `MMMSerializer` round-trips `model_params` as a plain dict
  (the reloaded model re-validates against the — possibly evolved — schema), with
  a `model_params_schema_version` so drift is visible.
- **UI:** the garden manifest can carry a `config_schema` (the schema's
  `model_json_schema()`) so the frontend renders a dynamic params form, the same
  way `dataset_schema` is rendered.

## 2. Pluggable likelihood (`LikelihoodConfig`)

`ModelConfig` gains a declared field `likelihood: LikelihoodConfig`
(`{family, link, params}`), defaulting to `normal` / `identity`:

```python
from mmm_framework.config import ModelConfig, LikelihoodConfig
ModelConfig(likelihood=LikelihoodConfig.student_t(nu=5))
ModelConfig(likelihood=LikelihoodConfig.binomial(n_trials=1000))
```

`LikelihoodFamily`: `normal`, `student_t`, `lognormal`, `binomial`,
`beta_binomial`, `poisson`, `negative_binomial`, `beta`. `LinkFunction`:
`identity`, `logit`, `log` (canonical link resolved per family; an incoherent
link is rejected).

- **Conditional standardization** (`_prepare_data`): Gaussian-scale families
  (`normal`/`student_t`/`lognormal`) z-score `y` as before; count/bounded families
  keep the **natural scale** (`y_mean=0, y_std=1`, so `y_obs_scaled` and every
  downstream `* y_std` bridge stay identity no-ops).
- **Built-in additive dispatch** (`_build_likelihood`): fits only the Gaussian
  families — `normal` (byte-identical to the old hard-coded `pm.Normal`) and
  `student_t`. **Non-Gaussian families raise a clear `NotImplementedError`** on
  the built-in additive model: its component priors are calibrated for
  standardized-Normal `y` on an identity link, so recalibrating them for a logit
  or log link is a research task, not a config switch.
- **Non-Gaussian belongs to models that own their observation block.** A garden
  model that overrides `_build_model` reads `self.model_config.likelihood` and
  writes the likelihood itself. The family still drives the standardization
  decision upstream; the bespoke count (e.g. `number_of_trials`) lives in
  `CONFIG_SCHEMA`, so the two compose without redundancy (`n_trials` is **not**
  required in `LikelihoodConfig.params`).

### Worked example — the awareness garden model

`examples/garden_models/awareness_structural_mmm.py` demonstrates both axes
together. Its `AwarenessParams` `CONFIG_SCHEMA` carries `number_of_trials`,
`retention_prior_alpha/beta`, `level_innovation_sigma`. With the default Normal
likelihood the KPI is a continuous awareness *index*; with
`spec["likelihood"] = {"family": "binomial"}` the model writes its own
`pm.Binomial(n=number_of_trials, p=sigmoid(awareness_state), observed=y)` and the
KPI is a survey *count*. It declares `DEFAULT_ESTIMANDS = ["awareness_lift",
"contribution_roi"]` (built-in names resolved via the estimand registry), so the
Phase-1 `awareness_lift` estimand is reported on top.

## 3. Authoring a non-default-likelihood garden model

1. Subclass `CustomMMM`; set `CONFIG_SCHEMA` for bespoke params.
2. In the spec, set `model_params` (your fields) and `likelihood`
   (`{"family": ...}`).
3. In `_build_model`, read `self.model_params.<field>` and
   `self.model_config.likelihood`; for a non-Gaussian family, write the
   observation node yourself (`y` is the natural-scale value when the family
   doesn't standardize). Register the usual deterministics (`channel_contributions`,
   `y_obs_scaled`, …) so the read-ops + estimands still resolve.

## 4. What's intentionally deferred

- Prior **recalibration** for non-Gaussian families on the *built-in additive*
  model (gated by the `NotImplementedError`).
- `BaseModelConfig` discriminated-subclass refactor (this `CONFIG_SCHEMA` +
  `model_params` route is the lighter, garden-first step).
- Capability-**gating** of the garden contract's MMM-specific required attributes
  + a real non-MMM family (CFA/LCA/EFA). The capability machinery exists
  (`estimands/capabilities.py`); promoting a non-MMM family is the next axis.

## Key files

- `src/mmm_framework/config/likelihood.py`, `config/enums.py`
  (`LikelihoodFamily`/`LinkFunction`), `config/model.py` (`ModelConfig.likelihood`).
- `src/mmm_framework/model/base.py`: `CONFIG_SCHEMA`, `model_params` ctor arg +
  `_coerce_model_params`, `_standardizes_y` / `_likelihood_config`,
  `_build_likelihood`, `_resolve_estimand`.
- `src/mmm_framework/builders/model.py`: `with_likelihood`.
- `src/mmm_framework/agents/fitting.py`: `spec["likelihood"]` + `spec["model_params"]`.
- `src/mmm_framework/serialization.py`: `model_params` round-trip.
- `src/mmm_framework/agents/garden_registry.py`: manifest `config_schema`.
- `examples/garden_models/awareness_structural_mmm.py`: the worked example.
- Tests: `tests/test_likelihood_config.py`, `test_model_likelihood_dispatch.py`,
  `test_model_params.py`, `test_awareness_garden_model.py`.

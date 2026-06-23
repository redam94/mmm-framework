# Flexible role-tagged `Dataset`

## Why

The model layer is extensible along four axes — `CONFIG_SCHEMA` + `model_params`,
`LikelihoodConfig`, capability-gated estimands, and `__garden_model_kind__`. The
**data** layer was not. The canonical transport `PanelDataset`
(`data_loader.py`) is shaped around the MMM roles `y` / `X_media` / `X_controls`
with an `MFFConfig` that forces every column into kpi / media / control. A
genuinely non-MMM family that wants to treat every measured column as a manifest
**indicator** (CFA, LCA) had to *abuse* the panel — concatenate `y + X_media +
X_controls` back into one matrix and hand-set ~15 contract attributes.

`Dataset` fixes this with an **additive** abstraction that mirrors the model-config
extensibility pattern: a column carries a `DatasetRole`, a model declares the data
it needs the way it declares `model_params`, and the MMM read-surface is preserved
so nothing else changes.

## Pieces

| Piece | File | Role |
|-------|------|------|
| `DatasetRole` | `config/roles.py` | Role enum — superset of `VariableRole`: `TARGET, PREDICTOR, CONTROL, INDICATOR, GROUP, TIME, OFFSET, WEIGHT, TRIALS, AUXILIARY`. `MFF_ROLE_TO_DATASET`/`DATASET_ROLE_TO_MFF` bridge the four shared roles losslessly. `OBSERVED_ROLES = (TARGET, PREDICTOR, CONTROL, INDICATOR)`. |
| `RoleBinding`, `DatasetSchema` | `config/dataset.py` | Pure-Pydantic (`extra="forbid"`, `schema_version`). `DatasetSchema` = the role mapping that generalizes `MFFConfig`. Role views (`target_names`, `indicator_names`, …) + lossless `from_mff(MFFConfig)` / `to_mff()`. |
| `Dataset` | `dataset.py` | Container: one tidy role-tagged `table` + `schema` + reused `PanelCoordinates`. Generic accessors `columns_for` / `frame_for` / `matrix` / `observed()`, **and** the MMM views `.y` / `.X_media` / `.X_controls` / `.coords` so it is drop-in where a panel is *read*. Adapters `from_panel` / `as_panel`; `PanelDataset.as_dataset()` is the reverse entry. |

## Model-side declaration (mirrors `CONFIG_SCHEMA`)

`BayesianMMM` gains three class attrs + a coercion helper (`model/base.py`):

- `DATASET_SCHEMA: type[DatasetSchema] | None = None` — a family's data schema (None ⇒ default MMM roles).
- `REQUIRED_ROLES: tuple[DatasetRole, ...] = ()` — roles the model needs; enforced **only when non-empty** (the base declares none, so existing flows never gate).
- `REQUIRED_DATASET_CAPABILITIES: tuple[str, ...] = ()` — duck-typed needs, checked against `dataset_capabilities(ds)` (`GEO_PANEL`, `HAS_INDICATORS`, `HAS_TRIALS`).
- `_coerce_dataset(data)` — wraps a `PanelDataset` (no data motion) or takes a `Dataset`, validates the declared needs with a clear `ValueError`. Mirrors `_coerce_model_params`.

The constructor keeps `self.panel` a `PanelDataset` (every existing reader unchanged) and adds `self.dataset` (the role-tagged view). A `Dataset` passed as `panel` is converted back to a panel view via `as_panel()`.

## What changed for the non-MMM families

The concat hack is gone. CFA/LCA `_prepare_data` now reads `self.dataset.observed()`
(one role-aware call returning the measured columns in `[target, *predictors,
*controls, *indicators]` order — **identical** to the legacy concat order, so results
are byte-stable) and call the shared `CustomMMM._set_non_mmm_defaults()` (which fills
the ~15 model-agnostic contract attributes from `self.dataset.coords`). The awareness
model declares `REQUIRED_ROLES = (TARGET, PREDICTOR)` as a positive demonstration of
the gate.

## Backward compatibility

Strictly additive. `MFFConfig`/`PanelDataset` are untouched; `load_mff` still returns
a `PanelDataset`; the model constructor still accepts a `panel`; reporting/analysis
read the same MMM surface. `Dataset` exposes that surface as views, so it is
duck-type-droppable wherever a panel is read.

## Roadmap

- **A1 (done):** the three modules + adapters + model declaration + kill the CFA/LCA concat hack. (`tests/test_dataset.py`.)
- **A2:** thread an optional `spec["dataset"]` `DatasetSchema` through `agents/fitting.build_model` (validated against `DATASET_SCHEMA`, like `model_params`); round-trip `dataset_schema` in the serializer; populate the garden manifest `dataset_schema` field (AST + `model_json_schema()`); teach `_garden_schema_warnings` about `required_roles`.
- **A3:** `synth/mff.py` emits a `DatasetSchema` (TRIALS role for binomial worlds); make `Dataset` the primary internal read in `_prepare_data`; optionally route TRIALS/OFFSET/WEIGHT into the base likelihood block.

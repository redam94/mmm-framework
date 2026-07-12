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

## Native loading (A3)

`Dataset.from_wide(table, schema)` and `dataset_loader.load_dataset(source, schema)`
load a **wide, role-tagged table** (CSV/parquet/DataFrame) into a `Dataset`
*directly* — no kpi/media/control classification. This is how a genuinely non-MMM
family brings its own data shape: a CFA/LCA indicator matrix or a survey is loaded
with INDICATOR-tagged columns and fits without ever constructing an MMM panel.
Scope: a flat / cross-sectional table (one row per observation), with optional
`time_col` / `group_cols` for the coordinate axes; geo/product panels keep using
`load_mff`. The agent's `build_model` takes this native path whenever `spec["dataset"]`
is present (skipping the MFF kpi requirement entirely). `Dataset.trials()` exposes a
`TRIALS`-role column so a count family reads the binomial denominator **per
observation** from the data (the awareness model prefers it over the scalar
`number_of_trials`). Tests: `tests/test_dataset_native.py`.

## Roadmap

- **A1 (done):** the three modules + adapters + model declaration + kill the CFA/LCA concat hack. (`tests/test_dataset.py`.)
- **A2 (done):** `spec["dataset"]` validated through `build_model`; serializer round-trips `dataset_schema`; garden manifest `dataset_schema` populated (AST `static_dataset_requirements`); `_garden_schema_warnings` understands `required_roles`. (`tests/test_dataset_threading.py`.)
- **A3 (done):** native `Dataset.from_wide` / `load_dataset` + the `build_model` native path + `TRIALS` per-observation likelihood hook. (`tests/test_dataset_native.py`.)
- **Deferred:** make `Dataset` the primary internal read in the base `_prepare_data` (architectural; the MMM hot path currently reads `self.panel`); `src/mmm_framework/synth/mff.py` emitting a `DatasetSchema` answer-key; OFFSET/WEIGHT routing into the base likelihood; a long-format / geo-panel native loader.

# Sub-channel measurement — creative / keyword / campaign arms

> **Status:** Design doc, 2026-07-02. Companion to
> `technical-docs/continuous-learning-wiring.md` (the pinned implementation
> contract, §2.5/§3.3) and `technical-docs/continuous-learning.md` (the
> engine). This document records what sub-channel measurement *is* in this
> codebase today, the formulation we chose for measuring below the channel
> level, and what is deliberately deferred.

## The problem

"Which channel" is often not the live question. The money question is which
creative family inside a channel, which keyword group inside search, which
campaign inside a platform. Today the repo has **three disconnected
capabilities** that are each sub-channel-shaped, none of which share a
`subchannel` concept end-to-end:

1. **The breakout-weighted MMM** (`examples/garden_models/breakout_weighted_mmm.py`)
   — the only working sub-channel *model*. `model_params.breakout_groups`
   (`{parent → [sub_columns]}`, ≥2 members) groups raw media columns into a
   virtual parent channel whose input is a partial-pooled, sum-preserving
   weighted aggregate (`logtau ~ HalfNormal(σ_w)`, `w_raw = exp(logtau·z)`,
   exact renorm so `Σ w_k S_k = Σ S_k`; breakout_weighted_mmm.py:296-346) fed
   into ONE shared adstock→saturation→β curve, plugged in via the
   `BayesianMMM._channel_media_input` hook (`model/base.py:1089`, called at
   1481 on the parametric-adstock path only). τ→0 nests the plain pre-summed
   channel. It is a **mixing model**: it learns the within-channel *weights*,
   not per-sub response curves, and it cannot express sibling cannibalization
   (a weighted sum into one saturation is substitution-by-construction).
2. **The MFF data format** already reserves sub-channel dimensions —
   `DimensionType.{CAMPAIGN, OUTLET, CREATIVE}` (`config/enums.py:8-16`) and
   the canonical columns `Campaign`/`Outlet`/`Creative`
   (`config/mff.py:38-67`); the ad-platform connectors emit them
   (`integrations/ad_platforms/base.py:28-37`). But the loader only builds
   panels on Period/Geography/Product: `split_dimensions` retains
   Outlet/Campaign rows at extraction (`data_loader.py:531-536`) with **no
   downstream consumer** (and enabling it produces a duplicated index that
   `_align_to_target_dimensions` cannot reindex, data_loader.py:639-678);
   `Creative` is never read at all. `MediaChannelConfig.parent_channel` and
   `MFFConfig.get_hierarchical_media_groups()` (`config/mff.py:207`) are
   metadata with zero consumers.
3. **The experiment stack** is uniformly keyed on a single `channel: str`:
   the design engine (`planning/design.py:42/348/640/846`), the lifecycle
   registry (`channel TEXT NOT NULL`, `api/sessions.py:167`), the calibration
   likelihood (`ExperimentMeasurement.channel` "must be one of the model's
   channels", `calibration/likelihood.py:138-139`), and the agent/REST
   surfaces. There is no sub-channel field anywhere in the experiment
   lifecycle.

The practical consequence: a creative-level lift test can be *designed* today
(pass the sub-stream's own MFF variable name as `channel`), but its readout
has nowhere first-class to live and nothing structured to calibrate.

## Chosen formulation: arms on the continuous-learning surface

For *measuring* sub-channel effects (as opposed to decomposing an
observational MMM), the natural home is `mmm_framework.continuous_learning`,
whose surface is already K-generic — "channels" are just named dimensions of a
spend vector (`surface.py:65`, `model.py:285`). Three candidate formulations
were considered (analysis in the wiring contract's Phase A):

* **(a) Arms as extra surface dims + grouped budget constraints — chosen.**
  `"Search │ Brand"` and `"Search │ NonBrand"` become two entries in
  `channels`, each with its own `(β, κ, α)`; the parent's budget becomes an
  SLSQP equality constraint `Σ_{i∈g} s_i = B_g` alongside the global
  `Σ s = B` (the eq-constraint machinery already exists in
  `planner.py::_slsqp_allocate`). Within-parent cannibalization is expressed
  natively as sign-informed `γ < 0` priors on sibling pairs (`PAIR_SIGNS`,
  `model.py`), and `arms.within_parent_pairs(spec)` restricts probing to
  within-parent pairs (with `cross_parent_pairs(spec, pairs_of_parents=...)`
  for explicitly declared cross-parent probes) — containing the quadratic
  pair blow-up. (`probe_pairs_excluding` / `demote_channel` remain the
  walled-garden demotion tools: they *exclude* one channel's pairs and cannot
  express "within-parent only".) Crucially, the CCD's *designed* variation
  manufactures exactly the independent sub-stream variation whose absence
  makes the observational breakout mix unidentifiable (the
  `breakout_collinear` lesson, `synth/dgp.py:1639`). The flatten-to-named-arms
  pattern is precedented by `planning/budget.py:370-490` (`GEO_ARM_SEP`,
  per-(geo,channel) arms into the unchanged channel-generic optimizer).
* **(b) Nested activation (weighted aggregate inside one curve)** — the
  breakout model's formulation transplanted. Rejected for the bandit:
  activations are elementwise on a scalar spend per dim; nesting requires
  ragged per-channel sub-vectors through `surface_value`/likelihood/planner,
  breaking the single-source-of-truth `jax.grad` path — and it cannot
  represent sibling cannibalization at all.
* **(c) Per-channel mini-programs** — one independent loop per parent.
  Rejected: discards cross-channel γ and the shared budget, the two things
  the module exists to learn, and forfeits joint `expected_regret`/ENBS.

### Implementation surface (`continuous_learning/arms.py`, per contract §2.5)

```python
ARM_SEP = " │ "                    # same separator as planning/budget.py geo arms

@dataclass
class ArmSpec:
    channels: list[str]            # flattened arm names, e.g. "Search │ Brand"
    parents: list[str]             # parent per arm (== name when unsplit)
    groups: dict[str, list[int]]   # parent -> arm indices

def expand_arms(channels, arms) -> ArmSpec
def within_parent_pairs(spec) -> list[tuple[int, int]]
def cross_parent_pairs(spec, pairs_of_parents=None) -> list[tuple[int, int]]
def default_arm_pair_signs(spec, *, within="neg", base=None) -> dict
```

Within-parent siblings default to `"neg"` (shared-audience substitution);
cross-parent pairs default `"weak"` unless overridden. The planner functions
(`_slsqp_allocate`, `allocate_under_sample`, `thompson_wave`,
`expected_regret`, `plan_from_posterior`) gain optional
`group_budgets: list[tuple[list[int], float]] | None` with a feasibility check
(`Σ B_g <= B`, groups disjoint), so every Thompson draw, funding-line read,
and regret pass respects the same constraints.

### Cell-count economics (the real constraint)

`central_composite` yields `1 + 2K + 2·|probe| + K` cells
(`continuous_learning/design.py:44`). Each extra arm therefore costs
**≈ 3 cells per wave** (2 axial + 1 shutoff) plus 2 per probed pair, and each
cell needs ≥ 1 geo. With default all-pairs probing the pair count is quadratic
in K — sub-channel use *must* restrict probe pairs to within-parent siblings.
The service layer warns when `n_cells > n_geo` rather than silently designing
an unrunnable wave. Practical guidance: split one or two parents at a time —
the ones whose mix decision is worth money — and keep the rest as single arms.

## The `subchannel` registry column (contract §3.3)

The experiment lifecycle gains a **nullable attribution field**, not a new
unit of analysis:

* `ALTER TABLE experiments ADD COLUMN subchannel TEXT` in the existing
  migration loop (`api/sessions.py:189-200` pattern), with passthrough on
  `upsert_experiment` / `_experiment_row_to_dict` and a
  `list_experiments(subchannel=...)` filter.
* Agent tools `log_experiment` / `plan_experiment` /
  `record_experiment_readout` and `POST /experiments` accept optional
  `subchannel` (a creative/keyword/campaign identifier).
* **MMM calibration stays channel-level.** `ExperimentMeasurement` does not
  grow a subchannel field: a sub-channel readout feeds either a learning
  program with arms (via `evidence.experiments_to_summaries`, which matches a
  subchannel readout to the arm named `f"{channel}{ARM_SEP}{subchannel}"`) or
  the breakout-model **share likelihood** (next section). A channel-level
  readout on a split parent is skipped with an explicit reason ("channel-level
  readout on a split parent") rather than mis-assigned to one arm.

## Share-based breakout calibration (implemented)

The in-graph route for sub-channel evidence to constrain the breakout model's
weights. A learning program (or any creative-level study) measures the
*within-channel composition* — what fraction of the parent's incremental
response each sub-stream carries — and that composition becomes a likelihood
term on the breakout model's `breakout_share_<C>` simplex Deterministic
(`breakout_weighted_mmm.py`, `share_k = w_k S_k / Σ_j w_j S_j`). This is
exactly the evidence the `breakout_collinear` world lacks: when sub-streams
share one flighting calendar the time-series likelihood is flat in the mix, and
a designed share measurement restores identification (the CCD's independent
sub-stream variation, imported as data).

**The measurement** — `calibration/likelihood.py::ShareMeasurement`, a frozen
dataclass beside `ExperimentMeasurement` (which stays scalar; the vector
composition deliberately does NOT ride `spec["experiments"]` / the registry):

* `channel` (the VIRTUAL parent), `breakouts` (MFF sub-stream column names in
  an **explicit order**, ≥ 2), `shares` (same order; floored at ε, must sum to
  ~1, renormalized to an exact simplex).
* Two measurement-error families. `distribution="logistic_normal"` (default,
  preferred): an `MvNormal` on the `K−1` **additive log-ratios** w.r.t. the
  LAST breakout as reference — matches the renormalized weights' `K−1` dof
  exactly and preserves the source draws' correlation structure; requires
  `log_ratio_cov` (`(K−1)×(K−1)`, symmetric, **strictly** PD — checked via
  Cholesky at construction, so a singular/PSD matrix fails there with an
  actionable "add a small diagonal ridge (e.g. 1e-9)" message instead of
  surfacing as pm.MvNormal's cryptic `logp = -inf` at sampler init deep
  inside the fit job).
  `distribution="dirichlet"`: `shares ~ Dirichlet(concentration ·
  model_share)` — a single-precision alternative with the Dirichlet's rigid
  negative-correlation structure. Exactly one of `log_ratio_cov` /
  `concentration` must be set, matching the distribution.
* `attach_share_likelihood(name, share_expr, measurement)` mirrors
  `attach_experiment_likelihood`: registers a `{name}_model_share`
  Deterministic (model-implied vs observed shares, the overconfidence
  diagnostic) then the observed node. `to_dict`/`from_dict` round-trip.

**Threading** — `BreakoutWeightedParams.share_calibrations: list[dict]` (each
entry a `ShareMeasurement` dict), riding the existing `spec["model_params"]`
path untouched (constructor pass-through, serializer round-trip, garden
manifest `config_schema`). A pydantic validator round-trips every entry through
`ShareMeasurement.from_dict` at construction (bad specs fail at build time with
a clear message) and checks the entry's `channel` is a `breakout_groups` parent
whose `breakouts` all come from that group's columns.

**Attachment** — the base gate now calls `_add_experiment_likelihoods`
**unconditionally** (`model/base.py`; the method early-returns when no scalar
experiments are registered, so every existing model's graph is byte-identical)
— this guarantees subclass overrides always run, so a share-ONLY calibration
attaches. `BreakoutWeightedMMM._add_experiment_likelihoods` attaches each share
calibration to the channel's `breakout_share_<C>` (guaranteed built —
`_channel_media_input` runs earlier in the base build), then calls `super()`.

* **Order is load-bearing:** the ALR covariance is defined w.r.t. the ordered
  breakouts, so the measurement's `breakouts` must match the MODEL's breakout
  order (`self._breakout_names[C]`) **exactly** — a mismatch or strict subset
  raises with both sides listed rather than silently reindexing (reordering
  requires re-deriving the covariance).
* **Double-counting guard:** a parent carrying BOTH a scalar
  `ExperimentMeasurement` and a share calibration triggers a warning — if both
  derive from the same program/wave, the evidence enters the posterior twice
  with correlated errors. Rule: per program export EITHER per-arm scalar
  readouts OR (one parent-level readout + one share vector), never both; the
  `source` provenance block exists for this audit.

**The CL exporter** — `continuous_learning/arms.py::arm_shares(post, spec,
parent, spend_ref, *, breakout_name_map, mode="zero_out", draws=500, rng=None)`
returns the `ShareMeasurement`-shaped payload (`channel`/`breakouts`/`shares`/
`log_ratio_cov`/`source`). Per posterior draw: `mode="zero_out"` reads arm `i`
as `R(s_ref) − R(s_ref with arm i zeroed)` (captures `γ`);
`mode="main_effect"` reads `β_i · act(s_ref_i)` (strictly positive). Shares are
normalized within `spec.groups[parent]`.

* **Non-positive responses are EXCLUDED, not floored.** A zero-out draw where
  any arm's response goes non-positive (strong cannibalization at `spend_ref`)
  has no well-defined share, and flooring it at ε would inject
  `z ≈ log(ε) ≈ −20` outliers into the ALR covariance — even one floored draw
  in 500 inflates a diagonal of O(0.3) by ~1.5, silently down-weighting the
  share evidence ~20–40×. Such draws are dropped from BOTH the shares and the
  covariance: any exclusion warns; > 20 % excluded raises (zero-out shares are
  ill-defined — use `mode="main_effect"` or a different `spend_ref`); < 10
  surviving draws raises. `source` records `n_draws` (surviving) and
  `n_excluded`. A tiny `1e-9` floor remains on the surviving draws' shares as
  a pure numerical guard before the log.
* **Location/cov consistency.** The exported `shares` are the **inverse-ALR
  (softmax) of `mean(z)`** over the surviving draws — not the arithmetic mean
  of the share draws — so the consumer's observed `z_hat = ALR(shares)` is
  exactly `mean(z)`: the MvNormal location and the empirical ALR covariance
  (+ `1e-9` ridge) summarize the same draws of the same distribution.

`breakout_name_map` (arm
sub-name → MFF breakout column) is **required** with no fuzzy fallback — the
three naming conventions (MFF columns, `ARM_SEP` arm names, registry
`subchannel`) must be bridged explicitly or siblings get silently swapped.

**Estimand caveat (identification story):** the MMM's share is an
*effectiveness share through ONE shared curve at the panel's spend mix*; the CL
share is a ratio of PER-ARM curves at `spend_ref`. They coincide near the
panel's operating point under the breakout model's own shared-curve
assumption — so export shares at (approximately) the panel's sub-stream spend
mix, not at a CL-optimal reallocated point. Any understatement of CL
uncertainty transfers directly into over-tight MMM weight HDIs on collinear
panels (where the share likelihood dominates the mix posterior); compare
`{name}_model_share` to the observed shares as the diagnostic.

Tests: `tests/test_breakout_weighted_mmm.py` (graph wiring, order/validation
errors, Dirichlet variant, double-count warning, and the slow
`test_share_calibration_restores_identification` demo on `breakout_collinear`)
and `tests/test_arm_shares.py` (the exporter on a fabricated posterior).

## MFF guidance

Until the `split_dimensions` path has a real consumer, the supported route to
creative/keyword granularity in panel data is:

* **One media variable per sub-stream, encoded in `VariableName`**
  (e.g. `Search_Brand`, `Search_NonBrand`) — exactly the breakout model's data
  contract, and what the loader fully supports today.
* The `Creative` / `Campaign` / `Outlet` columns remain **reserved**: valid
  MFF, emitted by connectors, carried through validation — but not modeled.
  Do not configure `split_dimensions` expecting per-creative columns; the
  extraction retains the rows but nothing downstream pivots them (and geo/product
  panels will hit the duplicate-index reindex failure, data_loader.py:639-678).
* Keywords have no dedicated column; they ride `VariableName` (or `Campaign`
  for ad-group-level ingestion via the connectors).

## Deferred

* **Per-arm cost descriptors** — `measurement_unit`/`spend_column`/`cpm`/`cpc`
  are per-*channel* (`config/variables.py:90-103`); arms currently inherit the
  parent's cost basis. Per-arm CPMs (common when creatives run in different
  placements) need a per-arm descriptor and per-arm dollar conversion in the
  learning-program service.
* **Total-lift constraints across an arm group** — a channel-level readout on
  a split parent is a constraint on the *sum* of its arms' responses; wiring
  it as a summary observation over the group (rather than skipping it) is
  future work in `evidence.py`.
* **A first-class `split_dimensions` consumer** — pivoting Outlet/Campaign/
  Creative variation into model columns at load time (and fixing the
  duplicate-index reindex) remains unplanned; revisit if a second modeling
  consumer beyond the breakout pattern appears.

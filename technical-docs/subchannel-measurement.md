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
  subchannel readout to the arm named `f"{channel}{ARM_SEP}{subchannel}"`) or,
  in the future, a breakout-model share likelihood. A channel-level readout on
  a split parent is skipped with an explicit reason ("channel-level readout on
  a split parent") rather than mis-assigned to one arm.

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

* **Share-based breakout calibration likelihood** — an in-graph route for a
  creative-level lift test to constrain the breakout model's weights: the
  estimand `share_k × parent_contribution` is buildable from the existing
  `breakout_weights_<C>` / `breakout_share_<C>` deterministics
  (breakout_weighted_mmm.py:331-339) plus a `subchannel` branch in
  `_add_experiment_likelihoods` (model/base.py:1862). Nothing structural
  blocks it; it needs its own identification story (weights are only
  identified when sub-stream flighting varies independently).
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

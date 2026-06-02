# Requirements: Deferred Causal-MMM Features

**Status:** Specification for work intentionally deferred during the `critique.md`
P0–P2 implementation. **None of these are implemented.** This document is the
contract for implementing them later.

**Why deferred:** all three change the *likelihood / posteriors* (or, for Full-ID,
require a research-grade algorithm + heavy dependency). Every feature here must
clear a stricter bar than the diagnostics and identification *checks* already
shipped: it must be validated against **held-out experimental evidence** before
it can be recommended, because a feature that changes the numbers can be
confidently wrong in a way a report section cannot.

| ID | Feature | Effort | Changes posteriors? | Gating validation |
|----|---------|--------|--------------------|-------------------|
| DF-1 | Geo-heterogeneous media coefficients | L | Yes | Geo-holdout backtest vs. geo-lift |
| DF-2 | Grouped / hierarchical priors for collinear channels | M | Yes | Incrementality calibration on pooled channels |
| DF-3 | Full-ID (Tian–Pearl) | M–L | No (identification layer) | Cross-check vs. vetted library |

---

## 0. Cross-cutting requirements (apply to every feature below)

These are non-negotiable and exist because the current framework's credibility
rests on *not* silently changing established results.

- **R0.1 — Off by default.** Each feature is gated behind a config flag whose
  default reproduces today's behavior exactly. No existing model's posterior may
  change unless the user opts in.
- **R0.2 — Zero-change regression test.** A test must prove that, with the
  feature disabled, the fitted posterior is bit-for-bit identical (same seed) to
  the pre-feature code path. This is the acceptance gate for "off by default".
- **R0.3 — Held-out experimental validation before default-recommend.** A
  posterior-changing feature may ship *behind a flag* once R0.1/R0.2 hold, but it
  may not be *recommended as a default* until it is shown to improve (or at least
  not degrade) out-of-sample prediction against **held-out geo-lift /
  incrementality data**, on at least one real or realistically-simulated dataset
  with known ground-truth effects.
- **R0.4 — Naming/contract preservation.** Downstream code reads specific PyMC
  variable names. Any refactor of the media/coefficient block MUST keep emitting:
  - `beta_{channel}` (a scalar Deterministic) — read by the unobserved-confounding
    sensitivity (P0-2, `validation/sensitivity_unobserved.py`) and reporting.
  - `channel_contributions` (Deterministic, dims `(obs, channel)`).
  - `beta_controls` as a single `(n_controls,)` vector — read by
    `reporting/extractors/bayesian.py` and `validation/validator.py`.
  If a feature makes a coefficient geo/group-varying, `beta_{channel}` must be
  emitted as the *summary* coefficient (e.g., the hierarchical mean) so existing
  consumers keep working and assess the pooled effect.
- **R0.5 — Integrate with existing causal machinery.** Each feature must compose
  with P0 (experiment-calibrated `roi_prior`, robustness value, refutation), P1
  (causal role typing), and P2 (collinearity + geo diagnostics) rather than
  bypass them.
- **R0.6 — Honest reporting.** New estimates must be reported with the assumption
  they rest on and with HDIs, in the same register as the shipped diagnostics
  ("this rests on geo-exogeneity / partial pooling", not "geo ROAS is X").

---

## DF-1. Geo-heterogeneous media coefficients

### Motivation
Cross-geo spend variation is quasi-experimental identifying signal (critique.md
§3.7). Today the core model pools media fully: a single scalar coefficient per
channel, with geography entering only as an additive intercept offset. The geo
diagnostic shipped in P2-1 (`validation/geo_identification.py`) only *reports*
whether enough cross-geo variation exists; it does not let the model use it.
This feature lets the model estimate **geo-level treatment heterogeneity** while
borrowing strength across geos.

### Current state (anchors)
- Media coefficient: `model/base.py:907-919` — `beta = _sample_from_prior_config("beta_{channel}", roi_prior, default=Gamma(mu=1.5, sigma=1.0))`, scalar, then `channel_contrib = beta * x_saturated`.
- Geo effect: `model/base.py:867-873` — `geo_sigma = HalfNormal(0.3)`, `geo_offset = Normal(0,1,shape=n_geos)`, `geo_contribution = (geo_sigma*geo_offset)[geo_idx]`. **Additive intercept only**; media is geo-invariant.
- `geo_idx` (obs → geo code), `n_geos`, `geo_names`: `model/base.py` `_prepare_data`.
- Toggle precedent: `HierarchicalConfig.pool_across_geo` (`config/model.py:19`), `use_non_centered` (`:23`).

### Goal
Optionally estimate `beta_geo[g, c]` (per geo × channel) under hierarchical
partial pooling so that (a) geo-specific ROAS is estimable, (b) sparse geos
shrink toward the channel mean, and (c) the channel mean remains the
experiment-anchored quantity.

### Functional requirements
- **DF1.1** New flag `HierarchicalConfig.geo_varying_media: bool = False`
  (default off ⇒ R0.1). Only active when `has_geo and n_geos >= 2`.
- **DF1.2 — Hierarchical prior.** When active, sample
  `beta_geo[g,c] ~ Normal(mu_c, tau_c)` with a **non-centered** parameterization
  when `use_non_centered` (required for sparse geos):
  `beta_geo = mu_c + tau_c * z_gc`, `z_gc ~ Normal(0,1)`, `tau_c ~ HalfNormal(.)`.
- **DF1.3 — Calibration anchors the mean, not the geos.** The channel-level mean
  `mu_c` MUST honor `MediaChannelConfig.roi_prior` (P0-1) when set — i.e. the
  experiment pins the *pooled* effect and geos vary around it. Per-geo
  coefficients must NOT each independently consume the calibrated prior.
- **DF1.4 — Per-obs contribution** uses `beta_geo[geo_idx[obs], c] * sat_c(adstock_c(x))`.
- **DF1.5 — Contract (R0.4).** Emit `beta_{channel}` as a Deterministic equal to
  `mu_c` (the pooled mean) and keep `channel_contributions` populated per obs.
- **DF1.6 — Identifiability pre-check.** Before fitting with this feature, run
  `geo_spend_variation_diagnostic` (P2-1); **refuse or loudly warn** when a
  channel's cross-geo spend CV is below threshold (insufficient variation ⇒
  unidentified geo coefficients). Surface which channels are under-identified.
- **DF1.7 — Reporting.** Add a per-geo, per-channel ROAS table with HDIs and the
  cross-geo heterogeneity `tau_c`; flag geos with low per-geo ESS.

### Backward-compat requirements
- R0.1/R0.2: with the flag off, the geo block and media block are byte-identical
  to today.
- The number of parameters jumps from `n_channels` to `n_channels*(1 + n_geos)`;
  this is acceptable *only* behind the flag and the DF1.6 gate.

### Acceptance criteria (the deferral gate)
- **A1.** R0.2 regression test passes (off ⇒ identical posterior).
- **A2.** Per-geo-channel `r_hat < 1.01` and ESS above threshold on a dataset
  with genuine cross-geo variation; the fit must *fail loudly* (not silently
  return garbage) when DF1.6 detects insufficient variation.
- **A3 — Held-out geo backtest (R0.3).** On a panel with held-out geos (or a
  simulation with known per-geo effects), the geo-varying model's per-geo ROAS
  predictions must be **no worse** (ideally better) than the pooled model at
  predicting the held-out geos' outcomes / known geo-lift. Ships behind a flag
  regardless; becomes recommendable only if A3 passes.
- **A4.** Prior-sensitivity check: the geo split must be driven by data, not by
  `tau_c`'s prior, when the data is informative.

### Risks & mitigations
- **Parameter explosion / weak identification** when spend scales together across
  geos → DF1.6 hard gate + hierarchical shrinkage (DF1.2) + A2.
- **Endogenous geo targeting** (spend follows local demand) → the geo-exogeneity
  caveat already authored in P2-1 must be attached to all geo-level output; geo
  heterogeneity is "conditional on geo-assignment exogeneity", never asserted.
- **Calibration scope mismatch** → DF1.3 (mean-anchoring) resolves it; document
  that `roi_prior` is interpreted as a prior on the *national/pooled* effect.

### Sequencing
After DF-1's DF1.6 reuses P2-1 (done). Independent of DF-2. **Effort: L.**

---

## DF-2. Grouped / hierarchical priors for collinear channels

### Motivation
Brands spend on all channels together, so per-channel coefficients are often
weakly identified regardless of confounding (critique.md §3.5). P2-2 ships the
*detection* (`validation/channel_diagnostics.py::_detect_collinear_clusters`) and
*recommends* grouped priors but does not apply them. This feature applies a
hierarchical prior across related channels so collinear channels borrow strength
and their split is regularized rather than overconfident.

### Current state (anchors)
- Media coefficients are independent per channel (`model/base.py:907-919`).
- `media_groups` (from `MediaChannelConfig.parent_channel`,
  `config/mff.py::get_hierarchical_media_groups`) is computed but **unused for
  priors** (used only for coord naming).
- Shrinkage infra already exists for *controls* in
  `mmm_extensions/components/variable_selection.py` (horseshoe / spike-slab /
  LASSO, returning a `VariableSelectionResult`) — a reference, not a drop-in.
- P2-2 produces `grouped_prior_recommendations` (reporting only).

### Goal
Optionally place a partial-pooling prior on the coefficients of channels in the
same group: `beta_c ~ Normal(mu_group, tau_group)`, so collinear channels are
shrunk toward a shared group mean by an amount the data chooses (`tau_group`).

### Functional requirements
- **DF2.1** New flag `ModelConfig.use_grouped_media_priors: bool = False`
  (default off ⇒ R0.1).
- **DF2.2 — Group source.** Groups come from explicit `parent_channel`
  (`media_groups`) by default. Auto-grouping from P2-2 detected clusters is
  **opt-in only** and must be logged (data-driven grouping is riskier; never the
  default).
- **DF2.3 — Hierarchical prior.** For each group: `mu_group ~ <weakly-informative>`,
  `tau_group ~ HalfNormal(.)`, `beta_c = mu_group + tau_group * z_c` (non-centered).
  Channels in no group keep today's independent prior.
- **DF2.4 — Calibration precedence.** A channel with an experiment-calibrated
  `roi_prior` (P0-1) is **excluded from group shrinkage** (its randomized prior
  wins; it must not be pulled toward a weak group mean). Optionally it may
  *inform* `mu_group`, but it may not be shrunk by it. This must be explicit and
  tested.
- **DF2.5 — Contract (R0.4).** Keep emitting per-channel `beta_{channel}` and
  `channel_contributions`.
- **DF2.6 — Reporting.** Mark which channels were pooled and disclose that their
  per-channel ROIs are *partially pooled* (not independent), linking back to the
  P2-2 collinearity finding that motivated it.

### Backward-compat requirements
- R0.1/R0.2: flag off ⇒ identical posterior.
- Must not change the meaning of `beta_{channel}` for un-grouped channels.

### Acceptance criteria (the deferral gate)
- **A1.** R0.2 regression test passes.
- **A2 — No silent over-shrinkage.** On a dataset where one channel is strongly
  identified and a collinear neighbor is not, the strong channel's posterior must
  not be materially dragged toward the weak neighbor (the adaptive `tau_group`
  must let an informative channel escape the pool). Tested explicitly.
- **A3 — Calibration precedence (DF2.4).** A calibrated channel's posterior must
  match its `roi_prior`-anchored estimate whether or not grouping is on.
- **A4 — Held-out validation (R0.3).** On data with known channel effects (or a
  geo-lift on the pooled channels), grouped-prior per-channel ROIs must be
  better-calibrated (interval coverage) than independent priors, and total group
  effect must be unbiased. Recommendable only if A4 passes.

### Risks & mitigations
- **Over-shrinkage of a strong channel** → adaptive `tau_group` + A2.
- **Wrong groups** → default to explicit `parent_channel`; auto-clusters opt-in
  + logged (DF2.2).
- **Interaction with calibration** → DF2.4 + A3.

### Sequencing
Builds on P2-2 (done) for group detection and on the `variable_selection`
component as a structural reference. Independent of DF-1. **Effort: M.**

---

## DF-3. Full identification (Tian–Pearl ID) — out of scope, spec for completeness

### Why out of scope
The shipped identification layer covers **backdoor + front-door + IV**
(`dag_model_builder/identification.py`), which handle ~95% of real MMM DAGs. The
complete ID algorithm (Tian–Pearl / Shpitser–Pearl) adds the remaining cases
(effects identifiable via neither pure backdoor nor front-door, via
c-component / hedge reasoning). It is deferred because:
1. Its output is a general **estimand (a do-expression)**, which the current
   additive likelihood has **no estimator** for — identifying an effect the
   engine cannot estimate adds no user value yet.
2. The algorithm is research-grade and its diagnostics are cryptic to the
   non-expert audience the agent layer targets.

### Requirements *if* ever pursued
- **DF3.1** Implement the ID algorithm (recursive c-component factorization with
  hedge detection) returning, for `do(T)` on `Y`: identifiable Y/N **and** the
  symbolic estimand, not just a boolean.
- **DF3.2 — Do not hand-roll alone.** Cross-check against a vetted library
  (e.g. Ananke, pgmpy, DoWhy) over a corpus of DAGs — mirroring the P2-6 networkx
  cross-check pattern. This implies a heavier optional dependency; gate it as a
  dev/optional extra, never a hard runtime dep.
- **DF3.3 — Pair with an estimator or don't surface it.** Only expose a Full-ID
  verdict once there is an estimator (e.g. a front-door/IV/weighting estimator)
  that can actually compute the identified estimand; otherwise it misleads.
- **DF3.4 — Trigger.** Pursue only when a real DAG appears that is identifiable
  but not by backdoor/front-door/IV. Until then, the three implemented criteria
  + experiment calibration are the supported path.

**Effort: M–L (algorithm) + dependency decision.**

---

## Appendix: definition of done for a deferred feature

A deferred feature is "done" when: R0.1–R0.6 hold; the feature-off regression
test (R0.2) passes; the feature-on path passes convergence + identifiability
gates; the held-out experimental validation (R0.3) is documented; reporting
discloses the assumption; and the relevant `critique.md` roadmap memory is
updated. Shipping behind a flag is allowed after R0.1/R0.2; recommending as a
default requires R0.3.

# Multiplicative effects in the continuous-learning loop — justification for the additive surface + implementation plan

**Status: PLAN (not implemented).** Written 2026-07-03, alongside the
non-Gaussian follow-ups (Student-t likelihood, unconstrained-space `ThetaMap`
acquisition, GLM Fisher weights — all shipped; see `continuous-learning.md`).

## 0. The question

Classical MMMs usually make media **multiplicative**: the KPI is modeled as
`y = baseline(t) · Π_c effect_c(spend)` (or additively in log space), because a
bigger market/season both *spends more* and *responds more in absolute terms*.
The continuous-learning module instead fits a purely **additive** surface

    y_gt = a_g + R(s_gt) + ε_gt,      R(s) = Σ_c β_c f_c(s_c) + Σ_{c<c'} γ_cc' f_c f_c'

with a common `R` across geos. Why is that reasonable, when is it not, and
what would a multiplicative variant look like?

## 1. Why the additive surface is reasonable HERE (and why the MMM argument doesn't transfer)

The short version: **in the MMM, multiplicative structure is a causal-
identification necessity; in the CL loop it is only a variance/heterogeneity
refinement, because randomization already removes the confounding that makes
multiplicativity dangerous to omit.** Four specific arguments:

### 1.1 Randomization breaks the baseline↔effect correlation that motivates multiplicative MMMs

The MMM's problem: baseline demand `a(t)` is *correlated with spend* in
observational data (budgets chase seasons). If effects truly scale with
baseline and you fit additively, the misattribution lands **in the media
coefficients** — a bias.

In the CL loop, geo→cell assignment is randomized (and stratified on
accumulated KPI since the 2026-07-02 follow-ups), so `E[a_g | cell] = ā` for
every cell by construction. Under a multiplicative truth
`y_g = a_g · (1 + r(s))`:

    E[y | cell c] = ā · (1 + r(s_c)) = ā + ā·r(s_c)

The cell means — the only thing the designed contrast identifies — trace
`ā·r(s)`: the **population-average incremental response**. The additive fit's
`R(s)` estimates exactly that, and `ā·r(s)` is exactly the quantity the
allocator needs (total incremental KPI over the panel at allocation `s`).
Misspecification does not bias the estimand; it becomes **heterogeneity noise**
(§1.3). The pre-period further pins each `a_g` (CUPED-style), absorbing the
level part of the geo size differences into the random intercept.

### 1.2 The trust region + re-centering makes any smooth truth locally additive

Every wave is a central-composite design at `center · (1 ± δ)` — a *local*
perturbation. Any smooth multiplicative surface is additive to first order in
that region: `a·g(s₀+Δ) ≈ a·g(s₀) + a·g'(s₀)·Δ`. The loop refits at each new
center, so the additive Hill surface is a **moving local approximation**, not a
global commitment. This is the same mechanism the misspecification study
already verified empirically for the activation family (single Hill fit on
two-Hill truth: profit gap 0.9% → 0.5% → 0.2% → 0.3% across waves, tracking
the correctly-specified loop) — the sequential trust-region loop erases mild
structural misspecification *for the decision*. The known caveat carries over
verbatim: the guarantee needs the **stable geo set** rule, and calibration
(interval coverage) degrades before decisions do — trust the ranking/funded
set, distrust channel-by-channel magnitudes.

### 1.3 The γ interaction term IS the leading multiplicative cross-term

If the truth composes channels multiplicatively,

    y = a · Π_c (1 + ρ_c f_c(s_c))
      = a · (1 + Σ_c ρ_c f_c + Σ_{c<c'} ρ_c ρ_c' f_c f_c' + O(f³))

the second-order term is exactly the surface's `γ_cc' f_c f_c'` block with
`γ_cc' = a·ρ_c·ρ_c' > 0`. So a multiplicative-composition world does not sit
*outside* the additive model class — it shows up as **positive synergies**, and
the `PAIR_SIGNS` machinery ("pos") lets you encode that prior belief directly.
What the additive surface cannot represent is the third-order and higher terms
— material only when several channels are simultaneously deep into saturation
at high fractional lifts, i.e. outside the trust region anyway.

### 1.4 What omitting multiplicativity actually costs: efficiency, not bias

Under multiplicative truth with randomized cells, the residual around the
additive fit is `(a_g − ā)·r(s_cell)` — **heteroskedastic** (variance grows
with the cell's response and with `Var(a_g)`), not confounded. A Gaussian
homoskedastic likelihood remains a consistent estimator of the cell means
(QMLE logic); the cost is mis-weighted observations (efficiency) and mildly
optimistic intervals in the high-response cells. Two shipped features already
blunt this: **CUPED** removes the `a_g` *level* variation (the dominant part
when `r` is small), and the **Student-t likelihood** (2026-07-03) stops the
inflated-residual cells from dragging the surface. Note honestly: CUPED
removes the level, not the effect-scale interaction — `y_adj ≈ ā + (a_g−ā)·r(s) + ε`
still carries the heteroskedastic term, just without the baseline spread.

### 1.5 Where the additive surface genuinely strains

Be precise about the failure modes, because they define the acceptance tests
for any multiplicative variant:

1. **Very unequal geo sizes + per-geo allocation decisions.** The additive fit
   recovers the population-average response; if you need *per-geo* funding
   decisions and effects scale with size, a common `R` misallocates across
   geos (it can't, and shouldn't, be read per-geo today — the planner
   allocates a national mix).
2. **Wide intensity ranges.** `intensity_min = −100%` (go dark) to `+150%`
   stretches "local". The first-order argument weakens; the loop mitigates by
   re-centering but single-wave readouts at the range edges are the least
   trustworthy — same caveat class as the activation misspecification.
3. **Rate/share KPIs** (conversion rate, awareness share): bounded outcomes
   whose effects compose multiplicatively/logit by nature. The additive
   surface can locally approximate but the observation family is wrong too —
   this is a *likelihood + link* problem, not just an effect-scale problem.
4. **Cross-wave drift in baseline scale.** If `ā` doubles between waves
   (seasonality) under multiplicative truth, the additive `R` fitted on pooled
   waves mixes two effect scales. `time_effect="national"` absorbs the level
   shift but not the effect-scale shift.

## 2. Diagnostic first: detect it before modeling it (Phase 0)

A cheap, pure-numpy check that runs on any accumulated program state — no new
model:

- **Effect-scale test.** Split geos into baseline terciles (pre-period means —
  already computed for CUPED/stratification). Within each tercile, compute the
  designed contrast `lift_T = mean(y | high-spend cells) − mean(y | shutoff/low cells)`.
  Under additive truth `lift_T` is flat in `T`; under multiplicative truth it
  is proportional to the tercile baseline. Report the slope of
  `lift_T / lift_pooled` against `baseline_T / baseline_pooled` with a
  bootstrap CI: slope ≈ 0 → additive fine; slope ≈ 1 → multiplicative.
- **Heteroskedasticity slope.** Regress `log |residual|` on `log(pre-period
  baseline)` (Breusch–Pagan flavor). Positive slope corroborates §1.4.

Wire both into `fit()`'s `diagnostics` dict (advisory, never failing a fit)
and surface them in the Sextant program readout. **This is the gate for the
rest of the plan**: if real programs show slope ≈ 0, stop here — §1 says the
additive surface is not just defensible but preferable (fewer parameters, no
new identification burden).

## 3. Implementation plan for multiplicative effects

Two variants, in increasing order of departure. Both keep the module's core
invariant: **one differentiable JAX surface shared by likelihood, DGP and
planner** — the multiplicative structure wraps *around* `R`, so `surface.py`
does not change at all.

### Phase 1 — DGP + misspecification study (evidence before machinery)

* `dgp.simulate_panel(effect_scale="proportional")` (default `"additive"`,
  byte-identical rng stream): `mu = a_g · (1 + r(s)/a_ref)` with `a_ref` the
  world's `a_level`, so the same `TrueWorld` surface doubles as fractional
  truth. Optional `make_world_multiplicative` convenience.
* Extend the misspecification study (notebook §14 pattern, real
  `LearningState` accumulation, fixed `a_geo`): fit the ADDITIVE model on
  multiplicative truth; measure per-wave profit gap, funded-set agreement,
  marginal-ROAS CI coverage; with/without CUPED; Gaussian vs Student-t.
  **Hypothesis (from §1): decisions near-optimal, coverage degraded in
  proportion to `Var(a_g)/ā²`.** The measured numbers become the doc's
  justification artifact, exactly like the activation-misspecification
  finding.
* Tests: DGP invariance (`effect_scale="additive"` byte-identical), the
  study's headline gate marked `@pytest.mark.slow`.

### Phase 2 — proportional-effects likelihood (`effect_scale="proportional"`)

The minimal genuine multiplicative model — effects scale with the geo
baseline, channels still compose through `R`:

    mu_gt = a_g · (1 + R(s_gt))          # R now a FRACTIONAL lift surface

* **Model** (`model.py`): new `effect_scale` knob (default `"additive"`, no
  new sites, byte-identical). Priors: `beta` becomes fractional
  (`HalfNormal(beta_scale)` — NO `y_scale` multiplier; the scale rides on
  `a_g`), `gamma` likewise. `a_g` keeps its natural-scale prior. Works under
  all three observation families; for NB the composition is
  `softplus(a_g·(1+R))` (unchanged guard semantics).
* **Identification.** The pre-period pins `a_g·(1+R(center))`, not `a_g` — a
  multiplicative confound between the intercept and the center's lift. The
  CCD's **shutoff cells** resolve it: at `s=0`, `f(0)=0 ⇒ mu = a_g` exactly,
  so shutoffs anchor the baseline the way they already anchor β/γ
  collinearity. Consequence to enforce loudly: `effect_scale="proportional"`
  **requires shutoff cells in the design** (validate in `fit`; the CCD always
  has them, arbitrary imported panels may not).
* **Planner** — one number changes. Incremental KPI at allocation `s` is
  `Σ_g a_g·R(s) = A_tot·R(s)`, so every profit expression multiplies `value`
  by the posterior-mean total baseline: `profit = value·Ā_tot·R(s) − cost`.
  Implement as a `value_scale` resolved from the posterior
  (`draw_params`/`PlanResult` carry it; additive → 1.0). Funding line:
  `P(value·Ā_tot·∂R/∂s_c > 1)`. Thompson/regret/ENBS inherit automatically
  since they all go through the same profit callables. Per-draw `A_tot^{(d)}`
  keeps the uncertainty honest (do NOT plug the mean into per-draw optima).
* **Acquisition.** `ThetaMap` gains nothing (β/γ transforms unchanged); the
  per-cell Fisher weights pick up the baseline factor: under proportional
  effects `∂mu/∂θ = ā·∂R/∂θ`, so `design_information` multiplies the centered
  gradients by `ā` (equivalently `unit_info` scales by `ā²`). One extra kwarg
  threaded from `observation_unit_info`.
* **Summaries.** A historical `lift ± se` from a known cell footprint measures
  `Σ_g∈test a_g·(R(test)−R(base))`. The summary dict gains an optional
  `baseline_total` (default: the fitted `Ā` × `scale`, with a documented
  degradation when absent). Structural-stationarity caveat now covers the
  baseline scale too — state it.
* **Serialization/wiring:** payload v2 semantic key `effect_scale` (back-compat
  default `"additive"`, same pattern as `likelihood`); `service.new_program_state`
  config knob; tool/REST passthrough; `WaveRecord` untouched.
* **Tests:** recovery gate on the proportional DGP (β fractional signs +
  funded set), shutoff-required validation, planner `value_scale`
  equivalence (additive world: proportional fit with `Var(a_g)→0` matches the
  additive fit's plan), serialization round-trip.

### Phase 3 — log-link family (full multiplicative channel composition)

Only if Phase 1/2 evidence shows real programs need channel effects to
*compose* multiplicatively (not just scale with baseline):

    y ~ LogNormal(log a_g + R(s), sigma)        # or NB with exp link

* In log space the additive surface becomes exact multiplicative composition:
  `E[y] ∝ a_g·exp(R) = a_g·Π_c e^{β_c f_c}·e^{Σγff'}` — and `γ` now measures
  synergy **beyond** the product baseline (the `Π(1+ρf)` cross-terms are
  already implied by `exp(Σβf)`), so `PAIR_SIGNS` semantics change and must be
  re-documented. Default-`zero` pairs become the natural multiplicative prior.
* The "never log y" data-contract rule is NOT violated: that rule bans
  *preprocessing* (transforming the data corrupts the natural-scale estimand
  and NB counts); a log **link** keeps `y` natural and moves the
  transformation into the likelihood, where the posterior can undo it.
* Planner: incremental KPI = `Σ_g a_g·(exp(R(s))−exp(R(s_base)))`; marginal =
  `value·Ā_tot·exp(R)·∂R/∂s`. Same `value_scale` seam as Phase 2, plus the
  `exp(R)` factor — still one shared surface, differentiated by JAX.
* Summaries need a delta-method bridge (natural-scale lift ↔ log-scale
  contrast). Acquisition Fisher weights: `∂mu/∂θ = ā·exp(R)·∂R/∂θ` — again a
  per-cell multiplier through the existing `unit_info` seam.
* This phase subsumes Phase 2 for LogNormal-appropriate KPIs; keep both knobs
  (`effect_scale`, `likelihood`) orthogonal so `proportional`+`normal` and
  `log-link` coexist.

### Phase 4 — decision gates and non-goals

* **Gate to build Phase 2:** Phase 0 diagnostic slope materially > 0 on a real
  program (or a client KPI that is a rate/share), AND the Phase 1 study shows
  either a persistent profit gap the loop does not erase (> ~2% after 3
  waves) or coverage below ~70% where the additive loop holds ~90%. Absent
  that, the additive surface + CUPED + Student-t remains the recommended
  default — with the §1 arguments as the client-facing justification.
* **Non-goals:** per-geo response curves (`β_gc`) — that is a hierarchy, not
  an effect-scale question, and it explodes the design burden; time-varying
  effect scale (interact with `tau_t`) — revisit only after Phase 2 ships;
  multiplicative *error* (log-scale sigma) independent of the mean structure.

## 4. Summary for the impatient

The MMM's "media is multiplicative" instinct is about **confounded
observational data**; the CL loop's designed, randomized, locally-perturbed,
re-centered contrasts remove precisely that hazard, so the additive surface
estimates the decision-relevant population-average incremental response
without bias — misspecification shows up as heteroskedastic noise (efficiency)
and as positive γ (which the model already represents). The empirical
misspecification finding — decisions robust, calibration fragile, the
sequential loop erases the gap — is expected to extend, and Phase 1 measures
exactly that before any machinery is built. If the evidence disagrees, the
proportional-effects variant (Phase 2) is a small, seam-respecting change:
`mu = a_g·(1+R)`, shutoff cells as the identification anchor, one
`value_scale` number in the planner, one per-cell factor in the Fisher
weights.

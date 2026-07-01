# Continuous sequential learning — a model-free geo response-surface bandit

`mmm_framework.continuous_learning` is a self-contained Bayesian
sequential-experimentation loop that allocates continuous budget across channels
by repeatedly (1) fitting a response surface **directly from designed experiment
data**, (2) choosing the most valuable next experiment, and (3) stopping when
further testing no longer pays — **without requiring a pre-fit MMM**. The
experiment's designed cross-sectional variation identifies the surface, so the
priors inform but the data dominates.

This is the implementation of `assets/continous_learning.md` (the design guide),
adapted to fit the rest of the framework. Read that guide for the math and the
productionization path; this doc maps it to the code and records the integration
decisions.

## Where it sits

The framework already has a mature decision layer in `mmm_framework.planning`,
but everything there is a layer **on top of a fitted
`BayesianMMM`** — its EIG/EVOI/funding-line machinery consumes
`ResponseCurves` produced by `compute_response_curves(mmm, …)`. This subsystem is
the **inverse**: there is no observational model. It learns a lightweight
response surface from experiments alone and carries the posterior across waves.

| | `planning/` (model-anchored) | `continuous_learning/` (model-free) |
|---|---|---|
| Needs a fitted MMM | yes | **no** |
| Source of the surface | observational time series | designed geo experiments |
| Backend | PyMC | NumPyro / JAX |
| Decision | one-shot allocate + score next experiment | sequential loop, posterior carried across waves |
| Stopping | information-decay re-test trigger | ENBS (`E[regret]·margin·pop − cost ≤ 0`) |

The two are complementary. A team with no usable observational history can run
the model-free loop; a team with a fitted MMM can use `planning/` and fold
results back via the experiment-calibration likelihood.

## The loop

```
   ┌──────────────────────────────────────────────────────────────┐
   ▼                                                                │
 FIT posterior ─▶ SCORE & PICK (acquisition) ─▶ RUN WAVE ─▶ UPDATE  │
 (model.fit)      (planner.thompson_wave,        (designed   data ──┘
                   marginal_roas, knowledge_      holdouts)
                   gradient)                          │
   ▲                                                  ▼
   └────────────────────────────────  STOP?  (planner.expected_regret + enbs)
```

Each pass is a *wave*: a designed batch of geo cells run for a fixed window. The
posterior is carried across waves by refitting on **all** accumulated data
(`LearningState.fit`), so every wave borrows strength from all prior data.

## Modules

| Module | Role |
|---|---|
| `surface.py` | The differentiable JAX response surface — the **single source of truth**. `R(s) = Σ β_c·f_c(s_c) + Σ_{c<c'} γ_{cc'} f_c f_c'`. Used by the likelihood, the DGP, **and** the allocator (which `jax.grad`s it), so the optimizer can never disagree with the fitted surface. The activation `f_c` is **pluggable** (an `ACTIVATIONS` registry): any smooth, monotone, saturating curve with `f(0)=0` and a finite gradient. `hill` (default), `logistic` (`f=1−e^{−λs}`, concave), and `hill_mixture` (`w·Hill+(1−w)·Hill`, a two-phase misspecification stress test) ship; add a family with a registry entry + a `_sample_activation_shape` case. `surface_value` / `surface_over_rows` evaluate any family. |
| `model.py` | The NumPyro generative model + priors (guide §4.2), sign-informed interaction priors (`PAIR_SIGNS`, guide §4.3), the `Posterior` container, `fit`, and `demote_channel` / `probe_pairs_excluding` for non-randomizable channels (guide §5.4). |
| `design.py` | `central_composite(center, delta, probe_pairs)` (1 center + 2K axial + 2·\|probe\| off-axis + K shutoff cells) and `assign_geos` (round-robin, optional holdouts). |
| `dgp.py` | A synthetic `TrueWorld` with causal ground truth, `simulate_panel` (the recovery harness), `simulate_wave` (later waves over the same geos), and `make_world` (a worked 4-channel world). Evaluates the **same** `surface` functions the model fits. |
| `planner.py` | The allocator (`allocate_under_sample`, multi-start SLSQP on the non-concave surface), `thompson_wave` / `recommend_allocation`, the funding line `marginal_roas`, the stopping rule `expected_regret` + `enbs` / `should_stop`, and decision-aware EVSI `knowledge_gradient`. |
| `preprocess.py` | Baseline realism (guide §9.3/9.4): `adstock_panel` / `adstock_prepass` (geometric-adstock the spend series, reusing `transforms.adstock`; `dgp.simulate_panel(adstock_alpha=…)` adds carryover) and `cuped_adjust` / `cuped_covariate` (CUPED variance reduction, `1 − ρ²`). |
| `acquisition.py` | Fast acquisition with **no MCMC** (guide §9.1/9.2): `laplace_knowledge_gradient` (Gaussian-linear EVSI), `design_eig` (`target="all"` D-optimal / `target="gamma"` D_s-optimal pure EIG), and the shared `theta_moments` / `design_information` Fisher machinery. |
| `loop.py` | `LearningState` (carry the posterior across waves) and `run_closed_loop` (the end-to-end demo / closure test); `due_for_retest` reuses `planning.eig.reexperiment_due`. |

## Data contract (guide §3)

A **long (tidy) geo-week panel**, one row per (geo, week):

| field | shape | meaning |
|---|---|---|
| `spend` | `(N, K)` | per-channel **scaled** spend (divide each channel by a fixed reference constant — never a cluster mean) |
| `geo_idx` | `(N,)` | 0-based geo index |
| `n_geo` | int | number of geos |
| `y` | `(N,)` | KPI in **natural units** (never normalized/centered/logged) |

**Identification is non-negotiable** (guide §3.2): the panel must contain a
**pre-period** where every geo shares the status-quo `center` allocation (this
pins each geo intercept and breaks the baseline↔incremental collinearity) and a
**test-period** of designed cross-sectional variation (the CCD cells). Fit on
observational spend with no designed variation and the output is not causal — the
loop degenerates to MMM with the usual confounding. `simulate_panel` builds the
pre/test split in; a real ingestion must provide it.

## Framework alignment

* **Hill convention.** `surface.activation` is `s^α / (κ^α + s^α)` — exactly the
  framework's `SaturationType.HILL` (`x^slope / (x^slope + sat_half^slope)`) with
  `slope = α`, `sat_half = κ`. A continuous-learning posterior is therefore
  directly comparable to a `BayesianMMM` Hill fit on the same channel.
* **Information decay.** `loop.due_for_retest` reuses
  `planning.eig.reexperiment_due` (`σ_eff²(t) = σ_post² exp(λt)`,
  `λ = ln2 / half_life`), so a continuous-learning program and a model-anchored
  program agree on when evidence has gone stale.
* **Self-contained.** Following the guide, the DGP, design, and planner live in
  this package rather than coupling to `synth/` — the bandit is meant to run
  without the MMM stack, and the "one JAX surface" property only holds if the DGP
  evaluates the same function the planner differentiates.

## Acquisition & stopping

* **Thompson wave** (`thompson_wave`) — solve the allocation for each posterior
  draw → a posterior over the optimal split. The mean is the recommendation; the
  spread is the exploration signal. The surface is non-concave (negative γ), so
  the allocator multi-starts and keeps the best.
* **Funding line** (`marginal_roas`) — posterior of `value·∂R/∂s_c` at the
  recommendation; a channel is funded where `P(value·∂R/∂s_c > 1) > 0.5`.
* **Expected regret / ENBS** (`expected_regret`, `enbs`) —
  `regret_d = profit(best-for-draw-d under d) − profit(consensus under d) ≥ 0`
  (warm-started from the consensus so it cannot go negative). Stop when
  `E[regret]·margin·population − wave_cost ≤ 0`.
* **Knowledge gradient** (`knowledge_gradient`) — one-step-lookahead EVSI for
  scoring candidate *test* designs. The expensive path: `refit_fn` runs a NUTS
  chain per fantasy (`refit_fn_from_data` builds one). In production swap it for
  a Laplace/conjugate update (guide §9.1).

## Validation (guide §8)

`tests/test_continuous_learning.py` exercises the three feasibility gates:

1. **Recovery** (`test_recovery_*`) — fit a known world; the main-effect ordering
   is preserved (Spearman ≥ 0.8, strongest channel correct) and the sign-informed
   synergy **signs** recover (cannibalization negative, complementarities
   positive). Magnitudes are prior-sensitive — assert signs, not magnitudes.
2. **Prior-sensitivity audit** (`test_prior_sensitivity_audit_gamma_scale`) — a
   prior-dominated (weak, true≈0) synergy's posterior spread grows with
   `gamma_scale`; flag such pairs as sign-reliable / magnitude-assumed.
3. **Closure & stopping** (`test_closure_and_stopping`) — `E[regret]` shrinks as
   the posterior is carried across waves, the recommendation tracks the
   truth-optimal profit (gap < 10%), and the ENBS rule fires before `max_waves`.

Worked end-to-end demo: `examples/ex_continuous_learning.py` (terminal) and
`nbs/continuous_learning.ipynb` — a baked, plot-rich visual walkthrough (the
response surface and its synergy contours, the central-composite design, β/γ
recovery forests, fitted-vs-true response curves, the Thompson allocation
posterior + funding line, the closed-loop regret/ENBS/convergence dashboard, the
`gamma_scale` prior-sensitivity audit, the adstock/CUPED and Laplace-KG/pure-EIG
sections, and the **acquisition & uncertainty surface**). Re-author + re-bake
with `nbs/build_continuous_learning.py` (header has the exact commands).

The acquisition/uncertainty visualization also stands alone:
`nbs/build_acquisition_viz.py` fits a posterior, evaluates the mean-profit,
uncertainty (`value·SD[R]`), and UCB-acquisition surfaces over a 2-D allocation
slice, marks the exploit vs acquisition optima (they differ — the acquisition
point is pulled toward the under-probed region), and writes a PNG + standalone
HTML to `nbs/artifacts/`. `nbs/build_acquisition_animation.py` runs the loop
wave by wave and stitches the per-wave surfaces into a GIF
(`nbs/artifacts/continuous_learning_acquisition.gif`) — colour ranges held fixed
so the **uncertainty surface shrinks** and the optima drift toward the truth as
tests accumulate, over the generic social-network channels
`["Chatter", "Pulse", "Orbit", "Vibe"]` (slice = Pulse × Orbit). It defaults to
**18 small, lean, noisy waves** (few cells + few geos each, ~15 cells via
`ACQ_PROBE="slice"`, `N_GEO=34`, `noise=1.0`) so each wave learns little and the
trust region **keeps hunting** — a **gentle, non-static** convergence rather than
locking early and sitting still. It also **tracks the experiment history**: the
probed central-composite cells and the recommendation trajectory are overlaid on
the uncertainty panel (the parameters searched), and a readout panel logs each
wave's observed incremental KPI with its observation uncertainty (±SE). A second
**hard-problem** variant (`ACQ_WORLD="hard"` →
`continuous_learning_acquisition_hard.gif`) gives the two slice channels a strong
audience-overlap **cannibalization** (a ridge in the response surface, close main
effects, slow saturation) so learning is genuinely hard — the exploit optimum
wanders along the ridge and convergence is slow and non-monotone. All cadence
knobs are env-overridable (`ACQ_N_WAVES`, `ACQ_T_TEST`, `ACQ_N_GEO`, `ACQ_NOISE`,
`ACQ_PROBE` (`all`/`slice`/`none`), `ACQ_DELTA`, `ACQ_WORLD` (`easy`/`hard`),
`ACQ_TAG`, …) so alternative cadences render side-by-side.

## When the response family is wrong (misspecification)

A pluggable activation invites the honest question: what if *none* of the
available families is the true one? We stress-tested this with a
`make_world_hill_mixture` DGP whose per-channel truth is a **weighted sum of two
Hills** — an early, steep (low-κ, high-α) component plus a later, gentler one — a
two-phase shape a single Hill can only average over and a logistic (concave, no
inflection) cannot represent at all. The same mixture-truth panel was fit with
the correct `hill_mixture`, a `hill` (mild misspecification), and a `logistic`
(severe). The result separates two things that are usually conflated:

* **Decisions are robust.** The recommended allocation captured ~99% of the true
  optimum's profit under *every* family (profit gap: Hill 0.9%, logistic 1.4%,
  mixture 0.9%). Near an interior optimum the profit surface is flat, and any
  smooth monotone-saturating curve fit to the probed cells reproduces the *local
  marginal ordering* — which is all the allocator needs.
* **Calibration is not.** The marginal-ROAS 90% credible interval covered the
  true value 4/4 channels for the (widest) mixture, 3/4 for the single Hill, and
  only 2/4 for the logistic — which was also the **narrowest**. A misspecified
  model does not know it is wrong, so it under-states its own uncertainty:
  confidently biased exactly where the experiments did not probe (see the
  anchored response-curve recovery, `nbs/artifacts/continuous_learning_misspec.png`).
* **Non-convergence is a tell.** A single Hill fit to two-Hill data frequently
  fails to mix (R̂ ≈ 1.5): there is no single (κ, α) that reconciles all the
  cross-sectional cells, so the sampler wanders. Poor R̂ / systematic residuals
  across the CCD cross-section are the practical signal that the family is too
  rigid — the cue to widen the activation, not to trust the tight intervals.
* **The loop erases the mild misspecification.** Because each wave re-probes
  *locally* over the same geos (fixed `a_geo`) and refits on all accumulated data,
  the loop never commits to a far extrapolation. Running the real accumulating
  loop (`LearningState` + `simulate_wave`) under the mild misspecification, the
  profit gap converged 0.9% → 0.5% → 0.2% → 0.3% — **tracking the correctly-
  specified mixture loop** (0.9% → 0.5% → 0.1% → 0.1%) to a fraction of a percent,
  with healthy R̂ throughout. Local sequential experimentation makes the wrong
  family nearly irrelevant to the *decision trajectory*. (An earlier draft that
  re-drew geo baselines each wave instead of holding `a_geo` fixed diverged — a
  reminder that the loop's guarantee rests on a **stable geo set**; conflating two
  intercept draws under one `geo_idx` corrupts the fit.)

**Operating rule:** trust the *ranking* and the funded set; distrust the
channel-by-channel magnitudes and their intervals. Fit the most flexible
activation you can identify (it is honest-but-uncertain rather than
tight-but-wrong), and treat R̂ / cross-section residuals as a misspecification
alarm. Verified in `tests/test_continuous_learning.py`
(`test_misspecified_single_hill_still_makes_a_near_optimal_decision`) and
demonstrated in notebook §14 — including an animation
(`nbs/build_misspecification_animation.py`) that runs the correct-mixture and
wrong-single-Hill loops side by side: the profit-gap trackers overlap and descend
(the *decision* converges) while the wrong family's credible-interval width shrinks
*below* the correct one and its coverage collapses (calibration stays confidently
wrong).

## Failure modes & gotchas (guide §11)

* **β attenuation / β↔γ trade-off** → include the shutoff cells; widen `delta`.
* **κ/α weakly identified** → expected from a narrow local design; trust the
  local gradient within a wave, not the global Hill shape. Global shape
  accumulates only as the trust region moves across waves.
* **Prior-dominated γ** (especially demoted-channel pairs) → never present those
  magnitudes as data-driven.
* **Outcome transformed** → never normalize/center/log `y`; only scale spend, by
  a fixed global constant (not a cluster mean).
* **No designed variation** → output is not causal.

## Deferred

The productionization upgrades from guide §9 are now implemented and
demonstrated in the notebook (§7–9): the **adstock pre-pass + CUPED**
(`preprocess.py`), the **fast Laplace knowledge-gradient**
(`acquisition.laplace_knowledge_gradient`), and the **pure-EIG D/D_s-optimal**
acquisition (`acquisition.design_eig`). Still deferred:

* **Agent / API / UI wiring** — agent tools (`start_learning_program`,
  `design_next_wave`, `record_wave_readout`, `recommend_allocation`,
  `check_stopping`), REST endpoints + non-blocking jobs (reuse
  `_run_model_op_job` / `_spawn_job_task`), sessions persistence
  (`learning_program` / `learning_wave` tables), and a React "Learning Programs"
  page.
* **Laplace KG *inside the loop*** — the Laplace update is available as a
  standalone acquisition; wiring it into `knowledge_gradient` / the closed loop
  (so wave selection never pays for a NUTS refit) is not done.
* **Richer baseline** — a national time effect `τ_t`, and a per-geo (rather than
  pre-pass) adstock parameter in the graph.

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
| `design.py` | `central_composite(center, delta, probe_pairs)` (1 center + 2K axial + 2·\|probe\| off-axis + K shutoff cells) and `assign_geos` (shuffled round-robin, optional holdouts; pass a per-geo `baseline` for **stratified/blocked randomization** — geos sorted by baseline, each block of `n_cells` gets a random permutation of the cells, holdouts at EXACTLY `n_holdout` evenly spaced positions in baseline-sorted order so the counterfactual spans the range and honors the requested count). |
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
* **KG-driven design selection** (`loop.select_next_design`) — the Laplace KG
  is now WIRED into design selection: candidates
  `central_composite(center, δ, probe_set)` over a `candidate_deltas` grid (×
  optional probe sets) are scored with
  `acquisition.laplace_knowledge_gradient` under **one shared seed** (common
  random numbers, so the Monte-Carlo argmax doesn't flap) and the EVSI-best
  design runs. Opt-in everywhere the default stays byte-stable:
  `run_closed_loop(use_laplace_kg=True, candidate_deltas=…)` (recorded on the
  next `WaveRecord` as `kg_used`/`chosen_delta`), `service.design_wave(
  optimize=True)` → agent tool `design_learning_wave(optimize=true)` → REST
  `POST …/design-wave {"optimize": true}` (the response's `kg` key carries the
  per-candidate scores). The fast acquisition works for **any registered
  activation** (the `ThetaMap` in `acquisition.py` moment-matches in an
  unconstrained reparameterization — log for positive parameters, scaled
  logit for bounded ones, sign-aware log for `neg`/`pos` synergies — so every
  fantasy maps back to valid parameters with no clipping) and for **any
  fitted observation family** (per-cell GLM Fisher weights via
  `observation_unit_info`: `1/σ²` Gaussian, `(ν+1)/((ν+3)σ²)` Student-t,
  `sigmoid(η)²/(m+m²/φ)` softplus-link NegBinomial). The selector still falls
  back to the fixed-`delta` design with an explicit reason/warning for an
  activation with no `SHAPE_TRANSFORMS` entry, an unknown likelihood family,
  or a posterior missing its observation sites (a summaries-only fit: its θ
  lives on the KPI's natural scale under `prior_scaling="auto"`, so no fixed
  noise guess is meaningful — Fisher weights would make every candidate score
  identically while claiming `kg_used=True`) and for any
  non-finite candidate score (NaN would silently argmax to the first
  candidate). The REST body bounds `kg_n_outcomes` (8–256) and
  `candidate_deltas` (≤8 entries in (0, 1.5]) and the endpoint runs the
  scoring via `asyncio.to_thread`, so one request cannot freeze the API.
* **Stratified geo assignment** — `assign_geos(baseline=…)` blocks the
  geo→cell randomization on a per-geo covariate. `service.design_wave`
  supplies the accumulated per-geo mean KPI once the program has ingested data
  (`assignment.stratified_on = "accumulated_kpi"`; `stratify=False` opts out);
  the DGP/loop can stratify on the true `a_geo`
  (`simulate_panel`/`simulate_wave`/`run_closed_loop` `stratify…=True`, default
  OFF to keep the pinned seeded gates byte-identical).

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

## Count KPIs — the NegativeBinomial likelihood (opt-in)

`fit(likelihood="negbinomial")` (default `"normal"`, byte-identical graph) swaps
the panel observation for `y ~ NegativeBinomial2(softplus(mu), phi)` with
concentration `phi ~ LogNormal(log 30, 2)` replacing the `sigma` site (the
log-scale tail matters: NB2 variance is `m + m²/phi`, so near-Poisson behavior
at row mean `m` needs `phi ≳ m` — an exponential-tailed `Gamma(2, 0.1)` prior
caps `phi` at O(10²) regardless of the data and forces a variance floor on
high-mean weakly-overdispersed KPIs, while the LogNormal keeps the same O(10)
center but lets the data buy `phi` in the thousands); the identity-link
mean `mu = a_geo[geo_idx] + R(spend) (+ tau)` is unchanged (softplus is only a
positivity guard), so β keeps its per-scaled-unit meaning and the planner
readouts (Thompson wave, funding line, regret/ENBS) run untouched — they read
only the surface parameters. **Link caveat**: the marginal readouts
(`marginal_roas`, `expected_regret`, free-mode allocation) differentiate the
LATENT surface `R`, but the observable count mean is `softplus(mu)`, whose
derivative is `sigmoid(mu)` — so the count-scale marginal response is
`sigmoid(mu)·dR/ds`. Since `sigmoid(mu) ≈ 1` whenever `mu >> 1` (typical
counts), the readouts are asymptotically exact; near `mu ≈ 0` (per-row means
of a few counts) they OVERSTATE the count-scale marginal response by up to
`1/sigmoid(mu)` (an optimistic funding line). `fit()` warns loudly when
`mean(y) < 20`. The fixed-budget Thompson argmax is unaffected — softplus is
monotone, so per-draw optima are invariant. `y` must be non-negative integer counts
(`_validate_panel` enforces it loudly); `preprocess.cuped_adjust` is
**incompatible** (it mutates `y` to non-integer/negative values). The summary
block stays **Gaussian for all** likelihoods: a `lift ± se` readout is already
a normal-approximation aggregate. The previously Gaussian-only paths now
dispatch on the family instead of raising: `planner.knowledge_gradient`
fantasizes from the fitted family (`_fantasy_outcomes` — Gaussian, Student-t
with the posterior-mean `nu`, or gamma–Poisson counts with the posterior-mean
`phi`; build the refit closure with the matching
`refit_fn_from_data(likelihood=…)`), and the Laplace/EIG acquisition uses the
family's per-cell GLM Fisher weights (`observation_unit_info` — for NB,
`sigmoid(η)²/(m+m²/φ)` through the softplus link at the design cell's
predicted mean). An *unknown* family still raises `NotImplementedError`
rather than returning plausible nonsense. Threaded end-to-end like `activation`:
`LearningState.likelihood`, `config["likelihood"]` in `service.new_program_state`,
serialization with `"normal"` back-compat defaults, and a DGP
`noise_family="negbinomial"` (gamma–Poisson around `softplus(mu)`, world
`phi_true`) for the recovery gate
(`test_negbinomial_recovery_counts_world`).

## Heavy-tailed KPIs — the Student-t likelihood (opt-in)

`fit(likelihood="studentt")` keeps the Gaussian's location/scale structure and
adds a learned tail df `nu ~ Gamma(2, 0.1)` (the Juárez–Steel robustness
prior: mean 20 ≈ near-Gaussian a priori, real mass below `nu ~ 5` so a few
wild geo-weeks buy heavy tails instead of dragging `beta`/`gamma`);
`y ~ StudentT(nu, mu, sigma)` with `sigma` the t SCALE (sd is
`sigma·sqrt(nu/(nu−2))`). Everything reading only the surface parameters is
untouched; the acquisition discounts the per-observation Fisher information by
`(nu+1)/(nu+3)` (the classic heavy-tail efficiency factor, → 1 as `nu → ∞`).
`y` stays continuous/natural-scale (CUPED **is** compatible, unlike NB). DGP:
`noise_family="studentt"` with world `nu_true`. Recovery gate:
`test_studentt_recovery_heavy_tailed_world` (t(3) residuals; beta ordering
recovered, posterior `nu` concentrates well below the prior mean).

## National time effect τ_t (opt-in)

`fit(time_effect="national")` (default `"none"`, no new sites) adds a
**zero-centered hierarchical** national per-period shock to the panel mean:
`sigma_tau ~ HalfNormal(y_scale)`, `tau_t ~ Normal(0, sigma_tau)`,
`mu += tau[period_idx]`. Zero-centered partial pooling, NOT fixed effects — a
free τ level is exactly collinear with the intercept hyper `A`. The data
contract gains an optional `period_idx (N,)` (0-based int, validated as loudly
as `geo_idx` — JAX clamps out-of-bounds indices silently). **Producers always
provide, consumers opt in**: `simulate_panel` / `simulate_wave` always return
`period_idx` (`np.repeat(arange(T), n_geo)`, matching the week-major row order)
plus optional true shocks via `tau_scale`; `LearningState.ingest` accumulates a
GLOBAL period index only when `time_effect != "none"` (each wave's local
0..T−1 shifted by the accumulated max+1 — exact offsets, or two waves' shocks
alias onto one τ) and refuses mixed with/without-period waves;
`ingest_wave_rows(period_col=…)` maps period labels to sorted local indices —
`period_col` is exposed end-to-end (the REST wave-ingest body and the
`record_learning_wave` agent tool both take it), and when omitted the service
auto-detects a `period`/`week`/`date` column with a warning note (no
recognizable column → a loud error naming `period_col`).
`state.npz` carries `panel_period_idx` (presence-guarded on load; a state
using the new semantics — non-default likelihood/time_effect or a persisted
`period_idx` — is stamped `schema_version` 2 so OLD readers refuse it loudly
instead of silently refitting a count/time-effect program as Gaussian/no-tau,
while default-config states keep stamping 1 and stay loadable both ways).
Summary observations need **nothing**: a national per-period constant cancels
in the lift difference exactly as the geo intercept does — and a
**summaries-only fit** (`{"summaries": [...], "n_geo": 0}`) therefore needs no
`period_idx` at all: there are no panel rows for τ to index, so no tau sites
are sampled and `fit(time_effect="national")` succeeds. `refit_fn_from_data`
raises for `time_effect != "none"` (KG fantasy rows have no period identity).
Recovery gate: `test_national_time_effect_recovery` (corr(τ̂, τ_true) > 0.7
with β recovery intact).

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
acquisition (`acquisition.design_eig`). The Laplace KG is also **wired into
design selection** (`loop.select_next_design`, opt-in via
`run_closed_loop(use_laplace_kg=True)` / `design_wave(optimize=True)`), and
`assign_geos` supports **stratified/blocked geo randomization** on a per-geo
baseline (the service stratifies on the accumulated KPI by default).

**Agent / API wiring has now shipped** (Phase B of
`technical-docs/continuous-learning-wiring.md`):

* **Service layer** — `continuous_learning/service.py`: dollars-at-the-boundary
  config validation (`new_program_state`, arms-aware), state-file IO
  (`<workspace>/projects/<pid>/learning/<prog>/state.npz`), `design_wave`,
  `ingest_wave_rows` / `rows_from_csv`, `import_experiment_summaries` (the
  model-free past-experiment bridge), and `fit_and_plan` — ONE Thompson pass
  producing the pinned SNAPSHOT (per-dollar mROAS funding line with
  FUND/HOLD/CUT verdicts, ENBS in dollars, response curves, prior-domination
  and shape-identification flags).
* **Sessions persistence** — `learning_programs` / `learning_waves` tables +
  CRUD in `api/sessions.py`; a nullable `subchannel` column on the
  `experiments` registry (threaded through `log_experiment` /
  `plan_experiment` / `record_experiment_readout` and `POST /experiments`).
* **REST endpoints** — `GET/POST/DELETE /projects/{pid}/learning-programs…`,
  sync `…/design-wave`, non-blocking `…/waves` + `…/fit` jobs on the synthetic
  thread `__learnjobs__{pid}` (bespoke model-free worker `_run_learning_job` —
  it never loads an MMM), and the `…/jobs/{job_id}` poller.
* **Agent tools** — `agents/learning_tools.py`: `start_learning_program`,
  `import_past_experiments`, `design_learning_wave`, `record_learning_wave`,
  `get_learning_program_status`, `check_learning_stopping` (all spine tools;
  prompt guidance in `MMM_SYSTEM_PROMPT` + a `_LIBRARY_MENU` section).
* **React "Sextant" page** (`/learning`) — built in Phase C against the same
  §3.1 SNAPSHOT schema and §3.5 endpoint paths.
* **Docs pages** (Phase D) — `docs/continuous-learning.html` /
  `-math.html` gained the past-experiment ingestion (summary-observation
  likelihood), creative/keyword arms, and in-the-app (Sextant + agent tools)
  sections, plus the interactive surface/ENBS calculators.

Still deferred:

* **Laplace KG *inside the loop*** — the Laplace update is available as a
  standalone acquisition; wiring it into `knowledge_gradient` / the closed loop
  (so wave selection never pays for a NUTS refit) is not done.
* **Richer baseline** — the national time effect `τ_t` is now in the graph
  (opt-in, see above); a per-geo (rather than pre-pass) adstock parameter is
  still deferred.
* **Heterogeneous `spend_ref` budget constraint** — the planner enforces the
  fixed budget in scaled units; with a non-uniform per-channel `spend_ref`
  that is a reference-weighted budget, not the exact dollar simplex (the
  snapshot carries a warning). The default `spend_ref` — one global constant,
  the mean of the channel centers — is exact.
* **Total-lift constraints across an arm group** — a channel-level readout on a
  split parent is skipped by the evidence converter rather than being
  distributed over its arms.

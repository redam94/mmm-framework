# Continuous Sequential Learning — Python Implementation Guide

Technical documentation for the geo response-surface bandit: a Bayesian
sequential-experimentation system that allocates continuous budget across a set
of channels by repeatedly (1) fitting a response surface from experimental data,
(2) choosing the most valuable next experiment, and (3) stopping when further
testing no longer pays.

This document maps the framework's concepts (EIG, EVOI/EVSI, Bayesian
optimization, ENBS) to concrete Python, specifies the data contracts and module
APIs, and lays out the path from the reference prototype to production.

---

## 1. Architecture

### 1.1 The loop

```
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         │
   ┌─────────┐    ┌──────────────┐    ┌────────────┐   ┌────────┐ │
   │  FIT     │──▶│  SCORE & PICK │──▶│  RUN WAVE   │─▶│ UPDATE │─┘
   │ posterior│    │ (acquisition)│    │ (holdouts) │   │  data  │
   └─────────┘    └──────────────┘    └────────────┘   └────────┘
        ▲                  │
        │                  ▼
        │            ┌───────────┐
        └────────────│  STOP?     │  ENBS ≤ 0  →  halt
                     │  (ENBS)    │
                     └───────────┘
```

Each pass is a *wave*: a designed batch of geo/audience cells run for a fixed
window. The posterior is carried across waves, so every wave borrows strength
from all prior data.

### 1.2 Concept → code

| Framework concept | What it is | Reference implementation |
|---|---|---|
| Response surface (belief) | Posterior over how spend drives outcome | `model.model` (NumPyro) → posterior dict |
| Bayesian optimization | Choose next point via a surrogate + acquisition | `planner.thompson_wave` (+ `knowledge_gradient`) |
| Thompson sampling | Sample a world, act optimally, repeat | `thompson_wave` → `allocate_under_sample` per draw |
| EVOI / EVSI | Value of an experiment *to the decision* | `knowledge_gradient` (EVSI for the allocation argmax) |
| EIG | Pure information gain | *not built-in* — see §9.2 (extension) |
| ENBS (stopping) | EVSI − cost ≤ 0 → stop | `expected_regret` × margin vs. wave cost (§7.4) |
| Funding line | `value·∂R/∂s_c > 1` per channel | `marginal_roas` |
| Counterfactual | Geo/audience holdout baseline | pre/test split in `design.simulate_panel` |

### 1.3 Modules

```
geo_rsm/
  model.py      response surface (JAX) + NumPyro generative model + priors
  design.py     central-composite geo design + synthetic DGP (recovery / fantasies)
  planner.py    allocator, Thompson wave, funding line, KG, ENBS stopping
  demo.py       end-to-end: simulate → fit → recover → plan → stop
  fit_once.py   cache a posterior for fast planner iteration
```

One JAX function, `model.incremental`, is the single source of truth for the
response surface; it is used inside both the NumPyro likelihood and the
`planner` allocator, so the optimizer can never disagree with the fitted model.

---

## 2. Installation

```bash
pip install numpyro jax scipy numpy pandas      # CPU is sufficient at geo scale
# optional, for figure rendering / notebooks
pip install matplotlib arviz
```

JAX defaults to CPU. At geo scale (tens–hundreds of geos × tens of weeks) NUTS
runs in seconds–minutes on CPU; GPU is unnecessary and often slower due to
transfer overhead.

```python
import jax; jax.config.update("jax_platform_name", "cpu")
```

---

## 3. Data contract

The fit consumes a **long (tidy) geo-week panel**. One row per (geo, week).

| field | shape / type | meaning |
|---|---|---|
| `spend` | `(N, K)` float | per-channel spend for that geo-week, **scaled** (see §3.1) |
| `geo_idx` | `(N,)` int | 0-based geo index, contiguous `0..n_geo-1` |
| `n_geo` | int | number of geos |
| `y` | `(N,)` float | KPI in **natural units** (revenue or conversions) |

`K = len(model.CHANNELS)`; the default channel order is
`["pmax", "search", "reddit", "amazon"]`.

### 3.1 Scaling conventions (important)

- **Spend is divided by a global per-channel reference constant** before the
  fit (e.g. each channel's median or a fixed planning spend). This is a unit
  choice, not a transformation: it keeps the half-saturation prior on `kappa`
  sane (`κ ~ O(1)`). The reference is a *fixed constant per channel*, never a
  cluster-specific mean — dividing by per-geo means induces the ratio/between-
  signal-erasure pathology and is prohibited.
- **The outcome `y` is never normalized, centered, or logged.** The model is
  additive in natural units so that incrementality and marginal ROAS stay
  interpretable and the KKT funding-line algebra holds. Store the spend
  reference vector alongside the data so you can map recommended *scaled*
  allocations back to dollars.

### 3.2 Identification requirement (non-negotiable)

The model is causal **only because spend is experimentally assigned**. The DGP
and any real ingestion must provide:

1. A **pre-period** where all geos share a common (status-quo) allocation. This
   pins each geo's baseline intercept `a_geo` (CUPED-style) and breaks the
   within-geo collinearity between baseline and incremental response.
2. A **test-period** where geos receive *designed*, cross-sectionally varied
   allocations (the central-composite cells of §5). This variation identifies
   the response surface.

Fit on observational spend (no holdout, no designed variation) the model is just
MMM with the usual confounding; the priors will carry everything and the output
is not causal.

---

## 4. Layer 1 — Response model (`model.py`)

### 4.1 The surface

Per geo-week, incremental response to a scaled spend vector `s ∈ R^K`:

```
f_c(s_c) = s_c^{α_c} / (κ_c^{α_c} + s_c^{α_c})          # Hill activation ∈ [0,1]
incremental(s) = Σ_c β_c · f_c(s_c)  +  Σ_{c<c'} γ_cc' · f_c(s_c) · f_c'(s_c')
```

`activation()` returns the β-stripped Hill fraction; `incremental()` adds the
main effects and the upper-triangular interaction block. The cross-partial is
`∂²/∂s_c∂s_c' = γ_cc' · f_c'(s_c) · f_c'(s_c')`, so **`γ` carries the sign of the
synergy** (positive = complementarity, negative = cannibalization).

```python
def incremental(spend, beta, kappa, alpha, gamma):   # gamma: (K,K) upper-tri
    f = activation(spend, kappa, alpha)
    return jnp.sum(beta * f) + jnp.sum(gamma * jnp.outer(f, f))
```

### 4.2 Priors

| parameter | prior | rationale |
|---|---|---|
| `beta` `(K,)` | `HalfNormal(beta_scale=1)` | non-negative channel ceilings |
| `kappa` `(K,)` | `LogNormal(0, 0.5)` | positive half-saturation, O(1) in scaled units |
| `alpha` `(K,)` | `TruncatedNormal(2, 0.5, [0.5, 5])` | shape; informative (weakly identified locally) |
| `gamma` (per pair) | sign-informed (below) | encodes domain knowledge |
| `A`, `sigma_a` | `Normal(0,5)`, `HalfNormal(2)` | geo-intercept hyperpriors |
| `a_geo` `(n_geo,)` | `Normal(A, sigma_a)` | baseline random intercept |
| `sigma` | `HalfNormal(1)` | observation noise |

Likelihood: `y ~ Normal(a_geo[geo_idx] + incremental(spend), sigma)`.

### 4.3 Sign-informed interaction priors

`PAIR_SIGNS` maps each channel pair to a prior family:

```python
PAIR_SIGNS = {
    (0,1): "neg",   # pmax  × search  → shared-inventory cannibalization,  γ ≤ 0
    (0,2): "weak",  # pmax  × reddit
    (0,3): "zero",  # pmax  × amazon  → ~0, leave to prior
    (1,2): "pos",   # search × reddit → demand-gen complementarity,        γ ≥ 0
    (1,3): "weak",  # search × amazon
    (2,3): "pos",   # reddit × amazon → demand-gen complementarity,        γ ≥ 0
}
```

| sign | prior on γ | sampled as |
|---|---|---|
| `neg` | `γ ≤ 0` | `-HalfNormal(gamma_scale)` |
| `pos` | `γ ≥ 0` | `HalfNormal(gamma_scale)` |
| `zero` | `~0` | `Normal(0, 0.05·gamma_scale)` |
| `weak` | weakly informative | `Normal(0, gamma_scale)` |

`gamma_scale` (default `0.8`) is the most consequential knob: too tight and a
strong true synergy is crushed toward the prior; the **sign recovers robustly,
the magnitude is prior-sensitive** (audit it, §8.2).

### 4.4 API

```python
model.model(spend, geo_idx, n_geo, y=None,
            beta_scale=1.0, gamma_scale=0.8, pair_signs=None)
```

- `pair_signs` overrides `PAIR_SIGNS` (e.g. Amazon demotion, §5.4).
- `y=None` draws from the prior predictive (use for prior checks).

Posterior site names: `beta`, `kappa`, `alpha`, `A`, `sigma_a`, `a_geo`,
`sigma`, and one `gamma_<ci>_<cj>` per pair (deterministic for `neg`/`pos`,
sampled for `zero`/`weak`).

---

## 5. Layer 2 — Experimental design (`design.py`)

### 5.1 Central-composite geo design

```python
design.central_composite(center, delta, probe_pairs)
```

Returns a stack of scaled allocations (rows = cells):

- **1 center cell** — the current operating allocation (trust-region center).
- **2K axial cells** — each channel ±`delta` (gives the gradient / main effects).
- **2 off-axis cells per pair in `probe_pairs`** — two channels moved *jointly*
  ±`delta` (the only way to recover `∂²R/∂s_c∂s_c'`; pick the decision-pivotal
  pairs).
- **K shutoff cells** — one channel set to 0 (`f_c → 0`). These isolate the
  remaining terms and **break the β/γ collinearity**; without them β attenuates.

### 5.2 Geo assignment

```python
design.assign_geos(design, n_geo, rng)   # round-robin, shuffled → (n_geo, K)
```

Round-robin keeps cells balanced across geos. In production, stratify the
assignment on pre-period KPI level/variance (matched-market style) rather than
pure round-robin, to minimize baseline imbalance.

### 5.3 Synthetic DGP (recovery + fantasies)

```python
design.simulate_panel(true, center, n_geo=80, t_pre=6, t_test=10,
                      delta=0.6, noise=0.6, seed=0)
```

`true` is a dict with `beta, kappa, alpha` (each `(K,)`) and `gamma`
(`len(PAIRS),`). Pre-period spend = `center` for all geos; test-period spend =
each geo's assigned cell. Returns the data-contract dict plus `design`,
`geo_alloc`, `a_geo`. This is both the recovery harness (§8.1) and the engine
that fantasizes wave outcomes inside `knowledge_gradient`.

### 5.4 Amazon (or any non-randomizable channel) demotion

When a channel cannot be geo-co-randomized (a walled garden), demote it to
clean-room-main-effect-only:

```python
signs = model.demote_channel("amazon")              # its interactions → "zero"
probe = model.probe_pairs_excluding("amazon")       # drop its off-axis cells
design = central_composite(center, delta, probe)
mcmc.run(..., pair_signs=signs)
```

The main effect is still identified; its interactions become prior-dominated.
Flag `γ(·,amazon)` as the least-trustworthy parameter in any decision.

---

## 6. Layer 3 — Inference

```python
from numpyro.infer import MCMC, NUTS
mcmc = MCMC(NUTS(model.model), num_warmup=500, num_samples=500,
            num_chains=2, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0),
         spend=data["spend"], geo_idx=data["geo_idx"],
         n_geo=data["n_geo"], y=data["y"])
post = {k: np.array(v) for k, v in mcmc.get_samples().items()}
```

**Diagnostics:** check `mcmc.print_summary()` for `r_hat ≤ 1.01` and adequate
`n_eff`, especially on `kappa`/`alpha` (the most likely to be weakly identified
from a local design) and on prior-dominated `gamma` pairs. Divergences usually
signal a too-narrow spend range or a `kappa` prior mismatched to the scaling.

**Caching:** persist the posterior dict (`pickle`) so the planner can iterate
without refitting (`fit_once.py`). The planner only needs the posterior arrays.

---

## 7. Layer 4 — Planner / acquisition (`planner.py`)

The decision: `max_s  value·R(s) − 1ᵀs  s.t. constraints`. Two regimes:

- **fixed budget** (`mode="fixed"`): `1ᵀs ≤ B` — pure simplex.
- **free budget** (`mode="free"`): `0 ≤ s_c ≤ cap` — total is itself a decision;
  each channel self-funds until its marginal ROAS hits 1.

### 7.1 Allocator (one posterior draw)

```python
planner.allocate_under_sample(p, B, value_per_unit, x0=None,
                              n_starts=3, seed=0, mode="fixed", cap=None)
# -> (allocation (K,), profit)
```

SLSQP with analytic JAX gradient. The surface is **non-concave** (negative γ),
so the allocator multi-starts (center + optional `x0` + Dirichlet starts) and
keeps the best. `draw_params(post, d)` extracts one draw and assembles its
`(K,K)` γ matrix.

### 7.2 Thompson wave (primary acquisition)

```python
planner.thompson_wave(post, B, value_per_unit, q=300, seed=0,
                      mode="fixed", cap=None)
# -> (allocs (q,K), profits (q,))
```

For each of `q` posterior draws, solve the allocation → a posterior over the
optimal split. The spread *is* the exploration signal. The mean is the
recommended allocation; the per-draw optima seed the next wave's candidate
cells. This is the continuous-action analog of the bid-multiplier Thompson
sampling already in use.

### 7.3 Funding line

```python
planner.marginal_roas(post, alloc, value_per_unit, q=300, seed=1)
# -> (mean mROAS per channel (K,), P(above funding line) (K,))
```

Posterior of `value·∂R/∂s_c` at `alloc`. A channel is funded where
`P(value·∂R/∂s_c > 1) > 0.5`. This is the decision readout each wave.

### 7.4 Stopping — expected regret (ENBS)

```python
planner.expected_regret(post, B, value_per_unit, q=300, seed=2,
                        mode="fixed", cap=None)
# -> (E[regret], consensus alloc (K,), alloc sd (K,), sd of optimal profit)
```

`regret_d = profit(best-for-draw-d under d) − profit(consensus under d) ≥ 0`,
**guaranteed non-negative** because each draw's optimization is warm-started from
the consensus and takes the max. `E[regret]` is the profit still on the table
from posterior uncertainty.

**ENBS stopping rule:** stop when no wave's expected reallocation value clears
its cost:

```
ENBS(wave) ≈ E[regret] · margin · affected_population  −  wave_cost
stop when   max_wave ENBS(wave) ≤ 0
```

In practice: keep probing a channel/pair while its share of `E[regret]` exceeds
the cost of a wave that would resolve it; otherwise halt.

### 7.5 Knowledge gradient (decision-aware insurance)

```python
planner.knowledge_gradient(post, candidate_test_alloc, B, value_per_unit,
                           refit_fn, n_fantasy=20, seed=3)   # -> KG value
```

One-step lookahead — EVSI specialized to the allocation argmax:

```
KG(d) = E_y[ max_a profit(a | posterior updated with fantasised y) ]
        − max_a profit(a | current posterior)
```

Use it to score candidate **test** allocations (which wave to run), not just the
live allocation. It is the **expensive path**: `refit_fn(extra_spend, extra_y)`
must return an updated posterior, and the reference calls a full refit per
fantasy. In production replace that with a Laplace / conjugate linearized update
(§9.1) so many candidate designs can be scored cheaply.

---

## 8. Validation (the pilot checks)

These three checks, in order, are the feasibility gates.

### 8.1 Recovery

Simulate from known `true` params on a CCD, fit, and compare:

```python
data = simulate_panel(true, center, n_geo=90, t_pre=6, t_test=12, delta=0.7)
# ... fit ...
for (i,j), tg in zip(model.PAIRS, true["gamma"]):
    s = post[f"gamma_{model.CHANNELS[i]}_{model.CHANNELS[j]}"]
    print(model.CHANNELS[i], model.CHANNELS[j], tg, "->", s.mean(),
          np.percentile(s,[5,95]))
```

Expected: `kappa` tight; main-effect ordering preserved; pivotal-pair γ **signs**
recover robustly; magnitudes track wherever the design reaches. See `demo.py`.

### 8.2 Prior-sensitivity audit (γ)

Re-fit at `gamma_scale ∈ {0.4, 0.8, 1.6}` and report how much each γ moves.
Pairs whose posterior tracks the prior are prior-dominated — flag them as
sign-reliable / magnitude-assumed in any decision.

### 8.3 Closure & stopping

- **Closure:** a simulated wave updates the posterior, `thompson_wave` returns a
  sensible split, and the recommendation moves toward the true optimum across
  waves.
- **Stopping:** `expected_regret` shrinks as data accumulates and the rule fires
  before the budget runs out (no infinite testing, no chasing noise).

---

## 9. Productionization

### 9.1 Replace the per-fantasy refit in KG

Full NUTS per fantasy is too slow to score many designs. Swap `refit_fn` for a
**Laplace approximation** at the current MAP (Gaussian posterior update under a
local linearization of `incremental` around the operating point) or a
conjugate update on a linearized surrogate. This turns KG from minutes-per-
candidate into milliseconds and makes EVSI-based wave selection practical.

### 9.2 Add a pure-EIG acquisition (optional)

For information-only objectives, add D-optimal / D_s-optimal scoring. In the
Gaussian-linear regime the posterior precision updates as
`Λ_post = Λ_prior + σ⁻² XᵀX`; maximize `log det` of the block you care about
(use the **γ sub-block** for synergy-targeted waves — D_s-optimality). Prefer
EVSI/KG (§7.5) when the goal is the decision; use EIG only to shore up
decision-pivotal interactions the exploit-heavy Thompson waves under-probe.

### 9.3 Baseline realism

The reference baseline is a geo intercept only. For production add:

- a **national time effect** `τ_t` (iid, spline, or GP) to absorb seasonality
  and shared shocks;
- an explicit **CUPED covariate** (pre-period KPI) if you want variance
  reduction beyond what the shared `a_geo` provides — it shrinks the geo group
  size needed per MDE, a direct testing-budget lever;
- a count likelihood (`NegativeBinomial`) if `y` is conversions at low volume
  (keep the additive-mean structure via an identity-link mean, not a logged
  outcome).

### 9.4 Continuous intensity & carryover

The reference treats spend as the lever directly. If carryover matters, apply
**adstock before saturation** (geometric adstock on each channel's spend series)
and fit on the adstocked series; do not log or mean-center the outcome. If the
journey is genuinely funnel-staged, the same surface sits on top of a two-stage
mediator (Reddit → Search/Amazon) — run the terminal-KPI version first, add the
mediator once the loop is stable.

### 9.5 Orchestration

```
state = load_state()                      # posterior, history, trust-region center
while True:
    post = fit(model, state.all_data)                      # §6
    rec, _ = thompson_wave(post, B, value, mode=...)       # §7.2
    funded = marginal_roas(post, rec.mean(0), value)       # §7.3
    reg, *_ = expected_regret(post, B, value, mode=...)    # §7.4
    if reg * margin * population <= wave_cost:              # ENBS ≤ 0
        break
    probe = choose_pairs_by_KG_or_regret(post)             # §7.5 / §7.4
    design = central_composite(state.center, delta, probe) # §5
    run_wave(design)                                        # geo/audience holdouts
    state.append(collect_results()); state.recenter(rec.mean(0))
```

Persist `state` (PostgreSQL/checkpoint) with full provenance: the posterior, the
design run each wave, the recommendation, and the stop decision. Run on a cadence
matched to the holdout window (typically 2–6 weeks per wave).

### 9.6 Scaling notes

- NUTS cost grows with `n_geo + n_params`; geo scale is comfortable on CPU.
- `thompson_wave`/`expected_regret` cost is `q × n_starts` SLSQP solves; tune
  `q` (200–400) and `n_starts` (3–8) for the runtime/quality trade-off. Cache
  the posterior; vectorize allocations across draws if you need more `q`.

---

## 10. Configuration reference

| knob | location | default | effect |
|---|---|---|---|
| `CHANNELS`, `PAIR_SIGNS` | `model.py` | 4 ch / 6 pairs | channel set and synergy priors |
| `beta_scale` | `model.model` | 1.0 | main-effect ceiling prior scale |
| `gamma_scale` | `model.model` | 0.8 | interaction prior scale (audit!) |
| `pair_signs` | `model.model` | `PAIR_SIGNS` | override (e.g. demotion) |
| `delta` | `central_composite` | — | trust-region step (local gradient vs. global shape) |
| `probe_pairs` | `central_composite` | — | which cross-partials to identify |
| `t_pre`, `t_test` | `simulate_panel` | 6, 10 | baseline vs. signal weeks |
| `B`, `value_per_unit` | planner | — | budget and $/unit (sets funding line) |
| `mode`, `cap` | planner | `"fixed"`, — | fixed simplex vs. free box |
| `q`, `n_starts` | planner | 300, 3 | MC draws / multistart (runtime vs. quality) |
| `num_warmup/samples/chains` | NUTS | 500/500/2 | inference budget |

---

## 11. Failure modes & gotchas

- **β attenuation / β↔γ trade-off** → include the shutoff cells (§5.1); widen
  `delta`; check the γ prior isn't absorbing main-effect signal.
- **Negative or unstable regret** → the allocator hit a local optimum; raise
  `n_starts`, and rely on the consensus warm-start in `expected_regret`.
- **`kappa`/`alpha` non-identified** → expected from a narrow local design;
  trust the local gradient, not the global Hill shape, within a single wave;
  global shape accumulates only as the trust region moves across waves.
- **Prior-dominated γ** → especially demoted-channel pairs; never present those
  magnitudes as data-driven.
- **Outcome transformed** → never normalize/center/log `y`; only scale spend, by
  a fixed global constant (not a cluster mean).
- **No designed variation** → output is not causal; the loop degenerates to MMM.

---

## 12. Glossary

- **EIG** — expected information gain; uncertainty reduction of an experiment.
- **EVOI / EVSI** — expected value of (sample) information; value of an
  experiment *to the downstream decision*. Implemented as `knowledge_gradient`.
- **ENBS** — expected net benefit of sampling = EVSI − cost; the stopping rule.
- **Knowledge gradient** — EVSI specialized to the argmax-allocation decision.
- **Funding line** — `value·∂R/∂s_c = 1`; fund channels above it.
- **Trust region** — the local neighborhood of the current allocation a wave
  measures; recentered each wave.
- **Holdout / counterfactual** — geo or audience group kept at status quo to
  estimate what would have happened anyway.

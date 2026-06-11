# MMM Robustness Report — Convergence, Fit-Time & Failure Modes

**Scope.** This report answers three questions about `BayesianMMM`:

1. **Convergence** — does the sampler actually converge, where does it struggle, and is the convergence *gate* trustworthy?
2. **Fit-time & scaling** — how long do fits take and how does that grow with draws, chains, weeks, channels, sampler, and the downstream analyses?
3. **Other failure modes** — beyond the recovery-bias study, what else can go wrong (numerical, geometric, statistical, operational)?

It is built on three evidence sources, all reproducible:

| evidence | what | how to reproduce |
|---|---|---|
| **Recovery/coverage matrix** | 13 structural-violation scenarios fit & scored | `uv run python -m tests.synth.run_stress_matrix --all` → `tests/synth/results/stress_matrix.md` |
| **Fit-time/convergence benchmark** | one-knob-at-a-time sweep + post-fit op costs | `uv run python -m tests.synth.bench` → `tests/synth/results/bench.json` |
| **Code audit** | failure-mode trace of the model graph (4-way parallel review) | `src/mmm_framework/model/base.py`, `transforms/`, `validation/` |

Benchmarks: macOS, PyMC NUTS, clean 156-week / 4-channel data unless noted. **Numbers are floors** from a fast config — production runs are slower and tighter.

---

## 1. Executive summary

* **Convergence is fine on well-specified data, but the diagnostic *flag* is not.** Every representable scenario sampled cleanly (r̂ ≈ 1.005–1.01, ESS 700–3 100, **0 divergences**). The only converged-failure was the Aurora kitchen-sink (r̂ = 1.11, **108 divergences**). Yet the production gate (`r-hat < 1.01`, strict) marks nearly every fast fit "not converged" because r-hat sits *right on* 1.01 — a false alarm driven by an inherent **β·λ identifiability ridge** that a diagonal NUTS mass matrix cannot rotate out.
* **Fits are fast; the *analyses around* them dominate.** A 156-week / 4-channel fit is **~17 s** (pyMC) or **~8.5 s** (numpyro). But the **causal refutation suite costs ~25 s** (4 refits) — more than the fit itself — and per-channel counterfactual contributions cost one posterior-predictive pass *per channel*. The stress matrix's ~40–70 s/scenario is mostly refutation, not fitting.
* **Scaling is gentle in time, steep in channels.** ~linear in weeks (104→12 s, 260→21 s); **super-linear in channels** (4→17 s, 8→45 s). Geo-level panels scale with `weeks × geos`.
* **The biggest latent risks are numerical, not statistical:** no NaN/inf input validation (a single bad cell → cryptic "Bad initial energy"), an **unguarded `w/w.sum()` in delayed/Weibull adstock that can divide 0/0 → NaN**, and near-constant-column degeneracy. None hit the default geometric path, but all are one config/data-quirk away.
* **The one true fix for the identifiability problems** (the β·λ ridge, collinear channels, confounding) is the same as in the recovery study: **experiment-anchored priors** (`calibration/`). Sampler tuning treats symptoms; a measured lift removes the ridge.

---

## 2. Convergence

### 2.1 What actually happens

Pure-`fit()` convergence on the **clean** world across configs (`bench.json`):

| config | sample s | r̂ max | min ESS (bulk) | divergences |
|---|--:|--:|--:|--:|
| 600 draws / 4 chains (baseline) | 17.1 | 1.007 | 1 427 | 0 |
| 300 draws / 4 chains | 9.2 | 1.010 | 790 | 0 |
| 1200 draws / 4 chains | 26.6 | **1.004** | **3 112** | 0 |
| 600 draws / 2 chains | 13.7 | 1.008 | 722 | 0 |
| target_accept = 0.99 | 14.9 | 1.007 | 1 427 | 0 |
| numpyro backend | 8.5 | 1.005 | 1 482 | 0 |

Per-scenario, on the structural-violation data (`stress_matrix.json`, 600/4 + refutation):

| | r̂ max | divergences |
|---|--:|--:|
| all 12 *representable/national* scenarios | **1.01** | **0** |
| `aurora_kitchen_sink` (extension model) | **1.11** | **108** |

**Takeaways.** (1) The national single-KPI model converges robustly — no divergences anywhere except the kitchen-sink. (2) More draws genuinely help: 300→1200 draws lifts min-ESS 790→3 112 and drops r̂ to 1.004. (3) `target_accept` 0.9→0.99 changes nothing on clean data (no divergences to suppress) — it matters only for the pathological geometries below.

### 2.2 Why r-hat sticks at ~1.01 (the β·λ ridge)

The lowest-ESS parameters are always the **media coefficients and saturation rates** — baseline benchmark worst-5: `beta_Search` (ESS 1427), `sat_lam_Search` (1447), `beta_TV`, `channel_contributions`, `sat_lam_TV`. This is structural, not a tuning artifact:

* Each channel contributes `beta · (1 − exp(−sat_lam · x))`. In the near-linear regime (most weeks, where normalized spend is below the channel max), `1 − exp(−λx) ≈ λx`, so the contribution is `≈ beta·λ·x` — **only the *product* `beta·λ` is identified**, not the two factors. The posterior is a curved ridge.
* NUTS with `init="adapt_diag"` (hard-coded for the pyMC backend, `base.py:1342`) tunes a **diagonal** mass matrix — per-parameter scales only, no rotation. It cannot align trajectories with a *correlated* ridge, so it takes small steps along it: low ESS on `beta`/`sat_lam`, r̂ that plateaus near 1.01 even as you add draws.
* Worse, the explicit `"adapt_diag"` **overrides PyMC's default `"jitter+adapt_diag"`**, removing the per-chain initial jitter — so chains can settle into different parts of a near-degenerate sheet, inflating r-hat. The default numpyro backend is likewise diagonal (`dense_mass=False`), so the diagnosis holds on both backends.

This is benign for the headline number (the *product*, i.e. the contribution, is well-determined and recovers to ~7% on clean data) but it means **the per-factor posteriors are weakly identified by design.**

### 2.3 Where divergences appear

Divergences signal genuinely pathological geometry. Observed and code-confirmed triggers:

| trigger | mechanism | evidence |
|---|---|---|
| **Collinear channel spend** | saturated regressors near-parallel → `beta_A+beta_B` identified, `beta_A−beta_B` not → strong negative β-correlation, divergences on the pair | earlier collinear run: 6 divergences |
| **Hill + latent mediation (extensions)** | stiff Hill curvature (`slope~Gamma(3,1)` can be near-step; unbounded `kappa` drifts off the data range) **compounded** with bilinear mediator terms | `aurora`: r̂ 1.11, 108 div |
| **Legacy adstock blend** | default `adstock_mix·Beta(2,2)` mixes two *highly correlated* fixed-α series (α=0 vs α=0.9) → weak ID, boundary pile-up at 0/1 | `base.py:953-956`; low ESS on `adstock_*` |
| **Single-group hierarchy** | `n_geos=1` with `pool_across_geo=True` → `geo_sigma` unidentified (a population SD from one group) → funnel | `base.py:911-931` |

### 2.4 The convergence *gate* is the real problem

`ConvergenceSummary` passes only if `r-hat < 1.01` strictly (`validation/results.py`). Because the β·λ ridge parks r-hat at ~1.005–1.01, **fast fits are routinely flagged "not converged" even when they are fine** — every row of the stress matrix printed `conv=False` at r̂≈1.01 despite 0 divergences and ESS>700. Two compounding issues:

* The threshold is a production-grade bar applied to a fast config.
* r-hat/ESS are computed over **all** posterior variables including large deterministics (`channel_contributions` is `obs × channel`), which can inflate the reported max.

**Recommendations (convergence).**
- For production, sample **≥1 000 draws / ≥1 000 tune, 4 chains**; budget more for collinear or extension models.
- For the pyMC backend, prefer `init="jitter+adapt_diag"` (restore jitter) and consider a **dense mass matrix** (`init="jitter+adapt_full"`) to absorb the β·λ ridge; raise `target_accept` to 0.95–0.99 for collinear/extension fits.
- Treat the strict `r-hat < 1.01` flag as advisory at fast settings — **read divergences and ESS directly**; ~1.01 with 0 divergences and ESS in the thousands is converged.
- The structural fix for the ridge is an **experiment-calibrated `roi_prior`** on a channel — pinning `beta` collapses the ridge to a point.

---

## 3. Fit-time & scaling

### 3.1 Benchmark (pure `fit()` wall-clock)

| knob | value | sample s | draws/s | note |
|---|---|--:|--:|---|
| **baseline** | 600d/4c/param/pyMC/156w/4ch | **17.1** | 140 | reference |
| draws | 300 | 9.2 | 130 | ~linear in draws |
| draws | 1200 | 26.6 | 180 | |
| chains | 2 | 13.7 | 88 | 4 chains ≈ 1.25× the 2-chain time (good parallelism) |
| adstock | legacy 2-point blend | 13.2 | 182 | ~25% faster than parametric, but **lower ESS** (1105 vs 1427) |
| **sampler** | **numpyro** | **8.5** | **282** | **~2× faster than pyMC** |
| target_accept | 0.99 | 14.9 | 161 | negligible cost on clean data |
| weeks | 104 / 156 / 260 | 11.8 / 17.1 / 21.3 | — | **~linear in n_obs** |
| channels | 8 (vs 4) | **45.4** | 53 | **~2.7× for 2× channels — super-linear** |

### 3.2 The analyses cost more than the fit

Post-fit operations on one baseline fit:

| operation | cost | why |
|---|--:|---|
| `compute_counterfactual_contributions` (uncertainty, 4 ch) | **1.3 s** | `N+1` posterior-predictive passes (baseline + one per channel) |
| `compute_parameter_learning` | 0.2 s | one prior-predictive draw + arithmetic |
| **causal refutation suite** (4 refits @150 draws) | **25.4 s** | refits the *entire model* 4× (placebo, negative-control, random-common-cause, data-subset) |

So a "scored" scenario ≈ **fit (17 s) + refutation (25 s) + ops (~2 s) ≈ 44 s**, matching the matrix's 37–72 s/scenario. **Refutation, not fitting, is the cost center.** `compute_counterfactual_contributions` and `compute_marginal_contributions` scale as **`O(n_channels)` posterior-predictive passes**, so they get expensive on wide channel sets even though one pass is cheap.

### 3.3 Scaling story (mechanisms behind the numbers)

* **Time ∝ n_obs ∝ `weeks × geos`, and ∝ `draws`.** Geo is the true scaling axis: a stacked panel multiplies n_obs by the geo count *and* adds `geo_offset`/`product_offset` latent dims (NUTS per-step cost ~linear in dimension; step counts also grow), so geo scale is **super-linear** overall. National-weekly (~156 obs) → seconds; geo-level (thousands of obs) → minutes-to-hours.
* **Why parametric adstock is slower / channels scale super-linearly.** On the parametric path each channel builds a dense `(n_obs, l_max)` window matrix and does `pt.dot(windows, weights)` *inside every leapfrog gradient evaluation* (`adstock_pt.py`), because the kernel params are RVs — cost/grad is `O(n_channels · n_obs · l_max)`. The legacy path precomputes two fixed-α adstocks once in NumPy and the graph only does a scalar blend (`O(n_obs)`, no `l_max`). That is why parametric is ~25% slower at 4 channels and the 8-channel fit is ~2.7× the 4-channel one. Weibull/delayed (2–3 shape params, default `l_max` 12) is the worst-conditioned and slowest. *Mitigation:* FFT/`pt.signal` convolution or precompute the windowed tensor once outside the graph (windows don't depend on params, only the weights do); keep `l_max` minimal.
* **Trace memory blows up with n_obs (latent — geo scale not run).** `_build_model` registers ~8 per-obs `Deterministic`s stored *every draw* (`intercept_component`, `trend_component`, `seasonality_component`, `geo_component`, `product_component`, `media_total`, `controls_total`, `y_obs_scaled`) **plus** `channel_contributions` (`obs × channel`) and `control_contributions` (`obs × control`). Trace size ≈ `draws · chains · n_obs · (8 + n_channels + n_controls)` float64. National (156 obs) is a few MB and harmless; **156 wk × 50 geo, 4 ch, 1000×4 draws ≈ ~4 GB before posterior-predictive** → OOM/swap and slow `az.summary`/r-hat/serialization. *Mitigation:* compute the decomposition post-hoc (`pm.compute_deterministics`) from the small parameter posterior rather than storing obs-sized tensors during sampling.
* **Contribution/ROAS analysis is quadratic in channels.** `compute_counterfactual_contributions` and `compute_marginal_contributions` run `N+1` posterior-predictive passes (baseline + one per channel), and **each pass re-evaluates all `N` channel adstock kernels** → `O((N+1)·draws·chains·n_obs·l_max·N)`. Cheap at 4 channels (1.3 s) but dominates at 10–20 channels / geo scale. *Mitigation:* reconstruct contributions analytically from the already-stored `channel_contributions` Deterministic instead of re-sampling; or batch all channel knock-outs into one `set_data` pass.
* **Validation refits multiply the whole fit.** Each enabled block re-fits the full model: refutation ≤4×, cross-validation per fold (~5×), sensitivity per multiplier, bootstrap up to ~20× — and every clone re-incurs the trace-memory cost above.

**Recommendations (performance).**
- Switch the inference backend to **numpyro** (`InferenceMethod.BAYESIAN_NUMPYRO`) for ~2× speed at equal/better convergence; nutpie is installed but not yet wired into `fit()`.
- **Fit once, analyse many** — contributions/marginals/validation all reuse the trace; don't refit. Reconstruct contributions from the stored deterministic rather than re-sampling per channel.
- Run the **refutation/CV/bootstrap suites selectively** (they are the dominant cost) — on the final model, with a time budget, parallelized across processes.
- At geo scale, reduce stored deterministics (compute post-hoc), prefer numpyro/nutpie + GPU, and expect channel-count and n_obs to dominate.

---

## 4. Other failure modes

Severity reflects likelihood × impact on the **default** path (national, geometric adstock, Normal likelihood). "Confirmed" = observed empirically here; "latent" = code-confirmed but not on the default path.

### A. Convergence & posterior geometry

| # | failure | mechanism | trigger | mitigation | status |
|---|---|---|---|---|---|
| A1 | **β·λ ridge** | contribution `≈ beta·λ·x` in the linear regime → only the product is identified | always (worse at low spend variation) | experiment `roi_prior`; dense mass matrix; read ESS not r̂ | confirmed |
| A2 | **Diagonal metric can't rotate ridges** | `adapt_diag` tunes only per-param scales | always | `jitter+adapt_full`; more tune | confirmed |
| A3 | **Collinear channels → divergences** | parallel regressors; difference of βs unidentified | correlated spend (paired/always-on campaigns) | calibrate one channel; ↑target_accept; aggregate | confirmed |
| A4 | **Legacy adstock-blend multimodality** | convex mix of two collinear fixed-α series | default `use_parametric_adstock=False` on interior-decay data | use parametric path with informative α prior | latent |
| A5 | **Extension stiffness (Hill + latent mediation)** | near-step Hill × bilinear mediator | `NestedMMM`/Hill with under-observed mediator | data-anchored bounded `kappa`; tighten `slope` prior | confirmed (aurora) |
| A6 | **Level degeneracy (geo/product/trend vs intercept)** | group offsets & trend levels trade off with the intercept | hierarchy with few groups; piecewise/HSGP trend | `ZeroSumNormal` offsets; mean-center trend basis | latent |

### B. Numerical & data edge cases

| # | failure | mechanism | trigger | mitigation | status |
|---|---|---|---|---|---|
| B1 | **Delayed/Weibull adstock NaN kernel** | `w = w/w.sum()` (`adstock_pt.py:68`) unguarded; large `theta` or tiny `scale` underflows *all* lag weights → `0/0` | `use_parametric_adstock=True` with a `delayed`/`weibull` channel | guard: `w = switch(sum>0, w/sum, unit_impulse)`; bound `theta`/`scale` priors | latent (verified in code) |
| B2 | **No NaN/inf/shape input validation** | `_prepare_data` standardizes raw y/X with no finiteness check; one NaN → NaN `y_std` → whole graph NaN | any NaN/inf/sentinel in KPI or media (missing weeks, sentinels) | assert `np.isfinite(...)` naming the offending column; assert `n_obs` minimum | latent (high impact) |
| B3 | **Near-constant column degeneracy** | the `+1e-8` floor only saves *exactly* constant columns; a near-constant column gets a tiny std so one outlier standardizes to a huge z (a 0/0/…/1 flag → **z ≈ 12.4**, a 12-σ leverage point). A media channel whose max is a lone spike normalizes typical weeks to ~0.01 → `1−exp(−λx)` is locally **linear** → `sat_lam` unidentifiable (this is the mechanism behind the `spend_outliers` silent failure) | near-constant control/flag, low-variance KPI, one big spend spike | warn/drop on coefficient-of-variation below threshold; **robust (high-percentile) media scaling** instead of raw max | latent (boundary-verified) |
| B4 | **Saturation clip floors gradient** | `1 − exp(clip(−λx, −20, 0))` → exactly 0 (zero gradient) for negative adstocked input | perturbed/counterfactual or future spend going negative; high normalized spend | document expected low `sat_lam` ESS; soft-saturation alternative | confirmed (low-ESS) |
| B5 | **Single-geo / 1-obs panels** | one-group hierarchy unidentified; std/trend paths degenerate | `n_geos=1` with pooling on; n_obs ≤ 2 | force `pool_*_=False` for <2 groups; require min n_obs | latent |
| B6 | **Geo cross-boundary adstock bleed** *(correctness)* | adstock is applied per channel along the **full observation axis with no geo segmentation** (`geometric_adstock_2d` = "independently to each column"; parametric `apply_adstock_pt` convolves the whole column). On a stacked multi-geo panel, carryover bleeds from one geo's last weeks into the next geo's first weeks → biased contributions in the first `l_max−1` rows of each geo block | any multi-geo panel (both adstock paths) | segment-aware adstock: reshape to `(geo, weeks)` and convolve per geo | **latent (mechanism confirmed in source; geo scale not run)** |

### C. Statistical & identifiability

| # | failure | mechanism | trigger | mitigation | status |
|---|---|---|---|---|---|
| C1 | **adstock·saturation·β equifinality** | normalized adstock folds magnitude into β; many (decay, λ, β) triples fit equally well | always (worse with little spend variation) | experiment priors; report the *contribution*, not the factors | confirmed |
| C2 | **Prior-data conflict (downward shrink)** | `beta ~ Gamma(μ=1.5)` pulls large true effects down | true standardized effect ≫ 1.5 | calibrate `roi_prior`; widen with justification | confirmed (clean control under-recovers ~7%) |
| C3 | **Counterfactual HDI inflation** | `compute_counterfactual_contributions` passes `random_seed` to *both* predict calls, but the **default `None` re-seeds independently** → unpaired noise → CI too wide by ~√2·σ | calling it with `compute_uncertainty=True` and default `random_seed=None` | always pass a fixed seed (as the stress harness does — so its coverage is *not* inflated); fix to mirror `compute_marginal_contributions`' shared pair-seed | latent (verified; mitigated in our harness) |
| C4 | **Silent recovery failures** | see the recovery study | confounding, collinearity, saturation-shape, spend-outliers, imperfect controls | experiment anchoring | confirmed (5 silent failures — `tests/synth/README.md`) |

### D. Operational / API

| # | failure | mechanism | mitigation | status |
|---|---|---|---|---|
| D1 | **False "not converged"** | strict `r-hat < 1.01` gate vs the ~1.01 ridge floor | treat as advisory; gate on divergences + ESS | confirmed |
| D2 | **Backend availability/parity** | `fit()` selects pyMC or numpyro via `use_numpyro`; numpyro needs JAX (present here) and has a first-call compile cost; nutpie is installed but not wired in | document the JAX dependency; wire nutpie if desired | informational |
| D3 | **Downstream output flooring** | ROI/MAPE guards (`analysis.py`, `validation/validator.py`) substitute `0`/`1.0` for non-finite or zero-spend cases | prefer NaN + exclude from aggregation so degenerate cases stay visible | latent |

---

## 5. Consolidated recommendations

**Run configuration**
1. Default the backend to **numpyro** (~2× faster, equal convergence).
2. Production sampling: **≥1 000 draws / ≥1 000 tune / 4 chains**, `target_accept` 0.9; bump to **0.95–0.99** and `init="jitter+adapt_full"` for collinear or extension models.
3. **Fit once, analyse many.** Run the refutation suite (the dominant cost) only on the final model.

**Trust the right signals**
4. Don't reject a fit on `r-hat ≈ 1.01` alone — require **0 divergences and ESS in the hundreds–thousands**. Loosen the production gate or report r-hat alongside divergences/ESS.
5. Expect low ESS on `sat_lam`/`beta`/`adstock_*`: that is the identifiability ridge, not a bug.

**Hardening (recommended code changes)**
6. Add **input validation** in `_prepare_data` (finite check naming the bad column; minimum n_obs) — fixes the most common cryptic crash (B2).
7. **Guard the parametric adstock normalization** (`adstock_pt.py:68`) against `0/0` for delayed/Weibull kernels (B1).
8. **Seed-pair** `compute_counterfactual_contributions` by default, mirroring `compute_marginal_contributions` (C3).
9. Auto-disable single-group pooling and warn on near-constant columns (B3, B5).
10. **Before any geo rollout:** make adstock **segment-aware** (per-geo convolution) — the current path bleeds carryover across geo boundaries (§4B B6) — and stop storing obs-sized `Deterministic`s in the sampling graph (compute the decomposition post-hoc) to avoid multi-GB traces (§3.3).

**The real fix for the hard cases**
11. The ridge (A1/C1), collinearity (A3), and confounding (C4) are **identification** problems no sampler setting solves. Anchor effects with **randomized geo-lift / incrementality experiments** via `calibration/` — a measured lift pins `beta`, collapses the ridge, and is the one signal the silent failures cannot fake.

---

## Appendix — reproduce

```bash
# recovery/coverage matrix (13 scenarios)
uv run python -m tests.synth.run_stress_matrix --all
# fit-time / convergence benchmark
uv run python -m tests.synth.bench
# fast generator self-checks (no MCMC)
uv run pytest tests/synth/test_dgp.py -q
```

Raw data: `tests/synth/results/stress_matrix.{md,csv,json}`, `tests/synth/results/bench.json`.
Companion: `tests/synth/README.md` (the recovery study and its silent-failure findings).

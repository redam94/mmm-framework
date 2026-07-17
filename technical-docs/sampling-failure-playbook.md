# Sampling-failure playbook — diagnosing a Bayesian fit and what to try next

The framework's formal escalation ladder for a fit that fails: what each
diagnostic actually tells you, which rung of fixes it points to, and where each
`FitMethod` belongs. The `fit()` docstring, the agent's fit tooling, the docs
troubleshooting page and the research-blog post ("When sampling fails") all
point here; this file is the source of truth for the ordering.

The one-line version: **sampler failures are almost always model failures.**
The ladder is ordered so you spend seconds before minutes and fix
identification before reaching for a different algorithm.

## 1. What "failed" means — reading the diagnostics

Every exact fit stamps `diagnostics` and emits a `ConvergenceWarning` on
failure (`diagnostics/convergence.py`, single source of truth; thresholds per
Vehtari et al. 2021):

| Signal | Threshold | What it means |
|---|---|---|
| `divergences` | `> 0` | NUTS hit geometry it cannot integrate — funnels, cliffs, stiff curvature. The posterior is *biased where it matters most* (divergences cluster where the sampler was repelled). Not applicable to SMC (`None`, never flagged). |
| `rhat_max` | `> 1.01` | Chains (or independent SMC runs) disagree. With clean-looking per-chain traces this is the **multimodality** signature — each chain found a different mode. |
| `ess_bulk_min` | `< 400` | High autocorrelation — usually a ridge (two parameters trading off) or weak identification, not a bug. Intervals are computed from too few effective draws. |

Three distinct failure *syndromes* hide behind these numbers:

1. **Divergences** → curvature/scale problems. Fix: reparameterize
   (non-centered), rescale data, raise `target_accept`.
2. **High R-hat, clean traces** → multimodality. Fix: **identification**
   (anchoring, sign constraints, centering rules) — a different sampler only
   *diagnoses* this, it does not remove the modes. This codebase's documented
   cases: the StructuralNestedMMM reflected-factor mode (split R-hat 1.755
   until the sign anchor moved to a mediator whose loading the data holds
   nonzero), label switching in LCA/latent-factor models, the adstock↔AR
   `α↔ρ` ridge.
3. **Low ESS everywhere** → ridges/weak identification. Fix: priors that break
   the ridge, or accept and run longer.

A fourth, silent one: **wall-clock explosion** (tuning never settles). Treat
it as syndrome 1/3 — the geometry is bad even if no threshold has tripped yet.

## 2. The ladder

Ordered cheapest-first. Each rung names the framework hook.

**Rung 0 — check the model before the sampler.**
`prior_predictive_check` / the prefit Model Design Readout
(`generate_model_design_readout`): implausible prior KPI ranges,
`scale_z_abs_mean` blowups, prior ROI nonsense. Run the smoke SBC
(`run_calibration_check`) — a mis-calibrated refit loop fails *before* you
burn hours on NUTS. Mis-scaled data and absurd priors are the leading cause of
"sampler problems".

**Rung 1 — approximate reconnaissance (seconds).**
`fit(method="map")` → does the optimizer even find a mode? A stall or NaN
usually means a scaling/NaN-gradient bug (e.g. variance-outside-sqrt guards).
`fit(method="laplace")` → MAP + Gaussian curvature: if the Hessian is not
positive-definite (the fit warns/fails), some direction of parameter space is
flat — a weak-identification flag *before* any MCMC. `advi`/`pathfinder` give
a cheap posterior shape check. All are `approximate=True`: never for final
inference.

**Rung 2 — identification and reparameterization.**
The rung that actually fixes syndromes 2 and 3. The codebase's hardened rules:
sign-anchor a latent factor at a loading the data holds materially nonzero
(HalfNormal anchor on a near-zero loading has a cost-free zero mode — the
reflection escapes); zero-center non-linear trend families so their level goes
to the intercept, not the media coefficients; never center media-dependent
signals in-graph; standardize latent scales in-graph (variance inside the
sqrt); non-centered state parameterizations for sparse measurement.
Tighter/asymmetric priors that break an `α↔ρ`-style ridge belong here too.

**Rung 3 — sampler settings.**
`target_accept` 0.9 → 0.95 → 0.99 (smaller steps through tight curvature —
costs time, cures *false* divergences only); more `tune`; more draws/chains
for a pure-ESS shortfall; `nuts_sampler="numpyro"` for speed. Settings are the
*last* resort for divergences, not the first: papering over a divergence with
`target_accept=0.999` on unfixed geometry just hides the bias.

**Rung 4 — SMC: confirm multimodality, get evidence.**
`fit(method="smc")` (exact; `approximate` stays `False`). Independent tempered
particle populations do not get mode-locked, so R-hat *across SMC runs*
separates "NUTS was stuck" from "the posterior really has modes". If the modes
are real → go back to Rung 2 and fix identification; if the multimodality is
*semantic* (a mixture you mean to have), the SMC posterior is the honest
answer. SMC also estimates the **log marginal likelihood**
(`diagnostics["log_marginal_likelihood"]`, per-run values in
`…_per_run`) — Bayes-factor comparison of candidate fixes (adstock families,
mediation structures) that no other method in the framework provides.

**Rung 5 — escalate the model, not the chain count.**
If Rungs 0–4 keep failing, the model is asking more of the data than it holds:
simplify (drop a latent state, pool harder), or bring more information
(experiment calibration priors, `roi_prior`).

## 3. Method reference

| `FitMethod` | Exact? | Cost | Reach for it when |
|---|---|---|---|
| `nuts` | yes | minutes | Default. Final inference. |
| `smc` | **yes** | minutes–hours | Suspected multimodality; model evidence (lml). NOT a speedup — the IMH kernel degrades in high dimension (large geo hierarchies / latent-state models will crawl). |
| `map` | no (point) | seconds | Fastest "does a mode exist" probe. |
| `laplace` | no | seconds | MAP + curvature uncertainty; Hessian failure = identification flag. More stable read than bare MAP on high-dim models. |
| `advi` / `fullrank_advi` | no | seconds–minutes | Cheap posterior-shape check; ELBO trend as a geometry smell test. |
| `pathfinder` | no | seconds | Multi-path quasi-Newton VI; also a good NUTS-init candidate (future). |

Semantics pinned by `FitMethod.is_approximate` (`nuts`/`smc` → `False`):
report banners, serializer metadata, run-history provenance and the agent
registry all key off it — SMC must never be labelled "uncertainty not
calibrated".

## 4. Implementation notes

- **Engines** (`model/base.py`): `run_approximate_fit` (map/laplace/advi/
  fullrank_advi/pathfinder — laplace via `pymc_extras.fit_laplace`, BFGS,
  `progressbar=False` default) and `run_smc_fit` (`pm.sample_smc`;
  `draws`=particles per run, `chains`=independent runs). Both are
  model-agnostic and shared with `BaseExtendedMMM.fit`, so extension models
  (Nested/MV/Combined/Structural) get every method.
- **Laplace optimizer gotcha** (measured on the clean synth MMM,
  `nbs/demos/approximate_posteriors.ipynb`): the default quasi-Newton (BFGS)
  optimizer is run-to-run fragile on ridgey MMM posteriors — one run stopped
  short of the mode and reported a **median SD 76× NUTS** (curvature measured
  off-mode is meaningless). Passing
  `fit(method="laplace", optimize_method="trust-ncg", use_hess=True)` (kwargs
  flow through to `fit_laplace`) reliably reaches the mode: median SD ratio
  ≈ 0.94 vs NUTS, at ~3–4× the BFGS wall-clock. If a Laplace fit's spreads
  look absurd, suspect the optimizer before the method. (Kept as a kwarg, not
  the framework default — exact-Hessian assembly scales poorly to
  hundreds-of-parameter geo hierarchies.)
- **SMC particle-count gotcha** (same notebook): at 500 particles × 4 runs the
  independent runs *disagreed* (R-hat 1.43, 14-nat evidence spread) on a clean
  unimodal world — under-populated runs mimic the multimodality signature.
  2000 particles → R-hat 1.014, ~2-nat spread. Disagreement that *persists as
  particles grow* is real structure; disagreement that merges is Monte-Carlo
  noise.
- **SMC lml extraction gotcha**: `sample_stats["log_marginal_likelihood"]` is
  per-stage per-chain; with **ragged** stage counts across runs xarray stores
  a 1-D object array of per-chain *lists* (not a padded 2-D array), and rows
  are NaN-padded when counts agree. `run_smc_fit` takes the last *finite*
  value per chain — do not index `[:, -1]`.
- **SMC + `pm.Potential`** (LCA garden model, experiment calibration): PyMC
  folds Potentials into the tempered likelihood term with a warning —
  semantically what we want here (the calibration likelihood should be
  tempered too).
- **SMC diagnostics**: `compute_convergence` handles the missing `diverging`
  stat (→ `None`, never flagged); R-hat/ESS come from the independent runs;
  `warn_if_not_converged` still fires — for SMC that warning *is* the
  multimodality signal, not a "run longer" nag.
- **Where non-nuts used to be conflated with approximate** (all fixed to
  respect `is_approximate`): serializer `metadata.json`,
  `planning/history.compute_run_metrics`, the saved-settings digest
  (`agents/fitting.settings_digest_markdown`), the prefit Inference-plan row,
  `validation/spec_curve` fit kwargs, FE `ModelSpecWidget`/`ArtifactsPanel`
  badges.
- **Registry**: `agents/fitting._INFERENCE_METHODS` = {nuts, smc, map,
  laplace, advi, fullrank_advi, pathfinder}; `unconsumed_spec_path` validates
  values up front; builder sugar `.laplace()` / `.smc()`.

## 5. Deferred (known next wins, deliberately not in this change)

- **nutpie routing** — `nutpie>=0.16` is a declared dependency but
  `fit()` only dispatches `nuts_sampler="numpyro"|"pymc"` (and passing
  `nuts_sampler` via kwargs collides with the explicit argument). Wiring it in
  (and its experimental normalizing-flow adaptation) is the highest-value
  "NUTS fails on geometry" fix after this playbook: a learned
  reparameterization without touching the model.
- Pathfinder/ADVI-initialized NUTS (`init=` is hardcoded `"adapt_diag"`).
- GPU vectorized chains via numpyro for big geo panels.

## 6. Tests

`tests/test_approx_fit.py` (laplace contract + `TestSMCFit` exact-contract:
not-approximate, R-hat/ESS finite, lml present incl. the ragged-stage layout,
predict works), `tests/test_extension_fit_path.py::test_smc_fit_supported_for_extensions`,
`tests/test_spec_path_validation.py` (registry accepts laplace/smc, rejects
typos).

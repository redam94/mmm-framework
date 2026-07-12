# Validation Plan: Deferred Causal-MMM Features (DF-1, DF-2)

**Status:** Validation strategy for the two deferred, *posterior-changing* features
specified in [`deferred-causal-features.md`](./deferred-causal-features.md). Neither
is implemented. This document is the **proving harness contract**: it turns the
abstract acceptance gates (A1–A4) in the spec into runnable procedures with named
substrate, reused harnesses, explicit metrics, and numeric thresholds.

**Why a separate plan.** The spec answers *what to build and what must hold*. This
answers *how we prove it holds before it ships, and what number decides*. Both
features change the likelihood, so per **R0.3** they clear a stricter bar than any
shipped diagnostic: a feature that changes the numbers can be confidently wrong in a
way a report section cannot. DF-3 (Full-ID) is **excluded here** — it changes only
the identification layer, not the posterior, so it is validated by the existing
networkx-style cross-check (P2-6 pattern), not by this plan.

**Scope:** `DF-1` (geo-heterogeneous media coefficients) and `DF-2`
(grouped/hierarchical priors for collinear channels).

---

## 0. The decision ladder (what each gate buys)

Every gate maps to a **ship decision**, not just a green test. A feature moves up the
ladder only by clearing the gate at that rung:

| Rung | Decision | Gate that unlocks it | Reversible? |
|------|----------|----------------------|-------------|
| 0 | Code merged, flag absent from all defaults | **A1** (R0.2 regression: flag-off ≡ today) | n/a |
| 1 | **Ship behind flag**, opt-in only, documented as experimental | **A1 + A2** (converges + fails loud when unidentified) | yes — flag stays off |
| 2 | **Recommendable as a default** for qualifying datasets | **A3 + A4** (held-out improvement + data-driven, not prior-driven) | yes — demote to rung 1 |
| — | **Abandon / redesign** | any *kill criterion* (§6) trips | — |

Rung 1 is cheap to hold forever (a dormant flag). Rung 2 is the expensive claim and
is the only one that requires held-out experimental evidence. **We do not need rung 2
to ship; we need it to recommend.** This is the core risk control.

---

## 1. Shared validation infrastructure (reuse, don't reinvent)

All of this exists today. The plan composes it; it does not rebuild it.

| Capability | Where | Used for |
|---|---|---|
| Stress-matrix recovery harness (`_build_model` → `fit` → recovery metrics → `silent_failure` flag) | `tests/synth/harness.py:107-150`, `tests/synth/run_stress_matrix.py` | The fit-and-grade loop for every recovery test below |
| Recovery metrics: `median_abs_rel_error`, `max_abs_rel_error`, 90% CI `coverage_rate` | `tests/synth/harness.py:50-99` (`ScenarioResult`) | Per-channel / per-geo recovery grading |
| Rolling-origin / frozen-predict backtest | `validation/backtest.py` (`PosteriorForecaster`; `FrozenPredictor` used at `validator.py:82`) | Out-of-sample A/B of two specs on the **same** fold |
| Geo identifiability pre-gate (cross-geo spend CV vs 0.15) | `validation/geo_identification.py:74-123` (`geo_spend_variation_diagnostic`) | DF-1 A2 loud-failure gate (this is DF1.6) |
| Collinearity detection (union-find clusters, condition number) | `validation/channel_diagnostics.py:26-85, 415-427` | DF-2 substrate selection + before/after condition-number reporting |
| Lift-test agreement (`within_ci`, `coverage_rate`, `relative_error`) | `validation/validator.py:1425-1575`, `validation/results.py:806-851` (`CalibrationResults`) | Held-out experimental validation (R0.3) for both features |
| Experiment-calibrated prior path (`roi_prior`) | `calibration/experiment.py`, `model/base.py:1300-1306` | DF1.3 / DF2.4 precedence tests |
| Panel adstock cell-isolation harness | `tests/test_panel_adstock.py` (`_impulse_panel`) | DF-1 no-cross-geo-bleed check |

### Ground-truth substrate already on disk

| Scenario | Factory | Known truth it exposes | Which feature |
|---|---|---|---|
| `geo_heterogeneous` | `src/mmm_framework/synth/dgp_geo.py:430-477` (`make_geo_heterogeneous`) | Per-geo channel effectiveness multipliers (0.3–1.8×), performance-chasing budgets; `true_contribution_by_geo`, `true_roas_by_geo`. **Pooled β cannot represent this** → the discriminating world for DF-1. | DF-1 |
| `geo_clean` | `src/mmm_framework/synth/dgp_geo.py` (`make_geo_clean`) | Geo differences are **level shifts only**, exact model family. | DF-1 positive control |
| geo×product | `src/mmm_framework/synth/dgp_geo.py:493-525` | Additive geo + product offsets, **shared β**, per-cell truth. | DF-1 positive control |
| `multicollinearity` | `src/mmm_framework/synth/dgp.py:450-487` (`make_multicollinearity`) | Synchronized flighting, mean pairwise \|r\| ≈ 0.85; `true_contribution` per channel via counterfactual zero-out on the noiseless mean. | DF-2 |
| `dense_controls` | `src/mmm_framework/synth/dgp.py:953-1051` | 1 confounder + 2 precision + 18 noise + 4 decoys; `true_control_effects`, `latent_demand`. | DF-2 confounder-interaction |

**Two substrate gaps this plan must add** (small, scoped):
1. `make_geo_heterogeneous` already carries per-geo truth but **no geo-lift answer
   key** — add a derived per-geo lift fixture (treat the known per-geo incremental
   contribution over a test window as a synthetic lift test) so DF-1 A3 can run the
   `CalibrationResults` path, not just point-recovery. *(One helper in `src/mmm_framework/synth/mff.py`.)*
2. DF-2 A2 needs an **asymmetric-collinearity** world: two collinear channels where
   one *also* has independent spend variation (so it is strongly identified and the
   other is not). Add `make_multicollinearity_asymmetric` to `src/mmm_framework/synth/dgp.py`. Today's
   `multicollinearity` makes both channels equally weak, which cannot test "the strong
   channel escapes the pool."

---

## 2. DF-1 — Geo-heterogeneous media coefficients

**What changes (from spec §DF-1):** media coefficient goes from a global scalar
`beta_{channel}` (`model/base.py:1294-1308`) to `beta_geo[g,c] ~ Normal(mu_c, tau_c)`
(non-centered), gated by `HierarchicalConfig.geo_varying_media=False`. `mu_c` honors
`roi_prior`; `beta_{channel}` is re-emitted as `mu_c` for downstream contract (R0.4).

### Gate A1 — Zero-change regression (unlocks rung 0 → 1)

- **Procedure.** Before the feature lands, capture a **golden posterior**: fit the
  current code on a small fixed panel (e.g. 3 geos × 60 weeks, 3 channels) at a fixed
  seed; persist the `idata.posterior` to `tests/golden/geo_pooled_golden.nc`. After
  the feature lands, fit with `geo_varying_media=False`, same seed.
- **Metric / threshold.** `xr.testing.assert_allclose(golden, new, atol=1e-10)` over
  every shared variable, **and** identical variable names/shapes/coords. Any RV the
  golden has that the new run lacks (or vice versa) fails the gate.
- **Belt-and-suspenders.** A structural assertion that, when the flag is off, the media
  block executes the unchanged scalar path (guard the new branch behind
  `if has_geo and n_geos >= 2 and geo_varying_media:` and assert the else-branch is the
  literal pre-feature code).
- **Pass = rung 1 eligible.**

### Gate A2 — Converges where identified, fails loud where not (rung 1)

- **Procedure (identified case).** Fit `geo_varying_media=True` on `geo_heterogeneous`
  (genuine cross-geo variation) via the stress harness; draws/tune sufficient for the
  geo dimension (start 1500/1500, 4 chains).
- **Metric / threshold.** On every `beta_geo[g,c]`: `r_hat < 1.01`, `ess_bulk > 400`,
  **0 divergences**. (Per spec A2.) Use `utils/arviz_compat.dataset_extremum` to reduce
  R̂/ESS over `beta_geo`.
- **Procedure (unidentified case).** Build a degenerate panel where geo spend is
  **synchronized** (cross-geo CV < 0.15 for ≥1 channel — reuse a `multicollinearity`-style
  flighting pattern replicated across geos).
- **Metric / threshold.** `geo_spend_variation_diagnostic` flags the channel, and the
  fit path **raises or warns loudly** (DF1.6) naming the under-identified channel — it
  must **not** silently return a fit. Assert the warning/exception fires and names the
  channel. A silent green fit on this panel is an **automatic A2 fail** (it is the exact
  silent-failure mode the framework exists to prevent).

### Gate A3 — Held-out geo backtest (the rung-2 gate, R0.3)

This is the only gate that licenses recommending DF-1 as a default. Run **three**
comparisons; all must hold.

1. **Per-geo recovery on the discriminating world.** On `geo_heterogeneous`, fit both
   the pooled model and the geo-varying model.
   - *Metric:* per-geo ROAS error vs `true_roas_by_geo`:
     `median_g,c |roas_est[g,c] − roas_true[g,c]| / |roas_true[g,c]|`.
   - *Threshold:* geo-varying median rel-error **≤ pooled median rel-error** (non-inferiority
     margin 0; ideally strictly lower), **and** 90% HDI `coverage_rate` of the true
     per-geo ROAS **≥ 0.85**. The pooled model is *expected* to fail coverage here
     (it cannot represent heterogeneity) — that failure is what justifies the feature.
2. **Held-out-geo prediction.** Leave-one-geo-out: train on geos `1..K-1`, predict geo
   `K`'s outcome with `FrozenPredictor` on the same fold for both specs.
   - *Metric:* held-out-geo outcome MAPE (rolling-origin via `PosteriorForecaster`).
   - *Threshold:* geo-varying MAPE ≤ pooled MAPE + 0.5pp (non-inferior), averaged over
     all leave-one-out folds.
3. **Held-out geo-lift agreement.** Using the synthetic per-geo lift fixture (§1 gap 1),
   hold the lift test out of the fit, then check it lands in the posterior via
   `CalibrationResults`.
   - *Metric:* `coverage_rate` (fraction of held-out per-geo lifts inside the 90% HDI)
     and `mean_absolute_calibration_error`.
   - *Threshold:* geo-varying `coverage_rate` ≥ pooled, and ≥ 0.85 absolute; calibration
     error not worse than pooled.

- **Positive controls (must NOT regress).** Repeat (1) on `geo_clean` and the
  geo×product world (shared-β truth): the geo-varying model must **not invent spurious
  heterogeneity** — `tau_c` posterior 90% HDI upper bound must concentrate near 0
  (threshold: `tau_c` HDI-upper < 20% of `mu_c`), and national fit (LOO/MAPE) must not
  degrade vs pooled. If the feature manufactures heterogeneity on a homogeneous world,
  **A3 fails** regardless of its wins on `geo_heterogeneous`.

### Gate A4 — Data-driven, not prior-driven (rung 2)

- **Procedure.** On `geo_heterogeneous`, refit at two `tau_c` prior scales
  (`HalfNormal(0.25)` vs `HalfNormal(1.0)`).
- **Metric / threshold.** The recovered geo split (spread of `beta_geo[·,c]` posterior
  means across geos) changes by **< 15%** between the two priors when the data is
  informative — i.e. the data, not the prior, drives the heterogeneity. Pair with a
  prior-vs-posterior contraction check (reuse `diagnostics/learning.py`): `tau_c` must
  contract from its prior.

### Cell-isolation sanity (always-on)

- Reuse `tests/test_panel_adstock.py` impulse harness: a unit impulse in one geo must
  produce **zero** contribution in other geos under the geo-varying path (no
  carryover/coefficient bleed across the `geo_idx` boundary). Cheap, fast, non-slow.

### Test files & markers

- `tests/test_geo_varying_media.py`:
  - `test_regression_flag_off_identical` (A1, fast, golden compare)
  - `test_unidentified_geo_fails_loud` (A2 loud-failure, fast — diagnostic + warns path)
  - `test_geo_recovery_beats_pooled` (A3.1, `@pytest.mark.slow`)
  - `test_held_out_geo_backtest` (A3.2, `slow`)
  - `test_geo_lift_coverage` (A3.3, `slow`)
  - `test_no_spurious_heterogeneity_on_clean` (A3 positive control, `slow`)
  - `test_tau_prior_sensitivity` (A4, `slow`)
  - `test_geo_impulse_isolation` (cell-isolation, fast)

---

## 3. DF-2 — Grouped / hierarchical priors for collinear channels

**What changes (from spec §DF-2):** independent per-channel `beta_{channel}` becomes
`beta_c ~ Normal(mu_group, tau_group)` (non-centered) for channels sharing a group,
gated by `ModelConfig.use_grouped_media_priors=False`. Groups come from explicit
`parent_channel`/`media_groups` by default; calibrated channels (`roi_prior`) are
**excluded from group shrinkage** (DF2.4).

### Gate A1 — Zero-change regression (rung 0 → 1)

- Same mechanism as DF-1 A1, on a 4-channel fixture (2 grouped, 2 ungrouped). Golden at
  `tests/golden/grouped_off_golden.nc`; `assert_allclose(atol=1e-10)` with
  `use_grouped_media_priors=False`. Additionally assert **un-grouped** channels'
  `beta_{channel}` are unchanged when the flag is *on* but they belong to no group
  (per spec backward-compat: grouping must not change the meaning of ungrouped betas).

### Gate A2 — No silent over-shrinkage (rung 1)

This is the load-bearing safety gate for DF-2 — the failure that would quietly corrupt a
strong channel's ROI.

- **Procedure.** Add and use `make_multicollinearity_asymmetric` (§1 gap 2): channel `A`
  is collinear with `B` but `A` *also* carries independent spend variation (strongly
  identified); `B` is weak. Fit independent priors, then grouped priors with `A,B` in
  one group.
- **Metric / threshold.** `|beta_A_grouped_mean − beta_A_independent_mean| / |beta_A_independent_mean| < 0.10`
  — the strong channel's posterior must not be materially dragged toward the weak
  neighbor; the adaptive `tau_group` must let it escape the pool. Also: `tau_group`
  posterior must be **non-degenerate** (its HDI must exclude 0 only if the data supports
  pooling; on this asymmetric world it should be wide → little shrinkage). A grouped fit
  that pulls `beta_A` >10% toward `B` is an **A2 fail**.

### Gate A3 — Calibration precedence (rung 1; correctness, not performance)

- **Procedure.** Reuse the `tests/test_calibration.py` harness. On `multicollinearity`,
  set a `roi_prior` (experiment-calibrated) on channel `A`, which is grouped with `B`.
  Fit with grouping **off** and **on**.
- **Metric / threshold.** `A`'s posterior must match between the two runs (within
  `roi_prior` is excluded from shrinkage per DF2.4): mean rel-difference < 5% **and** HDI
  overlap > 0.9. If grouping drags a calibrated channel toward a weak group mean, **A3
  fails** — randomized evidence must win over an observational group prior.

### Gate A4 — Held-out calibration improvement (the rung-2 gate, R0.3)

- **Procedure.** On `multicollinearity` (known `true_contribution` per channel), fit
  independent vs grouped priors. Hold out a lift test on the pooled channels and run
  `CalibrationResults`.
- **Metrics / thresholds (all three must hold):**
  1. **Per-channel interval calibration.** 90% HDI `coverage_rate` of `true_contribution`
     under grouping is **closer to nominal 0.90** than independent priors (independent
     priors over-cover-with-width or under-cover; grouped should tighten toward nominal).
  2. **Total group unbiasedness.** `|Σ_group est − Σ_group true| / |Σ_group true|`
     under grouping ≤ independent + 2pp. Borrowing strength may reallocate *within* the
     group but must not bias the *total* — this is the non-negotiable.
  3. **Held-out lift agreement.** Grouped `coverage_rate` on the held-out lift ≥
     independent, ≥ 0.85 absolute.
- **Decision.** Recommendable as default only if A4 holds; otherwise it ships at rung 1
  (opt-in) as a regularizer for users who already know their channels are collinear.

### Condition-number disclosure (always-on, cheap)

- Report `channel_diagnostics.condition_number` and the detected `collinear_clusters`
  before/after enabling grouping in the test output, so the reviewer sees the
  identification problem the feature targets and that grouping addresses the *reported*
  cluster (R0.5: compose with P2-2, don't bypass it).

### Test files & markers

- `tests/test_grouped_media_priors.py`:
  - `test_regression_flag_off_identical` + `test_ungrouped_beta_unchanged` (A1, fast)
  - `test_strong_channel_escapes_pool` (A2, `slow`)
  - `test_calibrated_channel_excluded_from_group` (A3, `slow`)
  - `test_grouped_interval_calibration` + `test_group_total_unbiased` (A4, `slow`)
  - `test_held_out_lift_coverage` (A4.3, `slow`)

---

## 4. Cross-feature mechanics

### The golden-regression mechanism (A1 for both)

The R0.2 gate is the cheapest, highest-value test and the one most likely to catch a
botched refactor. Implement once as a shared helper:

```
tests/golden/  ── checked-in NetCDF posteriors from the PRE-feature code path
tests/_golden.py ── assert_posterior_matches(idata, golden_path, atol=1e-10)
```

Capture the golden **on `main` before the feature branch** (so it reflects today's
numbers), commit it, then the feature branch asserts against it. If float
non-reproducibility across the refactor makes `atol=1e-10` infeasible, that itself is a
finding — investigate before loosening, and never loosen past `atol=1e-6` without a
documented reason (a real posterior shift hides below that).

### Where the held-out experimental evidence comes from (R0.3)

Neither feature can reach rung 2 on synthetic recovery alone — the spec demands
held-out *experimental* validation. Two sources, in priority order:
1. **Synthetic geo-lift / channel-lift fixtures** derived from the DGP's known
   per-geo/per-channel incremental truth (§1 gaps). These are deterministic and gate CI.
2. **A real dataset with a real lift test**, if/when available (the spec's
   "realistically-simulated or real"). Document the result in this file's changelog;
   do not block the rung-1 ship on it.

### CI cost control

All recovery/backtest tests are `@pytest.mark.slow` and excluded from `make fast_tests`.
Add a `make df_validation` target that runs `tests/test_geo_varying_media.py` and
`tests/test_grouped_media_priors.py` with `-m slow`. Run it in a nightly job, not on
every push. The A1 regression tests and the A2 loud-failure tests are **fast** and run
on every push (they are the rung-1 safety net).

---

## 5. Sequencing

1. **DF-2 first** (effort M, independent of DF-1). Smaller change surface (group
   hyperprior on existing per-channel betas), and its A-gates exercise the calibration
   precedence logic both features share. Land A1+A2+A3 → ship at rung 1. Run A4 nightly;
   promote when it holds on the synthetic lift fixture.
2. **DF-1 second** (effort L). Reuses the golden-regression helper and the held-out
   calibration plumbing built for DF-2. Its DF1.6 gate is already shipped
   (`geo_spend_variation_diagnostic`), so A2's loud-failure path is mostly wiring.

Each is independently shippable at rung 1. Neither blocks the other.

---

## 6. Kill criteria (when to abandon or redesign, not loosen a threshold)

A feature is **pulled back to the drawing board** — not granted a threshold waiver — if:

- **K1.** A1 cannot be made to pass at `atol ≤ 1e-6` (the "off" path is not actually
  off → the refactor changed established results; unacceptable per R0.2).
- **K2.** A2's loud-failure path does not fire on a provably unidentified panel (silent
  garbage on weak identification is the framework's cardinal sin).
- **K3 (DF-1).** On the `geo_clean` / geo×product positive controls, the feature
  manufactures heterogeneity (`tau_c` does not collapse) → it is fitting noise.
- **K4 (DF-2).** A2 shows >10% drag on a strongly-identified channel that cannot be
  removed by widening the `tau_group` prior → the pooling is not adaptive enough.
- **K5 (either).** A4 shows held-out calibration *degrades* vs the independent/pooled
  baseline and the degradation is not explained by a fixable prior/parameterization
  choice → the feature does not earn rung 2; it stays opt-in indefinitely or is dropped.

Kill criteria protect the project's central credibility claim: **we do not silently
change established numbers, and we do not recommend a number we cannot defend against
held-out evidence.**

---

## 7. Definition of done (extends the spec appendix)

A deferred feature is **rung-1 done** when:
- [ ] R0.1 flag exists, default reproduces today.
- [ ] A1 golden-regression test passes (`atol ≤ 1e-10`) and runs on every push.
- [ ] A2 convergence test passes on the identified world; A2 loud-failure test passes
      on the unidentified world; both wired into the fast suite where possible.
- [ ] R0.4 contract preserved (`beta_{channel}` emitted as the summary scalar;
      `channel_contributions` dims unchanged) — assert variable names in the test.
- [ ] R0.6 reporting discloses the assumption + HDIs.
- [ ] Flag documented as experimental in `CLAUDE.md` troubleshooting + the feature's
      builder docstring.

It is **rung-2 done** (recommendable as default) when additionally:
- [ ] A3 (DF-1) / A4 (DF-2) held-out validation passes on the synthetic lift fixture
      and the result + numbers are recorded in this file's changelog.
- [ ] A4 (DF-1) / A2 (DF-2) prior-sensitivity / over-shrinkage gates pass.
- [ ] The `causal-mmm-critique-roadmap` memory is updated to move the item from
      "deferred" to "shipped (rung 2)".

---

*Anchored to the tree as of 2026-06-20. Code references
(`model/base.py:1294-1308` media block, `:1236-1243` geo intercept,
`validation/geo_identification.py:74-123`, `validation/channel_diagnostics.py:26-85`,
`src/mmm_framework/synth/dgp_geo.py`, `src/mmm_framework/synth/dgp.py:450-487`, `tests/synth/harness.py:107-150`) were
verified against source; thresholds are initial values to calibrate on the first
nightly run and are deliberately stated as inequalities so they self-document when
tightened.*

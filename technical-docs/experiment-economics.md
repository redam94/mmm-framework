# Model-Anchored Experiment Economics — Methodology & Build Spec

> Status: implementation spec (adversarially verified). This document is the single
> source of truth for `planning/opportunity_cost.py`, `planning/simulation.py`,
> `planning/design_anchor.py`, the EIG/EVOI loopback, and the full-stack wiring.
> It answers three client asks: (1) use the **latest fitted model** to aid experiment
> design, (2) quantify the **short-term risk / opportunity cost** of deviating media
> from business-as-usual (BAU), (3) run **A/A and A/B simulations** on historical data
> to compare experiment methodologies on power, MDE, and opportunity cost.

## 0. Verified codebase facts that pin every correction

| # | Fact | Source |
|---|---|---|
| F1 | `sample_channel_contributions(X_media, max_draws, random_seed) -> (n_draws, n_obs, n_channels)` evaluates the `channel_contributions` **Deterministic** — NOT a likelihood node → **zero observation noise**. Identical `max_draws` → deterministic thinning → two passes are **draw-paired exactly**; per-draw delta of the perturbed channel is exact. | base.py:2129 |
| F2 | Rows are stacked `(period × geo × product)`. Per-row codes: `mmm.time_idx` (code into `panel.coords.periods`), `mmm.geo_idx` (code into `mmm.geo_names`). National → `geo_idx` all zeros. | base.py:331-352, 548-572 |
| F3 | The agent path is **RAGGED** (`MFFLoader.build_panel` → `from_frame`, observed rows only; `coords.periods` = union across geos). "last `duration` codes" can be absent in some geos. | data_loader.py:632,647 |
| F4 | `_perturbed_contribution_sum`: perturb `X_media_raw[mask,ch] *= (1+lift)`, sum contribution **window-only** (post-window carryover deliberately not counted). | base.py:1460 |
| F6 | `design.geo_lift_design` returns `treatment_geos`/`control_geos` (names from a raw-csv pivot — a **different** pipeline than `mmm.geo_names`), `weekly_spend_delta = abs(...)` (**unsigned**), `intensity_pct` (signed: -100 holdout). | design.py:410,482 |
| F7 | `_placebo_did` = existing A/A (sliding overlapping windows). `_did_se_total_kpi` assumes **iid** first-diff noise. `MDE_FACTOR=2.8`. | design.py:279-315 |
| F8 | `compute_response_curves` scales **all** rows by scalar `m`; `base_spend = X.sum(0)` is **global**. `priority.py roi_draws = contrib_at_current[:,c]/base_spend[c]` = window-total **average** ROAS. | budget.py:63, priority.py:140 |
| F9 | `priority.compute_experiment_priorities(sigma_exp_overrides={name:float})` honored before fallback. `compute_evoi_for_channel(curves, idx, roi_draws, sigma_exp, ...)` takes `roi_draws` directly. | priority.py:172,186 |
| F10 | Model-op contract `{content, dashboard, error, tables}`; `_ok/_err`; `OPS` registry; `op.allow_unfitted=True` attr. | model_ops.py:35-40 |
| F11 | `run_model_op` reads `_MODEL_CACHE["fitted_model"]` keyed by active-thread ContextVar. `asyncio.to_thread` **copies** caller context → set+load+run must be in **ONE** worker call. | kernels.py:120, runtime.py |
| F12 | Design endpoint returns `JSONResponse(safe_json_dumps_load(design))` — no strict `response_model`, extra keys can't 422. | main.py:1910,1938 |
| F13 | `sessions.add_artifact`/`get_artifact` exist; **`update_artifact_payload` does NOT** — add it. | sessions.py:474 |
| F14 | `load_model_core(thread_id, name, spec, dataset_path)` needs `spec.kpi` + `dataset_path`; deposits into in-process `_MODEL_CACHE`. | tools.py:2075 |

## 1. Corrected math (the claims verification flagged)

**OC-1 Ragged window** — recent window = **per-geo intersection** of available period codes, last `duration` of the sorted intersection. Report `duration_effective`; warn when `< duration`. (F3)

**OC-2 Exact paired delta** — `channel_contributions` is Deterministic (F1): per-draw delta has zero sampling noise; sum only the tested channel over treated rows. No seed-pairing needed.

**OC-3 Three distinct quantities** (a single "opportunity cost" scalar is incoherent across families): `kpi_delta` (signed) & `forgone_kpi = max(0,-kpi_delta)`; `net_profit_impact = margin*kpi_delta - spend_delta` (signed); `opportunity_cost_dollar = max(0,-net_profit_impact)` (non-negative downside — the client's "cost"). Headline the downside.

**OC-4 Spend sign** — `spend_delta = spend_exp_total - spend_bau_total` computed **internally**, NEGATIVE for holdout. **Never** import `design['weekly_spend_delta']` (it's `abs()`, F6 — would invert the holdout net).

**OC-5 Learning-to-cost per-week** — `evoi_per_week = EVOI/response_horizon_weeks`; `cost_per_week = |kpi_delta_with_carryover|/duration_effective`; `ratio = evoi_per_week/cost_per_week`. Cap EVOI at EVPI. `cost_per_week≈0` (budget-neutral flighting) → `ratio=None`, basis `net_neutral_design`; EVOI floored → `channel_already_precise`.

**OC-6 Geo name resolution** — `{str(g).strip(): j}` + case-insensitive fallback; **raise ValueError listing missing geos**; assert `mask.any()`. (F6)

**OC-7 Carryover** — report BOTH `kpi_delta` (window-only headline, matches DiD estimand F4) and `kpi_delta_with_carryover` (all treated-geo rows). `cost_per_week` uses with-carryover.

**SIM-1 A/A FPR measured, not assumed** — A/A computes a **design-calibrated critical value** = empirical `(1-alpha)` quantile of `|estimate|` over null windows (block-bootstrap, block ≥ adstock memory); report `fpr_at_nominal` (using `z*analytic_se`) only as an **inflation diagnostic**. (F7 — analytic FPR inflates 2–13× under autocorrelation.)

**SIM-2 Effective windows** — report `n_eff_windows ≈ n/t_test` + Wilson CI; `status='insufficient_windows'` when `n_eff<30`. `fpr_tolerance` adaptive: `max(0.075, 0.05 + 1.645*sqrt(0.05*0.95/n_eff))`.

**SIM-3 MDE via probit fit** — fit `Phi^{-1}(power_i)` on `effect_i` by lstsq, solve 0.80 crossing; isotonic-clean first; `interp_fallback` only when <3 points; `mde_ci` via block-bootstrap.

**SIM-4 Injection contract** — default **model-anchored** injection (perturb spend, read per-cell ΔKPI from `sample_channel_contributions`); additive on treated×test cells only; pre-window/control cells byte-identical. Fixed fallback: uniform-per-cell or ∝spend-share (clipped ≥0), NEVER ∝KPI-level. `injection_basis ∈ {model_anchored, spend_share, uniform}`.

**SIM-5 Injected lift = in-window estimand** — inject AND label the **window-only** ΔKPI (F4); analysis window == injection window. A/A injects exactly 0.

**SIM-6 Fixed assignment across estimators** — build the `Assignment` **once** per family (fixed seed/seed-bank shared across all estimators). Vary estimator within family; cross-family numbers are different DGP slices, not estimator-comparable.

**ANCHOR-1 sigma_exp pairs with the INCREMENTAL estimand** — loopback passes BOTH `sigma_exp_overrides={ch:|se_roas|}` AND `roi_draws` override = `incremental_roas_draws` into `priority.compute_experiment_priorities` (needs a small additive `roi_draws_overrides` kwarg). For a `-100%` holdout the estimand IS average ROAS → keep `roi_sd`. Select by `design_key`.

**ANCHOR-2 Honest treated-row perturbation** — `model_anchored_effect` perturbs ONLY treated geo × test-window rows at full intensity via `sample_channel_contributions` — does NOT interpolate the global `compute_response_curves` (F8 — global curve scales all geos uniformly, mis-states the treated subset under heterogeneous saturation).

**ANCHOR-3 Signed assurance** — `assurance = mean_d[Phi(eff_d/se - z) + Phi(-eff_d/se - z)]` (signed two-sided, NOT `|eff|`), so a null channel scores ≈ alpha. Lead the verdict with `prob_detectable = mean_d(|eff_d|>mde_roas)`. `z=1.959963984540054`.

**ANCHOR-4 sigma_exp clamp** — `sigma_exp ≥ eps*max(|median_roi|, roi_floor)` so flat-placebo `se_roas→0` can't return absurd EIG.

## 2. Public API (see the source files for the authoritative signatures)

- `planning/opportunity_cost.py`: `OpportunityCostResult` dataclass + `compute_opportunity_cost(mmm, design, *, margin_per_kpi, kpi_kind, price, preferences, branding, loss_threshold, evoi_kpi_units, response_horizon_weeks, max_draws, random_seed)`; internal `_resolve_treated_rows`, `build_experiment_media`, `_resolve_margin`.
- `planning/design_anchor.py`: `model_anchored_effect(mmm, design, ...)`, `powered_to_detect(effect, power_curve, duration, se_roas, ...)`, `realized_sigma_exp_for_anchor(incremental_roas_draws, se_roas, ...)`.
- `planning/simulation.py`: `Window`, `Assignment`, `SimPanel`, `EstimatorResult`, `AAResult`, `ABResult`; estimators `pooled_did_estimator`/`per_pair_did_estimator`/`regadj_geo_estimator`/`national_onoff_estimator`; `build_sim_panel`, `run_aa_simulation`, `run_ab_simulation`, `model_anchored_injector`/`fixed_lift_injector`, `methodology_leaderboard`, `simulate_methodologies` (model-op).
- `planning/priority.py`: add `roi_draws_overrides: dict[str,np.ndarray] | None = None`.
- `agents/model_ops.py`: `experiment_economics(mmm, results, *, design_params, run_simulation, margin, kpi_kind, max_draws, random_seed)` (`allow_unfitted=True`).

## 3. Wiring

- **Model-op** `experiment_economics` → `dashboard["experiment_economics"]` = `{channel, kpi, design_key, design_type, duration, randomized, se_roas, mde_roas, se_source, model_anchored, anchor{...}|null, opportunity_cost{...}, simulation{...}, design{...}}`. All scalars `float()`-cast; non-finite → `None`.
- **Agent tools**: enrich `design_experiment_plan` (route through op when fitted; pure fallback pre-fit); new `simulate_experiment` (HEAVY).
- **Endpoint** (NOT ARQ — this app uses per-session kernels): `POST /projects/{id}/experiment-design/simulate` → `{job_id,'pending'}` (background `asyncio.create_task`, strong-ref set); `GET .../simulate/{job_id}` polls. Job target runs `set_current_thread`+`load_model_core`+`run_model_op` in **one** `asyncio.to_thread` (F11), pins `MMM_AGENT_KERNEL=inprocess`. `sessions.update_artifact_payload` added (F13). Synthetic tid `__simjobs__{project_id}` (server-minted).
- **Frontend**: `ExperimentEconomicsPayload` + `model_anchor?` on `ExperimentDesignPayload`; `startSimulation`/`pollSimulation`; `useExperimentSimulation` hook (poll `refetchInterval` until done/error); two `DesignStudio.tsx` panels (Opportunity cost, Methodology comparison).

## 3b. Experiment optimizer (suggest a setup + Pareto front)

`planning/experiment_optimizer.py` turns "test this channel" into a recommended,
runnable setup and a **Pareto front** of designs trading FOUR objectives the
client weighs (all lower-is-better): **lowest MDE**, **highest statistical power**
(as a `power_shortfall = max(0, target − power)` axis, default target 80%),
**smallest short-term tradeoff** (opportunity cost), **shortest duration**.

Power is the model-anchored probability each design detects the expected effect:
`power = mean_d[Φ(eff_d/se − z) + Φ(−eff_d/se − z)]` with `se = mde_roas/2.8` and
`eff_d` the reference incremental-ROAS draws (so at an effect equal to the MDE,
power ≈ 80% by construction). It needs no extra posterior passes. `power_shortfall`
is 0 once a design reaches the target, so above-target designs compete on the
other three axes and below-target ones are pushed toward the bar; the
recommendation prefers the ≥target set. Power unknown (no model effect) → neutral
on that axis.

- `cooldown_weeks(mmm, channel)` — the washout after a holdout (or between
  flights) before the treated cells are back to BAU, from the fitted geometric
  adstock: smallest `k` with `alpha**k < threshold` (default 5%), clamped to
  `[1, 26]`. Unknown adstock → a moderate `default_weeks` (4); no carryover → 1.
  Also sets the minimum flighting block length (a block shorter than the memory
  smears the on/off contrast).
- `evaluate_experiment_grid(...)` — sweeps a design space the caller BOUNDS with
  ranges: durations in `[duration_min, duration_max]` weeks and **signed** spend
  variations in `[intensity_min, intensity_max]` % (`-100` go dark … `+150` scale
  up; a variation at `-100%` renders as a go-dark holdout, negatives are spend
  reductions). `_duration_grid`/`_intensity_grid` auto-sample a few endpoint-
  inclusive points within each range (explicit `durations`/`scaling_intensities`
  override). **National flighting** turns the spend range into the schedule's
  spend LEVELS (multipliers of mean spend): a 2-level on/off pins the average
  ROAS, and a ≥3-level schedule (`flighting_design(levels=…)`) additionally
  traces the response CURVE so the saturation / marginal ROAS is identified. For
  flighting, power is computed for **three estimands separately** — ROAS,
  contribution, mROAS (`_flighting_power_breakdown` + `flighting_estimand_ses`):
  a local quadratic fit `y=g(x)` on the design's spend regressors gives
  `σ²(XᵀX)⁻¹`, from which the LEVEL `g(x₀)` SE → contribution / average-ROAS
  detectability (the same test rescaled by known spend, so those two powers
  coincide) and the SLOPE `g'(x₀)` SE → mROAS power. The tangent mROAS is only
  identifiable with ≥3 distinct levels (binary = a secant, `mroas_identified=
  False`); the binding `min(roas, mroas)` drives the Pareto objective so a
  flighting design must pin the curve, not just the average. The **MDE for every
  duration comes from one pure-pandas power-curve call per (footprint,
  intensity)**; the **opportunity cost reuses a
  single shared BAU posterior pass** across the whole grid (the `contrib_bau`
  kwarg), so the grid costs ~1 BAU + 1 perturbed pass per config. The tradeoff
  axis is the net-$ opportunity-cost downside when a margin is known (a
  money-saving holdout scores ~0 — a true signal), else the forgone KPI.
- `pareto_front(cands)` — non-dominated set over `(mde_roas, tradeoff, duration)`
  (a dominates b iff ≤ on all three and < on one); non-finite objectives can
  never be on the front. `recommend(...)` picks the **knee** (closest to the
  ideal point in front-normalized objectives) among the **powered** front
  designs (MDE ≤ the model's reference incremental ROAS), or all front designs
  if none are powered. Indexed by `CandidateEval.index` via a map (robust to
  index≠position).
- `suggest_experiment(...)` — orchestrates the above and returns the recommended
  design's runnable setup: test/control geo groups (or flighting schedule +
  block length), intensity, duration, and the cool-down.

Wiring: model-op `experiment_optimizer` (`agents/model_ops.py`, requires a fit);
agent tool `suggest_experiment`; non-blocking `POST
/projects/{id}/experiment-design/optimize` + `GET .../optimize/{job_id}` (reuses
the generalized `_load_and_run_op` / `_run_model_op_job` / `_spawn_job_task` job
machinery); React DesignStudio "Optimize (Pareto front)" panel (scatter +
recommended-setup card + front table). Tests:
`tests/test_planning_experiment_optimizer.py`.

## 4. Build order

1. `opportunity_cost.py` core mechanics (`_resolve_treated_rows`, `build_experiment_media`) — reused everywhere.
2. `opportunity_cost.compute_opportunity_cost` + `_resolve_margin`.
3. `priority.py` `roi_draws_overrides` + `design_anchor.py`.
4. `simulation.py`.
5. `model_ops.experiment_economics`.
6. `agents/tools.py`.
7. `sessions.update_artifact_payload` + `api/main.py` endpoints.
8. Frontend.
9. Tests + docs.

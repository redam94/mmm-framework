# Non-Geo Experiment Methods — Ghost Ads + Switchback (Phase 2)

> Status: implementation spec. Companion to `experiment-methods-framework.md`
> (Phase 1). Adds two methods that do **not** ride the geo panel: user/ad-server
> level ghost ads (individual randomization) and time-randomized switchback
> designs. Both register through the Phase-1 `planning/methods/` registry with
> their own `DataRequirement`, so the DesignStudio, agent, and loop treat them
> uniformly with the geo methods.

## 0. Why these are different

The geo methods (Phase 1) operate on a `SimPanel` (geo × week KPI). Ghost ads
and switchback have fundamentally different units of randomization and therefore
different power math:

- **Ghost ads** — randomization is at the **user/impression** level via the ad
  server (treated users see the real ad; "ghost"/PSA control users are *eligible
  and would-have-been-served* but see a placebo). The estimand is a
  **conversion-rate (or per-user count) lift**. Power is a two-proportion / count
  problem driven by baseline conversion rate, users reached, and the ghost/treated
  split — **no time series, no fitted MMM required**. This makes it a clean
  **standalone, pre-fit calculator**.
- **Switchback** — randomization is over **time blocks** at the national (or per
  unit) level: the treatment is toggled on/off on a schedule, and the estimand is
  the on-vs-off contrast. The dominant power driver is **carryover + temporal
  autocorrelation**, so the analytic i.i.d. SE badly understates variance; power
  must be block-bootstrap calibrated (the `national_onoff_estimator` + A/A path
  from Phase 1 already models this; switchback formalizes the *design* — block
  length ≥ carryover memory — and the calculator).

## 1. Verified facts

| # | Fact | Source |
|---|---|---|
| N1 | The Phase-1 registry `MethodSpec` supports `family` in `DataRequirement` — `'user'` and `'switchback'` slot in without touching geo code. | `experiment-methods-framework.md` §2.1 |
| N2 | `national_onoff_estimator` (`simulation.py:220`) already does the on/off contrast + block noise; A/A calibration handles autocorrelation via block bootstrap. Switchback reuses this analysis path. | `simulation.py:220, 341` |
| N3 | `cooldown_weeks(mmm, channel)` (`experiment_optimizer.py:268`) derives carryover washout from fitted adstock — the minimum switchback block length. | `experiment_optimizer.py:268` |
| N4 | Design endpoint returns loose JSON (extra keys can't 422); options endpoint enumerates methods. | `main.py:3388, 3356` |
| N5 | The blog `docs/blog-switchback-experiments.html` already narrates the switchback carryover/timing tradeoff — link it as the method `references`. | `docs/` |

## 2. Ghost ads — `planning/methods/ghost_ads.py`

A **standalone power/MDE calculator** for individual-level incrementality. No
panel, no fitted model — it takes design inputs and returns sample-size / power /
MDE. It is a `MethodSpec` with `DataRequirement(family="user", needs_panel=False,
needs_pre_period=False)`, so it appears in `design_options` whenever the user
declares user-level reach (not gated on geos).

### 2.1 Inputs (a small pydantic `GhostAdsDesign`)

- `baseline_rate` — control conversion rate `p0` (per user), OR `baseline_mean` +
  `baseline_dispersion` for a count/Poisson-Gamma outcome.
- `users_reached` — total eligible users (treated + ghost).
- `treated_fraction` — share served the real ad (default 0.5; ghost = remainder).
- `outcome` — `'binary'` (conversion) | `'count'` (purchases/user) | `'revenue'`
  (mean value/user, with a variance input).
- `alpha` (0.05), `power_target` (0.80), one/two-sided.
- `cost_per_user` and `value_per_conversion` (optional) → feeds Phase-3 economics.

### 2.2 Math (closed form + simulator)

**Binary (two-proportion).** MDE (absolute lift `Δp`) at target power:

```
n_t = treated users, n_c = ghost users
se(Δp̂) = sqrt( p0(1-p0)/n_c + p1(1-p1)/n_t )   # p1 = p0 + Δp
Δp_MDE solves  Δp = (z_{1-α/2} + z_{power}) · se(Δp̂)     # solved by fixed-point
```

Report absolute and **relative** lift MDE (`Δp/p0`), the implied incremental
conversions, and — inverting — the **users required** to detect a target lift.
Also report an exact-ish check via a Monte-Carlo simulator (binomial draws) so the
normal approximation is validated near small `p0` (rare-event regime, where the
normal approx is optimistic).

**Count (Poisson / NB).** SE from the Poisson (or Gamma-overdispersed) variance;
same `z`-based MDE. **Revenue** — mean lift with a supplied per-user variance
(or a Gamma-Gamma-style CV); Phase 4's LTV model can supply a per-user value
distribution to make this an **LTV-based** ghost-ads power calc later.

**Ghost-ads-specific correction.** Only a fraction of eligible users actually
*convert-eligible* window; expose an `exposure_rate`/`intent_to_treat` deflation
so the calculator reports both ITT and treatment-on-treated MDE (the standard
ghost-ads dilution).

### 2.3 API

```python
def ghost_ads_power(design: GhostAdsDesign) -> dict
    # -> {mde_abs, mde_rel, incremental_conversions, users_required_for_target,
    #     se, power_at_target_mde, itt_mde, tot_mde, method='ghost_ads', ...}
def ghost_ads_users_for_mde(design, target_lift) -> int
def ghost_ads_simulate(design, true_lift, *, n_sims, seed) -> dict  # empirical power/FPR
```

`estimator_fn` for ghost ads is a **user-level two-proportion / count estimator**
(not the `SimPanel` contract) — the registry marks it `family="user"` so the
generic geo A/A·A/B harness is skipped; its own `ghost_ads_simulate` provides the
empirical power/FPR instead.

## 3. Switchback — `planning/methods/switchback.py`

Formalizes the time-randomized design already half-present via
`national_onoff_estimator` + `flighting_design`.

### 3.1 Design generator

- `switchback_design(dataset_path, kpi, channel, *, block_weeks=None, duration,
  budget_neutral=True, seed)` — builds a randomized on/off (or high/low)
  block schedule. **Block length defaults to `cooldown_weeks` (N3)** so each
  block exceeds the carryover memory (a block shorter than adstock memory smears
  the on/off contrast — the calculator warns and the power collapses).
- Returns `schedule_mult` (the `Assignment.schedule_mult` the
  `national_onoff_estimator` already consumes, M2/N2) + block boundaries for the
  UI calendar.

### 3.2 Power

- Analytic: on/off contrast SE inflated by the AR(1)/block **design effect**
  (reuse `identification.ar1_design_effect`, `identification.py:115`) — this is
  the honest switchback SE.
- Calibrated + empirical: the Phase-1 A/A·A/B harness with
  `national_onoff_estimator` and a **block bootstrap** (block ≥ carryover memory)
  gives the true FPR and empirical MDE under autocorrelation — exactly the
  regime switchback lives in.
- Report the **number of switches** and the tradeoff: more switches → more
  independent contrasts (↑power) but more carryover contamination near
  boundaries (the blog's timing story, N5). Include a recommended block length +
  buffer (drop the first `k` weeks of each block from analysis, `k` from adstock
  memory).

### 3.3 API

```python
def switchback_design(...) -> dict           # schedule + block plan
def switchback_power(design, *, sigma_y, design_effect) -> dict   # analytic MDE
# analysis reuses simulation.national_onoff_estimator (registered under 'switchback')
```

Register `switchback` into `_NATIONAL_ESTIMATORS` (pointing at the on/off
estimator) so the leaderboard runs it; `DataRequirement(family="switchback",
needs_panel=False)` (needs a national time series, not a geo panel).

## 4. DesignStudio surface

- A **standalone "Power calculator" panel** for ghost ads: pure inputs → MDE /
  users-required, no model/dataset needed (works on the Experiments page before
  any fit). This is the smallest self-contained deliverable and can ship first
  within Phase 2.
- Switchback joins the existing method picker (Phase 1) with a **block-length +
  switch-count** control and a schedule calendar (reuse `FlightingSchedule.tsx`).
- Both feed the same result cards + (Phase 3) economics.

## 5. Wiring

- **`planning/methods/ghost_ads.py`**, **`planning/methods/switchback.py`** —
  new, registered in `methods/__init__.py`.
- **`planning/simulation.py`** — register `switchback` → `national_onoff_estimator`
  in `_NATIONAL_ESTIMATORS`.
- **`agents/tools.py`** — a `ghost_ads_power` tool (pre-fit, `allow_unfitted`);
  switchback flows through `design_experiment_plan` with `method="switchback"`.
- **`src/mmm_framework/api/main.py`** — `POST /projects/{id}/ghost-ads/power` (stateless calc; no
  job needed) + switchback via the existing design endpoint.
- **Frontend** — `GhostAdsCalculator.tsx` panel; switchback controls in
  `DesignStudio.tsx`.
- **`synth/`** — a user-level ghost-ads DGP (binomial conversions with a known
  lift) for the simulator's recovery test; switchback rides the national DGP.

## 6. Build order

1. `ghost_ads.py` closed-form + simulator (standalone; no repo coupling) + tests.
2. Ghost-ads agent tool + `/ghost-ads/power` endpoint + `GhostAdsCalculator.tsx`.
3. `switchback.py` design + block-length-from-adstock + power (reuse ar1 design
   effect + on/off estimator).
4. Register both in the Phase-1 registry + leaderboard.
5. DesignStudio switchback controls; docs cross-links.

## 7. Tests — `tests/test_nongeo_methods.py`

- Ghost ads: two-proportion MDE matches a known closed-form reference; simulator
  empirical power ≈ analytic at the MDE; users-for-MDE inverts consistently;
  rare-event regime flagged where normal approx diverges from simulation; ITT vs
  TOT dilution correct.
- Switchback: block length ≥ adstock memory enforced; design effect inflates SE
  above i.i.d.; A/A FPR under autocorrelation calibrated to alpha; empirical MDE
  finite and larger than the naive i.i.d. MDE.
- Registry: both gated by `methods_for_data` (ghost ads needs user counts;
  switchback needs a national series; neither offered as a geo method).

# Changelog

All notable changes to `mmm-framework` are documented here. The project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html): the major version changes when a
frozen public contract breaks, and the contract itself is pinned by
`tests/test_api_contracts.py` and `tests/test_lean_imports.py`.

> **Maintainers:** every entry added here must also be mirrored into
> [`docs/changelog.html`](docs/changelog.html) — the public documentation site — in the same PR
> as the version bump, including moving the `Current` chip onto the new release. That page is
> hand-authored HTML; nothing generates it from this file and no test gates it, so it rots
> silently (1.1.0 and 1.2.0 both shipped while the site still announced 1.0.0). The full
> checklist is in `CLAUDE.md` § "Release checklist".

## [Unreleased]

### Added

- **The graph-fingerprint safety net** ([#279] PR 0.1 — the entry gate for the src refactor;
  spec `technical-docs/src-refactor.md`). Structure-only fingerprinting produces a false
  negative — `michaelis_menten` and `tanh` saturation collide on a structural hash, and only
  the numeric block (per-factor logp + Deterministic sums/abs-sums at a name-offset probe
  point) separates them; the test proving that pair is part of the contract.
  `tests/contracts/graph_fingerprint.py` (the engine, all four spec gotchas kept),
  `tests/contracts/model_matrix.py` (38 importable case builders across every
  trend/saturation/adstock family, geo/geo×product panels, both prior modes, levers, extended
  models and the agent spec path — with `binomial_refused` pinning a refusal as its golden),
  38 full-dict goldens (readable diffs; regen behind `MMM_REGEN_FINGERPRINTS=1`), and
  `tests/contracts/invariants.md` — the numbered invariant table (39 rows, file:lines
  re-verified). The matrix caught a real silent no-op on its first run: `control_selection`
  fingerprinted identical to the default model because selection activates only with ≥2
  selectable controls. 43 unmarked tests, ~44 s warm.

[#279]: https://github.com/redam94/mmm-framework/issues/279

- **The feature-showcase notebook series** (`nbs/showcase/` 00–06). Seven chart-first
  notebooks covering every subpackage in `src/mmm_framework/` — the measurement loop
  end-to-end in miniature, data foundations, model anatomy (every transform family
  charted), the extension families, the trust toolchain (estimands, diagnostics,
  validation, calibration on a deliberately confounded world), the v1.4 plan-and-commit
  loop, and the operating surface (experiments, the geo bandit, reporting, platform
  services). Baked with zero error outputs; a coverage sweep verified no package is
  absent (cloud-edge modules — integrations, storage — appear as prose with real call
  shapes rather than live network calls). Registered in `nbs/README.md`.

### Fixed

- **Async numpyro jobs failed at their saving stage, every time** (found by the showcase
  series' own demo). The fit succeeded, then `jobs.py`'s worker hit stdlib pickle's
  `'functools.partial' object has no attribute '__name__'` on the fitted model and the job
  died as `failed`. `jobs.py` now serializes with cloudpickle — already a core dependency
  and what `MMMSerializer` standardized on; cloudpickle reads stdlib-pickle files, so
  existing job directories still load. Pinned by a fast invariant that fails on the
  unfixed code plus a slow end-to-end job-completion test (`tests/test_jobs.py`).

- **`eligible['alpha']` is now design-gated on temporal contrast** ([#293], split out of
  [#224]). `planning/identification.py` marked the carryover decay eligible unconditionally —
  the temporal-contrast requirement lived only in the module docstring — so a constant-spend
  channel (zero level changes inside the window, alpha entirely prior-determined) was still
  reported alpha-eligible, and the payback surface had to build its own posterior-side
  prior-domination gate because the design-side one was missing. The gate counts separated
  week-over-week level changes (mirroring how `lam` is gated on spend-level contrast) and
  requires at least one — a step's decay tail is the minimum that expresses carryover inside
  the window; pulse-count beyond that is a power question, not eligibility. The result carries
  `n_transitions` + the threshold so readouts can say WHY alpha was gated. Regression: a
  constant 1.3× schedule comes back alpha-ineligible (verified to fail on the unfixed code).

[#293]: https://github.com/redam94/mmm-framework/issues/293

## [1.4.0] — 2026-08-06

**Finance-grade planning** (epic [#195]): the measurement loop now closes at the CFO's desk. A KPI
valuation that is never silently one dollar; a forward forecast whose caveats lead its headline; an
append-only, hash-chained plan of record that reproduces from provenance to 1e-9; payback horizons
that ship with their truncation and prior-domination disclosures; promo-depth optimization with
per-arm cost bases; and a variance-to-plan bridge that sums to actual − committed exactly and
refuses the refit "effectiveness" split. Throughout, the design language is the refusal: fund-to-
breakeven without a valuation, promo ROI on a flag promo, a price recommendation, a bridge over a
partially-realized window — each is a stated refusal, not a silent default.

**Behaviour changes to read before upgrading:** free-mode budget allocation raises
`UnresolvedValueError` without a valuation; forecasts refuse when a model has controls and no
future values were supplied; decomposition shares are computed against the signed total (previously
rendered "% of total" figures meant something else); the CFO baseline is the model's fitted
non-marketing outcome with the residual named; multiplicative models refuse
`sample_channel_contributions()` instead of returning log-scale numbers; the report and the slide
deck resolve one shared break-even, which changes tier recommendations for projects with a saved
margin.

### Added

- **Enablement: the gates that make silent unreachability and artifact drift fail CI** ([#228]).

  Three failure modes kept recurring, all invisible to CI: a shipped capability nobody can invoke
  (price levers spent a release as config + builder + spec + passing tests while no agent spec
  path read them), three hand-maintained API artifacts that each new endpoint silently staled
  (the checked-in `openapi.json` was 11 operations behind the live app), and docs registration
  that fails by rendering nothing (a nav page without an audience tier, a changelog page
  announcing a version two releases old).

  - **`tests/test_capability_reachability.py`** — every `ModelConfig` field is either wired to a
    spec path `unconsumed_spec_path` accepts or allowlisted with a substantive reason; every
    model op is either mapped in `session_export._OP_TOOLS` or allowlisted with the reason replay
    cannot be deterministic; the tool registration sets (TOOLS / HEAVY / MMM-only / causal /
    milestone / export map) must agree. The allowlists are the trap the issue named: empty
    reasons and non-reasons ("TODO", "later", "n/a") fail, and — the self-cleaning half — an
    allowlisted capability that BECOMES reachable fails until its row is deleted. Planted-omission
    tests prove the gate detects a dummy field and a dummy op by name.
  - **`scripts/sync_api_surface.py` + `make api-sync`** — regenerates
    `tests/contracts/rest_routes.json` (canonical order), `docs/shared/openapi.json` (info block
    overridden: version from `mmm_framework.__version__`, provenance description re-injected — a
    naive re-export downgrades both), `EXPECTED_OPS`, and `docs/rest-api.html`, idempotently;
    prints ADDED/REMOVED routes; **refuses to write while any `/projects/{project_id}` route
    lacks a `_proj_*` tenant guard**. Its first run caught a real one:
    `POST /projects/{id}/plan-of-record` shipped with only a rate limit — any authenticated
    principal could commit a plan of record into another org's project. Fixed
    (`_proj_write` added) and now impossible to reintroduce silently
    (`tests/test_api_surface_sync.py`).
  - **Contract fixes** — `FROZEN_ENUM_VALUES` now freezes `InferenceMethod` (which carries the
    paradigm and is serialized into specs and run metrics; note `frequentist_ridge`/`cvxpy` are
    InferenceMethod values, NOT FitMethod — the obvious edit was a no-op), `LikelihoodFamily`,
    `LinkFunction`, `MeasurementUnit`, and the previously-missing `root` saturation value.
  - **Docs gates extended** — every `NAV_GROUPS` page must exist on disk and carry a
    `PAGE_TIERS` audience tier (`tests/test_docs_nav_registration.py`); a new
    `tests/test_docs_versions.py` fails when `docs/changelog.html` announces a version other
    than `pyproject.toml`, when the `Current` chip count is not exactly one, or when any
    `mmm-framework==X.Y.Z` pin across the site is stale.
  - **Read the Docs** — `frequentist` and `finance` API pages existed nowhere (v1.3.0's headline
    subpackage was absent from the reference); both now have `.rst` pages + toctree entries,
    `cvxpy` joined `autodoc_mock_imports`, and `mmm_framework.frequentist` joined the
    lean-import gate — before which the cvxpy entry in `BLOCKED_PACKAGES` guarded nothing.
  - **v1.4 documentation** — seven `technical-docs/` specs that did not exist:
    `kpi-valuation.md`, `planning-calendar.md`, `forward-forecast.md`, `plan-of-record.md`,
    `payback-and-carryover.md`, `lever-optimization.md`, `variance-to-plan.md` (all
    snippet-gated), plus `docs/finance-planning.html` (the planning loop with its refusal
    index, registered in nav/tiers/SEO), a valuation-refusal correction to
    `docs/workflow-budget-optimization.html` (its uplift example read in dollars
    unconditionally; the shipped optimizer refuses `mode="free"` without a valuation and the
    uplift is KPI-units otherwise), and the `docs/modeling-guide.html` promo passage now points
    at the decision-arm layer that actually computes a promo ROI — and says what it refuses.
  - **`technical-docs/integration-checklist.md`** — the full touch-point list (core → spec
    registry → op → tool sets → export map → REST + guards → FE → reporting → persistence →
    docs → verification), so "register an agent tool" stops being seven undiscoverable edits.

- **Sensitivity to unmeasured confounding, on the decision scale and for experiments too.**
  Spec: `technical-docs/confounding-sensitivity.md`.

  The framework already reported a Cinelli–Hazlett robustness value per channel, but on the
  *coefficient* scale and with no way to say whether the confounder strength it demands is
  plausible; experiments had nothing at all, so a matched-market DiD readout anchored the model
  exactly as firmly as a randomized one. Following [the PyMC Labs treatment][pymc-sens], the
  observed effect is now decomposed into a causal part and a bias part with a **prior on the
  bias**, and the report says which conclusions survive which priors.

  - **Engine** `diagnostics/bias_sensitivity.py` — a **tipping point** ("TV's ROI would have to
    be overstated by more than 24% of its own size to stop clearing break-even"), a
    `(mu, sigma)` **sensitivity surface**, and a VanderWeele–Ding **E-value** restricted to
    ratio measures (an ROI is not a risk ratio, and the function refuses rather than returning
    an uninterpretable number). Everything is closed-form — the de-biased posterior is a
    Gaussian mixture over existing draws, so it is deterministic, exact at small draw counts,
    and the identical arithmetic runs in a browser.
  - **MMM** `validation/confounding_sensitivity.py` — per-channel tipping points against the
    measurement-aware break-even reference, plus **Cinelli–Hazlett benchmarking** against the
    covariates that *were* measured, which is what turns a slider into an argument: "a
    confounder as strong as Price implies 9%, well inside the 24% it would take".
  - **Experiments** `planning/experiment_sensitivity.py` — the bias prior is *measured* from
    the design's own placebo distribution rather than asserted, and the threat model follows
    **assignment, not the estimator**: a randomized geo split faces interference and concurrent
    shocks, not confounding, and is told so instead of being handed a confounding tipping point.
  - **Propagation** — `ExperimentMeasurement` gains `bias_mu` / `bias_sigma` / `bias_scale` /
    `bias_source`, consumed by `attach_experiment_likelihood`, so a fragile readout stops
    anchoring the MMM as if it were clean. Opt-in via
    `apply_experiment_calibration(bias_mode="stored")`; the registry keeps the raw measurement
    and only the staged spec entry carries the assumption.
  - **In-graph sweep** (opt-in, exact) `run_confounder_sweep` — the post-hoc layers re-weight a
    posterior and so move every channel the same way; this puts a *fixed* hypothetical confounder
    into the graph and re-fits, so coefficients move relative to one another. On
    `unobserved_confounding` at assumed partial R² 0.25, TV falls 2.83 → 2.18 while Display rises
    1.95 → 2.19. `BayesianMMM.add_latent_confounder` keeps the default graph byte-identical (the
    term is absent, not zero). Refuses by name for extension models, the frequentist paradigm,
    multiplicative specs and GP trends.
  - **Surfaces** — agent op `confounding_sensitivity`, tool `run_sensitivity_analysis`,
    `POST /projects/{id}/validate` with `check="sensitivity"`, an Oracle Validation-tab button,
    a tipping-point table in `CausalAssumptionsSection`, and three charts in
    `reporting/charts/sensitivity_bias.py`.

  Three decisions worth recording because they are the ones that would have produced confident
  wrong numbers:

  1. **No per-draw multiplicative bias.** `tau_d = x_d * (1 - b_d)` is the obvious way to say
     "relative" and is a trap: an efficiency channel's break-even reference is 0 and every media
     coefficient here has positive support, so `P(tau > 0) = P(b < 1)` **exactly** — the same
     number for every channel, with no data in it. The shipped `fraction_of_mean` scale
     (`tau_d = x_d - b*|mean(x)|`) has none of that, and a regression gate pins it.
  2. **The sensemakr validity condition is `r2dxj_x < 1/(1+kd)`**, not the `kd*r2dxj_x < 1` one
     might reach for — at `kd=1` that is 0.5, not 1.0. Past it the published formula returns
     `NaN`, and a `NaN` compares `False` against every fragility test, i.e. it would read as
     "not fragile". Breaches are now explicit refusals naming the largest admissible `kd`.
  3. **The benchmark uses an OLS standard error from the design matrix, never the posterior sd.**
     The bias identity is calibrated on `se*sqrt(df) = ||y_res||/||d_res||`; an informative
     positive-support media prior shrinks the posterior sd, which would shrink the implied bias
     and manufacture robustness. Validated against the exact Cinelli–Hazlett identity — fitting
     with and without a known confounder reproduces the predicted bias to ~1e-15.

  Tests: `tests/test_bias_sensitivity.py`, `tests/test_confounding_sensitivity.py`,
  `tests/test_experiment_sensitivity.py` (planted-truth gates on the
  `unobserved_confounding` / `confounding_controlled` synth pair).

[pymc-sens]: https://www.pymc.io/projects/examples/en/latest/causal_inference/sensitivity_unmeasured_confounding.html


- **Per-channel payback horizon: per-draw intervals, truncation disclosure, and a
  prior-domination gate** ([#224]). "When does a dollar pay back" is the most CFO-legible number
  the model can emit and rests on its least identified parameter — so nothing here ships without
  its epistemics attached.

  `planning/payback.py::channel_payback()` reports ONE named quantity: the per-draw interpolated
  lag at which the fitted carryover kernel crosses 50% (`t50`) and 90% (`t90`) of its total —
  built on `transforms/carryover.py`'s family-aware per-draw reader (#218) via a new
  `carryover_crossing_lags(kernel, share)` generalization, so t50/t90 cannot drift apart the way
  the four shipped "half-life" definitions did. Intervals are the ETI **of the per-draw
  transform**, never the transform of a mean parameter (the crossing is convex in α: measured, the
  two differ by ~0.4 lags at the extremes); no public function derives a horizon from
  `mean(alpha)`.

  Every result carries what the number cannot travel without:

  - **The truncated tail mass.** `normalize=True` renormalizes the truncated kernel to sum 1,
    redistributing the untruncated tail INSIDE the window — geometric α=0.8 at the default
    `l_max=8` discards 13.4% of its mass and reads t90≈6.8 against an untruncated ≈10.3, under
    the canonical lag-interval convention. Tails ≥10% promote the disclosure to a sentence.
  - **A carryover-learning verdict.** `reporting/evidence.py` gains
    `_CHANNEL_CARRYOVER_PREFIXES` + `carryover_learning()` — the extension point the issue names.
    Deliberately NOT folded into the ROI evidence tier: the ROI tier answers "is the effect SIZE
    evidence-backed?", and a slow-to-learn alpha does not bear on that; the payback surface gates
    on exactly the parameters IT depends on. Two defensible default priors ship with 2x different
    implied half-lives (Beta(1,3) vs Beta(2,2)), and a test pins that two fits differing only in
    that prior produce a flagged verdict, not two equally confident horizons.
  - **An autocorrelation gate, two tests deep.** The residual Ljung-Box alone is insufficient:
    measured on `adstock_misspec` (NUTS, 156w) it reads p=0.16 while the posterior-predictive
    lag-1 check is extreme at p=0.01. Both run; either fires the downgrade, with the DIRECTION
    stated — a truncated kernel cannot express mass beyond `l_max`, so on such fits every horizon
    is BIASED SHORT (measured: fitted t90 5.2–7.1 against planted 9.1–11.9, short on 4/4
    channels).
  - **Refusals by name.** Extension models (one hardcoded geometric family), `StructuralNestedMMM`
    with AR(1) mediators (persistence lives in the state's ρ, on a stated ridge with α), and
    dual-stock brand models (detected by their registered `brand_retention`/`long_term_fraction`
    variables, not class name — `adstock_alpha_<ch>` there is only the FAST stock, so a horizon
    read off it is dramatically too short while the model says the opposite).
  - **Provenance.** Frequentist fits flip the interval noun to *confidence*; MAP/ADVI single-draw
    fits report the interval as **collapsed/absent** rather than `[x, x]` (#249); the serializer's
    `metadata.json` and `planning/history` run metrics (schema v4) record basis / family / l_max /
    tail mass so a reloaded model cannot re-derive a horizon on a different basis.

  `payback_breakeven()` is the separately-named **finance** sense — cumulative discounted dollar
  return per dollar of spend reaching 1, per draw, with `prob_never` for draws that never repay.
  It refuses on efficiency-measured channels (`MetricMeta.is_monetary` False, #221) and raises
  `UnresolvedValueError` without a valuation — never a silent `value_per_kpi=1.0`.

  `planning/discount.py` extracts the repo's two disagreeing discount implementations
  (`experiment_value`'s per-week mean weight; `bayesian_clv`'s mid-horizon factor) into
  `discount_weights()` / `mid_horizon_discount_factor()`, byte-compatible with both, default rate
  **0.0** (measured: a 10%/yr rate moves repo-horizon numbers by 0.33%–2.4% — an assumed nonzero
  rate would be a silent input to every payback).

  Surfaces: report bundle field + classic/Augur sections + interactive card, agent op
  `payback_horizon` + tool `get_payback_horizon` + session-export entry, REST
  `POST/GET /projects/{id}/planner/payback[/{job_id}]`.

  Answer-key fixes that make the criteria gradeable: `make_realistic` now exports `true_adstock`
  (it bypassed `_finish` and had NO carryover truth), and `make_adstock_misspec` exports the
  planted **Weibull** truth instead of inheriting the geometric `_ALPHA` table — its answer key
  used to describe the model's family rather than the planted kernel.

  Measured recovery: on `clean` (NUTS 4×800, seed 7) the 90% interval covers planted t50 on 4/4
  channels (t50 truth/est — TV 1.38/1.42, Search 0.62/0.63, Social 0.83/1.04, Display 1.00/1.06);
  on `realistic` 6/7 with the miss on the deliberately near-collinear Print. Tests:
  `tests/test_planning_payback.py` (34).

- **Promo-depth optimization: per-arm cost bases, a profit objective, and the world that grades
  it** ([#226]). The optimizer's decision vector was dollars of media spend — no slot for a
  decision whose cost is margin given away rather than a spend line. The epic's original headline,
  "trade a price cut against media", is deliberately NOT shipped as a recommendation: the repo's
  own published measurement (`docs/blog-modelled-one-p.html`) recovers **39%** of a planted price
  elasticity confidently. **Promo depth is the shipping headline; price is a labelled what-if
  evaluator that refuses to recommend.**

  The governing reduction (`planning/decision_arms.py`): every arm re-parameterized by its
  **realized cost** — media at its spend (identity; the media-only path is bit-identical, pinned
  by test), a promo arm at `depth × unit_cost`. The existing allocator then runs untouched: the
  budget constraint is already `Σcost = B`, `mode='free'` already maximizes `value·KPI − Σcost`,
  and the KKT water level is again a single number because the decision space is homogeneous
  dollars — the per-group shadow-price correction the issue anticipated for a level-space vector
  is made unnecessary by construction. `DecisionArm` / `ArmCurves` / `build_arm_curves` /
  `optimize_arms` (result rows gain `arm_kind` / `level_units` / `optimal_level`), plus
  `promo_roi` and `price_whatif`. `BayesianMMM.sample_lever_contributions` is the new
  per-draw reader for `price_component` / `promo_component` (original KPI units — the graph
  registers them standardized, and the `× y_std` is load-bearing).

  The analytic gate: a two-arm fixture with per-arm margins (0.6, 0.2) whose closed-form
  equal-marginal-PROFIT optimum (750, 250) lands exactly on grid knots — the allocator hits it to
  1e-6 on both the greedy and constrained paths, and the margin-blind equal-marginal-KPI optimum
  is (250, 750), flipped, proving the objective changed.

  **What refuses, and why:**

  - `promo_roi` on a **0/1 event flag** (no depth ⇒ no ΔP×Q cost; a ratio with no units) and on a
    column **outside [0, 1]** (unknown units; the model's internal max-normalization must not be
    priced).
  - `promo_roi` / fund-to-breakeven **without a valuation** — and the agent layer's
    `value_per_kpi` default changed from a silent `1.0` to `None`: the chat tool could previously
    reach `optimize_budget(mode='free')` only with a fabricated $1/KPI, making the planning
    layer's own refusal unreachable.
  - `compute_response_curves` on a **non-monetary channel**: `base_spend` is
    `X_media_raw.sum(axis=0)`, so an impressions channel was summed into a dollar budget, bounded
    in "dollars", and traded at one shadow price. Now a named refusal pointing at the CPM/CPC fix.
  - `goal_seek` on a **mixed-arm portfolio**: its monotone-frontier bisection is proven for
    concave spend curves; a promo arm's depth response need not be concave.
  - `price_whatif` **always** refuses to emit a recommendation, with the 39% measurement in the
    message — evaluation of a stated hypothetical only.

  **Concavity is finally checked** — the greedy allocator's own docstring said "exact for concave
  curves" since it shipped, and nothing verified it. `check_concavity` (second differences of the
  interpolant) runs on every `optimize_budget` call; a failing arm forces the #290 multi-start
  constrained solver with a note.

  **The worlds** (`synth/dgp.py`): `make_promo_and_media` — planted `elasticity·log(price/ref)` +
  `amp·geom_adstock(depth, α)` exactly in the model's lever family, media near-saturated at
  current spend, and the **answer key frozen before any optimizer runs** (`gross_margin`,
  `promo_unit_cost`, `true_optimal_split` computed from DGP parameters alone). Measured planted
  optimum: promo share 55% of outlay against an observed 19%. `make_promo_endogenous` — LAST
  week's soft demand triggers this week's deeper promo and price cut, so the naive lift attenuates
  (measured 89% → 70% recovery) and `diagnostics/endogeneity.py` — extended to walk lever columns
  with a `kind` field — flags the lever with the clearance mechanism named. Both registered in
  `SCENARIOS` and `PRIORITY`.

  **The milestone criterion, measured**: the joint solve beats the media-only solve on TRUE
  planted profit (noiseless structural mean, frozen economics) for **10 of 10 seeds**; the
  recommended promo share moves toward the planted split (measured 19% → 28% toward 55%); and a
  model without the promo lever misses the split by >15pp — the discrimination that keeps the
  world from being decoration. Two findings the tests preserve: on MAP point fits the milestone
  scored 8/10 (two seeds under-fit media saturation, saw phantom headroom, and the joint solve CUT
  profitable promo — a point estimate of a saturation curve is exactly the input this allocator
  should not be trusted with), and the profit claims are labelled conditional on the stated
  economics throughout.

  Frontend: Planner allocation tables gain **Kind** and **Level** columns whenever a non-media arm
  is present, so a promo row's dollars (margin given away) cannot be misread as a media buy.
  Tests: `tests/test_decision_arms.py` (31). Notebook:
  `nbs/demos/promo_depth_optimization.ipynb` (baked, 19 cells, 0 errors).

- **A realized-KPI actuals store — the hard blocker for variance-to-plan** ([#227], first
  deliverable; the bridge itself follows). There was no realized-KPI record anywhere — `delivery`
  holds spend only — so a "variance to plan" could only ever restate a forecast under actual
  spend.

  The `actuals` table is **as-of-dated and append-preserving**: `UNIQUE(project_id, period,
  as_of)`, so re-stating a period is a new row under a new `as_of` and the old statement stays
  readable — a restatement is a visible event, never an overwrite (acceptance criterion pinned by
  test). `record_actuals` / `list_actuals` / `latest_actuals_for_project` in the sessions store;
  `platform/actuals.py` (lean-core, AST-checked) adds the CSV/TSV/JSON parser and
  `reconcile_against_panel` — when the fitted panel and the uploaded actuals disagree for the same
  period it REPORTS the signed per-period gap with both numbers in hand, never silently preferring
  one source, and an out-of-vocabulary period is reported as unmatched and shifts nothing.
  REST: `GET/POST /projects/{id}/actuals` (upload mirrors the delivery ingest; `as_of` and
  `kpi_name` as query params). 15 tests in `tests/test_actuals_store.py`.

- **Variance to plan: delivery-driven vs unexplained, and the bridge closes** ([#227] complete).

  A plan was committed; a season happened; the CFO asks why the miss. Without a refit only two
  buckets are identifiable, and the surface ships exactly those. **Delivery variance** re-runs
  the committed forecast twice with the recorded seed — under the plan and under actual spend —
  on the **committed posterior**: per-channel rows from the forecast's own decomposition, a
  paired-draw interval on the total. **Unexplained** is realized KPI minus the forecast under
  actual spend, LABELLED for what it mixes (baseline movement, competitor action, data error,
  model error, noise) rather than attributed. The refit "effectiveness" split
  (`g_new(S_actual) − g_plan(S_actual)`) is **refused with the reason stated** — that subtraction
  mixes more data, a different window, spec changes and MC noise — and the `compare_runs` diff
  (what actually changed) is attached in its place; the word "effectiveness" appears on no
  surface, by test.

  - **Engine** `planning/variance.py` — rows sum to actual − committed **to 1e-9 by
    construction** (sub-tolerance reproduction drift is carried as its own row; a model that
    does not reproduce the committed snapshot is refused outright — a delivery bucket on a
    different posterior would be the refit comparison in disguise). The
    **committed-interval verdict leads** every rendering, computed from the committed
    window-total *draws* (per-period bounds cannot give it). Refusals: partial actuals coverage;
    supplied lines on a non-dollar KPI; per-channel supplied restatement. `sum_equals`-pinned
    frequentist fits suppress the independent-reconciliation framing; mixed efficiency-measured
    portfolios suppress the blended dollar headline.
  - **Provenance** `finance/lines.py` gains `SUPPLIED` — a human adjustment line (gross-to-net,
    returns) with a **required source note** and no invented interval; supplied lines subtract
    from the remainder, never from a channel.
  - **The #225 criterion actually holds now.** The `forecast_plan` op emits
    `plan_media`/`plan_controls`/`random_seed` inside the forecast snapshot and
    `reproduce_committed_plan` reads them (with snapshot fallback) — previously every real
    commitment refused reproduction with "records no per-period spend plan" while the criterion
    read as met. An end-to-end roundtrip test now reproduces a commitment from provenance to
    1e-9 and refuses on a mutated dataset. Also fixed: `CommitRefusal.overridable` (the commit
    tool crashed on `AttributeError` before any refusal could render).
  - **Pacing window fix.** `expected_outcome_delta` fed *elapsed* totals into *full-window*
    response curves (landing on the steep left; `np.interp` clamps silently past the grid).
    With `elapsed_fraction` it projects totals to the curve's axis and scales the delta back,
    and the payload names its `window_basis`; `compute_pacing` reports the fraction.
  - **Platform** `platform/variance.py` (lean-core) assembles inputs from the stores and
    refuses at assembly time: no committed plan, no realized KPI, delivery gaps over the
    committed window ("assuming plan-as-delivered would fabricate a zero delivery variance"),
    a changed dataset. **REST**: `POST /projects/{id}/variance` (refusals are 409s at POST
    time; the job loads the **committed** run, never the latest) + poll endpoint. **Agent**:
    `get_variance_to_plan` tool + `variance_to_plan` model op. **Report**: `VarianceSection` +
    `AugurVarianceSection`, verdict first (`MMMReportGenerator(..., variance=...)`).
    **Frontend**: Performance → Variance panel (start/poll, provenance chips, refusals
    verbatim).
  - **Notebook** `nbs/demos/variance_to_plan.ipynb` — the whole loop on a world with a causal
    answer key: the delivery rows graded against `response_fn` truth (TV planted at 1.3×,
    Search at 0.7×; signs and interval coverage verified), every refusal demonstrated live.
    38 tests in `tests/test_variance_bridge.py`.

- **The plan of record is reachable: REST, agent tool, Planner commit action — and pacing now
  grades against what was promised** ([#225] remainder; the append-only store, gates and
  reproducibility landed in #267–#269).

  - **REST** — `POST /projects/{id}/plan-of-record` (assess with `assess_only`, commit otherwise;
    a refusal is a **422 naming each gate** and whether it is overridable — 2xx would let a client
    treat "not committable" as committed), `GET .../plan-of-record` (latest committed version),
    `GET .../plan-of-record/history` (all versions, payloads elided, **chain verdict included in
    the listing** so tampering surfaces without an audit someone must remember to run). Contract
    snapshot updated.
  - **Agent tool** `commit_plan_of_record` — assesses, reports refusals as data, records gate
    overrides in the committed payload; provenance gaps are never overridable.
  - **Planner UI** — a "Commit as plan of record" flow on the forecast panel: check
    committability first (refusals render as the disabled button's reason — the refusal is the
    feature, not an error state), commit second, versions confirmed inline.
  - **The retarget** — `latest_budget_plan_for_project` now prefers the latest COMMITTED version:
    pacing and variance compare delivery against the number that was promised, not against
    whatever the draft was last edited to. The committed payload freezes the working plan's
    `plan_payload` verbatim (new `build_commit_payload(plan_payload=...)`) so the pacing join
    reads the same shape either way; drafts stay editable without moving the committed pointer —
    pinned by test.

### Fixed

- **The residual-autocorrelation caveat never fired for core models** (found while wiring #224's
  gate). `planning/forecast.py::_residual_autocorrelation` fell back to
  `compute_component_decomposition().fitted_mean()` — a method that **does not exist** — and the
  `AttributeError` was swallowed by the surrounding best-effort `except`, so for every core
  `BayesianMMM` (which registers no `mu` Deterministic) the check returned all-`None`. Blast
  radius: the forecast's "interval TOO NARROW" caveat and the plan-of-record
  `GATE_RESIDUAL_AUTOCORRELATION` commit gate silently never fired for core models. The fitted
  mean is now the sum of the decomposition's per-obs components (original scale, no heuristic
  rescale).

- **The decomposition residual is now disclosed, and a bridge says whether it closed honestly**
  ([#220]). Completes the issue: the absorption sites were fixed earlier, and this adds the shared
  module they were each reimplementing, plus the two defects that survived.

  `finance/lines.py` gives a bridge line a **provenance** — `modelled`, `observed`, `residual`, or
  `absorbing` for the leftover kind computed as `observed − modelled media`. **Absence reads as
  `absorbing`**, matching `diagnostics/provenance.py`'s discipline: every line written before this
  existed was a leftover, and defaulting the other way would launder exactly the numbers this
  issue is about. `bridge_gap()` reports closure, and when a leftover line is present it says the
  bridge closed *by construction* rather than returning a satisfied zero.

  `finance/closure.py::decomposition_closure()` reconciles a fit's components against the observed
  KPI. `fitted_total` resolves through `compute_component_decomposition()` (core `BayesianMMM`),
  then the `mu` Deterministic (the extension families register one; core does not), then the
  predictive mean, and labels which it used. Measured on a `make_clean` MAP fit: observed
  46801.1, components 46868.4, residual **−67.3 (−0.144%)**, and the bridge closes to 1e-6 on
  additive **and** multiplicative specs. Under a multiplicative spec the total is the exact LMDI
  reconstruction and the **Jensen gap** against `predict()` (measured −281.7) is carried as its
  own field instead of being folded into a component.

  Two things it deliberately refuses to imply. A near-zero residual is an accounting property of
  a fit with a free intercept, not a validated baseline — measured, the residual understates the
  baseline's true error against planted truth by ~14x — so `ClosureFacts` pairs it with an
  interval on the modelled baseline and `residual_reading()` writes the caveat. And on a MAP/ADVI
  fit that interval **collapses**, so it is reported absent with the reason rather than rendered
  as `[x, x]`, following [#249].

  `MediaReconciliation` is the guard that matters most: a bridge can close perfectly around a
  media number that is badly wrong. Measured on a `NestedMMM`, the reporting extractor's media
  total is **2108.8** against `sample_channel_contributions`' **22634.2** and a planted truth of
  **19591.7** — the closure closes to −0.005% either way. The disagreement is now flagged, and
  the nearer-truth reading is the one published; the suspect total is never silently substituted.

  Two remaining defects: `reporting/helpers/decomposition.py::_convert_model_decomposition`
  dropped **Events, Synergy, Price & Promotion, Geo and Product** from both the rows and the share
  denominator, while `extractors/bayesian.py` emits all five under those labels — so a model that
  fit any of them got a decomposition that did not describe it, and the surviving rows' shares
  were too large by exactly the omitted blocks' share. Each is now emitted on `is not None` rather
  than `!= 0`: a fitted block that nets to zero is still part of the identity, and dropping it
  moves the denominator for every other row. And `reporting/helpers/cfo.py::_fitted_total` now
  delegates to `finance.closure.fitted_total` rather than keeping a second copy, so the rollup and
  the bridge cannot disagree about what "fitted" means.

- **The constrained budget allocator returned its warm start — which is today's plan — and called it
  optimal** ([#290]). `planning/budget.py::_solve_allocation`. Found while building
  `nbs/demos/critique_to_decision.ipynb`; verified by direct search rather than by reading.

  Two things combined. The default warm start **works out to the current allocation**: with
  `lo_spend=0` and `hi_spend=2·base`, spreading the current total proportionally over the headroom
  reproduces current spend exactly. And the channel curves are piecewise linear, so the gradient
  (`_segment_marginal`) is piecewise *constant* and SLSQP's line search could fail to find any
  improving step, exiting at `nit=2` with `success=True` and `x` unmoved. `res.success` was never
  checked, so the untouched iterate was returned as the recommended plan.

  The output of that failure is `optimal_allocation == current allocation` with
  `expected_uplift ≈ 0` — "your plan is already optimal", the most trustworthy-looking thing a
  planning tool can emit, and the one nobody questions. Measured on a real fit, a 60,000-sample
  random search over the same feasible set beat it by **+399.7** (`p10`) and **+523.2** (`cvar5`),
  with 452 and 662 samples respectively landing above what the solver returned.

  Every solve now runs from **several starting points** — the caller's `x0`, the proportional
  spread, and the greedy water-fill on the same objective curve (seeded at several budget levels
  under `mode="free"`, where the total is itself a decision) — and returns the best-scoring
  **feasible** candidate among the starts *and* the solver's answers. Two properties follow: a
  solver can never return something worse than a point it was handed, and a stall no longer hides a
  better point another start reached. On the reproduction the allocator now returns **11,582.6**
  against the 11,160.7 it used to, which also beats the random search's 11,560.4. Feasibility is
  checked explicitly (bounds, the budget equality, and every group constraint) because the greedy
  candidate knows nothing about `groups` and a higher-scoring infeasible point must never win.

  Failed inner solves are now counted and surfaced in `BudgetOptimizationResult.notes` instead of
  being accepted silently; on the reproduction 4–5 of 201 per-draw solves genuinely fail.

  **Blast radius**: every risk objective (`p10`, any `p<q>`, `cvar5`, any `cvar<a>`), `mode="free"`,
  and the `groups` / `abs_bounds` / `min_channel_spend` constraints — i.e. everything that is not
  (mean objective + fixed mode + no advanced constraints). The `objective="mean"` default path uses
  `_greedy_allocate` and is untouched, and a test pins that it stayed untouched.

  A note on the regression test, because it is the part that nearly went wrong: the first version
  built its own synthetic concave curves and **passed against the unfixed code**. Smooth textbook
  curves do not stall SLSQP; the specific segment slopes of a real fit do. The shipped test carries
  the P10 objective curve from the reproducing fit verbatim, and 4 of its assertions fail on the
  old code.

- **A windowed `contribution_roi` silently returned the full-series value, and the estimand engine
  had no multiplicative guard** ([#278]). Two gaps in `estimands/evaluate.py`, both of which
  returned a plausible number rather than an error.

  `_contribution_quantity` accepted a time mask and never used it, and
  `_get_contribution_samples` had no mask parameter at all. Measured on a 60-week MAP fit: an
  estimand with `window=TimeWindow(0, 9)` returned **0.4335166608306849** — byte-identical to the
  full-series value, `status="ok"` — where the true windowed ROI is **0.3942**. The mask is now
  threaded through, and the denominator honours it too, since a windowed numerator over a
  full-series divisor is a ratio of neither period. Where a shape genuinely cannot carry a window
  — a per-draw scalar contribution has no observation axis — it raises
  `ContributionWindowUnsupported` by name. The unwindowed path is untouched (an all-true mask
  takes the original branch), so the bit-stability gate in `tests/test_estimands.py` still holds.

  Five further defects, found by an adversarial review of the first commit. The scale refusal now
  lives in `_get_contribution_samples` itself rather than in one caller, because that function has
  **two** consumers — the estimand engine and `compute_roi_with_uncertainty`, which the classic
  report's ROI table renders — and guarding one produced a *self-contradicting report*: the
  Estimand Results section omitted `contribution_roi` as unsupported while the ROI table in the
  same file printed 0.00 "Underperforming", a 550x understatement. The predicate also covers
  **link-scale** models, not just multiplicative ones: a count/bounded likelihood sets
  `y_std = 1.0`, so on the shipped binomial awareness garden model an unguarded `contribution_roi`
  published 0.0071 against an original-scale ROI-equivalent of ~1.5 — over 200x too small, with
  `status="ok"` and `units="ROI"`. It refuses only on a **known-bad** configuration, never on an
  unrecognized one, since a loose predicate refuses every duck-typed model and test double. A
  refusal raised from a **denominator** no longer escapes as a private control-flow exception —
  that discarded every result already computed in the batch, contradicting the documented "never
  raises" contract. A window selecting **no observations** now says so instead of vanishing from
  the results dict. And `contribution_roi`'s stated assumptions no longer claim "over the full
  period" on a windowed instance.

  Separately, `model/base.py` refused a multiplicative specification in **two** places
  (`sample_channel_contributions`, `compute_marginal_contributions`) and the estimand engine in
  **none** — and `contribution_roi` reaches the in-graph Deterministic without calling either, so
  it was the one remaining unguarded instance of the same defect: an additive-scale quantity
  rescaled by `y_std` and published as an original-scale number, when the graph is additive in
  *log* space.

  The asymmetry was the actual bug, and it is resolved **per quantity, not per engine**. The
  in-graph contribution path now refuses. The **contrast**-based estimands stay available on
  purpose: they go through `predict_under`, which returns the original scale, so differencing them
  is precisely the remedy both of the model's own error messages prescribe. Refusing those would
  have been the over-broad reading.
- **The classic report published `contribution_roi` twice, at two masses and two interval
  definitions** ([#277]). `ChannelROISection` renders it at an **80% equal-tailed** interval and
  `EstimandsSection` at a **94% true HDI**; both are default-on, neither said which it was, and
  there was no cross-reference. The trap is that one interval is visibly narrower, which invites
  the reader to treat it as the better estimate. It is not — it is the same posterior at a lower
  mass under a different definition, and on a skewed posterior the two differ in both endpoints.

  Two renders of one estimand is a product decision rather than a bug in either section. The
  decision taken is to **keep both and label each**: every rendered interval now states its mass
  *and* its definition ("80% ETI", "94% HDI"), sourced from provenance that travels with the
  number rather than from a literal at the render site. `EstimandResult` gains `interval_kind`,
  derived from the estimand's own `Realization.hdi_method` — only `az_hdi` is a true
  highest-density interval, since `compute_hdi_bounds` is percentile-based despite its name. A
  bundle carrying no provenance renders the neutral "N% CI" rather than asserting a definition it
  does not have.

  The two defaults were set **1,500 lines apart in different packages**; they are now stated
  together in `estimands.spec` (`ESTIMAND_INTERVAL_MASS` = 0.94, `DASHBOARD_INTERVAL_MASS` = 0.80)
  with the reason they differ. Both are left as they are — changing either would move published
  numbers — so labelling is what closes the gap.

  The definition is stated **per row** in the estimand table, not once in its header, because the
  default estimand set is deliberately mixed — `contribution_roi` is a true HDI while
  `marginal_roas` and `contribution` are equal-tailed. A single modal header let ETI outvote HDI
  2:1 and published `contribution_roi`'s true HDI as an equal-tailed interval in **every** default
  report, turning a missing label into a wrong one on the exact estimand this issue is about. The
  header now states the mass only. "HDI"/"ETI" is posterior vocabulary, so it is suppressed for a
  fit with no posterior, where the existing `interval_noun` already says "confidence interval".
  The extended-model extractor stamps its provenance too, so a Nested/MV/Combined report is
  labelled like a core one.

  Two smaller corrections: the label no longer rounds to whole percent (0.995 read as "100%", a
  claim of certainty), and the new result field is `interval_definition` rather than
  `interval_kind`, which was already taken by a different vocabulary on the bundle
  (`bootstrap_percentile` / `credible` / `confidence`).

  A pre-existing mislabel is fixed as a side effect: `BayesianMMMExtractor.ci_prob` is pinned at
  0.8 and is not wired to `SectionConfig.credible_interval`, so a section configured at 90%
  previously printed "90% CI" over 80% percentiles. The label now follows the arithmetic. The
  underlying disconnect — a `credible_interval` setting that never reaches the extractor — is a
  separate bug and is untouched here.

  The other two shells were checked for the same duplication and do not have it: the Augur deck
  wires neither section, and the interactive report renders one estimand panel with a selector.

- **Two arithmetically wrong fallback branches in the report's ROI extractor** ([#276]).
  `BayesianMMMExtractor._compute_channel_roi` re-implemented
  `reporting.helpers.roi._get_contribution_samples`' three-branch precedence rather than calling
  it, and got two branches wrong: a scalar-per-draw contribution was multiplied by `n_obs` (the
  canonical form has no such factor), and the `beta_<channel>` fallback used
  `beta * y_std * n_obs * 0.5` — under a comment calling itself a rough estimate — against a
  canonical `beta * media_sum * y_std`.

  The second contains **no spend term at all**, so every channel received the same contribution
  however much was spent on it. Measured on a 60-week panel with `y_std = 56.66` and `beta = 0.4`
  everywhere: all four channels reported 679.9, against canonical totals of 79,575 / 52,300 /
  41,060 / 33,343 — ratios of 117.05x, 76.93x, 60.39x, 49.04x, each exactly
  `media_sum / (0.5 · n_obs)`.

  These branches fire precisely for models exposing only a coefficient — the bespoke garden
  models — and the section rendered a clean per-channel ROI table with no marker distinguishing
  them from the primary path. **This changes a published number on those paths**, which is the
  point: the previous one was wrong.

  The two other `# Rough estimate` sites in the same file are fixed with it, since this was one
  habit rather than four incidents. `_get_component_totals` now reads through the same canonical
  path (it held the second copy of the `* 0.5` formula, and which branch fired depended on whether
  spend happened to be available — so one model could report two different totals for one channel).
  And `_compute_marketing_attribution` no longer manufactures a `±15%` interval when the model
  gives no HDI: it returns no interval, which both render sites already degrade gracefully on. A
  number with no posterior behind it, rendered in the credible-interval slot, is indistinguishable
  from a real one.

  Models on the primary branch (a registered `channel_contributions`) are byte-identical, verified
  by running both revisions against the same fitted model.

  Four further defects, found by an adversarial review of the first commit rather than by the
  issue. Returning `None` bounds is only honest if every consumer survives them, and there were
  **four** consumers, two of which are derived metrics rather than render sites:
  `_compute_marketing_contribution_pct` divided by the bound with no guard and `extract()` calls
  it unguarded, so an absent interval took down the **entire report** instead of leaving one gap
  in it; `_compute_blended_roi` did the same inside a bare `except`, silently dropping a
  *computable mean* because a bound was absent; and `insights._triple` raised on `float(None)`,
  dropping the whole revenue line — mean included — from the narrative. The two sibling Augur KPI
  cards still rendered "— – —" under a "80% range" label. All four now keep the mean and say the
  interval is unavailable.

  Two indexing defects came with the delegation. The extractor iterated a **non-deduplicated**
  channel list while the canonical reader indexes a deduplicated one, so on a model with a
  repeated channel name every channel after the duplicate read the wrong column of
  `channel_contributions`. And a `(chain, draw, obs)` `channel_contributions` — which has no
  channel axis — had its **time** axis indexed, returning the value at one period instead of the
  window total, roughly `1/n_obs` of the answer. Both are now size-checked rather than assumed.
  Models on the primary branch (a registered `channel_contributions`) are byte-identical.
- **The extension seasonal period was 52.178571, not 52.0, under a docstring claiming
  comparability** ([#275]). The core graph looks the data frequency up in a table and gets exactly
  52.0 weekly observations per year; the extension graphs
  (`NestedMMM` / `StructuralNestedMMM` / `MultivariateMMM` / `CombinedMMM`) divide 365.25 by the
  datetime index's median spacing and get 52.178571. `components/temporal.py` stated that its
  periods mirrored "the core model's frequency→period logic … so the component is comparable
  across models". They did not, and it was not: measured on the yearly Fourier design over 104
  weekly points, max |Δ| is 0.04216 at order 1, **0.08174** at order 2 and 0.12376 at order 3 — on
  a basis whose amplitude is O(1). Small enough to look like noise in a plot, large enough to move
  a decomposition.

  Because fixing it changes extension-model numbers, it ships behind a flag whose default
  reproduces today (R0.1). `SeasonalityConfig.period_source` is `None` by default, leaving every
  site on its historical source; set it to `SeasonalityPeriodSource.FREQUENCY_TABLE` and an
  extension model's seasonal basis becomes identical to a core model's, which is what makes the
  two genuinely comparable. A spacing matching no tabulated frequency warns and falls back rather
  than inventing a period. The core model has no median-spacing path and **refuses** an explicit
  `DATETIME_MEDIAN` instead of ignoring it — a silently-ignored setting is how the two diverged
  unnoticed.

  Two things the flag needed to be usable rather than merely present. The table's rows are
  deliberately **partial** — weekly data tabulates no `weekly` period, monthly data only `yearly`
  — while the median rule always yields all three, so the opt-in path crashed with a bare
  `KeyError` on configurations the default path (and the core model) accept and skip with a
  warning; it now mirrors the core's warn-and-skip. And the frequency is recognised from the index
  spacing within **5%**, not 25%: at 25% a 4-weekly retail calendar (28-day median) resolved to
  "monthly" and was handed a yearly period of 12.0 against a true 13.04, a max |Δ| of **1.99999**
  — fully anti-phase, 24x the divergence the flag exists to remove, delivered silently. It now
  warns and falls back. `seasonality.period_source` is also settable from the agent spec, which is
  the path that builds every extension model in the product.

  The frequency→period table now has one definition,
  `transforms.seasonality.PERIODS_BY_FREQ`. `validation/backtest.py` held a copy-pasted literal
  linked to the core only by a comment; the drift guard added in #216 scraped that literal with a
  regex, and now asserts the stronger property — both sites read the same object.

- **Extension models' components and ROI were scaled by `y_std` twice** ([#274]). The two model
  families register the same component Deterministics on **different scales** — the core graph
  standardized, the extension graphs (`NestedMMM` / `StructuralNestedMMM` / `MultivariateMMM` /
  `CombinedMMM`) already multiplied by `y_std`, i.e. in KPI units — and nothing said so. Three
  consumers hard-coded the core rule.

  In the **pre-fit Model Design Readout**, `prior_component_facts` rendered an extension model's
  trend and seasonality bands scaled by `y_std**2`. Measured on a 60-week nested model with
  `y_std = 68.8` and a KPI topping out at 1,646: the seasonality band reached **3,940** — 2.4x the
  entire KPI. That reads as "the prior is uninformative", the opposite of what a pre-registration
  document is for.

  `prior_estimand_facts` — the same file, the same `prefit_facts()` call, rendered on the same
  page — had it on `channel_contributions`, so the readout's prior-ROI section was `y_std` times
  too large beside the now-corrected bands.

  Worst, `reporting/helpers/roi.py::_get_contribution_samples` had it on the **default post-fit
  path**, reached by `compute_roi_with_uncertainty` from the Oracle's `roi_metrics` op,
  `build_and_fit`, the Augur deck and the summary helpers. Measured on the same nested model: a
  channel contribution of **797,050 against a total KPI of 88,685** — nine times the whole KPI —
  published as **ROI 132.9**. It is 11,585 and ROI 1.93.

  The convention now lives in one place, `mmm_framework.model.component_scale`, and each family
  *declares* which side it is on. A model that declares nothing is treated as standardized, the
  historical assumption, so every existing consumer is byte-identical. Consumers bridge through
  `to_kpi_units` rather than multiplying by `y_std` — and that function documents what it is not:
  the standardization bridge, not a link inverse, so on a multiplicative or count-likelihood model
  the result is on the model's own scale rather than the KPI's.
- **Rolling-window cross-validation on a geo panel forecast out of phase with its own training
  window** ([#273]). `validator.py` passed `min(train_indices)` — an *observation* index — into
  `PosteriorForecaster.forecast`'s `train_offset`, which is a *period* offset. The two axes
  coincide on a national panel and differ by `n_cells` on a geo/product one, so a 4-cell window
  starting at observation 80 (period 20) shifted the Fourier basis by 80 periods and drove the
  linear trend far negative. Measured on a 4-cell panel: the forecast mean was **175.2 against an
  actual of 232.3** (25% low), a 139.6-unit error against a 35.3-unit noise tolerance. A
  mis-phased seasonality moves the forecast *level* while leaving the interval width untouched, so
  the miss reads as an effectiveness change rather than as an index.

  `forecast()` now takes `train_positions` — observation positions, the same units as `positions`
  — and derives the period offset itself; `train_offset` remains, documented as a period offset.
  A window that does not start on a period boundary is refused by name rather than rounded.

  Three structural fixes ship with it.

  **CV splits are now generated on the period axis** and expanded to observations, so every
  window covers whole periods. `CrossValidationConfig`'s `min_train_size`, `test_size` and `gap`
  therefore count **periods** — unchanged on a national panel, where `n_cells == 1`. Because the
  shipped default of 52 is a full year of weekly history, a one-year *geo* panel can no longer
  spare it, so that case now raises a message naming both numbers instead of returning zero folds
  and surfacing as a generic warning with no CV section and no stated reason.

  **`_slice_panel_data` refuses a slice that does not cover whole periods** instead of rebuilding
  coordinates from whatever index values survive: 101 observations of a 4-cell panel previously
  produced a clone claiming `n_periods=26` — 104 observations' worth — after which every
  downstream reshape read the wrong cell. The check is exact rather than a divisibility test,
  because `[0,1,2,3,5,6,7,8]` is eight observations starting on a period boundary and still
  straddles three periods.

  **The causal-refutation `data_subset` test samples whole periods on a geo panel.** It drew a
  random subset of raw observations, so the refit ran against a fabricated period axis and the
  "effects should be stable" verdict was partly measuring that. The national path — including its
  RNG draw — is byte-identical.

- **The HTML report and the slide deck gave opposite recommendations for the same channel**
  ([#221]). `deck/engine.py` computed a margin-adjusted break-even (`1/margin`) while the Augur HTML
  took `channel_rows`' default of `1.0`. At a 40% gross margin a channel with ROI 1.8 was tiered
  **Scale** in the report and **Reduce** in the deck, with no cross-reference. Both now resolve one
  break-even through `reporting.helpers.measurement.resolve_break_even`.

  The convention is **move the reference, not the number**: a margin-scaled figure would be a profit
  number wearing a revenue label. `MetricMeta` gains `basis` (`"revenue"` by default, so every
  existing number is byte-identical), `value_per_kpi` and `value_source`, so a metric's definition
  travels with it. Any artifact judging channels on a profit basis now carries a banner naming the
  margin, its source, and the constant-margin assumption — triggered by the data, not a config flag.
  A margin passed as a percentage (`40` rather than `0.40`) is refused rather than silently
  producing a 0.025 break-even that tiers everything Scale.

- **`_masked_sum` silently returned an unwindowed divisor** ([#221]). On a dtype or length mismatch
  it fell back to the FULL-series sum, so a windowed ROI divided by every period's spend instead of
  the window's — understating the metric with no error, across windowed marginal ROAS, the
  interactive per-period divisor and `analysis.py`. It now raises, naming the channel and both
  shapes. The geo-panel case its comment cited as justification does not reach it: measured on a geo
  panel through `build_and_fit`, it ran 12 times and took the masked branch every time.

- **The CFO baseline is the model's fitted non-marketing outcome, with the residual named**
  ([#220]). It was `observed_total − modelled_media`, so model error landed inside a number read as
  base demand. `cfo_facts` now reports `fitted_total`, `unexplained` and `baseline_basis`, the
  rollup reconciles as `base + marketing + unexplained == observed`, and both surfaces state which
  basis they are on rather than leaving it to be assumed.

- **Models that consume a control column could be saved but never reloaded** ([#237], [#222]).
  A reach/frequency `frequency_column`, and price/promo levers, are stripped out of
  `model.control_names` because they are consumed as model terms rather than linear controls. The
  serializer recorded that **post-strip** list and compared it against the panel's **pre-strip**
  columns, so `MMMSerializer.load` raised
  `ValueError: Panel controls ['Price'] don't match saved model controls []` every time. Save
  succeeded; load always failed. Metadata now carries `panel_controls` — mirroring the
  `panel_channels` field that already existed for the same reason on the media axis — and the gate
  compares against that, falling back to `control_names` for saves written before it existed. Scoped
  by the strip mechanism rather than by any one feature, so it covers both.

- **`predict(X_controls=...)` silently returned the baseline** when levers or reach/frequency had
  consumed every control ([#222]). The guard was `if X_controls is not None and self.n_controls > 0`,
  so the argument was dropped without an error. A narrower case was also wrong: with one surviving
  control, a mismatched-width array was applied to whichever control happened to survive. Both now
  raise, naming the consumed columns and the accepted order. The legitimate partial swap — passing
  the surviving controls — still works and is not blanket-refused.

- **`sample_channel_contributions()` now refuses a multiplicative specification** instead of
  returning log-scale numbers as though they were original-scale contributions ([#220]). Its
  docstring rests on "the model is additive in channels" and it ends in `contrib * y_std`, both of
  which hold only on the log scale for a multiplicative fit. It was wrong *quietly*: measured on a
  MAP fit, the CFO one-pager reported marketing at **0.005%** of KPI where the additive equivalent
  of the same world reported **10.7%**. Its sibling `compute_marginal_contributions()` has refused
  this since it shipped; this closes the asymmetry.

  **Behaviour change.** Roughly twenty reporting and planning call sites reach this method, so a
  multiplicative model now loses those surfaces rather than showing wrong numbers — reports still
  render, with the affected sections absent. `garden/compat.py::_ops_smoke_tier` executes real ops,
  so a multiplicative garden model's ops-smoke tier turns **red**; red-because-refused is the
  intended outcome. Use `compute_component_decomposition()` (exact LMDI, original scale) or
  `compute_counterfactual_contributions()` / `compute_channel_roi()`.

- **Component shares are computed against the signed total, not a sum of magnitudes** ([#220]).
  Components sum to the fitted outcome, so a magnitude denominator made "% of total" wrong whenever
  any component was negative — a declining trend or a negative control was enough, and shares did
  not add to 1. A stability rule falls back to the magnitude denominator, flagged, when the signed
  total is near zero (a naive signed share there renders Baseline −1105.8% and Trend +1613.0%).

- **The fallback decomposition's Baseline was short by `y_mean × n_obs`** ([#220]).
  `_compute_decomposition_from_trace` computed `intercept * y_std` with no `+ y_mean`, and that
  path is taken exactly when `compute_component_decomposition()` raised — when a trustworthy number
  matters most.

[#220]: https://github.com/redam94/mmm-framework/issues/220
[#221]: https://github.com/redam94/mmm-framework/issues/221
[#222]: https://github.com/redam94/mmm-framework/issues/222
[#237]: https://github.com/redam94/mmm-framework/issues/237
[#249]: https://github.com/redam94/mmm-framework/issues/249
[#224]: https://github.com/redam94/mmm-framework/issues/224
[#226]: https://github.com/redam94/mmm-framework/issues/226
[#225]: https://github.com/redam94/mmm-framework/issues/225
[#195]: https://github.com/redam94/mmm-framework/issues/195
[#227]: https://github.com/redam94/mmm-framework/issues/227
[#228]: https://github.com/redam94/mmm-framework/issues/228
[#273]: https://github.com/redam94/mmm-framework/issues/273
[#274]: https://github.com/redam94/mmm-framework/issues/274
[#275]: https://github.com/redam94/mmm-framework/issues/275
[#276]: https://github.com/redam94/mmm-framework/issues/276
[#277]: https://github.com/redam94/mmm-framework/issues/277
[#278]: https://github.com/redam94/mmm-framework/issues/278
[#290]: https://github.com/redam94/mmm-framework/issues/290

## [1.3.3] — 2026-07-27

A security release. Cut from the `v1.3.2` tag with only the fix below, so 1.3.x users can take it
without adopting in-progress v1.4 work.

### Fixed

- **Generated HTML reports did not escape `</` in embedded chart JSON**
  ([GHSA-7q6v-xpwm-4937]). `reporting/charts/base.py::_to_json` serialised Plotly payloads with a
  bare `json.dumps`, which escapes neither `<` nor `/`. Every caller interpolates the result
  straight into an inline `<script>` block, so a payload string containing `</script>` closed the
  block early and the browser parsed the remainder as **HTML**.

  Affects the **classic** (`ReportConfig.full()` / `.minimal()` / `.presentation()`) and **augur
  readout** shells. `InteractiveReportGenerator` was never affected — it already applied the same
  guard.

  Channel names could not carry the payload, because PyMC rejects `/` in random-variable names and
  a hostile channel name therefore fails at fit time. **Control-variable and geography names are
  xarray coords / DataFrame columns, not RV names**, and they reach chart traces, hovertemplates
  and axis titles — so the vector is reachable from any untrusted modeled dataset (a client CSV, an
  upload, a Data Studio import). HTML-body interpolation of the same names was already correctly
  `html.escape`d; only the chart-JSON path was affected.

  `_to_json` now appends `.replace("</", "<\/")`. `\/` is a legal JSON string escape and legal
  JavaScript, so Plotly decodes a byte-identical string while the literal `</script>` never
  appears. Because `_to_json` is the single choke point for those shells, one change covers every
  chart in them.

  The premature tag also corrupted the surrounding chart payloads, so this is a rendering fix as
  well: in the reproduction, charts rendered went from 26/28 to 28/28 and JS errors from 2 to 0.

  Found by pressure-testing the report generators against a nine-model matrix that included
  deliberately hostile channel, control and geography names.

[GHSA-7q6v-xpwm-4937]: https://github.com/redam94/mmm-framework/security/advisories/GHSA-7q6v-xpwm-4937

## [1.3.2] — 2026-07-27

### Fixed

- **Three names v1.3.1 documented as public were not actually exported.** Its changelog stated
  that `rebuild_like()`, `audit_forward_pass()` and `audit_refit()` were "exported from
  `mmm_framework.validation`". They were reachable only from
  `mmm_framework.validation.backtest`. The claim is now true.

  Found by importing the published wheel and checking it against its own release notes rather
  than against the working tree. Nothing pinned the export set — `PUBLIC_SURFACE` in
  `tests/test_api_contracts.py` did not cover `mmm_framework.validation`, and the docs-snippet
  gate reads code fences, so a *prose* claim about the API surface had nothing checking it.
  `mmm_framework.validation` is now in `PUBLIC_SURFACE`, which makes removing or renaming any of
  these a declared breaking change.

## [1.3.1] — 2026-07-26

A correctness release for the out-of-time forecast path. One bug, of the class this project
treats as most serious: **a code path that reads as complete and silently does nothing for one
configuration** — the same shape as the v1.2.0 sampler fixes (#169/#171) and the root-saturation
fix in 1.3.0 (#202).

### Fixed

- **`PosteriorForecaster` summed five of the fitted mean's ten terms** ([#219]). The model's
  `mu` is `intercept + trend + seasonality + geo + product + media + controls`, plus conditional
  `event` (#143), cross-channel `interaction` (#142) and price/promo `lever` (#138) blocks. The
  forward pass replayed only intercept, trend, seasonality, media and controls — and raised
  nothing. It also applied a **time-averaged** coefficient to a time-varying channel
  (`beta_<ch>` is `pt.mean(beta_t)`), and convolved **raw reach** for a reach/frequency channel,
  omitting the frequency gain.

  Every metric computed from that forward pass was affected on such a model — MAPE, sMAPE,
  RMSE, MAE, bias, MASE, the naive-baseline comparison and 50/80/95% interval coverage — and the
  failure was silent: the backtest completed and reported plausible accuracy. On a
  geo × product panel the dropped product offset alone moved the forecast by up to **113 KPI
  units**, on every observation.

  These numbers ship through `run_backtest`, the `cross_validation` agent tool, the
  Validation-tab REST job and a client artifact.

  The fix follows the frequentist path's refusal convention (#183): the **product level offset is
  now replayed**, and every term the forward pass genuinely cannot reproduce raises the new
  `ForecastUnsupportedError` — naming the feature, listing every blocker at once, and doing so at
  **construction time**, so a caller cannot hold a forecaster whose output it is not allowed to
  trust. Refused: price and promotion levers, events, cross-channel interactions,
  reach/frequency, time-varying coefficients, and the multiplicative specification.

  **This turns some previously "working" backtests into hard refusals.** That is the point: the
  numbers they produced were wrong. The `cross_validation` agent tool reports it as a stated
  "not assessable for this model" reason rather than a failure.

- **`_clone_for_prefix` downcast custom model classes** ([#219]). It hard-constructed
  `BayesianMMM(...)`, so backtesting any garden or custom model — a `CustomMMM` subclass such as
  `LatentFactorMMM` or the awareness model — silently fit and graded a **plain additive MMM** and
  reported its accuracy under the custom model's name. It now reconstructs `type(model)` and
  forwards `model_params`, refusing when the class cannot be rebuilt rather than falling back.

- **`run_backtest` now refuses an experiment-calibrated model.** The refit drops calibration
  likelihoods (their estimands reference full-period spend), so the reported accuracy was an
  *uncalibrated* model's, carrying the calibrated model's name.

- **A geo/product panel with a spline, GP or piecewise trend replayed the trend at the wrong
  time index** ([#219]). `trend_component` is registered per-*observation*, but the forward pass
  indexed it with *period* positions — so on an `n_cells`-wide panel, period `p` read
  observation `p`, which belongs to period `p // n_cells`. The trend was stretched by a factor of
  `n_cells`, and the documented hold-last-flat clamp never fired because positions never reached
  the end of the obs axis. Measured at **up to 26% of KPI level** on a 6-cell spline panel. A
  national panel was unaffected (`n_obs == n_periods` makes the two indexings identical), which
  is why it went unseen.

- **Student-t fits drew Gaussian observation noise.** `LikelihoodFamily.STUDENT_T` is a supported
  additive family (default `nu = 4.0`, genuinely heavy-tailed), but `include_noise=True` always
  drew `Normal(0, 1) * sigma`. Since the whole purpose of that flag is the predictive
  distribution that `BacktestResult.coverage()` grades, the reported interval coverage was not
  the model's. The noise draw now dispatches on the fitted family.

- **The rolling-window cross-validation forecast seasonality out of phase.** `_trend_at` honoured
  `train_offset` but `_seasonality_at` did not, so a clone whose window starts at absolute period
  `s` evaluated the Fourier basis `s` periods out of phase. `run_backtest` always passes
  `train_offset=0` and was unaffected; `ModelValidator.cross_validate(strategy="rolling")` was not.

- **`ModelValidator.cross_validate` swallowed the new refusal into a per-fold warning.** Its broad
  `except Exception` downgraded a configuration-level refusal to a log line — and not uniformly,
  since event regressors are rebuilt per fold, so a fold containing no configured holiday passed
  while one containing a holiday was silently dropped. It now propagates, and the audit is hoisted
  above the fold loop so the refusal costs zero refits instead of one per fold. Cross-validation
  also now applies `audit_refit`: it drops `experiments` exactly as the backtest refit does, so it
  was silently grading an uncalibrated model while the backtest harness refused to.

- **Four more clone sites in `validation/validator.py` downcast custom model classes** — the
  cross-validation, prior-sensitivity and stability clones, and `_fit_clone`, which powers the
  causal refutation suite behind the **Model Defense** document. A garden model's defense report
  was showing a plain additive MMM's refutation results under the garden model's name. All five
  clone sites now share one `rebuild_like()` helper.

### Added

- `ForecastUnsupportedError` (with `.feature`, `.reason`, `.all_unsupported`),
  `TrendExtrapolation`, `rebuild_like()`, and the public `audit_forward_pass()` /
  `audit_refit()` helpers, all exported from `mmm_framework.validation`.
- **A subclass that overrides `_build_model` is now refused**, because it defines its own mean
  and this forward pass reproduces only the base additive one. A class whose mean *is* the base
  mean opts in with `__forecast_forward_pass__ = "base"`. This is the necessary counterpart to
  class-preserving cloning: fitting the right class while replaying the base forward pass would
  have traded one silent-drop path for another.
- **`PosteriorForecaster.trend_extrapolation`** states how the trend is continued past the
  training window instead of leaving it implicit: `linear` extrapolates in closed form, while
  spline/GP/piecewise **hold the last fitted level flat** — so those forecast intervals do not
  widen with horizon. `is_model_defined` distinguishes the model's own continuation from the
  heuristic. It also records the training length, because a linear trend's slope *per period* is
  a function of it.
- `PosteriorForecaster(..., strict=False)` downgrades a refusal to a warning for diagnostic use.
  It omits a term the model estimated and must not be used for planning.

### Notes

The regression test (`tests/test_backtest_completeness.py::TestComponentSumIdentity`) asserts
that `forecast(include_noise=False)` over training positions equals the sum of the model's
registered component Deterministics to `atol=1e-9`. **That test is the audit**: any term the
forward pass fails to sum appears as a residual rather than as a plausible number. It
deliberately does not compare against `predict()`, which draws from the observation likelihood
and carries MC noise no seed removes.

The module and `validator.py` docstrings claimed "national data only; trend NONE or LINEAR"
while geo panels and every trend family in fact ran. They now state the actual scope.

[#219]: https://github.com/redam94/mmm-framework/issues/219

## [1.3.0] — 2026-07-26

### Changed

- **The default logistic saturation prior is now stated in units of observed spend.**
  Media reaches saturation normalized by the channel's training maximum, so `sat_lam` *is*
  the curve's elbow — half-saturation sits at `ln(2)/lam` in units of that maximum. The
  previous default, `Exponential(lam=0.5)`, was never reparameterized after that
  normalization: its **mode was `lam = 0`** (a channel that never saturates) and **29.3% of
  its mass placed the elbow beyond maximum observed spend**, where no observational data can
  move it.

  That matters more than an ordinary prior choice because saturation is **not identified from
  the sales likelihood** — Jin et al. (Google, 2017) call the Hill parameters "essentially
  unidentifiable"; Dew et al. (2024) show predictive fit cannot arbitrate between
  observationally equivalent response curves. When a parameter is unidentified the prior is
  largely the answer, so it has to be defensible.

  The new default places the median elbow at half of maximum observed spend, with a 90%
  interval of roughly 26–97% of maximum and ~4% of mass beyond it — the same move Robyn
  (`inflexion = max(x)·γ`) and Meridian (`ec` scaled to median spend) make.

  **This changes fitted results for any model that did not set an explicit `lam_prior`.**
  To restore the old prior, pass
  `SaturationConfig.logistic(lam_prior=PriorConfig(distribution="Gamma", params={"alpha": 1.0, "beta": 0.5}))`
  — `Gamma(1, rate)` *is* the Exponential, and `PriorType` has no Exponential member.
  ([#207](https://github.com/redam94/mmm-framework/issues/207))

### Added

- **`SaturationConfig.anchor_kappa_to_data`** — confines the Hill half-saturation prior to the
  channel's own observed spend percentiles rather than the whole normalized `[0, 1]` range,
  via the existing `compute_kappa_bounds_from_data` (previously reachable only from the
  extension priors). Opt-in, because it changes fitted results; an explicit `kappa_prior`
  still wins. ([#207](https://github.com/redam94/mmm-framework/issues/207))

- **`mmm_framework.diagnostics.saturation`** — the pair of checks for a parameter whose value
  is mostly its prior. `saturation_prior_report` asks, *before* fitting, where the prior puts
  the curve's elbow relative to spend you have actually observed, and grades it
  `anchored` / `diffuse` / `unanchored`; `warn_if_saturation_prior_is_unanchored` warns on the
  last of those and is silent otherwise; `saturation_learning` reports post-fit
  prior→posterior contraction for the saturation block, where a `"prior-dominated"` verdict
  means the fitted curve is the prior you chose. Neither makes the parameter identifiable —
  only dose spread does that. ([#207](https://github.com/redam94/mmm-framework/issues/207))

- **`mmm_framework.frequentist`** — the estimation layer for epic
  [#180](https://github.com/redam94/mmm-framework/issues/180). With per-channel adstock and
  saturation held fixed the core model is linear in its remaining parameters, so its mean is
  exactly `X @ theta`. `build_design_matrix` produces that design in pure NumPy and an
  equivalence test pins it against the PyTensor graph to 1e-12 across every supported
  transform, trend and panel shape; `fit_ridge` solves it in closed form (NumPy/SciPy only,
  diagonal penalty, effective degrees of freedom, optional non-negativity); and
  `search_transforms` chooses the transforms by rolling-origin out-of-sample error.

  Note what the search does **not** do. Graded against the planted truth in
  `synth.dgp.make_clean`, carryover is recovered better than chance (mean absolute error
  ~0.17 on α against ~0.26 for an uninformed draw) but **saturation is not identified** —
  among candidates scoring within 10% of the best, one channel's `sat_lam` ranges over
  ~0.16 to 7.8 while the planted value is 1.6. At a large budget the winner *beats the true
  parameters* out of sample while sitting further from them. Read `SearchResult.spread()`,
  not the winner, and render `SearchResult.caveat` wherever its ROI is shown.

  No estimator is wired to `fit()` yet — `frequentist_ridge` and `frequentist_cvxpy` still
  refuse. Design spec: `technical-docs/frequentist-estimation.md`.

- **`mmm_framework.frequentist.bootstrap_fit`** — moving-block residual bootstrap turning the
  ridge point estimate into `(chain=1, draw=n_boot)` `InferenceData`, so `predict`, the
  estimand engine and reporting all work unchanged. Every deterministic in the container is
  evaluated out of the model's own PyTensor graph at each replicate's parameter vector, so a
  bootstrap `channel_contributions` cannot drift from the Bayesian definition of one.
  ([#186](https://github.com/redam94/mmm-framework/issues/186))

  Three deliberate choices, each addressing a way a plausible bootstrap ships intervals that
  do not cover:

  - **Blocks, with a data-driven length.** MMM residuals are serially correlated, and an iid
    residual bootstrap treats each week as exchangeable. Measured over 60 simulations × 300
    replicates of the `make_clean` world with AR(1) errors at ρ = 0.6, per-channel
    contribution coverage of 90% intervals is **79.6%** for the iid bootstrap and **90.4%**
    for the block version (median block length 7). At ρ = 0 the estimated block length
    collapses to 1 and the two agree (92.9% vs 93.3%), so nothing pays a width penalty for a
    dependence that is not there. Panel cells resample the *same* period sequence, because
    resampling geographies independently destroys the contemporaneous correlation that makes
    a panel more informative than one national series.
  - **The cheap interval is labelled, not silently shipped.** The transforms and penalty are
    chosen once by search, and every replicate conditioning on that choice omits selection
    uncertainty — which matters more than usual here, because saturation is not identified by
    the criterion at all. `refit_search=True` re-runs the search inside each replicate and is
    correct; the default is `False` and stamps
    `diagnostics["interval_semantics"] = "conditional_on_selection"` plus a plain-language
    caveat, on the reasoning that an unaffordable honest default gets switched off and takes
    its own label with it.
  - **The bias is stated, not papered over.** Percentile intervals around a shrunk estimator
    cover the estimator's sampling distribution, not the true parameter, and no interval
    method fixes that. `bc_interval` / `bca_interval` correct the bootstrap distribution's
    median-bias and skewness for a named scalar; the caveat carries the fit's effective
    degrees of freedom, which is the honest measure of how much work the penalty is doing.

  Every fit stamps `inference_family="frequentist"`, `interval_kind="bootstrap_percentile"`
  and `approximate=False` — `approximate` is the wrong flag to reuse, since an approximate fit
  is a badly-estimated posterior and a ridge fit is not a posterior at all. Nothing renders
  these yet; that is [#188](https://github.com/redam94/mmm-framework/issues/188).

- **`fit()` dispatches the frequentist path, and every surface tells the truth about it.**
  `InferenceMethod.FREQUENTIST_RIDGE` / `FREQUENTIST_CVXPY` now select a real estimator instead
  of raising. `ModelConfigBuilder().frequentist_ridge()` / `.frequentist_cvxpy()` are live, as is
  the generic `.with_inference_method(...)`; `ridge_alpha`, `bootstrap_samples` and
  `optim_maxiter` stop being inert and become the penalty fallback, replicate count and search
  budget. The agent spec accepts `inference.method = "frequentist_ridge"`.
  ([#188](https://github.com/redam94/mmm-framework/issues/188))

  **`inference_method` selects the paradigm; `fit_method` selects among the Bayesian
  estimators.** A frequentist fit leaves `fit_method` as `None` — `FitMethod` has no frequentist
  member, and a stray `"nuts"` there is precisely what made downstream surfaces announce a full
  MCMC posterior for a ridge fit.

  The integration half is the larger one, because every report section, estimand label and
  diagnostic in this codebase was written assuming a posterior:

  - **`MMMResults.converged` is `None`** for a frequentist fit, and the sampler metrics are
    nulled rather than computed. This is not defensive: a `(chain=1, draw=B)` bootstrap trace
    passed *every* convergence gate as `True` — `az.rhat` on one chain is NaN and a `None` metric
    raises no flag, while `az.ess` returns ≈`B` because bootstrap replicates are iid. The
    estimator was silently green everywhere the verdict is consumed.
  - **Interval wording is family-aware.** A bootstrap percentile interval is a **confidence**
    interval; "there is a 90% probability the ROI is in this range" is true of a credible
    interval and false of this one. `mmm_framework.diagnostics.provenance` is the single source
    of that vocabulary, and the classic report, the Augur client deck and the interactive
    report's JavaScript all read it instead of hard-coding "credible".
  - **Posterior-only views are gated off with a stated reason**, not blanked: the convergence
    table, posterior-predictive checks and Bayesian p-values, prior-predictive and
    prior-vs-posterior contraction. A missing table reads as an oversight; "not applicable
    because there is no chain" reads as a property of the method.
  - **A banner names the estimator, the selection criterion and the interval semantics** in all
    three shells — including the Augur client deck, which previously rendered a frequentist fit
    with *zero* caveat, since its only stop sign fires on `approximate` (False here) or
    `is_converged is False` (`None` here).
  - **`inference_family` is a distinct provenance field**, carried into report bundles, the
    serializer's `metadata.json` and `planning/history` run metrics. `approximate` is the wrong
    flag to reuse: an approximate fit is a badly-estimated posterior, a ridge fit is not a
    posterior at all and its point estimate may be excellent. Absence of the field reads as
    Bayesian, so every fit produced before this release is unaffected.

  Selecting `frequentist_cvxpy` with no explicit `constraints=` applies non-negative media — the
  one restriction every MMM wants and the one a prior can only express softly. It needs the
  optional extra (`pip install 'mmm-framework[frequentist]'`).

  The Excel template's `Inference Method` cell accepts both values too — it had its own hard
  refusal, added while they were unimplemented, which would have outlived the thing it gated.

- **The frequentist path, graded — and the verdict written down.** Epic
  [#180](https://github.com/redam94/mmm-framework/issues/180) opened by challenging itself
  ("ridge is MAP with Gaussian priors, so this risks shipping a synonym").
  [#189](https://github.com/redam94/mmm-framework/issues/189) answers it with numbers on five
  synthetic worlds with answer keys, and records the answer where it is unflattering.

  Mean |relative error| on per-channel contribution (NUTS at 4 × 1000; full table and signed
  bias in `tests/frequentist/test_recovery_comparison.py`):

  | world | ridge | map | nuts |
  |---|---|---|---|
  | `clean` (control) | **0.052** | 0.073 | 0.076 |
  | `realistic` | 0.317 | 0.216 | **0.182** |
  | `unobserved_confounding` | 0.670 | **0.297** | 0.302 |
  | `adstock_misspec` | 0.959 | 0.955 | **0.809** |
  | `saturation_misspec` | 0.688 | 0.399 | **0.389** |

  Runtime: ridge 1.7–2.9 s, MAP 2.6–5.6 s, NUTS 8.9–19.6 s.

  **Ridge is not a synonym for MAP** — the two differ by roughly 2× on three of five worlds
  and in opposite directions on `adstock_misspec`. **But it is not more accurate.** It wins
  only on the positive control, and under unobserved confounding it over-credits media by
  **+41.6%** against MAP's +5.9% — because the shipped `Gamma` / `LogNormal` media priors
  shrink media effects in a way that happens to counteract back-door bias, and a penalty
  selected by out-of-sample error carries no such opinion.

  **So the honest recommendation is: use the Bayesian path.** Reach for the frequentist one
  when you need a **hard constraint** a prior cannot express (β ≥ 0, an ordering, a
  contribution total that must match a booked number — the epic's strongest justification),
  a **fast second opinion** for triangulation, or iteration speed while a specification is
  still moving. Do not reach for it to publish intervals, to encode knowledge, or on data you
  suspect of confounding.

  Ships with `nbs/demos/frequentist_vs_bayesian.ipynb` — the **paradigm** axis, sibling to
  `approximate_posteriors.ipynb` (method) and `nuts_backends.ipynb` (backend) — which measures
  all of the above at bake time rather than quoting it.

- **`run_recovery_coverage(refit=...)`** — estimator injection for
  `diagnostics/coverage.py`, which was hard-wired to `pm.observe` + NUTS. Supplying a callable
  grades a non-PyMC estimator against the same θ*, the same simulated datasets and the same
  central equal-tailed intervals as the Bayesian path, which is what made the #186 coverage
  table possible. `extra_caveats=` was added alongside it: the function auto-attaches an
  uncertainty caveat for ADVI only, so ridge shrinkage bias and conditional-on-selection
  intervals have to be passed explicitly or they go unstated.

### Fixed

- **`fit(method="map")` no longer fails with a bare `ZeroDivisionError`.** `find_MAP`
  optimizes in the unconstrained space, so a `[0, 1]`-bounded parameter — every `Beta`-prior
  parameter in the model: `adstock_alpha`, `sat_half`, `sat_exponent` — reaches the optimizer
  as a logit, and float64 `sigmoid` returns *exactly* `1.0` past about 37. `1 - alpha` then
  becomes exactly zero and the `Beta` log-density's gradient divides by zero inside the
  compiled graph, surfacing as a bare `ZeroDivisionError` naming a `Composite` op and nothing
  actionable.

  The error now names the mechanism, lists the bounded parameters in your model, and gives the
  ways out (ADVI or NUTS; a prior with less mass at the edge; or checking the channel has
  enough spend variation to inform its carryover at all). This does not prevent the crash —
  preventing it means changing priors, which is the caller's decision. Note the `#207` prior
  change above happens to stop the originally-reported configuration from crashing, but the
  failure mode is unchanged. ([#203](https://github.com/redam94/mmm-framework/issues/203))

- **Backtests on a root-saturation model were computed from unsaturated forecasts.**
  `PosteriorForecaster._saturate` — the NumPy forward pass behind `run_backtest` —
  dispatched on four saturation families and returned its input unchanged for anything
  else, so `SaturationType.ROOT` fell through to the no-saturation return. Every metric
  the backtest reported for such a model was therefore wrong: MAPE, sMAPE, RMSE, MAE,
  bias, MASE, the naive-baseline comparison and the 50/80/95% interval coverage. Nothing
  raised; the backtest completed and reported plausible numbers. Measured on a real MAP
  fit, the saturated response was off by 0.27 on a transform whose range is `[0, 1]`.

  Saturation now has **one** NumPy definition — `mmm_framework.frequentist._transforms`,
  a bit-for-bit mirror of the in-graph `_apply_saturation_pt` — which both the forecaster
  and the new frequentist design matrix import, and an unhandled family raises rather than
  degrading to identity. If you have run a backtest on a model with `saturation="root"`,
  re-run it. ([#202](https://github.com/redam94/mmm-framework/issues/202))

## [1.2.0] — 2026-07-25

Sampler selection worked in one place and silently failed in four. Fixing that required new
public API — a third NUTS backend that was a declared dependency no code path could reach —
so this is a minor rather than a patch bump. It also raises an install floor and marks two
inference methods as deprecated, neither of which is patch-safe.

Plus a reporting correctness fix: cross-effect tables in multi-outcome reports listed pairs
the model never estimated.

The theme is knobs that read as configuration and were no-ops. If you set `target_accept`,
selected a frequentist inference method, or read a multi-outcome cross-effect table, your
results change.

### Fixed

- **`ModelConfig.target_accept` is now honored by `fit()`.** The fallback went straight to a
  literal `0.9`, so `ModelConfigBuilder().with_target_accept(0.95)` was a silent no-op — the
  first knob the sampling-failure playbook tells you to reach for, and the one that does
  nothing about divergences if it never reaches `pm.sample`. Precedence is now explicit
  argument → config → `0.9`. The field's own default is `0.9`, so an untouched config
  samples byte-identically to 1.1.0; if you *had* set it, your fits change.

- **`fit(nuts_sampler=...)` no longer raises on the core model.** `BayesianMMM.fit()` had no
  such parameter, so the keyword — which `BaseExtendedMMM.fit()` does accept, and which four
  places in the docs recommended — fell into `**kwargs` and collided with the explicit
  argument already passed to `pm.sample`, raising a `TypeError` that named `pm.sample` rather
  than anything the caller wrote. The same line worked on an extension model and failed on a
  plain one.

- **`frequentist_ridge` / `frequentist_cvxpy` no longer silently fit Bayesian NUTS.**
  Both are declared `InferenceMethod` values with builder methods advertising "Ridge
  regression (fast, frequentist)" and "CVXPY for constrained optimization". Neither has ever
  been implemented: `fit()` dispatches on `FitMethod`, never on `InferenceMethod`, and the
  package depends on neither `scikit-learn` nor `cvxpy`. Selecting one did not raise — it
  fitted a **full Bayesian posterior** via the `"pymc"` fallback in `ModelConfig.nuts_sampler`,
  so you asked for a fast frequentist point estimate, waited out MCMC, and got a posterior
  with no indication the request had been ignored.

  `fit()` now raises `NotImplementedError` naming the supported alternative, and constructing
  such a config emits a `DeprecationWarning`. **The alternative is `fit(method="map")`**: under
  Gaussian coefficient priors, maximum a posteriori estimation *is* ridge regression, so the
  capability the name promised has effectively been available all along.

  The enum values are **retained** — stored configs still parse and the frozen-enum contract
  stays green. The Excel template parser rejects the strings at parse time. The config fields
  that exist only for this path (`ridge_alpha`, `bootstrap_samples`, `optim_maxiter`) are
  likewise inert and now marked as such; they are the reserved surface for the real
  implementation, tracked in
  [#180](https://github.com/redam94/mmm-framework/issues/180).

- **Cross-effect summaries no longer report structurally-zero outcome pairs as estimated.**
  `reporting/helpers/mediated.py::compute_cross_effects` probed for `get_cross_effect_summary`
  (singular); both `MultivariateMMM` and `CombinedMMM` spell it `get_cross_effects_summary`,
  so the `hasattr` check never matched and every report fell through to the manual branch.
  That branch walks every off-diagonal `psi` entry, and the matrix starts from zeros with
  only declared specs filled — so undeclared pairs appeared alongside the real ones, and
  `effect_type` was dropped. If you have a multi-outcome report with cross-effect rows you
  did not declare, that is this bug; re-run to get the declared set.

### Added

- `InferenceMethod.BAYESIAN_NUTPIE` — the `nutpie` Rust NUTS sampler has been a core
  dependency since 1.0.0, but sampler choice was the binary `use_numpyro` with no
  representable value for it. Existing enum values are unchanged.
- `ModelConfigBuilder.bayesian_nutpie()` and `DAGModelBuilder.bayesian_nutpie()`, alongside
  the existing `.bayesian_pymc()` / `.bayesian_numpyro()`. The three sample the same graph;
  only the NUTS implementation differs. Verified end-to-end — a real MMM fit through nutpie
  agrees with the pymc backend.
- `ModelConfig.nuts_sampler` — the single resolver from inference method to the
  `pm.sample(nuts_sampler=...)` string. Non-Bayesian methods report `"pymc"` rather than
  raising, so a caller reading it on a frequentist config still gets the historical default.
- `BayesianMMM.fit(nuts_sampler=...)` — explicit argument, defaulting to the config. Accepts
  `"pymc"`, `"numpyro"`, `"nutpie"` or `"blackjax"`; ignored by the non-NUTS fit methods.
- `"bayesian_nutpie"` is accepted by the Excel config parser's inference-method mapping.

### Changed

- **`nutpie>=0.16.4` → `>=0.16.10`.** pymc 6.0 raises `ImportError` below `0.16.10`, so the
  old pin shipped a sampler that could not have started even if it had been reachable. A
  locked environment resolving the old floor needs to update.

### Deprecated

- `InferenceMethod.FREQUENTIST_RIDGE` and `FREQUENTIST_CVXPY`, and the corresponding
  `ModelConfigBuilder.frequentist_ridge()` / `.frequentist_cvxpy()`. They are unimplemented
  and now refuse rather than falling through to NUTS (see Fixed). They are **not** removed:
  removal would break the frozen-enum contract and is reserved for a major version. They will
  become live — not removed — when [#180](https://github.com/redam94/mmm-framework/issues/180)
  lands.

### Notes

- Extension models (`BaseExtendedMMM` — Nested / MV / Combined / Structural) keep their
  `nuts_sampler="pymc"` default deliberately. Their bespoke graphs are not all JAX-traceable,
  so inheriting a numpyro config would break fits rather than speed them up.
- If a pipeline of yours currently runs with a frequentist inference method, it will now fail
  loudly. That is the point: it was returning a Bayesian posterior for a frequentist request.
  Switch to `fit(method="map")` for the fast penalized estimate, or to an explicit Bayesian
  method to keep exactly what you were already getting.
- Internal only, no package surface: the docs code-snippet gate now covers Markdown
  (`README.md`, `CLAUDE.md`, `technical-docs/*.md`) as well as HTML, docs navigation
  registration is gated, and `docs/tools/build_seo.py` is idempotent.

## [1.1.0] — 2026-07-25

Two methodological fixes in `validation/`. Both corrected numbers that read as **more
trustworthy than they were**, so both change output you may have quoted.

A minor rather than a patch bump: the fixes are accompanied by new public API (a `weighting`
argument, new result fields, two new exported names), and one of them changes a default so
that a re-run returns a different number than 1.0.0 did.

### Fixed

- **Spec-curve model averaging no longer weights a causal estimand by predictive skill.**
  `run_spec_curve` applied LOO-stacking weights to per-channel ROI. Stacking
  (Yao et al. 2018) maximizes expected *predictive* utility — it answers "which mixture
  forecasts held-out `y` best?" — while a spec curve averages a *causal* estimand. The two
  objectives come apart exactly where MMM specs differ, because two specs can predict the
  KPI equally well while splitting that same fitted mean very differently between media and
  baseline. Worse, the direction inverts: a spec that overfits the confounder block often
  predicts *better* while being *less* trustworthy for the causal contrast, so stacking
  systematically upweighted the specs a causal analyst should trust least.

  The default is now **equal weights** over the pre-registered set, on the reasoning that
  pre-registration already asserted every variant is defensible, so there is no post-hoc
  predictive ground to promote one.

- **The unobserved-confounding robustness value is no longer inflated by tight priors.**
  `robustness_value` is strictly increasing in `|t|`, and `t = posterior_mean / posterior_sd`.
  Tightening a prior shrinks the posterior sd, which *raised* the reported robustness with no
  new evidence — so the most prior-dominated channel could report the most robust value, and
  robustness values were not comparable across channels with differently tight priors.

  Prior contraction is now computed per channel where a prior group exists. A channel below
  the `PRIOR_DOMINATED_CONTRACTION` threshold (0.20) renders as **"Not assessable
  (prior-driven)"** rather than a green "Robust", with a footnote explaining the inversion.
  "Could not check" is now reported distinctly from "checked and passed".

### Added

- `run_spec_curve(..., weighting=...)` — `"equal"` (default) or `"stacking"`. Requesting
  stacking opts back into the old behavior and logs a warning; it is defensible only when
  every spec in the set is identified for the same estimand and you genuinely want a
  predictive mixture. An unrecognized value raises `ValueError`.
- `SpecCurveResult.weighting` and `.predictive_weights` — the stacking weights are still
  computed and reported, deliberately, as a *diagnostic* rather than an input: divergence
  from uniform says predictive fit discriminates between your specifications, which is worth
  seeing and not worth acting on. `to_dict()` also emits `weighting_caveat`.
- `validation.spec_curve.WEIGHTING_CAVEAT` — the caveat text keyed by weighting mode, so
  reports and payloads state which weighting produced a number.
- `ChannelRobustness.prior_contraction`, `.is_prior_dominated` and `.rv_is_quotable`, plus
  `validation.sensitivity_unobserved.prior_inflation_warning()` and
  `PRIOR_DOMINATED_CONTRACTION`.

### Changed

- `_stacking_weights` returns `{}` on every fallback path instead of fabricating a uniform
  vector, so `predictive_weights` honestly distinguishes "unavailable" from "uniform".
- Report surfaces relabelled to match: the spec-curve blend prose is now derived from the
  weighting actually used, the weights table separates the applied weight from the
  (unapplied) predictive weight, and a prior-driven channel can no longer render green.

### Notes

If you have a stored `SpecCurveResult` or a report produced by 1.0.0, its model-averaged ROI
was stacking-weighted. Re-run to get the equal-weight number, or read the new
`predictive_weights` field to see how far the two diverge on your set.

## [1.0.0] — 2026-07-24

First stable release: the package was split into a lean modeling core and optional
application layers, the public contracts were audited and frozen, and the project adopted
strict semantic versioning. See the
[v1.0.0 release notes](https://github.com/redam94/mmm-framework/releases/tag/v1.0.0) for the
full list of breaking packaging changes, the contract freeze, and the audit fixes.

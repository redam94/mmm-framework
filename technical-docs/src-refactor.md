# Refactoring `src/mmm_framework` — implementation guide

A sequenced plan for reducing coupling and duplication in the core package, written
**after** an audit rather than before it. Every claim below carries a `file:line` that was
read, and every number was measured on this checkout at **v1.3.3** (`pyproject.toml:3`,
`src/mmm_framework/__init__.py:155`).

The audit changed the plan in four places, and an adversarial verification pass then
changed it in a dozen more. Both are recorded rather than quietly folded in, because the
things that looked obvious and safe were mostly neither.

**Scope.** `src/mmm_framework/` only — 161,225 lines, 356 files, 30 sub-packages, 8 root
modules. `server/`, `frontend/`, `examples/` and `nbs/` appear only as blast radius
([§9.5](#95-blast-radius-outside-src)).

---

## 0. Assumed semantics (read before use)

Four rules govern every step. They are the project's, not this document's, and each has
already changed a decision here.

* **A consolidation must not move a published number.** The convention is stated as R0.1
  in `technical-docs/deferred-causal-features.md:27` — "each feature is gated behind a
  config flag whose default reproduces today". Thirty distinct byte-identical invariants
  bind `model/base.py` and the reporting stack; PR 0.1 checks them into
  `tests/contracts/invariants.md` ([§2.1](#21-the-invariants-being-protected)). If a
  cleanup would change a number, it is not a cleanup — it is a behaviour change, and it
  ships in its own PR with a changelog entry.

* **Strict SemVer since 1.0.0.** `CHANGELOG.md:3-6`: *"the major version changes when a
  frozen public contract breaks, and the contract itself is pinned by
  `tests/test_api_contracts.py` and `tests/test_lean_imports.py`."* The 1.2.0 precedent is
  `CHANGELOG.md:571-578`, where dead enum values were *retained*, not removed.

* **The v2.0 entry rule.** `technical-docs/roadmap.md:349` (epic
  [#193](https://github.com/redam94/mmm-framework/issues/193)): *"Nothing lands here for
  being tidier. It lands because keeping it costs something measurable — a real bug, a
  real support burden, or a real performance floor."* Every PR below states its
  measurable cost. One candidate fails the bar and is dropped ([§1.3](#13-do-not-rename-mmm_frameworkplatform)).

* **The safety net precedes the restructuring.** The strongest single-config graph pin
  today is `tests/test_model_likelihood_dispatch.py:88` (`compile_logp()` equality at
  `rtol=0, atol=0`). `tests/test_model.py:856` — the test named
  `test_model_has_expected_variables` — checks for `"intercept"`, `"sigma"` and a fuzzy
  beta match, and nothing else. Four tests pin a Deterministic **sum identity** at
  `rtol=1e-8` (`test_multiplicative_spec.py:224`, `test_channel_interactions.py:144`,
  `test_price_promo_levers.py:183`, `test_holiday_events.py:182`), which is blind to a
  refactor that moves value *between* `intercept_component` and `trend_component`, and
  which runs only on non-default configs. **No test pins the absolute value of a graph
  Deterministic at a fixed point.** Phase 0 fixes that before anything moves.

---

## 1. What this guide deliberately does not do

Four things that look like obvious wins and are not. Each was killed by evidence.

### 1.1 Do not unify the ROI computations

Four estimands are pinned bit-for-bit *against* four legacy functions:
`tests/test_estimands.py:229` (`@pytest.mark.slow class TestEstimandEquivalence`) holds
`contribution_roi`, `counterfactual_roi`, `marginal_roas` and `contribution` to
`rtol=1e-9` against `compute_roi_with_uncertainty`, `MMMAnalyzer.compute_channel_roi`,
`compute_marginal_contributions` and `compute_counterfactual_contributions`. The scoping
note is at `technical-docs/estimands.md:132-143`.

**That pinning makes each estimand equal its legacy twin. It does not make the legacy
functions equal each other.** They are different statistical objects:

| Path | Numerator | Point rule | Interval |
|---|---|---|---|
| `reporting/helpers/roi.py:26` | in-graph `channel_contributions` from the stored posterior | mean of ratios | 94% `az.hdi` |
| `analysis.py:272` | `y_obs` posterior-predictive counterfactual difference, **unpaired** | ratio of means | none on ROI |
| `reporting/extractors/bayesian.py:1582` | in-graph deterministic, own extraction | mean of ratios | **80% ETI** |
| `reporting/extractors/extended.py:743` | coefficient draws × transform at posterior **mean** | mean of ratios | 80% ETI |

The first carries only parameter uncertainty; the second carries `2·n_obs·σ²` of unpaired
observation noise (`model/base.py:4135-4153` forwards `random_seed` to both `predict()`
calls with no synthesis — contrast `:4265-4267`, which synthesises a shared `pair_seed`);
the fourth is a pure rescale of the coefficient posterior with all transform uncertainty
removed. Making them agree is a product decision about which number to publish.

Row 3 is the one most likely to be "unified" by mistake: it is what the classic report
actually publishes, and it carries two arithmetically wrong fallback branches
([§8.4](#84-two-wrong-branches-in-the-reports-roi-extractor)).

**What to share instead** is the mechanics: divisor resolution, sample extraction,
thinning, interval computation, zero-denominator policy, metric metadata.
That boundary is [PR 6](#pr-6--ratio_summary-and-the-divisor-boundary).

### 1.2 Do not unify the trend/seasonality RV builders

`model/base.py:1562-1724` and `mmm_extensions/components/temporal.py:53-182` compute the
same families and **cannot** be merged behind one default, because they differ in *which
random variables exist*:

| Conflict | Core | Extension |
|---|---|---|
| piecewise offset | free RV `trend_m ~ Normal(0, 0.5)` (`base.py:1611`) | absent — pinned by `tests/mmm_extensions/test_extension_priors_wiring.py:425` |
| piecewise centering | uncentered (`base.py:1628`) | centered (`temporal.py:124`) |
| HSGP centering | uncentered (`base.py:1681`) | centered (`temporal.py:173`) |
| seasonality RVs | N RVs, `season_{yearly,monthly,weekly}`, per-component sigma (`base.py:2490-2493`) | 1 RV `seasonality_coefs` over the concatenation (`temporal.py:265`) |
| yearly period | frequency enum table → **52.0** (`base.py:1452`) | datetime median → **52.178571…** (`temporal.py:220`) |

The first two are a *package*, not two flags: with `trend_m` present, centering would make
it exactly unidentified — which is why the extension dropped it. Note the centering
difference is narrower than "core never centers": the core **does** center its spline
(`base.py:1651`) and its numpy GP fallback (`base.py:1722`); only piecewise and HSGP are
uncentered.

And `temporal.py:195-196` claims the periods mirror "the core model's frequency→period
logic, so the component is comparable across models". Measured, the order-2 yearly Fourier
design differs by up to **0.08174** over 104 weekly points. That is a latent defect
([§8.3](#83-the-extension-seasonal-period-does-not-match-the-core)), not a refactor target.

Only the **numpy basis layer** is safely shareable. That is [PR 7](#pr-7--temporal_featurespy-basis-only).

### 1.3 Do not rename `mmm_framework.platform`

I flagged the stdlib shadowing in the initial review. Measured: there is **zero** stdlib
`import platform` anywhere in `src/`, `server/src/`, `tests/`, `scripts/`, `examples/` or
`deploy/` — in fact anywhere in the repo. Absolute imports make the collision unreachable;
it reproduces only by putting the package directory itself on `sys.path` head, and no
`sys.path` entry anywhere does that.

The cost side: 78 occurrences in `src/`, 123 across 50 test files, 37 in `server/`, 6 in
`scripts/` and 2 in `examples/`, plus a `sys.modules` aliasing shim for the package and
its 14 submodules (`platform.sessions` must alias the *module object*, not re-import —
`DB_PATH = resolve_db_path()` at `sessions.py:53` is import-time state, and two module
objects mean two DB handles). `PUBLIC_SURFACE["mmm_framework.platform.sessions"]`
(`tests/test_api_contracts.py:110`) and seven `CORE_IMPORTS` entries
(`tests/test_lean_imports.py:78-88`) make it a frozen-contract break.

Against the v2.0 entry rule this is not a bug, a support burden, or a performance floor.
**Dropped.** Add a comment at the top of `platform/__init__.py` noting the shadow hazard
and why it is tolerated.

### 1.4 Do not delete `src/mmm_framework/api/` — move it

This is the correction that matters most. I described the directory as a stray leftover.
It contains **~83 MB of live development data**: 30 tables including `organizations`,
`users` (with password hashes), `projects`, `experiments`, `run_metrics`, `checkpoints`,
and `garden_models` with **19 rows**. It is gitignored (`.gitignore:45`), so there is no
git recovery, and it has zero git-tracked files.

It is also load-bearing right now. `platform/sessions.py:46-50` prefers it when it exists,
`src/mmm_framework/platform/sessions.db` does **not** exist, and `MMM_SESSIONS_DB` is
unset — so every dev, test and seeder invocation resolves to the legacy path today. Note
there are **two** resolvers that must both be considered: `platform/sessions.py:46-50` and
its byte-for-byte twin at `auth/store.py:29-31`.

Deleting it destroys every local org, project and garden registration. The correct
operation is a **move plus a staged fallback retirement**, and it is already scheduled at
v2.0 (`roadmap.md:347` lists "the legacy sessions-DB fallback"). See
[PR 3](#pr-3--retire-the-legacy-sessions-db-fallback-v20).

**Also not done:**

* Collapsing ETI and HDI behind one *default*. PR 5 unifies the mechanics behind an
  explicit `method=` and leaves the three true-HDI callers on `method="hdi"`; merging them
  silently would move the dashboard ROI interval.
* Consolidating the four `_pct` string formatters (`reporting/insights.py:63`,
  `augur_sections.py:105`, `agents/report_builder.py:523`, `agents/model_ops.py:3263`) —
  each emits a *different published string*, and `report_builder.py:523` emits the literal
  `nan%`. (`deck/builder.py:46 _pct` is a name collision: it is an interval helper.)
* Consolidating `_now` — a 2-line `time.time()` alias in a module whose imports are
  deliberately stdlib-only (`platform/sessions.py:14-23`).
* Unifying the six normal-quantile implementations (the sixth, `frequentist/bootstrap.py:265
  _norm_ppf`, is a scipy delegation a `_phi*`-shaped grep misses). Only two are hand-rolled
  approximations — Winitzki at `estimators/causal.py:173` and Acklam at
  `planning/simulation.py:563`; the other three are stdlib/scipy. The Winitzki error is
  9.15e-4 at q=0.975 (0.047%), but `causal.py:161` short-circuits the default 95% case to
  the exact `_Z975` constant, so it bites only on non-95% CIs.

---

## 2. Phase 0 — the safety net

Nothing in Phases 1–5 lands before this does.

**Runtime budget, stated honestly: +45–60 s cold on a 131 s fast suite.** PR 0.1 and PR 0.2
must be **unmarked** (fast tier). A `@pytest.mark.slow` safety net runs only nightly
(`ci-slow.yml` is nightly + `workflow_dispatch`) and would gate none of Phases 1–5.

### 2.1 The invariants being protected

Thirty documented byte-identical invariants bind this refactor. **PR 0.1 checks the full
numbered table into `tests/contracts/invariants.md`** — id, one-line statement, enforcing
`file:line`, pinning test — because the fingerprint matrix references them by id and an
implementer cannot otherwise tell whether their case covers "inv. 12".

The three with no pin today, which is where a restructuring breaks something silently:

| # | Invariant | Pinned today? |
|---|---|---|
| 19 | The ROI-mode denominator reads a **frozen `pt.constant`** copy of media, never the mutable `pm.Data("X_media_raw")` — else `set_data` counterfactuals silently rescale `beta` (`model/base.py:2757-2769`, `CLAUDE.md:472`) | **No test names this hazard** |
| 26 | `ColorScheme.font_sans` default `"Source Sans 3, sans-serif"` keeps existing reports byte-identical (`reporting/config.py:46`) | **No test** — zero occurrences of `font_sans` in `tests/` |
| — | The absolute value of any reporting-facing Deterministic at a fixed point | **No test** (see §0) |

Calibration for the strongest existing pins: `tests/test_grouped_media_priors.py:192`
(same-seed MAP posterior means at `rtol=1e-10`, slow); `tests/test_price_promo_levers.py:347`
(`_NO_LEVER_NAMED_VAR_COUNT = 26` at `:333` plus a 9-entry `initial_point()` sums dict at
`:334-344`).

### PR 0.1 — graph fingerprint contract

**Why (measured):** structure-only fingerprinting produces a **false negative**.
`saturation_michaelis_menten` and `saturation_tanh` collide on a structural hash — both
emit `sat_half_<ch>` with the same Beta prior, and only the Deterministic formula differs
(`model/base.py:269-270` `x/(x+sat_half)` vs `:272-273` `pt.tanh(x/sat_half)`). Numeric
evaluation separates them. This was reproduced independently twice, on different panels.
Any refactor of `_apply_saturation_pt` could otherwise swap two saturation families past a
structural test.

**New:** `tests/contracts/graph_fingerprint.py`, `tests/contracts/model_matrix.py`,
`tests/contracts/invariants.md`, `tests/test_graph_fingerprint.py`, and one golden JSON per
case under `tests/contracts/graph_fingerprints/`. (`tests/contracts/` already exists —
`rest_routes.json` — so no new directory convention.)

**The matrix must be an importable fixture module, not test-local code.** PR 0.2 and every
Phase 4 PR import it; otherwise they re-implement it and the two drift.

The fingerprint shape:

```text
model_fingerprint(model, numeric=True) -> dict with keys
  free_RVs            [name, op.name, shape, dist_params, transform-class-name]  sorted
  observed_RVs        [name, op.name, shape]                                     sorted
  deterministics      [name, shape]                                              sorted
  potentials          [name]                                                     sorted
  data_vars           [name, shape-of-get_value()]                               sorted
  coords              {dim: labels}
  dims                {var: dim-tuple}
  initial_point       {name: [shape, round(sum, 9)]}
  logp_terms          {factor-name: round(logp, 9)}      <- at a probe point
  total_logp          round(sum, 9)
  deterministic_values{name: [round(sum,9), round(abs-sum,9)]}
```

Four gotchas found by implementing it. All four must survive into the code:

1. **`initial_point()` is degenerate.** `intercept_component`, `trend_component`,
   `seasonality_component` and `controls_total` all evaluate to exactly `0.0`. Offset each
   transformed-space entry before probing.
2. **The offset must be name-derived, not position-indexed.** `+0.1*(i+1)` over
   `sorted(ip.items())` re-randomises every alphabetically-later variable whenever an RV is
   added, removed or renamed — which destroys the readable-diff property the goldens exist
   for. Use a stable hash of the name.
3. **The offset does not rescue zero-mean blocks under `sum`.** On a whole-cycle weekly
   panel, `seasonality_component` and `controls_total` still sum to exactly 0 for *any*
   coefficient (a whole number of Fourier cycles; a z-scored control). **The `abs-sum`
   second element of `deterministic_values` is load-bearing — do not drop it.**
4. **`replace_rvs_by_values` is mandatory.** Naive
   `model.compile_fn(inputs=model.value_vars, outs=[model[d.name] ...])` still contains the
   RandomVariables and re-samples them: measured `intercept_component` = 55.66 / −70.81 /
   1.38 across three separate processes. With `replace_rvs_by_values` the same call is
   byte-identical in all three. `inputs=model.value_vars` is also required — without it
   `PointFunc` raises `TypeError: Unknown input: sigma_log__`, because the Deterministics do
   not depend on every value var.

`initial_point(random_seed=…)` is seed-invariant here (verified element-wise for seeds 0
and 12345 across seven configurations — the strategies in use are moment-based, not prior
draws), and `total_logp` reproduced byte-identically across three separate processes.

**Matrix:** 31 configurations were built and hashed, covering the trend families, every
saturation type, every adstock type, geo and `vary_media_by_geo`, both media-prior modes,
explicit media and control priors, grouped priors, Student-t, multiplicative, legacy
adstock, and `binomial_refused` — where *the refusal is the contract*. The only collisions
were `{default_national, trend_linear, saturation_logistic}`, which are genuinely the same
configuration (`TrendConfig().type is TrendType.LINEAR`; a channel with no saturation falls
back to `SaturationConfig.logistic()`).

**Must be added before merge**, each mapping to an invariant with no other coverage:
`events`, `price`/`promotions`, `channel_interactions`, `control_selection.method != "none"`,
`reach_frequency`, time-varying betas, a geo×product panel, a `cpm`/`cpc` measurement
channel, confounder-role controls, the agent `build_model(spec, csv)` path with `priors.*`
writes, and **at least two `BaseExtendedMMM` configurations** — PR 6 touches
`mmm_extensions/models/structural.py:1097` and two of the §8 bugs are extension bugs, so
extension coverage is required before PR 6, not just before the deferred Layer 2.

**Runtime:** the numeric matrix measured 41.3 s cold in one implementation and ~1.8 s/case
warm in another (≈55 s warm for 31 cases). CI has no PyTensor compiledir cache
(`ci.yml` — `enable-cache: true` at `:44,122` is `setup-uv`, not pytensor), so budget the
cold number.

**Verify:** `uv run pytest tests/test_graph_fingerprint.py -q`. Store full fingerprint
dicts rather than bare hashes so a failure is a readable diff. **Do not check in literal
hashes from this document** — they depend on the exact probe panel and serialization, and
the panel is a property of `model_matrix.py`, not of prose. Attach
`pymc.printing.str_for_model(model)` as a failure artifact; `Model.str_repr()` does **not**
exist in pymc 6.0.1 (`hasattr(pm.Model, "str_repr")` is `False`). Gate golden regeneration
behind an env var.

**Rollback:** delete the test and the contracts; nothing depends on them.

### PR 0.2 — serializer round-trip contract

**Why:** `serialization.py` writes instance attributes directly at 13 sites (`:286`,
`:289`, `:448`, `:464`, `:486`, `:496`, `:500`, `:513`, `:517`, `:936-947`, `:960`, `:974`)
— 15 distinct attributes, 11 by assignment and 4 by in-place container mutation. Any state
that moves behind a collaborator in Phase 4 breaks `load`, and the existing coverage is
mostly `MockModel` unit tests (`tests/test_serialization.py:511-537`; 37 `Mock*`
occurrences against 2 real `BayesianMMM(` constructions, at `:610` and `:639`).

**New:** a real save → load → `predict()` round-trip over `model_matrix.py`'s national,
geo, roi-mode, garden-subclass and extension cases, asserting `metadata.json` field-set
equality against a checked-in snapshot and `predict()` equality at `rtol=0`. Unmarked.

**Verify:** `uv run pytest tests/test_serialization.py tests/test_serialization_extended.py -q`
(measured: 1.31 s fast; 36.5 s all tiers).

**Rollback:** delete the test and the snapshot.

### PR 0.3 — import-layer gate

**Why (measured):** the *module-level* package graph has **exactly one** cycle —
`{agents, platform}` — and it is closed by a single line:

```text
src/mmm_framework/platform/runs.py:19
    from mmm_framework.agents.spec_locks import flatten_leaves
```

Remove that one edge and the module-level graph is acyclic (7–8 levels depending on how
root modules are bucketed). Everything else that looks like a cycle is produced by deferred
imports: there are **632 function-local imports** of the package's own modules
(`agents/tools.py` alone has 108), and counting them collapses 18 packages into one giant
SCC. The gate exists to stop that number growing back. It is a **ratchet**: today's
violations are allowlisted and the allowlist may only shrink.

**What the gate counts — this is the whole design, and it must be stated in the test
docstring:**

* Every `Import`/`ImportFrom` node resolving to a sibling top-level package, **including
  function-local ones**. A module-level-only gate is defeated by
  `def f(): from ..planning import x`, which is precisely the habit being capped.
* **`if TYPE_CHECKING:` bodies are excluded.** They do not execute, and including them
  manufactures a `calibration → validation` edge (`calibration/__init__.py:59`,
  `experiment.py:70`, `likelihood.py:94`) and a phantom
  `{model, calibration, validation, frequentist}` cycle.
* Dynamic imports are invisible to AST and out of scope — `agents/tools.py:1648`
  (`importlib.import_module(f"mmm_framework.{_sub}")`) and the PEP-562 `__getattr__`
  façades in `agents/__init__.py:54`, `estimands/__init__.py:79-85`, `garden/__init__.py:49`
  are edges the gate cannot see. Say so, so nobody trusts it as complete.
* **The unit is the import occurrence**, aggregated per `(edge, file)` record. State this
  explicitly: counting names, statements or files gives answers that differ by up to 3×,
  and a ratchet on an undefined unit is decorative.
* An **unclassified** package is a hard failure naming the module. A silent default tier
  makes the gate useless the first time someone adds a package.

**Tier map** (revised from my initial sketch — two corrections below):

```text
t0  config  utils  transforms  integrations  security  storage
    data_loader.py  dataset.py
t1  model  estimands  diagnostics  calibration  frequentist  mmm_extensions
    dag_model_builder  datasets  synth  builders
    dataset_loader.py  data_preparation.py
t2  validation  planning  reporting  eda  garden  finance  ltv  estimators
    excel_config  continuous_learning  platform
    analysis.py  serialization.py  lineage.py
t3  agents  data_studio  jobs.py
t4  auth
__init__.py  = façade: exempt as a source, forbidden as a target
```

* **`platform` belongs at tier 2, not 4.** It is the persistence layer and the agent tools
  are its heaviest client; putting the store above its callers inverts reality and
  manufactures the `agents → platform` edge — 55 import occurrences across 8 files, ~54 of
  the difference between the two maps, and the single largest block in the violation set.
* **`integrations`, `security`, `storage` belong at tier 0.** All three have zero outgoing
  package edges. Note `security` and `estimators` are *fully isolated* — no package edges in
  or out — so their placement is unconstrained by evidence; mark them as such rather than
  implying they were placed by counting.

**Rule:** `tier(dst) <= tier(src)`. Sideways legal, upward not. A strict `<` rule would need
~8 more tiers and would ratchet on the legitimate `model ⇄ estimands/diagnostics/
calibration/mmm_extensions` peer cluster.

**The baseline is whatever the shipped script measures — do not hard-code a number here.**
Two independent implementations of the rules above produced 21 edges / 46 occurrences and
12 edges / 31 occurrences respectively. Neither reproduces the other, which is precisely the
failure mode a ratchet cannot survive: if the baseline cannot be regenerated from a stated
algorithm, the first regeneration silently loosens it. **PR 0.3 ships the measuring script
and derives the allowlist from it in the same commit**; the number in the PR body is
whatever that script prints.

For orientation only, the current violating edges (statement unit, HEAD) are
`platform → agents` (7 across 5 files, of which **exactly one is module-level** — the PR 10
line), `estimands → reporting` (6, PR 11 removes three and PR 6 two),
`model → serialization.py` (4), `garden → agents` (3), `calibration → validation` (2),
`continuous_learning → agents` (2), `reporting → agents` (2), `frequentist → validation`,
`mmm_extensions → serialization.py`, `model → reporting`, `validation → agents`, and
**`transforms → model` — new on 2026-07-28**, `transforms/carryover.py:216`
(`from ..model.base import _ADSTOCK_KIND`, function-local) landed by #218/PR #236. That last
one is a **tier-0 module reaching tier 1** and is the first of its kind in the repo; it
arrived after the guide's first draft, which is the argument for the script over a literal.

Two enumeration details the walker must handle or it fails on day one:

* **`src/mmm_framework/api/` is not a package** — it holds `sessions.db{,-wal,-shm}` and a
  `.sql.gz`, no `.py`. It must be skipped explicitly, or the "unclassified package is a hard
  failure" rule fires on the first run.
* Adopt #228's anti-rubber-stamp rule for the `LAYERS` comments: assert every reason string
  is non-empty and matches no banned non-reason (`TODO`, `later`, `n/a`). #228 worked this
  out for its own gate; PR 0.3 and PR 0.4 are the same construction and should inherit it
  rather than rediscover it.

**Bonus move, justified by import weight rather than the tier rule:** `model/trend_config.py`
(108 lines, stdlib-only — `dataclasses` + `enum`) is a config artefact in `model/`. Three
consumers reach it through `mmm_framework.model`, which drags in `model/base.py` and the
whole PyMC stack — which is *why* `builders/model.py:712,903` and `data_preparation.py:434`
import it lazily. Moving it to `config/trend.py` (re-export from `model/__init__.py:23`)
lets `excel_config/parser.py:36` keep its module-level import and lets the three lazy
imports become eager. It fixes no tier violation; all four sites are already legal under
the map above.

**Storage,** matching the house split in `tests/test_api_contracts.py`: the `LAYERS` map
inline in the test file (it is a rule; it must be code-reviewed; each entry deserves a
comment — same role as `PUBLIC_SURFACE` at `:69-114`), and today's violations in
`tests/contracts/import_layer_allowlist.json` (mechanically generated and churny — same
role as `rest_routes.json`). Key it `"src_pkg -> dst_pkg" → {src_file: count}`, **not by
line number**: line numbers move on every unrelated edit above them. The per-file count is
the ceiling, so adding an import to an already-allowlisted file fails.

**Runtime:** `ast.parse` over all 356 files measured **0.390 s** (five runs, 0.378–0.392);
the full resolution pass is 0.72–0.89 s wall. For comparison `uv run pytest
tests/test_lean_imports.py -q` is **4.18 s** — about 5.8× — so the gate is safe for
`make fast_tests` and the pre-commit hook. (The `timeout=600` at
`tests/test_lean_imports.py:152,177` is a subprocess ceiling, not a runtime.)

**Failure message must name the fix**, per house style
(`tests/test_api_contracts.py:121-126`): which edge, which file, which tier each side is
in, and the three legal remedies — move the symbol down, invert with a Protocol, or amend
`LAYERS` with a reviewed comment.

**Rollback:** delete the test and the allowlist; the `LAYERS` map has no runtime effect.

### PR 0.4 — private authoring-surface contract

**Why:** Phase 4's "the facade preserves the public surface, therefore minor" is
unverifiable today. The five public methods are pinned; the ~12 **private** members that
registered garden models actually call are pinned by nothing. `agents/garden_authoring.py:71-77`
promises a specific list to the LLM as the authoring API, and **no test asserts any of
those names exist**. `examples/garden_models/nested_survey_mediation_mmm.py` — the one model
that resolves solely through `find_garden_class` — has no test coverage at all.

**New:** `AUTHORING_SURFACE` in `tests/test_api_contracts.py` asserting the promised
members exist with `inspect.signature` equality against a checked-in snapshot, plus the
required instance attributes; and a corpus test that imports and builds the graph of all
eight `examples/garden_models/*.py` subclasses.

**Also:** export the `garden_models.source` rows into `tests/fixtures/garden_corpus/`
(scrubbed of org identifiers) and run the corpus test over them. Of the 19 registered rows,
only **6 source paths still resolve on disk** — the other 13 point at deleted `/private/tmp`
workspaces. The database is gitignored and CI has no rows, so without this, "we preserved
the private surface" is an assertion about one developer's SQLite file.

---

## 3. Phase 1 — deletions and moves (no behaviour change)

> **Standing rule for every PR in this section and the next.** Any PR that adds, deletes or
> relocates a top-level module or package must, in the same commit: (a) update the `LAYERS`
> map with a reviewed comment — new entries are `mmm_framework/measurement.py` (PR 6, t0),
> `mmm_framework/specs/` (PR 10, t0), `config/trend.py` (PR 0.3's bonus move, t0); the
> deleted entry is `data_preparation.py` (PR 1); and (b) regenerate
> `tests/contracts/import_layer_allowlist.json` and show the diff is a **shrink**.

### PR 1 — delete `data_preparation.py`

**Why (measurable):** 509 lines that no production path imports, carrying a *second*
implementation of the trend-feature dispatch. `data_preparation.py:425-462` is
branch-for-branch and key-for-key identical to `model/base.py:1486-1518` (SPLINE /
PIECEWISE / GP; the dead copy has one extra `trend_config is None` guard), and it has
already drifted — `:417` is literally `period = 52  # Weekly data` where the live path
reads a frequency table at `base.py:1451-1456`. It is autodoc'd at
`docs/api/source/api/data_loader.rst:20`, so a reader can find it and follow it into a dead
end. That is the support burden.

**Reachability, verified:** zero runtime importers in `src/`, `server/`, `examples/`,
`scripts/`, `nbs/`. Not in `PUBLIC_SURFACE` and not exported from
`mmm_framework/__init__.py`, so this is minor-safe by the letter of the policy.

**Edits:**

```text
DELETE  src/mmm_framework/data_preparation.py                (509 lines)
DELETE  tests/test_data_preparation.py                       (1201 lines, 41 tests)
EDIT    src/mmm_framework/model/components/trend.py:19       delete the TYPE_CHECKING
        `from ...data_preparation import PreparedData`; replace the 8 `data: PreparedData`
        annotations (:29,:59,:72,:108,:156,:200,:314 + 2 docstrings) with `data: Any`.
        PR 2 deletes this file, but PR 2 is v2.0-gated — this cannot wait, or the tree
        carries a dangling type reference on a publicly re-exported @runtime_checkable
        Protocol for all of v1.x.
EDIT    docs/api/source/api/data_loader.rst                  delete lines 16-23
EDIT    CLAUDE.md:155                                        drop the tree line
EDIT    CLAUDE.md:276                                        drop the key-modules row
EDIT    docs/data-requirements.html:423                      re-anchor the HTML comment to
        model/base.py:965-987 (`_media_max`, the per-channel max AFTER adstock — which is
        what the sentence at :421-422 describes). NOT :926-931, which is `_media_raw_max`,
        the raw-scale sibling used by the in-graph adstock path.
```

`docs/blog-aggregation-bends-the-curve.html:171` cites `DataPreparer.prepare
(data_preparation.py:240)` — already wrong twice (the class is `DataPreparator` at `:185`;
`prepare` is at `:230`). Leaving a dated research post citing the code as-of-then is
defensible. If you do touch it, re-run from `docs/`: `python3 tools/build_search_index.py`
then `python3 tools/build_seo.py`, and commit `shared/seo-manifest.json`.

**Verify:** `make lint && uv run pytest -m "not slow" -n logical`, plus
`uv run --group docs sphinx-build -b html docs/api/source /tmp/apidocs`. RTD is not `-W`
(`.readthedocs.yaml:25 fail_on_warning: false`), so a missed reference warns rather than
fails — read the log, don't trust the exit code.

### PR 2 — retire `model/components/trend.py` (interim now; deletion is owned by #193)

> **This is already a v2.0 candidate.** Epic
> [#193](https://github.com/redam94/mmm-framework/issues/193) names it verbatim —
> "`model/components/trend.py::TrendBuilder` — dead code with a `TrendConfig` shape
> incompatible with the live one (`.trend_type`, tuple GP priors)". **Do not file a duplicate.**
> Comment the two scope corrections this audit found onto that bullet: the removal is
> **six exported names across two `__all__` lists**, not one class, and only three of the five
> strategies actually raise (table below). Then land the interim here.

**Why:** `TrendBuilder` and the strategy classes are unreachable, and three of the five
exported names are also **un-runnable** against the live `TrendConfig`. Measured by
instantiating each:

```text
TrendBuilder.build       FAILED  AttributeError 'TrendConfig' has no attribute 'trend_type'
SplineTrendStrategy      FAILED  AttributeError 'spline_prior_scale'
GPTrendStrategy          FAILED  AttributeError 'gp_lengthscale_prior'
LinearTrendStrategy      OK   (never reads config at all)
PiecewiseTrendStrategy   OK   (reads only changepoint_prior_scale, which exists)
```

So the honest statement is: the entry point and two of four strategies raise
`AttributeError`; the other two build but are unreachable. Following the exported name gets
you a failure more often than not. `roadmap.md:346` already parks it in v2.0.

**Version verdict:** unlike PR 1, these **are** importable public names — re-exported from
`model/__init__.py:41-46` (`__all__:72-77`) and `model/components/__init__.py:8-14`
(`__all__:18-23`). Removing a name from a package `__all__` in a minor contradicts the 1.2.0
precedent. **Treat as major** unless #193 is re-scoped in writing.

**Interim (minor-safe, do this now):** attach the deprecation to the **names**, not the
module. Add a PEP-562 `__getattr__` in `model/components/trend.py` and convert the eager
re-export blocks in `model/__init__.py:40-47` and `model/components/__init__.py:8-14` to
lazy `__getattr__` in the same commit. A module-level `warnings.warn` would fire on
`import mmm_framework` for **every user** — `model/__init__.py:40-47` imports these eagerly
and `mmm_framework/__init__.py` imports the model package — and `pytest.ini:8-17`
deliberately keeps first-party `DeprecationWarning`s visible, so it would also spam the
test log.

**Edits at v2.0:** delete the module, the two import blocks, the two `__all__` entries.
Nothing persisted references it, so no data shim is needed.

### PR 3 — retire the legacy sessions-DB fallback (3a now, 3b later)

> **Also already a #193 candidate** — "Legacy sessions-DB fallback: `platform/sessions.py`
> still prefers a pre-2026-07-24 dev DB at the old `api/sessions.db` path when present."
> Comment on that bullet rather than filing a duplicate; the scope correction to record is
> that there are **two** resolvers, not one (`auth/store.py:29-31` carries a copy).

**Why:** the fallback silently forks state, and CI cannot see it. `src/mmm_framework/api/`
has zero git-tracked files, so on a fresh checkout the legacy branch is **never exercised**
— a divergence introduced only in that arm passes CI and reproduces exclusively on developer
machines and upgraded deployments, which is exactly the population the fallback exists for.

**Most of this should not wait for v2.0, and the guide's first draft was wrong to say it
should.** The cost is present-tense and compounding — every new machine and every upgraded
deployment joins the exposed population — while the frozen contract, by the project's own
definition (`CHANGELOG.md:3-6`), is what `test_api_contracts.py` and `test_lean_imports.py`
pin. Those pin **symbol presence**: `resolve_db_path` keeps existing with an unchanged
signature; only its return value in one un-pinned environment changes.
`docs/api-contracts.html` marks the sessions store *"internal — stable semantics"*, not
`frozen`, and nothing anywhere freezes filesystem path-resolution **order**. So:

* **PR 3a — next minor.** Steps 1 (the test CI has never had), 2.5 (the one-time migration),
  and the `auth/store.py` delegation. The delegation is the single highest-value line here —
  it eliminates the two-resolver lockstep hazard immediately — and has no business waiting
  for a major. Add a `### Deprecated` changelog entry naming the legacy path and the release
  that removes it, matching the 1.2.0 announce-don't-remove precedent.
* **PR 3b — one minor after 3a**, not v2.0: steps 3 and 4. By then the migration has run on
  every machine that imported the package once. If maximum caution is wanted, gate it on a
  named release ("deprecated 1.4.0, removed 1.6.0"). Keep it at v2.0 *only* if the project
  decides path-resolution order is itself a published contract — it is not in
  `docs/api-contracts.html`, so make that decision visible rather than inherited.

Step 2 ("physically move the dev DB") is not a shippable step at all — it is a local
operation on one machine, and once 2.5 ships the migration performs it. Keep it in the
runbook, not the sequence.

**Sequence (order is load-bearing):**

```text
0. Stop the API server and any seeder. The DB is in WAL mode
   (platform/sessions.py:73, auth/store.py:59). Either run
   PRAGMA wal_checkpoint(TRUNCATE) or move sessions.db-wal and sessions.db-shm
   with the file — moving the .db alone silently drops the most recent
   transactions, in a step whose stated stake is "no git recovery".
1. Add the test CI has never had: create a fake api/sessions.db under a tmp
   package root and assert BOTH resolvers pick it.
2. MOVE  src/mmm_framework/api/sessions.db{,-wal,-shm}
      -> src/mmm_framework/platform/
   MOVE  sessions_backup_20260713_163813.sql.gz -> out of the source tree
         (it is a backup, not package data)
2.5 Ship a one-time migration IN resolve_db_path: if the legacy file exists and
   the new one does not, MOVE it and log. Release that. Only after one release
   does step 3 happen — otherwise every developer and every upgraded deployment
   starts v2.0 against a new empty database, which is worse than the fork.
3. DELETE the legacy branch: platform/sessions.py:47-49, auth/store.py:29-31.
   rm -rf src/mmm_framework/api/  (now only stale __pycache__)
4. REWRITE tests/test_sessions_db_path.py:22-27 and update the prose that goes
   stale with it: the module docstring at :8-11, platform/sessions.py:35-38
   (the resolve_db_path docstring), auth/store.py:23-25 (the lockstep comment),
   CLAUDE.md:446 (the Sessions DB troubleshooting row), docs/changelog.html:513,
   scripts/seed_workflow_demo.py:19. Check deploy/gcp/vm/vm_setup.sh's
   MMM_SESSIONS_DB.
```

**Also in this PR** (it is the same single-source problem): `auth/store.py:22-36` duplicates
the resolver's *logic* — not its text; `store.py:29,32` uses
`Path(__file__).resolve().parents[1]` where `sessions.py:46` uses a named `package_root`,
and the docstrings differ entirely. Make it delegate:

```text
# src/mmm_framework/auth/store.py
def _resolve_default_db_path() -> Path:
    from mmm_framework.platform.sessions import resolve_db_path   # lazy: no cycle
    return resolve_db_path()
```

Keep the name and the `DEFAULT_DB_PATH` global — 7 test files reference it and
`tests/test_sessions_db_path.py` calls the private function directly. `platform/*.py` has
zero `mmm_framework.auth` references, so this creates no cycle, and `platform/sessions.py:16-23`
is stdlib-only, so it adds no dependency.

### PR 4 — move `InProcessKernel` beside its siblings

**Why:** the third kernel implementation lives at `agents/tools.py:1579-1909` — 331 lines
inside a 7,931-line tool module — while `agents/kernels.py` holds the `Kernel` Protocol
(`:81`), `KernelManager` (`:95`) and `SubprocessKernel` (`:571`), and
`agents/container_kernel.py` holds `ContainerKernel`. `isinstance(InProcessKernel(), Kernel)`
is `True` today. The cost is discoverability: changing the kernel contract requires finding
an implementation that is not where the other three are.

**The move is cleaner than it looks.** The class references **zero** `tools.py`-defined
names — every free name it uses is itself an import into `tools.py` (`_MODEL_CACHE`,
`_NAMESPACE_CACHE` from `agents/runtime.py`; `_normalize_figure`, `format_execution_error`
from `agents/figures.py`; `build_and_fit` from `agents/fitting.py`; `_ws`, `_model_ops`,
`ExecuteResult`). No module-level cycle: `fitting`, `model_ops`, `runtime`, `figures` and
`workspace` never import `kernels`.

**Leave the `_KERNELS` registry where it is.** `tools.py:1913-1921` builds
`KernelManager(..., {"inprocess": …, "subprocess": …, "container": …})` and registers the
`atexit` hook; `server/src/mmm_framework_server/main.py:229, 311, 7310` do
`from mmm_framework.agents.tools import _KERNELS`. Moving it would drag `ContainerKernel`
and `agents.profile` into `kernels.py` and invert the registry. Import `InProcessKernel`
into it instead.

Note this adds ~4 deferred imports back the other way (`kernels.py:776` already does
`from mmm_framework.agents.tools import format_execution_error`), which counts against
PR 0.3's ratchet — allowlist them explicitly with a comment.

**Also:** update `agents/kernels.py:5`, whose docstring says where `InProcessKernel` lives.

**Verify:** `uv run pytest tests/test_kernels.py tests/test_oracle_checkpoint_serde.py -q`,
and `make lint` (which covers `server/src`).

---

## 4. Phase 2 — mechanical consolidation

Each PR here is provably number-preserving. **Where it is not, the PR says so and splits the
behaviour change into its own changelogged PR** — this applies uniformly, including to
PR 6's resolver adoption and PR 9's `synth/mff.py` and `eda/results.py` migrations.

### PR 5 — `utils/intervals.py`

**Why (measurable):** `compute_hdi_bounds` (`utils/statistics.py:17`) computes an
**equal-tailed percentile interval**, not a highest-density interval. The name has been
copied outward. In a framework whose stated differentiator is honest uncertainty, a
function that says HDI and returns ETI is a real support burden — and the ambiguity is
load-bearing: the browser recomputation in `reporting/interactive/script.py` must agree
with the Python convention, and the only enforcement is a docstring (`facts.py:21-27`,
`script.py:21-27`).

**Twelve sites: nine ETI, three true `az.hdi`.** The ten below were independently
re-classified three times and every cited line is exact. Two more shipped on 2026-07-28,
after the first draft, both in `planning/forecast.py` (#223/PR #263):
`:233 window_total_interval` and the inline `:593` — **both already in PR 5's proposed
canonical spelling**, which makes the consolidation a drop-in there. A thirteenth,
`planning/pacing.py:324-325`, uses the `100 - lo_q` form and was verified bit-identical at
p ∈ {0.8, 0.9, 0.94, 0.95}.

The freshest evidence for this PR is `planning/forecast.py:498`, which documents its
parameter as *"Central interval width; equal-tailed, matching `compute_hdi_bounds`"* — the
name-lies-about-ETI defect has now propagated into a brand-new v1.4 planning surface. And
`planning/forecast.py:601-602` sits under #225's **1e-9 reproduce-from-provenance contract**
for already-committed plan versions: any shift there breaks every committed plan. PR 5 is
safe there **if and only if** it adopts that spelling literally, which it already promises —
add `planning/forecast.py` to the mapping table with that note.

**Precedent to cite in the PR body:** #218 ("One family-aware, per-draw carryover-profile
reader; delete the four wrong ones") catalogued *"five horizon definitions under one word"*,
was accepted as a **bug** rather than a refactor, and shipped in two PRs. That is
structurally the same argument, already ratified by this project.

```text
ETI (seven):   utils/statistics.py:17           compute_hdi_bounds
               model/base.py:116                _hdi_finite
               reporting/helpers/cfo.py:29      _eti
               reporting/interactive/facts.py:126  _eti
               reporting/deck/builder.py:46     _pct
               validation/spec_curve.py:287     _eti
               reporting/extractors/base.py:97  DataExtractor._compute_percentile_bounds

TRUE HDI (three):  reporting/helpers/utils.py:81      _compute_hdi
                   reporting/extractors/base.py:63    DataExtractor._compute_hdi
                   utils/arviz_compat.py:62           hdi_bounds   (the shared routing target)
```

**Merging all ten changes the dashboard ROI interval.** Both `_compute_hdi` copies fall back
to ETI when arviz is absent (`extractors/base.py:90-95`, `helpers/utils.py:106+`), so "true
HDI" holds only because arviz is a hard dependency — worth a comment.

**Proposed API** — four orthogonal knobs, which is what the ten call patterns decompose into:

```text
interval(samples, prob=0.94, *, method="eti"|"hdi", axis=0|None,
         nonfinite="none"|"nan"|"finite",
         on_degenerate="raise"|"nan"|"passthrough",
         min_draws=1) -> tuple[ndarray|float, ndarray|float]
```

Bit-stability contract, to be honoured literally:

* `method="eti"` computes `lo_q = (1 - prob) / 2 * 100` and `hi_q = (1 + prob) / 2 * 100`
  **in that spelling** and calls `np.percentile(..., axis=axis)`. The four existing
  spellings were verified bit-identical (quantiles *and* resulting percentiles compared with
  `==`) at p ∈ {0.8, 0.9, 0.94, 0.95}; pinning one removes the ULP hazard for user-supplied
  probs.
* `method="hdi"` routes through `arviz_compat.hdi_bounds` and is legal only with `axis=None`.
* `nonfinite="nan"` uses `~np.isnan`; `"finite"` uses `np.isfinite`; both mask-index and so
  flatten, forcing `axis=None` — matching `_hdi_finite`, `_compute_hdi` and `spec_curve._eti`.

**The migration is a signature-preserving wrapper, not an alias.** `compute_hdi_bounds =
eti_bounds` **breaks every call site**: the live signature is
`compute_hdi_bounds(samples, hdi_prob=0.94, axis=0)` and `hdi_prob=`/`axis` are passed by
keyword at ~15 sites (`estimands/evaluate.py:538`, `model/base.py:127, 3837, 4202, 4600`,
`mmm_extensions/models/base.py:778`, `reporting/interactive/facts.py:195`, plus
`tests/test_utils.py:22, 38, 52, 53, 68, 72, 82, 96`), while the new API renames `hdi_prob`
→ `prob` and makes `axis` keyword-only. Each legacy name stays a **real function with its
original signature**, delegating to `interval(...)`.

**Ship a helper → knob mapping table in the PR body** (helper, `method`, `axis`,
`nonfinite`, `on_degenerate`, `min_draws`). That table, not the prose, is the migration
spec. Call sites measured: ~50 in `src/`, ~80 including `tests/`.

`compute_hdi_bounds` is not in `PUBLIC_SURFACE` or `CORE_IMPORTS`, so exporting
`eti_bounds` as the primary name is minor-safe.

**Also in this PR:** copy the canonical `lo_q`/`hi_q` spelling into the docstrings at
`facts.py:21-27` and `script.py:21-27` — they are the only thing keeping the browser and
Python in agreement.

**Out of scope:** `hdi_dataset` (arviz container, different return type),
`bc_interval`/`bca_interval` (need `point`/jackknife — they stay in
`frequentist/bootstrap.py`), `jeffreys_interval` and the analytic CIs (not draw quantiles).

### PR 6 — `ratio_summary` and the divisor boundary

**Why:** the same `num/den → mean/lo/hi` shape is hand-rolled repeatedly, differing along
axes nobody chose. Parameterise the axes instead of the code:

```text
ratio_summary(num_samples, denom, *,
              point_rule="mean_of_ratios"|"ratio_of_means",
              zero_policy="skip"|"zero"|"nan"|"keep",
              interval_fn=eti|hdi|finite_eti, prob) -> RatioSummary
```

Every call site keeps its current triple, so no number moves — and the differences become
declarative and greppable rather than buried in re-implementations.

**The divisor move is the layering half.** `reporting/helpers/measurement.py::resolve_channel_divisor`
is a *measurement* concept, not a reporting one, and `model/base.py:4269` already imports it
upward. Move `measurement.py` to `mmm_framework/measurement.py` (t0) and re-export from
`reporting.helpers.measurement`. That also removes the `estimands/evaluate.py:436,455`
upward reaches.

**Four sites can then adopt the resolver:** `planning/budget.py:111`, `planning/history.py`,
`mmm_extensions/models/structural.py:1097`, `model/base.py:3261`. Adoption is
**byte-identical for SPEND channels** (`measurement.py:479-481` is the legacy raw sum) and
changes numbers **only** for `cpm`/`cpc`/`spend_column`/efficiency channels — where the
current value is wrong: the Performance-page ROI trajectory and the dashboard ROI are on
different scales today for any non-SPEND channel. **Ship that adoption as its own
changelogged fix**, not inside the consolidation.

**`model/base.py:1806 _roi_mode_divisor` is explicitly OUT OF SCOPE.** It looks like a
fifth adoption site and is not. It returns `None` for non-SPEND channels *by design*
(`:1810-1813`): an efficiency channel's break-even reference is 0, so the ROI-scale
`LogNormal(0,1)` prior — whose median 1.0 *means* break-even — is meaningless there, and the
channel deliberately falls back to the coefficient prior. Returning a number instead would
create `roi_<ch>` where `beta_<ch>`'s Gamma prior was: **a different set of free RVs, i.e. a
PR 0.1 fingerprint break, not a published-number tweak.** It would also move
`frequentist/design.py`'s matrix, which is pinned to the PyTensor graph at `TOL = 1e-12`
(`tests/frequentist/test_design_equivalence.py:51`).

### PR 7 — `temporal_features.py` (basis only)

Per [§1.2](#12-do-not-unify-the-trendseasonality-rv-builders), only the numpy layer is
shareable. Extract into `transforms/temporal_features.py`:

```text
trend_basis(n, trend_config) -> TrendBasis
seasonality_basis(t, config, *, component_periods, require_min_period=None)
    -> SeasonalityBasis
```

**`n` must be an explicit axis choice, and this is the whole subtlety.** The core builds its
grids over `self.n_periods` (`base.py:1488`, `:1613`, `:1678`) and then gathers with
`[time_idx]`; `temporal.py:84` and `:171` use `n_obs`. These are identical on a national
panel and differ by a factor of `n_cells` on a geo panel. The signature must make the caller
say which, or the "lift" becomes a behaviour change on panels.

Note the lift spans more than one method on the core side: `base.py:1486-1518`
(`_prepare_trend`) has exactly one `np.linspace` and produces `spline_basis` /
`changepoints` / `changepoint_matrix` / `gp_config`; the piecewise `t_unique` (`:1613`) and
the GP grid `np.linspace(-1, 1, n_periods)` (`:1678`) live in `_build_trend_component`.

**The caller supplies `component_periods`**, which is what keeps the 52.0-vs-52.179
divergence from being silently resolved in one direction.

**The win beyond DRY:** `validation/backtest.py:170-176` holds a copy-pasted literal of the
core period table, linked only by a comment; values match today and nothing asserts it.
Repointing the forecaster at the shared helper kills that drift class. Add the missing test:
assert `seasonality_basis(...).periods` equals the backtest's table for all three
frequencies.

**Do not attempt Layer 2** (the RV builders with `CORE`/`EXTENSION` dialect records) without
PR 0.1 fingerprints covering NestedMMM and MultivariateMMM at all four trend families. The
dialect design is sound but it is a graph change with a naming contract on both sides.

### PR 8 — `reporting/shell.py` + `reporting/format.py`

**Why:** nine sites emit a full HTML document — `reporting/generator.py:446` and `:592`,
`prefit.py:1558`, `interactive/generator.py:1102`, `consultant_artifacts.py:1448`,
`model_defense.py:203`, `agents/report_builder.py:1386` and `:1745`,
`validation/results.py:1499`. The scrollspy block at `generator.py:576`, `prefit.py:1542`
and `interactive/generator.py:1086` is **byte-identical** — md5 `6544f7c99b486ead5bd772c5f84c206d`
at all three, reproduced independently. The Plotly CDN tag appears at 5 sites in 4 distinct
spellings.

The formatter sprawl collapses cleanly: 4 identical escape functions → 1; 4 near-twin money
formatters → 1; 6 twin finite-coercers → 1. **Do not** add `_pct` to that list (see §1.4),
and do not merge the two LLM-reply cleaner families into one — ship
`extract_llm_text(reply, *, collapse=False)`.

**Security angle, which is the real justification:** the shell is where escaping is applied
or forgotten, and it is currently forgotten. `ReportConfig.generated_date` is user-supplied
(`reporting/config.py:191`) and is interpolated unescaped at **four** sites:
`reporting/generator.py:373`, `:402`, `:504`, `:564` — next to correctly escaped neighbours
at `:370`, `:376`, `:501`, `:503`. Centralising means the shell escapes `title` /
`Masthead.title` / `eyebrow` / `meta_bits` / `NavItem.title` by contract, with callers
passing raw for those and pre-escaped-or-trusted for `body` / `footer_html` / `logo_svg` /
scripts.

**Migration order,** safest first: `model_defense.py` (smallest, no nav) →
`consultant_artifacts.py` → `validation/results.py` → `prefit.py` →
`interactive/generator.py` → `generator.py` (both shells) → `report_builder.py` (two shells;
the Reveal.js one is structurally different — last, or not at all).

**Verify with golden-HTML diffs, and pin the non-determinism first.**
`reporting/generator.py:362` is `generated_date = self.config.generated_date or
datetime.now().strftime("%B %Y")`. Before capturing goldens, pin `ReportConfig(generated_date=…)`
on all nine shells and any `datetime.now()`/`uuid` in the other six modules, and assert
`div_id`s derive from section names. **This adds fixture work the Phase 2 estimate does not
contain:** the goldens need a fitted model per shell, and the reporting suite uses fakes
today (588 collected; 568 pass / 20 deselected in 4.41 s fast). Its assertions are mostly
presence and shape — `test_sections.py` and `test_generator.py` carry no assertion on a
computed report number, and `test_ppc_estimands_reporting.py` only asserts loose bounds
(`r2 > 0.5` at `:126`, `cov90 > 0.6` at `:333`).

### PR 9 — the small utility clusters

Ranked by evidence. Two carry traps.

1. **`_finite` F1 → one function** (`utils/numeric.py::finite_or_none`). Five copies —
   `reporting/evidence.py:142`, `reporting/extractors/mixins.py:19`,
   `diagnostics/convergence.py:49`, `diagnostics/snapshot.py:34`,
   `reporting/triangulation.py:153` — verified behaviourally identical across 11 probe
   inputs. **Do not absorb the two F2 copies** (`estimands/evaluate.py:581`,
   `planning/simulation.py:766`): they call `isfinite` *before* `float()`, so they raise
   where F1 returns `None`. Merging turns a crash into a silent `None` on the
   estimand-evaluation path.
2. **`_phi` → one package-private `planning/_stats.py`.** There are **seven** `_phi*`
   definitions — plus an eighth normal CDF a `_phi`-shaped grep misses,
   `frequentist/bootstrap.py:271 _norm_cdf` (scipy; leave it alone, it is in a different
   package and a different tier). Of the seven: six normal **CDFs** (`cpa.py:46 _phi_cdf`, `evoi.py:243 _Phi`,
   `experiment_optimizer.py:49`, `design_anchor.py:39`, `identification.py:91`,
   `methods/ghost_ads.py:61`) and one **PDF** — `evoi.py:239 _phi` — with both consumed
   together at `evoi.py:252`. **A mechanical "consolidate all `_phi`" swaps a PDF for a CDF
   and corrupts every EVOI number.** Rename them `_norm_pdf`/`_norm_cdf` in the same commit.
   An eighth copy is inlined at `experiment_optimizer.py:90-91`.
   Keep this out of `utils/`, with a comment: **the erf formula is load-bearing.** Over
   `linspace(-6,6,100001)` only **48,779 of 100,001** values are exactly equal to
   `scipy.stats.norm.cdf` (max |Δ| 2.22e-16 — ULP level on that grid); the qualitative
   failure is further out, at x ≈ −37, where the erf form underflows to `0.0` and scipy
   returns `5.73e-300`. Two separate facts; do not present them as one.
3. **`_clean` to_dict scrubber** — three nested closures (`experiment_optimizer.py:394`,
   `experiment_value.py:74`, `opportunity_cost.py:111`), behaviourally equivalent for floats
   though textually different (`math.isfinite` vs `np.isfinite`; different comprehensions).
4. **`_json_safe` → `utils/jsonsafe.py`.** There are exactly **four** definitions in the
   package — `eda/results.py:22`, `continuous_learning/serialize.py:68`,
   `agents/kernels.py:198`, `synth/mff.py:96`. (There is **no** `_json_safe` in
   `reporting/`.) The canonical needs three explicit axes because the policies genuinely
   conflict:
   `json_safe(obj, *, nonfinite="null"|"keep", arrays="list"|"keep", unknown="passthrough"|"str"|"drop"|"raise")`.
   * Land `utils/jsonsafe.py` with the **two zero-risk copies** only
     (`continuous_learning/serialize.py`, and `synth/mff.py` behind the diff below).
   * `synth/mff.py` changes `synthetic_truth.json` content (datetimes stop being dropped) —
     **its own changelogged PR**, behind a diff of `truth_summary()` across every
     `dgp.SCENARIOS`/`dgp_geo.SCENARIOS`.
   * `eda/results.py:22` changes output **types** — its `json.dumps(obj, cls=NumpyEncoder,
     default=str)` shadows `NumpyEncoder.default`, so `np.int64(5)` currently serialises as
     `'5'` and `np.array([1,2])` as `'[1 2]'` (verified by execution). **Its own
     changelogged PR**, behind a before/after output-diff harness.
   * `agents/kernels.py:198` is **last, and optional**. It is the checkpoint-safety copy; its
     ndarray and NaN policies are deliberately opposite to the JSON copies
     (`agents/serde.py:43-46` documents why) and five test functions in
     `tests/test_oracle_checkpoint_serde.py` (`:41, :57, :63, :68, :75`) pin it.
   * Leave the server's `safe_json_dumps` and `_msgpack_coerce` alone.

**Do not touch:** `_pct`, `_now`, the five quantile implementations.

---

## 5. Phase 3 — break the one real cycle

### PR 10 — move `spec_locks` out of `agents/`, and repoint the one importer

**Why:** `platform/runs.py:19` is the single module-level import that closes the only
package-level cycle in the codebase. `agents/spec_locks.py` is 209 lines with **zero**
`mmm_framework` imports — a pure spec-path utility that happens to live in the agent package.

**The import site is the edge, not the definition site.** Moving the module and leaving a
re-export in `agents/spec_locks.py` changes nothing if `runs.py` keeps importing through the
shim: the `platform → agents` edge persists verbatim and the SCC survives.

```text
1. MOVE   src/mmm_framework/agents/spec_locks.py -> src/mmm_framework/specs/locks.py
2. EDIT   src/mmm_framework/platform/runs.py:19
             from mmm_framework.specs.locks import flatten_leaves
          <- THIS SINGLE LINE IS THE PR. Everything else is bookkeeping.
3. Re-export from agents/spec_locks.py for the other importers, including the
   module-level one at server/src/mmm_framework_server/main.py:33.
4. Add `specs` to LAYERS at t0, and `mmm_framework.specs.locks` to CORE_IMPORTS in
   tests/test_lean_imports.py (agents.spec_locks is listed there at :97, and the
   lean contract must follow the code).
5. EDIT CLAUDE.md:16, which lists spec_locks among the langchain-free agents modules.
```

**Result:** the module-level package graph becomes acyclic. State that in the PR body — it
is the headline, and it is one line of code.

### PR 11 — move the posterior helpers out of `reporting/`

`estimands/evaluate.py` reaches *up* into the reporting layer for three private helpers:
`_get_posterior` (`:402`, `:464`), `_get_scaling_params` (`:464`) and `_compute_hdi`
(`:529`). `reporting/helpers/utils.py` is 242 lines importing only `typing`, `numpy` and
`loguru` (plus a `TYPE_CHECKING`-only `arviz`) and has no `mmm_framework` import at all, so
nothing about these functions is a reporting concern.

```text
MOVE  reporting/helpers/utils.py  _get_posterior, _get_scaling_params, _compute_hdi
                                  (+ _flatten_samples, for cohesion — estimands does
                                   not use it, but its three siblings do)
  ->  mmm_framework/utils/posterior.py     (t0)
      re-export from reporting.helpers.utils, and keep the
      reporting/helpers/__init__.py:50-53,146-149 __all__ bindings intact
```

Then delete the three function-local imports at `estimands/evaluate.py:402, 464, 529`.

**PR 11 must land after PR 5.** Both rewrite `reporting/helpers/utils.py:81 _compute_hdi`:
PR 5 turns it into an `interval(method="hdi")` wrapper, PR 11 moves it to
`utils/posterior.py`. This is a guide-internal ordering constraint, not a GitHub collision,
and it is the only one in the plan.

**Two of the six sites are not this PR's:** `:436` and `:455` (`resolve_channel_divisor`) go
away with PR 6's `measurement.py` move. **And one is addressed by neither:** `:463` imports
`_get_contribution_samples` from `reporting/helpers/roi.py`, which no PR in this plan moves —
it remains an upward reach and stays on the allowlist. Say so rather than implying the edge
is gone.

**Verify:** `uv run pytest tests/test_estimands.py -q` **including the slow tier** — the
`rtol=1e-9` equivalence gate is `@pytest.mark.slow` (`:229`) and would otherwise not run.

---

## 6. Phase 4 — split `BayesianMMM`

4,806 lines; **88** method definitions; `_prepare_data` writes 45 attributes;
`_build_model` spans `2419-2920`; `fit` is `3320-3572` (253 lines). This phase is where the
value is and where the risk is. Do not start before Phase 0 is merged and the fingerprint
matrix is complete.

### 6.1 The frozen surface — what may not move

**Public methods** (`docs/api-contracts.html:178` lists them as frozen; every one must stay
callable on `BayesianMMM` or it is a major-version event *and* it fails
`tests/test_docs_snippets.py`, which binds `mmm`/`model`/`fitted_model` to the class at
`:50-52`): `fit()` (`:3320`), `predict()` (`:3782`), `sample_channel_contributions()`
(`:4610`), `predict_under()` (`:4438`), `evaluate_estimands()` (`:4493`).

**Also frozen:** `mmm_framework.analysis`'s `MMMAnalyzer`, `MarginalAnalysisResult` and
`compute_contribution_summary` are in `PUBLIC_SURFACE` (`tests/test_api_contracts.py:89`).
PR 12 folds analysis logic *into* `MMMAnalyzer`; it must not drop the names.

**The de-facto override contract** — exactly seven members, from reading all eight garden
models:

```text
_prepare_data(self)                                  # base.py:808 has no -> None annotation
_build_model(self) -> pm.Model
_build_coords(self) -> dict
_channel_media_input(self, c, channel_name, X_media_raw_data)
_add_experiment_likelihoods(self, channel_handles) -> None
_default_estimands(self) -> list
fit(self, *args, **kwargs)      # latent_factor_mmm.py:435 wraps it to inject initvals
```

**A second, undocumented contract:** the private helpers those overrides call on `self` —
`_build_coords()`, `_prepare_raw_media_for_model()`, `_build_channel_saturation(ch)`,
`_channel_adstock_apply(ch)`, `_selection_active()`, `_geometric_adstock_per_cell(X, α)`,
**`_build_control_betas(sigma)`** (called by `awareness_structural_mmm`, `latent_factor_mmm`,
`long_term_brand_mmm`) and **`_resolve_estimand(ref)`** (called by `breakout_weighted_mmm`,
`latent_factor_mmm`) — plus the module-level `_apply_saturation_pt` and
`_sample_from_prior_config`.

`agents/garden_authoring.py:71-77` (plus `:182`, `:205`) documents seven of these to the LLM
as the authoring API: `_build_coords`, `_prepare_raw_media_for_model`,
`_build_channel_saturation`, **`_build_trend_component`**, **`_build_control_betas`**,
`_apply_saturation_pt`, `_sample_from_prior_config`. It does **not** name
`_channel_adstock_apply`, `_selection_active` or `_geometric_adstock_per_cell`, which the
example models do use — an inconsistency to fix in the same PR. Note `_build_control_betas`
also appears in PR 14's extraction list; the two must be reconciled, because a `self`-less
`GraphKit` breaks the documented `self._build_control_betas(sigma)` call.

None of this is pinned by any test today. **PR 0.4 pins it.**

Non-subclass consumers of the same private surface: `frequentist/design.py` (six methods —
`_get_adstock_config`, `_get_saturation_config`, `_likelihood_config`,
`_prepare_raw_media_for_model`, `_roi_mode_divisor`, `_selection_active` — plus 30 distinct
attributes across its two model receivers), `frequentist/search.py:257`,
`validation/backtest.py:711,725,727,773`, `reporting/helpers/saturation.py:108,217,577`,
`reporting/extractors/bayesian.py:2126,2213`, `diagnostics/saturation.py:118`,
`transforms/carryover.py:247`, `agents/model_ops.py:2993`, `calibration/experiment.py:433`,
`estimands/evaluate.py:515,533`, `analysis.py:126,155,375`.

### 6.2 Order of extraction — safest first

**PR 12 — `Analyzer`.** `predict`, `predict_under`, `compute_component_decomposition`,
`compute_counterfactual_contributions`, `compute_marginal_contributions`,
`what_if_scenario`, `sample_channel_contributions`, `sample_latent_under`,
`_swapped_media_data`, `_intervention_to_*`.

The reason this goes first is **not** that it is the smallest read surface — measured, the
group touches ~18 state attributes and calls 6 internal helpers. Beyond the obvious nine
(`_trace`, `model`, `y_mean`, `y_std`, `_multiplicative`, `channel_names`, `X_media_raw`,
`panel.index`, `n_obs`) it reads `control_mean`, `control_std`, `control_names`,
`n_controls`, `X_levers_raw`, `_lever_names`, `_price_lever`, `n_levers`, `has_geo`,
`has_product`, `use_parametric_adstock`, and calls `_check_controls_swappable`,
`_get_time_mask`, `_intervention_to_inputs`, `_multiplicative_decomposition`,
`_prepare_media_data_for_model`, `_prepare_raw_media_for_model` — which must move with it or
be injected as callables. The control-scaling triple and the lever fields are what make
`predict_under` not a pure posterior reader.

**The real reason it goes first: it is the only group that writes nothing to `self`**
(except `_swapped_media_data`, which mutates the graph via `pm.set_data` inside a
try/finally). `BayesianMMM` keeps thin delegating methods so the frozen surface is untouched.

Fold `MMMAnalyzer` in at the same time. Seven of its ten methods are pure pass-throughs with
zero logic to preserve; only `compute_channel_roi` (`analysis.py:272-341`) and
`compute_saturation_curves` (`:343-405`) hold behaviour that does not exist on the model.
`compute_channel_roi` must survive because the **class is in `PUBLIC_SURFACE`** — not because
it is the only path to the number: `tests/test_estimands.py:254-284` pins the estimand
engine's `counterfactual_roi` bit-identical to it. Widen `what_if_scenario` while you are
there; it exposes 3 of the model's 6 parameters (`analysis.py:266-270`).

`_intervention_to_X_media` (`model/base.py:4423`) has no caller in `src/`, `examples/` or
`nbs/`, **but `tests/test_price_promo_levers.py:415,417` pins its lever-refusal behaviour as
a regression guard** (the bug is recorded at `:362`). Deleting it means deleting that guard —
either keep the method or move the guard onto `_intervention_to_inputs`.

**PR 13 — `Estimator`.** `run_approximate_fit(model, method, draws, seed)` and `run_smc_fit`
are already the pattern applied to fitting: they take a bare `pm.Model` and are reused by
`BaseExtendedMMM` and `planning/methods/tbr.py:206`. `fit()` is a dispatcher with three
near-duplicate `MMMResults(...)` construction sites (`:3473`, `:3519`, `:3563`) sharing six
identical kwargs. Collapse those, then hoist the four estimators behind one protocol.
`run_frequentist_fit(mmm, seed, **kw)` breaks the pattern — it takes the whole model because
it needs `model_config`, `panel`, `trend_config`. Leave it; note the exception.

**Fix two mutation hazards here, not later:** `fit()` writes
`self.model_config.fit_method = method` at `:3417` and `_fit_frequentist` writes `= None` at
`:3630`. Both mutate a **shared config object the caller may hold** — a `ModelConfig` reused
across two models is cross-contaminated today. Copy-on-write it.

**PR 14 — `GraphKit` (the RV factories).** `_build_channel_saturation` (`def` at `:2129`),
`_channel_adstock_apply` (`:2255`), `_build_channel_betas_geo` (`:1841`),
`_build_channel_beta_tvp` (`:2098`), `_build_control_betas` (`:1115`), `_build_coords`
(`:1527`), `_prepare_raw_media_for_model` (`:1763`), `_geometric_adstock_per_cell` (`:1726`),
plus the already-pure module-level `_apply_saturation_pt` (`:224`) and
`_sample_from_prior_config` (`:154`). Most read almost no `self` state:
`_channel_adstock_apply` reads two references, `_build_channel_saturation` two.

Follow the shape the codebase already proved in `mmm_extensions/components/temporal.py` — no
`self`, a `prefix`/`name` parameter instead of the implicit naming namespace,
`getattr`-defensive config reads, and `None` rather than a zeros tensor when off.

**Three landmines, each of which breaks naive extraction:**

* `_build_reach_frequency_gains` (`def` at `:1997`) **writes graph tensors into
  `self._freq_gain` at `:2034`**, and `_channel_media_input` reads them later in the same
  build (`:1801`).
* `_build_grouped_media_betas` (`def` at `:1859`) writes `self._pooled_channels` at `:1920`
  and **accumulates across builds** — it is only reset in `_prepare_data`.
* `_build_channel_interactions` resets `self._interaction_names` at `:2063` — *after* two
  early `return None` at `:2054` and `:2061`, so flipping a config from "has pairs" to "no
  pairs" leaves stale names.

`_kappa_bounds_for` (`def` at `:2204`) recomputes the entire normalized media matrix at
`:2214` to slice one column; pass the column in.

**Reconcile with §6.1 before starting:** `_build_control_betas` is on the documented
authoring surface. Either keep a `self._build_control_betas(sigma)` shim on the facade or
update `garden_authoring.py` and re-register the stored sources.

**PR 15 — `DataPrep`.** Last and hardest. `_prepare_data` writes 45 attributes, of which
**37 are read outside `base.py`**; only 8 are genuinely private (`X_levers_raw`,
`_freq_gain`, `_interaction_names`, `_lever_names`, `_price_ref`, `_promo_scale`,
`has_media_hierarchy`, `t_scaled`). The serializer writes 8 of them directly (11 counting
`_scaling_params`, `trend_features`, `seasonality_features`, which `_prepare_data`'s helpers
produce).

`_prepare_levers` (`:1343-1345`) and `_prepare_reach_frequency` (`:1404-1406`) are
**destructive, order-dependent in-place narrowings** of `control_names`, `X_controls_raw`
and `n_controls`. `breakout_weighted_mmm.py:310-330` rewrites four prepared attributes
*after* calling `super()._prepare_data()` — subclasses treat prepared state as a public
mutable surface.

A frozen `PreparedData` value object is the right destination, but ship it first as a
**read-through facade** (attributes still live on the model; `PreparedData` is a view) and
freeze only once the serializer is migrated.

### 6.3 The lazy-graph rule

`model` (`:3303-3308`) builds on first access, never in `__init__`. Nothing below may change:

* **Warnings raised inside the build are deferred to first `.model` access.** Nineteen
  `warnings.warn` sites across nine methods are reachable from `_build_model`; only two are
  in `_build_model` itself (`:2691`, `:2746`).
  `tests/test_prior_config_flow.py:253-256` requires the `.model` access to be *inside* the
  `pytest.warns` block — wrapping only `build_model` catches nothing.
* Config edits between construction and `.model` are picked up; data-prep input edits are
  not, because `_prepare_data` already ran.
* **`_build_model` is not idempotent with respect to `self`** — the three writers above, plus
  `fit()`'s `model_config.fit_method` write (`:3417`) and `_fit_frequentist`'s (`:3630`),
  which survive a `.model` rebuild. A second build re-runs them.

---

## 7. Phase 5 — the garden contract (v2.0)

De-inheriting `CustomMMM` is a **major-version** change. It breaks stored user artifacts,
which `docs/api-contracts.html:281` marks frozen. Do not attempt it before Phase 4 lands and
PR 0.4's corpus test is green.

### 7.1 What actually blocks it

There is exactly **one** `isinstance` on a model base class in production code —
`serialization.py:56`, `isinstance(model, BaseExtendedMMM)` — and it does not name
`BayesianMMM`. The real blockers are MRO-*name* checks an `isinstance` grep misses:

| Blocker | Mechanism | Breaks |
|---|---|---|
| `garden/contract.py:99-102` `is_bayesian_mmm_subclass` | walks `cls.__mro__` for the literal strings `"BayesianMMM"`/`"BaseExtendedMMM"` | `validate_class` (`:200`) starts demanding `predict` + `sample_channel_contributions`. **None of the eight example models defines either method.** Three (`bayesian_cfa.py:67` `"cfa"`, `bayesian_lca.py:69` `"latent_class"`, `bayesian_clv.py:97` `"clv"`) declare a non-MMM `__garden_model_kind__` and are exempt at `contract.py:199`; the remaining five are MMM-kind → static tier fails → `load_garden_class_from_path` raises → **every registered MMM garden model becomes unloadable** |
| `garden/contract.py:311` `find_garden_class` | same check | `nested_survey_mediation_mmm.py` has no `GARDEN_MODEL` marker and resolves solely by this path |
| `garden/contract.py:121` `model_kind` fallback | `DEFAULT_MODEL_KIND if is_bayesian_mmm_subclass(cls) else "unknown"` | `serialization.py:478` then **skips y/media/control re-standardization** on reload — a silently mis-scaled model. **Narrower than it looks:** `CustomMMM` sets `__garden_model_kind__ = "mmm"` at `garden/base.py:111`, so its subclasses return at `:119` and never reach `:121`. The exposure is confined to garden models that subclass `BayesianMMM` *directly* — none in `examples/`, but the registry accepts them |
| `serialization.py:430-437` | fixed ctor keyword set `(panel, model_config, trend_config, adstock_alphas, experiments, model_params)` | a composed model must keep it exactly, or every core-flavor reload raises `TypeError`. Same contract positionally at `agents/fitting.py:1653` |
| cloudpickle `model.pkl` (`serialization.py:282`) | resolves imports itself; an alias table does not help | **Extended flavor only** (`_load_extended`, `:253`). Core/garden loads reconstruct via `_resolve_model_class` + the ctor at `:430` with no pickle, so this does **not** block de-inheriting `CustomMMM` — it blocks moving the `mmm_extensions.models.*` classes |

Plus **19 registered garden rows** in the dev database, of which **6 sources still resolve on
disk**; between them they call 12 inherited private members. And 9 sites where model source
is embedded in **strings** that get `exec`'d — grep-and-replace with no runtime guard.

### 7.2 The design, and the payoff

Components: `DataPrep` → frozen `PreparedData`; `GraphKit` (stateless RV factories);
`GraphSpec` (**the only thing an author implements**: `build(prepared, kit, params) -> pm.Model`);
`Estimator`; `PosteriorSurface`; `EstimandEngine`; `ModelPersistence`; `ModelFacade`.

`CustomMMM` becomes `ModelFacade(graph_spec=…, data_prep=…, estimator=…)`.
`CustomMMM._set_non_mmm_defaults` (`garden/base.py:72`, whose body at `:80-101` writes **17**
base attributes) becomes a `NullDataPrep` — an alternative implementation rather than a
de-neutralizing hack.

**The payoff is measurable: the 9-tier compatibility suite drops to 5.**

| Tier | Fate | Why |
|---|---|---|
| static | **gone** | `fit`/`predict`/`sample_channel_contributions` live on the facade by construction; what remains is `isinstance(spec, GraphSpec)` at registration |
| build | keep | author code still runs arbitrary logic |
| fit | **gone as a graded tier** | the `Estimator` owns `_trace` assignment and the `approximate` flag; the author never writes them. Keep it as a fixture step for tiers 5–9 |
| instance | **gone** | all 8 `REQUIRED_ATTRS` are produced by `DataPrep` and enforced at `PreparedData` construction |
| trace | reduce to the `beta_<channel>` naming check | the chain/draw dim check is guaranteed by the `Estimator`; the naming convention is author-written and is not |
| scaling | keep | original-scale `predict()` depends on author registration inside `build` |
| ops_smoke | keep — **most valuable post-refactor** | it checks author-declared posterior *names*; only execution can |
| carryover | **gone** | read the body (`compat.py:349-386`): after the `has_geo` check it only asserts `_trace is not None` after a second panel fit. It never checks the bleed invariant its docstring claims. Most expensive tier, duplicates tier 3. Replace with a `_geometric_adstock_per_cell` unit test — microseconds, no fit |
| accuracy | keep (advisory) | orthogonal; grades recovery against the synth answer key |

`BLOCKING_TIERS` (`compat.py:32-34`) becomes `{"build", "trace", "scaling", "ops_smoke"}`.

### 7.3 `BaseExtendedMMM` — the same problem, solved separately

`mmm_extensions/models/base.py` is 951 lines and inherits from `object`
(`contract.is_bayesian_mmm_subclass` nevertheless treats it as a model base by name). Its
data layer and graph are genuinely different from `BayesianMMM`'s — raw arrays vs
panel/`Dataset`/roles; joint multi-outcome likelihood. **Everything after the graph is
duplicated: 471 of its 951 lines, just under half.**

```text
model 154-159 (6) · save 261-277 (17) · load 279-302 (24)
add_experiment_calibration 308-323 (16) · fit 480-650 (171)
_swapped_data 665-706 (42) · predict 708-786 (79)
sample_channel_contributions 830-885 (56) · summary 887-893 (7)
sample_prior_predictive 895-902 (8) · compute_parameter_learning 904-948 (45)
```

Precision matters on two of these, because the document's first draft got them backwards:

* **`sample_channel_contributions` — both return original scale.** The core's docstring
  (`model/base.py:4610`) says "ORIGINAL-scale" and the method ends `return contrib *
  self.y_std` (`:4674`); the extension's deterministic already carries `y_std` so its method
  must *not* rescale (`mmm_extensions/models/base.py:830`). There is no caller-visible
  divergence. The divergence is in the **deterministic's scale convention** — standardized in
  the core graph, original in the extension graph — and any composition boundary must pick
  one, or the rescale lands twice. (That same convention split is what produces
  [§8.2](#82-double-y_std-scaling-in-the-pre-fit-readout-for-extension-models).)
* **`summary` is semantically equivalent, textually divergent** — different docstring,
  different fitted-check (`_check_fitted()` vs an inline raise), different import style.
  That is a *stronger* argument for consolidation, not a weaker one. `sample_prior_predictive`
  has the same name on both sides (`model/base.py:4708`, `models/base.py:895`) and an
  identical two-line body; only a local import differs. `add_experiment_calibration` really
  is line-for-line identical (`:321-323` vs `:790-792`).

Third foot-gun, real: `self.y` means **standardized** in the core and **raw** in the
extension.

`BaseExtendedMMM` has already reverse-engineered the base's attribute contract — the
synthesized `X_media_raw`/`y_raw`/`time_idx` properties at `:798-820` exist explicitly to
match it. That discovered contract *is* the composition boundary.

---

## 8. Bugs the audit found — fix separately, do not bundle

Each is a defect found while reading, not a refactor. Each deserves its own issue, its own
changelog line, and a stated reproduction.

**All six were searched against the full 79-issue history (open and closed, 20 keyword
queries). None is filed.** Two have close relatives whose framing matters: §8.1 is a
regression that **#219 half-created** on 2026-07-27, and §8.6's second half is the *sibling*
of a defect #220 named and PR #251 **already fixed** in a different file — do not restate
that one as live. Routed by owner rather than into the refactor tracker: §8.1 is filed as a `follow-up` to
#219, and §8.5 as a rider to the open #220. **All six were filed on 2026-07-30 as
[#273](https://github.com/redam94/mmm-framework/issues/273)–[#278](https://github.com/redam94/mmm-framework/issues/278).**

### 8.1 `train_offset` unit mismatch on a geo panel — [#273](https://github.com/redam94/mmm-framework/issues/273)

`validation/validator.py:755` passes `train_offset=int(np.asarray(train_indices).min())`
where `train_indices` are **observation** indices, but `backtest.py:926, 939-940` subtracts
it from **period** positions:

```text
all_periods = np.arange(n_full)
shared += self._trend_at(all_periods, train_offset)
shared += self._seasonality_at(all_periods, train_offset)
```

(`_trend_at`/`_seasonality_at` subtract at `:605`, `:614`, `:630`.) With `n_cells = 6` and a
window starting at obs 120 (period 20), the Fourier phase shifts by 120 periods instead of
20 and the linear trend evaluates far negative across the whole forecast window.

**Trigger:** geo/product panel + `cv_config.strategy == "rolling"` + `_run_cross_validation`.
Only `rolling` (`validator.py:586`) gives `min(train_idx) > 0`; the default is `"expanding"`
(`validation/config.py:70`), `expanding` (`:569`) and `blocked` (`:603`) both start at 0, and
neither `run_backtest` (`backtest.py:1328`) nor `planning/forecast.py:573` (shipped by #223)
passes `train_offset`. The only in-repo use of `"rolling"` is
`tests/validation/test_validation.py:561`. Confidence: high.

**#219 (PR #230, merged 2026-07-27) created half of this and sharpened the other half.** Its
own commit message lists *"Rolling-window CV forecast seasonality out of phase (`train_offset`
was honoured by the trend but not the seasonality)"* — the diff **added** `train_offset` to
`_seasonality_at` and passed it at both `:838` (national) and `:940` (geo). On a national
panel that is a correct fix; on a geo panel it propagated the obs-vs-period unit error from
the trend into the seasonality. Before #219, only the trend was mis-shifted. Separately,
#219's decimation guard at `backtest.py:533-537` made `_trend_component` a clean
length-`n_periods` array, so `np.clip(positions - train_offset, 0, n_train-1)` with an
obs-scale offset now pins the **entire** forecast window to index 0 whenever
`train_offset >= n_periods` — a trend frozen at the training-window start, not merely shifted.
File it as a #219 follow-up in the v1.4 milestone, not as a refactor item.

**#219's own safety net cannot see it:** `grep -rn train_offset tests/` returns **zero hits**,
and the component-sum identity tests (`tests/test_backtest_completeness.py:385-451`, including
two geo cases at `:401` and `:434`) all call `forecast(...)` at the default offset of 0.

**Secondary, same path — file it with the primary, the evidence is stronger than first
thought.** `_slice_panel_data(panel, train_indices)` (`validator.py:631`, def at `:640`)
slices a geo panel by raw obs index, so a boundary that is not a multiple of `n_cells`
produces a ragged panel that `_forecast_geo`'s reshape (`:925`, `:928`) and the decimation
guard (`backtest.py:533-537`) both assume away. Decisive:
`grep -n n_cells src/mmm_framework/validation/validator.py` returns **nothing** — the whole
CV-split machinery is cell-blind for all three strategies, and `rebuild_like`
(`backtest.py:1154`, new in #219) adds no ragged guard either.

### 8.2 Double `y_std` scaling in the pre-fit readout for extension models — [#274](https://github.com/redam94/mmm-framework/issues/274)

`trend_component` and `seasonality_component` are registered on the **standardized** scale in
a core graph (`model/base.py:2482`, consumed `* y_std` at `:3910-3911`) but in **original KPI
units** in an extension graph (`nested.py:358`, `structural.py:783`).
`reporting/helpers/prefit.py:598 prior_component_facts` is reachable on an extension model
via `agents/tools.py:3545-3547` → `build_model` → `_build_extension_model`, and the helper
multiplies by `y_std` a second time at `reporting/helpers/prefit.py:644` (reached from
`reporting/prefit.py:114`) — a `y_std²`-scaled band in the Model Design Readout.

Code path confidence high. **Repro to construct before filing:** an extension spec
(`dag_model_type="nested_mmm"`) with trend or seasonality configured, then
`generate_model_design_readout`.

### 8.3 The extension seasonal period does not match the core — [#275](https://github.com/redam94/mmm-framework/issues/275)

52.178571… (`temporal.py:220`) vs 52.0 (`base.py:1452`) for weekly data, against a docstring
at `temporal.py:195-196` claiming the component "is comparable across models". Measured max
|Δ| of the order-2 yearly Fourier design over 104 weekly points: **0.08174** (reproduced
independently; order 1 → 0.04216, order 3 → 0.12376).

Fixing it changes extension-model numbers. Ship behind a `SeasonalityPeriodSource` enum
defaulting to current per-site behaviour, so the fix is opt-in and the refactor stays
bit-stable.

### 8.4 Two wrong branches in the report's ROI extractor — [#276](https://github.com/redam94/mmm-framework/issues/276)

`reporting/extractors/bayesian.py:1630` is `contrib_samples = vals.flatten() * y_std * n_obs`
where the canonical path (`reporting/helpers/roi.py:208`) is `samples * y_std` with no
`n_obs`. `:1657` is `beta_vals * y_std * n_obs * 0.5` — under a `# Rough estimate` comment at
`:1656` — where the canonical (`roi.py:268`) is `beta_samples * media_sum * y_std`. The ratio
is `media_sum / (0.5·n_obs)`, i.e. twice the mean per-period spend, so for any real spend
series these differ by orders of magnitude. Both are fallback branches, firing only for
models that expose a scalar-per-draw contribution or only `beta_<ch>`.

**File all four sites, or the other two get re-found.** The same `# Rough estimate` comment
introduces two more hand-rolled fallbacks in the same extractor:
`reporting/extractors/bayesian.py:1797` and `:1925` (`# Rough estimate: +/- 15% of total`).

### 8.5 The classic report publishes `contribution_roi` twice — [#277](https://github.com/redam94/mmm-framework/issues/277)

`reporting/generator.py:223` and `:225` both wire their sections, and `SectionConfig.enabled`
defaults to `True` (`reporting/config.py:159`), so one document renders the same estimand as
an **80% ETI** (`bayesian.py:1665-1670`, `ci_prob: float = 0.8` at `:58`) and a **94%
`az.hdi`** (`estimands/registry.py:63` `Realization(point_rule="mean_of_samples",
hdi_method="az_hdi")` + `estimands/spec.py:265` default `hdi_prob = 0.94`, carried unchanged
through `reporting/extractors/mixins.py:487`), with no cross-reference. Two masses and two
interval definitions. Pick one or label both.

**File it as the second half of #221** (closed): *"one value basis and one break-even
convention, gated at every render site"* unified the ROI **reference** across
`ChannelROISection`, `EstimandsSection`, `augur_sections.py:52` and `deck/engine.py:143` via
`MetricMeta`, and explicitly did not touch interval mass or the duplicate render. Framing
this as #221's residue on the same two sections will get it prioritised correctly.

### 8.6 Two gaps in the estimand engine — [#278](https://github.com/redam94/mmm-framework/issues/278)

**`contribution_roi` silently ignores `Estimand.window`** — verified by execution: a
`window=TimeWindow(0, 9)` returned exactly the full-series value with no warning. Mechanism:
`evaluate.py:441-443` forces `mask=None` for `ObservedInput(source="panel")` and `:471-478`
sums unmasked.

**`estimands/evaluate.py` has no multiplicative guard** — zero occurrences of
`multiplicative` anywhere in `estimands/` — so `marginal_roas` returns a number where the
model refuses.

**This is now an asymmetry, not a symmetry, and the framing must change.** #220 named the
same class of defect on `sample_channel_contributions`, and **PR #251 fixed it on
2026-07-28**: `model/base.py:4647-4655` now raises, mirroring the long-standing guard on
`compute_marginal_contributions` (`:4247-4254`). So `model/base.py` refuses in **two**
places and `estimands/evaluate.py` refuses in **none**. The new guard does not cover the
estimand path: `contribution_roi` reaches `_get_contribution_samples`, which reads
`posterior[var_name].values` directly and never calls `sample_channel_contributions`. File
this as *"extend #251's guard to the estimand engine — or decide the model's guard is
over-broad"*, citing #220 and #251. **Do not restate #220's defect #4 as live.**

The over-broad reading is still open: `predict_under` (`base.py:4438`) has no guard and
hard-codes `return_original_scale=True` at `:4455`, so the engine diffs exp-back-transformed
predictions — exactly the remedy `base.py:4252-4255`'s own error text prescribes.

**A note on reading #220 itself:** its cites (`base.py:4378`, `:4074-4081`) were *correct
when written* on 2026-07-26 and have since drifted by +232 and +174 lines, via #251, #253 and
#261. Its analysis is trustworthy; only its coordinates are stale. Issue bodies in this repo
go stale against the code within days — cite behaviour, and re-derive line numbers at the
time you act.

---

## 9. Sequencing, versioning and effort

### 9.1 The table

| Phase | PRs | Version | Blocks | Blocked by | Est. |
|---|---|---|---|---|---|
| 0 safety net | 0.1, 0.2, 0.3, 0.4 | minor | everything | — | 5–7 d |
| 1 deletions/moves | 1, 4 | minor | — | 0.3 (LAYERS) | 1–2 d |
| 1 deletions/moves | 2 interim, 3a | minor | — | — | 2 d |
| 1 deletions/moves | 2 deletion, 3b | **major / later minor** — owned by #193 | — | 3a shipped | 1 d + migration |
| 2 consolidation | 5, 6, 7, 8, 9 | minor | — | 0.1, 0.3 | 10–15 d |
| 3 cycle break | 10, 11 | minor | — | 0.3 | 1–2 d |
| 4 split | 12, 13, 14, 15 | minor *(facade preserved)* | 5 | 0.1, 0.2, 0.4 | 15–25 d |
| 5 garden contract | — | **major (v2.0)** | — | 4 | 15–20 d |
| §8 bug fixes | 6 issues | minor | — | — | 4–6 d |

Phase 2's estimate excludes PR 8's golden-model fixtures (see PR 8). Phase 4 is "minor" only
while the facade preserves **both** `BayesianMMM`'s five frozen methods **and**
`mmm_framework.analysis`'s three `PUBLIC_SURFACE` names, and while PR 0.4's authoring-surface
snapshot stays green.

### 9.2 Where this work lives on GitHub

**Not a new milestone.** `roadmap.md:6` — "Milestones on GitHub mirror this file" — and
`roadmap.md:9-17` is one milestone per quarter from Q3 2026 to Q1 2028 with no free slot.
Opening one forces a roadmap row, which forces a theme, which forces "The bar"
(`roadmap.md:55-61`: *a feature ships with evidence that it recovers a planted truth*). At
32–51 days this is quarter-sized, so it would have to **displace** a quarter — and it invites
exactly the reading v2.0's entry rule exists to prevent: a refactor quarter justified by
tidiness.

**Not v1.5 / #191 either.** #191's thesis is *bias*; a refactor makes no bias claim, and
folding this into a milestone whose sole issue carries two planted-truth success criteria
makes the milestone unmeasurable. There **is** a real dependency to record instead: #191's
proposed state-space trend family (`TrendType.LOCAL_LEVEL`/`LOCAL_LINEAR`) and its forecast
semantics land in `model/base.py:1486-1518` and `validation/backtest.py:170-176` — the exact
code PR 7 lifts and PR 14 extracts. **Land PR 7 before #191 is decomposed**, and note it on
both sides.

**Two artifacts, in the #176 style:**

* **[#279](https://github.com/redam94/mmm-framework/issues/279) — `[Tracking] src/ refactor: the safety net and the independent cleanups`.** Filed 2026-07-30, unmilestoned, label `tracking`. Spec is this document. Checkboxes for PR 0.1–0.4, 1, 4, 5,
  6, 7, 8, 9, 10, 11 with the §9.1 estimates and the order in §9.3. Children land on whatever
  minor is open — exactly how #169–#171 shipped under #176.
* **[#280](https://github.com/redam94/mmm-framework/issues/280) — `[Epic] Split BayesianMMM behind a preserved facade`** for Phase 4 (PR 12–15). Filed 2026-07-30, blocked on #279.
  An epic, not a checkbox: a 15–25 d sequenced program with a frozen surface (§6.1), an
  undocumented override contract, a lazy-graph rule (§6.3) and its own acceptance criteria.
  Leave it unmilestoned and blocked on #279 until Phase 0 is green, then milestone it into
  whichever quarter has capacity — realistically v1.6/Q2 2027. Decide then, not now.

**PR 2, PR 3b and Phase 5 belong to #193** — but only two of the three are already there.
#193 names `TrendBuilder` and the legacy sessions-DB fallback verbatim, so the scope
corrections were **posted as comments on #193** (2026-07-30) rather than filed as duplicates. Phase 5 is new
and should be a **separate epic linked from #193**, not a seventh candidate bullet — and it
must argue itself past the entry rule on its own terms. "The compat suite drops from 9 tiers
to 5" is a *benefit of changing*, not a *cost of keeping*; the cost of keeping is that the
private surface 19 registered models depend on is pinned by nothing, which is what PR 0.4
measures.

**A date correction the guide previously got from the wrong source.** The v2.0 milestone is
**Q1 2028 / 2028-03-31** (`roadmap.md:17`, `:341`, and milestone #7's `due_on`). Both #193's
body and milestone #7's *description* still read "Q2 2027" and are stale — an implementer
opening #193 plans nine months early. Cite the roadmap, and fix the two GitHub strings.

### 9.3 Order, given what is in flight

v1.4 is due 2026-12-31 with 7 open issues. The capacity conflict is smaller than it looks —
six of the seven are finance/planning/reporting/server/docs work — but three sequencing rules
fall out of the file-level collision matrix, and all three argue for coordination rather than
deferral:

1. **Ship the boundary PRs (5, 6) before the v1.4 features that would otherwise hand-roll
   around them.** #224 would ship a thirteenth `_eti` without PR 5. #226 replaces
   `planning/budget.py`'s `base_spend = X.sum(axis=0)` with `DecisionArm.cost_fn` and plans to
   *copy* `MetricMeta` — so **PR 6's `measurement.py` move must land first**, after which #226
   owns the `budget.py` divisor adoption instead of PR 6.
2. **A ratchet trap worth naming.** The allowlist keys on a per-file count, so adding an
   import to an already-allowlisted file fails. `model/base.py` already carries the
   `reporting.helpers.measurement` reach at `:4269`. If #226's `sample_lever_contributions`
   adds a second upward reach from that file, the gate fails — **unless PR 6 has moved
   `measurement.py` to tier 0 first.** Rule 1 is what closes this.
3. **Capture PR 8's golden HTML last.** #224, #226 and #227 each add a report section. PR 8's
   verification is golden diffs across nine shells; capturing them before those sections land
   means capturing them twice. (#220 is *not* the collision here — its remaining scope is
   `finance/closure.py` + `finance/lines.py`, two modules with no HTML. Its landed section
   work touched none of the nine shell sites.)

Two more: **PR 4 goes late** — it renumbers `agents/tools.py` by −331 lines and every v1.4
tool registration plus #228's integration checklist sit downstream. And **PR 0.3/0.4 sequence
*with* #228, not against it**: #228 already extends `FROZEN_ENUM_VALUES` and `CORE_IMPORTS` in
the same two files, ships `scripts/sync_api_surface.py`, and has worked out the
anti-rubber-stamp rule the other two gates lack. Its capability-reachability gate and PR 0.4's
`AUTHORING_SURFACE` are the same idea at two layers — say so, or they get built twice.

### 9.4 Branching

Minor PRs land on short-lived branches off `main`, one PR each. **PR 2's deletion, PR 3b and
Phase 5 do not merge to `main`** until a `v2` line opens; land only PR 2's lazy deprecation
and PR 3a. A reader who follows §9.1 literally deletes the legacy sessions-DB fallback in
week two and orphans their own 19 garden registrations.

### 9.5 Blast radius outside `src/`

**No PR in this plan requires a frontend change.** Server coordination is limited to three
points: PR 4's `_KERNELS` import (`server/src/mmm_framework_server/main.py:229, 311, 7310`),
PR 10's module-level `from mmm_framework.agents.spec_locks import …` (`main.py:33`, covered
by the re-export), and PR 3's `MMM_SESSIONS_DB` deploy wiring
(`deploy/gcp/vm/vm_setup.sh`). `make lint` covers `server/src`, and
`tests/contracts/rest_routes.json` is untouched by everything here.

### 9.6 If you only do three things

PR 0.1 (the fingerprint — the precondition for everything, and it already caught a false
collision), PR 5 (the interval consolidation and the honest rename — highest correctness
payoff per line changed), and PR 10 (one line; makes the module-level package graph acyclic).

**If you only do one:** PR 0.1. Everything else is safer with it and reckless without it.

---

## 10. Conventions for this work

**Commits:** `type(scope[,scope]): lowercase sentence stating the resulting invariant (#PR)`.
Types in use: `feat`, `fix`, `test`, `chore`, `release`. Scopes are package names —
`model`, `reporting`, `planning`, `platform`, `agents`, `server`, `frontend`, `config`,
`serialization`, `validation`, `transforms`, `synth`, `finance`, `skill`. Multi-scope commas
are common (`fix(planning,platform,frontend): …`). **The subject is declarative, not
imperative** — house style states the invariant that now holds ("a monthly plan starts on the
next month, not 30 days later"), not the action taken.

**Gates every PR must pass:** `make lint` — `ruff check src server/src` (`Makefile:8`,
identical at `ci.yml:53` and in `.githooks/pre-commit`, so it is one gate, not three) — and
`make fast_tests` (5,160 of 5,435 collected; 275 `@pytest.mark.slow`, all per-test; no
module-level `pytestmark` applies the slow marker — the one that exists,
`tests/test_prior_config_flow.py:31`, is a `filterwarnings` mark).
`.github/workflows/ci.yml` has no path filter and runs on every change; `docs.yml` filters
`src/**`/`docs/**` and does not fire on `technical-docs/`; `ci-slow.yml` is nightly +
`workflow_dispatch`.

**Changelog.** Behaviour-changing PRs here — PR 6's resolver adoption, PR 9's `synth/mff.py`
and `eda/results.py` migrations, all six §8 bugs — add an entry under `## [Unreleased]` in
`CHANGELOG.md`. Pure moves and consolidations that provably preserve numbers do not. At the
release carrying this work, follow CLAUDE.md's release checklist: bump **both**
`pyproject.toml:3` and `src/mmm_framework/__init__.py:155` (gated by
`test_package_version_matches_pyproject`), mirror the section into `docs/changelog.html` and
move the `Current` chip, sweep the version strings the checklist names, then re-run
`docs/tools/build_search_index.py` and `build_seo.py` and commit `shared/seo-manifest.json`.
Nothing gates `docs/changelog.html`; it rots silently.

**The docs snippet gate is the one that will surprise you.** `tests/test_docs_snippets.py:493-494`
globs `technical-docs/*.md` and AST-checks every ` ```python ` fence against the **real
installed package**. Measured behaviour:

```text
FAIL   from mmm_framework.utils.intervals import summarize     (module doesn't exist)
FAIL   from mmm_framework.utils import summarize_intervals     (attribute doesn't exist)
FAIL   mmm.summarize_intervals()                               (bound name, no such method)
pass   summarize_intervals(trace)                              (free call, unmapped name)
pass   class IntervalSummary: ...                              (class body)
pass   pl.optimize_budget(mmm, ...)                            (unmapped module alias)
```

So: write proposed APIs in ` ```text ` or ` ```diff ` fences (never collected), or as free
functions / class bodies with no `mmm_framework` import, or mark the block `# pseudocode` as
its literal first line, or `<!-- doc-snippet: skip -->` on the nearest preceding non-blank
line (`SKIP_MARKER`/`PSEUDOCODE_MARKERS` at `:43-44`). **Never** ` ```python ` plus
`from mmm_framework.<new_module> import …`. That is why every proposed signature in this
document is in a `text` fence.

A new `technical-docs/*.md` needs **no** nav, SEO, sitemap, search-index or toctree
registration — verified against all three docs gates and both builders. Its only gate is
`tests/test_docs_snippets.py::test_markdown_snippets[src-refactor.md]`.

---

## Deferred

* **Layer 2 of the temporal split** (RV builders with `CORE`/`EXTENSION` dialect records).
  Designed and sound, but a graph change with a naming contract on both sides. Needs the
  fingerprint matrix extended to NestedMMM and MultivariateMMM at all four trend families.
* **A third `model_flavor: "composed"`** serializer branch. Only needed if composition means
  a bespoke model owns arrays `_prepare_data` cannot re-derive from a panel.
* **`_conn()` resolving the DB path per call** instead of freezing at import. Both stores
  freeze today; changing it touches ~60 monkeypatching tests and is orthogonal to PR 3.
* **The `agents/tools.py` split** (7,931 lines, 108 function-local imports). Real, but an
  agent-layer concern with no bearing on the library contract; sequence it after Phase 4.
* **`reporting/helpers/roi.py::_get_contribution_samples`** stays an upward reach from
  `estimands/evaluate.py:463` — no PR here moves it (see PR 11).

## Key files

| Concern | File |
|---|---|
| The god object | `src/mmm_framework/model/base.py` |
| The extension contract | `src/mmm_framework/garden/{base,contract,compat}.py` |
| The LLM-facing authoring surface | `src/mmm_framework/agents/garden_authoring.py:71-77` |
| The parallel base class | `src/mmm_framework/mmm_extensions/models/base.py` |
| Interval helpers | `src/mmm_framework/utils/{statistics,arviz_compat}.py`, `reporting/helpers/utils.py` |
| The divisor resolver | `src/mmm_framework/reporting/helpers/measurement.py` |
| The one cycle-closing import | `src/mmm_framework/platform/runs.py:19` |
| Contract gates | `tests/test_api_contracts.py`, `tests/test_lean_imports.py`, `tests/test_docs_snippets.py` |
| Deprecation policy | `CHANGELOG.md:3-6`, `docs/api-contracts.html:160-167`, `technical-docs/roadmap.md:340-350` |

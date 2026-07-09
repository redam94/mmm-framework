# Continuous learning — agent/API/UI wiring + past-experiment ingestion (pinned contract)

> **Status:** Implementation contract, 2026-07-02. This document PINS the schemas,
> function signatures, endpoint payloads, and component plan for wiring
> `mmm_framework.continuous_learning` into the agent, REST API, and React UI —
> plus the model-free past-experiment ingestion path and the sub-channel (arms)
> extension. Implementation agents build against this document; deviations must
> be recorded here. Companion docs: `technical-docs/continuous-learning.md`
> (the engine), `assets/continous_learning.md` (the design guide).

---

## 0. What this closes

From `technical-docs/continuous-learning.md` §Deferred:

1. **Agent / API / UI wiring** — agent tools, REST endpoints + non-blocking jobs,
   sessions persistence (`learning_programs` / `learning_waves`), React
   "Learning Programs" page.
2. **Past experiments as evidence without a model** — a summary-observation
   likelihood on the CL surface (`lift = value ± SE at spend s_test vs s_base`),
   plus a converter from the experiment lifecycle registry
   (`sessions.list_experiments`) into those summaries. A team with historical
   lift tests and **no MMM and no panel** can now fit a response surface from
   readouts alone (with honest prior-domination warnings when the summaries are
   too few to identify shape).
3. **Sub-channel measurement (creative / keyword arms)** — arms-as-surface-dims
   with within-parent probe pairs and grouped budget constraints
   (`continuous_learning/arms.py`), and a nullable `subchannel` column on the
   `experiments` registry.
4. **Review fixes** — the P0/P1 items from the 2026-07-02 module review (§2).

Naming: the React page is **“Sextant”** (`/learning`) — the navigation
instrument you re-sight every wave. House naming family: Orrery, Auspices,
Chronicle, Almanac, Atelier.

---

## 1. Architecture decisions (with rationale)

* **Host-side compute, shared service.** CL is model-free (no fitted-MMM kernel
  state), NumPyro/JAX CPU, seconds-to-a-minute per fit. Both agent tools and
  REST endpoints call ONE orchestration module,
  `src/mmm_framework/continuous_learning/service.py` (precedent:
  `data_studio/service.py`). Tools run it host-side in the tool body (precedent:
  `design_experiment_plan`, `agents/tools.py:4609`); endpoints run it inside a
  bespoke job worker (precedent: `_run_validation_job`, `src/mmm_framework/api/main.py:3335` —
  NOT `_run_model_op_job`, which hard-requires a saved MMM run).
* **First-class tables, not artifacts.** A learning program is a statused,
  project-scoped, longitudinal entity feeding a UI page and re-test triggers —
  exactly the `experiments` + `run_metrics` idiom (`api/sessions.py`). Job
  status uses artifacts under a synthetic thread `__learnjobs__{project_id}`
  (precedent: `__simjobs__`/`__valjobs__`).
* **Heavy state on disk, summaries in SQLite.** Posterior draws + the
  accumulated panel go to an `.npz` + JSON sidecar under
  `<workspace_root>/projects/<project_id>/learning/<program_id>/state.npz`
  (idiom: “bytes live on disk; SQLite holds paths + metadata”). Wave rows and
  program summaries store JSON *snapshots* only (like `run_metrics`).
* **One Thompson run per plan.** `recommend`, `funding`, and `regret` previously
  re-sampled with different seeds (planner seeds 0/1/2), so the funded set and
  the reported regret referred to different consensus vectors and paid 2–3× the
  SLSQP cost. The service computes ONE `plan()` per fit (see §3.4).

---

## 2. Phase A — core library changes (`src/mmm_framework/continuous_learning/`)

### 2.1 Review fixes (all with tests)

| # | Fix | Where |
|---|---|---|
| F1 | Validate the data contract in `fit()`: `geo_idx` int, `min>=0`, `max<n_geo`, contiguous dtype; `spend`/`y` finite, row counts agree. JAX clamps out-of-bounds indices silently — a wrong `n_geo` corrupts the fit with no signal. | `model.py::fit` |
| F2 | Stop flipping global JAX config per call. `jax.config.update("jax_platform_name","cpu")` moves the whole process to CPU. Replace with a module-level, once-only guard honoring `MMM_CL_JAX_PLATFORM` (default: set `cpu` only if the flag was never touched — use a module `_PLATFORM_SET` sentinel). | `model.py` |
| F3 | `expected_regret`: actually warm-start each per-draw solve from the consensus (`allocate_under_sample(..., x0=consensus)`) in a second pass, keeping the post-hoc `max()` guarantee. Reduces the regret-biased-low → premature-stop failure. | `planner.py::expected_regret` |
| F4 | `plan_from_posterior(post, B, value, *, q, seed, mode, cap, n_starts) -> PlanResult` — NEW: one Thompson run producing `recommendation`, `allocs`, `profits`, `funding` (mroas at the recommendation), `e_regret/consensus/alloc_sd/profit_sd` (regret pass warm-started from that same consensus), so all readouts share one sample. `LearningState.plan(**kw)` delegates. Existing functions unchanged (back-compat). | `planner.py`, `loop.py` |
| F5 | `LearningState.recenter(alloc, *, min_frac=0.05)`: clamp each channel's new center to `>= min_frac * (B / K)` so the multiplicative CCD never collapses a channel to zero variation (unresurrectable channel + unidentified β). `min_frac=0` restores old behavior. Warn when clamping. | `loop.py` |
| F6 | Geo-identity guard: the data contract gains an optional `geo_ids: list[str]` key. `LearningState.ingest` stores it on first wave and raises on mismatch later (the misspec study showed re-drawn geo baselines make the loop diverge). Count-only check retained when absent. | `loop.py::ingest` |
| F7 | `knowledge_gradient(..., noise=None)` → default fantasy noise becomes `post.samples["sigma"].mean()` instead of the DGP's 0.6. | `planner.py` |
| F8 | Make `spend_ref` real: `to_scaled(dollars, spend_ref)` / `to_dollars(scaled, spend_ref)` helpers in a new `scaling.py`; `Posterior`/`LearningState` docstrings updated. The service layer (§3) works in dollars at the boundary and scaled units internally. | `scaling.py` (new) |
| F9 | `_diagnostics` no longer swallows all exceptions silently — log at debug + return Nones. | `model.py` |

Deliberately NOT fixed (documented as known limitations in
`continuous-learning.md`): float32 Laplace path fragility at large K; Python
draw-loop performance; `cuped_adjust` ddof mix (benign under the `a_geo`
absorption invariant). (The "Hill-only acquisition" limitation listed here
historically was LIFTED on 2026-07-03: `acquisition.ThetaMap` moment-matches
any registered activation in an unconstrained reparameterization, and
`observation_unit_info` supplies GLM Fisher weights for
normal/studentt/negbinomial — see `continuous-learning.md`.)

### 2.2 Summary-observation likelihood (past experiments, no panel required)

The atomic evidence unit is a **summary observation**:

```python
{
  "spend_test":  (K,) float,   # scaled spend vector during the test
  "spend_base":  (K,) float,   # scaled counterfactual/baseline spend vector
  "lift":        float,        # measured incremental KPI (natural units, total)
  "se":          float,        # standard error of that lift, same units
  "scale":       float,        # geo-weeks the lift aggregates over: n_units * n_periods
}
```

Likelihood: `lift ~ Normal(scale * (R(spend_test) − R(spend_base)), se)` —
the geo intercept cancels in the difference, so **no pre-period is required**
(same structural-stationarity caveat as the MMM's off-panel calibration).

* `model.model` gains optional kwargs `summary_test (M,K)`, `summary_base (M,K)`,
  `summary_scale (M,)`, `summary_lift (M,)`, `summary_se (M,)`; one
  `numpyro.sample("summary_obs", dist.Normal(...), obs=...)` block after the
  surface params. Works with any activation.
* `fit(data)` accepts an optional `data["summaries"]: list[dict]` (the shape
  above) and a **panel-optional** mode: if `data` has zero panel rows
  (`spend` shape `(0,K)` or absent), fit on summaries alone (`n_geo=1` dummy
  plate, no panel likelihood). At least one of panel/summaries required.
* `LearningState` gains `summaries: list[dict]` + `ingest_summaries(items)`;
  `fit` passes both through. `WaveRecord` gains `n_summaries`.
* Honesty gate: `fit` attaches `diagnostics["evidence"] =
  {"n_rows": N, "n_summaries": M}`; the service (§3) flags
  `shape_identified=False` when the evidence contains fewer than 3 distinct
  spend levels for a channel (κ/α prior-dominated → trust the funded set, not
  the curve).

### 2.3 Registry → summaries converter (`evidence.py`, new)

```python
def experiments_to_summaries(
    experiments: list[dict],       # sessions.list_experiments(...) dicts
    *,
    channels: list[str],
    spend_ref: np.ndarray,         # (K,) dollars per scaled unit
    center_scaled: np.ndarray,     # (K,) baseline allocation, scaled units
    period_days: float = 7.0,
) -> tuple[list[dict], list[dict]]   # (summaries, skipped: [{id, reason}])
```

Mapping rules (channel-granular; readout fields per
`agents/tools.py::record_experiment_readout`):

* Only `status in {"completed","calibrated"}` with `value`+`se` and a channel
  in `channels` (case-insensitive match; else skipped with reason).
* `n_units = readout.n_treated_units or len(design.treatment_geos) or 1`
  (resolved **before** the spend delta — the design branch divides by it);
  `n_periods` from the `start_date`/`end_date` span at `period_days` cadence
  (fallback `design.duration`).
* Spend delta per period **per treated unit** comes from, in order:
  `readout.spend_per_period` (already per unit, signed by the analyst) →
  `design.weekly_spend_delta / n_units` (`weekly_spend_delta` is the
  treated-cell **TOTAL**, summed over all treated geos in
  `planning/design.py`, and stored `abs()` — the sign is restored from the
  design: `design.design_type == "holdout"` or `design.design == "holdout"`
  ⇒ negative) → skipped.
* Estimand → lift (**signed** — the summary is a test-minus-base contrast,
  negative for holdouts even though the registry records effects positive):
  * `contribution`: `lift = value` for a scale-up, `lift = -value` for a
    holdout (`spend_delta < 0`); `se_lift = se`.
  * `roas`: `lift = value * total_spend_delta` (signed),
    `se_lift = se * |total_spend_delta|`,
    where `total_spend_delta = spend_delta_per_period * n_units * n_periods`.
  * `mroas`: skipped (reason `"mroas readouts are slopes, not lifts"`).
* `spend_base = center_scaled`; `spend_test = center_scaled` with the channel's
  entry shifted by `spend_delta_per_period / spend_ref[c]` (floored at 0).
* `scale = n_units * n_periods`, and `lift`/`se` are divided by nothing — the
  summary keeps the TOTAL lift with `scale` making the per-geo-week surface
  comparable. (`pred = scale * (R(test) − R(base))`.)

This is **the** model-free past-experiment bridge; the model-anchored bridge
(`apply_experiment_calibration` → in-graph MMM likelihood) is unchanged.

### 2.4 Persistence (`serialize.py`, new)

* `posterior_to_payload(post) -> dict` / `posterior_from_payload(d) -> Posterior`
  — JSON-safe: samples → lists (float32→float64), pairs → `[[i,j],...]`,
  `pair_signs` → `{"i,j": sign}`.
* `state_to_npz(state, path)` / `state_from_npz(path) -> LearningState` — the
  accumulated panel arrays, summaries, posterior samples, config, history, and
  `geo_ids` in one `.npz` (arrays) + embedded JSON string (config/history).
  Round-trip test: fit → save → load → `plan()` reproduces bit-identical
  recommendation for a fixed seed.

### 2.5 Sub-channel arms (`arms.py`, new)

```python
ARM_SEP = " │ "                    # same separator as planning/budget.py geo arms

@dataclass
class ArmSpec:
    channels: list[str]            # flattened arm names, e.g. "Search │ Brand"
    parents: list[str]             # parent per arm (== name when unsplit)
    groups: dict[str, list[int]]   # parent -> arm indices

def expand_arms(channels: list[str], arms: dict[str, list[str]]) -> ArmSpec
def within_parent_pairs(spec: ArmSpec) -> list[tuple[int,int]]
def cross_parent_pairs(spec: ArmSpec, pairs_of_parents: list[tuple[str,str]] | None = None) -> list[tuple[int,int]]
def default_arm_pair_signs(spec: ArmSpec, *, within: str = "neg", base: dict | None = None) -> dict
```

* Within-parent siblings default to `"neg"` (shared-audience substitution —
  creatives/keywords cannibalize); cross-parent pairs default `"weak"` unless
  overridden.
* Planner: `_slsqp_allocate`/`allocate_under_sample`/`thompson_wave`/
  `expected_regret`/`plan_from_posterior` gain optional
  `group_budgets: list[tuple[list[int], float]] | None` — extra SLSQP equality
  constraints `Σ_{i∈g} s_i = B_g` (parent budget fixed, mix free). Feasibility
  check: `Σ B_g <= B` and groups disjoint.
* CCD cell-count guard: `central_composite` is unchanged; the SERVICE warns
  when `n_cells > n_geo` (each cell needs ≥1 geo) and the docs state the cost:
  every extra arm ≈ 3 cells (2 axial + 1 shutoff) + **2 per probed pair**
  (a joint up- and a joint down-cell, `design.py`).

### 2.6 Tests (Phase A)

`tests/test_continuous_learning_wiring.py` (fast, no-MCMC where possible;
tiny-NUTS where needed, ≤50 draws):
validation errors (F1); summary-only fit recovers a planted β ordering from 6
synthetic summaries (tiny NUTS, slow-marked); summaries+panel joint fit runs;
`experiments_to_summaries` mapping table incl. holdout sign, roas→lift math,
skip reasons; serialize round-trips; `plan_from_posterior` consistency
(funding evaluated at the same recommendation the regret consensus used);
recenter floor; geo-identity guard; arms helpers + grouped-budget allocation
(two arms of one parent sum to the parent budget); KG noise default.

---

## 3. Phase B — service, sessions, tools, endpoints

### 3.1 Service (`src/mmm_framework/continuous_learning/service.py`)

Host-side orchestration; no sqlite access (callers own the DB); file IO under a
caller-supplied directory. All money at this boundary is **dollars per
geo-period** — center/budget/cells/recommendation are per-geo spend levels
(the engine fits geo-period rows; divide a national budget by the number of
test geos), never national totals. The service converts via `spend_ref`.

```python
PROGRAM_STATE_FILENAME = "state.npz"

def program_dir(project_id: str, program_id: str) -> Path      # <workspace_root>/projects/<pid>/learning/<prog>
def new_program_state(config: dict) -> LearningState           # validates + builds (dollars → scaled)
def load_program_state(project_id, program_id) -> LearningState
def save_program_state(project_id, program_id, state) -> str   # returns path

def design_wave(state, *, delta=0.6, probe_pairs=None, n_geo=None, n_holdout=0, seed=0) -> dict
    # {cells_scaled, cells_dollars, cell_labels, assignment?, n_cells, delta, probe_pairs, warnings[]}
def ingest_wave_rows(state, rows: list[dict], *, geo_col="geo", y_col="y") -> dict
    # rows: [{geo, y, <channel dollar spends...>}] -> panel dict appended; returns {n_rows, n_geo, warnings[]}
def import_experiment_summaries(state, experiments: list[dict]) -> dict
    # wraps evidence.experiments_to_summaries with the state's channels/spend_ref/center
    # -> {imported: n, skipped: [{id, reason}]}
def fit_and_plan(state, *, fit_kwargs=None, plan_kwargs=None, margin, population, wave_cost) -> dict
    # fit on ALL accumulated evidence, one plan_from_posterior pass, ENBS/stop -> SNAPSHOT (below)
```

**Program config dict** (stored as `config_json`; ALL dollars are **per
geo-period** — per-geo spend levels, not national totals):

```json
{
  "channels": ["Chatter", "Pulse"],
  "arms": {"Search": ["Brand", "NonBrand"]},          // optional; expands via arms.py
  "center": {"Chatter": 2800, "Pulse": 2800},         // $ per geo-period per channel/arm
  "budget": 11200,                                     // $ per geo-period
  "value_per_unit": 5.0,                               // $ per KPI unit (funding line)
  "spend_ref": 2800,                                   // $ per scaled unit — ONE global constant (default: mean of the centers)
  "mode": "fixed", "cap": null,                        // mode "free" requires spend_ref = 1 (raw-dollar spend)
  "activation": "hill", "gamma_scale": 0.8, "beta_scale": 1.0,
  "pair_signs": {"0,1": "neg"},                        // optional
  "kpi": "sales", "cadence_weeks": 2,
  "margin": 1.0, "horizon_periods": 13, "wave_cost": 25000  // ENBS in $; population = n_geos × horizon_periods at fit time
}
```

**SNAPSHOT** (returned by `fit_and_plan`; persisted on the program row and each
wave row; this is THE UI payload — pin it):

```json
{
  "schema_version": 1,
  "fitted_at": 1751470000.0,
  "evidence": {"n_rows": 1280, "n_summaries": 4, "n_waves": 2, "shape_identified": {"Chatter": true}},
  "diagnostics": {"max_rhat": 1.01, "min_ess": 350, "n_draws": 1000, "flags": []},
  "recommendation": {"Chatter": 2548.0},
  "recommendation_scaled": {"Chatter": 0.91},
  "allocation_sd": {"Chatter": 290.0},
  "funding": [{"channel": "Chatter", "mroas_mean": 1.8, "mroas_margin_adjusted": 1.8,
               "prob_above_line": 0.94, "funded": true, "verdict": "FUND"}],
  "regret": {"e_regret_kpi": 160.0, "e_regret_dollars": 49920.0, "enbs": 24920.0, "stop": false,
              "margin": 1.0, "population": 312, "wave_cost": 25000},
  "gamma": [{"pair": ["Chatter","Pulse"], "mean": -0.42, "p5": -0.7, "p95": -0.1, "sign": "neg", "prior_dominated": false}],
  "response_curves": {"Chatter": {"spend_dollars": [...], "mean": [...], "lo": [...], "hi": [...], "current": 2800.0}},
  "warnings": ["..."]
}
```

Verdicts: `FUND` if `prob_above_line > 0.65`, `CUT` if `< 0.35`, else `HOLD`
(story-notebook convention); `mroas_margin_adjusted = margin × mroas_mean`
(the profit-dollar read alongside the value-dollar `mroas_mean`).
`e_regret_kpi` is per-geo-period profit already in value-dollars;
`population` is **geo-periods** — `n_geos × horizon_periods`, computed at fit
time once the geo set is pinned (falls back to `horizon_periods` alone for
summaries-only programs; 312 = 24 geos × 13 periods in the example).
`e_regret_dollars = e_regret_kpi * margin * population`;
`enbs = e_regret_dollars − wave_cost`.
Response curves: 25-point grid per channel over `[0, 2×center]` via
`planner.response_grid`, 90% band, computed on ≤200 thinned draws.

### 3.2 Sessions tables (`api/sessions.py::init_db`, idiom: experiments/run_metrics)

```sql
CREATE TABLE IF NOT EXISTS learning_programs (
  id TEXT PRIMARY KEY, project_id TEXT, thread_id TEXT,
  name TEXT, status TEXT NOT NULL DEFAULT 'active',          -- active|stopped|archived
  channels_json TEXT NOT NULL, config_json TEXT NOT NULL,
  state_path TEXT, summary_json TEXT,
  created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_learning_programs_project ON learning_programs(project_id, updated_at);

CREATE TABLE IF NOT EXISTS learning_waves (
  id TEXT PRIMARY KEY, program_id TEXT NOT NULL, project_id TEXT,
  wave_index INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'designed', -- designed|ingested
  source TEXT,                                                -- wave|experiment_import|manual
  design_json TEXT, observations_json TEXT, snapshot_json TEXT,
  experiment_ids_json TEXT, created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_learning_waves_program ON learning_waves(program_id, wave_index);
```

Store functions (mirror experiments'): `create_learning_program`,
`get_learning_program`, `list_learning_programs(project_id, status=None)`,
`update_learning_program(program_id, **fields)` (status validated against
`{'active','stopped','archived'}`), `delete_learning_program` (cascades waves),
`add_learning_wave`, `list_learning_waves(program_id)`,
`update_learning_wave(wave_id, **fields)`. Row→dict adapters parse the JSON
columns (`_learning_program_row_to_dict`, guards for pre-migration rows).

### 3.3 `subchannel` on experiments

* `ALTER TABLE experiments ADD COLUMN subchannel TEXT` in the existing
  migration loop (sessions.py:188-200 pattern).
* Passthrough: `upsert_experiment(subchannel=None)`, `_experiment_row_to_dict`,
  `list_experiments(subchannel=None)` filter.
* Agent tools `log_experiment` / `plan_experiment` /
  `record_experiment_readout` gain optional `subchannel: str | None` arg
  (docstring: creative/keyword/campaign identifier; calibration to an MMM stays
  channel-level — sub-channel readouts feed **learning programs with arms**, or
  a breakout-model share likelihood, future).
* `POST /experiments` request schema gains `subchannel`.
* `evidence.experiments_to_summaries` matches a subchannel readout to an arm
  named `f"{channel}{ARM_SEP}{subchannel}"` when the program has arms;
  channel-level readouts on a split parent are skipped with reason
  `"channel-level readout on a split parent"` (a total-lift constraint across
  an arm group is future work).

### 3.4 Agent tools (`src/mmm_framework/agents/learning_tools.py`, new module → `LEARNING_TOOLS` list appended in `tools.py::TOOLS`)

All follow the canonical recipe (`_activate_thread` → lazy imports → work →
`publish_tables`/`_publish_figures` → `Command`). Project id via
`sessions_store.get_session(tid)["project_id"]`.

1. `start_learning_program(name, channels, budget_per_period, value_per_unit, center=None, arms=None, activation="hill", kpi=None, wave_cost=None, horizon_periods=13)` —
   creates the sessions row + state file; returns a markdown summary + a
   `dashboard_data["learning_program"]` payload. `budget_per_period`/`center`
   are **$ per geo-period** (divide a national budget by the number of test
   geos); `horizon_periods` feeds `population = n_geos × horizon_periods` at
   fit time.
2. `import_past_experiments(program_id=None, experiment_ids=None)` — pulls
   completed/calibrated registry readouts (`list_experiments`), converts via
   the service, refits, transitions nothing (registry untouched), stores an
   `experiment_import` wave row + snapshot; reports imported/skipped with
   reasons. **This is the “leverage past experiments without a model” tool.**
3. `design_learning_wave(program_id=None, delta=0.6, probe_pairs="auto", n_geo=None, n_holdout=0)` —
   returns the CCD cells in dollars (table) + assignment; stores a `designed`
   wave row.
4. `record_learning_wave(program_id=None, csv_path=None, rows=None)` — ingest a
   geo,week,spend,y panel (CSV from the workspace or inline rows), refit,
   snapshot; publishes the funding-line figure + allocation table.
5. `get_learning_program_status(program_id=None)` — snapshot → markdown +
   funding table + response-curve/synergy figures.
6. `check_learning_stopping(program_id=None, margin=None, population=None, wave_cost=None)` —
   re-evaluates ENBS with overridden economics; recommends stop/continue; on
   stop, `update_learning_program(status="stopped")` only when the user
   confirms (tool arg `confirm_stop=False`).

Guidance: a “Continuous learning (no model required)” subsection in
`prompts.py::MMM_SYSTEM_PROMPT` Step-9 region + a `## Continuous learning`
entry in `_LIBRARY_MENU`. Tool-name lists: all six are **spine** tools (work
without media-channel spec), none heavy.

### 3.5 REST endpoints (`src/mmm_framework/api/main.py`; all `dependencies=[_proj_read]` reads / `[_proj_write]` writes; 404-check project; `safe_json_dumps_load` responses)

```
GET    /projects/{pid}/learning-programs                     → {programs: [{...row, summary}]}
POST   /projects/{pid}/learning-programs                     → create (LearningProgramCreateRequest)
GET    /projects/{pid}/learning-programs/{prog}              → {program, waves: [...]}
DELETE /projects/{pid}/learning-programs/{prog}
POST   /projects/{pid}/learning-programs/{prog}/design-wave  → sync; body {delta, probe_pairs?, n_geo?, n_holdout?}
POST   /projects/{pid}/learning-programs/{prog}/waves        → ingest; body {rows?: [...], experiment_ids?: [...], csv_text?: str}
                                                               → 202 {job_id} (fit job spawned)
POST   /projects/{pid}/learning-programs/{prog}/fit          → 202 {job_id} (refit with current evidence; body {margin?, population?, wave_cost?})
GET    /projects/{pid}/learning-programs/{prog}/jobs/{job_id}→ poll artifact {status, result: SNAPSHOT, error}
```

Job worker `_run_learning_job` (copy `_run_validation_job` shape): synthetic
tid `__learnjobs__{pid}`, artifact kind `learning_fit`, ONE
`asyncio.to_thread` wrapping `set_current_thread` + service calls + sessions
writes; patches status via `_sim_job_patch`. No `latest_model_run_payload` —
model-free.

### 3.6 Tests (Phase B)

`tests/test_learning_sessions.py` (store CRUD + status validation + cascade),
`tests/test_learning_endpoints.py` (httpx TestClient: create → design →
ingest tiny wave → poll job to done (tiny fit kwargs via request override
`fit_kwargs: {num_warmup: 30, num_samples: 30, num_chains: 1}` — the request
schema accepts it for tests) → snapshot shape; import-experiments path with a
seeded registry row; subchannel passthrough), `tests/test_learning_tools.py`
(tool bodies with a stubbed thread/session).

---

## 4. Phase C — React “Sextant” page

* Route `/learning`, `PAGES` entry `{path:'/learning', name:'Sextant', hint:'Continuous learning programs'}`, `NAV_ICONS` entry.
* `api/services/learningService.ts` — typed mirror of §3.5 (types:
  `LearningProgram`, `LearningWave`, `LearningSnapshot`, `FundingRow`,
  `DesignWavePayload`, `LearningJob`); `api/hooks/useLearning.ts`
  (`learningKeys`, `useLearningPrograms`, `useLearningProgram`,
  `useCreateProgram`, `useDesignWave`, `useIngestWave` +
  `useLearningFitJob` polling à la `useExperimentSimulation`).
* Components (`pages/Learning/`):
  * `index.tsx` — program list / empty state / program detail.
  * `ProgramCreateWizard.tsx` — channels+arms, budget/center, value-per-unit,
    economics (margin/horizon/wave cost); modeled on `ProjectOnboardingWizard`.
  * `WaveTimeline.tsx` — per-wave cards (design → ingested → snapshot deltas);
    base: `RunsTimeline`.
  * `FundingLineChart.tsx` — per-channel P(mROAS>1) bars + verdict chips, and
    the response-curve-with-band panel (`ResponseCurvesPanel` idiom) with the
    funding threshold line.
  * `SynergyHeatmap.tsx` — FIRST Plotly heatmap in the app: γ matrix, diverging
    colorscale anchored in `COLORS` (rust↔sage), prior-dominated cells hatched
    via text annotation `†`.
  * `EnbsCard.tsx` — stop/continue verdict: E[regret]$ vs wave cost, `StatHero`
    + verdict chip.
  * `DesignWaveStudio.tsx` — drawer: delta/probe/holdout inputs → design table
    (cells in dollars) → “Record results” CSV paste → ingest+fit job progress.
  * `ImportExperimentsPanel.tsx` — lists completed/calibrated registry
    experiments (reuse `useExperimentRegistry`), pick → POST
    `{experiment_ids}` → job → imported/skipped report.
* Experiments page: show a `subchannel` chip on `LifecycleBoard` cards and the
  drawer when present; `LogExperimentModal` gains the optional field.
* Tests: vitest specs for `learningService` (payload mapping), one hook
  (polling stop condition), `EnbsCard` + `FundingLineChart` render; `tsc -b` +
  eslint clean.

---

## 5. Phase D — docs

1. **`docs/continuous-learning.html`** — add: the misspecification study section
   (embed `continuous-learning-misspec.gif` + `cl-misspec.png`, copied from
   `nbs/artifacts/continuous_learning_misspecification.gif` and
   `continuous_learning_misspec.png`); a “Carry-over and noise” subsection
   (adstock pre-pass + CUPED, Tier-2 framing); “Start from the tests you've
   already run” section (past-experiment ingestion — the summary likelihood in
   plain language + the `import_past_experiments` agent flow); “Creative and
   keyword arms” section (sub-channel); “In the app” section (Sextant page +
   agent tools); direct links to both notebooks; embed or delete the stray
   `continuous-learning-acquisition.png` (embed in §cycle as the static
   fallback). **Interactive example #1**: “Play with the surface” — sliders
   (β₁, β₂, γ, κ, α, budget) driving a two-panel Plotly (response curves +
   profit vs split with funded/unfunded shading), the house
   `math-02-saturation.html` pattern (`plotly-2.27.0.min.js`, `.controls-row`,
   `Plotly.react`). **Interactive example #2**: ENBS stop/continue calculator
   (E[regret], margin, horizon, wave cost sliders → verdict + bar).
2. **`docs/continuous-learning-math.html`** — add: the summary-observation
   likelihood (equation + “geo intercept cancels ⇒ no pre-period” note); the
   `hill_mixture` fittable activation + `make_world_*` constructors; a note
   that the Laplace/EIG acquisition covers any registered activation +
   normal/studentt/negbinomial (unconstrained-space `ThetaMap` + GLM Fisher
   weights, 2026-07-03 — it was Hill/Gaussian-only before, and still falls
   back loudly for unknown families); preprocess/design/dgp in the closing
   code map; the stable-geo-set gotcha stated as a rule; grouped-budget arms
   formulation (`Σ_{i∈g} s_i = B_g`); fix the `mu, sigma = theta_moments(...)`
   naming collision in the snippet.
3. **Links/nav**: reciprocal link from `measurement-calibration.html`; entries
   on `demos.html` (both CL notebooks) and a mention on `index.html`; update
   `sitemap.xml` lastmod for both pages.
4. **Snippet gate** (`tests/test_docs_snippets.py`): add
   `"post" → continuous_learning.model.Posterior` and
   `"state" → continuous_learning.loop.LearningState` to `CONDITIONAL_BINDINGS`
   (conditional on `continuous_learning` appearing in the snippet, to avoid
   colliding with other pages' `post`/`state` vars), and a `TRUSTED_PRODUCERS`
   entry for `cl.fit`/`fit`. Keep all pages green.
5. **`technical-docs/subchannel-measurement.md`** — the design doc: today's
   three disconnected capabilities (breakout model / MFF dims / channel-keyed
   experiments), the chosen CL arms formulation with cell-count economics, the
   `subchannel` registry column, MFF guidance (creative/keyword rides
   `VariableName`; `Creative`/`Campaign` columns reserved), and the deferred
   items (share-based breakout calibration likelihood, per-arm cost
   descriptors, total-lift constraints across an arm group).

---

## 6. Recorded deviations (implementation, 2026-07-02)

* **Holdout detection** (`evidence._is_holdout`): the literal
  `design_type == "holdout"` check is a subset of reality — real design
  snapshots encode `design_type = f"{method} — holdout"` and
  `intensity_pct = -100`. Implemented as the superset: literal checks OR
  substring `"holdout"` in `design_type`/`design_key` OR `intensity_pct < 0`.
* **Missing test window**: when both readout dates and `design.duration` are
  absent the experiment is **skipped** (reason
  `"no test window: start/end dates or design.duration required"`) rather than
  defaulting `n_periods=1`.
* **Summaries-only fits skip the geo plate entirely** — no `a_geo`/`sigma`/`A`
  sites (rather than a dummy `n_geo=1` plate). `knowledge_gradient` falls back
  to `noise=0.6` for such posteriors.
* **Create request body** is `{name, config}` (config = §3.1 dict); channels
  ride inside `config`.
* **Import job report**: `imported`/`skipped` ride at the top level of the job
  `result`, alongside the SNAPSHOT keys.
* **Funding line is per-dollar**: `plan_from_posterior` yields mROAS per
  *scaled* unit; the service divides the draw matrix by `spend_ref[c]` before
  the break-even-at-1.0 test so verdicts compare dollars-to-dollars.
* **ENBS conversion**: `e_regret_dollars = e_regret_kpi × margin × population`
  where `e_regret_kpi` is per-geo-period profit already in value-dollars (in
  fixed mode the spend term cancels in the regret difference).
* **Group budgets**: a split parent's scaled budget is
  `Σ arms' scaled center × (B_scaled / Σ center_scaled)` — equal to the pinned
  rule when `budget == Σ center` (the wizard's case), jointly feasible
  otherwise. Heterogeneous `spend_ref` makes the fixed-budget simplex
  reference-weighted (snapshot warning; exact when `spend_ref = center`).
* Job payloads carry an extra `program_id`; `design-wave` responses include
  `wave_id`; pure `/fit` refits update the program summary without adding a
  wave row.
* `state_to_npz` returns the written path (str).

**Post-review corrections (2026-07-02)** — the adversarial review of the
shipped wiring surfaced units/economics errors in the contract itself; the
code implements the corrected semantics below, and §2.3/§3.1/the SNAPSHOT
example have been updated in place to match:

* **Signed roas lift** — §2.3 originally pinned `lift = value × |total_spend_delta|`,
  which inverted every holdout's evidence (the summary likelihood predicts
  `scale × (R(test) − R(base))`, negative for a holdout). Corrected:
  `lift = value × total_spend_delta` (signed, negative for holdouts);
  `se_lift = se × |total_spend_delta|` stays absolute.
* **Contribution lift sign flips for holdouts** — the registry records a
  holdout's measured contribution positive, but the test-vs-base contrast is
  negative: `lift = -value` when `spend_delta < 0`; `se_lift = se` unchanged.
* **`design.weekly_spend_delta` is the treated-cell TOTAL** (summed over all
  treated geos in `planning/design.py`), not per-treated-unit; the converter
  divides it by `n_units` (resolved first) before using it as the per-unit
  per-period delta. `readout.spend_per_period` was already per unit.
* **All dollars at the service boundary are per geo-period** — center,
  budget, design cells, and the recommendation are per-geo spend levels, not
  national totals (§3.1's 140k/560k example was restated as 2,800/11,200
  per geo-period).
* **`horizon_periods` replaces a pre-baked `population`** — the config stores
  the horizon; `population = n_geos × horizon_periods` is computed at fit
  time once the geo set is pinned (falls back to the horizon alone for
  summaries-only programs). Regret is a per-geo-period quantity and every geo
  runs the learned allocation, so horizon-only populations understated VOI
  ~n_geo× and stopped the loop prematurely.
* **Default `spend_ref` is ONE global constant** (the mean of the channel
  centers), not per-channel centers — a uniform reference keeps the scaled
  budget simplex and grouped arm budgets dollar-exact.
* **Funding rows gain `mroas_margin_adjusted`** (`= margin × mroas_mean`) —
  the profit-dollar funding read alongside the value-dollar `mroas_mean`, so
  FUND verdicts are not issued at revenue break-even when `margin ≠ 1`.
* **`mode: "free"` requires `spend_ref = 1`** — the free-mode objective prices
  spend in scaled units, so a non-unit reference understates the marginal
  cost of a dollar by ~spend_ref×; `new_program_state` rejects the
  combination (use fixed mode, or spend in raw dollars with `spend_ref: 1`).
* **Prior scaling for natural-unit KPIs** — `fit(prior_scaling="auto")`
  (default) scales the intercept/noise/β/γ prior scales by
  `y_scale = 10^round(log10(std(y)))` (decade-quantized, `mean(y)` centers `A`;
  summaries-only fits derive the decade from `|lift|/scale`). Decade
  quantization rather than raw `std(y)` is deliberate and measured: raw std
  (1.22 on the synthetic worlds) widened the β/γ priors enough to push a
  marginally-identified direction into non-identification (recovery-gate R̂
  1.04→1.32, ESS 60→2.8), while `10^k` is exactly 1.0 for O(1) worlds
  (byte-identical priors) and exactly right at revenue scales.
  `prior_scaling="unit"` reproduces the original graph.
* **Import idempotency** — repeat `import_past_experiments` calls skip
  already-imported experiment ids (summaries carry `experiment_id`
  provenance); re-imports report as skipped with reason "already imported".
* **Job/state robustness** — `save_program_state` writes atomically
  (tmp + `os.replace`); missing/corrupt state raises `ProgramStateError`
  (endpoints → 409-with-message); a per-program `threading.Lock` serializes
  the load→ingest→fit→save critical section in the worker and both tool fit
  paths; `/waves`, `/fit`, and DELETE 409 while a fit job is pending/running
  (artifact scan with a 6h staleness cutoff); rows+csv_text in one ingest →
  400; imported experiment ids are project-scoped; deleting a program reaps
  its state directory.
* **One wave row per real wave** — a rows/CSV ingest resolves the program's
  latest still-`designed` wave row to `ingested` (design preserved) instead of
  appending a duplicate; experiment imports always append; pure refits add no
  row and don't increment `evidence.n_waves`.

## 7. Out of scope (recorded as deferred)

* Vectorized Thompson solves.
* Postgres migration (WS-4) — new tables use the existing SQLite idiom.
* Automatic experiment-registry → learning-program sync (import is explicit).

Shipped in the follow-up hardening/features pass (same branch, after this
contract's phases): Laplace KG inside the closed loop
(`loop.select_next_design` + `design_wave(optimize=…)`), NegBinomial
likelihood (`fit(likelihood="negbinomial")`), national time effect τ_t
(`fit(time_effect="national")` + `period_idx` plumbing), stratified geo
assignment (`assign_geos(baseline=…)` + `design_wave(stratify=…)`), the
breakout-model share likelihood (`calibration.likelihood.ShareMeasurement` +
`arms.arm_shares`), and the experiment-registry hardening (audit events via
`append_experiment_event`, state-machine enforcement in `upsert_experiment`,
`record_experiment_readout(overwrite_calibrated=…)`,
`apply_experiment_calibration` merge-by-default + `replace=`). See
`technical-docs/continuous-learning.md` and
`technical-docs/subchannel-measurement.md` for the feature docs.

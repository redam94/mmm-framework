# Phase 2 Scoping — Fits + Model Tools in the Kernel

**Parent design:** `agent-session-kernels.md` (v2) §3.4/§3.5/§6. **Builds on:** Phase 1
(`-phase1.md`, COMPLETE — `KernelManager` + `InProcessKernel`/`SubprocessKernel` behind
`MMM_AGENT_KERNEL`). **Goal:** move model **fits** and the **~14 model-reading tools** to run
where the model lives, so `mmm`/`results` are available under `subprocess` and the documented
Phase-1 boundary is removed. **Hard invariant (every PR):** default `inprocess` behavior is
provably unchanged.

> Grounded by a 4-region risk-scan of every `_MODEL_CACHE` consumer. The scan **confirmed the
> uniform model** (collapses §3.5's 3-way classification) and surfaced the serialization,
> stdout, file-path, and two latent-bug details below.

---

## 1. The uniform model (the core decision)

Every model tool has the **same shape**: guard (model present?) → a model-touching **compute**
that reads `model._trace.posterior` (and sometimes re-samples live PyMC) → **extract** scalars +
build markdown → return a `Command` that merges a JSON dict into `dashboard_data`.

So routing is uniform: **run the model-touching compute where the model lives.** "Reload-from-
disk" was never a routing requirement — it's only an optimization, and it's *impossible* for the
live-PyMC tools (you can't `sample_prior_predictive` on a deserialized trace). Drop the 3-way
split. The seam:

```
agents/model_ops.py   # NEW: relocatable, framework-only, NO langchain/state
  def roi_metrics(mmm, results, **kw) -> {"content": str, "dashboard": {...}, "error": str|None}
  def model_diagnostics(mmm, results, **kw) -> {...}
  ...  # one per model tool; each does the compute AND the scalar/markdown extraction

Kernel gains:
  run_model_op(op_name: str, kwargs: dict) -> dict        # the model-touching half
  fit(model_spec, dataset_path) -> dict                   # build_and_fit, relocated
```

- **`InProcessKernel.run_model_op`** calls `model_ops.<op>(MODEL_CACHE['fitted_model'],
  MODEL_CACHE['fit_results'], **kwargs)` directly → today's behavior, zero change.
- **`SubprocessKernel.run_model_op`** runs the op **in the kernel** on its in-kernel
  `mmm`/`results` and returns the dict over the **result channel** (§2).

**The kernel/host split (from the scan):** the **kernel** does only the model-touching compute
→ a JSON projection `{content, dashboard, error}`. The **API-process tool wrapper** keeps
everything stateful: `_activate_thread`, the `dashboard_data` read-merge, **`record_assumption`**
(causal tools write to the session store — must NOT happen in the kernel), and building the
`Command`. None of the computes need `InjectedState` beyond `dashboard_data` or `InjectedConfig`
beyond thread context.

## 2. Result channel — display_data MIME, NOT stdout (load-bearing)

The scan **confirmed a stdout-scrape would be corrupted**, and worse than expected: PR3 merged
`sys.stderr → sys.stdout`, and the computes are noisy — `MMMSerializer.save/load` do bare
`print()` to stdout (`serialization.py:100,222`), and the live-PyMC tools emit PyMC/rich
**progress bars** to stdout/stderr (`pm.sample_posterior_predictive`/`sample_prior_predictive`).
So:

- The op publishes its result over a **dedicated MIME**:
  `publish_display_data({"application/vnd.mmm-modelop+json": result})` — exactly parallel to the
  existing `application/vnd.plotly.v1+json` plot capture. The `_run` loop gains one capture
  branch; `run_model_op` collects that payload and ignores stdout (stdout/warnings still flow to
  the normal text buffer, harmlessly).
- **NaN/Inf sanitize at the boundary.** HDI can return `np.nan`; `json.dumps` emits bare `NaN`
  (invalid strict JSON / rejected by JS `JSON.parse`). Reuse the existing
  `src/mmm_framework/api/main.py:safe_json_dumps` logic (NaN/Inf → `None`, numpy scalars → python) in the op-result
  encoder, both impls. (np-scalar leakage is mostly already avoided by the tools' `float()` casts,
  but don't rely on it — sanitize.)
- **Projection after extraction.** The *raw* computes return DataFrames / ndarrays /
  `InferenceData` / dataclasses-with-ndarrays — **none cross cleanly**. The op function must do
  the scalar/markdown extraction (the existing per-tool `*_json` dict building) and return only
  that. The serialization boundary sits **after** extraction, inside `model_ops.<op>`.

## 3. Migration table (the ~14 sites)

| Tool | Class | Model-touch | Notes for relocation |
|---|---|---|---|
| `get_roi_metrics` | A compute | posterior + `az.hdi` | NaN risk from empty-sample HDI |
| `get_component_decomposition` | A compute | `model.compute_component_decomposition()` (posterior) | extract `component/total/pct` before serialize |
| `get_model_diagnostics` | A compute | `az.summary` (r_hat/ess) | needs `arviz` in kernel env (already a dep) |
| `get_adstock_weights` | A compute | posterior alpha | drop `decay_weights` ndarray; keep scalars |
| `get_saturation_curves` | A compute | posterior, `np.random.choice` | **seed it** — non-deterministic subsample |
| `leave_one_out_decomposition` (causal) | A compute | decomposition (posterior) | `record_assumption` stays API-side |
| `prior_predictive_check` (causal) | **B live-PyMC** | `mmm.sample_prior_predictive` | **cannot** reload-from-disk; in-kernel mandatory; reduce idata→summary in kernel; `record_assumption` API-side |
| `run_budget_scenario` | **B live-PyMC** | `what_if_scenario` → 2× `predict` | raw dict has numpy arrays → `.tolist()`/`default=str`; markdown-only return |
| `run_marginal_analysis` | **B live-PyMC** | `compute_marginal_contributions` → N+1 `predict` | return `df.to_dict('records')` not truncated `to_string()`; slow |
| `generate_client_report` | **B + C file** | report build runs `predict` (live PPC) | writes HTML; host-shared volume |
| `generate_client_slides` | C file | saturation + mROI (posterior only, no live PyMC) | writes HTML; clean JSON enrichment |
| `save_fitted_model` | C file | serialize to disk | **LATENT BUG** (§5); host-shared volume |
| `load_fitted_model` | C file + mutate | `MMMSerializer.load` (rebuilds live PyMC) | **LATENT BUG** (§5); needs session **dataset** in kernel (§6) |
| `get_session_status` | D trivial | "model present?" boolean | lightweight kernel ping, not full relocation |
| `list_saved_models` | — (no model) | lists `mmm_models/` dir | not a model op; shares the shared-volume need (§4) |

`generate_project_report` reads **only** `dashboard_data` (no `_MODEL_CACHE`) → **not** a model
tool, no migration.

## 4. Files & the host-shared volume

File-writers use **relative** paths resolved vs process cwd: `agent_client_report.html`,
`agent_client_slides.html`, and `_MODELS_DIR='mmm_models'`/`_CONFIGS_DIR='mmm_configs'`
(module-level relative, **not** rebound per-thread). Under `subprocess`:

- The kernel's cwd **is** `work_dir` (PR3 re-chdir), so a bare `report.to_html('agent_client_
  report.html')` lands in `work_dir/` — **host-visible** (the workspace mount). Good — but the
  returned path key in `dashboard_data` must resolve there for the Artifacts/download tab.
- **`mmm_models/` is the problem:** the kernel writes `work_dir/mmm_models/...` while the API's
  `save_config`/`list_saved_models`/download resolve `mmm_models/` vs the **API** cwd — divergent.
  **Fix:** make the model/config dirs **workspace-resolved** (absolute, under a shared root) so
  kernel-written models are listable/loadable/downloadable by the API. This also fixes the
  in-process case (today they're CWD-relative).

## 5. Two latent bugs to fix in passing (the scan found these — they fail *today*)

- **`save_fitted_model`** calls `MMMSerializer().save(fitted, results, save_dir)` but the
  signature is `save(model, path, save_trace, compress)` — `results` mis-binds to `path`. It
  **always** returns "Save failed". Relocation must **fix the call** (drop `results`; pass
  `save_dir` as `path`), not just move it.
- **`load_fitted_model`** calls `load(save_dir)` and unpacks a 2-tuple, but `load(path, panel,
  rebuild_model=True)` **requires the session `PanelDataset`** to rebuild the model and returns a
  **single** instance. It **always** returns "Load failed". Relocation must supply the panel from
  the session dataset (§6) and fix the unpacking.

## 6. Cold-kernel model reload (the production gap)

PR3's `MMM_MAX_KERNELS` LRU cap and the death/respawn path can drop a kernel **mid-session** →
respawned **cold** (no `mmm`/`results`). A model op (or `execute_python` cell) then finds no
model — even though one was fit and is on disk via `MMMSerializer`. This is the model analogue of
Phase 1's reconstitution decision (reload from disk, don't replay the fit). So:

- `run_model_op`/`fit-dependent` paths **rehydrate** `mmm`/`results` from the latest saved
  `model_path` (the most-recent `model_run` artifact) **before** running, if the kernel is cold.
- This needs the model dir on the shared volume (§4) **and** the session **dataset** in the
  kernel (the serializer's `load` rebuilds the live PyMC graph from the panel — `load_fitted_
  model`'s latent bug exposed this). The dataset is already staged in `work_dir`/`dataset_path`.
- **Without this, the exit criterion "fit→interpret round-trips" passes warm-kernel tests and
  fails in production after any eviction** — name it now.

## 7. Fits in the kernel

`fit_mmm_model`'s `build_and_fit` body (build from spec + dataset → `mmm.fit` → report → serialize
→ `model_run` record) becomes `kernel.fit(model_spec, dataset_path)`:
- `InProcessKernel.fit` runs it locally, deposits `mmm`/`results` in `MODEL_CACHE` (unchanged).
- `SubprocessKernel.fit` runs it **in the kernel**; `mmm`/`results` become kernel globals; the
  serialized `model_path` lands on the shared volume; returns `{summary, model_run}` over the
  result channel.
- **Progress side-channel** (Phase 1 decision §7.6): the kernel emits "fitting n/N" into a
  per-thread `asyncio.Queue` that `event_generator` drains as an ephemeral SSE event; the final
  result is the single `Command`.
- **Stop = interrupt → kill** (Phase 1 wall-clock path already built); note NumPyro/JAX ignores
  SIGINT so a stopped fit is kill+respawn (loses the run — no mid-fit checkpoint).

## 8. `MODEL_CACHE` doesn't retire — it's `InProcessKernel`'s store

Keep `MODEL_CACHE` as the in-process backing store. "Retirement" means the ~14 sites **stop
reading `_MODEL_CACHE` directly** and go through `kernel.run_model_op`/`kernel.fit`. For
`inprocess` it's the same object → zero behavior change. The causal-tools lazy
`from tools import _MODEL_CACHE` circular-import workaround moves into `model_ops`/the kernel.

## 9. PR breakdown (mirror PR1–4 discipline)

- **PR-A — extract `model_ops` registry (pure refactor).** ✅ DONE (`4fd0ed0`): the 5
  interpretation ops (roi/decomposition/diagnostics/adstock/saturation) extracted into
  `agents/model_ops.py` (`op(mmm, results, **kw) -> {content, dashboard, error}`); tools dispatch
  **directly** (in-process) via `_modelop_command`; saturation RNG pinned; **`save_fitted_model`
  bug fixed** + regression test. **Deferred:** the live-PyMC / report / causal ops extract with
  their PRs (B–D); the **`load_fitted_model` bug fix moves to PR-C** (it needs panel
  reconstruction from the session dataset — the same mechanism PR-C builds). Full fast suite green.
- **PR-B — `run_model_op` dispatch + result channel.** ✅ DONE (`ad0482b`): `Kernel.run_model_op`
  (InProcess calls the op on the MODEL_CACHE model; Subprocess runs it in-kernel and returns it
  over the `application/vnd.mmm-modelop+json` display_data MIME — **off stdout**, captured by
  `_run`); `_json_safe` NaN/Inf→None + numpy→python; the 5 interpretation tools dispatch via
  `_KERNELS.get_or_spawn(tid).run_model_op(...)`; no-model/unknown-op return as the result `error`.
  Tested: channel round-trip (op runs in-kernel, dict transported). **Note:** the *full real-model*
  round-trip through `subprocess` lands in PR-C — until fits move into the kernel, subprocess has
  no model so `run_model_op` returns no-model (the Phase-1 boundary, by design).
- **PR-C — fit in the kernel + cold reload + shared volume.** Split:
  - **PR-C.1 ✅ DONE (`7eba59b`): persistence + reload actually work.** Fixed the remaining two
    serializer bugs — `fit_mmm_model` auto-save (`save(mmm, model_path)`; it never persisted
    before, so cold-reload had nothing) and `load_fitted_model` (rebuild the panel +
    `load(save_dir, panel)`). Extracted `_mff_config_from_spec` shared by fit + load. Slow
    end-to-end fit→auto-save→load→interpret round-trip test proves all three latent bugs fixed.
  - **PR-C.2 ✅ DONE (`416ae94`): fit in the kernel → the boundary is removed.** `build_and_fit`
    extracted into `agents/fitting.py` (relocatable; `_build_prior`/`_mff_config_from_spec` moved
    too); `Kernel.fit` — InProcess fits + stores in MODEL_CACHE (unchanged); Subprocess fits IN
    the kernel via the `_mmm_emit_fit` startup driver so `mmm`/`results` become kernel **globals**
    (run_model_op/execute_python then see the model). Slow test: fit IN a real subprocess kernel →
    interpret the in-kernel model. (This already achieves the "post-fit mmm works in subprocess"
    flip the old PR-D bullet named.)
  - **PR-C.3 ✅ DONE (`1f3b496`): cold-kernel reload-from-disk.** Each `SubprocessKernel` gets its
    `thread_id` (so fit/run_model_op/execute share one work_dir = the session's thread_dir);
    `run_model_op` spawns a cold kernel and the `_mmm_rehydrate` startup driver reloads the latest
    `<work_dir>/mmm_models/run_*` model (panel rebuilt from the spec+dataset persisted in
    `run_metadata.json`). **Also fixed a real serializer bug:** `_load_trace` deleted the temp
    `.nc` before the (lazy) `az.from_netcdf` materialized → "Unable to open file" on reload; now
    materializes first. Slow test: fit→evict (`MMM_MAX_KERNELS=1`, 2nd session)→cold respawn
    rehydrates→interpret works. **Deferred:** in-process MODEL_CACHE-eviction cold-reload
    (symmetry); fit-progress `asyncio.Queue`; unifying explicit `save_fitted_model`/list dirs with
    the per-session location.
- **PR-D — reports/save/load/causal.** (The "remove the boundary + flip the test" goal was already
  achieved in PR-C.2: post-fit `mmm` works in a subprocess cell, proven by the fit→interpret test.)
  - **PR-D.1 ✅ DONE (`8f7a2ff`): the analysis tools.** `run_budget_scenario` / `run_marginal_analysis`
    (live-PyMC) migrated to `model_ops` ops dispatched via `run_model_op` — they now run where the
    model lives; input validation stays host-side; progress-bar stdout no longer matters (result on
    the MIME channel).
  - **PR-D.2 ✅ DONE (`92b07de`): the causal tools.** `prior_predictive_check` (live PyMC) /
    `leave_one_out_decomposition` → `model_ops` ops returning `{content, dashboard, error}` + an
    `assumption` payload; the wrappers dispatch via `run_model_op` and keep `record_assumption`
    **host-side** (never in the kernel). OPS now has 9 ops.
  - **PR-D.3 ✅ DONE (`af868db`): `save_fitted_model` runs in the kernel** (`model_ops.save_model`
    → in-process: API-cwd/mmm_models; subprocess: the in-kernel model → work_dir/mmm_models).
    `get_session_status` already works in subprocess (reports fitted on `model_status=="completed"`)
    — no change. OPS now 10 ops.
  - **PR-D.4 (remaining — needs API-side work, NOT a clean model_ops move):** the **file-writing**
    report tools `generate_client_report`/`generate_client_slides` write relative-cwd HTML that the
    `/report*` endpoints read from the API cwd — a subprocess kernel writes to `work_dir`, which
    those endpoints wouldn't find, so this needs **coordinated per-session path resolution in the API
    endpoints**. Also: subprocess `load_fitted_model`-into-the-kernel + `list_saved_models` dir
    unification. Recommend a foundational "workspace-resolve report/model/config paths + serve them
    per-session" PR before migrating these.

## 10. Risks

- **Default-path regressions** — PR-A moves ~14 tool bodies; keep it a mechanical extraction
  reviewed against the suite (the PR2 lesson).
- **Serialization edges** — NaN/Inf, numpy scalars, DataFrames; the encoder + projection-after-
  extraction handle them, but every op needs a JSON-round-trip test.
- **Live-PyMC cost** — report/budget/marginal re-sample on every call (slow, non-deterministic
  without a seed); document and seed where feasible.
- **Cold-reload correctness** — the highest-value test (and the one a warm-kernel suite misses);
  it depends on the shared volume + the session dataset being in the kernel.
- **Two latent bugs** — already broken; Phase 2 is the natural place to fix, with a regression
  test that a fitted model can actually save→load round-trip (it can't today).

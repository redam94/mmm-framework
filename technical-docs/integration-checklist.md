# Integration checklist — every surface a new capability must touch

A shipped capability can be unreachable while CI stays green: config + code +
spec + a passing test file, and no user-facing surface can invoke it (price
levers lived that way for a release). This checklist is the full touch-point
list. `tests/test_capability_reachability.py` enforces the two registries that
used to drift silently (spec paths and the session-export map); the rest is
here so the decision is deliberate, not forgotten.

Work top-down; stop when a layer genuinely does not apply — and if a layer is
skipped on purpose, the reachability gate's allowlists are where that decision
is recorded WITH ITS REASON.

## 1. Core library

- [ ] Engine module under `src/mmm_framework/` (lean: no fastapi/langchain/
      httpx at module scope — `tests/test_lean_imports.py` CORE_IMPORTS gates
      it; ADD the new module to that list).
- [ ] Config: field on `ModelConfig` (`config/model.py`) or a dedicated
      Pydantic config; builder method on `ModelConfigBuilder`
      (`builders/model.py`).
- [ ] Enum values that serialize into specs/run metrics → add to
      `FROZEN_ENUM_VALUES` in `tests/test_api_contracts.py` (losing a value
      strands stored runs).

## 2. Agent spec registry

- [ ] `agents/fitting.build_model` reads the spec key; register it in the
      `unconsumed_spec_path` registries (`_SIMPLE_TOP_KEYS` /
      `_FREEFORM_TOP_KEYS` / `_ENUM_TOP_VALUES` / lever registries — all in
      `agents/fitting.py`).
- [ ] Add the path to `CONSUMED_SPEC_PATHS` (`tests/test_api_contracts.py`)
      and to `FIELD_SPEC_PATHS` in `tests/test_capability_reachability.py`;
      if deliberately NOT spec-wired, add a `FIELD_NOT_REACHABLE` reason
      instead — the gate refuses empty or non-reasons.

## 3. Model op

- [ ] Op function in `agents/model_ops.py` + entry in `OPS`; update the
      registry set in `tests/test_model_ops.py`.
- [ ] Decide `allow_unfitted`: ops that run without a posterior are checked at
      `agents/kernels.py` (`run_model_op`) and `agents/tools.py`
      (`_modelop_command` path) — a fitted-only op on an unfitted model must
      refuse, not crash.

## 4. Agent tool

- [ ] `@tool` function in `agents/tools.py`; register in `TOOLS` (and note
      `EXPERT_TOOLS` snapshots BEFORE the delegate append — a tool appended
      after the snapshot is orchestrator-only).
- [ ] Membership sets, each a deliberate decision:
      `HEAVY_TOOL_NAMES` (excluded from the orchestrator's default set),
      `_MMM_ONLY_TOOL_NAMES` / `_CAUSAL_TOOL_NAMES` (mode gating),
      `MILESTONE_TOOLS` + `STEP_LABELS` in `agents/workflow_guard.py`
      (per-turn milestone budget — the two must stay equal-keyed).
- [ ] Session export: map the tool in `agents/session_export.py::_OP_TOOLS`
      (op key + arg-mapper) so "export as Python" replays it — or add an
      `OP_NOT_EXPORTABLE` reason in `tests/test_capability_reachability.py`
      saying why replay cannot be deterministic (store reads, interactive
      args). Prefit ops also join `_PREFIT_OPS`.
- [ ] Dashboard state: decide whether the op's dashboard key survives into
      LangGraph state or is dropped (`_STATE_DROPPED_DASHBOARD_KEYS` in
      `agents/tools.py`) — numpy payloads must pass `_json_safe` either way
      (the msgpack checkpointer crashes on numpy scalars).

## 5. REST

- [ ] Route in `server/src/mmm_framework_server/main.py`, Pydantic schemas
      beside it. EVERY `/projects/{project_id}` route carries a tenant guard
      (`_proj_read`/`_proj_write`/`_proj_admin`) — `scripts/sync_api_surface.py`
      REFUSES to sync while one is missing, and
      `tests/test_api_surface_sync.py` fails CI. Rate limits (`_rl_*`) are
      not tenant guards.
- [ ] Long-running compute → the non-blocking planner-job pattern
      (`_start_planner_job` / `_poll_planner_job`), result_key matching the
      op's dashboard key.
- [ ] `make api-sync` — regenerates `tests/contracts/rest_routes.json`,
      `docs/shared/openapi.json`, `EXPECTED_OPS` + `docs/rest-api.html`
      together. Never hand-edit those four again.

## 6. Frontend

- [ ] Service in `frontend/src/api/services/`, hook in `api/hooks/`
      (start/poll shape for jobs), page/panel under `pages/`, route in
      `App.tsx` (tab registries live in each page's `index.tsx`).
- [ ] `appIdentity.PAGES` + `NAV_ICONS` when it is a new page; GuideBubble
      `page_context` so the guide chat can explain the surface.
- [ ] Enums mirrored from Python come from `scripts/gen_fe_enums.py`
      (`tests/test_fe_enum_mirror.py` gates drift) — never hand-copy.

## 7. Reporting

- [ ] Chart in `reporting/charts/`, extractor in `reporting/extractors/`,
      classic section in `reporting/sections.py` + registry, Augur section in
      `reporting/augur_sections.py` + `_SECTIONS` registry + `__all__`,
      bundle field in `reporting/extractors/bundle.py`, config field in
      `reporting/config.py`, generator ctor param (data-gated pattern —
      render only when the payload is attached).
- [ ] Interactive report (`reporting/interactive/`) if the section's numbers
      should recompute in-browser.

## 8. Persistence

- [ ] Serializer round-trip (`serialization.py`) for anything that must
      survive save/load; extended-flavor saves carry their own arrays.
- [ ] Sessions-store tables/accessors in `platform/sessions.py`; migrations
      are additive `CREATE TABLE IF NOT EXISTS` / `ALTER` guards in
      `init_db`.

## 9. Docs

- [ ] Technical spec in `technical-docs/` (top-level — the snippet gate only
      collects top-level `*.md`); every `mmm_framework` import in a fenced
      python block must resolve (`tests/test_docs_snippets.py` — no exec,
      import + attribute checks; `# pseudocode` first line opts a block out).
- [ ] Docs site page: `NAV_GROUPS` + `PAGE_TIERS` in
      `docs/shared/components.js` (the nav gate now fails on a tierless nav
      page), `GROUPS` in `docs/tools/build_seo.py` for llms.txt, then re-run
      `build_search_index.py` + `build_seo.py` from `docs/` and commit the
      regenerated `shared/*.json`.
- [ ] Sphinx: module `.rst` under `docs/api/source/api/` + toctree entry in
      `api/index.rst`; heavy optional deps go in `conf.py`
      `autodoc_mock_imports`.
- [ ] `CHANGELOG.md` entry; at release time mirror into
      `docs/changelog.html` — `tests/test_docs_versions.py` now fails when
      the page announces a version other than `pyproject.toml` or any
      `mmm-framework==X.Y.Z` pin is stale.
- [ ] Engineering-notes entry (`technical-docs/engineering-notes.md`) +
      CLAUDE.md subsystem-index line for non-obvious invariants.

## 10. Verification

- [ ] Synthetic-world gate where a causal truth exists (`synth/` worlds keep
      `response_fn` so window truths are recomputable — grade against the
      world's physics, not against the model's own output).
- [ ] Notebook walkthrough (`nbs/builders/build_<name>.py` → baked
      `nbs/demos/<name>.ipynb`, registered in `nbs/README.md`) for every new
      user-facing feature.
- [ ] Regression tests must FAIL on the unfixed code (stash the fix and run
      them once) — synthetic fixtures can pass against real bugs.

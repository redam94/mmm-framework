# Code-Side Notes — issues surfaced during docs work

Running tracker of code (not docs) issues raised while auditing and rewriting the
documentation site. Each entry: what was found, where, why it matters, suggested fix.
Add new entries at the top of the relevant section as they surface; move items to
**Resolved** with the fixing commit when closed.

_Last updated: 2026-06-12._

## Open — behavior bugs

1. **`fit()` ignores `ModelConfig.target_accept`.**
   `src/mmm_framework/model/base.py:1633` hardcodes `target_accept = target_accept or 0.9`
   and never reads `self.model_config.target_accept`, so
   `ModelConfigBuilder().with_target_accept(0.95)` silently does nothing unless the value
   is also passed to `fit()`. Fix: default from the config
   (`target_accept or self.model_config.target_accept or 0.9`).

2. **`compute_saturation_curves` is half-implemented.**
   Known from the workshop-notebook build (see memory/workshop notes); the docs avoid it
   (`workflow-budget-optimization.html` was rewritten to use
   `reporting/helpers/results.py` saturation fields instead). Decide: finish or remove.


## Open — security / API posture

4. **The agent API has no authentication dependency.**
   `src/mmm_framework/api/main.py`: no route carries an auth dep; the `X-API-Key`
   header there is the *LLM key passthrough*, not auth. The Models API shared key is
   off by default (`api/auth.py`, `api/config.py:60`). `docs/security.html` now states
   this honestly; the mitigation is perimeter/SSO until real auth lands (phase-3 §1.1
   defers it deliberately — this note is a reminder that the docs promise it).

5. **`technical-docs/agent-knowledge-workspace.md` §9 claims "All new routes require
   the existing X-API-Key dep" — false** against current `api/main.py` (see #4).
   Fix the doc or (better) make it true.

6. **Container egress on macOS dev is recorded as `open:unenforced-macos-dev`**
   (`agents/container_kernel.py`), while CLAUDE.md says hosted "egress is denied"
   unqualified. True on Linux; unenforced on macOS dev. Keep the caveat in any
   security statement (docs/security.html carries it).

## Open — stale internal docs

7. **`docs/model-configuration.md` provider table omits `lmstudio`**, which
   `agents/llm.py` fully supports. CLAUDE.md has it right; the md doc is stale.

8. **CLAUDE.md's audit-event list is incomplete**: names 5 events; the code emits 12
   (`kernel_respawn`, `kernel_timeout_interrupt`, `kernel_fit_start/done`,
   `kernel_egress`, `spawn_refused`, `overlay_wiped` in addition).
   `docs/security.html` documents the verified full set.

9. **`planning/eig.py` / `planning/design.py` apply no serial-correlation design
   effect analytically** — the EIG closed form assumes independent geo-weeks;
   `design.py` compensates empirically via placebo-calibrated power. Fine, but the
   docs (measurement-calibration.html, workflow-calibration-decisions.html) now
   describe exactly this split — if the analytic correction is ever added, update both
   pages. The in-page duration calculators also use the independent-weeks
   approximation (disclosed on-page).

## Open — examples

12. **`examples/ex_fit_and_report.py` hand-rolls HTML** instead of using
    `MMMReportGenerator(model=…, panel=…, results=…, config=ReportConfig(…)).to_html(path)`
    (the real pattern, used by the aurora builder and `nbs/demos/real_data_onboarding.ipynb`).
    Update the example so it demonstrates the shipped generator.

13. **Two `/projects` APIs exist and answer differently.** The top-level Models API
    (`api/main.py`, run as `cd api && uvicorn main:app`) serves a StorageService-backed
    project registry; the agent API (`uvicorn mmm_framework.api.main:app`) serves the
    sessions-store projects the React app and `scripts/seed_demo_project.py` use. CLAUDE.md's
    React run instructions point at the former, which shows an empty project list after
    seeding (hit during screenshot capture 2026-06-12). Fix the run instructions and/or
    unify the registries.

14. **Duplicate prior-predictive APIs**: `BayesianMMM.get_prior()` (base.py:1597) and
    `BayesianMMM.sample_prior_predictive()` (base.py:2180) overlap. Docs were
    standardized on `get_prior`; consider deprecating one. (Also: long-lived agent
    session kernels from prior days were found still alive on 2026-06-12 and one
    appears implicated in a hung notebook bake — check kernel LRU eviction is
    actually reaping idle kernels in the dev posture.)

## Open — docs infrastructure

10. **`docs/scientific-workflow-simple.html` has a pre-existing div imbalance**
    (60 open / 59 close by tag count; possibly a counter false-positive from markup in
    a string). Untouched during the 2026-06-12 overhaul; verify and fix.

11. **Hardcoded measured numbers across docs go stale on notebook re-bake.**
    Partial mitigation now exists (canonical scorecard footnotes, artifact JSONs in
    `nbs/artifacts/`, the doc-snippet API gate in `tests/test_docs_snippets.py`), but
    there is still no bake-time generation binding HTML constants to recorded outputs.

## Resolved

- **Learning-diagnostic verdict mislabeled evidence-driven relocation** (raised by
  Matthew 2026-06-12): `diagnostics/learning.py` computed `shift_z` but never used it
  in the verdict, so negative contraction + a ≥1-prior-sd mean shift (evidence
  dominating the location; width from likelihood flatness or tail/mixing inflation)
  read as "weak"/"prior-dominated". Fixed same day: new first-rank "relocated"
  verdict (`z_relocated=1.0`), plus `contraction_robust` (IQR-based — separates
  tail-driven widening from bulk widening) and `post_ess_bulk` (low ESS ⇒ the width
  estimate may be a sampler artifact) columns. Canonical case: `adstock_alpha_Search`
  under the parametric default (shift +1.6 prior-sd, c ≈ −1.0) → relocated.

- **Validator CV forward pass only supported the legacy blend-adstock model**
  (was #3): `_predict_at_indices` silently fell back to `mix=0.5` for parametric
  models. Fixed 2026-06-12 alongside the default flip: it now delegates to
  `validation/backtest.py:PosteriorForecaster` (both adstock paths, all saturation
  types, `train_offset` for rolling windows) and raises `NotImplementedError` on
  panel models / non-linear trends instead of silently mispredicting.
- **Adstock default**: `ModelConfig.use_parametric_adstock` flipped to `True`
  (2026-06-12) — closes the long-standing "make parametric the default or document
  why not" open finding. Legacy path retained via explicit `False` /
  `ModelConfigBuilder().with_legacy_blend_adstock()`; old pickles unaffected
  (the `getattr(..., False)` fallback in `model/base.py` is deliberate).

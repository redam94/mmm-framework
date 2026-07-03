# Production-Readiness Action Plan

> **Status:** Draft for review — uncommitted.
> **Source:** Multi-persona production-readiness review (media exec / planner / analyst / data scientist), 2026-06-24.
> **Owner:** Matthew Reda
> **Purpose:** Convert the review's findings into a sequenced, trackable plan with concrete tasks, acceptance criteria, and file pointers.

---

## 0. How to read this document

- **Severity:** `P0` = blocks any production use / can silently produce wrong numbers · `P1` = blocks commercial/hosted adoption · `P2` = quality, credibility, polish.
- **Effort:** `S` ≈ ≤1 day · `M` ≈ 2–5 days · `L` ≈ 1–3 weeks · `XL` ≈ multi-week / multi-person.
- **Track:** `LIB` = needed even for a self-hosted library release · `SAAS` = only needed for hosted multi-tenant · `BOTH`.
- Every task has **acceptance criteria** and a **test gate** so "done" is unambiguous.
- File/line pointers were verified during the review; line numbers may drift — treat them as anchors, not exact.

---

## 1. The strategic fork (decide this first)

Half the roadmap's cost (the `XL` data-tier migration, secure-by-default, DR, multi-tenant ops) is **only required if you are shipping a hosted SaaS**. The review's headline: *GA-ready as a self-hosted library; ~2–3 quarters from hosted multi-tenant SaaS.*

**Decision required — pick the near-term target:**

| Option | What ships | What's in scope now |
|---|---|---|
| **A. Self-hosted library / single-tenant** (recommended first milestone) | `pip install` + run-your-own-instance | WS-1, WS-2, WS-3, WS-6 (lineage hardening), WS-8, WS-9, WS-10, WS-11. **Defers** the XL data tier, multi-tenant security, DR. |
| **B. Hosted multi-tenant SaaS** | claude.ai-style hosted product | Everything in A **plus** WS-4, WS-5, WS-7. Adds ~2 quarters. |

> **Recommendation:** Commit to **Option A** as the next release (it's mostly correctness work on code you already wrote), and treat Option B as the following milestone. The plan below is structured so Option A is Phases 1–2 and Option B work is Phase 3+.

This document assumes you may pursue B eventually; tasks are tagged so you can drop the SAAS-only ones if you stay with A.

---

## 2. The two unanimous quick wins (start here)

Both are **P0**, both were flagged independently by 3–4 personas, and both are small relative to their value. The defining risk of the whole product — *"confident output, silent failures"* — is closed primarily by these two.

### QW-1 — Wire EDA validation into the fit path + fix the loader (`P0`, `M`, `LIB`)

**Problem.** The production loader silently corrupts data and the (good) EDA validators are never called by the fit path.
- Duplicate `(Period, variable)` rows are silently **summed** → KPI doubled with no warning. `data_loader.py` `_extract_variable` `groupby(dim_cols).sum()` (~line 416).
- Currency/thousands-formatted strings (`"$1,234"`, `"1,234"`) raise an opaque `TypeError: Could not convert string ... to numeric` deep in `groupby().sum()`.
- `validate_mff_structure` only parses `df[period].iloc[0]` (first row) → mixed/partly-bad dates pass validation, then raise a raw pandas `ValueError` inside `load()` (~line 350), bypassing `MFFValidationError`.
- `eda/validators.py` (`validate_dataset`) and `eda/outliers.py` are **advisory-only** — not called by `api/routes/data.py` (upload) or `agents/fitting.py::build_model` (fit).

**Actions.**
1. In `_extract_variable`, **detect duplicate keys before aggregating**. Default policy: raise `MFFValidationError` listing the offending `(Period, [dims], variable)` rows; allow an explicit `duplicate_policy={"error"|"sum"|"mean"|"first"}` config (default `"error"`).
2. Add **numeric coercion** for `VariableValue` in the loader: strip currency symbols/thousands separators, `pd.to_numeric(errors="raise")`, and on failure raise a clear `MFFValidationError` naming the column + a sample bad value. (Note: same root cause as the i18n locale gap — see WS-10/T-LOC.)
3. Fix `validate_mff_structure` to parse the **whole** period column (`pd.to_datetime(df[period], errors="coerce")`, report all NaT positions) so bad dates surface as `MFFValidationError`, not a mid-load crash.
4. **Wire validators into both entry points:**
   - `api/routes/data.py` upload → run `validate_dataset` and return structured issues (warn-tier vs error-tier) to the UI EDA tab (the dashboard already renders `dashboard_data.eda`).
   - `agents/fitting.py::build_model` (and the core `load_mff` path) → run `validate_dataset` with an **error-tier gate**: error-tier issues block the fit with an actionable message; warn-tier issues attach to results/diagnostics.
5. Add a `--force`/`allow_quality_warnings` escape hatch on the gate so an expert can override warn-tier (never error-tier silently).

**Acceptance criteria.**
- A dataset with a duplicated row **fails loudly by default** (or is aggregated only under an explicit policy), never silently summed.
- A `"$1,234"` column produces a clear validation error naming the column, not a `groupby` `TypeError`.
- A file with a bad date in row N (N>0) raises `MFFValidationError`, not a raw pandas error.
- `build_model` refuses to fit a dataset with error-tier issues (date gaps that shift adstock, negative spend, >X% missingness) unless overridden.

**Test gate.** New `tests/test_data_loader_messy_inputs.py`: duplicate rows, currency strings, mixed/partial bad dates, negative spend, zero-inflation, date gaps. Plus a `tests/test_fit_quality_gate.py` asserting `build_model` blocks an error-tier dataset and passes a clean one. Both in `make fast_tests`.

---

### QW-2 — Convergence gate for core *and* extended models (`P0`, `S`–`M`, `LIB`)

**Problem.** `fit()` records `divergences`/`rhat_max`/`ess_bulk_min` into a dict and **never warns**; `MMMResults` has no `converged` property; the gate exists only in opt-in `ModelValidator`. Extended models (`mmm_extensions/models/base.py::fit`) compute **zero** diagnostics. The headline ROI carries no trust signal.
- Core: `model/base.py::fit` (~lines 2012–2037) records but doesn't act. Thresholds already exist: `RHAT_OK=1.01`, `ESS_BULK_OK=400` in `validation/validator.py` / `diagnostics/snapshot.py`.
- Extended: `mmm_extensions/models/base.py::fit` (~lines 378–417) returns `ModelResults` without diagnostics; `ModelResults.diagnostics` defaults to `{}`.
- *(Correction from review: the React `ModelHealthPanel` already renders a `ConvergenceSection`/`ConvergenceTile`. The gap is programmatic gate + extended-model diagnostics + report-headline stamp — not "no signal exists at all.")*

**Actions.**
1. Add a shared `compute_convergence_diagnostics(idata)` helper (or reuse `diagnostics/snapshot.py`) and call it at the end of **both** `fit()` paths; populate `diagnostics` consistently for core and extended models.
2. Add a `converged: bool` (+ `convergence_summary`) property to `MMMResults` and the extended `ModelResults`, computed from rhat/ess/divergence thresholds.
3. `warnings.warn(...)` by default on any of: divergences > 0, `rhat_max > 1.01`, `ess_bulk_min < 400`. Make the threshold/severity configurable; never silent.
4. **Stamp the report headline.** In `reporting/` (executive summary / EstimandsSection), render a prominent banner when `not converged` ("⚠ This model has not converged — do not act on these numbers"). Optionally gate the executive ROI render behind a `render_unconverged=False` flag.
5. Surface `converged` in the API response and the agent's fit-summary so the chat/agent reports it.

**Acceptance criteria.**
- A deliberately under-tuned fit (few draws/tune) yields `results.converged == False`, emits a warning, and produces a report with a visible non-convergence banner.
- A NestedMMM / MultivariateMMM / CombinedMMM fit returns populated `diagnostics` with rhat/ess/divergences.
- The agent's fit summary states convergence status in plain language.

**Test gate.** `tests/test_convergence_gate.py` (fast: fake/short fits asserting `converged` flag + warning emitted) and one slow real-fit asserting populated extended-model diagnostics. Report test asserting the banner renders when `converged=False`.

---

## 3. Workstreams

Each task: **ID · title · severity · effort · track**. Tasks within a workstream are roughly ordered.

### WS-1 — Data integrity (`P0`, `LIB`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| D1 | QW-1 above (dup rows, currency, dates, validators-into-fit) | P0 | M | **Quick win** |
| D2 | Fix or remove `RaggedMFFLoader` / `load_ragged_mff` | P0 | M | Dead code: passes `explicit_nan_mask=` to `PanelDataset` (no such field) and reads `self.config.preserve_explicit_nan` (absent → `AttributeError`). It's in `__all__` and the agent tools prompt. Either implement explicit-NaN preservation properly or remove from `__all__`/prompt and raise `NotImplementedError`. |
| D3 | Replace silent KPI zero-fill with explicit missingness handling | P1 | M | Basic loader reindex-fills KPI gaps with `0` (~line 614) and media with a constant (~line 519) — distorts seasonality/adstock. Track missingness explicitly; warn; offer interpolation/NaN-aware options. |
| D4 | Content-type / magic-byte validation on upload | P2 | S | `api/routes/data.py` checks extension only. |

### WS-2 — Inference trust (`P0`/`P1`, `LIB`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| I1 | QW-2 above (convergence gate, core + extended, report stamp) | P0 | S–M | **Quick win** |
| I2 | Add **SBC** (simulation-based calibration) + **LOO-PIT** | P1 | L | The "genuine uncertainty quantification" claim is asserted but never machine-verified. Add SBC over the model's own prior for the structural params; add LOO-PIT to PPC. Wire into the validation suite (need not run on every fit). |
| I3 | Surface + soften the default prior | P1 | M | Default `beta ~ Gamma(μ=1.5, σ=1.0)` favors ROI>1 and forbids negative/zero effects → a dead channel can't show as a loser from observational data alone. Disclose it in the report; offer a prior that admits negative effects; document at call site. *(Correction: there IS an in-code comment at `base.py:~1463`; "undocumented" was overstated — the report-disclosure gap stands.)* |
| I4 | Guard bad LOO Pareto-k | P2 | S | `bad-k` count is surfaced but nothing warns when quoting LOO with k>0.7 common. Add a warning/annotation. |
| I5 | Restore `predict()` state (re-entrancy) | P2 | S | `predict()` mutates shared `pm.Data` via `set_data` with no `finally` restore (~lines 2192–2208) → non-reentrant; concurrent counterfactual calls in the agent service can read another request's media state. Wrap in `try/finally` restoring training data. |

### WS-3 — CI/CD & dependency hygiene (`P1`, `BOTH`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| C1 | Stand up CI (GitHub Actions) running `make fast_tests` + slow on a schedule | P1 | M | Only `.github/workflows/docs.yml` exists today. Gate PRs on the test suite. |
| C2 | Add ruff config + enforce in CI | P1 | S | No `[tool.ruff]` section; default rules already report 131 errors in `src` (75 unused imports, 12 unused vars), 46 in `api`. Add rule selection, fix, gate. |
| C3 | Add type-checking (mypy or pyright) in CI | P1 | M | `py.typed` shipped + hints "required" but no checker runs anywhere. Start non-strict, ratchet. |
| C4 | Raise dependency floors to lockfile reality | P1 | S | `langgraph >=0.2.14` (locked 1.2.0), `langchain-core >=0.2.33` (locked 1.4.0), `langchain-anthropic >=0.1.20` (locked 1.4.3) — a clean `pip install` without the lock can resolve to broken APIs. Bump floors. |
| C5 | Declare missing dev deps; pin asyncio mode | P2 | S | `pytest-asyncio` is undeclared (44 async tests rely on it transitively); `black` is used by `make format` but not a declared dep. Add both; pin `asyncio_mode`. |
| C6 | Add coverage threshold (`fail_under`) + dependency/security scan (pip-audit) | P2 | S | Coverage collected but never gated; no SBOM/vuln scan. |
| C7 | Stop suppressing the warnings that signal drift | P2 | S | `pytest.ini` does `ignore::DeprecationWarning` — hides the next arviz/pymc/pydantic break that `utils/arviz_compat.py` exists to absorb. Scope the ignore narrowly. |

### WS-4 — Data tier & deployment (`P0` for SAAS, `XL`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| P1a | Migrate off single-file SQLite → Postgres + Alembic migrations | P0(SAAS) | XL | All tenant data + auth + langgraph checkpoints + run_metrics in one ~3.2 GB local `sessions.db`; schema evolution is `try/except ALTER TABLE` with bare `except: pass`. SQLite serializes writes (throughput ceiling) and WAL is single-host. |
| P1b | Object store for artifacts (implement the declared S3 backend) | P0(SAAS) | L | `api/config.py` declares `storage_backend: Literal['local','s3']` + s3 creds and the docstring claims S3, but `StorageService` only implements local FS (no `boto3`). Selecting `s3` silently writes local → data loss/invisibility across pods. |
| P1c | Backups / PITR / restore drill | P0(SAAS) | M | No backup, replication, or recovery for any state today. |
| P1d | Implement the missing `KubernetesKernel` provisioner — or correct the deploy story | P0(SAAS) | XL | `deploy/k8s/api.yaml` sets `MMM_AGENT_KERNEL=kubernetes`, but `kernels.py` registers only `inprocess/subprocess/container`; unknown impls fall back to `inprocess`. The per-session-pod + gVisor + warm-pool topology has **no implementing code**. Either build it or rewrite the manifests to the container-kernel reality. |
| P1e | Fix `replicas:2` against pod-local SQLite | P0(SAAS) | — | Resolved by P1a+P1b+shared persistence; until then, two replicas hold divergent auth/session DBs and silently break tenant isolation (a correctness bug, not perf). |
| P1f | Ship an API/worker Dockerfile + compose | P1(SAAS) | M | Only the kernel image has a Containerfile; the API deploy story is `uvicorn --reload`. |
| P1g | ARQ/Redis resilience | P2(SAAS) | S | Redis is single-replica, no AOF/persistence; job state is Redis-only. Remove `print("DEBUG:")` spam; fix the no-op bug `pytensor.config.mode == "NUMBA"` (should be `=`). |

### WS-5 — Security-by-default (`P1`, mostly SAAS)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| S1 | Fix CORS | P1(SAAS) | S | `src/mmm_framework/api/main.py` sets `allow_origins=["*"]` **with** `allow_credentials=True` (browser-invalid + over-broad). Use an explicit allowlist (the legacy `api/` already does). |
| S2 | Redis-back the rate limiter | P1(SAAS) | M | `auth/ratelimit.py` is module-global in-memory → effective limit = N×configured across uvicorn workers / ARQ / replicas, resets on restart. Redis is already a hard dep. |
| S3 | Secure-by-default posture when hosted | P1(SAAS) | M | Auth, rate-limit, sandbox all OFF by default; default kernel `inprocess` runs untrusted LLM-authored code in the API process with full env (incl. cloud creds). Make `MMM_AGENT_HOSTED=1` imply the full secure chain; fail closed. |
| S4 | Quarantine/retire the legacy backend | P1 | M | Two backends with incompatible auth ship together: legacy `api/` uses plaintext static `X-API-Key` and `cloudpickle` artifact load (RCE vector if storage is attacker-influenced); agent API uses JWT/org. Pick one canonical stack; remove or clearly gate the other. |
| S5 | Secrets management + rotation runbook | P2(SAAS) | M | `MMM_AUTH_SECRET` is a raw env string; rotating it invalidates all live tokens (no dual-key overlap). Add KMS/Vault integration option + a documented rotation procedure with overlap window. |

### WS-6 — Governance, lineage & compliance (`P1`/`P2`, `BOTH`)
> *Correction from review: less missing than first thought. Lineage stamping exists on the agent path (`agents/tools.py` ~910–947 stamps `data_fingerprint`, `spec_hash`, `parent_run_id`, assumption stack; `api/runs.py` builds an MLflow-style `/runs` timeline). Governance docs exist (`technical-docs/soc2-readiness.md`, `security-questionnaire.md`, `data-processing.md`, public `docs/security.html`). Reframe from "absent" to "harden + implement what's documented."*

| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| G1 | Harden lineage into a content-addressed dataset store | P2 | L | Today it's per-fit content fingerprinting (good) but not an immutable version chain; the legacy `storage.py` path stores only a hash. Make a fitted model reliably resolvable to exact training bytes even across same-name re-deliveries. |
| G2 | Implement encryption-at-rest + retention/delete sweep | P1(SAAS) | L | Client files written as plaintext to disk; `data_retention_days` only purges Redis job records (`worker.py:~1278`). Add at-rest encryption, a TTL/retention sweep, and a GDPR/CCPA delete workflow. |
| G3 | PII screening on ingest | P2(SAAS) | M | No detection/redaction/masking. Add a screening pass + policy. |
| G4 | Side-by-side run/model comparison view | P1 | M | `/runs` timeline + `RunsTimeline.tsx`/`ModelHealthPanel.tsx`/`AgreementLog.tsx` exist (diff-timeline), but no per-channel ROI/contribution **delta table** across two runs — the analyst's most frequent ask ("why did TV ROI change since last refresh?"). |

### WS-7 — Operability (`P1` for SAAS) — *dimension the personas missed*
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| O1 | DR procedure + RTO/RPO + tested restore | P1(SAAS) | M | `soc2-readiness.md` itself admits "no documented RTO/RPO, no backup/restore tooling, no tested DR runbook, no HA config." |
| O2 | Runbooks / on-call / incident process | P1(SAAS) | M | Zero operational docs for a product whose DS persona "carries the pager." |
| O3 | Billing/quota enforcement | P2(SAAS) | M | `commercialization-summary.md`: entitlements **meter but don't block**. Add hard fit-quota blocking + (Stripe) billing integration. |
| O4 | Structured logging + tracing + durable metrics | P2(SAAS) | M | `logging.basicConfig` to stdout; `/metrics` counters are in-process (reset per restart/replica). Add request-id/correlation, OpenTelemetry, durable metrics. |

### WS-8 — Validation & causal coverage (`P2`, `LIB`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| V1 | Extend backtest to geo-panel and spline/GP/piecewise trend | P2 | L | `validation/backtest.py` `PosteriorForecaster` raises `NotImplementedError` for `n_cells>1` and trend ∉ {none,linear} — so the models real deployments use have **no out-of-time validation**, exactly where the CFO-facing credibility argument is needed. |
| V2 | Implement front-door / 2SLS estimators **or** downgrade the claim | P2 | L | DAG/agent layer reports front-door & IV identifiability, but no estimator exists — the user is told an effect is "identifiable" and still gets the plain back-door additive estimate. Either ship the estimator or label the check "identifiable in principle; not estimated." |
| V3 | Geo-heterogeneous estimation | P2 | L | `geo_identification.py` is a sufficiency report only; geo-level inference is deferred. |

### WS-9 — Planner surface (`P1`, `BOTH`)
> *Correction: budget-plan CRUD already exists in legacy `api/routes/budget_plans.py` (POST/GET/DELETE); the React app targets the agent API where `GET /budget-plans` is a `[]` stub. The fix is the two-backend unification (WS-5/S4), not building CRUD from scratch.*

| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| B1 | Expose budget-plan persistence on the canonical backend + build the scenario page | P1 | L | Wire `useBudgetPlans` to a real backend; ship a persistent, side-by-side scenario workspace (save/compare/reload), not a disappearing chat table. |
| B2 | Expose per-channel constraints on the agent/tool path | P0(planner) | S | The optimizer function supports a per-channel bounds dict; the agent tool only takes one global min/max multiplier. Plumb bounds through so partner caps / contractual floors / frozen lines are encodable. **Cheap, high-value.** |
| B3 | Add uncertainty to `what_if_scenario` | P0(planner) | M | Returns a point estimate; needs HDI + P(beats baseline). The optimizer already propagates uncertainty — reuse that path. |
| B4 | Geo/DMA + tactic-level allocation | P1 | L | `optimize_budget` sums to national channel-level; model supports geo panels. |
| B5 | Plan export (CSV / flight plan) + report allocation section | P1 | M | No way to hand a partner an executable deliverable; client report has no allocation/flighting section. |
| B6 | Flighting / temporal plan (forward calendar) | P1 | XL | "Flighting" currently means experiment pulses; `what_if`/optimizer only rescale the historical window. The temporal half of planning is absent. |

### WS-10 — UX, first-run & i18n (`P1`/`P2`, `BOTH`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| U1 | Async fits with progress/ETA + fit-specific cancel | P1 | M | `fit_mmm_model` runs NUTS synchronously in the chat turn (`MMM_FIT_TIMEOUT` default 1800s); user sees only a spinner with no progress or distinct cancel. Add a confirmation ("~N min, proceed?"), progress, and cancel. |
| U2 | Make React the canonical UI; deprecate Streamlit; rewrite README launch path | P1 | S | README still leads with "Launch the Streamlit App :8501" (stale `app/api_client.py`, last touched 2026-02-07) → new operators land on a half-broken UI. |
| U3 | No-API-key first-run / demo path | P1 | M | Login asks for an LLM API key (or server model) by default; password auth gated off → only an analyst can self-serve, contradicting the exec/planner pitch. Add a guided demo project + managed-key-default. |
| T-LOC | Internationalization / locale + currency | P1 | L | *Cross-cutting blocker no persona named.* Frontend hardcodes `Intl.NumberFormat('en-US')`; backend can't parse non-US number formats (same root cause as the QW-1 currency crash). A EU/UK/APAC client can neither ingest data nor read output. Add locale/currency config end-to-end. |
| U4 | In-app per-session report viewer/download | P2 | M | Agent report HTML is a disk artifact; dev `/report` is global; per-session routing only under hosted profile. |

### WS-11 — Code hygiene & accessibility (`P2`, `BOTH`)
| ID | Task | Sev | Eff | Notes |
|---|---|---|---|---|
| H1 | Fail-loud error handling | P1 | L | 442 `except Exception` + 87 swallow-to-`pass`/`continue` in `src` (33 swallow in `agents/tools.py` alone) → wrong-but-confident empty/partial output (e.g. a silently empty report section). Triage: re-raise or log+surface; reserve swallowing for truly optional paths. |
| H2 | Accessibility baseline | P1(SAAS) | L | 2 `aria-label` across 107 components, 227 `title=` on icon-only buttons, native `confirm()` for destructive deletes. Disqualifies enterprise/gov procurement (ADA/508). Add aria, focus management, keyboard nav, accessible dialogs, contrast audit. |
| H3 | Frontend test harness + tests | P1 | M | 0 test/spec files across ~30K LOC. Start with the SSE chat-stream parser and the spec-lock/diff logic. |
| H4 | Decompose god-modules | P2 | L | `agents/tools.py` (5,863 LOC), `api/main.py` (5,113), `model/base.py` (2,907), `api/sessions.py` (2,467). |
| H5 | Remove dead/debug code | P2 | S | `app/dep/*.py` (~245 KB), `debug_posterior_structure()` print dumps, `serialization.py` `print()` for save/load status, the `tests/synth` shim. |
| H6 | Fix fake reporting examples | P1 | S | `ex_fit_and_report.py` / `ex_reporter.py` hand-code simulated posteriors — analysts copy a fabricated pipeline. Make a real end-to-end fit the canonical example; label the simulated ones loudly. |

---

## 4. Phased timeline

### Phase 1 — "Stop being confidently wrong" (Weeks 1–4) · `LIB`
The two quick wins + the cheapest credibility fixes. After this, no silent corruption and no silent non-convergence.
- **D1 (QW-1)**, **I1 (QW-2)**, **B2**, **B3**, **C4**, **U2**, **I5**.
- **Milestone:** a dirty file fails loudly; a bad fit is flagged; planner can set per-channel constraints with uncertainty.

### Phase 2 — "Trustworthy library release" (Weeks 5–10) · `LIB`
- **C1, C2, C3, C5, C6, C7** (full CI gate), **D2, D3**, **I2 (SBC/LOO-PIT), I3, I4**, **H1, H6**, **G4**, **U1, U3**.
- **Milestone:** Option A GA — a self-hosted, CI-gated, calibration-verified library an expert can run with confidence; convergence + data quality enforced; planner constraints/uncertainty shipped.

### Phase 3 — "Hosted SaaS foundation" (Weeks 11–20) · `SAAS`
- **P1a, P1b, P1c, P1e, P1f** (data tier + backups), **S1, S2, S3, S4** (secure-by-default), **G2, G3** (compliance impl), **O1, O2** (DR + runbooks).
- **Milestone:** a multi-tenant deployment that can't lose data, is secure by default, and is operable.

### Phase 4 — "Commercial polish" (Weeks 21+) · `SAAS` + `LIB`
- **P1d** (or manifest rewrite), **P1g**, **S5**, **G1**, **O3, O4**, **V1, V2, V3**, **B1, B4, B5, B6**, **T-LOC**, **H2, H3, H4, H5**, **U4**.
- **Milestone:** planner desk fully built, global-ready, backtest covers production models, accessible.

---

## 5. Tracking checklist

> Update `[ ]` → `[x]` as tasks land. Keep this section as the single source of truth for status.

**Phase 1 (P0 quick wins + cheap credibility) — ✅ COMPLETE (PRs #16–#22, merged to main 2026-06-24)**
- [x] D1 — QW-1 loader hygiene + validators-into-fit *(PR #20)*
- [x] I1 — QW-2 convergence gate (core + extended + report stamp) *(PR #19)*
- [x] B2 — per-channel constraints on agent path *(PR #21)*
- [x] B3 — uncertainty on `what_if_scenario` *(PR #22)*
- [x] C4 — raise dependency floors *(PR #17)*
- [x] U2 — README → React; deprecate Streamlit *(PR #16)*
- [x] I5 — `predict()` re-entrancy *(PR #18)*

> Phase 1 verification: full fast-test suite green (2833 passed, 2 skipped); 6 new
> test files added (`test_predict_reentrancy`, `test_convergence_gate`,
> `test_data_loader_messy_inputs`, `test_fit_quality_gate`, `test_budget_constraints`,
> `test_what_if_uncertainty`). The two highest-risk "silent failure" paths
> (duplicate-row summing, non-convergence) are now fail-loud.

**Phase 2 (trustworthy library) — 14/16 done (PRs #23–#32, merged to main 2026-06-25)**
- [x] C1 CI tests *(#24)* · [x] C2 ruff *(#23)* · [x] C3 types *(#25)* · [x] C5 dev deps *(#23)* · [x] C6 coverage/scan *(#24)* · [x] C7 warnings *(#23)*
- [x] D2 RaggedMFFLoader *(#26)* · [x] D3 missingness handling *(#27)*
- [x] I2 SBC/LOO-PIT *(#30)* · [x] I3 prior disclosure *(#28)* · [x] I4 Pareto-k guard *(#28)*
- [x] H1 fail-loud errors *(#31, scoped: logged_suppress util + extractor hotspot; broad 442-site sweep is a follow-up)* · [x] H6 fix fake examples *(#29)*
- [x] G4 run-comparison delta **backend** *(#32)* — `compare_runs` + `/runs/compare`; **FE side-by-side view deferred** (no FE test harness; see H3)
- [ ] U1 async fit UX — **deferred** (architecturally significant FE+kernel change; risks the working synchronous fit path; not verifiable without the FE harness)
- [ ] U3 no-API-key first-run — **deferred** (FE login change; not verifiable without the FE harness)

> Phase 2 verification: full fast-test suite green (2853 passed, 2 skipped); CI live
> (`.github/workflows/ci.yml` + nightly `ci-slow.yml`); ruff clean on src+api; mypy
> non-blocking baseline (643 errors to ratchet); 7 new test files. **U1, U3, and
> G4's FE view need the frontend test harness (H3, Phase 4) before they can be
> shipped verifiably — flagged for an explicit FE pass rather than shipped blind.**

**Phase 3 (SaaS foundation)**
- [ ] P1a Postgres+migrations · [ ] P1b object store/S3 · [ ] P1c backups/PITR · [ ] P1e replicas fix · [ ] P1f Dockerfile
- [ ] S1 CORS · [ ] S2 Redis rate limit · [ ] S3 secure-by-default · [ ] S4 retire legacy stack
- [ ] G2 encryption/retention · [ ] G3 PII screening
- [ ] O1 DR/RTO-RPO · [ ] O2 runbooks/on-call

**Phase 4 (polish) — testable slice done (PR #34); rest deferred (infra/FE/large-modeling)**
- [x] S5 secrets rotation *(#34, docs; JWT dual-key noted as follow-up, encryption dual-key real)*
- [x] V2 front-door/2SLS — **downgraded honestly** *(#34: identifiable≠estimated caveat; a real 2SLS/front-door estimator stays deferred)*
- [x] T-LOC i18n/locale — **backend** *(#34: locale-aware numeric parsing; FE i18n deferred)*
- [x] H5 dead code *(#34)*
- [x] H4 decompose god-modules — **done** *(#43: agents/spec_normalize.py extracted from tools.py)*
- [x] V1 backtest geo/spline — **done** *(#44: geo-panel PosteriorForecaster, per-cell forward pass; MAPE 0.063 < seasonal-naive 0.141 across 4 geos)*
- [x] V2 front-door/2SLS estimator — **done** *(#39: real two_stage_least_squares + frontdoor_estimate in estimators/causal.py)*
- [x] V3 geo-hetero estimation — **done** *(#45: vary_media_by_geo partial-pooled per-geo betas; recovers planted mults mean corr +0.90 vs pooled scalar)*
- [x] G1 content-addressed lineage store — **done** *(#41: lineage.py DatasetLineage)*
- [x] B1/B4/B5/B6 planner FE + U4 in-app report — **done** (the **Almanac** planner page). B1: real budget-plan persistence on the agent backend (`budget_plans` table + CRUD in `sessions.py`; `/budget-plans` GET/POST/GET-one/DELETE replacing the `[]` stub) + non-blocking compute (`POST/GET /projects/{id}/planner/optimize|scenario` → new `plan_budget`/`plan_scenario` ops) + a save/compare/reload scenario workspace. B4: per-(geo,channel) joint allocation (`planning/budget.py::optimize_budget_by_geo` reusing the greedy allocator over flattened arms) surfaced as a per-geo table. B5: CSV flight-plan export (`/budget-plans/{id}/export.csv`) + a data-gated, default-off `AllocationSection` in the report. B6: forward flighting calendar (`planning/flighting.py`, even/front/back/pulsed/seasonal) rendered as a stacked-bar calendar. U4: per-session `ReportViewer` (`?thread_id=`), reused on the planner and wired into the workspace report widget.
- [x] H3 frontend test harness — **already present** (vitest); the planner ships with service/hook/component tests (`plannerService`, `ReportViewer`, `PlanCompare`, `AllocationResult`), which is what previously blocked B1/B4/B5/B6/U4.
- [ ] P1d KubernetesKernel · P1g ARQ/Redis · O3 billing · O4 observability — **deferred (need live infra)**
- [ ] H2 a11y — **deferred** (broad audit across all components)

> Phase 4 verification: full fast suite green (2885 passed, 2 skipped); 4 new test
> files; ruff clean. The deferred items split into (a) needs-live-infra
> (Postgres/Redis/K8s) and (b) the H2 a11y sweep — none shippable *verifiably*
> here, so flagged rather than shipped blind.
>
> **Planner/U4 follow-on (this PR):** B1/B4/B5/B6 + U4 shipped as the Almanac
> page. Backend: `tests/test_planner_budget.py`, `tests/test_budget_plan_endpoints.py`,
> `tests/reporting/test_allocation_section.py` (28 fast tests). Frontend: 5 vitest
> specs (51 FE tests total green), `tsc -b` + eslint clean. The studio exposes
> **per-channel** spend bounds (min×/max× per media channel, on top of a global
> default) so a planner can cap how far budget moves / freeze a line — these flow
> through as `channel_bounds` to the already-supporting `plan_budget` op (geo path
> expands them per-arm). Remaining planner follow-up: auto-attaching a live plan to
> the agent's generated client report (the AllocationSection renders today only
> when allocation data is attached to the bundle).

---

## 6. Risk register (top items)

| Risk | If unaddressed | Mitigation task |
|---|---|---|
| Silent KPI corruption (dup rows / currency) | Wrong ROI presented to CFO; trust destroyed at first reconciliation | **D1** |
| Non-converged model shipped as confident ROI | Budget decisions on meaningless numbers | **I1** |
| Single-file SQLite, no backups | Total tenant data loss on one disk failure | **P1a–P1c** |
| Advertised cloud/K8s/S3 deploy doesn't exist | Procurement/legal blocker; can't actually stand up the sold architecture | **P1b, P1d** |
| No CI on a 48K-LOC test suite | Silent regressions in the ROI math across versions | **C1** |
| Uncertainty asserted, not verified (no SBC) | Headline intervals may be optimistic; core differentiator unproven | **I2** |
| i18n/locale absent | Entire non-US market can't ingest or read output | **D1 + T-LOC** |
| No DR / runbooks / billing enforcement | Unoperable + unmonetizable as a hosted product | **O1–O3** |

---

## 7. What's already strong (don't re-litigate)

Keep these as load-bearing assets; the plan above protects rather than replaces them:
- Single-source-of-truth saturation (`_apply_saturation_pt`) — attribution can't drift from the fit.
- Experiment calibration (two routes, in-graph likelihood on the same structural params) — the real causal anchor.
- Causal-rigor diagnostics (Cinelli-Hazlett sensitivity, fit-based refutation suite, parameter-learning contraction).
- Production-grade rolling-origin backtest (for the model families it supports).
- Decision-theoretic experiment planning + economics + Pareto optimizer.
- Synthetic DGP worlds with honest recovery grading.
- Client-ready branded HTML reporting.
- Solid auth crypto + hardened container kernel (when enabled).
- Honest governance documentation (SOC2 readiness, SIG-Lite, data-processing register) — a procurement asset, not a void.

---

*Generated from the 2026-06-24 multi-persona review. Edit freely; this is a working plan, not a contract.*

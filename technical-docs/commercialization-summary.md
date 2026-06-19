# Commercialization arc — summary

How the MMM Framework went from a rigorous research codebase to a multi-tenant,
sellable product. This is the engineering companion to the pitch deck
(`docs/business-readiness-report.html`) and the live tracker
(`technical-docs/business-readiness-p0.md`).

**Thesis.** The hard, risky part of becoming a SaaS — real authentication,
tenant isolation, a buyer-facing security posture, and entitlements that
actually enforce — is **shipped**. The science and the security are de-risked;
what remains is business motion (a signed design partner, billing) and a couple
of clearly-scoped integrations that need customer credentials. Every claim below
is backed by code + tests in the repo, and the honesty discipline (synthetic vs.
real validation, no overclaiming) was enforced throughout.

---

## At a glance

| Tier | Item | Status |
|---|---|---|
| **P0** | Auth · multi-tenancy · RBAC (both backends) | **Shipped** |
| **P0** | Trust & security collateral | **Shipped** |
| **P0** | Security hardening (adversarially reviewed) | **Shipped** |
| **P0** | Pricing / packaging with enforceable entitlements | **Shipped** |
| P0 tail | Billing integration (Stripe) + hard fit-quota *blocking* | Open (metered, not blocked) |
| **P1** | Self-serve onboarding + inline data validation | **Shipped** |
| **P1** | Reliability / observability (off-host audit, `/observability`) | **Shipped** |
| **P1** | Reference-customer enablers (design-partner kit, ROI template) | **Shipped** |
| P1 tail | A signed design partner | Open (non-code / sales) |
| **P2** | "Model-defense" report (one-click, brandable) | **Shipped** |
| **P2** | Portfolio benchmarking API + dashboard | **Shipped** |
| **P2** | GCS + BigQuery data integrations | **Shipped** |
| **P2** | Saved data connections + scheduled auto-sync | **Shipped** |
| **P2** | Org admin panel + user settings page | **Shipped** |
| P2 tail | Live ad-platform API clients | Open (needs customer credentials; BigQuery-transfer path works) |

---

## Locked decisions (made early, with the user)

1. **Built-in JWT auth now, IdP-ready later.** A pluggable `TokenVerifier`
   protocol lets `LocalJWTVerifier` (HS256, stdlib) be swapped for OIDC/JWKS
   without touching call sites. No third-party auth dependency today.
2. **An Organization tenant layer above projects.** Org → projects → sessions.
   Every row is org/project-scoped; cross-tenant access returns 404.
3. **Dependency-light.** Passwords use scrypt (`cryptography`); JWT is stdlib
   `hmac`/`hashlib`; the store is stdlib `sqlite3`. No new heavy runtime deps for
   the security core.

---

## P0 — table stakes

### Auth, multi-tenancy, RBAC  (`src/mmm_framework/auth/`)
A self-contained, layered package:
- `passwords.py` — scrypt hashing, self-describing encoding, constant-time verify.
- `tokens.py` — HS256 encode/verify, `make_claims` (carries `org`, `role`, and a
  `tv` token-version for **stateless revocation**), the `TokenVerifier` protocol.
- `store.py` — orgs, users, `org_members(org_id, user_id, role)`, invites,
  password resets, usage counters — raw `sqlite3`, idempotent schema init.
- `service.py` — signup, login, invite/accept (seat-limited), deactivate,
  change-password, revoke-all.
- `plans.py` — tiers + feature gates + usage metering (see Pricing).
- `deps.py` / `routes.py` — FastAPI principal resolution, `require_org_role`,
  `require_project_access`, and the `/auth/*` router.

RBAC roles, least→most: **viewer < analyst < admin < owner**. Tenant guards are
mounted on **both** backends — the classic `api/` and the agent
`src/mmm_framework/api/` — via each route's `dependencies=[...]`. A dev principal
(`is_dev`) bypasses checks for single-tenant local use; production enforces.

**Tests:** `tests/test_auth_{foundation,enforcement,lifecycle,plans,audit,ratelimit,admin_routes}.py`
(the enforcement matrix proves org A cannot read/write org B's projects).

### Trust & security collateral
Buyer-facing security posture documented honestly — **SOC 2 framed as
"readiness," not "certified."** Token revocation, tenant isolation, audit chain,
and the data-handling story are written down and kept in sync with the code (a
mid-build inventory caught and reconciled stale "no token revocation" claims).

### Security hardening (adversarially reviewed)
A dedicated hardening pass plus multi-agent adversarial reviews that found and
fixed **real** bugs before they shipped — see *Engineering rigor* below.

### Pricing & packaging  (`auth/plans.py`)
Four tiers with **enforceable** limits, not marketing copy:

| Plan | Seats | Feature gates |
|---|---|---|
| free | 2 | core |
| team | 5 | + collaboration |
| business | 25 | + audit export |
| enterprise | unlimited | + SSO / OIDC |

`org_usage(org_id)` returns live seats/projects/fits vs. limits;
`assert_within_seat_limit` blocks invite-accept past the cap; `PlanLimitError`
surfaces as a user-facing 402. **Caveat:** fits are *metered* today, not *blocked*
— hard quota enforcement + Stripe billing is the remaining P0 tail.

---

## P1 — make it stick

- **Self-serve onboarding.** A checklist API (`api/onboarding.py` +
  `GET /projects/{id}/onboarding-status`) drives a React `OnboardingChecklist`
  on the Program home; project onboarding renders a `project_brief.md` into the
  project KB so the agent retrieves it. Inline **data-quality verdict** via
  `GET /projects/{id}/data-quality` (`summarize_eda_issues`) so a user knows
  their data is model-ready before fitting.
- **Reliability / observability.** An off-host **audit shipper**
  (`agents/audit_shipper.py`, cursor-based, background lifespan tick →
  `MMM_AUDIT_SHIP_URL`) so the local hash-chained log isn't the only copy; a
  `GET /observability` endpoint (`api/observability.py`) reporting audit-chain
  health, event counts, ship backlog, and run-metrics activity.
- **Reference-customer enablers.** A design-partner program kit and an ROI
  template; the synthetic-world tooling + branded report generator make
  white-glove onboarding cheap to deliver.

**Tests:** `tests/test_{onboarding_status,audit_shipper,auth_audit}.py`.
**Remaining:** a *signed* design partner (sales, not code).

---

## P2 — accelerate

### Model-defense report  (`reporting/model_defense.py`)
A one-click, brandable report that answers "why should I trust this model?":
causal refutation (placebo / vanish-stable tests) + convergence (divergences,
r-hat) + calibration count → a verdict ladder (Robust / Qualified /
Needs-scrutiny, where non-convergence dominates), with plain-English per-test
explanations and honest caveats. Agent tool `generate_model_defense_report`
runs the validator and writes `agent_model_defense.html`; served at
`GET /model-defense`. **Tests:** `tests/test_model_defense{,_tool}.py`.

### Portfolio benchmarking  (`api/portfolio_benchmark.py` + `frontend/.../Portfolio/`)
For an agency / holding-co: aggregate the latest `run_metrics` per brand into a
cross-brand benchmark (channel ROI median / p25–p75 / percentile) + governance
(model freshness, calibration coverage). `GET /portfolio-benchmark` is
org-scoped. The **Constellation** dashboard renders it: governance tiles, a
per-channel ROI box-plot with every brand scattered as a dot (top/bottom-quartile
colored), and a brand table (fit recency, portfolio mROI, vs.-portfolio
leaders/laggards). **Tests:** `tests/test_portfolio_benchmark.py`.

### Data integrations  (`src/mmm_framework/integrations/`)
The land-and-expand surface, built to reuse the platform's existing **ADC**
identity (the same one Vertex uses) so there are **no new secrets to manage**.

- **GCS + BigQuery data sources** — a `DataSource` ABC + `GCSDataSource`
  (CSV/Parquet objects) and `BigQueryDataSource` (query or table → DataFrame),
  ADC-first auth, lazy SDK behind a new `[gcp]` extra, injectable clients for
  testing. `build_data_source` / `list_data_sources` power a catalog.
- **Agent tools** `load_from_bigquery` / `load_from_gcs` pull a DataFrame
  straight into the session's working dataset; `GET /integrations/catalog`
  (auth-gated) lists what's available + installed.
- **Ad-platform stubs + recommendation** (`integrations/ad_platforms/`,
  `technical-docs/ad-platform-integrations.md`): a shared `AdPlatformConnector`
  contract + `spend_to_mff` helper + Google/Meta/TikTok stubs. The recommended
  path is to **land spend in BigQuery via a managed transfer** (lowest-effort,
  GCP-native) and read it through the BigQuery source; direct-API ease is
  ranked **Meta (easy) > Google Ads / TikTok (moderate)**.
- **Saved data connections** — project-scoped, reusable connections (config is a
  **non-secret reference only** — bucket/object or project/dataset/query/table;
  auth stays ADC, so there's nothing to encrypt at rest). RBAC-gated CRUD + a
  `test` probe + a capped `preview`; an agent tool `sync_data_connection(name)`;
  a Settings → Data connections manager UI. *Save once, then "sync my weekly
  connection" in chat.*
- **Scheduled auto-sync** — connections refresh on an interval
  (Manual / Hourly / Daily / Weekly) via a background lifespan tick (mirrors the
  audit shipper). Each connection is isolated (per-read timeout, row cap, honest
  ok/error status, error backoff); a per-project snapshot CSV + freshness state
  (last-synced, row count, status) surface in the UI.

**Tests:** `tests/test_integrations_gcp.py`, `_ad_platform_stubs.py`,
`_integrations_wiring.py`, `_data_connections.py`, `_connection_sync.py`.
**Remaining:** live ad-platform API clients (need a customer's OAuth /
developer-token / system-user credentials; the BigQuery-transfer path works now).

### Admin & user management  (`auth/routes.py` + `frontend/.../{Admin,Settings}/`)
- **Org-admin endpoints**: list members, change role, remove member (only owners
  may grant/remove the owner role; the last owner is protected), pending-invite
  list + revoke, change-password (re-issues tokens so the caller stays in).
- **Admin panel "Curia"** (`/admin`, admin/owner only): invite-by-email with a
  copyable token, role management, seat-vs-tier usage tiles.
- **Settings page "Sanctum"** (`/settings`): Profile, Security (change password),
  Model & API (provider/model/key, ADC-aware), and Data connections.
- Role-gated nav (`pageVisibleToRole`) backed by a `/auth/me` current-user fetch.

**Tests:** `tests/test_auth_admin_routes.py`.

---

## Engineering rigor

**Per-task commits.** Every task lands as its own green commit
(`make format` + `tsc --noEmit` + `vite build` + the touched test files), with a
descriptive message and a co-author trailer. The integrations + admin arc alone
is 13 commits (`f0d074b` … `c70b72f`).

**Adversarial review caught real bugs.** Multi-agent review workflows
(find → independently refute → keep only confirmed) ran before declaring each
major change set done — and earned their keep:
- **Privilege escalation** — any *admin* could grant themselves the *owner* role
  (backend guard + frontend dropdown). Fixed: only owners manage the owner role.
- **SQL-injection vector** — a BigQuery table-id regex used `$`, which matches
  before a trailing newline, letting `dataset.table\n` smuggle a newline into the
  backticked SQL. Fixed with `\Z`.
- **Error-message leakage** — GCS/BigQuery SDK errors (project ids, service-account
  emails, credential paths) were echoed into chat / API responses. Fixed with a
  shared `scrub_cloud_error`.
- **Scheduler bugs** — silent snapshot overwrite on name collision, no read
  timeout (one hung pull stalled the batch), a failed write reported as `ok`,
  unbounded interval → `inf` → 500, and no error backoff. All fixed + tested.

**Honesty discipline.** Throughout the collateral: synthetic-world results are
labeled synthetic, real-data results labeled real; a corrected runtime multiplier
(6.0×, not the earlier 7.9×); SOC 2 as "readiness," not "certified"; and the
tracker marks integrations **Partial** while ad-platform clients are stubs — no
overclaiming.

---

## What remains (and why)

| Item | Why it's not done autonomously |
|---|---|
| Stripe billing + hard fit-quota *blocking* | Needs a Stripe account + webhook wiring; fits are metered today. |
| A signed design partner | Sales motion, not engineering. |
| Live ad-platform API clients | Need a customer's OAuth client / Google Ads developer token / Meta system-user token; the documented BigQuery-transfer path is fully working. |
| Per-tenant cloud identity for scheduled sync | Today's sync reads under the server's ambient ADC identity (same surface already exposed by `load_from_*`/preview); per-tenant workload identity is a hosted-posture item. |

---

## Key map

- **Security core:** `src/mmm_framework/auth/`
- **Integrations:** `src/mmm_framework/integrations/` (+ `ad_platforms/`, `connections.py`)
- **Scheduler:** `src/mmm_framework/api/connection_sync.py` (lifespan tick in `api/main.py`)
- **Reports:** `src/mmm_framework/reporting/model_defense.py`, `api/portfolio_benchmark.py`
- **Frontend:** `frontend/src/pages/{Admin,Settings,Portfolio,Program}/`
- **Pitch deck:** `docs/business-readiness-report.html`
- **Live tracker:** `technical-docs/business-readiness-p0.md`
- **Ad-platform guidance:** `technical-docs/ad-platform-integrations.md`

**Env knobs** (all inert by default): `MMM_AUTH_ENABLED`, `MMM_AUDIT_SHIP_URL`,
`MMM_CONNECTION_SYNC_INTERVAL` (0 disables) / `_MAX_ROWS` / `_READ_TIMEOUT`,
`MMM_GCP_*` / `MMM_GCS_*` / `MMM_BIGQUERY_*`, and the `[gcp]` extra
(`uv sync --extra gcp`).

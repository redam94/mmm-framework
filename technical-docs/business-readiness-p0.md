# Business Readiness — P0 Tracker

> **Pitch deliverable:** `docs/business-readiness-report.html` — a standalone,
> print/PDF-ready business readiness report (exec summary, opportunity, product,
> moat, honesty-labeled validation evidence, and the P0→P2 scorecard with live
> status). Self-contained (no nav inject); links to `trust.html`. Keep its status
> chips in sync with this tracker.


Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

This tracks the **P0** work that turns the MMM Framework from a capable internal
tool into a sellable, multi-tenant SaaS. Three tracks, executed in sequence but
sharing one foundation (the auth/tenant layer everything else references).

Decisions locked (2026-06-17):
- **Auth**: built-in JWT now, structured so an external IdP (OIDC/JWKS) plugs in
  later without changing call sites. No new runtime dependencies — HS256 via
  stdlib `hmac`, password hashing via scrypt (`cryptography`, already installed).
- **Tenant boundary**: an **Organization** layer above projects. Users belong to
  an org; projects belong to an org; access is checked org-first, then role.
- **System of record**: `src/mmm_framework/api/sessions.db` (already owns
  `projects` / `users` / `project_members`). The classic root-`api/` app shares
  the same org/principal layer via `mmm_framework.auth`.

---

## Track 1 — Authentication, tenancy & access control (engineering)

### Phase 1.1 — Shared auth foundation  `[x]`  (`src/mmm_framework/auth/`)
- [x] `passwords.py` — scrypt hash/verify (constant-time), dependency-free
- [x] `tokens.py` — HS256 JWT encode/verify + `TokenVerifier` protocol +
      `LocalJWTVerifier` + `OIDCVerifier` stub + `build_verifier()` factory
- [x] `config.py` — `AuthSettings` (`MMM_AUTH_*`): enabled flag, provider,
      secret, issuer/audience, token TTLs, optional bootstrap org/admin
- [x] `models.py` — `Role` enum + ordering, `AuthContext`, request/response DTOs
- [x] `store.py` — `organizations` + `org_members` tables; `users` gains
      `password_hash`/`org_id`/`status`/`last_login_at`; `projects` gains
      `org_id`. Idempotent migrations against `sessions.db`. CRUD.
- [x] `service.py` — `signup_organization`, `authenticate`, `issue_tokens`,
      `bootstrap_from_env`
- [x] `deps.py` — `get_current_principal`, `require_org_role`,
      `require_project_access` (404-on-cross-tenant to avoid existence leaks)
- [x] `routes.py` — `/auth/signup`, `/auth/login`, `/auth/refresh`, `/auth/me`
- [x] Unit tests — `tests/test_auth_foundation.py`

### Phase 1.2 — Mount & enable  `[x]`
- [x] Mount auth router in root `api/` (additive, safe; disabled by default)
- [x] Mount auth router in agent app (`src/mmm_framework/api/main.py`)
- [x] Resolve principal on the agent app's `/chat` + project list/create +
      experiment-create routes (dev principal when auth off → no behavior change)
- [x] Seed: `initialize_auth()` runs on startup in both apps — schema +
      optional bootstrap owner + one-time backfill attaching every org-less
      project/user to the primary org (idempotent; nothing 404s when 1.3 flips on)
- [x] `create_project(org_id=…)` stamps the tenant; `list_projects(org_id=…)`
      scopes to it; tests in `tests/test_auth_foundation.py`

Notes for 1.3: the frontend must start sending `Authorization: Bearer` on
`/chat` and project calls once `MMM_AUTH_ENABLED=1` (today it sends only the LLM
`x-api-key`). Until enforcement, the agent app's `/projects/{id}/...` subroutes
are still unscoped — that sweep is 1.3.

### Phase 1.3 — Enforce tenant isolation on every route  `[x]`
- [x] Agent app: `require_project_access(role)` on all **22** `/projects/{id}/...`
      routes via route-level `dependencies=[...]` (read=viewer, write=analyst,
      destructive=admin). `deps.ensure_project_access()` is the shared 404-on-
      cross-tenant primitive.
- [x] Agent app: experiment **by-id** routes (get/upsert/transition/delete)
      check ownership of the experiment's project.
- [x] Agent app: list endpoints scoped to the caller's org — `/projects`
      (1.2), `/experiments`, `/sessions` filter to `list_org_project_ids(org)`.
- [x] Handlers use the `Annotated[..., Depends] = _DEV_PRINCIPAL` pattern so
      direct (unit-test) calls still work; FastAPI injects the real principal.
- [x] Cross-tenant access test matrix — `tests/test_auth_enforcement.py`
      (A↔B read/write 404, role gating 403, 401 no-token, list scoping).
- [x] Agent app: **whole thread-keyed surface** guarded (audit found ~35 holes
      a path-param dep didn't cover). New `require_session_access(role)` factory
      (binds `thread_id` from path OR query) on all `/state`,`/history`,`/rewind`,
      `/dag`,`/spec`,`/outliers`,`/dataset/preview`,`/assumptions`,`/workflow`,
      `/files/{thread_id}`,`/workspace`,`/sessions/{thread_id}*`,`/artifacts/{thread_id}`
      routes + all 10 report routes (`deny_missing` 404s thread-less report under
      real auth). Helper `_ensure_session_access` resolves session→project→org;
      legacy project-less sessions allowed single-tenant, denied hosted.
- [x] Agent app: IDOR by-id routes — `/models/{id}`(+dashboard), `/files/{id}`
      (download/delete), `/artifacts/{id}` (download/delete), `/kb/{id}` delete,
      `/analysis-plans/{id}` get/delete — resolve the row's owning thread/project.
- [x] Agent app: body/aggregate holes — `/chat` (drive only own attributed
      session; new threads still allowed), `POST /sessions` (can't plant into
      another org's project), `/models` list, `/runs`, `/portfolio`,
      `/analysis-plans` list scoped to org.
- [x] Fixed a latent verifier-cache bug (rotated `MMM_AUTH_SECRET` was ignored).
- [x] Session/report cross-tenant tests added (`tests/test_auth_enforcement.py`).
- [x] Root `api/` app (classic, filesystem storage) — `org_id` stamped into
      `api/storage.py` metadata + list filtering + **IDOR ownership 404s on every
      by-id route** (data/configs/models+~13 result endpoints/extended/budget/
      projects), via shared `org_scope`/`assert_org_owns` + org-aware
      `_check_model_completed`/`_check_extended_model_*`. Fit validates data_id +
      config_id (+mediator) org before enqueue; budget-plan create validates the
      model org. `api/routes/sessions.py` by-id session + analysis-plan endpoints
      org-checked. Backfill: `api/backfill_org.py` + `storage.backfill_org_id()`
      stamps legacy org-less records. Tests: `api/tests/test_org_isolation.py`
      (cross-org 404 + list scoping + backfill).
- [x] Frontend — JWT store + login + `Authorization: Bearer` axios interceptor +
      `bearerHeader()` on all raw-fetch sites (incl. the two the impl missed:
      `useChatStream` artifacts load, `DatasetPanel` preview) + single-flight
      refresh-or-logout (keeps the LLM key on JWT expiry). `tsc --noEmit` +
      `vite build` clean. `VITE_AUTH_ENABLED === 'true'` opt-in gate.
- [ ] Residual (low sev, documented): `/plots/{id}`,`/tables/{id}` rely on the
      content-hash + thread salt rather than a positive ownership gate;
      `storage.count_by_project` not org-filtered (only reached post-ownership-
      check, so not exploitable). Migration note: run `python api/backfill_org.py
      <org>` before flipping `MMM_AUTH_ENABLED=1` on an existing install.

### Phase 1.4 — Lifecycle & hardening  `[x]`
- [x] **Invite flow + password reset + deactivate** — `/auth/invite` (admin),
      `/auth/accept-invite`, `/auth/password-reset/request` (202, no
      enumeration) + `/auth/password-reset/confirm`, `/auth/users/{id}/deactivate`
      (admin, same-org only). Opaque single-use tokens (`invites`,
      `password_resets` tables); reset/invite tokens delivered out-of-band.
- [x] **Refresh-token rotation + revocation** — `revoked_tokens` blocklist;
      `/auth/refresh` is single-use (revokes the presented token, mints a new
      pair); `/auth/logout` revokes; deactivated/disabled users fail refresh.
- [x] **Audit log** — `auth/audit.py` emits structured `auth.*` events
      (login/logout/refresh/invite/accept/reset/deactivate) to the `mmm_audit`
      logger tagged with `user_id` + `org_id`. Tests: `tests/test_auth_lifecycle.py`.
- [x] **Instant access-token kill** — a `tv` (token_version) claim stamped at
      issue + checked in `get_current_principal` (1 indexed read/request).
      `deactivate_user` / `confirm_password_reset` / new `/auth/logout-all`
      (`revoke_all_sessions`) bump the version → every live access AND refresh
      token for that user is rejected on next use, immediately. Legacy tokens
      (no `tv`) read as 0 and keep working until a bump. Tests:
      `test_auth_lifecycle.py::test_token_version_instant_kill` + `_deactivate_instant_access_kill`.
- [x] **Rate limits** — `auth/ratelimit.py` in-memory fixed-window (off until
      `MMM_RATELIMIT_ENABLED=1`; `Retry-After` via `math.ceil`; periodic bucket
      sweep so it can't grow unbounded). Two layers: **per-org** (`_rl_chat` on
      `/chat`; `_rl_heavy` on the 3 design jobs + `/kb` ingest + `/load-model` +
      `/upload` + `branding/extract`; dev principal never limited) **and
      per-IP** (`require_ip_rate_limit` on the unauthenticated `/auth/signup`
      `/login` `/refresh` `/password-reset/request` — the brute-force surface the
      per-org limiter structurally can't cover). Tests: `tests/test_auth_ratelimit.py`
      (per-org isolation, 429, dev/disabled bypass, IP brute-force throttle).
      *Single-process scope — Redis-back for multi-worker (documented); `X-Forwarded-For`
      honored only via `MMM_RATELIMIT_TRUST_FORWARDED`.* Billing quotas → Track 3.
- [x] **Adversarial review pass** (3-lens workflow) — token-version logic verdict
      *sound*; it caught + we fixed the **HIGH** gaps (unauthenticated auth routes
      had no throttle; `/kb`+`/load-model` missed the heavy limit) plus the
      `Retry-After: 0` boundary bug, the unbounded bucket dict, and the blocking
      per-request DB read (now `asyncio.to_thread`-offloaded in `deps.py`).
- [x] **Ship the kernel sandbox image** — `make kernel-lock|image|verify|push`
      targets + `deploy/kernel/README.md` runbook (build → verify under real
      sandbox flags → push → `MMM_AGENT_HOSTED=1` + `MMM_KERNEL_IMAGE`). The
      Containerfile + k8s manifests (gVisor runtimeclass, egress-deny netpol,
      warm-pool) already existed; build/push itself is a deploy-env step.

---

## Track 2 — Security & trust collateral (buyer-facing)  `[x]`
Authored by an inventory→author→adversarial-verify workflow (22-control verified
inventory; every claim cites file:line and honors its status). The verify pass
caught + fixed a real overclaim (audit sink "must be installed" → it IS
auto-installed at `main.py:167-169`). I then reconciled the docs with the
concurrently-shipped Phase 1.4 (the inventory predated it): updated all
"no token revocation" / RBAC-attribution-only claims across 4 files to reflect
refresh-token revocation/rotation/logout, org-role 403 gating, and the account
lifecycle. HTML well-formed (0 unclosed), docs-snippet gate 74/74, nav wired.
- [x] `docs/trust.html` — flagship buyer page (complements the existing
      `docs/security.html`; cross-linked), status pills, shared-responsibility
      table, "Current Limitations & Roadmap" box; in NAV_LINKS + sitemap.
- [x] `technical-docs/data-processing.md` — Data Processing & sub-processors
      (LLM providers applicable-only-if-configured; local-only via LM Studio).
- [x] `technical-docs/soc2-readiness.md` — TSC control map, **readiness not
      certified**, honest gap list.
- [x] `technical-docs/security-questionnaire.md` — SIG-lite Q&A boilerplate.
- [x] `docs/.well-known/security.txt` (RFC 9116) + `docs/responsible-disclosure.html`.
- [ ] Open item for the user: `security.txt` Contact is the git email
      (`m.reda94@gmail.com`) — swap for a dedicated `security@` alias before
      publishing. The collateral cites source file:line (NDA/reviewer-grade);
      consider a lighter first-touch handout for prospects.

## Track 3 — Pricing & packaging (strategy)  `[ ]`
- [ ] Tier definition (e.g. Team / Business / Enterprise) + feature gating map
- [ ] Per-fit cost model (PyMC compute is the COGS driver) → margin guardrails
- [ ] Metering: count fits / models / seats per org for billing
- [ ] Pricing page + one-page packaging doc

---

## How to enable (once 1.2/1.3 land)

```bash
export MMM_AUTH_ENABLED=1
export MMM_AUTH_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
# optional one-shot admin bootstrap:
export MMM_AUTH_BOOTSTRAP_ORG="Acme"
export MMM_AUTH_BOOTSTRAP_EMAIL="admin@acme.com"
export MMM_AUTH_BOOTSTRAP_PASSWORD="<strong-password>"
```

When `MMM_AUTH_ENABLED` is unset/false, `get_current_principal` returns a
single-tenant dev principal so existing local workflows keep working unchanged.

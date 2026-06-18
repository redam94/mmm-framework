# Security Questionnaire (SIG-Lite) — Boilerplate Answers

> **Status: internal sales-enablement boilerplate.** This is a pre-written Q&A to answer
> the common Standardized Information Gathering (SIG-lite) style security questionnaires
> from prospects. Every answer is drawn from the platform's **code-verified control
> inventory**; where the honest answer is "not yet" or "deployment-dependent," it says so.
> **Do not overclaim.** Companion documents: `technical-docs/soc2-readiness.md`
> (SOC 2 readiness map, **not** a certification) and `docs/security.html` (the public,
> file-cited security posture page).

> **Read-this-first framing.** The MMM Framework is **self-hosted software**, not a
> vendor-operated SaaS. There is no service we run on a customer's behalf. Therefore many
> answers below are **deployment-dependent** — the control exists in the software but is
> supplied/toggled by *your* deployment (e.g., TLS, disk encryption, SSO). The platform
> ships two named postures: **single-user development (default, no auth)** and **hosted
> multi-user (`MMM_AGENT_HOSTED=1`, fail-closed sandbox)**. Production answers assume the
> hosted posture behind a customer-operated perimeter.

---

## A. Access Control & Authentication

**A1. Do you enforce user authentication?**
**Partial / deployment-dependent.** Authentication is **OFF by default**
(`MMM_AUTH_ENABLED=False`, `auth/config.py:30`); the default install injects a single-tenant
dev principal (OWNER) that bypasses tenant scoping (`deps.py:50-57`). When enabled
(`MMM_AUTH_ENABLED=1` + `MMM_AUTH_SECRET` set, enforced by `require_secret()`,
`config.py:59-67`), the platform issues stateless **HS256 JWT** access/refresh tokens with
verified signature/`exp`/`nbf`/audience/issuer and `org`/`role`/`sub`/`jti` claims
(`auth/tokens.py:42-121`; 1 h access / 14 d refresh, `config.py:38-39`). The agent API
itself carries no per-route login in the default posture; production deployments must front
both APIs with their own perimeter (SSO/OIDC reverse proxy).

**A2. How are passwords stored?**
**scrypt** (memory-hard KDF, n=2¹⁵, r=8, p=1, 16-byte random salt) in a self-describing
`scrypt$n$r$p$salt$hash` encoding; verification is **constant-time** via `hmac.compare_digest`
and never raises on malformed input (`auth/passwords.py:26-29`, `:46-85`). *Note:* the
unknown-user branch returns `False` immediately (`service.py:74`), leaving a minor
user-enumeration timing residual.

**A3. Do you support SSO / SAML / OIDC?**
**Not yet (roadmap).** The OIDC/RS256 external-IdP path is an explicit `NotImplementedError`
stub (`tokens.py:149-166`). Today, SSO is supplied by the deployer's reverse proxy in front
of the platform.

**A4. Do you support MFA?**
**No (not implemented).** MFA must be provided by the deployer's perimeter/IdP.

**A5. Is role-based access control (RBAC) enforced?**
**Implemented (auth-enabled).** Guarded routes require a minimum **org role**
(viewer < analyst < admin < owner) and return **403** on insufficient role
(`auth/deps.py` `require_project_access` / `require_org_role`): reads need viewer, writes
need analyst, and admin operations (invites, member management, deactivation) need
admin/owner. The enforced gate is the org role carried in the token; the legacy
product-level owner/analyst/viewer roster labels remain attribution-only. Per-resource
ACLs below the org/role tier are a deferred workstream.

**A6. Can you revoke an active session/token?**
**Partial — refresh tokens yes, access tokens by expiry.** Refresh tokens are **single-use**:
`/auth/refresh` revokes the presented token (rotation) and `/auth/logout` revokes on demand,
backed by a `revoked_tokens` denylist (`auth/service.py`, `auth/store.py`); deactivating a
user blocks all subsequent refreshes. Access tokens are stateless and remain valid until
their short TTL expires (≤ 1 h) — instant access-token kill (a `token_version` claim) is the
remaining roadmap item.

**A7. How do you prevent cross-tenant data access (IDOR)?**
**Implemented (auth-enabled).** `ensure_project_access` returns **404 (not 403)** when a
principal's org does not own a project — preventing existence-probing — then 403 on
insufficient role (`auth/deps.py:119-138`). Storage records are stamped with `org_id` and all
list queries are org-scoped with a per-record `assert_org_owns` 404 on mismatch
(`api/storage.py:637-657`). *Caveat:* isolation is **enforced in application code**, not by a
database FK/ACL — the store is single-file SQLite (WAL); `org_id` lives in plaintext JSON
sidecars. In the **hosted profile**, the server-minted `thread_id` is treated as a bearer
capability and the API refuses guessable/client-invented/unknown thread_ids
(`api/main.py:450-473`).

**A8. How are accounts provisioned and de-provisioned?**
**Implemented (auth-enabled).** Owners/admins invite users with **single-use, expiring**
invite tokens (`/auth/invite` → `/auth/accept-invite`); self-service password reset uses
single-use expiring tokens delivered out-of-band (`/auth/password-reset/request` returns
**202 without revealing whether the email exists**, then `/auth/password-reset/confirm`); and
`/auth/users/{id}/deactivate` (admin-only, same-org) disables an account so its logins and
token refreshes fail immediately. Each action emits an `auth.*` audit event tagged with
`user_id`/`org_id`. *Not yet:* periodic access reviews and an automated joiner/mover/leaver
sync from an external IdP.

---

## B. Encryption

**B1. Is data encrypted in transit?**
**Deployment-dependent — not provided by the platform.** The application does not terminate
TLS; the deployer supplies a TLS-terminating reverse proxy. (CORS `allow_credentials` defaults
`True` with localhost dev origins, `api/config.py:64-72` — tighten for production.)

**B2. Is data encrypted at rest?**
**Deployment-dependent — not provided by the platform.** Datasets, models, the SQLite
sessions store, KB text/embeddings, and `org_id` sidecars are written as **plaintext on disk**.
At-rest encryption must come from the deployer's encrypted volumes / full-disk encryption.

**B3. How are application/credential secrets handled?**
**Implemented.** **No credentials are hardcoded** (a repo grep for high-entropy secret literals
returned no matches). LLM/provider keys, cloud creds, S3/Redis secrets, and the JWT signing
secret are read from **environment / ADC only** via pydantic `BaseSettings`
(`api/config.py:33-44`, `auth/config.py:34`). Secrets may load from a `.env` file if present
(`config.py:16-20`) — that file must be protected. Subprocess/container code-execution kernels
launch with a **fail-closed allowlisted, secret-scrubbed environment**: a denylist strips
anything ending in `_API_KEY`/`_TOKEN`/`_SECRET`/`_CREDENTIALS`/`_PASSWORD`/`_PRIVATE_KEY` plus
`GOOGLE_APPLICATION_CREDENTIALS` and `MMM_LLM_*`, and the denylist wins over any passthrough
(`agents/kernels.py:317-357`). *Caveat:* the in-process kernel (dev default) shares the full
API env — no scrub possible; env-scrub does not block the cloud metadata server (that is an
egress control).

**B4. What key-management / TLS-cert process do you use?**
**Deployer-owned.** The platform does not manage TLS certs or a KMS; the symmetric JWT secret
is provided via env. There is **no built-in automated key rotation.**

---

## C. Data Handling, Storage & Residency

**C1. Where is customer data stored? Do you operate a SaaS?**
**No SaaS.** The platform is self-hosted; **all state lives on infrastructure the customer
controls.** Per-session outputs under `$MMM_AGENT_WORKSPACE/threads/<thread_id>/`, project KB
sources under `.../projects/<project_id>/kb/`, dataset uploads under `uploads/`. KB text and
embedding vectors are stored locally in the SQLite sessions store — **not** in any external
vector database (`docs/security.html` §6).

**C2. What customer data leaves our network, and to whom?**
Only the **configured LLM/embedding traffic** to the provider *you* choose (Anthropic, OpenAI,
Google/Vertex), or **nothing** if you configure the fully-local `lmstudio` provider for both
chat and embeddings (`agents/llm.py`, `agents/embeddings.py`). Important nuance: raw datasets
are **not** uploaded by design, but whatever the agent *prints or reads into model context*
(e.g., a `df.head()`, summary stats, a ≤20 KB file excerpt) **is** sent to the LLM provider as
a tool result. There is **no telemetry, license-check, or auto-update call** in the codebase.

**C3. Is captured/agent output treated as untrusted?**
**Implemented.** Captured Plotly figures and tabular tool output are schema-filtered to an
allowed-key set, hard size-capped (`MMM_PLOT_MAX_BYTES` 5 MiB, `MMM_TABLE_MAX_BYTES` 1 MiB),
and content-addressed with IDs **salted by `thread_id`** so identical payloads don't dedup or
become guessable across sessions (`agents/workspace.py:77-158`). *Caveat:* the salted ID is the
**only** capability for `GET /plots/{id}` and `GET /tables/{id}` — there is no separate
per-tenant ACL on those endpoints.

**C4. Is file download path-traversal / symlink safe?**
**Implemented.** File serving is TOCTOU-safe and traversal-guarded: the resolved path must be
inside an allowlisted root; the realpath is opened `O_NOFOLLOW|O_CLOEXEC` (rejecting a symlinked
final component); the fd is confirmed a regular file; the response streams from that exact
validated fd (`api/main.py:2723-2792`; `workspace.py:202-226`). *Caveat:* a narrow
parent-directory-swap race is acknowledged as not fully closed without a read-only mount
namespace — defense-in-depth.

**C5. Do you protect against SSRF on server-side fetches?**
**Implemented.** The only server-side outbound fetch (website brand extraction) is SSRF-guarded:
http/https + ports 80/443 only, no embedded credentials, **every resolved DNS address must be
globally routable** (rejects loopback, RFC 1918, link-local `169.254` metadata, ULA, multicast,
reserved), redirects manually re-vetted (max 3, never auto-followed), with a byte cap and
content-type check (`agents/brand_extract.py:48-143`). Runs host-side only, never in the kernel;
**disabled in the hosted profile unless `MMM_BRAND_FETCH_ALLOW=1`.** *Caveat:* a documented
residual DNS-rebind window exists (address check just before connect; httpx re-resolves).

**C6. What is your data retention / disposal policy?**
**Partial.** A `data_retention_days` setting (default 30) drives `cleanup_old_jobs`, which
purges **Redis job records** older than the cutoff (`api/config.py:52`, `api/worker.py:1268-1292`).
It does **NOT** delete stored datasets/models/results on disk — those are deleted **on demand**
via `delete_*` methods. There is no automated on-disk data-disposal control today.

**C7. Do you support an S3 / object-storage backend?**
**Not yet (roadmap).** An `s3` backend is **declared in settings** (`api/config.py:37-44`) but
the `StorageService` implements **only the local filesystem path** — no S3 code path is present
(`api/storage.py`). Storage today is local disk.

**C8. Is the code-execution sandbox isolated from data and network?**
**Deployment-dependent.** The **container kernel** tier provides drop-ALL-capabilities,
no-new-privileges, read-only rootfs, cgroup memory/pids/CPU caps, ulimits, a tmpfs-only writable
area, and **egress denied by default** (`--network none` on the production ipc transport),
with a fail-closed gate that refuses to spawn an incomplete sandbox
(`agents/container_kernel.py:154-403`). **This is active only when `MMM_AGENT_KERNEL=container`
with a built image + runtime present.** The default `inprocess` kernel has **no sandbox**, and
`subprocess` is **also not sandboxed**. On macOS dev boxes the tcp transport leaves egress open
and unenforced. The hosted profile (`MMM_AGENT_HOSTED=1`) force-upgrades to `container` and
refuses to start otherwise.

---

## D. Sub-Processors / Third Parties

**D1. List your sub-processors.**
The platform operates **no sub-processors of its own** (no SaaS). The data-flow third parties
are the **customer-selected LLM/embedding providers**, configured per deployment:

| Provider (configurable) | Used for | Data sent | Customer can avoid? |
|---|---|---|---|
| Anthropic / OpenAI / Google (direct API) | Chat completion / embeddings | Chat + tool-result context; KB chunks at ingest | Yes |
| Google Vertex AI (`vertex_*`) | Chat / embeddings via ADC | Same, to the customer's own GCP project | Yes |
| **LM Studio (local)** | Fully-local chat + embeddings | **Nothing leaves the host** | — (this *is* the no-egress option) |

The customer chooses and contracts directly with these providers; the MMM Framework does not
manage those relationships or DPAs.

**D2. Do you have a vendor/sub-processor risk-management program?**
**No (gap).** There is no formal sub-processor inventory, DPA governance, or vendor security
review process shipped with or operated by the platform. This is the deployer's responsibility.

---

## E. Incident Response & Logging

**E1. Do you maintain security audit logs?**
**Implemented (with limits).** Security/lifecycle events on the `mmm_audit` logger are written
as a **tamper-evident hash-chained JSONL**: each record's hash = `sha256(prev_hash +
canonical(record))`, so any edit/delete/reorder breaks the chain and is detected by `verify()`;
the chain resumes across restarts (`agents/audit_sink.py:45-157`). Events include kernel
spawn/evict/death/timeout, egress posture, sandbox refusals, overlay wipes, and plot rejections.
*Caveat:* **tamper-EVIDENT, not tamper-PROOF** — the JSONL is local; true tamper resistance needs
the documented off-host shipper (roadmap). The sink **is installed by the API app at startup**
via `install_audit_sink()` (FastAPI lifespan, `src/mmm_framework/api/main.py:167-169`); the
install is **best-effort** (failure logged and swallowed) and runs only when the API process
boots — a library/standalone use that never starts the API gets no sink.

**E2. Do you have SIEM integration / real-time alerting / intrusion detection?**
**Not yet (roadmap).** No SIEM integration, alerting, IDS/IPS, or centralized log aggregation is
shipped. The audit chain is local and detects tampering after the fact; an off-host shipper to
durable storage is planned. Deployers should forward `mmm_audit` to their own SIEM.

**E3. Do you have a documented Incident Response plan?**
**No (gap).** No formal IR policy, runbook, breach-notification SLA, or on-call process is
shipped or operated. As self-hosted software, incident response is the deployer's responsibility.

**E4. Do you perform vulnerability scanning / penetration testing?**
**Partial / not formalized.** The security-sensitive subsystems (kernel sandbox, egress, SSRF,
file serving) were designed and **adversarially reviewed**, with documented exit tests
(`technical-docs/agent-session-kernels*.md`). However, there is **no documented recurring
vulnerability-scan or third-party pen-test cadence.**

---

## F. Business Continuity & Disaster Recovery (BCP/DR)

**F1. Do you have a documented BCP/DR plan with RTO/RPO?**
**No (gap).** No documented RTO/RPO, no tested DR runbook. As self-hosted software, continuity
is the deployer's responsibility.

**F2. Do you provide backup and restore tooling?**
**No (gap).** No shipped backup/restore tooling. State lives on local disk (SQLite sessions
store, workspace files, uploads) and Redis; the deployer must back these up using their own
infrastructure tooling.

**F3. Is the platform highly available / fault tolerant?**
**Partial / deployer-owned.** Long-running model fits are decoupled via an async ARQ/Redis job
queue, and per-session kernels have resource caps and an LRU cap to contain a single tenant's
resource use (`agents/container_kernel.py`, `agents/kernels.py`). However, **Redis and the
single-file SQLite store are single points of failure** with no shipped HA/clustering/failover
configuration — the deployer must make them redundant.

---

## G. Secure Software Development Lifecycle (SDLC)

**G1. Do you follow a secure SDLC?**
**Partial.** Security-sensitive features are designed against an adversarially-verified spec, the
full security posture is documented and file-cited (`docs/security.html`,
`technical-docs/agent-session-kernels*.md`), code review is practiced, and a docs/code-snippet
gate exists in CI. **But** there is no formal, evidenced secure-SDLC *program*: no documented
change-approval workflow tied to a ticketing system, no mandated SAST/dependency-scanning gate,
and no release sign-off control. We do not overclaim a certified SDLC.

**G2. Are dependencies and third-party libraries managed?**
**Partial.** Dependencies are pinned and managed via `uv` (lockfile-based), and the framework's
core stack (PyMC, NumPyro, FastAPI, Pydantic, etc.) is version-pinned. There is **no shipped,
automated CVE/dependency-vulnerability scanning gate** documented.

**G3. Are secrets kept out of source control?**
**Yes.** No hardcoded credentials in the repo (verified by grep); all secrets are read from env
/ ADC / a protected `.env` file (`api/config.py`, `auth/config.py`).

---

## H. Summary Posture Statement (drop-in for questionnaire cover pages)

> The MMM Framework is **self-hosted software**, not a vendor-operated SaaS, so the security
> boundary is the customer's own deployment. The platform ships a substantial set of
> **code-verified technical controls** — scrypt password hashing, IDOR-resistant (404) org
> tenant isolation, an HS256 JWT session model, a fail-closed container kernel sandbox with
> default-deny egress, secret scrubbing for code-execution kernels, SSRF-guarded server-side
> fetch, TOCTOU-safe file serving, size-capped/thread-salted agent output, and a tamper-evident
> hash-chained audit log. Several of these are **deployment-dependent** (the sandbox requires a
> built image + runtime; encryption in transit/at rest is supplied by the deployer's TLS and
> encrypted volumes) and the **default install is single-user with no authentication.**
> The platform has **not** undergone a SOC 2 examination and is **not certified**; a SOC 2
> readiness control map is available (`technical-docs/soc2-readiness.md`). Known gaps a buyer
> should account for: SSO/MFA, instant access-token kill (refresh-token revocation ships;
> access tokens expire by short TTL), S3 storage, on-disk data-disposal automation,
> SIEM/alerting, formal IR/BCP/DR plans, sub-processor governance, and a formalized secure-SDLC
> program — all either roadmap items or controls the deployer owns.

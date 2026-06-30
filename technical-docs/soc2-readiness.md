# SOC 2 Readiness Map — MMM Framework

> **Status: READINESS, NOT CERTIFICATION.** The MMM Framework has **not** undergone
> a SOC 2 Type I or Type II examination. No auditor has issued a report, and no SOC 2
> certificate exists. This document is internal sales-enablement collateral that maps
> the platform's **existing, code-verified technical controls** to the relevant SOC 2
> Trust Services Criteria (TSC), and lists — honestly — what is still missing before
> an actual audit could be attempted. Every "implemented" claim below cites a specific
> source file. Where a control is deployment-dependent or absent, it says so.
>
> Use this with prospects who ask "are you SOC 2 compliant?" The honest answer is:
> *"We are self-hosted software, so SOC 2 applies to **your** deployment, not a vendor
> SaaS we operate. The platform ships technical controls that support several Security,
> Availability, and Confidentiality criteria; we have not pursued an audit, and we are
> transparent about the policy, monitoring, and vendor-management gaps that a real
> examination would require."*

---

## 0. How to read this — the deployment model changes everything

The MMM Framework is **self-hosted software**, not a vendor-operated SaaS. There is no
multi-tenant service that Anthropic/Matthew Reda operates on customers' behalf. As a
consequence:

- **The SOC 2 "service organization" is the customer's own deployment**, on
  infrastructure the customer controls. The platform contributes technical controls;
  the customer owns the operational, policy, and monitoring controls that a SOC 2
  examination scopes.
- **Security posture is a function of how you deploy it.** The codebase exposes two
  named postures (see `docs/security.html` and `technical-docs/agent-knowledge-workspace.md` §11):
  - **Single-user development (default).** No auth, in-process kernel, workspace is an
    organizational boundary, not a security one. **Not suitable for a SOC 2 scope.**
  - **Hosted multi-user (`MMM_AGENT_HOSTED=1`).** Fail-closed onto container-sandboxed
    kernels, denied egress, server-minted session IDs. This is the posture against which
    the technical controls below are meaningful.
- **Many controls are "deployment-dependent."** Encryption at rest/in transit, the
  kernel sandbox, and tenant isolation are all toggled or supplied by the deployment,
  not baked into a single shipped binary. A readiness assessment must state the assumed
  posture; ours assumes the hosted profile plus a customer-operated TLS terminator and
  encrypted volumes.

> **One-line framing for sales:** *"SOC 2 readiness here means: the platform gives you
> a strong set of technical controls to inherit; you (the deployer) still own the policy,
> monitoring, and vendor-management controls a SOC 2 auditor will test. We have not been
> audited."*

---

## 1. Security (Common Criteria) — control mapping

The SOC 2 **Security** category (Common Criteria, CC1–CC9) is the mandatory core of any
SOC 2 examination. Below, each relevant criterion is mapped to the **existing** controls
from the verified inventory, with status and the evidence file.

### CC6.1 — Logical access: identification, authentication, credential protection

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| Passwords hashed with **scrypt** (memory-hard KDF, n=2¹⁵, r=8, p=1, 16-byte random salt), self-describing `scrypt$n$r$p$salt$hash` encoding; verification is **constant-time** via `hmac.compare_digest` and never raises on malformed input | **Implemented** | `src/mmm_framework/auth/passwords.py:26-29` (params), `:46-61` (hash), `:64-85` (verify, compare_digest at `:85`) | `verify_password` returns `False` immediately when the stored hash is `None`/empty (the unknown-user branch, `service.py:74`), so timing is **not fully equalized** on unknown users despite the "constant-ish work" comment (`service.py:73`) — a minor user-enumeration timing residual. |
| Stateless **HS256 JWT** access/refresh tokens; decode verifies signature (constant-time), `exp`/`nbf`/audience/issuer; claims carry `org`/`role`/`sub`/`jti`; 1 h access / 14 d refresh TTL | **Implemented** | `auth/tokens.py:42-94` (encode/decode, compare_digest `:77`, claim checks `:86-93`), `:97-121` (make_claims), `config.py:38-39` (TTLs) | **HS256 symmetric secret only.** OIDC/RS256 external-IdP is an explicit `NotImplementedError` stub (`tokens.py:149-166`). |
| **Refresh-token revocation + single-use rotation**: `/auth/refresh` revokes the presented token and mints a new pair; `/auth/logout` revokes on demand; deactivated users fail refresh — backed by a `revoked_tokens` denylist | **Implemented** | `auth/service.py` (`refresh_tokens`/`logout`/`deactivate_user`), `auth/store.py` (`revoke_token`/`is_token_revoked`, `revoked_tokens` table) | Refresh tokens are revocable immediately. **Access tokens are stateless** and remain valid until their short TTL (≤ 1 h) — instant access-token kill (a `token_version` claim) is the remaining roadmap item. |
| **Account lifecycle**: org invites (single-use, expiring), self-service password reset (single-use tokens, no email enumeration), and admin user-deactivation — all org-scoped and audit-logged | **Implemented** | `auth/routes.py` (`/auth/invite`,`/accept-invite`,`/password-reset/*`,`/users/{id}/deactivate`), `auth/service.py`, `auth/store.py` (`invites`/`password_resets` tables) | Invite/reset tokens are delivered **out-of-band** (no email integration bundled — the deployer wires delivery). |
| No hardcoded credentials; LLM/provider keys and cloud creds read from **environment / ADC only** | **Implemented** | `api/config.py:33-44` (S3/Redis secrets via pydantic `BaseSettings`), `auth/config.py:34` (signing secret from env); repo grep for hardcoded high-entropy secrets returned no matches | Secrets load from a `.env` file if present (`config.py:16-20`, `auth/config.py:24`) — that file must be protected by the deployer. CORS `allow_credentials` defaults `True` with localhost dev origins (`api/config.py:64-72`) — tighten for production. |

> **Gap for CC6.1:** Authentication is **OFF by default** (`MMM_AUTH_ENABLED=False`,
> `auth/config.py:30`); the default install injects a single-tenant dev principal
> (`deps.py:50-57`, OWNER) that bypasses tenant scoping. **No MFA, no SSO/OIDC, no
> password-policy enforcement, no automated credential rotation.** A SOC 2 scope requires
> auth enabled (`MMM_AUTH_ENABLED=1` + `MMM_AUTH_SECRET` set, enforced by
> `require_secret()`, `config.py:59-67`) **plus** the deployer supplying SSO/MFA at a
> reverse-proxy perimeter.

### CC6.1 / CC6.6 — Boundary protection & cross-tenant access control

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| Cross-tenant access returns **404, not 403** (IDOR-resistant, no project-existence probing); `ensure_project_access` 404s unless the principal's org owns the project, then 403 on insufficient role | **Implemented** | `auth/deps.py:119-138` (404 `:133`, 403 `:135`), `:141-157` (dependency factory) | Enforcement is **gated on auth being enabled**; the dev principal short-circuits (`deps.py:129-130`). Correctness depends on **each route mounting the dependency** — 22 enforcement call-sites in the agent API plus the root Models API routes. |
| Storage records **stamped with `org_id`**; list queries org-scoped; `assert_org_owns` 404s on mismatch; legacy records default to `DEFAULT_ORG_ID` | **Implemented** | `api/storage.py` save-stamping (`:107-110`,`:181-182`,`:424-427`,`:505-508`), org-scoped list filters, `org_scope`/`assert_org_owns` (`:637-657`, 404 `:654-657`), `DEFAULT_ORG_ID` `:26`, backfill `:660-696` | Org scoping activates **only when a non-None org is passed** (non-dev principal). `org_id` is stored in **plaintext JSON sidecar files**, not a DB-enforced foreign key — isolation is **application-code-enforced, not datastore-enforced**. |
| Hosted profile treats `thread_id` as a **bearer capability**: refuses the guessable default and any client-invented/unknown thread_id, requires a server-minted session, refuses to drive another org's session | **Deployment-dependent** | `src/mmm_framework/api/main.py:450-465` (reject non-server-minted, 403), `:467-473` (`ensure_project_access` on existing session) | Only applies when `MMM_AGENT_HOSTED=1`. In the dev posture, thread_ids are accepted and auto-created. |
| Org/user/membership schema is **additive & idempotent** (`organizations`, `org_members` tables; users gain `password_hash`/`org_id`/`status`; projects gain indexed `org_id`); one-time backfill attaches orphans | **Implemented** | `auth/store.py:60-126` (idempotent ALTERs), `:131-149` (org creation, slug-collision suffix), `:298-345` (default-org + orphan attach) | Backed by **SQLite (`sessions.db`, WAL mode)** — single-file DB, **no row-level DB ACLs**; isolation is enforced in application code. |

> **Gap for CC6.6:** Tenant isolation is **enforced in application code, not the datastore**,
> and is **inert in the default posture**. A SOC 2 examination would require a documented,
> tested boundary with periodic access reviews — neither exists today.

### CC6.7 — Data transmission & code-execution boundary controls

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| **Container kernel sandbox**: drop-ALL-capabilities, no-new-privileges, read-only rootfs, cgroup memory(+swap)/pids/CPU caps, nofile/nproc ulimits, exec-able tmpfs as the only writable area; podman default seccomp | **Deployment-dependent** | `agents/container_kernel.py:202-221` (resources), `:223-234` (security), `:236-244` (tmpfs) | **Active only when `MMM_AGENT_KERNEL=container` AND a built image + container runtime are present.** Default kernel is `inprocess` (**no sandbox**); `subprocess` is **also not sandboxed** (`profile.py:26` `SANDBOXED_IMPLS={'container'}`). Containerfile cap-drop/read-only are applied by the provisioner **at run time**, not baked into the image. |
| **Fail-closed isolation gate**: refuses to spawn unless read-only rootfs, cap-drop ALL, a memory cap, and denied egress are all present; auto-required in the hosted profile | **Deployment-dependent** | `agents/container_kernel.py:373-403` (`_verify_isolation`), `:379-381` (triggered by hosted or `MMM_KERNEL_REQUIRE_SANDBOX`) | Inert unless hosted or `MMM_KERNEL_REQUIRE_SANDBOX` set. Verifies the **command string it builds**, not the runtime's actually-applied state — defense-in-depth, not a kernel-level proof. |
| **Hosted profile** is a single fail-closed switch: force-upgrades kernel to `container`, requires a complete sandbox, denies egress, drops CWD from download roots, refuses to start on a non-sandboxed kernel | **Deployment-dependent** | `agents/profile.py:29-52`, `:55-65` (`assert_hosted_sandbox`), `src/mmm_framework/api/main.py:156-158` (boot guard) | **Deliberately inert until the Tier-2 container sandbox image/runtime exists.** Setting `MMM_AGENT_HOSTED=1` without the image causes a **refuse-to-start by design.** |
| **Container egress deny by default**: ipc transport uses `--network none` (no metadata server, no egress); Linux tcp uses an `--internal` network; egress opened only via `MMM_KERNEL_EGRESS=open` | **Deployment-dependent** | `agents/container_kernel.py:154-167` (`--internal`), `:169-192` (`--network none` for ipc, default `'deny'`) | **macOS dev boxes leave tcp egress OPEN and unenforced** (`:187-190`, posture `open:unenforced-macos-dev`). Egress-deny applies **only to the container kernel** — inprocess/subprocess have full host network. |
| **SSRF-guarded** brand-extraction fetch: http/https + ports 80/443 only, no embedded creds, every resolved DNS address must be globally routable (rejects loopback/RFC1918/link-local 169.254 metadata/ULA/multicast/reserved), ≤3 manually re-vetted redirects (never auto-followed), byte cap, content-type check | **Implemented** | `agents/brand_extract.py:48-83` (`_assert_public_http`), `:99-143` (`_fetch`) | Documented residual **DNS-rebind window** (address check just before connect; httpx re-resolves, `:14-17`). Runs **host-side only**, never in the kernel. Disabled in the hosted profile unless `MMM_BRAND_FETCH_ALLOW=1`. |
| **File-download serving** is TOCTOU-safe & path-traversal-guarded: resolved path must be inside an allowlisted root; realpath opened `O_NOFOLLOW|O_CLOEXEC` (rejects symlinked final component); fd confirmed a regular file; response streams from that exact fd | **Implemented** | `src/mmm_framework/api/main.py:2723-2761` (`_safe_open_within`), `:2764-2792` (`_iter_fd`/`_safe_serve`); `workspace.py:202-226` (traversal guards) | The narrow **parent-directory-swap race** is acknowledged as not fully closed without the Tier-2 read-only mount namespace (`src/mmm_framework/api/main.py:2732-2733`) — defense-in-depth. `O_NOFOLLOW` falls back to `0` on platforms lacking it. |
| **Subprocess/container kernels launch with a fail-closed allowlisted env**: regex denylist strips anything ending in `_API_KEY`/`_TOKEN`/`_SECRET`/`_CREDENTIALS`/`_PASSWORD`/`_PRIVATE_KEY` plus `GOOGLE_APPLICATION_CREDENTIALS` and `MMM_LLM_API_KEY`/`CREDENTIALS_PATH`; denylist wins over passthrough | **Implemented** | `agents/kernels.py:317-321` (`_ENV_DENY_RE`), `:327-357` (`_scrubbed_kernel_env`), `:573-576` (spawn); `container_kernel.py:140-152` (tighter container allowlist) | Opt-out exists (`MMM_KERNEL_SCRUB_ENV=0` → full `os.environ` inherited). **Env-scrub does NOT block the cloud metadata server** (ADC theft) — that is an egress control. The **in-process kernel shares the API process env entirely** (no scrub possible). |

> **Gap for CC6.7:** Encryption **in transit** (TLS) is **not provided by the platform** —
> the deployer must terminate TLS at a reverse proxy. The strongest code-execution controls
> (container sandbox, egress deny) are **deployment-dependent and inert by default**; on
> macOS dev they are explicitly unenforced.

### CC7.2 / CC7.3 — Monitoring, anomaly detection, and event logging

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| Security/lifecycle events on the `mmm_audit` logger written as a **tamper-evident hash-chained JSONL**: each record's hash = `sha256(prev_hash + canonical(record))`, so any edit/delete/reorder breaks the chain and is caught by `verify()`; chain resumes across restarts | **Implemented** | `agents/audit_sink.py:45-46` (`_chain_hash`), `:49-106` (`emit`, chain `:98`), `:131-157` (`verify`), `:59-77` (resume) | **Tamper-EVIDENT, not tamper-PROOF**: the JSONL is local; the chain only makes on-host pre-shipping tampering **detectable**. True tamper resistance needs the documented **off-host shipper to durable storage** (not yet built, `audit_sink.py:7-9`). The sink **is installed by the API app at startup** via `install_audit_sink()` (called from the FastAPI lifespan, `src/mmm_framework/api/main.py:167-169`); the install is **best-effort** (a failure is logged and swallowed) and only runs when the API process boots — a library/standalone use that never starts the API gets no sink. |

> **Gap for CC7.x:** There is **no SIEM integration, no real-time alerting, no intrusion
> detection, and no centralized log aggregation** shipped. The audit chain is local and
> only detects tampering *after* the fact; an off-host shipper is roadmap. A SOC 2 scope
> requires continuous monitoring and a documented incident-detection process — both are
> the deployer's responsibility and not provided.

---

## 2. Availability (A-series criteria) — control mapping

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| **Kernel resource caps** prevent a single session from exhausting the host: container cgroup memory (default 2 GB) / pids / CPU caps + ulimits; per-cell wall-clock timeout (`MMM_CELL_TIMEOUT`, default 600 s; SIGINT → kill); live-kernel LRU cap (`MMM_MAX_KERNELS`, default 8) | **Deployment-dependent / Implemented** | `agents/container_kernel.py:202-221` (cgroup caps, container tier); `agents/kernels.py` (cell timeout + LRU cap, subprocess tier) | cgroup caps apply **only to the container tier**; the in-process kernel has **no caps** and can exhaust the API process. Timeout/LRU apply to subprocess/container. |
| **Async job queue** decouples long fits from request threads (ARQ on Redis) | **Implemented** (architectural) | `api/worker.py`, `jobs.py` | Redis is a **single point of failure** the deployer must make HA; no shipped clustering/failover config. |
| **Storage retention cleanup**: `data_retention_days` (default 30) drives `cleanup_old_jobs`, purging Redis job records older than the cutoff | **Partial** | `api/config.py:52` (retention), `api/worker.py:1268-1292` (cleanup) | Retention currently purges **only Redis JOB records** — it does **NOT** delete stored datasets/models/results on disk. Data/model deletion is on-demand via `delete_*` methods. |

> **Gap for Availability:** There is **no shipped backup/restore tooling, no documented
> RTO/RPO, no health-check/SLA monitoring, no autoscaling, and no HA configuration** for
> Redis or the SQLite sessions store (single-file DB). Availability commitments and DR are
> entirely the deployer's responsibility. **No BCP/DR plan exists.**

---

## 3. Confidentiality (C-series criteria) — control mapping

| Control | Status | Evidence | Honest note |
|---|---|---|---|
| **No vendor SaaS** — all state stays on infrastructure the customer controls; only configured LLM/embedding traffic leaves the network (none with a local `lmstudio` provider) | **Implemented** (architectural) | `docs/security.html` §1, §6; `agents/llm.py`, `agents/embeddings.py` | "Raw data stays local" holds **by architecture, not a hard filter**: whatever the agent *prints* or *reads into context* (a `df.head()`, a 20 KB file excerpt) **is sent to the configured LLM provider** as a tool result. Use the fully-local `lmstudio` option or pre-anonymize if that is unacceptable. |
| **Captured plots & tabular tool output treated as untrusted egress**: schema-filtered to an allowed-key set, hard size-capped (`MMM_PLOT_MAX_BYTES` 5 MiB, `MMM_TABLE_MAX_BYTES` 1 MiB), content-addressed with IDs **salted by `thread_id`** so identical payloads don't dedup or become guessable across sessions | **Implemented** | `agents/workspace.py:77-105` (plot store), `:124-158` (table store), `:108-111`/`:161-164` (path lookup) | The **salted ID IS the only capability** for `GET /plots/{id}` and `GET /tables/{id}` — there is **no separate per-tenant ACL** on those endpoints. Size cap **rejects** (raises) rather than truncating. |
| **Secret scrubbing** keeps provider keys and cloud creds out of subprocess/container kernels (see CC6.7) | **Implemented** | `agents/kernels.py:327-357`; `container_kernel.py:140-152` | In-process kernel shares the full API env — no scrub possible. Env-scrub does not block the metadata server. |
| **Storage backend** supports local filesystem (default) with an S3 backend configurable via settings; `data_retention_days` (default 30) | **Partial** | `api/config.py:37-44` (`storage_backend` Literal `['local','s3']`), `:52`; `api/storage.py:48-57` | The **S3 backend is declared in settings but NOT implemented** — `StorageService` read/write all use the local `storage_path`; **no S3 code path is present.** Retention purges only Redis job records, not on-disk data. |

> **Gap for Confidentiality:** **Encryption at rest is NOT provided by the platform** —
> data, models, the SQLite store, and the `org_id` sidecars are written as plaintext on disk;
> at-rest encryption must come from the deployer's encrypted volumes / disk encryption.
> Plot/table endpoints lack a per-tenant ACL beyond the salted capability ID. There is **no
> data-classification scheme, no DLP, and no formal data-disposal control** (retention does
> not delete on-disk data).

---

## 4. The honest gap list — what's missing for an actual SOC 2 audit

A SOC 2 examination tests **operating effectiveness over time**, not just the presence of
technical controls. The following are absent or out of scope today. **None of these should
be represented to a prospect as existing.**

### 4.1 Formal policies & governance (largely absent)
- No written **Information Security Policy**, Access Control Policy, Acceptable Use,
  Change Management, Data Classification/Handling, or Incident Response policy set.
- No **risk assessment** process, risk register, or annual review cadence.
- No defined **roles & responsibilities** for security ownership at the org-governance level.
  (Note: the **org role** carried in the token *is* enforced as access control — guarded routes
  return 403 below the required role; the legacy product-level owner/analyst/viewer roster
  labels remain attribution-only — `docs/platform-overview.html`.)
- No **security awareness training** program or evidence of completion.

### 4.2 Access management (partial controls, no program)
- Authentication is **off by default**; no MFA; no SSO/OIDC (RS256 path is a stub).
- **Refresh-token revocation + rotation, logout, and admin deactivation ship**; the residual
  gap is **instant access-token kill** (stateless access tokens expire by their ≤ 1 h TTL — a
  `token_version` claim would close it).
- Account provisioning/deprovisioning exists (invites, password reset, deactivation) but there
  are **no periodic access reviews**, no automated joiner/mover/leaver IdP sync, and no
  privileged-access management program.

### 4.3 Monitoring & detection (minimal)
- Audit log is **local and tamper-evident only**; no off-host shipper, **no SIEM, no
  alerting, no IDS/IPS, no centralized aggregation.**
- No **vulnerability scanning** or penetration-test cadence documented.

### 4.4 Vendor / sub-processor management (absent)
- No **sub-processor inventory or DPA** governance. The relevant data-flow sub-processors
  are the **customer-chosen LLM/embedding providers** (Anthropic, OpenAI, Google/Vertex,
  or fully-local LM Studio) — the platform does not manage these relationships; the
  deployer does.
- No vendor security review or vendor risk-rating process.

### 4.5 Availability / BCP / DR (absent)
- **No documented RTO/RPO, no backup/restore tooling, no tested DR runbook, no HA config**
  for Redis or the SQLite sessions store.

### 4.6 Secure SDLC (partial)
- Code review and an adversarially-verified security design exist (`technical-docs/agent-session-kernels*.md`),
  and the security posture is fully documented (`docs/security.html`). **But** there is no
  formal, evidenced secure-SDLC *program*: no documented change-approval workflow tied to a
  ticketing system, no mandated SAST/dependency-scanning gate, no release sign-off control.

### 4.7 Encryption (deployment-dependent — not a platform guarantee)
- **At rest:** not provided; relies on the deployer's disk/volume encryption.
- **In transit:** not provided; relies on a deployer-supplied TLS terminator.
- Both must be claimed only as *"deployment-dependent"* — never as a built-in platform feature.

---

## 5. The honest sales script

When a prospect asks about SOC 2:

1. **Lead with the deployment model.** "We're self-hosted software, so SOC 2 scopes *your*
   environment. We ship technical controls you can inherit; we have not pursued a vendor
   SOC 2 examination, and there is no certificate."
2. **Point to the verifiable controls.** Sections 1–3 above are code-cited and real — scrypt
   password hashing, IDOR-resistant 404 tenant isolation, the container kernel sandbox (when
   deployed), SSRF-guarded fetch, TOCTOU-safe downloads, hash-chained audit, secret scrubbing.
3. **State the gaps proactively** (Section 4). This builds credibility; the security-questionnaire
   boilerplate (`technical-docs/security-questionnaire.md`) answers the standard SIG questions
   with the same honesty.
4. **Offer the readiness path, not a certificate.** The deployer enabling auth + hosted profile
   + TLS + encrypted volumes + their own SIEM/SSO closes most technical gaps; the policy,
   monitoring, vendor-management, and DR controls are an organizational program the customer
   owns. We can support a customer's own audit as the software vendor, but we are not the
   audited service organization.

> **Never** say "SOC 2 compliant," "SOC 2 certified," or imply an audit has occurred.
> The accurate phrase is **"SOC 2 readiness: a control map, not a certification."**

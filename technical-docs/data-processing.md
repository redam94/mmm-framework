# Data Processing & Sub-processors

This document summarizes **what data the MMM Framework platform processes, where
it lives, how long it is kept, and which third parties it can call out to**. It
is a factual, code-referenced summary for security and procurement reviewers.

> **Posture matters.** The platform is **self-hosted, open-source software** —
> there is no vendor-operated SaaS. You run it; you control the data residency,
> the storage backend, and which (if any) third-party providers are configured.
> Every control below is honored at its actual implementation status, and
> controls that depend on deployment flags are labelled as such. The companion
> security narrative is `docs/security.html`; the controls inventory it draws
> from is the source of truth for this document.

---

## 1. What data the platform processes

| Data category | What it is | Where it enters |
|---------------|-----------|-----------------|
| **Marketing spend & KPIs** | Per-channel media spend, the modeled KPI (sales/units/etc.), control variables — the core MMM inputs. | Uploaded datasets (MFF / Master Flat File). |
| **Uploaded datasets** | The raw CSV/data files an analyst uploads to fit a model. | Stored on local disk (uploads / the per-session workspace). |
| **Fitted models & results** | Saved model artifacts, posterior summaries, reports, and per-run history metrics. | Written by the storage layer and the workspace. |
| **Chat / agent interaction** | Chat messages, the system prompt, and **tool results** that return into model context (e.g. `stdout` of executed Python, KB snippets, file excerpts). | The agent subsystem; sent to the configured LLM provider per turn. |
| **Knowledge-base documents** | Project KB source documents; their full text is chunked and embedded at ingest. | Sent to the configured **embedding** provider (resolved separately from the chat model). |
| **Accounts & tenancy metadata** | Organizations, users (with `password_hash`), org memberships, and project↔org associations. | SQLite `sessions.db` (WAL mode). |

### What does **not** leave by architecture
Raw datasets are **not** uploaded to any LLM provider as part of normal
operation; analysis runs against local files inside the execution kernel.
Captured Plotly figures and tabular tool output are content-addressed and
streamed to the **browser** (`GET /plots/{id}`, `GET /tables/{id}`), not passed
through model context. This holds **by architecture, not by a hard filter**:
whatever the agent *prints* or *reads into context* (a `df.head()`, summary
stats, a file excerpt) does reach the LLM provider as a tool result. Tool output
is byte-capped and the prompt forbids printing full DataFrames, but sensitive
values surfaced into context will reach the provider. For a no-egress option,
use a fully-local LLM + embedding model (see §4).

---

## 2. Where the data lives (storage backends)

| Backend | Status | Notes |
|---------|--------|-------|
| **Local filesystem** (default) | Implemented | `storage_backend` defaults to `local`; all reads/writes go to `settings.storage_path`. Evidence: `api/config.py:37` (`storage_backend: Literal["local","s3"]`, `storage_path`), `api/storage.py` (local dir provisioning). |
| **S3** | **Declared only — not implemented** | `storage_backend="s3"` and S3 bucket/region/keys exist in settings (`api/config.py:37-44`), but `StorageService` only implements the local path — **there is no S3 code path present**. Treat S3 as a configuration placeholder, not a working backend, until implemented. |
| **Redis** (job queue) | Implemented (required for async jobs) | Holds async job records for model fitting (ARQ). Connection + optional password from settings/env (`api/config.py`). |
| **SQLite `sessions.db`** | Implemented | Backs sessions, organizations/users/memberships, projects (WAL mode; `src/mmm_framework/auth/store.py:21,44`). Single-file DB; **tenant isolation is enforced in application code, not by row-level DB ACLs**. |

Per-record tenancy: storage records are **stamped with `org_id`** and list
queries are **org-scoped**, with a per-record `assert_org_owns` that returns
**404 on mismatch** (not 403, to prevent existence probing). Legacy org-less
records default to a fixed `DEFAULT_ORG_ID`, and a one-time backfill attaches
orphan projects/users to a default org. Org scoping only activates when a
non-`None` org is passed (i.e. a non-dev principal). Evidence: `api/storage.py`
(stamping :107-110/181-182/424-427/505-508; org-scoped lists; `org_scope` /
`assert_org_owns` :637-657; `DEFAULT_ORG_ID` :26; backfill :660-696),
`src/mmm_framework/auth/store.py:298-345`.

> **Encryption.** Encryption **at rest** and **in transit** is
> **deployment-dependent** — it is a property of where and how you run the
> platform (disk/volume encryption, a TLS-terminating proxy, your S3/Redis
> configuration), not something this codebase asserts on its own. `org_id` and
> account records are stored in **plaintext** within the SQLite DB / JSON
> sidecars; protect those files at the OS/volume level.

---

## 3. Retention & deletion

| Mechanism | Status | What it does |
|-----------|--------|--------------|
| `data_retention_days` (default **30**) | **Partial** | Drives a cleanup job that purges **Redis job records** older than the cutoff (`api/worker.py:1268-1292`, `api/config.py:52`). |
| On-demand deletion of datasets/models/results | Implemented | Deletion is via the storage layer's `delete_*` methods. |

> **Important caveat (do not overclaim).** `data_retention_days` currently
> purges only **Redis job records** — it does **not** delete stored datasets,
> models, or results on disk. Deletion of data and models is **on-demand** via
> the `delete_*` methods, not an automatic time-based sweep. If you need
> time-bounded deletion of stored artifacts, implement or schedule it at the
> operations layer.

**Secrets handling.** No credentials are hardcoded; LLM/provider keys and cloud
credentials are read from environment/ADC only (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, Vertex ADC, S3 keys via settings). Secrets
may be loaded from a `.env` file if present — protect that file. Subprocess/
container kernels launch with a **fail-closed, allowlisted environment**: a
regex denylist strips anything ending in `_API_KEY` / `_TOKEN` / `_SECRET` /
`_CREDENTIALS` / `_PASSWORD` / `_PRIVATE_KEY` plus `GOOGLE_APPLICATION_CREDENTIALS`
and `MMM_LLM_*`, and the denylist wins over any passthrough (opt-out via
`MMM_KERNEL_SCRUB_ENV=0`, debug only). The **in-process** kernel shares the API
process env entirely (no scrub possible). Evidence: `api/config.py:33-44`,
`src/mmm_framework/auth/config.py:34`, `src/mmm_framework/agents/kernels.py:317-357,573-576`.

---

## 4. Sub-processors

A "sub-processor" here is any third party your deployment **can** send data to.
**Each is applicable only if you configure it** — none is required for a default
single-user, fully-local install, and the LLM provider is chosen by a model
configuration file (never hard-coded; see `docs/model-configuration.md`).

### 4a. LLM / embedding providers

Selected via the model-config file (chat) and resolved **separately** for
embeddings (because Anthropic offers no embedding model). Configure exactly one
chat provider and one embedding provider; the rest stay dormant.

| Sub-processor | Configured by | Data sent (if configured) | Egress |
|---------------|--------------|---------------------------|--------|
| **Anthropic** (`anthropic`) | `ANTHROPIC_API_KEY` | Chat messages, system prompt, tool results | To Anthropic |
| **Google Vertex AI** (`vertex_anthropic` / `vertex_gemini`) | Application Default Credentials (no API key) | Chat messages, system prompt, tool results; KB embeddings via Vertex `text-embedding-005` | To **your** GCP project's Vertex endpoint |
| **OpenAI** (`openai`) | `OPENAI_API_KEY` | Chat messages, system prompt, tool results; embeddings via `text-embedding-3-small` | To OpenAI |
| **Google Gemini (Developer API)** (`google_genai`) | `GOOGLE_API_KEY` | Chat messages, system prompt, tool results | To Google |
| **LM Studio (local)** (`lmstudio`) | none — defaults to `http://localhost:1234/v1` | Chat / tool results / embeddings | **No egress** — `localhost` |

> **Fully-local option.** `provider: lmstudio` with a locally loaded chat model
> **and** a local embedding model means **no chat, tool-result, or embedding
> traffic leaves the machine** — both endpoints are `localhost`, and no API key
> is needed or sent.

### 4b. Infrastructure sub-processors

| Sub-processor | Status / applicability | Notes |
|---------------|------------------------|-------|
| **Redis** | Required for async jobs | Self-hosted by you (the docs run `redis-server` locally). Holds job records; subject to the `data_retention_days` cleanup. Not a third-party-operated service unless you point it at a managed Redis. |
| **Amazon S3** | **Optional — applicable only if configured, and currently a settings placeholder** | S3 settings exist (`api/config.py:37-44`) but **no S3 code path is implemented** in `StorageService`. Do not rely on S3 as an active sub-processor until it is implemented. |

### 4c. Egress posture for executed code

Outbound network from agent-executed code depends on the kernel tier
(**deployment-dependent**). The **container** kernel denies outbound network by
default (`--network none` on the ipc transport; an `--internal` network on Linux
tcp; egress opened only via `MMM_KERNEL_EGRESS=open`). Egress-deny applies
**only** to the container kernel — the default `inprocess` and the `subprocess`
kernels have **full host network**. On macOS dev boxes the tcp transport leaves
egress open and unenforced. Evidence:
`src/mmm_framework/agents/container_kernel.py:154-192`.

---

## 5. Security controls that bear on data handling

These are referenced for completeness; full detail and file/line evidence is in
`docs/security.html`. Status labels are honored exactly.

- **Authentication — deployment-dependent.** OFF by default
  (`MMM_AUTH_ENABLED` defaults to `False`); when disabled, a single-tenant dev
  principal (OWNER) bypasses tenant scoping. When enabled with
  `MMM_AUTH_SECRET`, passwords are hashed with **scrypt** (memory-hard,
  self-describing encoding, constant-time verify), and sessions are **HS256
  JWTs** (signature/exp/nbf/aud/iss verified; 1h access / 14d refresh).
  Evidence: `src/mmm_framework/auth/config.py:30`, `deps.py:50-66`,
  `auth/passwords.py:26-85`, `auth/tokens.py:42-121`. Refresh tokens are
  **revocable** (single-use rotation + `/auth/logout` + a `revoked_tokens`
  denylist; deactivation blocks refresh), and account lifecycle (invites,
  password reset, deactivation) is implemented and audit-logged. Caveats: HS256
  symmetric only (OIDC/RS256 is a `NotImplementedError` stub); access tokens are
  stateless, so **instant access-token kill** is not yet implemented (they expire
  by their ≤ 1h TTL).
- **Tenant isolation — implemented (gated on auth being enabled).**
  Cross-tenant access returns **404** (IDOR-resistant); org-stamped storage and
  org-scoped queries (see §2). Correctness depends on each route mounting the
  access dependency.
- **Hosted-profile session capability — deployment-dependent.** With
  `MMM_AGENT_HOSTED=1`, the chat `thread_id` is a bearer capability: the API
  refuses guessable/client-invented/unknown thread IDs (server-minted sessions
  only) and refuses to drive another org's session. In dev posture, thread IDs
  are accepted and auto-created. Evidence: `src/mmm_framework/api/main.py:450-473`.
- **Kernel sandbox — deployment-dependent (requires the image to be built/
  deployed).** The container kernel applies cap-drop ALL, no-new-privileges,
  read-only rootfs, cgroup mem/pids/cpu caps, ulimits, and a tmpfs-only writable
  area, with a **fail-closed isolation gate** that refuses to spawn unless those
  controls + a denied-egress posture are present. **Active only when
  `MMM_AGENT_KERNEL=container` AND a built kernel image + a container runtime
  exist.** The default `inprocess` kernel is **not sandboxed**, and `subprocess`
  is **not** in the sandboxed set. The hosted profile force-upgrades to
  `container` and **refuses to start** if the sandbox is incomplete. Evidence:
  `src/mmm_framework/agents/container_kernel.py:202-244,373-403`,
  `agents/profile.py:29-65`, `api/main.py:156-158`.
- **SSRF guard on website brand-extraction — implemented.** http/https +
  ports 80/443 only, no embedded credentials, every resolved address must be
  globally routable (rejects loopback/RFC1918/link-local 169.254 metadata/ULA/
  multicast/reserved), redirects manually re-vetted (max 3), byte cap +
  content-type check. Runs **host-side only**, never in the kernel; disabled in
  the hosted profile unless `MMM_BRAND_FETCH_ALLOW=1`. Documented residual:
  a narrow DNS-rebind window. Evidence: `src/mmm_framework/agents/brand_extract.py:48-143`.
- **Untrusted kernel egress (plots/tables) — implemented.** Schema-filtered,
  hard size-capped (`MMM_PLOT_MAX_BYTES` 5 MiB / `MMM_TABLE_MAX_BYTES` 1 MiB),
  and content-addressed with **thread-salted** IDs so payloads don't dedup or
  become guessable across sessions. The salted ID is the only capability for the
  `GET /plots/{id}` / `GET /tables/{id}` endpoints — there is no separate
  per-tenant ACL on them. Evidence: `src/mmm_framework/agents/workspace.py:77-164`.
- **File-download serving — implemented (TOCTOU-safe, traversal-guarded).**
  Resolved path must be inside an allowlisted root; opened with
  `O_NOFOLLOW|O_CLOEXEC`; regular-file check; streams from the validated fd.
  Documented residual: a narrow parent-dir-swap race not fully closed without a
  read-only mount namespace (defense-in-depth). Evidence:
  `src/mmm_framework/api/main.py:2723-2792`, `agents/workspace.py:202-226`.
- **Audit logging — implemented (tamper-evident, not tamper-proof).** Security/
  lifecycle events are a hash-chained JSONL (each record's hash =
  `sha256(prev_hash + canonical(record))`); `verify()` detects any edit/delete/
  reorder, and the chain resumes across restarts. It is **tamper-evident, not
  tamper-proof** — the JSONL is local; true tamper resistance needs an off-host
  shipper. The sink is installed by the API app at startup via
  `install_audit_sink()` (best-effort; FastAPI lifespan,
  `src/mmm_framework/api/main.py:167-169`) — it runs only when the API process
  boots, so a library/standalone use that never starts the API gets no sink.
  Evidence: `src/mmm_framework/agents/audit_sink.py:45-157`.

---

## 6. Compliance posture

**SOC 2: readiness, not certified.** The controls above describe a SOC 2
*readiness* posture for self-hosted deployments — documented access control,
tenant isolation, audit logging, secret handling, and sandboxing. **The
platform is not SOC 2 certified**, and no third-party attestation is claimed.
Encryption at rest / in transit is **deployment-dependent** (your disk/volume
encryption and TLS termination), and the kernel sandbox is **deployment-
dependent** (requires the kernel image to be built and a container runtime
present). Do not represent any control as stronger than its status above.

---

## 7. Quick reference: data flow at a glance

```
Upload (CSV/MFF) ──► Local disk (uploads / workspace)   [default backend]
                        │
                        ├─► Execution kernel (local)  ──► fit ──► models/results on disk
                        │        (inprocess: no sandbox; container: sandboxed if image present)
                        │
Chat turn ─────────────►├─► LLM provider (Anthropic / Vertex / OpenAI / Gemini / LM Studio)
                        │      sends: messages + system prompt + tool results (NOT raw datasets)
                        │
KB ingest ─────────────►└─► Embedding provider (Vertex / Google / OpenAI / local LM Studio)

Async jobs ──► Redis (purged after data_retention_days)
Accounts/tenancy ──► SQLite sessions.db (org-scoped in app code)
Plots/tables ──► browser via GET /plots/{id} | /tables/{id} (thread-salted ids)
```

For the narrative version with full per-control evidence, see
`docs/security.html`. For responsible-disclosure / `security.txt`, see
`docs/responsible-disclosure.html` and `docs/.well-known/security.txt`.

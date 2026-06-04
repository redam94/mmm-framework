# Phase 3 Scoping — Sandbox + the Full Control Set

**Parent design:** `agent-session-kernels.md` (v2) §4/§5/§6. **Builds on:** Phase 1
(`-phase1.md`, COMPLETE — `KernelManager` + `InProcessKernel`/`SubprocessKernel`) and Phase 2
(`-phase2.md`, COMPLETE — fits + the ~14 model tools run *in* the kernel; cold-reload from the
session's `mmm_models/`). **Goal:** make a `SubprocessKernel` safe to run untrusted, LLM-authored
code next to other sessions and the host — env-scrubbed, FS/mount-isolated, resource-capped,
egress-denied, with a tenant-scoped plot channel and an audit trail.

> **Hard invariant (every PR):** the default `inprocess` path is provably unchanged. Every control
> here targets the **subprocess** kernel and the API endpoints that serve kernel-written artifacts.

> **Grounded by a 6-region surface scan (2026-06-03)** of the spawn/provisioner seam, the plot
> cross-tenant channel, the secret-env inventory, the tenant/auth model, the logging infra, and the
> download/TOCTOU surface. The scan changed the plan in four ways — recorded in §1.

---

## 1. What the surface scan changed (read this first)

1. **There is no multi-user auth today.** `thread_id` is client-supplied, auto-created
   (`touch_session`), and never ownership-checked; the Models API has only a single shared API
   key. The spec's "authorize the caller against the thread's owning tenant" (§3.2.2) and "plot
   channel can't cross tenants" (§6) **assume an identity layer that does not exist.** **Decision
   (user, 2026-06-03): treat `thread_id` as a bearer capability** — per-thread plot namespacing +
   retrieval ACL, plus (in the hosted profile) forbid guessable thread_ids (no `default_thread`;
   server-minted uuid4 only). Combined with Tier-2 mount isolation this closes the cross-tenant
   criterion *enough*. **Full multi-tenant auth is a separate, explicitly-deferred workstream.**

2. **Env-scrub is *not* a hosting-enabling switch.** On the actual target (GCP VM / Vertex / ADC)
   the dominant credential-theft path is the **metadata server (169.254.169.254)**, reachable from
   any same-host subprocess regardless of env. Blocking it is **egress (Tier 2)**, not env. So
   env-scrub is valid **defense-in-depth under today's single-user posture** — land it, but **do
   NOT wire a `hosted=on` switch that flips the trust model until the container tier completes it**
   (§4: "a half-applied sandbox is *more* dangerous than an honest single-user posture").

3. **Teardown-wipe is Tier-2-only (the ephemeral overlay), not the persistent workspace.** Phase-2
   PR-C.3 cold-reload reads `<work_dir>/mmm_models/run_*` from the **persistent** session dir after
   an LRU evict+respawn. Wiping `thread_dir` on evict would destroy that. The spec's teardown-wipe
   targets the container scratch overlay/tmpfs, which only exists in Tier 2.

4. **Tier-2's tightest constraint: TCP ZMQ and `--network none` are mutually exclusive.** A
   `--network none` container has only loopback, so the host `jupyter_client` cannot reach a TCP
   kernel inside it — the egress-deny control and the kernel control-channel collide. This dictates
   the provisioner architecture (IPC over a shared-mount unix socket vs. an egress-firewalled
   internal network vs. an in-container proxy). **Spike containerized-kernel connectivity as the
   first Tier-2 task**, before writing any other Tier-2 control.

## 2. Two tiers (sequencing decision, user 2026-06-03: Tier 1 first)

Phase 3's control set "must pass jointly" (§6) — but that bar is met by the **completed union of
both tiers**, not by a single PR. We land **Tier 1** (cross-platform, host-side, testable on the
macOS dev box) as incremental hardening, then **Tier 2** (the Linux container sandbox). The
hosted trust-model switch stays **inert/erroring** until Tier 2 completes.

| Tier | Controls | Platform | Trust-model claim |
|---|---|---|---|
| **Tier 1** | env-scrub, plot thread-scope+ACL+caps, path/TOCTOU hardening, audit log | cross-platform (here) | none new — defense-in-depth on the existing single-user posture |
| **Tier 2** | Podman provisioner + image, cgroups/seccomp/ulimit/pids, egress-deny + metadata-block, read-only image + ephemeral overlay + teardown-wipe, workspace-only mount, masked /proc//sys, disk/inode quota, fail-closed, denied-egress/syscall audit | Linux (Podman; Docker-compatible runtime configurable) | semi-trusted multi-user (the §4 posture flip) |

**Runtime (user 2026-06-03): Podman** (podman-machine being set up). The provisioner takes the
runtime binary from `MMM_KERNEL_RUNTIME` (default `podman`) so a Docker host works unchanged — both
are OCI.

## 3. Tier 1 — host-side hardening PRs (cross-platform, testable now)

> **Status: COMPLETE (2026-06-03).** PR-E.1 `7a1a201` · PR-E.2 `a150da9` · PR-E.3 + PR-E.4 (this
> series). Default `inprocess` unchanged; full fast suite green. Next: Tier 2 (§4), gated on the
> PR-F.0 connectivity spike. The hosted trust-model switch remains **inert** until Tier 2 lands.

### PR-E.1 — Env scrub at kernel spawn (the headline win) — ✅ DONE
`SubprocessKernel._start()` (`kernels.py:397`) calls `start_kernel(cwd=...)` with **no `env=`**, so
the kernel inherits the API's full `os.environ` — a hostile cell reads every API key + the ADC
path in one line. jupyter_client 8.7.0 passes `env=` straight through `LocalProvisioner` →
`Popen(env=...)`, so the fix is local and infra-free.

- Add `_scrubbed_kernel_env()` in `kernels.py`: **fail-closed allowlist** (drop everything not
  matched) that is *generous* enough not to break the PyTensor/JAX compile stack, plus a
  **secret-pattern denylist applied on top** (belt-and-suspenders).
  - **Allow (exact):** `MMM_AGENT_WORKSPACE`, `MMM_AGENT_KERNEL`, `MMM_MAX_KERNELS`,
    `MMM_KERNEL_*`, `MMM_CELL_TIMEOUT`, `MMM_FIT_TIMEOUT`, `MMM_MODEL_CONFIG`, the non-secret
    `MMM_LLM_*` config (`PROVIDER/MODEL/TEMPERATURE/MAX_TOKENS/PROJECT/LOCATION/BASE_URL`),
    `MMM_EMBED_{PROVIDER,MODEL,LOCATION}`, `GOOGLE_CLOUD_PROJECT`, and system
    `PATH/HOME/TMPDIR/TMP/USER/LOGNAME/LANG/TZ/TERM/SHELL`.
  - **Allow (prefix):** `LC_`, `PYTHON`, `PYTENSOR`, `JAX`, `XLA`, `OMP`, `MKL`, `NUMBA`,
    `OPENBLAS`, `CONDA_`, `VIRTUAL_ENV`, `MPLCONFIGDIR`, and the compiler vars
    `CC/CXX/CPATH/CPPFLAGS/LDFLAGS/LIBRARY_PATH/LD_LIBRARY_PATH/DYLD_*` (the macOS PyTensor clang
    fix lives in `~/.pytensorrc`, so `HOME` is required).
  - **Deny (pattern, even if allowed):** `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_CREDENTIAL*`,
    `*_PASSWORD`, `*_PRIVATE_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, `MMM_LLM_API_KEY`,
    `MMM_LLM_CREDENTIALS_PATH`, `AWS_*`, `AZURE_*`.
  - Escape hatch: `MMM_KERNEL_ENV_PASSTHROUGH` (comma-list of extra exact names) for the rare
    legitimately-needed var; and `MMM_KERNEL_SCRUB_ENV=0` to disable (debug only — default **on**
    for subprocess; never affects in-process).
- Pass it: `self._km.start_kernel(cwd=..., env=_scrubbed_kernel_env())`.
- **The kernel never calls the LLM/embedder** (those run in the API process), so this is safe —
  but the allowlist could still be too tight for the compile stack. **Gate = a real fit under the
  scrubbed env** (slow test) + a cell asserting `os.environ` carries no secret + that
  `google.auth.default()`/a dummy `*_API_KEY` are absent. (Note: env-scrub does **not** stop the
  metadata server — that's PR-F egress, Tier 2; say so in the test/docstring.)
- **Import-chain check (done):** the startup source imports `agents.tools` (`_normalize_figure`),
  `agents.model_ops`, `agents.fitting` — none import `agents.llm`/`graph` or read a secret at
  import, so scrub can't break startup. (If that changes, move `_normalize_figure` to a leaf
  `agents/_plot_capture.py` per the Phase-1 doc.)

### PR-E.2 — Path / TOCTOU hardening (zero container dependency; real holes today) — ✅ DONE
- **Guard the unguarded file-servers with `is_within()`:** the 10 hardcoded report/slides endpoints
  (`main.py`) call `FileResponse` with no allow-root check. Routed through the TOCTOU-safe
  `_safe_serve`/`_guarded_file_response`.
  - **DEFERRED follow-up (not kernel-isolation, so out of this PR):** the **Models API**
    `routes/models.py:283-330` report download serves a `filepath` read from Redis with no
    `is_within()` check. That path is written by *trusted* job code (not the kernel), and lives in a
    different app/trust domain — so it's a generic web-hardening item, not part of the kernel
    sandbox. The spec's §4 "every host-facing capability re-checked" still wants it eventually;
    named here so it isn't lost.
- **`safe_join` the upload filenames:** `/upload` (`main.py:~1496`) and the KB upload
  (`main.py:~1264`) build paths with raw `os.path.join`/`abspath` — traversal-able. Use
  `workspace.safe_join`.
- **Close the symlink/TOCTOU race in `_guarded_file_response`:** open with `O_NOFOLLOW` (and
  re-validate the opened fd's `realpath` is still within an allowed root) before serving, so a
  swap between the `is_within` check and `FileResponse` can't redirect to `/etc/passwd`.
- **Do NOT drop `Path.cwd()` from `allowed_roots` yet** — the legacy report endpoints serve
  `agent_*_report.html` from CWD, so dropping it regresses downloads. `O_NOFOLLOW` + `is_within`
  already kills the symlink hole. Dropping `cwd` pairs with moving reports into the workspace
  (Phase-2 PR-D.4 / Tier-2 mount work) — defer it there, behind the (inert) hosted profile.

### PR-E.3 — Plot channel: thread-salted ids + payload caps — ✅ DONE
The plot store is shared + content-addressed and `GET /plots/{id}` (`main.py:1184`) has **no
thread check**, so a content-guessable id crosses tenants; the plotly payload had no size or schema
cap (a cell can encode another session's data into a figure). **Decision (user 2026-06-03):
thread-*salted* ids, no frontend change** (over a path-scoped `/plots/{thread}/{id}` route).
- **Salt the id with `thread_id`** (`store_plot(fig, thread_id)`): `id = sha256(thread_id\0 +
  payload)[:24]`, stored flat at `plots/<id>.json`. The id stays a single opaque token — so
  `GET /plots/{id}` and the React/Streamlit fetch URL are **unchanged** — but it is no longer
  guessable from figure content across sessions (the id IS the capability, and ids only reach a
  client via that session's SSE stream), and identical figures no longer dedup across tenants.
  Within a session, identical figures still collapse to one id (immutable browser caching kept).
- **Cap + schema-validate at capture** (in `store_plot`, the single funnel for both kernels):
  reject a payload over `MMM_PLOT_MAX_BYTES` (default 5 MiB) or that isn't a `{"data": [...], …}`
  figure dict (raise `ValueError`); keep only the real figure keys (`data/layout/frames/config`)
  so a cell can't smuggle arbitrary content. The `execute_python` loop **drops** a rejected plot
  (with a `plot_rejected` audit line) instead of inlining it (which would defeat the cap).

### PR-E.4 — Audit logging — ✅ DONE
No audit log existed; kernel lifecycle was silent. Added a `mmm_audit` stdlib logger + `_audit(event,
**fields)` helper in `kernels.py`, emitting `key=value` lines at: `kernel_spawn`, `kernel_evict_lru`,
`kernel_respawn`, `kernel_died`, `kernel_timeout_interrupt`, `kernel_timeout_kill` (and
`plot_rejected` from `tools.py`). Single stdlib sink (stdout/container logs) for Tier 1; the
denied-egress/denied-syscall events and any off-host/tamper-evident sink are Tier 2 (PR-F) /
Phase 4d.

## 4. Tier 2 — the Linux container sandbox PRs (Podman)

> **PR-F.0 (the blocker, §1.4): spike connectivity — ✅ DONE.** See
> `deploy/kernel/F0-connectivity-findings.md`. **Decision:** two transports via
> `MMM_KERNEL_TRANSPORT`, platform-defaulted. **`ipc` + `--network none`** is the prod posture
> (no network ⇒ egress + metadata trivially denied; control over bind-mounted unix sockets;
> Linux-validated — virtiofs breaks unix sockets on the macOS dev box, as predicted). **`tcp` +
> per-port-forward** is the dev/macOS posture (proven `21*2→42` here); egress-deny over `tcp`
> uses an `--internal` network on Linux (proven to block both the internet and
> `169.254.169.254`), which rootless macOS can't combine with port-forward — so macOS-dev `tcp`
> has open egress (accepted; the metadata-block matters on the GCP VM, which runs `ipc`).

- **PR-F.1 — Container image — ✅ DONE** (`deploy/kernel/Containerfile` + `requirements.lock`):
  pinned dep closure (`uv export`), framework on PYTHONPATH, `g++`/`gcc` for runtime compile, `tini`
  as PID 1, non-root uid 10001, headless mpl, scratch under `/tmp`. Smoke-tested (2.53 GB).
  Read-only rootfs is a *run-time* flag (PR-F.3/F.5); hash-locking is a noted prod follow-up.
- **PR-F.2 — `ContainerKernel` — ✅ DONE** (`agents/container_kernel.py`, impl `container`):
  subclasses `SubprocessKernel`, **reuses the whole protocol** (`_run`/`execute`/`run_model_op`/
  `fit`/`_teardown`); only `_start`/`reset` differ — launch via `podman run`, connect with a
  `BlockingKernelClient`, and a `_ContainerManager` shim maps `is_alive`/`interrupt`/`shutdown`
  onto podman. Carries the PR-E.1 scrubbed env (config-only subset — host path/python vars are
  *not* forwarded), workspace bind-mount at the same abs path (`--userns=keep-id --user
  host-uid` so it's writable rootless), transport per PR-F.0. **Validated:** execute + host-
  visible workspace write + cross-cell persistence + in-container plot capture (slow test). The
  cgroup/seccomp/read-only/masked-proc *run flags* slot into `_resource_args`/`_security_args`
  (PR-F.3).
- **PR-F.3 — Resource caps:** cgroup mem (~2 GB), `pids.max`, ulimits, seccomp default-deny; the
  per-cell wall-clock cap escalates to an **out-of-band cgroup kill** (SIGINT can't stop a compiled
  sampler — §3.4). Per-kernel disk + inode quota at the mount (tmpfs/overlay `size=`).
- **PR-F.4 — Egress deny-by-default + metadata block:** deny all egress; explicitly block
  link-local/`169.254.169.254`/`metadata.google.internal`; the chat model API is **not**
  allowlisted (the kernel has no reason to call it). **Log every drop** (`mmm_audit`
  `denied_egress`). This is the control that actually stops ADC token theft.
- **PR-F.5 — Ephemeral overlay + teardown-wipe + fail-closed:** per-kernel ephemeral overlay
  separate from the persistent workspace mount; on evict/crash kill the cgroup, unmount and wipe
  the overlay before any reuse (never the persistent session dir — §1.3). **Fail-closed:** if any
  isolation control fails to apply at spawn, refuse to run the kernel.
- **PR-F.6 — Hosted profile activation:** flip the (until-now inert) `hosted` switch on — forbid
  guessable thread_ids (no `default_thread`, server-minted uuid4), drop `Path.cwd()` from
  `allowed_roots`, move report output into the workspace mount (closes Phase-2 PR-D.4). **Exit
  (must pass jointly, §6):** a hostile cell cannot read another session's workspace, host
  secrets/metadata, or the network; OOM is contained; the plot channel can't cross tenants.

## 5. What stays unchanged
The default `inprocess` path (every PR is subprocess/endpoint-only); the portable `.py` export; the
content-addressed plot store's *browser-immutable caching* (now thread-namespaced); the artifact
log; Phase-2 cold-reload from the **persistent** `mmm_models/` (Tier-1 teardown must not touch it).

## 6. Risks
- **Allowlist too tight → silent slow/broken PyTensor compile.** Mitigation: generous prefix-allow
  + the real-fit gate test.
- **Plot namespacing breaks existing dashboards.** Mitigation: read-only back-compat for flat ids.
- **Tier-2 connectivity (network-none vs ZMQ).** Mitigation: PR-F.0 spike before any other Tier-2
  control; virtiofs caveat noted.
- **False sense of safety.** Tier 1 is *not* a hosting green-light; the trust-model switch stays
  inert until Tier 2 (§1.2).

## 7. PR sequencing
Tier 1: **PR-E.1** (env scrub) → **PR-E.2** (path/TOCTOU) → **PR-E.3** (plot ACL+caps) → **PR-E.4**
(audit). E.1 and E.2 are independent and can land in either order. Tier 2: **PR-F.0** (spike) gates
F.1–F.6.

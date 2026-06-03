# Per-Session Sandboxed Kernels — Design & Phased Plan (v2, review-hardened)

**Status:** Draft, hardened by a 5-lens adversarial review (security, lifecycle/concurrency,
migration correctness, cost/ops, protocol) · **Date:** 2026-06-03 · **Owner:** Matthew Reda

**Decision context:** moving the agent toward **multi-user / hosted**. This supersedes the
in-process warm kernel (`NAMESPACE_CACHE`) and makes the planned ARQ-for-fits offload
redundant (fits run *inside* the kernel).

> **v2 changelog.** v1 named the right controls but asserted several as done. The review
> found **blockers** that v2 fixes: (1) the "secrets stay in the API" boundary is *false*
> unless the kernel env is scrubbed and the cloud metadata endpoint is blocked; (2) the
> Python path guards do **not** constrain a hostile kernel — FS isolation is the mount
> namespace; (3) the content-addressed plot store is a **cross-tenant exfiltration channel**;
> (4) collapsing `MODEL_CACHE` into the kernel silently breaks **~12 tools + causal_tools**;
> (5) cell-DONE detection requires correlating **two** ZMQ sockets; (6) interrupt does **not**
> stop a compiled NumPyro fit. Per-kernel resource math is now measured and the cheapest
> viable config is specified.

---

## 1. Goal

Every chat session gets a **live Python environment** that persists for the session, owns its
memory, runs **all** Python (`execute_python` *and* fits), keeps the API event loop free,
**isolates tenants** from each other and the host, and stays **portable** (the existing `.py`
export already snapshots the work, kernel-independently).

Non-goals: a general notebook product; GPU scheduling; arbitrary compute unrelated to a session.

## 2. Where we are now (what this replaces)

| Concern | Today | Limitation for hosted multi-user |
|---|---|---|
| `execute_python` state | in-process warm kernel (`runtime.NAMESPACE_CACHE`, thread-scoped, LRU(2)) | in the API process; GIL contention; no isolation; single process |
| Model fit | `fit_mmm_model` runs `mmm.fit()` in-process (`tools.py:527`) | blocks the API loop for *every* session on that process |
| Live `mmm`/`results` | `MODEL_CACHE`; read **synchronously by ~12 tools + causal_tools** (see §3.5) | not just `execute_python` — a large migration surface |
| Plot capture | monkeypatch `fig.show()` + `_normalize_figure` in-process (`tools.py:1250-1318`) | must cross a process boundary faithfully |
| Secrets | framework reads `*_API_KEY` from `os.environ`, Vertex via ADC metadata | **a hostile in-kernel cell reads them in one line** unless scrubbed |
| Portability | `session_export.build_session_script` → `.py` | ✅ kernel-independent; keep |

### 2.1 Measured resource facts (drive all sizing)

Measured on this repo — these numbers, not intuition, govern density:

| Quantity | Value |
|---|---|
| Bare interpreter + pandas/numpy/plotly (an **analysis-only** kernel) | **~94 MB RSS** |
| + `pymc/pytensor/numpyro/jax/nutpie/arviz` imported (a **fit-capable** kernel, idle) | **~370 MB RSS** |
| Peak RSS *during* a NUTS fit (XLA compile + `chains=4`) | **~1.5–2 GB** |
| Live `InferenceData` trace (4×1000 draws, ~104 obs) | **~40–100 MB** |
| Kernel image `site-packages` (jaxlib alone 228 MB) | **~1.3 GB** |

Implication: the **resident fit stack (~370 MB) is the density limiter**, not the trace; a host
budgets ~1.5–2 GB per kernel that *may* fit → **~6–8 fit-capable kernels on a 16 GB node,
~12–16 on 32 GB**. Fit **concurrency**, not session count, is the binding constraint.

## 3. Target architecture

```
┌────────────── API process (FastAPI, async) ──────────────┐
│  /chat ─→ LangGraph agent ─→ execute_python / model tools │
│                          KernelManager (registry+locks)    │   thread_id ownership
│                                   │ authorize tenant↔thread │   VERIFIED here
└───────────────────────────────────┼───────────────────────┘
                                     │ jupyter_client (IPC/ZMQ), allow_stdin=False
                  ┌──────────────────┼───────────────────┐
                  ▼                  ▼                   ▼
            ┌──────────┐       ┌──────────┐        ┌──────────┐  one sandboxed process
            │ kernel A │       │ kernel B │   ...  │ kernel N │  per session; scrubbed
            │ df,mmm,  │       │          │        │          │  env, no cloud identity,
            │ results, │       │          │        │          │  workspace-only mount,
            │ namespace│       │          │        │          │  egress deny-by-default
            └──────────┘       └──────────┘        └──────────┘
```

**The agent never executes Python itself.** But "behind `execute_python`" is **not** the whole
story — see §3.5: ~12 tools read live model objects today and must also be migrated.

### 3.1 Kernel runtime & protocol — DECIDED: `jupyter_client` + ipykernel

Option A (ipykernel over the Jupyter ZMQ protocol). **Decision closed**, because:
- `ipykernel 7.1.0` + `jupyter-client 8.7.0` are **already in `uv.lock`** (`pyproject.toml:46`) —
  **zero new dependency weight**.
- The only argued advantage of a custom subprocess (Option B) — interrupt control — is
  **illusory**: neither can interrupt a compiled NumPyro fit (§3.4). Cancellation reduces to
  kill+respawn either way, which `jupyter_client` already gives.
- Use `AsyncKernelManager`/`AsyncKernelClient` so the proxy is natively async.

The agent's pre-bound surface is installed two ways (the warm-kernel precedence contract,
faithfully reproduced — `tools.py:1051-1056`):
- **Startup file (run once):** module re-imports + builders + the monkeypatched `fig.show()`
  + standalone `save_result`/`load_result` (the closures in `tools.py:1128-1188` can't cross a
  process boundary — reuse the **known-correct standalone copies already in
  `session_export.py:70-111`**).
- **Per-call header (prepended by the proxy):** only the *reserved* names — `OUTPUT_DIR`,
  `dataset_path`, the `__mmm_df_source__` df-source sentinel — so system names re-layer each
  call and user vars are never clobbered.

### 3.2 The execute path (protocol spec)

1. Agent emits `execute_python(code)`. The `@tool` fn stays in the API process (so
   `InjectedState`/`InjectedConfig`/`Command`/ContextVar all survive unchanged); only code
   execution goes remote. The responsiveness win comes from the **CPU work being in another
   process (GIL released during `recv`)** — the existing sync-tool-in-executor model suffices.
2. `KernelManager.get_or_spawn(thread_id)` — **authorizes the caller against the thread's
   owning tenant** (never trust client-supplied `thread_id`, `main.py:230`), then returns the
   session's kernel under a **per-`thread_id` spawn lock** (double-checked) so two concurrent
   first-calls can't spawn two kernels over the same workspace mount (§5).
3. Send one `execute_request` with **`allow_stdin=False`** (an `input()` call otherwise hangs
   the kernel slot forever). Hold a **per-kernel asyncio lock** — one outstanding execute per
   kernel (matches today's one-cell-at-a-time semantics and keeps demux trivial).
4. **Correlate every message by `parent_header.msg_id`** (ignore others — startup/other
   sessions). Accumulate the iopub→artifact mapping:

   | message (socket) | handling |
   |---|---|
   | `status` busy/idle (iopub) | drives DONE detection; not an artifact |
   | `execute_input` (iopub) | ignore (code already persisted as `code_snippet`, `main.py:343`) |
   | `stream` {stdout/stderr} (iopub) | merge **both** into one text buffer in receipt order (matches today's combined `redirect_stderr→stdout`, `tools.py:1336`); **enforce a byte budget**, truncate with `...[output truncated at N MB]` |
   | `execute_result` / `display_data` (iopub) | `application/vnd.plotly.v1+json` → `workspace.store_plot`; `text/plain` → buffer; `image/png` → file store or drop (decide), never into the text buffer |
   | `error` {ename,evalue,traceback} (iopub) | **strip ANSI**; format with the exact substring **`Error executing code`** (load-bearing: `main.py:370` and `session_export.py:216` key off it); rebuild the NameError self-healing hint by `ename=='NameError'` + regex on `evalue` (the live `e.name` is gone over the wire) |
   | `execute_reply` (**shell**) {status, execution_count} | completion + ok/error |

5. **DONE = BOTH** iopub `status:idle` **and** shell `execute_reply` for this `msg_id` (two
   different sockets, no cross-socket ordering — waiting on only one truncates trailing
   output or loses ok/error). Then return the **single** `Command` exactly as today
   (`tools.py:1430`), so all downstream artifact logic is untouched.
- **Output bounding is a security control** (§4), not polish: IOPUB is a lossy PUB/SUB socket
  that silently drops past its HWM — a `while True: print()` either loses output undetectably
  or OOMs the consumer. Cap bytes at the consumer.
- **Trailing-expression echo — DECIDED: suppress** (preserve today's `print()`-only contract,
  `tools.py:1032`). `exec(code, ns)` never echoes a bare last expression, but ipykernel emits
  `execute_result` for `df` and could flood the checkpoint with a huge repr. The proxy wraps
  submitted code (e.g. AST-transform the trailing expression to a statement, or run with the
  display hook suppressed) so behavior matches `InProcessKernel` and the plain-script `.py`
  export exactly. No docstring change.

### 3.3 Plot capture across the boundary

Keep the **same** `fig.show()`/`pio.show` monkeypatch as today, but in the kernel startup
file: run `_normalize_figure` (the palette/layout remap, `tools.py:1250-1306` — it **must move
into the kernel** or charts render with default Plotly colors), then **explicitly**
`publish_display_data({'application/vnd.plotly.v1+json': fig.to_json()})` (a `display_data`
message, **not** a stateful comm). The proxy routes that payload straight into
`workspace.store_plot` and keeps the `{id,title}` ref — byte-identical to today's
content-addressed flow, so browser caching is unchanged. **Test:** assert byte-identical
`store_plot` id for a fixed figure across `InProcessKernel` and `SubprocessKernel`.
**Security:** the display payload is untrusted egress (§4) — size-cap + schema-validate it.

### 3.4 Fits run in the kernel

`fit_mmm_model`'s `build_and_fit` body relocates into the kernel. Consequences the review
forced into the open:
- **Interrupt does NOT stop a compiled NumPyro fit.** The default path is
  `.bayesian_numpyro()` (`tools.py:453`); `mcmc.run()` compiles to one XLA program and ignores
  SIGINT until it returns. **Stop = interrupt (short timeout, works for pure-Python PyMC at
  draw boundaries) → escalate to SIGKILL the kernel + respawn.** Run the kernel under an init
  shim (`tini`) so signals/zombies behave when it's PID 1 in a container. (Verify `nutpie`
  separately — Rust may poll at draw boundaries.)
- **A mid-fit kill loses the whole run.** The model is saved to disk only *after* `fit()`
  returns (`tools.py:552-566`) — there is no mid-sampling checkpoint. So a fitting kernel is
  **never evictable** (§5) and disconnect-during-fit must interrupt, not orphan.
- **Live progress — DECIDED: per-thread `asyncio.Queue`.** A tool returns one `Command` at the
  end and the SSE stream is at LangGraph **update** granularity (`main.py:285`), so iopub
  "fitting n/N" has nowhere to surface mid-cell. The proxy writes progress into a per-thread
  `asyncio.Queue` that `event_generator` drains alongside the agent stream and emits as an
  ephemeral SSE event (like the existing `dashboard_update`/plots event, `main.py:407-419`);
  the final cell result stays a single `Command`, so all artifact logic (`main.py:334-456`) is
  untouched. Not persisted as an artifact.

### 3.5 Migration surface — the ~12 tools that read live model objects (NEW, blocker)

"Behind `execute_python`" is false today. These read `mmm`/`results` **synchronously in the
API process, outside `execute_python`**, and break the moment those objects live only in the
kernel:

`get_roi_metrics` (`tools.py:674`), `get_component_decomposition` (`:731`),
`get_model_diagnostics` (`:798`), `get_adstock_weights` (`:872`),
`get_saturation_curves` (`:935`), the inspect/compare path (`:1854`),
`save_fitted_model` (`:1964`), client report (`:2241`), and `:2352/:2706/:2755`; plus
`causal_tools.py` `prior_predictive_check` (`:815/:833`) and
`leave_one_out_decomposition` (`:940`).

Per-tool decision (Phase 2, enumerated — not a one-line "collapse"):
- **In-kernel mandatory** where the **live** PyMC graph is needed — e.g. `prior_predictive_check`
  calls `mmm.sample_prior_predictive()`; reloading from disk in the API process does **not**
  rescue it. Run as an in-kernel execute returning markdown + dashboard JSON.
- **Reload-from-disk** acceptable for read-only/report tools that only need a deserialized
  model (`MMMSerializer().load` in the API process), if a live trace isn't required.
- **RPC scalars** otherwise: a thin call into the kernel returns computed rows/dicts (ROI,
  decomposition, diagnostics) — **never** ship the live PyMC objects over the wire.

All `_MODEL_CACHE` consumers (incl. `causal_tools`) move **together**, or the model becomes
unreachable from where those tools run.

## 4. Sandboxing & security (the §11 posture flip)

Hosted multi-user **inverts** today's "unsandboxed, single-user" posture: untrusted,
LLM-authored code in long-lived processes next to other tenants. **Control ordering (this is
the point):**

1. **Strongest boundary (deferred) = VM-level isolation** (gVisor or Kata) — adopted only in
   Phase 4c if the trust model escalates LLM code to *genuinely hostile* (§7.2).
2. **Chosen default boundary = container + cgroups + seccomp default-deny + non-root +
   read-only image + ulimits + egress-deny + workspace-only mount** (Podman). For the
   **semi-trusted** posture we're adopting, *this jointly-verified Phase 3 control set is the
   load-bearing isolation* — not the deferred VM tier.
3. **Python path guards (`workspace.safe_join`/`is_within`) are cosmetic for the kernel** —
   they run in the API process and gate *download*, not what in-kernel code reads/writes.

**Fail-closed:** if any isolation control fails to apply at spawn, **refuse to run the kernel**.
A half-applied sandbox (container without egress policy) is *more* dangerous than today's
honest single-user posture — "revertible" must mean fail-closed, not "add egress next sprint."

| Layer | Control (v2, specified) |
|---|---|
| **Secrets (was the false claim)** | Launch each kernel with a **scrubbed/allowlisted env** — no `*_API_KEY`, no `MMM_LLM_*`, no `GOOGLE_APPLICATION_CREDENTIALS` (the framework reads these from `os.environ`, `llm.py:119`/`embeddings.py`). Give the kernel **no cloud identity** distinct from the API SA. **Egress DENY `169.254.169.254` + `metadata.google.internal`** (ADC token theft, `llm.py:181`). **Exit test:** a cell dumping `os.environ` + `google.auth.default()` yields no usable credential. |
| **Data** | **API stages a per-session, read-only, quota'd copy** of the dataset into the workspace mount; **the kernel never holds data-source creds.** (If live in-kernel access is unavoidable: a per-tenant, per-table, minutes-TTL, audience-bound token injected per-execute — never persisted in the namespace.) |
| **Filesystem** | The boundary is the **per-kernel mount namespace** — workspace-only, no host FS, `nosuid,nodev,noexec`, no symlink-escape. **Not** the Python guards. Drop `Path.cwd()` from `allowed_roots` (`workspace.py:100`) for the hosted profile. For any API op opening a kernel-writable path, `O_NOFOLLOW` / re-validate the fd's realpath (TOCTOU/symlink swap). |
| **CPU/mem/pids** | cgroup mem cap (~2 GB), `pids.max` **+ ulimits + seccomp together** (pids alone won't stop FD/socket/thread exhaustion). Per-cell wall-clock timeout enforced by **out-of-band cgroup kill** (SIGINT is insufficient for native samplers, §3.4). |
| **Network egress** | **Deny by default.** Allowlist must **NOT** include the chat model API (the kernel has no reason to call it; allowing it is an exfil path). Block link-local/metadata. **Log every drop.** |
| **Display/plot channel** | **Untrusted egress, not just fidelity.** Today `GET /plots/{id}` has **no tenant ACL** and the store is shared+content-addressed (`main.py:1178`, `workspace.py:74`) — a cell can encode another tenant's data/secrets into a figure and get a fetchable id. **Scope plot ids to the owning thread, authorize retrieval, size-cap + schema-validate, strip free-text/embedded-image traces.** |
| **Image supply chain** | All kernels share one large image → one backdoored dep is cross-tenant root. Pinned/hash-locked deps, read-only image + per-kernel ephemeral overlay. |
| **Host disclosure** | Mask/minimize `/proc` and `/sys` (`/proc/self/environ`, `/proc/1/cgroup`, kernel version for targeted CVEs). |

**Teardown:** on evict/crash, **kill the cgroup, unmount and wipe** (discard the overlay/tmpfs)
of the workspace before any reuse — never reuse a dirty mount. **Do NOT serialize the namespace
to disk** (v1 option c): cloudpickle of untrusted-authored objects is **RCE-on-restore** (a
malicious `__reduce__`) *and* captures any injected token. **Hard per-kernel disk + inode
quotas**, enforced at the mount, separate from per-host.

## 5. Lifecycle & resource management

- **Spawn:** lazy on first `execute_python`, under the per-`thread_id` spawn lock (§3.2.2) —
  idempotent, single-writer workspace mount.
- **Busy-state eviction (not last-touch):** track `executing | idle-since` per kernel. **A
  kernel with an in-flight execute — especially a fit — is never the LRU victim or TTL casualty**
  (a 20-min fit receives no `execute_request`, so last-touch wrongly flags it idle). Combined
  with "no mid-fit checkpoint" (§3.4), last-touch eviction would kill the most expensive work.
- **Kernel-death contract (distinct from a code error):** an OOM/segfault/evict mid-execute
  emits **no reply** → the proxy await would hang and leave an orphan `tool_call`. The proxy
  must detect death (heartbeat timeout / connection reset), **synthesize an error `ToolMessage`
  bound to the `tool_call_id`** (so the graph stays consistent and `_repair_orphan_tool_calls`,
  `main.py:157`, isn't the only backstop), emit a "kernel restarted" SSE event (reuse the
  self-healing-hint pattern), and respawn.
- **Disconnect-during-fit:** the proxy concurrently watches `raw_request.is_disconnected()`
  while awaiting iopub and **interrupts/kills** on disconnect — otherwise an abandoned fit pins
  a kernel for its full duration, defeating the caps.
- **Backpressure (decide before Phase 1):** bounded queue with a max-wait, **plus dedicated SSE
  events** — `waiting for kernel capacity`, then `cold-starting kernel…` — so the held-open
  `/chat` connection is legible; on timeout, a typed retryable error.
- **Cold-namespace reconstitution:** **(a) accept-cold** (agent rebuilds via the self-healing
  hint + `load_result`) and **(b) reconstitute-then-replay-ALL** — the contract the `.py`
  export already proves: reload `df` + `mmm`/`results` from disk, then replay **every** cell in
  order (errored cells marked, not skipped). **Drop v1's "skip side-effecting cells"** — there
  is no purity classifier and skipping a cell that defines a later-used name guarantees
  divergence. Honest caveat: replay reproduces **code**, not necessarily identical live objects
  (RNG/trace).
- **Interrupt/restart/crash:** §3.4 (kill+respawn for compiled samplers; `tini` as PID 1). A
  "Restart kernel" UI control.

### 5.1 Capacity, density & the cheapest viable config

- **Two-tier kernels (biggest cost lever):** default to a **light kernel** (~94 MB, no sampler
  stack) for `execute_python`; **lazily upgrade** to a fit-capable kernel (~370 MB) only when
  `fit_mmm_model` is called. Analysis-only sessions (the majority) get **~4× density**.
- **Density math:** budget peak-fit ~1.5–2 GB per *fit-capable* kernel → the §2.1 ceilings.
  Bin-pack fits by **reserving cores per concurrent fit** (`chains=4` ⇒ 4 cores), not by
  session count.
- **Cold start:** baked image on every kernel host (**no per-spawn pull**) + a **small warm
  pool** of pre-imported kernels + lazy fit-stack import. Target SLO e.g. p95 first-cell <3 s
  (warm-pool hit), <30 s cold.
- **Autoscaling signal:** composite of **reserved-memory-headroom** (guarantee a fit can land)
  **and active-fit-count vs vCPU**; kernel-count only as a coarse cap (idle-but-pinned kernels
  inflate it).
- **ZMQ/fd accounting:** 5 sockets/kernel. **Transport=`ipc`** when co-located (Phase 1/2, zero
  TCP ports); TCP bound to loopback/private iface when containerized. Budget **~5 fds/kernel** in
  the API process and **raise its `ulimit -n`** (default 1024 bites at a few hundred kernels).
  Keep the connection-file HMAC key out of any tenant-readable path.
- **Cheapest viable hosted config (the floor):** single kernel-host or small fixed pool (no
  autoscaler); **Podman + cgroups (mem ~2 GB, pids cap, 4-vCPU/fit reservation) + egress-deny +
  non-root + seccomp + read-only framework mount** (NO gVisor/Kata); light-kernel default +
  lazy fit import; baked image + small warm pool; hard max-kernels with reject-when-full; idle
  TTL 15–30 min; accept-cold+replay. **Sacrifices:** strong isolation vs a determined escape
  (only OK if LLM code is *semi*-trusted), elasticity (queue/reject at peak), cross-host HA.

## 6. Phased migration plan

The enabler is a **`KernelManager` abstraction** so `execute_python`/the model tools stop
caring *where* code runs. Each phase ships value; **"revertible" means fail-closed** (§4).

- **Phase 0 — done.** Warm kernel + `save_result`/`load_result` + `reset_namespace`; portable
  `.py` export; frontend render fixes.
- **Phase 1 — Kernel abstraction behind a flag.** `KernelManager` with `InProcessKernel`
  (today's warm kernel) and `SubprocessKernel` (`jupyter_client`); shared startup/precedence
  logic (§3.1); the §3.2 protocol (DONE-by-two-messages, `allow_stdin=False`, byte budget,
  error-substring invariant, trailing-echo decision). **Fix the v1 contradiction:** either move
  `fit_mmm_model` into the kernel in Phase 1 too, **or** document that under the
  `SubprocessKernel` flag `mmm`/`results` are not yet available to `execute_python`. **Exit:**
  27 agent tests pass on `InProcessKernel`; the `SubprocessKernel` smoke test **fits then
  references `mmm` in `execute_python`** (so the gap can't ship behind a green test) + a
  plot-fidelity parity test.
- **Phase 2 — Fits + all model tools in the kernel.** Relocate `build_and_fit`; **migrate the
  ~12 §3.5 consumers** (in-kernel / reload-from-disk / RPC-scalars, per tool, `causal_tools`
  included); progress side-channel (§3.4); stop=interrupt→kill. `MODEL_CACHE` retires. **Exit:**
  fit→interpret round-trips through `SubprocessKernel`; event loop responsive under a
  concurrent fit (measured); `model_path` stays on a host-resolvable shared volume so the `.py`
  export's `MMMSerializer().load(...)` preamble and the zip-download still work.
- **Phase 3 — Sandbox + the FULL control set, jointly (single-host multi-user).** Containerize;
  **env scrub + no cloud identity + metadata-egress-deny**; workspace-only mount (TOCTOU-safe);
  cgroup/seccomp/ulimit/pids caps; egress deny-by-default (model API excluded) **with drop
  logging**; plot-id tenant ACL; teardown-wipe; per-kernel disk/inode quotas; **per-code +
  host-level audit** (spawn/evict/OOM/interrupt/denied-egress/denied-syscall) — moved here from
  v1's Phase 4 because hostile code first runs here. **Exit (must all pass jointly):** a hostile
  cell cannot read another session's workspace, host secrets/metadata, or the network; OOM is
  contained; the plot channel can't cross tenants. **Partial enablement is not a valid state.**
- **Phase 4 — Orchestrated pool (hosted, cross-host), split:**
  - **4a** orchestrator (k8s/Nomad) + per-user quotas (Docker/Podman carried from Phase 3).
  - **4b** autoscaling on the §5.1 signal + warm pool + load test to a stated concurrency SLO.
  - **4c** stronger isolation tier (gVisor/Kata) — gated on the §7.2 trust-model decision, with
    its cost (memory tax + spawn latency + syscall slowdown) folded into §2.1 density.
  - **4d** centralized metrics + tamper-evident, off-host audit.

## 7. Decisions (all resolved 2026-06-03)

1. **Kernel protocol — `jupyter_client`/ipykernel** (already in lockfile; zero new dep).
2. **Isolation tier — Podman + cgroups + seccomp + non-root** (cheapest viable; full density).
   gVisor/Kata are **deferred to Phase 4c** and adopted *only* if the trust model escalates LLM
   code from semi-trusted to genuinely hostile — each adds a per-kernel memory tax + spawn
   latency that lowers §2.1 density. The Phase 3 control set (env scrub, metadata-egress-deny,
   workspace-only mount, seccomp, egress-deny) is the load-bearing isolation either way.
3. **Reconstitution — accept-cold + reconstitute-then-replay-ALL** (no serialize-namespace;
   it's RCE-on-restore + secret capture).
4. **Kernel granularity — per-`session` (`thread_id`)**, not per-user.
5. **Trailing-expression echo — SUPPRESS** (preserve today's `print()`-only contract): the
   proxy wraps submitted code so a bare trailing expression is not echoed. Keeps `InProcess`↔
   `Subprocess` parity and the `.py` export faithful; no docstring change. (See §3.2.)
6. **Progress side-channel — per-thread `asyncio.Queue`**: the proxy writes iopub progress into
   a queue the `event_generator` drains alongside the agent stream (mirrors the ephemeral
   `dashboard_update`/plots event, `main.py:407-419`); the final cell result stays a single
   `Command`, so all artifact logic is untouched. (See §3.4.)
7. **Data path — API stages a per-session, read-only, quota'd copy** into the workspace mount;
   the **kernel never holds data-source credentials** (no token broker). (See §4 Data row.)

## 8. Risks

- **Resource blow-up is the dominant cost** — the resident fit stack (~370 MB), not the trace,
  caps density. Two-tier kernels + reserve-for-fit caps + eviction + quotas are mandatory.
- **Cold-namespace UX** — eviction loses live state; mitigated by export/replay/`load_result`;
  surface "kernel restarted" (reuse the self-healing hint).
- **Interrupt can't stop a compiled fit** — the stop button is kill+respawn for NumPyro; that
  loses the run (no mid-fit checkpoint). Set expectations in the UI.
- **Silent-fidelity regressions** — plot capture (normalize-in-kernel + explicit MIME), the
  `Error executing code` substring invariant, and the NameError-hint reconstruction must each
  have a parity test, or charts/error-marking silently break.
- **Security regressions** — every host-facing capability (downloads, plot retrieval, file
  reads) must be re-checked against the now-hostile kernel; the boundary is the mount/cgroup/
  egress, not the Python guards.

## 9. What stays unchanged

The portable `.py` export (artifact-driven, `session_export.py:150` — kernel-independent), the
content-addressed plot store + browser caching (now **tenant-scoped**, §4), the artifact log
(`code_snippet`/`text_output`/`model_run`), the workspace layout, and the frontend render-perf
fixes. The kernel work lives **behind** the tool layer — **except** the §3.5 model tools and the
§4 security controls, which the migration must explicitly carry. **Caveat:**
`model_run['model_path']` (`tools.py:560`) must point at a **host-resolvable shared volume** (not
a path that exists only inside the sandbox), or the export preamble and zip-download break.

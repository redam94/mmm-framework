# Phase 1 Implementation Plan — Kernel Abstraction Behind a Flag

> **Status: COMPLETE (2026-06-03).** PR1 (extractions) · PR2 (KernelManager seam +
> InProcessKernel) · PR3 (SubprocessKernel + adversarial-review fixes) · PR4 (these docs).
> Default `inprocess` is behavior-unchanged; `MMM_AGENT_KERNEL=subprocess` opt-in. Next:
> Phase 2 (fits + the ~12 model tools into the kernel) — see `agent-session-kernels.md` §3.5/§6.

**Parent design:** `agent-session-kernels.md` (v2). **Goal of Phase 1:** introduce a
`KernelManager` seam so `execute_python` stops caring *where* code runs, with two
implementations — `InProcessKernel` (today's warm kernel, the **default**, zero behavior
change) and `SubprocessKernel` (`jupyter_client`/ipykernel, opt-in). **No sandboxing, no
fits-in-kernel, no orchestration** — those are Phases 2–4.

The whole value of Phase 1 is de-risking: land the abstraction and the protocol against a
green test suite while the default path is byte-for-byte unchanged.

---

## 1. Scope

**In:**
- A `Kernel` interface + `KernelManager` registry (per-`thread_id`, spawn lock, minimal LRU).
- `InProcessKernel` — extract today's `execute_python` execution body unchanged.
- `SubprocessKernel` — ipykernel via `jupyter_client`, implementing the §3.2 protocol.
- A flag (`MMM_AGENT_KERNEL`) selecting the impl; **default `inprocess`**.
- Refactor `execute_python`/`reset_namespace` to route through the manager.
- Extract shared pieces (`_normalize_figure`, the error+hint formatter) to module level.
- Tests: InProcess parity (existing 27 stay green) + SubprocessKernel smoke/parity.

**Out (explicitly deferred):**
- Sandboxing / containers / cgroups / egress (Phase 3).
- Fits-in-kernel and the ~12 model-tool migration (Phase 2). **Consequence:** under
  `SubprocessKernel`, `mmm`/`results` are **not** available to `execute_python` (they live in
  the API-process `MODEL_CACHE`). This is a *documented, tested* boundary, not a silent gap.
- Eviction policy, busy-state tracking, autoscaling (later).
- Progress side-channel, data staging, security controls.

## 2. The seam

The split is already clean in `execute_python` (`tools.py:995`): everything up to and including
`exec` produces `(stdout, captured_plots, is_error)`; everything from `tools.py:1383` on
(content-address plots, register files, build `Command`) is **API-side and stays in the tool**.

```python
# agents/kernels.py  (new)
from dataclasses import dataclass, field

@dataclass
class KernelContext:
    thread_id: str
    work_dir: str | None          # OUTPUT_DIR / cwd for the run
    dataset_path: str | None      # for df auto-bind
    mmm: object | None = None     # InProcess only (from MODEL_CACHE); None for Subprocess in P1
    results: object | None = None

@dataclass
class ExecuteResult:
    stdout: str                   # combined stdout/stderr, ALREADY including the formatted
                                  # "Error executing code:" + NameError hint when is_error
    plots: list[dict] = field(default_factory=list)   # normalized Plotly figure JSONs
    is_error: bool = False

class Kernel(Protocol):
    def execute(self, code: str, ctx: KernelContext) -> ExecuteResult: ...
    def reset(self) -> None: ...        # fresh namespace
    def shutdown(self) -> None: ...      # release resources
```

`KernelManager` owns the registry + lifecycle:

```python
class KernelManager:
    def __init__(self, impl: str): self._impl = impl; self._kernels = {}; self._locks = {}
    def get_or_spawn(self, thread_id: str) -> Kernel:
        # per-thread spawn lock (double-checked) so two concurrent first-calls don't
        # spawn two kernels over the same workspace mount (design §3.2.2)
    def reset(self, thread_id: str) -> None: ...
    def shutdown(self, thread_id: str) -> None: ...

MANAGER = KernelManager(os.environ.get("MMM_AGENT_KERNEL", "inprocess"))
```

`execute_python` becomes (sketch):

```python
ctx = KernelContext(thread_id, str(work_dir), state.get("dataset_path"),
                    mmm=_MODEL_CACHE.get("fitted_model"), results=_MODEL_CACHE.get("fit_results"))
result = MANAGER.get_or_spawn(thread_id).execute(code, ctx)
# --- unchanged API-side post-processing (tools.py:1383+) ---
for fig in result.plots: store_plot(fig) ...           # content-address + dashboard refs
register_generated_files(thread_id, before, exclude_dirs=("results",))
return Command(update={"messages": [ToolMessage(format_content(result), tool_call_id)], "dashboard_data": dashboard_data})
```

`reset_namespace` → `MANAGER.reset(thread_id)` (InProcess: clears `NAMESPACE_CACHE`;
Subprocess: kernel restart).

## 3. Shared extractions (do first — both impls depend on them)

1. **`_normalize_figure` + `_PALETTE` → module level** in `tools.py` (or a new
   `agents/_plot_capture.py`). Today it's a closure inside `execute_python` (`tools.py:1250`);
   both `InProcessKernel` and the SubprocessKernel **startup file** must call the *same*
   function or charts render with default colors (design §3.3).
2. **Error+hint formatter → `format_execution_error(traceback_str, ename, missing_name) -> str`.**
   Produces the exact `"Error executing code:\n…"` text (load-bearing substring — `main.py:370`,
   `session_export.py:216`) plus the NameError hint. `InProcessKernel` calls it with the live
   exception (`e.name`); `SubprocessKernel` parses `ename`/`evalue` and calls the same fn — so
   the invariant holds in both (design §3.2, migration finding).
3. **Reuse the standalone `save_result`/`load_result`** already in `session_export.py:70-111`
   for the SubprocessKernel startup (the in-process versions are closures over `work_dir`).

## 4. `InProcessKernel` (the default — must not change behavior)

Move the body of today's `execute_python` (`tools.py:1057-1362`) into
`InProcessKernel.execute`: pull `NAMESPACE_CACHE.namespace()`, build the reserved `env`
(pd/np/plt/mmf/builders, `save_result`/`load_result`, df auto-bind, `mmm`/`results` from
`ctx`), install the `fig.show()` interception (now calling the module-level `_normalize_figure`),
`chdir` into `work_dir`, `ns.update(env)`, `exec(code, ns)`, format errors via the shared
helper, return `ExecuteResult(stdout, plots=captured_plots, is_error)`. **No logic changes** —
this is a pure move, so the existing 27 tests pass untouched. `reset()` → `NAMESPACE_CACHE.reset()`.

## 5. `SubprocessKernel` (opt-in)

```python
from jupyter_client.manager import AsyncKernelManager   # already in uv.lock
```

**Spawn (lazy, once per thread):** `AsyncKernelManager`, `start_kernel(cwd=work_dir)`, client
`start_channels()`, `await wait_for_ready()`, then run the **startup source** once.

**Startup source** (a string the kernel execs once) — the shared surface:
- `import` block mirroring `tools.py` pre-binds (reuse `session_export._PREAMBLE_IMPORTS`).
- the standalone `save_result`/`load_result` (`session_export._PREAMBLE_HELPERS`).
- the `fig.show()`/`pio.show` monkeypatch → `_normalize_figure(fig)` then
  `publish_display_data({'application/vnd.plotly.v1+json': fig.to_json()})` (design §3.3 — a
  `display_data`, not a comm).
- **echo suppression:** `get_ipython().ast_node_interactivity = 'none'` — one line that makes a
  bare trailing expression *not* auto-display, matching `exec()`’s print()-only contract
  (decision §7.5). (`_normalize_figure` is included in the startup source or imported from the
  module-level extraction.)

**Per-call header** (prepended to each `execute_request` — only reserved names, design §3.1):
`OUTPUT_DIR = "…"; dataset_path = "…"` and the df auto-bind sentinel logic.

**Execute (the §3.2 protocol):**
```python
msg_id = kc.execute(header + "\n" + code, allow_stdin=False)   # input() must not hang the slot
stdout, plots, err = [], [], None
idle = reply = False
while not (idle and reply):
    try: m = await kc.get_iopub_msg(timeout=T)
    except Empty: ... heartbeat/death check ...
    if m["parent_header"].get("msg_id") != msg_id: continue   # ignore other requests
    t = m["msg_type"]
    if t == "status" and m["content"]["execution_state"] == "idle": idle = True
    elif t == "stream": stdout.append(m["content"]["text"])     # both stdout+stderr, in order
    elif t in ("display_data","execute_result"):
        data = m["content"]["data"]
        if "application/vnd.plotly.v1+json" in data: plots.append(json.loads(data[...]))
    elif t == "error": err = m["content"]                       # {ename,evalue,traceback}
    # shell reply (separate socket) — poll/await get_shell_msg() for execute_reply→reply=True
```
- **DONE = idle (iopub) AND execute_reply (shell)**, both correlated by `msg_id`.
- Enforce a **byte budget** on accumulated stdout (truncate with a marker) — IOPUB is lossy/
  unbounded (design §3.2 security note).
- On `error`: strip ANSI, parse the missing name from `evalue`, call
  `format_execution_error(...)` → same text as InProcess.
- Hold a **per-kernel asyncio lock** so only one execute is outstanding.
- **Transport=`ipc`** (co-located in P1 → zero TCP ports). Raise the API `ulimit -n` note for
  many kernels (design §5.1).

`reset()` → `await km.restart_kernel()` + re-run startup. `shutdown()` → `await
km.shutdown_kernel()`.

## 6. `fit_mmm_model` in Phase 1

**Stays in-process** (Phase 2 relocates it). It writes `mmm`/`results` to the API-process
`MODEL_CACHE`. `InProcessKernel` reads them from `ctx` (so post-fit `execute_python` works as
today). `SubprocessKernel` runs in a different process and **cannot** receive the live PyMC
object → `mmm`/`results` are **undefined there in Phase 1**. This is the documented boundary;
the smoke test (below) asserts it explicitly so it can't regress silently.

## 7. Flag & config

`MMM_AGENT_KERNEL=inprocess|subprocess` (env), optionally surfaced in `model_config.yaml`.
Default `inprocess`. Selected once at `MANAGER` construction. No per-request switching in P1.

## 8. Lifecycle (minimal in P1)

Spawn lazily under the per-thread lock; `shutdown` on `reset_namespace` (Subprocess restart) and
on session delete (wire into the existing session-delete path). A simple **max-kernels cap**
with LRU `shutdown_kernel()` of the least-recently-used (Subprocess only). Full busy-state
eviction, TTLs, and kernel-death→orphan-repair are Phase 3 — but add a **TODO + a death check**
(heartbeat timeout) that synthesizes an error `ExecuteResult` rather than hanging, so the seam is
death-aware from day one (design §5).

## 9. Tests & exit criteria

| Test | Asserts |
|---|---|
| Existing `test_agent_workspace_kb.py` (27) on `InProcessKernel` (default) | **byte-for-byte behavior unchanged** — the refactor is pure |
| `test_kernels.py::subprocess_smoke` (guarded on `ipykernel` import) | a cell defines `x`; next cell reads `x` → persists; `print` output round-trips |
| `…::subprocess_fit_then_reference_mmm` | after an in-process fit, referencing `mmm` in a `SubprocessKernel` cell **raises `NameError`** — pins the documented Phase-1 boundary so Phase 2 must consciously remove it |
| `…::plot_fidelity_parity` | a fixed figure yields the **same `store_plot` id** via InProcess and Subprocess (normalize-in-kernel works) |
| `…::error_substring_parity` | an erroring cell yields `is_error=True` + the `"Error executing code"` substring + the NameError hint, identically in both impls |
| `…::echo_suppression` | a bare trailing `df.head()` produces **no** output under Subprocess (matches `exec()`) |

**Phase 1 is done when** all of the above pass and the default path (`inprocess`) is unchanged.

## 10. Sequencing (suggested PRs)

1. **PR1 — extractions (no behavior change):** move `_normalize_figure`/palette to module level;
   add `format_execution_error`. Existing tests green.
2. **PR2 — the seam:** `agents/kernels.py` (interface + `KernelManager` + `InProcessKernel` as a
   pure move); `execute_python`/`reset_namespace` route through `MANAGER`; default `inprocess`.
   Existing 27 green = the move is faithful.
3. **PR3 — `SubprocessKernel`:** startup source, per-call header, the §3.2 protocol, echo
   suppression, plot capture; the flag; minimal lifecycle. The Subprocess test matrix (§9).
4. **PR4 — docs:** update `agent-knowledge-workspace.md` (the warm kernel is now
   `InProcessKernel`) and `CLAUDE.md` troubleshooting (`MMM_AGENT_KERNEL`).

## 11. Risks & rollback

- **Refactor drift (PR2):** the InProcess move could subtly change behavior. Mitigation: it's a
  *move*, reviewed against the 27 tests; keep the diff mechanical.
- **jupyter_client async quirks:** shell/iopub ordering, `wait_for_ready` timeouts. Mitigation:
  the explicit two-message DONE rule + per-kernel lock; a spawn/ready timeout that fails the cell
  cleanly.
- **Rollback:** the flag defaults to `inprocess`; if `SubprocessKernel` misbehaves, it's never on
  the default path. PR2 is the only one touching the default, and it's behavior-preserving.

## 12. Effort

PR1 ~0.5 day · PR2 ~1.5 days · PR3 ~2–3 days (protocol + tests) · PR4 ~0.5 day. ~1 week for a
working `SubprocessKernel` behind the flag with the default untouched — the foundation Phases 2–4
build on.

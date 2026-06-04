"""Kernel abstraction for ``execute_python`` (Phase 1 of
``technical-docs/agent-session-kernels.md``).

The seam lets ``execute_python`` stop caring *where* Python runs. Phase 1 ships
``InProcessKernel`` (the default — today's in-process warm namespace; defined in
``tools.py`` where its dependencies already live). ``SubprocessKernel``
(``jupyter_client``, one process per session) lands in PR3 behind
``MMM_AGENT_KERNEL=subprocess``.

``KernelManager`` selects the active implementation and caches instances. A
``per_session`` kernel (subprocess) gets one instance per ``thread_id``; a
non-``per_session`` kernel (in-process) is stateless at this layer — its
per-thread state lives in the thread-scoped ``NAMESPACE_CACHE`` — so a single
shared instance is correct.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class KernelContext:
    """Everything a kernel needs to run a cell, decoupled from the tool/graph.

    ``work_dir``/``dataset_path`` are strings (serializable across a process
    boundary for the subprocess impl). ``mmm``/``results`` are passed in by the
    tool from ``MODEL_CACHE`` for the in-process impl; they are ``None`` for the
    subprocess impl until fits move into the kernel (Phase 2) — a documented,
    tested boundary, not a silent gap.
    """

    thread_id: str
    work_dir: str | None
    dataset_path: str | None = None
    mmm: Any = None
    results: Any = None


@dataclass
class ExecuteResult:
    """What a kernel returns for one cell — the API-side post-processing
    (content-address plots, register files, build the ``Command``) is identical
    regardless of which kernel produced this."""

    stdout: str
    plots: list = field(default_factory=list)
    is_error: bool = False


@runtime_checkable
class Kernel(Protocol):
    per_session: bool

    def execute(self, code: str, ctx: KernelContext) -> ExecuteResult: ...

    def run_model_op(self, op_name: str, kwargs: dict) -> dict: ...

    def fit(self, model_spec: dict, dataset_path: str) -> dict: ...

    def reset(self) -> None: ...

    def shutdown(self) -> None: ...


class KernelManager:
    """Selects a kernel implementation and caches instances per session.

    A ``per_session`` impl gets one instance per ``thread_id`` (spawned under a
    double-checked lock so two concurrent first-calls don't spawn two kernels
    over the same workspace — design §3.2.2). A non-``per_session`` impl shares a
    single instance.
    """

    def __init__(self, impl: str, factories: dict[str, Callable[[], Kernel]]):
        self._impl = impl if impl in factories else "inprocess"
        self._factories = factories
        self._instances: "OrderedDict[str, Kernel]" = OrderedDict()
        self._lock = threading.RLock()
        # Bound the number of live per-session kernels (each subprocess kernel is
        # a real process + fds). Phase 1 has no idle-eviction; this LRU cap is the
        # backstop so distinct thread_ids can't leak processes without limit.
        self._max = max(1, int(os.environ.get("MMM_MAX_KERNELS", "8")))

    @property
    def impl(self) -> str:
        return self._impl

    def _key(self, thread_id: str) -> str:
        cls = self._factories[self._impl]
        return thread_id if getattr(cls, "per_session", False) else "__shared__"

    def get_or_spawn(self, thread_id: str) -> Kernel:
        key = self._key(thread_id)
        inst = self._instances.get(key)
        if inst is not None:
            self._instances.move_to_end(key)  # LRU touch
            return inst
        with self._lock:  # double-checked: never spawn two kernels for one session
            inst = self._instances.get(key)
            if inst is None:
                inst = self._instances[key] = self._factories[self._impl]()
                # evict the least-recently-used kernel(s) beyond the cap
                while len(self._instances) > self._max:
                    _ek, _ev = self._instances.popitem(last=False)
                    if _ek == key:  # never evict the one we just created
                        self._instances[_ek] = _ev
                        break
                    try:
                        _ev.shutdown()
                    except Exception:
                        pass
            else:
                self._instances.move_to_end(key)
            return inst

    def reset(self, thread_id: str) -> None:
        inst = self._instances.get(self._key(thread_id))
        if inst is not None:
            inst.reset()

    def shutdown(self, thread_id: str) -> None:
        key = self._key(thread_id)
        with self._lock:
            inst = self._instances.pop(key, None)
        if inst is not None:
            try:
                inst.shutdown()
            except Exception:
                pass

    def shutdown_all(self) -> None:
        """Shut down every cached kernel (wire into process exit / app lifespan
        so subprocess kernels aren't orphaned)."""
        with self._lock:
            insts = list(self._instances.values())
            self._instances.clear()
        for inst in insts:
            try:
                inst.shutdown()
            except Exception:
                pass


# ── SubprocessKernel (PR3): one ipykernel process per session ─────────────────
#
# Talks the Jupyter messaging protocol via the SYNC jupyter_client API (the tool
# runs in a threadpool executor; a blocking recv() releases the GIL while the
# kernel subprocess does CPU-bound work, so the event loop stays responsive — no
# async conversion needed). Phase 1: NOT sandboxed (Phase 3) and fits still run
# in the API process (Phase 2), so `mmm`/`results` are intentionally NOT bound
# here — referencing them raises NameError, a documented + tested boundary.

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NAME_RE = re.compile(r"name '([^']+)' is not defined")
_MAX_OUTPUT_BYTES = 200_000

# Dedicated display_data MIME for model-op results — kept OFF stdout (which
# carries pymc/serializer prints + progress bars, and stderr is merged into it),
# so a noisy compute can never corrupt the structured result.
_MODELOP_MIME = "application/vnd.mmm-modelop+json"


def _json_safe(o):
    """Recursively coerce a value to strict-JSON-safe form: non-finite floats
    (NaN/Inf, e.g. from an HDI on empty samples) -> None, numpy scalars ->
    python. Applied to a model-op result so the dashboard payload is clean for
    the frontend regardless of which kernel produced it."""
    import math

    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes, int)):  # numpy scalar
        try:
            return _json_safe(o.item())
        except Exception:
            return o
    return o


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")


def _truncate(s: str, limit: int = _MAX_OUTPUT_BYTES) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[output truncated at {limit} bytes]"


def _build_startup_source() -> str:
    """The code run once per kernel: the framework surface (reused from the .py
    export so it stays in lock-step), the durable-result helpers, the dataset
    auto-bind, the fig.show()->display_data plot capture, and echo suppression."""
    from mmm_framework.agents.session_export import (
        _PREAMBLE_IMPORTS,
        _PREAMBLE_HELPERS,
    )

    extra = """
# ── reserved system bindings (match the in-process kernel) ──
import os
import sys as _sys
_sys.stderr = _sys.stdout  # one stream -> stdout/stderr keep write order, as in-process
try:
    import matplotlib
    matplotlib.use("agg")  # headless worker
    import matplotlib.pyplot as plt
except Exception:
    pass

# ── dataset auto-bind (reserved; invoked from the per-call header) ──
def __mmm_autobind_df(_path):
    global df, dataset_path, __mmm_df_source__
    if not _path:
        return
    dataset_path = _path
    try:
        if globals().get("__mmm_df_source__") == _path:
            return
        import os as _os
        _too_big = _os.path.exists(_path) and _os.path.getsize(_path) > 250 * 1024 * 1024
        if not _too_big and _path.lower().endswith(".csv"):
            df = pd.read_csv(_path)
            __mmm_df_source__ = _path
        elif not _too_big and _path.lower().endswith(".parquet"):
            df = pd.read_parquet(_path)
            __mmm_df_source__ = _path
    except Exception:
        pass

# ── plot capture: fig.show()/pio.show() -> display_data(plotly json) ──
try:
    import json as _json
    from mmm_framework.agents.tools import _normalize_figure as _normfig
    from IPython.display import publish_display_data as _pdd
    import plotly.io as _pio
    import plotly.basedatatypes as _pbd

    def _mmm_capture_show(fig_or_self, *a, **k):
        _f = _normfig(fig_or_self)
        _pdd({"application/vnd.plotly.v1+json": _json.loads(_f.to_json())})

    _pio.show = _mmm_capture_show
    _pbd.BaseFigure.show = _mmm_capture_show
except Exception:
    pass

# ── model-op driver (Phase 2 PR-B): run a model_ops op on the in-kernel model ──
try:
    from mmm_framework.agents import model_ops as _mmm_mo
    from IPython.display import publish_display_data as _mmm_pdd

    def _mmm_run_op(_name, _kw):
        _op = _mmm_mo.OPS.get(_name)
        if _op is None:
            return {"content": None, "dashboard": {}, "error": "Unknown model op: " + str(_name)}
        _m = globals().get("mmm")
        if _m is None:
            return {"content": None, "dashboard": {}, "error": _mmm_mo.NO_MODEL_MSG}
        return _op(_m, globals().get("results"), **(_kw or {}))

    def _mmm_emit_op(_name, _kw):
        _mmm_pdd({"application/vnd.mmm-modelop+json": _mmm_run_op(_name, _kw)})

    def _mmm_emit_fit(_spec, _dataset_path):
        # Fit IN the kernel: mmm/results become module GLOBALS so the subsequent
        # run_model_op / execute_python see the fitted model (removes the
        # Phase-1 boundary). Only the JSON `info` crosses the MIME channel.
        global mmm, results
        try:
            from mmm_framework.agents.fitting import build_and_fit

            mmm, results, _info = build_and_fit(_spec, _dataset_path)
        except Exception as _e:
            _mmm_pdd(
                {"application/vnd.mmm-modelop+json": {"error": "Error fitting model: " + str(_e)}}
            )
            return
        _mmm_pdd({"application/vnd.mmm-modelop+json": _info})
except Exception:
    pass

# ── echo suppression: match exec()'s print()-only contract ──
try:
    get_ipython().ast_node_interactivity = "none"
except Exception:
    pass
"""
    return _PREAMBLE_IMPORTS + "\n" + _PREAMBLE_HELPERS + "\n" + extra


def _per_call_header(ctx: "KernelContext") -> str:
    """Reserved bindings re-applied every call (OUTPUT_DIR + dataset/df), mirroring
    the in-process precedence contract. Run as a separate silent cell so user-code
    tracebacks keep correct line numbers."""
    wd = ctx.work_dir or os.getcwd()
    # Re-chdir every cell so a user os.chdir can't permanently relocate outputs
    # (bare-name writes, save_result -> ./results) out of work_dir, matching the
    # in-process kernel which re-chdirs on every call.
    return (
        f"import os as _os; _os.chdir({wd!r})\n"
        f"OUTPUT_DIR = {wd!r}\n"
        f"__mmm_autobind_df({(ctx.dataset_path or None)!r})\n"
    )


class SubprocessKernel:
    """One ipykernel process per session (``per_session=True``)."""

    per_session = True

    def __init__(self):
        self._km = None
        self._kc = None
        self._lock = threading.Lock()
        self._started = False
        # recv quantum per get_*_msg (NOT a total cap); death/timeout is checked on it
        self._recv_timeout = float(os.environ.get("MMM_KERNEL_RECV_TIMEOUT", "30"))
        self._ready_timeout = float(os.environ.get("MMM_KERNEL_READY_TIMEOUT", "60"))
        # wall-clock cap per cell: interrupt, then after a grace kill — so a hung
        # cell can't wedge the session (and the lock) forever
        self._cell_timeout = float(os.environ.get("MMM_CELL_TIMEOUT", "600"))
        # fits are long; their own (larger) wall-clock cap
        self._fit_timeout = float(os.environ.get("MMM_FIT_TIMEOUT", "1800"))
        self._interrupt_grace = float(
            os.environ.get("MMM_KERNEL_INTERRUPT_GRACE", "15")
        )

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _start(self, ctx: "KernelContext") -> None:
        from jupyter_client.manager import KernelManager as _JKM

        self._km = _JKM(kernel_name="python3")
        self._km.start_kernel(cwd=ctx.work_dir or None)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self._ready_timeout)
        self._run(_build_startup_source(), silent=True, capture=False)
        self._started = True

    def _ensure_started(self, ctx: "KernelContext") -> None:
        if self._started and self._km is not None and self._km.is_alive():
            return
        self._teardown()
        self._start(ctx)

    def _teardown(self) -> None:
        try:
            if self._kc is not None:
                self._kc.stop_channels()  # avoids the __del__ GC warning
        except Exception:
            pass
        try:
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
        except Exception:
            pass
        self._km = self._kc = None
        self._started = False

    def reset(self) -> None:
        # Restart for a truly fresh namespace, then re-run startup on next call.
        with self._lock:
            try:
                if self._km is not None and self._km.is_alive():
                    self._km.restart_kernel(now=True)
                    self._kc.wait_for_ready(timeout=self._ready_timeout)
                    self._run(_build_startup_source(), silent=True, capture=False)
                    return
            except Exception:
                pass
            self._teardown()  # next execute() lazily respawns

    def shutdown(self) -> None:
        with self._lock:
            self._teardown()

    # ── protocol ─────────────────────────────────────────────────────────────
    def _run(self, code: str, *, silent: bool, capture: bool, cell_timeout=None):
        """Send one execute_request, drain iopub to idle, then read the matching
        shell execute_reply. Returns (stdout, plots, err). ``cell_timeout`` bounds
        wall-clock time: interrupt at the cap, kill after a grace if SIGINT is
        ignored (e.g. a compiled sampler)."""
        import time
        from queue import Empty

        msg_id = self._kc.execute(
            code, silent=silent, store_history=False, allow_stdin=False
        )
        out: list[str] = []
        plots: list = []
        modelops: list = []
        err = None
        idle = died = False
        interrupted_at = None
        start = time.monotonic()
        while not idle:
            try:
                msg = self._kc.get_iopub_msg(timeout=self._recv_timeout)
            except Empty:
                if self._km is None or not self._km.is_alive():
                    err = {
                        "ename": "KernelError",
                        "evalue": "kernel died",
                        "traceback": [],
                    }
                    died = True
                    break
                if cell_timeout:
                    if (
                        interrupted_at is None
                        and (time.monotonic() - start) > cell_timeout
                    ):
                        interrupted_at = time.monotonic()
                        try:
                            self._km.interrupt_kernel()  # SIGINT — works for pure Python
                        except Exception:
                            pass
                    elif (
                        interrupted_at is not None
                        and (time.monotonic() - interrupted_at) > self._interrupt_grace
                    ):
                        self._teardown()  # interrupt ignored -> kill; next call respawns
                        err = {
                            "ename": "TimeoutError",
                            "evalue": f"cell exceeded {int(cell_timeout)}s; interrupted and the kernel was killed",
                            "traceback": [],
                        }
                        died = True
                        break
                continue  # a long-running cell emits no iopub — keep waiting
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "status":
                if content.get("execution_state") == "idle":
                    idle = True
            elif mtype == "stream" and capture:
                out.append(content.get("text", ""))
            elif mtype in ("display_data", "execute_result") and capture:
                data = content.get("data", {})
                fig = data.get("application/vnd.plotly.v1+json")
                if fig is not None:
                    plots.append(fig)
                mo = data.get(_MODELOP_MIME)
                if mo is not None:
                    modelops.append(mo)
            elif mtype == "error":
                err = content
        # DONE = idle (iopub) AND the shell execute_reply. Skip when the kernel
        # died (no reply will come — avoids a full-timeout stall), and correlate
        # by msg_id so a stale reply can never be mispaired onto the next request.
        if not died:
            while True:
                try:
                    reply = self._kc.get_shell_msg(timeout=self._recv_timeout)
                except Empty:
                    break
                if reply.get("parent_header", {}).get("msg_id") == msg_id:
                    break
        return "".join(out), plots, modelops, err

    def execute(self, code: str, ctx: "KernelContext") -> ExecuteResult:
        import traceback as _tb

        from mmm_framework.agents.tools import format_execution_error

        # execute() is TOTAL — it never raises (matches InProcessKernel). A spawn
        # failure (jupyter import / start_kernel / wait_for_ready timeout) or any
        # protocol error returns an is_error result and tears down the half-built
        # kernel so the next call respawns cleanly.
        with self._lock:
            try:
                self._ensure_started(ctx)
                _, _, _, h_err = self._run(
                    _per_call_header(ctx), silent=True, capture=False
                )
                stdout, plots, _modelops, err = self._run(
                    code, silent=False, capture=True, cell_timeout=self._cell_timeout
                )
            except Exception:
                self._teardown()
                return ExecuteResult(
                    stdout=format_execution_error(_tb.format_exc()),
                    plots=[],
                    is_error=True,
                )

        warn = ""
        if h_err is not None:  # reserved-binding setup failed — surface, don't hide
            warn = f"[reserved-binding setup warning: {h_err.get('ename', 'error')}]\n"

        if err is not None:
            ename = err.get("ename", "")
            evalue = err.get("evalue", "")
            tb = _strip_ansi("\n".join(err.get("traceback") or []) or evalue)
            is_name = ename in ("NameError", "UnboundLocalError")
            missing = None
            if ename == "NameError":
                m = _NAME_RE.search(evalue)
                missing = m.group(1) if m else None
            body = format_execution_error(
                tb, is_name_error=is_name, missing_name=missing
            )
            # Truncate ONLY the captured prefix; the error body (with the
            # load-bearing "Error executing code" marker + hint) survives intact.
            prefix = (_truncate(stdout) + "\n") if stdout else ""
            return ExecuteResult(
                stdout=warn + prefix + body, plots=plots, is_error=True
            )

        if not stdout:
            stdout = "Code executed successfully with no output."
        return ExecuteResult(
            stdout=warn + _truncate(stdout), plots=plots, is_error=False
        )

    def run_model_op(self, op_name: str, kwargs: dict) -> dict:
        from mmm_framework.agents.model_ops import NO_MODEL_MSG

        with self._lock:
            # A model op only has a model if the kernel is already live (the fit
            # moves here in PR-C). A cold/unstarted kernel has no model -> no-model
            # (don't spawn just to discover that). The op result rides the
            # dedicated MIME channel, NOT stdout (which carries pymc/serializer
            # prints + progress bars), so a noisy compute can't corrupt it.
            if not (self._started and self._km is not None and self._km.is_alive()):
                return {"content": None, "dashboard": {}, "error": NO_MODEL_MSG}
            try:
                code = f"_mmm_emit_op({op_name!r}, {dict(kwargs or {})!r})\n"
                _, _, modelops, err = self._run(
                    code, silent=False, capture=True, cell_timeout=self._cell_timeout
                )
            except Exception as e:  # noqa: BLE001
                return {
                    "content": None,
                    "dashboard": {},
                    "error": f"model op transport error: {e}",
                }
        if err is not None:
            detail = err.get("evalue") or err.get("ename") or "error"
            return {
                "content": None,
                "dashboard": {},
                "error": f"model op error: {detail}",
            }
        if not modelops:
            return {
                "content": None,
                "dashboard": {},
                "error": "model op returned no result",
            }
        return _json_safe(modelops[0])

    def fit(self, model_spec: dict, dataset_path: str) -> dict:
        # Fit IN the kernel so the model becomes a kernel global (run_model_op /
        # execute_python then see it). The kernel spawns in the dataset's
        # directory (= the session workspace) so the report/auto-save land there.
        ctx = KernelContext(
            thread_id="",
            work_dir=os.path.dirname(os.path.abspath(dataset_path)) or None,
            dataset_path=dataset_path,
        )
        with self._lock:
            try:
                self._ensure_started(ctx)
                code = f"_mmm_emit_fit({model_spec!r}, {dataset_path!r})\n"
                _, _, modelops, err = self._run(
                    code, silent=False, capture=True, cell_timeout=self._fit_timeout
                )
            except Exception as e:  # noqa: BLE001
                self._teardown()
                return {"error": f"Error fitting model: {e}"}
        if err is not None:
            detail = err.get("evalue") or err.get("ename") or "error"
            return {"error": f"Error fitting model: {detail}"}
        if not modelops:
            return {"error": "fit returned no result"}
        return _json_safe(modelops[0])

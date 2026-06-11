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

import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

# Structured audit trail for kernel lifecycle + security events (Phase 3 PR-E.4).
# Single stdlib sink for now; the denied-egress/denied-syscall events and an
# off-host/tamper-evident sink are Tier 2 (PR-F) / Phase 4d.
_audit_log = logging.getLogger("mmm_audit")


def _audit(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one audit event (never raises). Carries structured ``extra`` so the
    tamper-evident sink (Phase 4d) records clean event/fields, and a human
    ``key=value`` message for plain log readers."""
    try:
        parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        _audit_log.log(
            level,
            "%s %s",
            event,
            parts,
            extra={"audit_event": event, "audit_fields": dict(fields)},
        )
    except Exception:
        pass


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
    tables: list = field(default_factory=list)
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
                inst = self._instances[key] = self._factories[self._impl](thread_id)
                # evict the least-recently-used kernel(s) beyond the cap
                while len(self._instances) > self._max:
                    _ek, _ev = self._instances.popitem(last=False)
                    if _ek == key:  # never evict the one we just created
                        self._instances[_ek] = _ev
                        break
                    _audit("kernel_evict_lru", key=_ek, max=self._max)
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

# Dedicated display_data MIME for structured tables published via the
# `show_table(df)` kernel binding — same rationale as the plot/model-op MIMEs.
_TABLE_MIME = "application/vnd.mmm-table+json"


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


# ── env scrub (Phase 3 PR-E.1) ────────────────────────────────────────────────
#
# The subprocess kernel runs untrusted, LLM-authored code; never hand it the API
# process's credentials. A hostile cell would otherwise read every *_API_KEY +
# the ADC path in one line (the framework reads them from os.environ — llm.py,
# embeddings.py). The kernel NEVER calls the LLM or the embedder (those run in
# the API process), so it needs no API keys / ADC at all.
#
# Strategy: a FAIL-CLOSED allowlist (drop everything not matched) generous enough
# not to break the PyTensor/JAX compile stack, with a secret-pattern denylist
# applied ON TOP (belt-and-suspenders — a secret can never pass even if a future
# allow rule widens). jupyter_client adds its own required vars (JPY_PARENT_PID,
# the interrupt event) on top of whatever base env we hand start_kernel, and an
# absent `env=` inherits the full os.environ — so this only ever tightens, and
# only for the subprocess kernel (the in-process kernel shares the API env and is
# unaffected).
#
# NOTE (load-bearing): this does NOT stop the cloud metadata server
# (169.254.169.254 / metadata.google.internal) — that is the real ADC-token-theft
# vector and is closed by EGRESS deny (Phase 3 Tier 2 / PR-F.4), not by env scrub.

_ENV_ALLOW_EXACT = frozenset(
    {
        # framework / kernel config the in-kernel code may resolve
        "MMM_AGENT_WORKSPACE",
        "MMM_AGENT_KERNEL",
        "MMM_MAX_KERNELS",
        "MMM_CELL_TIMEOUT",
        "MMM_FIT_TIMEOUT",
        "MMM_MODEL_CONFIG",
        # non-secret LLM/embed config (no keys/creds — those are denied below)
        "MMM_LLM_PROVIDER",
        "MMM_LLM_MODEL",
        "MMM_LLM_TEMPERATURE",
        "MMM_LLM_MAX_TOKENS",
        "MMM_LLM_PROJECT",
        "MMM_LLM_LOCATION",
        "MMM_LLM_BASE_URL",
        "MMM_EMBED_PROVIDER",
        "MMM_EMBED_MODEL",
        "MMM_EMBED_LOCATION",
        "GOOGLE_CLOUD_PROJECT",  # project identity (a string, not a credential)
        # system
        "PATH",
        "HOME",
        "TMPDIR",
        "TMP",
        "USER",
        "LOGNAME",
        "LANG",
        "TZ",
        "TERM",
        "SHELL",
        # toolchain (PyTensor/numpy compile + link)
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPATH",
        "CPPFLAGS",
        "LDFLAGS",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "PKG_CONFIG_PATH",
    }
)

_ENV_ALLOW_PREFIX = (
    "MMM_KERNEL_",  # recv/ready/interrupt-grace knobs
    "LC_",
    "PYTHON",  # PYTHONPATH etc. (NOT a secret prefix)
    "PYTENSOR",
    "JAX",
    "XLA",
    "OMP",
    "MKL",
    "NUMBA",
    "OPENBLAS",
    "CONDA_",
    "VIRTUAL_ENV",
    "MPLCONFIGDIR",
    "DYLD_",  # macOS dynamic linker (PyTensor clang link path)
)

# Matched against the FULL name (case-insensitive); a match drops the var even if
# an allow rule would otherwise keep it.
_ENV_DENY_RE = re.compile(
    r"(_API_KEY|_APIKEY|_TOKEN|_SECRET|_CREDENTIALS?|_PASSWORD|_PRIVATE_KEY)$"
    r"|^GOOGLE_APPLICATION_CREDENTIALS$"
    r"|^MMM_LLM_API_KEY$"
    r"|^MMM_LLM_CREDENTIALS_PATH$"
    r"|^(AWS|AZURE|GCP)_",
    re.IGNORECASE,
)


def _scrubbed_kernel_env() -> "dict[str, str] | None":
    """The env to launch the subprocess kernel with: the API env minus secrets.

    Returns ``None`` when scrubbing is disabled (``MMM_KERNEL_SCRUB_ENV=0``,
    debug only) so the caller omits ``env=`` and the kernel inherits the full
    ``os.environ`` exactly as before. Otherwise returns a fail-closed allowlisted
    copy. ``MMM_KERNEL_ENV_PASSTHROUGH`` (comma-separated exact names) extends the
    allowlist for the rare legitimately-needed var; the denylist still wins.
    """
    if os.environ.get("MMM_KERNEL_SCRUB_ENV", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return None
    extra = {
        n.strip()
        for n in (os.environ.get("MMM_KERNEL_ENV_PASSTHROUGH", "") or "").split(",")
        if n.strip()
    }
    out: dict[str, str] = {}
    for name, value in os.environ.items():
        if _ENV_DENY_RE.search(name):  # secret pattern — never passes
            continue
        if (
            name in _ENV_ALLOW_EXACT
            or name in extra
            or name.startswith(_ENV_ALLOW_PREFIX)
        ):
            out[name] = value
    return out


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

# ── table capture: show_table(df) -> display_data(structured table json) ──
try:
    from IPython.display import publish_display_data as _mmm_tbl_pdd
    from mmm_framework.agents.tables import df_to_table_json as _mmm_df_to_table

    def show_table(df, title=None, group="repl"):
        \"\"\"Render a DataFrame as a formatted, sortable table in the dashboard
        (instead of printing it). Returns None.\"\"\"
        _payload = _mmm_df_to_table(
            df,
            title=str(title or "Table"),
            source="execute_python",
            group=str(group or "repl"),
        )
        _mmm_tbl_pdd({"application/vnd.mmm-table+json": _payload})
except Exception:
    pass

# ── model-op driver (Phase 2 PR-B): run a model_ops op on the in-kernel model ──
try:
    from mmm_framework.agents import model_ops as _mmm_mo
    from IPython.display import publish_display_data as _mmm_pdd

    def _mmm_rehydrate():
        # Cold-reload: a respawned (LRU-evicted) kernel has no `mmm` global, but
        # the session's last fit is on disk under <cwd>/mmm_models. Load the
        # latest, rebuilding the panel from the spec+dataset persisted in
        # run_metadata.json. Returns the model (and sets the globals) or None.
        global mmm, results
        import os as _os
        import json as _json
        import glob as _glob

        try:
            _dirs = sorted(_glob.glob(_os.path.join("mmm_models", "run_*")))
            if not _dirs:
                return None
            _latest = _dirs[-1]
            with open(_os.path.join(_latest, "run_metadata.json")) as _f:
                _meta = _json.load(_f)
            _spec = _meta.get("spec")
            _dsp = _meta.get("dataset_path")
            if not _spec or not _dsp:
                return None
            from mmm_framework.agents.fitting import _mff_config_from_spec
            from mmm_framework import load_mff
            from mmm_framework.serialization import MMMSerializer

            _panel = load_mff(_dsp, _mff_config_from_spec(_spec))
            mmm = MMMSerializer.load(_latest, _panel)
            results = None
            return mmm
        except Exception:
            return None

    def _mmm_run_op(_name, _kw):
        _op = _mmm_mo.OPS.get(_name)
        if _op is None:
            return {"content": None, "dashboard": {}, "error": "Unknown model op: " + str(_name)}
        _m = globals().get("mmm")
        if _m is None:
            _m = _mmm_rehydrate()  # cold kernel -> try loading the last fit from disk
        if _m is None and not getattr(_op, "allow_unfitted", False):
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

    def __init__(self, thread_id: str | None = None):
        # The session this kernel serves — so it always spawns in the SAME
        # work_dir (the session's thread_dir) for fit / run_model_op / execute,
        # which is what lets a respawned (LRU-evicted) cold kernel rehydrate the
        # model from disk (PR-C.3).
        self._thread_id = thread_id
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
    def _work_dir(self) -> str | None:
        """This session's workspace dir (the kernel's cwd) — where its dataset,
        auto-saved models, and outputs live."""
        if not self._thread_id:
            return None
        try:
            from mmm_framework.agents import workspace as _ws

            return str(_ws.thread_dir(self._thread_id))
        except Exception:
            return None

    def _start(self) -> None:
        from jupyter_client.manager import KernelManager as _JKM

        self._km = _JKM(kernel_name="python3")
        # Scrub secrets from the kernel's environment (Phase 3 PR-E.1). A None
        # return (scrubbing disabled) means omit env= so the kernel inherits the
        # full os.environ exactly as before.
        _env = _scrubbed_kernel_env()
        self._km.start_kernel(
            cwd=self._work_dir() or None, **({} if _env is None else {"env": _env})
        )
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self._ready_timeout)
        self._run(_build_startup_source(), silent=True, capture=False)
        self._started = True
        _audit("kernel_spawn", thread_id=self._thread_id)

    def _ensure_started(self) -> None:
        if self._started and self._km is not None and self._km.is_alive():
            return
        if self._started:  # was up but died/was killed -> cold respawn
            _audit("kernel_respawn", level=logging.WARNING, thread_id=self._thread_id)
        self._teardown()
        self._start()

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
        tables: list = []
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
                    _audit(
                        "kernel_died", level=logging.WARNING, thread_id=self._thread_id
                    )
                    break
                if cell_timeout:
                    if (
                        interrupted_at is None
                        and (time.monotonic() - start) > cell_timeout
                    ):
                        interrupted_at = time.monotonic()
                        _audit(
                            "kernel_timeout_interrupt",
                            level=logging.WARNING,
                            thread_id=self._thread_id,
                            cell_timeout=int(cell_timeout),
                        )
                        try:
                            self._km.interrupt_kernel()  # SIGINT — works for pure Python
                        except Exception:
                            pass
                    elif (
                        interrupted_at is not None
                        and (time.monotonic() - interrupted_at) > self._interrupt_grace
                    ):
                        _audit(
                            "kernel_timeout_kill",
                            level=logging.WARNING,
                            thread_id=self._thread_id,
                            cell_timeout=int(cell_timeout),
                        )
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
                tb = data.get(_TABLE_MIME)
                if tb is not None:
                    tables.append(tb)
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
        return "".join(out), plots, modelops, tables, err

    def execute(self, code: str, ctx: "KernelContext") -> ExecuteResult:
        import traceback as _tb

        from mmm_framework.agents.tools import format_execution_error

        # execute() is TOTAL — it never raises (matches InProcessKernel). A spawn
        # failure (jupyter import / start_kernel / wait_for_ready timeout) or any
        # protocol error returns an is_error result and tears down the half-built
        # kernel so the next call respawns cleanly.
        with self._lock:
            try:
                self._ensure_started()
                _, _, _, _, h_err = self._run(
                    _per_call_header(ctx), silent=True, capture=False
                )
                stdout, plots, _modelops, tables, err = self._run(
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
                stdout=warn + prefix + body, plots=plots, tables=tables, is_error=True
            )

        if not stdout:
            stdout = "Code executed successfully with no output."
        return ExecuteResult(
            stdout=warn + _truncate(stdout), plots=plots, tables=tables, is_error=False
        )

    def run_model_op(self, op_name: str, kwargs: dict) -> dict:
        from mmm_framework.agents.model_ops import NO_MODEL_MSG

        with self._lock:
            # Ensure the kernel is live (spawning in the session's work_dir), then
            # run the op. If the kernel is cold (LRU-evicted/respawned) the op's
            # driver rehydrates the model from disk first (PR-C.3). The op result
            # rides the dedicated MIME channel, NOT stdout (which carries
            # pymc/serializer prints + progress bars), so a noisy compute can't
            # corrupt it.
            try:
                self._ensure_started()
                code = f"_mmm_emit_op({op_name!r}, {dict(kwargs or {})!r})\n"
                _, _, modelops, _, err = self._run(
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
        # execute_python then see it). The kernel runs in the session work_dir, so
        # the auto-save lands in <work_dir>/mmm_models — findable by a later
        # (cold) kernel for the same session to rehydrate from.
        # fit_start/done bracket the expensive op -> feeds the active_fits metric
        # (Phase 4d) the autoscaler scales on (§5.1).
        _audit("kernel_fit_start", thread_id=self._thread_id)
        with self._lock:
            try:
                self._ensure_started()
                code = f"_mmm_emit_fit({model_spec!r}, {dataset_path!r})\n"
                _, _, modelops, _, err = self._run(
                    code, silent=False, capture=True, cell_timeout=self._fit_timeout
                )
            except Exception as e:  # noqa: BLE001
                self._teardown()
                _audit("kernel_fit_done", thread_id=self._thread_id)
                return {"error": f"Error fitting model: {e}"}
        _audit("kernel_fit_done", thread_id=self._thread_id)
        if err is not None:
            detail = err.get("evalue") or err.get("ename") or "error"
            return {"error": f"Error fitting model: {detail}"}
        if not modelops:
            return {"error": "fit returned no result"}
        return _json_safe(modelops[0])

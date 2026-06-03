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

import threading
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
        self._instances: dict[str, Kernel] = {}
        self._lock = threading.RLock()

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
            return inst
        with self._lock:  # double-checked: never spawn two kernels for one session
            inst = self._instances.get(key)
            if inst is None:
                inst = self._instances[key] = self._factories[self._impl]()
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

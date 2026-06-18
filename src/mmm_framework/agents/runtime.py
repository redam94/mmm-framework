"""Per-request runtime context for the MMM agent.

Two pieces of state need to follow the *active session* through a chat turn
without being threaded explicitly through every tool signature:

* ``current_thread_id`` — the LangGraph ``thread_id`` of the session currently
  being served. Set at the top of the ``/chat`` handler **and** at the top of
  every model-/workspace-using tool (belt + suspenders: LangGraph may run sync
  tools in an executor where the ContextVar does not auto-propagate).
* ``MODEL_CACHE`` — the fitted ``BayesianMMM`` / ``MMMResults`` objects. These
  cannot live in LangGraph state (msgpack can't serialize PyMC objects), so they
  are kept in a process-global cache. The cache is **scoped by thread** and
  bounded by an LRU so two concurrent sessions don't clobber each other and
  memory stays capped. Durable persistence is via ``MMMSerializer`` on disk.

The cache exposes the small ``dict``-like surface the existing tools already use
(``.get(key)``, ``cache[key]``, ``cache[key] = v``, ``key in cache``) so it is a
drop-in replacement for the old module-global ``_MODEL_CACHE = {}``.
"""

from __future__ import annotations

import contextvars
import threading
from collections import OrderedDict
from typing import Any

# ── Active-thread context ───────────────────────────────────────────────────

current_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mmm_agent_thread_id", default=None
)

_DEFAULT_BUCKET = "__default__"


def set_current_thread(thread_id: str | None) -> contextvars.Token:
    """Mark ``thread_id`` as the active session for the current context.

    Returns the ContextVar token so callers may ``reset`` it if they wish; most
    callers don't bother because each ``/chat`` request runs in its own asyncio
    task (and tool invocations re-set it defensively).
    """
    return current_thread_id.set(thread_id)


def get_current_thread() -> str:
    """The active ``thread_id``, or ``"__default__"`` when none is set."""
    return current_thread_id.get() or _DEFAULT_BUCKET


# ── Thread-scoped, LRU-bounded model cache ──────────────────────────────────


class _ThreadScopedModelCache:
    """A ``dict``-like cache whose backing store is selected by the active thread.

    Each thread gets its own bucket (a plain ``dict``). Buckets are kept in an
    ``OrderedDict`` ordered by recency; when more than ``maxsize`` threads have
    live models the least-recently-used bucket is dropped (its model is GC'd —
    the durable copy on disk via ``MMMSerializer`` remains and can be reloaded
    with ``load_fitted_model``).
    """

    def __init__(self, maxsize: int = 2) -> None:
        self._buckets: "OrderedDict[str, dict]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        self._lock = threading.RLock()

    def _bucket(self) -> dict:
        tid = get_current_thread()
        with self._lock:
            if tid in self._buckets:
                self._buckets.move_to_end(tid)
            else:
                self._buckets[tid] = {}
                while len(self._buckets) > self._maxsize:
                    self._buckets.popitem(last=False)
            return self._buckets[tid]

    # dict-compatible surface used by the tools ----------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._bucket().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._bucket()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._bucket()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._bucket()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._bucket()

    def pop(self, key: str, *default: Any) -> Any:
        return self._bucket().pop(key, *default)

    # introspection helpers ------------------------------------------------
    def has_model(self, thread_id: str | None = None) -> bool:
        with self._lock:
            tid = thread_id or get_current_thread()
            return bool(self._buckets.get(tid, {}).get("fitted_model"))

    def clear_thread(self, thread_id: str) -> None:
        with self._lock:
            self._buckets.pop(thread_id, None)

    def thread_ids(self) -> list[str]:
        with self._lock:
            return list(self._buckets.keys())


MODEL_CACHE = _ThreadScopedModelCache(maxsize=2)


# ── Thread-scoped REPL namespace ("warm kernel") ────────────────────────────


class _ThreadScopedNamespace(_ThreadScopedModelCache):
    """Per-thread persistent namespace for the ``execute_python`` warm kernel.

    Same thread-scoping + LRU-over-threads semantics as ``MODEL_CACHE``: a
    variable defined in one ``execute_python`` call survives into the next call
    **in the same live process** — so the agent can build up an analysis
    incrementally the way every notebook / REPL works, instead of each call
    starting from a blank slate.

    When the bucket is evicted (LRU) or the process restarts the namespace goes
    cold (empty). Durability of *specific* objects across that boundary is the
    job of the ``save_result``/``load_result`` helpers, which persist named
    objects to the on-disk workspace (mirroring how ``MMMSerializer`` is the
    durable fallback for the model cache). This is NOT a security or isolation
    boundary — like ``MODEL_CACHE`` it is an in-process convenience cache.
    """

    def namespace(self) -> dict:
        """The current thread's namespace dict (created on demand)."""
        return self._bucket()

    def reset(self, thread_id: str | None = None) -> None:
        """Clear the current thread's namespace (start a fresh kernel)."""
        with self._lock:
            tid = thread_id or get_current_thread()
            self._buckets[tid] = {}
            self._buckets.move_to_end(tid)


NAMESPACE_CACHE = _ThreadScopedNamespace(maxsize=2)

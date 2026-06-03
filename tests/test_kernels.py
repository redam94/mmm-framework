"""Tests for the Kernel abstraction (Phase 1 of agent-session-kernels.md).

PR2 covers the KernelManager seam + the default in-process routing. The
SubprocessKernel matrix (smoke / fit-boundary / plot-fidelity / echo) lands in
PR3.
"""

from mmm_framework.agents.kernels import ExecuteResult, KernelManager


class _Shared:
    per_session = False

    def execute(self, code, ctx):
        return ExecuteResult(stdout="ok")

    def reset(self):
        pass

    def shutdown(self):
        pass


class _PerSession:
    per_session = True

    def __init__(self):
        self.events = []

    def execute(self, code, ctx):
        return ExecuteResult(stdout="ok")

    def reset(self):
        self.events.append("reset")

    def shutdown(self):
        self.events.append("shutdown")


def test_manager_shares_non_per_session_instance():
    """A non-per_session impl (in-process) is shared across threads — its
    per-thread state lives in NAMESPACE_CACHE, not the instance."""
    m = KernelManager("inprocess", {"inprocess": _Shared})
    assert m.get_or_spawn("t1") is m.get_or_spawn("t2")


def test_manager_per_session_instance_per_thread():
    """A per_session impl (subprocess) gets one cached instance per thread_id."""
    m = KernelManager("sub", {"sub": _PerSession})
    a1, a1b, a2 = m.get_or_spawn("t1"), m.get_or_spawn("t1"), m.get_or_spawn("t2")
    assert a1 is a1b and a1 is not a2


def test_manager_reset_and_shutdown_route_to_instance():
    m = KernelManager("sub", {"sub": _PerSession})
    k = m.get_or_spawn("t1")
    m.reset("t1")
    assert k.events == ["reset"]
    m.shutdown("t1")
    assert k.events == ["reset", "shutdown"]
    m.shutdown("t1")  # already gone — must be a no-op, not an error
    # a fresh instance is spawned after shutdown
    assert m.get_or_spawn("t1") is not k


def test_unknown_impl_falls_back_to_inprocess():
    m = KernelManager("bogus", {"inprocess": _Shared})
    assert m.impl == "inprocess"


def test_execute_python_routes_through_inprocess_kernel():
    """The default manager is the in-process kernel; execute_python dispatches
    through it (the existing execute_python tests exercise the behavior)."""
    from mmm_framework.agents import tools as T

    assert T._KERNELS.impl == "inprocess"
    assert T.InProcessKernel.per_session is False

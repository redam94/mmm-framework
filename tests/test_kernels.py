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


# ── PR3: SubprocessKernel (spawns a real ipykernel; verifies the protocol) ────

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def subk(tmp_path_factory):
    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    from mmm_framework.agents.kernels import SubprocessKernel

    wd = str(tmp_path_factory.mktemp("subk_ws"))
    k = SubprocessKernel()
    yield k, wd
    k.shutdown()


def _ctx(wd, dataset_path=None, mmm=None):
    from mmm_framework.agents.kernels import KernelContext

    return KernelContext(thread_id="t", work_dir=wd, dataset_path=dataset_path, mmm=mmm)


def test_subprocess_persists_and_suppresses_echo(subk):
    k, wd = subk
    r1 = k.execute("a_sp = 7\nprint('set', a_sp)", _ctx(wd))
    assert not r1.is_error and "set 7" in r1.stdout
    # var persists into the next cell; a bare trailing expression is NOT echoed
    r2 = k.execute("print('a_sp is', a_sp + 1)\na_sp + 100", _ctx(wd))
    assert not r2.is_error and "a_sp is 8" in r2.stdout
    assert "107" not in r2.stdout  # echo suppression (print()-only contract)


def test_subprocess_nameerror_hint(subk):
    k, wd = subk
    r = k.execute("print(does_not_exist_xyz)", _ctx(wd))
    assert r.is_error
    assert "Error executing code" in r.stdout  # load-bearing substring
    assert "`does_not_exist_xyz`" in r.stdout and "load_result" in r.stdout


def test_subprocess_mmm_unavailable_boundary(subk):
    """Documented Phase-1 boundary: fits run in-process, so mmm/results are NOT
    in the subprocess. Referencing mmm raises NameError; Phase 2 removes this."""
    k, wd = subk
    r = k.execute("print(mmm)", _ctx(wd, mmm=object()))  # ctx.mmm set but NOT bound
    assert r.is_error and "`mmm`" in r.stdout


def test_subprocess_df_autobind(subk):
    k, wd = subk
    (Path(wd) / "sub_data.csv").write_text("a\n1\n2\n3\n")
    r = k.execute(
        "print('n', len(df)); print('cols', list(df.columns))",
        _ctx(wd, dataset_path="sub_data.csv"),
    )
    assert not r.is_error and "n 3" in r.stdout and "cols ['a']" in r.stdout


def test_subprocess_plot_fidelity_matches_inprocess(subk):
    """The same figure yields byte-identical normalized JSON via both kernels, so
    the content-addressed plot store dedups across implementations."""
    from mmm_framework.agents.tools import InProcessKernel

    k, wd = subk
    code = "import plotly.express as px\npx.bar(x=['a', 'b'], y=[1, 2]).show()"
    sp = k.execute(code, _ctx(wd))
    ip = InProcessKernel().execute(code, _ctx(wd))
    assert len(sp.plots) == 1 and len(ip.plots) == 1
    assert json.dumps(sp.plots[0], sort_keys=True) == json.dumps(
        ip.plots[0], sort_keys=True
    )


def test_subprocess_reset_clears_namespace(subk):
    k, wd = subk
    k.execute("reset_me = 123", _ctx(wd))
    assert "123" in k.execute("print(reset_me)", _ctx(wd)).stdout
    k.reset()
    r = k.execute("print(reset_me)", _ctx(wd))
    assert r.is_error and "`reset_me`" in r.stdout


# ── PR3 review fixes: parity + robustness the happy-path tests missed ─────────


def test_subprocess_binds_os_matplotlib_plt(subk):
    k, wd = subk
    r = k.execute(
        "print('os', bool(os.getcwd()))\n"
        "print('plt', hasattr(plt, 'plot'))\n"
        "print('backend', matplotlib.get_backend())",
        _ctx(wd),
    )
    assert not r.is_error, r.stdout
    assert "os True" in r.stdout and "plt True" in r.stdout
    assert "agg" in r.stdout.lower()


def test_subprocess_stderr_stdout_order_preserved(subk):
    k, wd = subk
    r = k.execute(
        "import sys\nprint('AA'); sys.stderr.write('BB\\n'); print('CC')", _ctx(wd)
    )
    assert not r.is_error
    assert r.stdout.index("AA") < r.stdout.index("BB") < r.stdout.index("CC")


def test_subprocess_rechdir_keeps_outputs_in_workdir(subk, tmp_path):
    k, wd = subk
    other = str(tmp_path / "elsewhere")
    Path(other).mkdir(parents=True, exist_ok=True)
    k.execute(f"import os; os.chdir({other!r})", _ctx(wd))  # user wanders off
    k.execute(
        "open('rechdir_probe.txt', 'w').write('hi')", _ctx(wd)
    )  # re-chdir -> work_dir
    assert (Path(wd) / "rechdir_probe.txt").exists()


def test_subprocess_error_marker_survives_truncation(subk, monkeypatch):
    """A cell that prints a lot then raises must still record is_error — the error
    body (with 'Error executing code') is appended AFTER truncating the prefix."""
    from mmm_framework.agents import kernels as K

    monkeypatch.setattr(
        K, "_truncate", lambda s, limit=40: s[:40] if len(s) > 40 else s
    )
    k, wd = subk
    r = k.execute("print('p' * 500)\nundefined_zzz", _ctx(wd))
    assert r.is_error
    assert "Error executing code" in r.stdout and "`undefined_zzz`" in r.stdout


def test_subprocess_execute_is_total_on_spawn_failure(monkeypatch, tmp_path):
    """execute() never raises (matches InProcess) even if spawn fails."""
    from mmm_framework.agents.kernels import SubprocessKernel

    k = SubprocessKernel()
    monkeypatch.setattr(
        k, "_start", lambda ctx: (_ for _ in ()).throw(RuntimeError("spawn boom"))
    )
    r = k.execute("print('hi')", _ctx(str(tmp_path)))
    assert r.is_error and "Error executing code" in r.stdout  # returned, did not raise


def test_subprocess_list_saved_results_bound(subk):
    k, wd = subk
    r = k.execute(
        "save_result('sr_demo', pd.DataFrame({'a': [1]}))\n"
        "print(sorted(list_saved_results()))",
        _ctx(wd),
    )
    assert not r.is_error and "sr_demo" in r.stdout


def test_subprocess_cell_timeout_interrupts_hung_cell(tmp_path):
    """The wall-clock cap must interrupt a hung cell so it can't wedge the session
    forever (the blocker the happy-path tests didn't cover)."""
    from mmm_framework.agents.kernels import SubprocessKernel

    k = SubprocessKernel()
    k._recv_timeout = 0.5
    k._cell_timeout = 1.5  # interrupt a hung cell after ~1.5s
    k._interrupt_grace = 5
    try:
        r = k.execute("while True:\n    pass", _ctx(str(tmp_path)))
        assert r.is_error and "Error executing code" in r.stdout
    finally:
        k.shutdown()

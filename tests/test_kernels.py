"""Tests for the Kernel abstraction (Phase 1 of agent-session-kernels.md).

PR2 covers the KernelManager seam + the default in-process routing. The
SubprocessKernel matrix (smoke / fit-boundary / plot-fidelity / echo) lands in
PR3.
"""

from mmm_framework.agents.kernels import ExecuteResult, KernelManager


class _Shared:
    per_session = False

    def __init__(self, thread_id=None):
        self.thread_id = thread_id

    def execute(self, code, ctx):
        return ExecuteResult(stdout="ok")

    def reset(self):
        pass

    def shutdown(self):
        pass


class _PerSession:
    per_session = True

    def __init__(self, thread_id=None):
        self.thread_id = thread_id
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


# ── PR-B: run_model_op dispatch + the model-op MIME result channel ────────────


def test_json_safe_sanitizes_nan_inf_numpy():
    import numpy as np
    from mmm_framework.agents.kernels import _json_safe

    out = _json_safe(
        {
            "a": float("nan"),
            "b": [1.0, float("inf"), -float("inf")],
            "c": np.float64(0.5),
            "d": "x",
            "e": True,
        }
    )
    assert out == {"a": None, "b": [1.0, None, None], "c": 0.5, "d": "x", "e": True}


def test_inprocess_run_model_op_no_model_and_unknown():
    from mmm_framework.agents import tools as T

    k = T._KERNELS.get_or_spawn("t_rmo")  # shared in-process kernel
    T._MODEL_CACHE.clear_thread("t_rmo")
    T.set_current_thread("t_rmo")
    assert "No fitted model" in k.run_model_op("roi_metrics", {})["error"]
    assert "Unknown model op" in k.run_model_op("does_not_exist", {})["error"]
    T.set_current_thread(None)


def test_subprocess_run_model_op_no_model_when_cold(tmp_path, monkeypatch):
    from mmm_framework.agents.kernels import SubprocessKernel
    from mmm_framework.agents.model_ops import NO_MODEL_MSG

    # Fresh session: the kernel spawns + tries to rehydrate from disk, finds no
    # saved model, and returns no-model.
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    k = SubprocessKernel(thread_id="t_empty")
    try:
        assert k.run_model_op("roi_metrics", {})["error"] == NO_MODEL_MSG
    finally:
        k.shutdown()


def test_subprocess_run_model_op_channel_roundtrip(subk):
    """An op runs IN the kernel on the in-kernel model and its result crosses the
    dedicated MIME channel (off stdout). A fake model makes the compute fail, so
    we get the op's error-as-data back — proving in-kernel execution + transport
    of a structured dict, uncorrupted by any stdout noise."""
    k, wd = subk
    k.execute("mmm = 'FAKE_MODEL'", _ctx(wd))  # inject a (non-real) model global
    res = k.run_model_op("roi_metrics", {})
    assert res["content"] is None and res["dashboard"] == {}
    assert "Error computing ROI" in res["error"]
    k.execute("del mmm", _ctx(wd))  # clean up the shared kernel


# ── Phase 3 PR-E.1: env scrub (the kernel runs untrusted code — no secrets) ────

_SECRETS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "MMM_LLM_API_KEY",
    "MMM_LLM_CREDENTIALS_PATH",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_CLIENT_SECRET",
    "SOME_VENDOR_TOKEN",
    "DB_PASSWORD",
    "FOO_CREDENTIALS",
)
_BENIGN = {
    "MMM_LLM_MODEL": "claude-benign",
    "MMM_LLM_PROVIDER": "vertex_anthropic",
    "MMM_AGENT_WORKSPACE": "/ws",
    "GOOGLE_CLOUD_PROJECT": "my-proj",  # project id is not a credential
    "PYTENSOR_FLAGS": "cxx=/usr/bin/clang++",
    "OMP_NUM_THREADS": "4",
    "MMM_KERNEL_RECV_TIMEOUT": "30",
}


def test_scrubbed_kernel_env_drops_secrets_keeps_benign(monkeypatch):
    from mmm_framework.agents.kernels import _scrubbed_kernel_env

    monkeypatch.setenv("MMM_KERNEL_SCRUB_ENV", "1")
    monkeypatch.delenv("MMM_KERNEL_ENV_PASSTHROUGH", raising=False)
    for s in _SECRETS:
        monkeypatch.setenv(s, "SECRET-VALUE")
    for k, v in _BENIGN.items():
        monkeypatch.setenv(k, v)

    env = _scrubbed_kernel_env()
    assert env is not None
    for s in _SECRETS:
        assert s not in env, f"secret leaked: {s}"
    for k, v in _BENIGN.items():
        assert env.get(k) == v, f"benign dropped: {k}"
    assert "PATH" in env  # needed to even boot the kernel


def test_scrubbed_kernel_env_disabled_returns_none(monkeypatch):
    """The opt-out (debug only) returns None so the caller omits env= and the
    kernel inherits the full os.environ exactly as before."""
    from mmm_framework.agents.kernels import _scrubbed_kernel_env

    monkeypatch.setenv("MMM_KERNEL_SCRUB_ENV", "0")
    assert _scrubbed_kernel_env() is None


def test_scrubbed_kernel_env_passthrough_extends_but_deny_still_wins(monkeypatch):
    from mmm_framework.agents.kernels import _scrubbed_kernel_env

    monkeypatch.setenv("MMM_KERNEL_SCRUB_ENV", "1")
    monkeypatch.setenv("WEIRD_NEEDED_VAR", "ok")  # not in any allowlist
    monkeypatch.setenv("STILL_A_TOKEN", "nope")  # passthrough must NOT rescue a secret
    monkeypatch.setenv("MMM_KERNEL_ENV_PASSTHROUGH", "WEIRD_NEEDED_VAR, STILL_A_TOKEN")
    env = _scrubbed_kernel_env()
    assert env.get("WEIRD_NEEDED_VAR") == "ok"
    assert "STILL_A_TOKEN" not in env  # denylist wins over passthrough


def test_subprocess_env_scrub_hides_secrets_from_cell(tmp_path, monkeypatch):
    """End-to-end: a real subprocess kernel spawned while secrets are in the API
    env cannot see them in its own os.environ; benign config still arrives."""
    import json as _json

    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    from mmm_framework.agents.kernels import SubprocessKernel

    monkeypatch.setenv("MMM_KERNEL_SCRUB_ENV", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-SECRET-anthropic")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")
    monkeypatch.setenv("MMM_LLM_API_KEY", "sk-SECRET-mmm")
    monkeypatch.setenv("MMM_LLM_MODEL", "claude-benign")  # benign config kept

    k = SubprocessKernel()
    try:
        r = k.execute(
            "import os, json\n"
            "print(json.dumps({"
            "'anthropic': os.environ.get('ANTHROPIC_API_KEY'),"
            "'adc': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),"
            "'mmm_key': os.environ.get('MMM_LLM_API_KEY'),"
            "'mmm_model': os.environ.get('MMM_LLM_MODEL')}))",
            _ctx(str(tmp_path)),
        )
        assert not r.is_error, r.stdout
        data = _json.loads(r.stdout.strip().splitlines()[-1])
        assert data["anthropic"] is None
        assert data["adc"] is None
        assert data["mmm_key"] is None
        assert data["mmm_model"] == "claude-benign"
    finally:
        k.shutdown()


def test_subprocess_env_scrub_survives_reset(tmp_path, monkeypatch):
    """The scrub must hold across reset_namespace too — reset() goes through
    restart_kernel(), which reuses the cached scrubbed env= (NOT os.environ).
    This is a user-reachable path, so pin it so a refactor can't regress it."""
    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    from mmm_framework.agents.kernels import SubprocessKernel

    monkeypatch.setenv("MMM_KERNEL_SCRUB_ENV", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-SECRET-reset")
    k = SubprocessKernel()
    try:
        assert not k.execute("x = 1", _ctx(str(tmp_path))).is_error  # initial spawn
        k.reset()  # restart_kernel path
        r = k.execute(
            "import os; print(repr(os.environ.get('ANTHROPIC_API_KEY')))",
            _ctx(str(tmp_path)),
        )
        assert not r.is_error, r.stdout
        assert "None" in r.stdout  # secret still absent after reset
    finally:
        k.shutdown()


# ── Phase 3 PR-E.4: audit logging ─────────────────────────────────────────────


def test_kernel_manager_audits_eviction(caplog):
    import logging

    from mmm_framework.agents.kernels import KernelManager

    m = KernelManager("sub", {"sub": _PerSession})
    m._max = 1
    with caplog.at_level(logging.INFO, logger="mmm_audit"):
        m.get_or_spawn("t1")
        m.get_or_spawn("t2")  # over the cap -> evicts t1
    msgs = [r.getMessage() for r in caplog.records if r.name == "mmm_audit"]
    assert any("kernel_evict_lru" in m and "key=t1" in m for m in msgs)


def test_subprocess_audits_spawn(caplog, tmp_path):
    import logging

    pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")
    from mmm_framework.agents.kernels import SubprocessKernel

    with caplog.at_level(logging.INFO, logger="mmm_audit"):
        k = SubprocessKernel(thread_id="t_audit")
        try:
            r = k.execute("x = 1", _ctx(str(tmp_path)))
            assert not r.is_error, r.stdout
        finally:
            k.shutdown()
    msgs = [r.getMessage() for r in caplog.records if r.name == "mmm_audit"]
    assert any("kernel_spawn" in m for m in msgs)


# ── Phase 3 PR-F.2: ContainerKernel (runs the kernel inside `podman run`) ──────


def _container_runtime():
    """The runtime bin iff a container kernel can actually run here (podman + the
    built image present). Returns None -> the container tests skip."""
    import os
    import shutil
    import subprocess

    b = (
        os.environ.get("MMM_KERNEL_RUNTIME_BIN")
        or shutil.which("podman")
        or "/opt/podman/bin/podman"
    )
    if not os.path.exists(b):
        return None
    img = os.environ.get("MMM_KERNEL_IMAGE", "mmm-kernel:latest")
    try:
        if subprocess.run([b, "image", "exists", img]).returncode != 0:
            return None
    except Exception:
        return None
    return b


@pytest.mark.slow
def test_container_kernel_executes_writes_workspace_and_captures_plot(monkeypatch):
    """The kernel runs inside a podman container: code executes non-root, the
    bind-mounted session workspace is writable (host-visible), state persists
    across cells, and an in-container plot is captured over the MIME channel."""
    import os
    import shutil
    import uuid as _uuid

    runtime = _container_runtime()
    if not runtime:
        pytest.skip("podman + mmm-kernel image not available")

    # Workspace must be under the user home (podman-machine shares it; the sandbox
    # tmp is not bind-mountable). Unique dir, cleaned up after.
    ws = os.path.join(
        os.path.expanduser("~"), ".cache", "mmm-ktest-" + _uuid.uuid4().hex[:8]
    )
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", ws)
    monkeypatch.setenv("MMM_KERNEL_RUNTIME_BIN", runtime)
    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "tcp")
    monkeypatch.setenv("MMM_KERNEL_READY_TIMEOUT", "120")

    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.container_kernel import ContainerKernel

    tid = "ctest"
    wd = str(W.thread_dir(tid))
    k = ContainerKernel(tid)
    try:
        r = k.execute(
            "import os\nprint('uid', os.getuid())\n"
            "open('probe.txt','w').write('ok')\nprint('wrote')",
            _ctx(wd),
        )
        assert not r.is_error, r.stdout
        assert "wrote" in r.stdout
        assert os.path.exists(os.path.join(wd, "probe.txt"))  # host sees the write
        # state persists in the same container; an in-container plot is captured
        r2 = k.execute(
            "import plotly.express as px\n"
            "print('persists', 'probe.txt' in os.listdir('.'))\n"
            "px.bar(x=['a'], y=[1]).show()",
            _ctx(wd),
        )
        assert not r2.is_error, r2.stdout
        assert "persists True" in r2.stdout
        assert len(r2.plots) == 1
    finally:
        k.shutdown()
        shutil.rmtree(ws, ignore_errors=True)


@pytest.mark.slow
def test_container_kernel_resource_caps_and_readonly(monkeypatch):
    """PR-F.3: the container drops all Linux capabilities, enforces the cgroup
    memory cap, and runs on a read-only rootfs (only the workspace + scratch
    tmpfs are writable)."""
    import os
    import shutil
    import uuid as _uuid

    runtime = _container_runtime()
    if not runtime:
        pytest.skip("podman + mmm-kernel image not available")

    ws = os.path.join(
        os.path.expanduser("~"), ".cache", "mmm-ktest-" + _uuid.uuid4().hex[:8]
    )
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", ws)
    monkeypatch.setenv("MMM_KERNEL_RUNTIME_BIN", runtime)
    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "tcp")
    monkeypatch.setenv("MMM_KERNEL_READY_TIMEOUT", "120")
    monkeypatch.setenv("MMM_KERNEL_MEM", "512m")  # deterministic cap to assert

    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.container_kernel import ContainerKernel

    wd = str(W.thread_dir("ctest"))
    k = ContainerKernel("ctest")
    try:
        r = k.execute(
            "print('cap', open('/proc/self/status').read().split('CapEff:')[1].split()[0])",
            _ctx(wd),
        )
        assert "cap 0000000000000000" in r.stdout, r.stdout  # all caps dropped
        r2 = k.execute(
            "print('mem', open('/sys/fs/cgroup/memory.max').read().strip())", _ctx(wd)
        )
        assert "mem 536870912" in r2.stdout, r2.stdout  # 512 MiB cgroup cap
        r3 = k.execute(
            "import os\n"
            "try:\n open('/opt/mmm/x','w').write('y'); print('WRITABLE')\n"
            "except OSError: print('READONLY')",
            _ctx(wd),
        )
        assert "READONLY" in r3.stdout, r3.stdout  # read-only rootfs
    finally:
        k.shutdown()
        shutil.rmtree(ws, ignore_errors=True)


# ── Phase 3 PR-F.4: egress deny + metadata block ──────────────────────────────


def test_container_kernel_egress_posture(monkeypatch):
    """ipc denies egress with no network at all; the posture is recorded (and an
    explicit open override is honoured for debug)."""
    from mmm_framework.agents.container_kernel import ContainerKernel

    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "ipc")
    monkeypatch.setenv("MMM_KERNEL_EGRESS", "deny")
    k = ContainerKernel("t")
    assert k._net_args() == ["--network", "none"]
    assert k._egress_posture == "denied:no-network"

    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "tcp")
    monkeypatch.setenv("MMM_KERNEL_EGRESS", "open")
    k2 = ContainerKernel("t")
    k2._net_args()
    assert k2._egress_posture == "open"


@pytest.mark.slow
def test_container_network_none_blocks_egress_and_metadata():
    """The prod egress posture (--network none, used by the ipc transport) makes
    the cloud metadata server AND the public internet unreachable."""
    import os
    import subprocess

    runtime = _container_runtime()
    if not runtime:
        pytest.skip("podman + mmm-kernel image not available")
    image = os.environ.get("MMM_KERNEL_IMAGE", "mmm-kernel:latest")
    code = (
        "import socket; socket.setdefaulttimeout(4)\n"
        "ok = True\n"
        "for t in [('169.254.169.254', 80), ('1.1.1.1', 53)]:\n"
        "    try:\n"
        "        socket.create_connection(t); ok = False\n"
        "    except Exception:\n"
        "        pass\n"
        "print('BLOCKED' if ok else 'REACHED')"
    )
    r = subprocess.run(
        [runtime, "run", "--rm", "--network", "none", image, "python", "-c", code],
        capture_output=True,
        text=True,
    )
    assert "BLOCKED" in r.stdout, (r.stdout, r.stderr)


# ── Phase 3 PR-F.5: ephemeral overlay + teardown-wipe + fail-closed ───────────


def test_container_kernel_fail_closed_when_sandbox_incomplete(monkeypatch, tmp_path):
    """With MMM_KERNEL_REQUIRE_SANDBOX on (the hosted profile), a weakened sandbox
    (read-only off) is refused at spawn — no kernel rather than a half-sandboxed
    one. Runs without podman: the refusal happens before `podman run`."""
    monkeypatch.setenv("MMM_KERNEL_REQUIRE_SANDBOX", "1")
    monkeypatch.setenv("MMM_KERNEL_READONLY", "0")  # the deliberate weakening
    monkeypatch.setenv(
        "MMM_KERNEL_TRANSPORT", "ipc"
    )  # egress denied; read-only is the gap
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))

    from mmm_framework.agents.container_kernel import ContainerKernel
    from mmm_framework.agents.kernels import KernelContext

    k = ContainerKernel("t")
    r = k.execute("print(1)", KernelContext(thread_id="t", work_dir=str(tmp_path)))
    assert r.is_error
    assert "fail-closed" in r.stdout.lower()
    assert "read-only" in r.stdout.lower()


@pytest.mark.slow
def test_container_kernel_teardown_wipes_ctrl_keeps_workspace(monkeypatch):
    """Teardown discards the ephemeral container + wipes the per-kernel control
    dir (which holds the ZMQ HMAC key), but the PERSISTENT workspace survives
    (cold-reload + .py export depend on it)."""
    import os
    import shutil
    import uuid as _uuid

    runtime = _container_runtime()
    if not runtime:
        pytest.skip("podman + mmm-kernel image not available")

    ws = os.path.join(
        os.path.expanduser("~"), ".cache", "mmm-ktest-" + _uuid.uuid4().hex[:8]
    )
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", ws)
    monkeypatch.setenv("MMM_KERNEL_RUNTIME_BIN", runtime)
    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "tcp")
    monkeypatch.setenv("MMM_KERNEL_READY_TIMEOUT", "120")

    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.container_kernel import ContainerKernel

    wd = str(W.thread_dir("ctest"))
    k = ContainerKernel("ctest")
    ctrl = None
    try:
        r = k.execute("open('keep.txt','w').write('persist')", _ctx(wd))
        assert not r.is_error, r.stdout
        ctrl = k._ctrl_dir
        assert ctrl and os.path.isdir(ctrl)  # control dir + conn file present
    finally:
        k.shutdown()
    assert ctrl is not None and not os.path.exists(ctrl)  # wiped (HMAC key gone)
    assert os.path.exists(os.path.join(wd, "keep.txt"))  # workspace survived
    shutil.rmtree(ws, ignore_errors=True)


# ── Phase 3 PR-F.6: hosted joint-exit criteria (under the container kernel) ────


@pytest.mark.slow
def test_container_kernel_joint_exit_criteria(monkeypatch):
    """The §6 joint exit, as far as the macOS dev box can exercise it: a hostile
    cell in session A's container cannot read host secrets, cannot see another
    session's workspace, runs with no capabilities on a read-only rootfs, and is
    memory-capped. (Egress-deny is validated separately by
    test_container_network_none_blocks_egress_and_metadata + enforced via the ipc
    posture in prod; macOS tcp egress is open by documented design.)"""
    import json
    import os
    import shutil
    import uuid as _uuid

    runtime = _container_runtime()
    if not runtime:
        pytest.skip("podman + mmm-kernel image not available")

    ws = os.path.join(
        os.path.expanduser("~"), ".cache", "mmm-ktest-" + _uuid.uuid4().hex[:8]
    )
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", ws)
    monkeypatch.setenv("MMM_KERNEL_RUNTIME_BIN", runtime)
    monkeypatch.setenv("MMM_KERNEL_TRANSPORT", "tcp")
    monkeypatch.setenv("MMM_KERNEL_READY_TIMEOUT", "120")
    monkeypatch.setenv("MMM_KERNEL_MEM", "512m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-SECRET-should-not-cross")

    from mmm_framework.agents import workspace as W
    from mmm_framework.agents.container_kernel import ContainerKernel

    # A sibling tenant's workspace holds a secret on the host (NOT mounted into A).
    sibling = W.thread_dir("sessB") / "tenantB_secret.txt"
    sibling.write_text("tenant-B-data")
    wdA = str(W.thread_dir("sessA"))

    k = ContainerKernel("sessA")
    try:
        probe = (
            "import os, json\n"
            "r = {}\n"
            "r['no_secret_env'] = (os.environ.get('ANTHROPIC_API_KEY') is None "
            "and os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is None)\n"
            "r['caps_dropped'] = open('/proc/self/status').read()"
            ".split('CapEff:')[1].split()[0] == '0000000000000000'\n"
            "try:\n"
            "    open('/opt/mmm/evil','w').write('x'); r['readonly_rootfs'] = False\n"
            "except OSError:\n"
            "    r['readonly_rootfs'] = True\n"
            "r['mem_capped'] = open('/sys/fs/cgroup/memory.max').read().strip() == '536870912'\n"
            "r['seccomp_active'] = open('/proc/self/status').read()"
            ".split('Seccomp:')[1].split()[0] == '2'\n"
            f"r['sibling_isolated'] = not os.path.exists({str(sibling)!r})\n"
            "print('JOINT', json.dumps(r))"
        )
        res = k.execute(probe, _ctx(wdA))
        assert not res.is_error, res.stdout
        line = [l for l in res.stdout.splitlines() if l.startswith("JOINT")][0]
        checks = json.loads(line[len("JOINT ") :])
        assert all(checks.values()), checks
    finally:
        k.shutdown()
        shutil.rmtree(ws, ignore_errors=True)

"""Containerized per-session kernel (Phase 3 Tier 2, PR-F.2).

``ContainerKernel`` runs the ipykernel inside a ``podman run`` container instead of
a bare local subprocess, so untrusted cell code is isolated by the container
boundary (env scrub + workspace-only mount now; cgroup/seccomp caps, egress-deny,
ephemeral overlay layered in PR-F.3/F.4/F.5). It **subclasses ``SubprocessKernel``
and reuses the entire Jupyter protocol** (``_run``/``execute``/``run_model_op``/
``fit``/``_teardown``): only kernel *launch* differs. The inherited code talks to
``self._kc`` (a ``BlockingKernelClient``) and ``self._km`` — here ``self._km`` is a
small ``_ContainerManager`` shim that maps ``is_alive``/``interrupt_kernel``/
``shutdown_kernel`` onto podman, so nothing else changes.

Transport (decided in PR-F.0, ``deploy/kernel/F0-connectivity-findings.md``):
  - ``ipc`` (default off-Darwin; prod): ``--network none`` + ZMQ over unix sockets
    on a per-kernel bind-mounted dir. No network ⇒ egress/metadata fully denied.
  - ``tcp`` (default on Darwin; dev): per-port host↔container forward; egress-deny
    is layered separately (PR-F.4).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import uuid

from mmm_framework.agents.kernels import (
    SubprocessKernel,
    _audit,
    _build_startup_source,
    _scrubbed_kernel_env,
)

_PORT_NAMES = ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")

# Config the kernel container legitimately needs (secrets already removed by
# _scrubbed_kernel_env). The image provides its own PATH/PYTHONPATH/HOME/toolchain,
# so host filesystem/python vars are NOT forwarded (they'd point at host paths that
# don't exist in the container). Only session config crosses.
_CONTAINER_ENV_ALLOW_EXACT = frozenset(
    {
        "MMM_AGENT_WORKSPACE",
        "MMM_MAX_KERNELS",
        "MMM_CELL_TIMEOUT",
        "MMM_FIT_TIMEOUT",
        "MMM_MODEL_CONFIG",
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
        "GOOGLE_CLOUD_PROJECT",
        "LANG",
        "TZ",
    }
)
_CONTAINER_ENV_ALLOW_PREFIX = ("MMM_KERNEL_", "LC_")


def _default_transport() -> str:
    # ipc needs unix sockets that survive the host<->container file share; that holds
    # on Linux but not the macOS podman-machine (virtiofs, PR-F.0), so default tcp there.
    forced = os.environ.get("MMM_KERNEL_TRANSPORT")
    if forced:
        return forced
    return "tcp" if sys.platform == "darwin" else "ipc"


def _free_ports(n: int) -> list[int]:
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


class _ContainerManager:
    """Quacks like the bits of jupyter_client's KernelManager that the inherited
    ``_run``/``_teardown`` use, backed by podman."""

    def __init__(self, runtime_bin: str, name: str, env: dict):
        self._bin = runtime_bin
        self._name = name
        self._env = env

    def _podman(self, *args: str):
        return subprocess.run(
            [self._bin, *args], env=self._env, capture_output=True, text=True
        )

    def is_alive(self) -> bool:
        r = self._podman("inspect", "-f", "{{.State.Running}}", self._name)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def interrupt_kernel(self) -> None:
        # SIGINT to PID 1 (tini), which forwards it — works for pure-Python cells;
        # a compiled sampler ignores it and the wall-clock path escalates to kill.
        self._podman("kill", "-s", "INT", self._name)

    def shutdown_kernel(self, now: bool = True) -> None:
        self._podman("rm", "-f", self._name)


class ContainerKernel(SubprocessKernel):
    """One ``podman``-run ipykernel container per session (``per_session=True``)."""

    per_session = True

    def __init__(self, thread_id: str | None = None):
        super().__init__(thread_id)
        self._runtime_bin = os.environ.get("MMM_KERNEL_RUNTIME_BIN") or os.environ.get(
            "MMM_KERNEL_RUNTIME", "podman"
        )
        self._image = os.environ.get("MMM_KERNEL_IMAGE", "mmm-kernel:latest")
        self._transport = _default_transport()
        self._cname: str | None = None
        self._ctrl_dir: str | None = None  # per-kernel host dir for conn/sockets
        # podman needs a clean registry auth file when the host docker config has a
        # global credsStore (otherwise anonymous pulls fail); harmless if unset.
        self._podman_env = {**os.environ}
        if "REGISTRY_AUTH_FILE" not in self._podman_env:
            auth = os.environ.get("MMM_KERNEL_REGISTRY_AUTH_FILE")
            if auth:
                self._podman_env["REGISTRY_AUTH_FILE"] = auth

    # ── env / mounts / launch args (extended by PR-F.3/F.4/F.5) ───────────────
    def _container_env(self) -> dict[str, str]:
        base = _scrubbed_kernel_env() or {}
        env = {
            k: v
            for k, v in base.items()
            if k in _CONTAINER_ENV_ALLOW_EXACT
            or k.startswith(_CONTAINER_ENV_ALLOW_PREFIX)
        }
        # Never let the in-container framework recurse into spawning kernels.
        env["MMM_AGENT_KERNEL"] = "inprocess"
        return env

    def _net_args(self) -> list[str]:
        # PR-F.4 hardens this (internal/egress-deny network for tcp). For ipc there
        # is no network at all.
        if self._transport == "ipc":
            return ["--network", "none"]
        net = os.environ.get("MMM_KERNEL_NET")
        return ["--network", net] if net else []

    def _resource_args(self) -> list[str]:
        return []  # PR-F.3

    def _security_args(self) -> list[str]:
        return []  # PR-F.3

    def _extra_mount_args(self) -> list[str]:
        return []  # PR-F.5 (ephemeral overlay / tmpfs)

    def _ctrl_root(self) -> str:
        """Per-kernel host dir (under the shared workspace root) for the connection
        file and, for ipc, the unix sockets. Must be on a path the runtime shares."""
        from mmm_framework.agents import workspace as _ws

        root = _ws.workspace_root() / ".kernels" / (self._thread_id or "default")
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    # ── launch (override) ─────────────────────────────────────────────────────
    def _start(self) -> None:
        from jupyter_client import BlockingKernelClient

        work_dir = self._work_dir()
        ctrl = self._ctrl_dir = self._ctrl_root()
        self._cname = name = (
            f"mmm-kernel-{(self._thread_id or 'def')[:24]}-{uuid.uuid4().hex[:8]}"
        )
        ports = _free_ports(5)
        key = uuid.uuid4().hex
        base = {
            **dict(zip(_PORT_NAMES, ports)),
            "key": key,
            "signature_scheme": "hmac-sha256",
            "kernel_name": "python3",
        }
        env_args: list[str] = []
        for k, v in self._container_env().items():
            env_args += ["-e", f"{k}={v}"]

        # workspace bind-mount (same absolute path in the container so OUTPUT_DIR /
        # cwd / mmm_models / dataset_path all resolve identically). Rootless: run as
        # the host uid mapped via keep-id so the (host-owned) bind mount is writable
        # while the process stays unprivileged. Overridable for a prod userns/idmap.
        uid = os.getuid() if hasattr(os, "getuid") else 0
        gid = os.getgid() if hasattr(os, "getgid") else 0
        userns = os.environ.get("MMM_KERNEL_USERNS", "keep-id")
        user = os.environ.get("MMM_KERNEL_USER", f"{uid}:{gid}")
        mount_args: list[str] = [f"--userns={userns}", "--user", user]
        if work_dir:
            mount_args += ["-v", f"{work_dir}:{work_dir}"]

        if self._transport == "ipc":
            ip = os.path.join(ctrl, "k")  # sockets at <ctrl>/k-<port>
            conn = os.path.join(ctrl, "conn.json")
            with open(conn, "w") as f:
                json.dump({**base, "ip": ip, "transport": "ipc"}, f)
            os.chmod(conn, 0o644)
            mount_args += ["-v", f"{ctrl}:{ctrl}"]
            net_pub: list[str] = []
            host_conn = conn
            launch_conn = conn
        else:
            conn = os.path.join(ctrl, "container.json")
            host_conn = os.path.join(ctrl, "host.json")
            with open(conn, "w") as f:
                json.dump({**base, "ip": "0.0.0.0", "transport": "tcp"}, f)
            with open(host_conn, "w") as f:
                json.dump({**base, "ip": "127.0.0.1", "transport": "tcp"}, f)
            os.chmod(conn, 0o644)
            mount_args += ["-v", f"{ctrl}:{ctrl}:ro"]
            net_pub = []
            for p in ports:
                net_pub += ["-p", f"127.0.0.1:{p}:{p}"]
            launch_conn = conn

        cmd = [
            self._runtime_bin,
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            *self._net_args(),
            *net_pub,
            *env_args,
            *mount_args,
            *self._extra_mount_args(),
            *self._resource_args(),
            *self._security_args(),
            self._image,
            "python",
            "-m",
            "ipykernel_launcher",
            "-f",
            launch_conn,
        ]
        proc = subprocess.run(cmd, env=self._podman_env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"podman run failed: {proc.stderr.strip()[:500]}")

        self._km = _ContainerManager(self._runtime_bin, name, self._podman_env)
        self._kc = BlockingKernelClient(connection_file=host_conn)
        self._kc.load_connection_file()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=self._ready_timeout)
        self._run(_build_startup_source(), silent=True, capture=False)
        self._started = True
        _audit(
            "kernel_spawn",
            thread_id=self._thread_id,
            kind="container",
            transport=self._transport,
        )

    def reset(self) -> None:
        # A container "restart" is teardown + fresh run (vs. jupyter restart_kernel).
        with self._lock:
            self._teardown()

    def shutdown(self) -> None:
        super().shutdown()

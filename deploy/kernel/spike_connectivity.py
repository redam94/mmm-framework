#!/usr/bin/env python
"""PR-F.0 connectivity spike: can the host jupyter_client drive an ipykernel that
runs *inside* a podman container, and under what transport / network posture?

This settles the provisioner architecture (design §1.4 / §4): TCP ZMQ and
``--network none`` are mutually exclusive. Two transports are spiked:

  - ``SPIKE_TRANSPORT=tcp`` (default): kernel binds ZMQ on 0.0.0.0:<ports> with a
    connection file mounted in (ip=0.0.0.0); podman maps host 127.0.0.1:P ->
    container P for each of 5 ports; the host client uses ip=127.0.0.1. Works on
    the default network. Egress must then be blocked some other way.
  - ``SPIKE_TRANSPORT=ipc``: kernel binds ZMQ to unix sockets under a bind-mounted
    dir (same absolute path host+container); ``--network none`` (no ports, egress
    fully denied — including the metadata server). The clean prod answer *iff*
    unix sockets survive the host<->VM file share.

``SPIKE_NETWORK=<name>`` adds ``--network <name>`` (tcp only). Run:
``uv run python deploy/kernel/spike_connectivity.py``  -> prints SPIKE-OK / SPIKE-FAIL.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid

PODMAN = os.environ.get("MMM_KERNEL_RUNTIME_BIN", "/opt/podman/bin/podman")
IMAGE = os.environ.get("SPIKE_IMAGE", "mmm-kernel-spike:latest")
TRANSPORT = os.environ.get("SPIKE_TRANSPORT", "tcp")
PORT_NAMES = ("shell_port", "iopub_port", "stdin_port", "control_port", "hb_port")
ENV = {**os.environ, "REGISTRY_AUTH_FILE": "/tmp/mmm-podman-auth.json"}


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


def _share_dir() -> str:
    # Must live on a path podman-machine shares into its VM (the user home on
    # macOS, NOT the sandbox $TMPDIR).
    root = os.path.join(os.path.expanduser("~"), ".cache", "mmm-kernel-spike")
    d = os.path.join(root, f"run-{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d


def _drive(host_conn: str, label: str) -> int:
    from jupyter_client import BlockingKernelClient

    kc = BlockingKernelClient(connection_file=host_conn)
    kc.load_connection_file()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=60)
        msg_id = kc.execute("print(21 * 2)")
        got = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                m = kc.get_iopub_msg(timeout=5)
            except Exception:
                continue
            if m["parent_header"].get("msg_id") != msg_id:
                continue
            if m["msg_type"] == "stream":
                got = (got or "") + m["content"].get("text", "")
            if m["msg_type"] == "status" and m["content"]["execution_state"] == "idle":
                break
        if got and "42" in got:
            print(f"SPIKE-OK transport={label} output={got.strip()!r}")
            return 0
        print(f"SPIKE-FAIL transport={label} (got={got!r})")
        return 1
    finally:
        kc.stop_channels()


def main() -> int:
    ports = _free_ports(5)
    key = uuid.uuid4().hex
    base = {
        **dict(zip(PORT_NAMES, ports)),
        "key": key,
        "signature_scheme": "hmac-sha256",
        "kernel_name": "python3",
    }
    share = _share_dir()
    name = f"mmm-spike-{uuid.uuid4().hex[:8]}"

    if TRANSPORT == "ipc":
        ip = os.path.join(share, "kernel")  # socket path prefix; sockets are ip-<port>
        conn = os.path.join(share, "conn.json")
        with open(conn, "w") as f:
            json.dump({**base, "ip": ip, "transport": "ipc"}, f)
        os.chmod(conn, 0o644)
        run_cmd = [
            PODMAN, "run", "-d", "--rm", "--name", name,
            "--network", "none",
            "--user", "0",  # spike: sidestep virtiofs uid-map perms on the socket dir
            "-v", f"{share}:{share}",
            IMAGE,
            "python", "-m", "ipykernel_launcher", "-f", conn,
        ]
        host_conn = conn  # same socket paths on both sides
        label = "ipc+network-none"
    else:
        pubs = []
        for p in ports:
            pubs += ["-p", f"127.0.0.1:{p}:{p}"]
        net = os.environ.get("SPIKE_NETWORK")
        net_args = ["--network", net] if net else []
        container_conn = os.path.join(share, "container.json")
        host_conn = os.path.join(share, "host.json")
        with open(container_conn, "w") as f:
            json.dump({**base, "ip": "0.0.0.0", "transport": "tcp"}, f)
        with open(host_conn, "w") as f:
            json.dump({**base, "ip": "127.0.0.1", "transport": "tcp"}, f)
        os.chmod(container_conn, 0o644)
        run_cmd = [
            PODMAN, "run", "-d", "--rm", "--name", name,
            *net_args, *pubs,
            "-v", f"{container_conn}:/conn.json:ro",
            IMAGE,
            "python", "-m", "ipykernel_launcher", "-f", "/conn.json",
        ]
        label = "tcp+portmap" + (f"+net={net}" if net else "")

    print("launching:", " ".join(run_cmd))
    cid = subprocess.run(run_cmd, env=ENV, capture_output=True, text=True)
    if cid.returncode != 0:
        print("podman run failed:", cid.stderr.strip())
        return 2
    try:
        return _drive(host_conn, label)
    finally:
        subprocess.run([PODMAN, "rm", "-f", name], env=ENV, capture_output=True)


if __name__ == "__main__":
    sys.exit(main())

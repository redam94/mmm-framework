# PR-F.0 — Containerized-kernel connectivity spike (findings)

**Question (design §1.4):** can the host `jupyter_client` drive an ipykernel running
inside `podman run`, and how do we reconcile the ZMQ control channel with egress-deny
(`--network none` and TCP port-publishing are mutually exclusive)?

**Spiked** with `deploy/kernel/spike_connectivity.py` against `mmm-kernel-spike`
(`Containerfile.spike`: `python:3.12-slim` + `ipykernel`/`jupyter-client`, non-root) on
podman 5.8.2 / libkrun machine, macOS (Apple Silicon).

## Results

| Transport / network | Control channel | Egress | Verdict |
|---|---|---|---|
| **`tcp` + per-port-forward, default net** | ✅ works (`21*2 → 42`) | open | dev-usable |
| **`ipc` + `--network none`** (shared unix sockets) | ❌ "Kernel died before replying" on macOS | n/a (no net) | **Linux-only** |
| **`--internal` network** (any transport) | ❌ rootless port-forward breaks | ✅ internet **and** `169.254.169.254` both `Network is unreachable` | egress proven |

- **IPC fails on macOS** because a unix-domain socket created inside the libkrun VM is not
  connectable from macOS across the virtiofs file share (exactly the advisor's prediction).
  On a real **Linux** host a bind-mounted unix socket is a normal socket, so `ipc` works there.
- **`--internal` denies egress for real** — both the public internet and the cloud metadata
  server are unreachable — but rootless podman's port-forwarder does not function on it, so it
  can't carry a `tcp` control channel.

## Decision (drives PR-F.2 / PR-F.4)

The provisioner supports **two transports**, platform-defaulted, overridable via
`MMM_KERNEL_TRANSPORT`:

- **`ipc` (default off-Darwin; the prod posture):** `--network none` + ZMQ over unix sockets on
  a per-kernel bind-mounted dir. **No network at all ⇒ egress and the metadata server are
  trivially, completely denied** (this *is* the control that stops ADC token theft). Validated
  on Linux (CI/host); **not validatable on the macOS dev box** (virtiofs, above).
- **`tcp` (default on Darwin; the dev posture):** per-port host↔container forward
  (`127.0.0.1:P → P` ×5). Control channel proven here. Egress-deny on this path uses an
  `--internal` network **on Linux** (where port-forward + internal coexist); on **rootless
  macOS** port-forward + internal don't coexist, so the macOS-dev `tcp` path has **open egress**
  — an accepted dev limitation (the laptop is single-user, not the multi-tenant threat surface;
  the metadata-block matters on the GCP VM, which runs the `ipc` prod posture).

**Net:** the full egress-deny exit criterion (§6) is met by the **`ipc`+`--network none` prod
posture**; the macOS dev box exercises everything else (env scrub, mount isolation, caps,
teardown) over `tcp`, with egress-deny as the one control validated on Linux rather than here.

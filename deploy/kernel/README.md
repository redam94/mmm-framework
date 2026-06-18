# Shipping the hardened agent kernel image

The agent's `execute_python` runs untrusted, model-generated code. In the default
posture it runs **in-process** (no sandbox) — fine for single-tenant/dev, unsafe
for multi-tenant hosting. `MMM_AGENT_HOSTED=1` flips the server to a **fail-closed**
posture that refuses to boot unless a real sandbox is present: each session's
kernel runs inside a container (`agents/container_kernel.py`) with a read-only
rootfs, all capabilities dropped, cgroup memory/pids/cpu caps, a scrubbed env, and
**egress denied**.

That sandbox needs **this image built and reachable by the host's container
runtime**. Until it exists, hosted mode is deliberately inert (`agents/profile.py`
+ the lifespan guard `assert_hosted_sandbox` in `src/mmm_framework/api/main.py`).
This is the one-time "ship" step.

## Build → verify → push

From the repo root (the build context is the repo so the framework source can be
copied onto `PYTHONPATH`):

```bash
make kernel-lock      # (re)freeze deploy/kernel/requirements.lock from uv.lock
make kernel-image     # podman build -t mmm-kernel:latest -f deploy/kernel/Containerfile .
make kernel-verify    # run the image under the real sandbox flags; imports mmm_framework + ipykernel
make kernel-push KERNEL_REGISTRY=registry.example.com/yourorg
```

Knobs (Makefile vars / env): `KERNEL_RUNTIME` (default `podman`; `docker` works),
`KERNEL_IMAGE` (default `mmm-kernel:latest`), `KERNEL_REGISTRY`.

`make kernel-verify` is the proof the image is sandbox-runnable — it launches the
container with `--read-only --tmpfs /tmp --network none --cap-drop ALL --user 10001`
(the same flags the provisioner applies) and imports the framework + ipykernel
inside it. If that passes, the image will serve real sessions.

## Enable hosted mode

Point the server at the image and flip the profile:

```bash
export MMM_AGENT_HOSTED=1            # fail-closed: refuses to boot without a sandbox
export MMM_AGENT_KERNEL=container    # forced by hosted mode anyway
export MMM_KERNEL_IMAGE=registry.example.com/yourorg/mmm-kernel:latest
export MMM_KERNEL_RUNTIME=podman     # or docker
# optional: MMM_KERNEL_MEM, MMM_KERNEL_PIDS, MMM_KERNEL_CPUS, MMM_KERNEL_TRANSPORT (ipc|tcp)
```

On boot, `assert_hosted_sandbox` verifies the kernel impl is the container sandbox
and refuses to start otherwise — a half-applied sandbox (hosted behavior over an
in-process kernel) is treated as more dangerous than an honest single-user
deployment. Pair this with `MMM_AUTH_ENABLED=1` + `MMM_AUTH_SECRET` so tenant
isolation is also in force.

## Kubernetes

Production manifests live in `deploy/k8s/`:

- `kernel-pod-template.yaml` — the per-session kernel pod the provisioner renders.
- `runtimeclass-gvisor.yaml` — run kernels under **gVisor** for a second isolation
  boundary beyond the container.
- `networkpolicy-kernel-egress-deny.yaml` — default-deny egress for kernel pods.
- `rbac.yaml`, `warm-pool.yaml`, `hpa.yaml`, `redis.yaml`, `api.yaml`, `namespace.yaml`.

Push the image to a registry the cluster can pull, set `MMM_KERNEL_IMAGE` to that
reference in `api.yaml`, and apply the manifests.

## Supply-chain note

Dependencies are pinned (`requirements.lock`). The documented prod-hardening
follow-up is `pip install --require-hashes` and signing/attesting the image
(cosign) — tracked, not yet wired.

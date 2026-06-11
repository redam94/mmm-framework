# Phase 4 Scoping — Orchestrated, Autoscaled, Hosted Pool

**Parent design:** `agent-session-kernels.md` §6 (Phase 4 a–d). **Builds on:** Phase 3
(Tier 1 + Tier 2 COMPLETE — the single-host sandboxed `ContainerKernel`). **Goal:** take the
single-host sandbox to a **cross-host, orchestrated, autoscaled** hosted service.

> **Validation honesty (read first).** Phase 4 is orchestration/infrastructure that **cannot be
> deployed or load-tested from the single dev box** (no cluster). So each artifact below carries an
> explicit **validation level**:
> - **CODE+TEST** — real code, unit-tested here (runs in CI).
> - **LOCAL-ASSERT** — code whose effect is asserted locally without the external system present
>   (e.g. the provisioner emits `--runtime=runsc` without gVisor installed).
> - **AUTHORED** — manifests/config that are schema/lint-clean but **NOT deploy-tested** — a real
>   cluster is required to exercise them. These are design-complete, not validated.
>
> **The single highest-value remaining validation is a first boot of the full prod posture
> (`ipc` + `--network none` + the complete `ContainerKernel`) on a Linux host** — Phase 3's joint
> criteria are each validated but never *jointly on one running kernel* (macOS can't: ipc→
> unreachable, tcp→egress open). Treat Linux first-boot as a required pre-deployment step, not a
> formality — it's where a scale-equivalent of the `nr_inodes` catch would surface.

---

## 4a — Orchestrator + per-user quotas  (AUTHORED)

`deploy/k8s/` — Kubernetes manifests for the hosted service:

- `namespace.yaml` — a namespace **per tenant** with a `ResourceQuota` (CPU/mem/pod ceilings) +
  `LimitRange` (per-pod defaults), so one tenant can't starve others (the §5 per-user quota).
- `api-deployment.yaml` / `api-service.yaml` — the FastAPI agent API (the orchestrator that mints
  kernel pods); its `ServiceAccount` is the only identity with cloud creds.
- `redis.yaml` — the ARQ/redis dependency.
- `kernel-pod-template.yaml` — the per-session kernel **pod** spec: the `securityContext` expression
  of Phase-3 PR-F.3 (`runAsNonRoot`, `readOnlyRootFilesystem`, `capabilities.drop:[ALL]`,
  `seccompProfile: RuntimeDefault`, `allowPrivilegeEscalation:false`), resource `limits`, an
  emptyDir/tmpfs scratch, and the workspace `PersistentVolumeClaim` mount.
- `networkpolicy-kernel-egress-deny.yaml` — the k8s-native egress-deny (default-deny egress + an
  explicit block of the metadata CIDR) — the cluster equivalent of `--network none`/`--internal`.
- `rbac.yaml` — the API SA may create/delete kernel pods in tenant namespaces; the **kernel** SA
  has *no* permissions and `automountServiceAccountToken: false` (no in-cluster identity).
- `runtimeclass-gvisor.yaml` — the gVisor `RuntimeClass` (4c).

**The provisioner.** The cross-host analogue of `ContainerKernel` is a **`KubernetesKernel`** that
creates a kernel **Pod** (instead of `podman run`) and connects over the same Jupyter protocol via
a headless `Service`. It reuses the entire `_run`/`execute`/`fit` protocol (as `ContainerKernel`
does) — only `_start`/`_teardown` change (k8s API create/delete Pod). **Design-complete; the
implementation + its cluster validation are deploy-time work** (the podman `ContainerKernel` is the
single-host implementation that proves the protocol).

## 4b — Autoscaling + warm pool + load test  (AUTHORED + CODE harness)

- `deploy/k8s/hpa.yaml` — `HorizontalPodAutoscaler` for the API/kernel pool. Baseline on CPU/mem;
  the §5.1 composite signal (**reserved-memory-headroom** + **active-fit-count vs vCPU**) is a
  **custom metric** (`active_fits`, exported by 4d's metrics endpoint) consumed via a
  prometheus-adapter — manifest authored, the custom-metrics pipeline is cluster work.
- `deploy/k8s/warm-pool.yaml` — a small `Deployment` of pre-imported kernel pods (the §5.1 cold-
  start mitigation) the API claims on first cell.
- `deploy/loadtest/chat_load.py` — **CODE**: an asyncio load driver that opens N concurrent `/chat`
  SSE sessions, drives a fit each, and reports p50/p95 first-cell + fit latency against a stated
  concurrency SLO. Runnable against any deployed API; **needs a target** to produce numbers.

## 4c — Stronger isolation tier (gVisor/Kata)  (LOCAL-ASSERT)

Gated on the §7.2 trust-model decision (only if LLM code escalates to *genuinely hostile*). The
hook is already present: `ContainerKernel` passes `--runtime=<MMM_KERNEL_OCI_RUNTIME>` to podman
(e.g. `runsc` for gVisor, `kata-runtime` for Kata), and the k8s `kernel-pod-template` references the
`gvisor` `RuntimeClass`. **LOCAL-ASSERT:** a unit test confirms the provisioner emits
`--runtime=runsc` when configured, *without* gVisor installed. Its cost (per-kernel memory tax +
spawn latency + syscall slowdown) folds into the §2.1 density math — measured at adoption.

## 4d — Centralized metrics + tamper-evident off-host audit  (CODE+TEST)

- **Tamper-evident audit** (`agents/audit_sink.py`): a `logging.Handler` for the `mmm_audit` logger
  that writes each event as a JSON line carrying a **hash chain** (`prev_hash` + `hash` over the
  record), so any deletion/edit/reordering breaks the chain — detectable by a bundled `verify()`.
  Ship the JSONL off-host (the file is the seam; a shipper tails it). **CODE+TEST**.
- **Metrics** (`/metrics`): a Prometheus-format endpoint exposing kernel counts, active fits,
  spawn/evict/OOM/denied-egress counters (sourced from the audit events) + the §5.1 autoscaling
  signal. Endpoint authored; scrape pipeline is cluster work.

## What stays unchanged
The single-host `ContainerKernel` (Phase 3) is the reference implementation; Phase 4 wraps it in
orchestration. The kernel protocol, the env scrub, the sandbox controls, and the audit events are
all carried forward unchanged.

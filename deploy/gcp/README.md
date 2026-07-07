# Deploying a multiseat Augur on GCP (Terraform)

This directory deploys the MMM Framework platform to Google Cloud in its
**hosted multi-user ("multiseat") posture**: JWT authentication with
organizations, roles and invites (`MMM_AUTH_ENABLED=1`), and the fail-closed
kernel sandbox (`MMM_AGENT_HOSTED=1`) so every seat's model-authored code runs
inside an egress-denied, read-only, capability-dropped container.

> **Why a single VM and not GKE?** The k8s manifests in `deploy/k8s/` are the
> forward-looking multi-node design, but the code has **no Kubernetes kernel
> provisioner yet** — the implemented sandbox is `MMM_AGENT_KERNEL=container`,
> which shells out to podman on the same host (`agents/container_kernel.py`).
> The platform also keeps live kernel state in the API process and its state in
> SQLite, so it scales **vertically**. This stack is therefore a hardened
> single-VM appliance behind a global HTTPS load balancer. When a Kubernetes
> provisioner lands, `deploy/k8s/` becomes the target.

## Architecture

```
                        ┌────────────────────────────────────────────────┐
 seats (browsers)       │ GCP project                                    │
   │ https://augur.example.com                                           │
   ▼                    │                                                │
 Global HTTPS LB ── managed cert, HTTP→HTTPS, 3600s timeout (SSE)        │
   │                    │                                                │
   ▼ :80                │  ┌──────────────────────────────────────────┐  │
 ┌─────────────────────────│ GCE VM (Ubuntu 24.04, no external IP)    │  │
 │ nginx                   │                                          │  │
 │  ├─ /      → React build (/opt/mmm/frontend)                       │  │
 │  └─ /api/* → uvicorn :8000 (strips /api, SSE unbuffered)           │  │
 │ mmm-api.service (user mmm)                                         │  │
 │  ├─ FastAPI agent API — MMM_AGENT_HOSTED=1, MMM_AUTH_ENABLED=1     │  │
 │  ├─ Vertex AI via VM service account ADC (LLM + KB embeddings)     │  │
 │  └─ podman run … mmm-kernel (per-session sandbox)                  │  │
 │       --network none · --read-only · --cap-drop ALL                │  │
 │       cgroup mem/cpu/pids caps · scrubbed env (no creds)           │  │
 │ /data (persistent disk, daily snapshots)                           │  │
 │  ├─ state/sessions.db   (auth, orgs, sessions, checkpoints)        │  │
 │  ├─ workspace/          (per-thread models, reports, project KBs)  │  │
 │  └─ audit/audit.jsonl   (tamper-evident audit log)                 │  │
 └─────────────────────────┬──────────────────────────────────────────┘  │
                        │  │ Cloud NAT (egress only)                      │
                        │  ▼                                              │
                        │ Artifact Registry (mmm-kernel image)            │
                        │ GCS release bucket (source + frontend bundles)  │
                        │ Secret Manager (JWT key, bootstrap password)    │
                        │ Vertex AI (chat model + text-embedding-005)     │
                        └────────────────────────────────────────────────┘
```

**What "multiseat" enforces** (all implemented in `src/mmm_framework/auth/` +
`agents/profile.py`, verified fail-closed at API boot):

| Boundary | Mechanism |
|---|---|
| Login / identity | Built-in HS256 JWT (`/auth/login`, 1 h access + 14 d refresh tokens) |
| Tenancy | Organizations; every project/session/artifact is org-scoped; cross-org access 404s (no existence leak) |
| Seats & roles | `viewer < analyst < admin < owner`; invites, role changes, deactivation via `/auth/*` |
| Untrusted code | Each session's `execute_python`/fits run in an `mmm-kernel` container: no network, no cloud metadata, read-only rootfs, no capabilities, cgroup caps, env scrubbed of every credential |
| Session IDs | Hosted mode refuses client-supplied thread IDs — only server-minted sessions |
| Audit | Hash-chained audit log on `/data`, exportable via `/auth/audit-export` |

## Prerequisites (operator machine)

- `terraform` ≥ 1.7, `gcloud` (authenticated, project set), `git`
- `podman` **or** `docker` (kernel image build), `node`/`npm` (frontend build),
  `uv` (only if `deploy/kernel/requirements.lock` needs regenerating)
- GCP project with billing; permissions to create the resources here
  (`roles/editor` + `roles/resourcemanager.projectIamAdmin`, or equivalent)
- For `update_app.sh` / SSH: `roles/iap.tunnelResourceAccessor` and
  `roles/compute.osAdminLogin` (OS Login with sudo) on the project or instance
- **Vertex AI**: the default LLM is Claude on Vertex — the model must be
  **enabled in the Model Garden** for your project (one-time console step),
  and `llm_location` must be a region that serves it (e.g. `us-east5`)

## Quick start

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars   # edit: project_id, domain, bootstrap_email
terraform init && terraform apply              # ~5 min; VM boots and waits for a release

cd ../../..                                    # repo root
deploy/gcp/scripts/build_kernel_image.sh       # build + verify + push the sandbox image
deploy/gcp/scripts/update_app.sh               # package source+frontend, upload, install

terraform -chdir=deploy/gcp/terraform output   # app_url, lb_ip, bootstrap_password_command
```

Point your DNS A record at `lb_ip` (or set `dns_managed_zone`), wait for the
managed certificate (typically 10–30 min after DNS resolves), then log in at
`app_url` as `bootstrap_email` with the password from:

```bash
gcloud secrets versions access latest --secret mmm-bootstrap-password --project <PROJECT>
```

## Step-by-step

### 1. Configure and apply Terraform

`terraform/terraform.tfvars` — the decisions that matter:

- **`domain`** — set it for the real deployment (HTTPS LB + managed cert; the
  VM gets no public IP). Leave empty only for a throwaway evaluation: that
  serves plain HTTP from a static IP to `allowed_ingress_cidrs` and is not
  acceptable for real client data.
- **`bootstrap_email` / `bootstrap_org`** — the first organization and its
  owner, created idempotently at API startup. The generated password is only
  in Secret Manager.
- **LLM** — default is `vertex_anthropic` via the VM's service account (no API
  key anywhere). For direct providers set `llm_provider`/`llm_model` and put
  the key in `llm_api_key` (it lands in Secret Manager and is injected only
  into the API process; the kernel env-scrub strips it from every sandbox).
- **Sizing** — `machine_type` must hold `max_kernels × kernel_mem` plus ~4 GB
  for API + OS. The default `n2-standard-8` (32 GB) fits 6 × 2 GB kernels.

`terraform apply` creates: VPC + NAT, the VM (Shielded, OS Login, IAP-only
SSH) with a snapshotted data disk, Artifact Registry, the release bucket
(seeded with `vm/vm_setup.sh`), Secret Manager secrets (generated JWT signing
key + bootstrap password), the service account
(`aiplatform.user` + logging/monitoring + repo-scoped pulls), and — when
`domain` is set — the HTTPS load balancer with managed cert and HTTP redirect.

### 2. Build and push the kernel sandbox image

```bash
deploy/gcp/scripts/build_kernel_image.sh
```

Builds `deploy/kernel/Containerfile` (pinned dependency closure), **verifies it
under the actual sandbox flags** (`--read-only --network none --cap-drop ALL
--user 10001`, importing `mmm_framework` + `ipykernel`), and pushes to Artifact
Registry. Hosted mode is deliberately **fail-closed**: the API refuses to serve
sessions without this image (`assert_hosted_sandbox`).

The script prints the pushed digest — **pin it** in `terraform.tfvars`
(`kernel_image = "…@sha256:…"`), re-apply, and reboot the VM (or
`sudo mmm-update` over SSH) so the sandbox is supply-chain pinned rather than
floating on `:latest`.

### 3. Package and install a release

```bash
deploy/gcp/scripts/update_app.sh
```

This packages `git archive HEAD` (source) + `npm run build` (frontend), uploads
both to the release bucket, marks the version `CURRENT`, and runs
`sudo mmm-update` on the VM over IAP SSH. On the VM, `mmm-update`:

1. unpacks the source under `/opt/mmm/releases/<version>` and runs
   `uv sync --frozen --no-dev` (uv-managed Python 3.12),
2. points the API at `/data/state/sessions.db` via `MMM_SESSIONS_DB` in
   `/etc/mmm/mmm.env` (this is what makes auth/org/session state survive
   releases; a symlink from the packaged tree's
   `src/mmm_framework/api/sessions.db` is kept as belt-and-braces for tools
   run without the unit's env),
3. unpacks the frontend under `/opt/mmm/frontends/<version>`,
4. pulls the kernel image and re-verifies it under the sandbox flags,
5. atomically flips the `/opt/mmm/app` + `/opt/mmm/frontend` symlinks, restarts
   `mmm-api`, and waits for `/health`.

**Upgrade** = run `update_app.sh` again. **Rollback** =
`SKIP_PACKAGE=1 deploy/gcp/scripts/update_app.sh <version>`.

### 4. DNS and certificate

Point an A record for `domain` at the `lb_ip` output (automatic if you set
`dns_managed_zone`). The Google-managed certificate provisions only after DNS
resolves — check with:

```bash
gcloud compute ssl-certificates list --project <PROJECT>
```

`FAILED_NOT_VISIBLE` means DNS isn't pointing at the LB yet; it retries
automatically.

### 5. First login and smoke test

```bash
URL=$(terraform -chdir=deploy/gcp/terraform output -raw app_url)
curl -s "$URL/api/health"     # {"status":"ok",...}

# Login as the bootstrap owner:
PW=$(gcloud secrets versions access latest --secret mmm-bootstrap-password --project <PROJECT>)
curl -s "$URL/api/auth/login" -H 'content-type: application/json' \
  -d "{\"email\":\"admin@example.com\",\"password\":\"$PW\"}"
```

Or just open `app_url` in a browser — the React login page drives the same
endpoints. After logging in, create a project, upload a dataset (or ask the
agent to generate synthetic data) and run a quick `map` fit to prove the
kernel sandbox works end-to-end.

## Managing seats

All endpoints are under `/api/auth/*` (the UI's Team page drives the same
API). Roles: `viewer` (read-only) < `analyst` (model ops) < `admin`
(destructive ops + seat admin) < `owner` (org management).

**Invite a seat into your org** (admin/owner):

```bash
TOKEN=$(curl -s "$URL/api/auth/login" -H 'content-type: application/json' \
  -d '{"email":"admin@example.com","password":"…"}' | jq -r .access_token)

curl -s "$URL/api/auth/invite" -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"email":"analyst@client.com","role":"analyst"}'
# → {"token":"…","expires_at":…}   (invites expire after 7 days by default)
```

Deliver the invite token out-of-band (there is no built-in email sender). The
invitee redeems it — no login required:

```bash
curl -s "$URL/api/auth/accept-invite" -H 'content-type: application/json' \
  -d '{"token":"<invite-token>","name":"Ada Analyst","password":"<their-password>"}'
```

**Other seat operations:**

| Action | Endpoint |
|---|---|
| List members | `GET /api/auth/members` |
| Change a role | `PATCH /api/auth/members/{user_id}` `{"role":"admin"}` |
| Deactivate a seat | `POST /api/auth/users/{user_id}/deactivate` |
| Password reset | `POST /api/auth/password-reset/request` → token → `…/confirm` |
| Log out everywhere | `POST /api/auth/logout-all` (bumps the user's token version) |

**Project-level access** within an org uses the roster endpoints
(`/api/users`, `/api/projects/{id}/members`) and the `/team` page.

**A caveat to know about:** `POST /auth/signup` is open — anyone who can reach
the site can create a *new* organization (tenants are still fully isolated
from each other). There is currently no server knob to disable signup. For a
closed deployment, restrict reachability (VPN / Cloud Armor allowlist on the
LB / IAP in front) or remove the signup route in a fork.

## LLM configuration

The default is **Vertex AI over the VM service account's ADC** — no API keys
exist anywhere in the deployment:

- `llm_provider = "vertex_anthropic"` needs the Claude model enabled in Model
  Garden and an `llm_location` that serves it (e.g. `us-east5`). Use the
  **exact** Model Garden ID (often with an `@version` suffix); a 404 usually
  means wrong ID or wrong region.
- `llm_provider = "vertex_gemini"` works the same way for Gemini models.
- KB embeddings default to Vertex `text-embedding-005` in `embed_location`
  (`us-central1`) over the same ADC. (The chat model and embedder are separate
  — Anthropic has no embeddings API.)
- Direct providers (`anthropic`/`openai`/`google_genai`): set `llm_api_key`;
  it is stored in Secret Manager and never enters kernel containers.

Changing any of these: edit `terraform.tfvars`, `terraform apply`, reboot the
VM (the env file is rebuilt from Terraform values + Secret Manager at boot):

```bash
gcloud compute instances reset mmm-app --zone <ZONE> --project <PROJECT>
```

## Operations

| Task | How |
|---|---|
| SSH | `terraform output -raw ssh_command` (IAP tunnel; no public port 22) |
| App logs | `journalctl -u mmm-api -f` on the VM, or Cloud Logging (Ops Agent) |
| Deploy an update | `deploy/gcp/scripts/update_app.sh` |
| Rollback | `SKIP_PACKAGE=1 deploy/gcp/scripts/update_app.sh <version>` |
| List releases | `gcloud storage ls gs://<bucket>/releases/` |
| Backups | Daily data-disk snapshots (14 d retention by default) — covers DB, workspace, audit log |
| Restore | Create a disk from a snapshot, swap it as `mmm-data`, reboot |
| Rotate JWT key | `terraform taint random_password.auth_secret && terraform apply`, reboot VM (all seats re-login) |
| Rotate bootstrap password | first login → `POST /api/auth/password-reset/request` flow, or add a new secret version |
| Resize | change `machine_type` (and `max_kernels`) in tfvars, `terraform apply` (VM restarts) |
| Grow data disk | raise `data_disk_size_gb`, apply, then on the VM: `sudo resize2fs /dev/disk/by-id/google-mmm-data` |
| Tear down | `terraform destroy` — deliberately blocked by the data disk's `prevent_destroy` (it holds every seat's data); remove that lifecycle block in `compute.tf` only if you truly mean it |

**Scaling note.** One uvicorn worker is deliberate: session kernels and the
SSE plot/table channels live in the API process, and state is SQLite. Scale up
(`machine_type`, `max_kernels`), not out. Multi-node arrives with the
`deploy/k8s/` design once a Kubernetes kernel provisioner exists in
`agents/kernels.py`.

## Security posture summary

- **Kernel sandbox (Tier 2, fail-closed):** every seat's model-authored code
  runs under rootless podman with `--network none` (blocks internet *and* the
  metadata server → no ADC theft), read-only rootfs, `--cap-drop ALL`,
  `no-new-privileges`, cgroup mem/cpu/pids caps, tmpfs scratch, and a scrubbed
  env (allowlist + credential denylist in `agents/kernels.py`).
- **Identity:** only the VM service account has cloud permissions
  (`aiplatform.user`, logging/monitoring, repo-scoped registry pull,
  bucket-scoped object read, per-secret access). Kernels have none.
- **Ingress:** LB-only (VM has no external IP); SSH via IAP only; Shielded VM
  + OS Login. Direct HTTP mode exists solely for evaluation.
- **Secrets:** generated by Terraform into Secret Manager, fetched at boot to
  a root:mmm 0640 env file; never in instance metadata, never in kernels.
  (They do live in the Terraform state — protect the state file / use a GCS
  backend with tight IAM.)
- **Left for you:** Cloud Armor / VPN if the login page itself must not be
  public; off-host audit shipping (`MMM_AUDIT_SHIP_URL` via `extra_env`);
  digest-pinning the kernel image (step 2); OS patch cadence (unattended
  upgrades are Ubuntu-default, but plan reboots).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `terraform apply` fails on APIs | first apply in a fresh project can race API enablement — re-run apply |
| VM up but `app_url` 502s | no release installed yet: `deploy/gcp/scripts/update_app.sh`; check `journalctl -u mmm-api` |
| API refuses to boot, log mentions sandbox | fail-closed hosted guard: kernel image missing/unpullable — run step 2, then `sudo mmm-update` |
| Kernel spawn fails at runtime | on the VM as `mmm`: `podman info`; re-run the verify line from `mmm-update`; check subuid/subgid + `loginctl enable-linger mmm` |
| Cert stuck `PROVISIONING`/`FAILED_NOT_VISIBLE` | DNS A record doesn't point at `lb_ip` yet; wait after fixing |
| Chat stream stalls behind LB | confirm nginx has `proxy_buffering off` and the backend service `timeout_sec = 3600` (both set by this stack — check for local edits) |
| Vertex 404 "model not found" | use the exact Model Garden ID (with `@version`) and a serving region (`llm_location`, e.g. `us-east5` for Claude) |
| KB ingest "no embedding backend" | embeddings are separate from chat: `embed_provider = "vertex"` needs `aiplatform.user` (granted) and a region serving `text-embedding-005` |
| Login 401 after re-apply | JWT secret rotated (taint/recreate) — seats must log in again |
| Fits die around 2 GB | per-kernel cgroup cap: raise `kernel_mem` (and check `machine_type` headroom) |
| Disk full on `/data` | old thread workspaces + model runs accumulate; grow the disk or prune old `threads/<id>` dirs (they are per-session artifacts) |

## Cost (rough, us-central1, on-demand)

| Item | ~$/mo |
|---|---|
| `n2-standard-8` VM | ~$230 |
| 200 GB pd-balanced + snapshots | ~$25–35 |
| HTTPS LB + forwarding rules | ~$20–40 (usage-dependent) |
| NAT, Artifact Registry, GCS, Secret Manager | ~$5–15 |
| Vertex AI | usage-based (per-token; dominates under heavy agent use) |

Stop the VM (`gcloud compute instances stop mmm-app`) when idle — state is on
the persistent disk; everything resumes on start.

## File map

```
deploy/gcp/
├── README.md                  ← you are here
├── terraform/
│   ├── versions.tf            providers + optional GCS state backend
│   ├── variables.tf           all knobs (sizing, domain, LLM, seats bootstrap)
│   ├── apis.tf                project service enablement
│   ├── network.tf             VPC, subnet, NAT, firewall (LB/HC, IAP SSH)
│   ├── iam.tf                 VM service account + least-privilege grants
│   ├── registry.tf            Artifact Registry repo + kernel image ref logic
│   ├── secrets.tf             generated JWT key + bootstrap password (+ LLM key)
│   ├── storage.tf             release bucket, seeds vm/vm_setup.sh
│   ├── compute.tf             VM, data disk, snapshots, startup wiring
│   ├── lb.tf                  HTTPS LB, managed cert, redirect, optional DNS
│   ├── outputs.tf             app_url, ssh_command, next_steps, …
│   ├── terraform.tfvars.example
│   └── templates/startup.sh.tpl   thin boot script (writes deploy.env, runs vm_setup)
├── vm/vm_setup.sh             full VM provisioning (packages, user, env, systemd,
│                              nginx, mmm-update) — idempotent, re-runs every boot
└── scripts/
    ├── build_kernel_image.sh  build + sandbox-verify + push mmm-kernel
    ├── package_release.sh     git archive + frontend build → GCS release
    └── update_app.sh          package (optional) + install/rollback over IAP SSH
```

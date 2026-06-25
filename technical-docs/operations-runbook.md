# Operations Runbook

On-call reference for running the MMM platform. Pair with
[disaster-recovery.md](disaster-recovery.md) for backup/restore.

## Services

| Service | Start | Port |
|---------|-------|------|
| Agent API (canonical) | `uv run uvicorn mmm_framework.api.main:app --host 0.0.0.0 --port 8000` | 8000 |
| React UI | `cd frontend && npm run dev` | 5173 |
| Legacy REST API (deprecated) | `cd api && uvicorn main:app` + `arq worker.WorkerSettings` + Redis | 8000 |

The agent API runs fits in-kernel — no Redis/worker required for the canonical path.

## Health & observability

- `GET /health` — liveness.
- `GET /metrics` — Prometheus text (audit-event counters, live kernel gauge,
  `mmm_active_fits`). NOTE: in-process counters reset on restart and are
  per-replica until a durable metrics store is wired (action plan O4).
- `GET /observability` — audit-chain integrity + ship backlog + fit activity.
- Logs: stdout (loguru). `mmm_audit` logger carries kernel lifecycle/security
  events. Swallowed best-effort failures now log via `logged_suppress` (DEBUG).

## Configuration (key env)

| Concern | Env |
|---------|-----|
| Hosted posture (sandbox, per-session reports, fail-closed) | `MMM_AGENT_HOSTED=1` |
| Auth on | `MMM_AUTH_ENABLED=1`, `MMM_AUTH_SECRET=…` |
| Rate limiting | `MMM_RATELIMIT_ENABLED=1`, `MMM_RATELIMIT_BACKEND=redis` (+ `_REDIS_URL`) for multi-replica |
| CORS allowlist | `MMM_CORS_ORIGINS=https://app.example.com,https://…` |
| Encryption-at-rest | `MMM_ENCRYPTION_KEY=<fernet key>` |
| Kernel sandbox | `MMM_AGENT_KERNEL=container` (needs the kernel image) |
| LLM provider | `config/model_config.yaml` / `MMM_LLM_*` |

## Common incidents

**API 5xx storm / high latency.** Check `/health`, `/metrics` (`mmm_active_fits`).
A long synchronous NUTS fit can occupy a worker for minutes — confirm via logs;
size workers accordingly. Roll back the last deploy if it correlates.

**Rate limiting ineffective across replicas.** The default limiter is per-process.
Set `MMM_RATELIMIT_BACKEND=redis` + `MMM_RATELIMIT_REDIS_URL` so the limit is global.

**Disk full.** Likely the SQLite DB (multi-GB), the agent workspace, or
`/backups`. Rotate backups (the DR cron prunes `-mtime +14`); move the workspace
to a larger volume (`MMM_AGENT_WORKSPACE`).

**Data loss / corruption.** Follow [disaster-recovery.md](disaster-recovery.md)
restore. Stop API + worker first.

**Bad migration / schema drift.** Schema changes are additive `ALTER TABLE`
guards; if a deploy wedges the DB, restore the pre-deploy backup and redeploy the
prior image.

**Untrusted code execution risk.** In hosted mode the kernel must be the
container sandbox; `assert_hosted_sandbox` refuses to boot otherwise. If you see
in-process kernel under `MMM_AGENT_HOSTED=1`, the image is missing — do not serve
untrusted sessions until fixed.

## Deploy / rollback

1. Build the API image (`deploy/api/Dockerfile`) and the kernel image
   (`deploy/kernel/Containerfile`).
2. Take a DB backup (DR doc) before any release that touches schema.
3. Deploy; verify `/health`, a login, a project/runs listing, one smoke fit.
4. Rollback = redeploy the prior image; restore the pre-deploy DB backup only if
   the schema changed.

## Escalation

Page the on-call data-science owner for: data loss, auth/isolation breach,
sandbox-escape suspicion, or sustained API unavailability.

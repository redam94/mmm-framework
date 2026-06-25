# Disaster Recovery (DR)

Status: **interim** — covers the current single-node SQLite deployment. Superseded
in part once the Postgres + object-store migration lands (action plan P1a/P1b).

## What state exists

| State | Location | Criticality |
|-------|----------|-------------|
| Sessions, auth (users/orgs/tokens), run_metrics, langgraph checkpoints | one SQLite file (`src/mmm_framework/api/sessions.db`) | **critical** |
| Fitted models, plots, tables, KB | agent workspace (`$MMM_AGENT_WORKSPACE`, default `./agent_workspace`) | high |
| Uploaded client datasets | storage dir (local FS today) | high |
| Audit chain | `mmm_audit` log + off-host shipper (if configured) | medium |

> **Known gap (P1a/P1b):** all tenant state in one local SQLite file + local-FS
> artifacts is a single point of failure. The Postgres + object-store migration
> removes it. Until then, the procedures below are the mitigation.

## Targets (interim, single node)

- **RPO** (max data loss): **≤ 24h** with a nightly backup; **≤ 1h** if the
  backup cron runs hourly (recommended for production tenants).
- **RTO** (max downtime to restore): **≤ 30 min** (restore the DB + restart).

## Backup

The SQLite store has an online, WAL-consistent backup (safe while running):

```bash
# one-off
python -m mmm_framework.api.backup backup /backups/sessions-$(date +%F).db

# hourly cron (example)
0 * * * * cd /app && python -m mmm_framework.api.backup backup \
    /backups/sessions-$(date +\%F-\%H).db && \
    find /backups -name 'sessions-*.db' -mtime +14 -delete
```

Also back up (rsync/object-store sync) the **agent workspace** and the **dataset
storage dir** on the same cadence — models/artifacts are not in the DB.

Off-site copies: push `/backups` to object storage (e.g. `aws s3 sync`) so a node
loss is recoverable. Verify backups by periodically restoring into a scratch path.

## Restore

```bash
# stop the API + worker first
python -m mmm_framework.api.backup restore /backups/sessions-2026-06-25.db
# restore the workspace + dataset dirs from their backups, then restart
```

`restore` writes a consistent copy into the live DB path via the SQLite backup
API (handles WAL). Confirm with a health check (`GET /health`) and a smoke login.

## Drill checklist (run quarterly)

1. Take a fresh backup; copy it to a clean host/scratch dir.
2. Restore into a scratch DB path (`--db /tmp/restore-test.db`).
3. Point a throwaway API instance at it (`MMM_SESSIONS_DB`/`DB_PATH`); verify
   `/health`, a login, and a project/runs listing.
4. Record the wall-clock restore time → update the RTO figure above.

## Post-migration (P1a/P1b)

Once on Postgres + object store: use managed automated backups + PITR (Postgres),
versioned object storage for artifacts, and a cross-region replica. Re-baseline
RPO/RTO and delete the SQLite-specific steps above.

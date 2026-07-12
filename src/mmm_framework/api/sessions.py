"""Session metadata store + per-session artifact log.

The langgraph SqliteSaver owns its own tables for checkpoints. We add two
sibling tables in the same DB file:

  sessions(thread_id PK, name, created_at, updated_at)
  artifacts(id PK, thread_id FK, kind, payload_json, created_at)

`kind` is one of: 'code_snippet', 'report', 'plot', 'saved_config',
'saved_model'. `payload_json` is whatever the frontend needs to render or
rerun the artifact (e.g. {"code": "..."} for code_snippet).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def resolve_db_path() -> Path:
    """Resolve the sessions DB location.

    ``MMM_SESSIONS_DB`` overrides the package-local default so deployments can
    keep state on a persistent disk without symlinking into the install tree
    (see deploy/gcp/vm/vm_setup.sh). ``api/main.py`` (the async checkpointer)
    and ``auth/store.py`` honor the same variable — all three must point at the
    same file.
    """
    env = os.environ.get("MMM_SESSIONS_DB", "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).parent / "sessions.db"


DB_PATH = resolve_db_path()


def _now() -> float:
    return time.time()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    # ``sessions.db`` is shared with the LangGraph AsyncSqliteSaver checkpointer
    # (see api/main.py), which writes a checkpoint on every graph step during a
    # chat stream. WAL mode lets a reader and a single writer coexist and avoids
    # the rollback-journal lock-escalation deadlock that surfaced as
    # "database is locked"; ``timeout`` makes a contending writer wait for the
    # lock instead of failing immediately. WAL is a persistent DB-level property,
    # so setting it here also covers the async checkpointer connection.
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id  TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                project_id TEXT,
                modeling_mode TEXT
            )
            """)
        # Migrate existing installs that predate the project_id column
        try:
            c.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")
        except Exception:
            pass
        # Migrate installs that predate the modeling_mode column (NULL == "mmm")
        try:
            c.execute("ALTER TABLE sessions ADD COLUMN modeling_mode TEXT")
        except Exception:
            pass
        c.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                kind         TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at   REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(thread_id, created_at)"
        )

        # Modeling assumptions: a key/value log per session with full versioned
        # history. Each row is an immutable snapshot; the "current" value for a
        # key is the row with the highest version. Updates write a new row.
        c.execute("""
            CREATE TABLE IF NOT EXISTS assumptions (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                key          TEXT NOT NULL,
                category     TEXT NOT NULL,
                value_json   TEXT NOT NULL,
                rationale    TEXT NOT NULL,
                change_note  TEXT,
                version      INTEGER NOT NULL,
                is_tombstone INTEGER NOT NULL DEFAULT 0,
                created_at   REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_assumptions_thread_key ON assumptions(thread_id, key, version)"
        )

        # Workflow status: which of the 9 canonical steps are done/in-progress.
        # We mostly infer status from state, but store manual overrides + notes
        # here. One row per (thread_id, step).
        c.execute("""
            CREATE TABLE IF NOT EXISTS workflow_status (
                thread_id   TEXT NOT NULL,
                step        INTEGER NOT NULL,
                status      TEXT NOT NULL,
                notes       TEXT,
                updated_at  REAL NOT NULL,
                PRIMARY KEY (thread_id, step)
            )
            """)

        # Data files associated with a session: uploads, generated datasets,
        # exports, plots, etc. Distinct from artifacts (which are code/report
        # snippets); this table is the "Files" tab in the UI.
        c.execute("""
            CREATE TABLE IF NOT EXISTS data_files (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                path         TEXT NOT NULL,
                name         TEXT NOT NULL,
                kind         TEXT NOT NULL,
                size_bytes   INTEGER,
                preview      TEXT,
                meta_json    TEXT,
                created_at   REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_data_files_thread ON data_files(thread_id, created_at)"
        )

        # Locked analysis plans: a snapshot of research_question + DAG + assumptions
        # at the moment the analyst decides to "lock" the pre-registration.
        c.execute("""
            CREATE TABLE IF NOT EXISTS analysis_plans (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                name         TEXT NOT NULL,
                locked_at    REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_plans_thread ON analysis_plans(thread_id, locked_at)"
        )

        # Lift-experiment registry: the project-level log of planned / running /
        # completed / calibrated experiments that the home page tracks and the
        # agent reads when deciding whether a model refresh should fold new
        # results in. status='completed' = measured but NOT yet calibrated into
        # a fit; 'calibrated' closes the loop.
        c.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          TEXT PRIMARY KEY,
                project_id  TEXT,
                thread_id   TEXT,
                channel     TEXT NOT NULL,
                design_type TEXT,
                status      TEXT NOT NULL DEFAULT 'planned',
                start_date  TEXT,
                end_date    TEXT,
                estimand    TEXT,
                value       REAL,
                se          REAL,
                notes       TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_project"
            " ON experiments(project_id, updated_at)"
        )
        # Lifecycle-registry columns (migrate installs that predate them):
        # links to the recommending/consuming model runs, the full design /
        # readout / priority payloads, the pre-registration timestamp, and an
        # append-only status audit trail.
        for col, decl in (
            ("recommending_run_id", "TEXT"),
            ("calibrated_run_id", "TEXT"),
            ("design_json", "TEXT"),
            ("readout_json", "TEXT"),
            ("priority_json", "TEXT"),
            ("preregistered_at", "REAL"),
            ("status_history_json", "TEXT"),
            # creative / keyword / campaign identifier (nullable) — sub-channel
            # readouts feed continuous-learning programs with arms; MMM
            # calibration stays channel-level.
            ("subchannel", "TEXT"),
        ):
            try:
                c.execute(f"ALTER TABLE experiments ADD COLUMN {col} {decl}")
            except Exception:
                pass
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_status"
            " ON experiments(project_id, status)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_channel"
            " ON experiments(project_id, channel, updated_at)"
        )

        # Per-run history metrics: one JSON snapshot per model run (ROI
        # posteriors, budget shares, EIG/EVOI, misallocation proxy) so the
        # Performance page plots trajectories without unpickling models.
        # metrics_json carries its own schema_version; rows are never mutated.
        c.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id         TEXT PRIMARY KEY,
                artifact_id    TEXT,
                thread_id      TEXT,
                project_id     TEXT,
                created_at     REAL NOT NULL,
                schema_version INTEGER NOT NULL,
                metrics_json   TEXT NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_metrics_project"
            " ON run_metrics(project_id, created_at)"
        )

        # Model Garden registry: bespoke, oracle-compatible MMM models authored
        # by experts, versioned + documented so the agent can re-fit them on any
        # project's data and they can be shared across a tenant's projects.
        # ORG-scoped (NOT project-scoped) so sharing is cross-project by
        # construction. manifest_json carries {contract_version, class_name,
        # dataset_schema, recommended_fit, tags}; source lives on disk at
        # source_path (workspace garden_dir); status_history_json is append-only.
        c.execute("""
            CREATE TABLE IF NOT EXISTS garden_models (
                id                      TEXT PRIMARY KEY,
                org_id                  TEXT NOT NULL,
                name                    TEXT NOT NULL,
                version                 INTEGER NOT NULL,
                owner_user_id           TEXT,
                status                  TEXT NOT NULL DEFAULT 'draft',
                docs                    TEXT,
                manifest_json           TEXT,
                source_path             TEXT,
                compat_report_json      TEXT,
                base_run_id             TEXT,
                reference_artifact_path TEXT,
                status_history_json     TEXT,
                created_at              REAL NOT NULL,
                updated_at              REAL NOT NULL,
                UNIQUE(org_id, name, version)
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_garden_models_org"
            " ON garden_models(org_id, status, updated_at)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_garden_models_name"
            " ON garden_models(org_id, name, version)"
        )

        # Saved data-source connections (project-scoped). config_json holds ONLY
        # a non-secret reference (bucket/prefix/object, or project/dataset/query/
        # table) — NEVER credentials. Auth stays ambient (ADC / the server's
        # MMM_GCP_CREDENTIALS_PATH), so there is no secret to encrypt at rest.
        c.execute("""
            CREATE TABLE IF NOT EXISTS data_connections (
                id          TEXT PRIMARY KEY,
                project_id  TEXT,
                name        TEXT NOT NULL,
                kind        TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                last_synced REAL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_data_connections_project"
            " ON data_connections(project_id, updated_at)"
        )
        # Scheduled-sync + freshness columns (migrate installs that predate them).
        # org_id mirrors the owning project's org for tenant-level auditability
        # (the scheduler is server-wide; access control is via project_id).
        for _col, _decl in (
            ("org_id", "TEXT"),
            ("sync_interval_minutes", "REAL"),
            ("next_sync_at", "REAL"),
            ("last_sync_status", "TEXT"),
            ("last_sync_error", "TEXT"),
            ("last_row_count", "INTEGER"),
            ("snapshot_path", "TEXT"),
        ):
            try:
                c.execute(f"ALTER TABLE data_connections ADD COLUMN {_col} {_decl}")
            except Exception:
                pass
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_data_connections_due"
            " ON data_connections(next_sync_at)"
        )

        # Projects: group sessions and own a knowledge base. A session belongs
        # to a project via sessions.project_id. meta_json carries the
        # onboarding profile (client info, goals, KPI/channel context) that
        # also gets rendered into a KB "project brief".
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                meta_json   TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """)
        try:
            c.execute("ALTER TABLE projects ADD COLUMN meta_json TEXT")
        except Exception:
            pass

        # Team roster + project membership. A registry for attribution and
        # assignment (who owns what, who signs off) — not an authentication
        # system; the hosted profile handles access control.
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                email      TEXT UNIQUE,
                role       TEXT NOT NULL DEFAULT 'analyst',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS project_members (
                project_id TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'analyst',
                PRIMARY KEY (project_id, user_id)
            )
            """)

        # Knowledge-base documents: the source files a user adds for context,
        # scoped to a project. Bytes live on disk (path); chunks+embeddings in
        # kb_chunks.
        c.execute("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id         TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name       TEXT NOT NULL,
                path       TEXT NOT NULL,
                kind       TEXT NOT NULL,
                size_bytes INTEGER,
                n_chunks   INTEGER NOT NULL DEFAULT 0,
                status     TEXT NOT NULL,
                error      TEXT,
                meta_json  TEXT,
                created_at REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_docs_project ON kb_documents(project_id, created_at)"
        )

        # Knowledge-base chunks: one row per text chunk with its embedding stored
        # as a float32 little-endian BLOB. Brute-force cosine search avoids a
        # vector-store dependency.
        c.execute("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id          TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                project_id  TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                dim         INTEGER NOT NULL,
                created_at  REAL NOT NULL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_project ON kb_chunks(project_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(document_id)"
        )

        # User/project preferences: small JSON values keyed by (scope, key).
        # scope is 'global' (deployment-wide defaults: favorite palette, number
        # formats) or a project_id (per-client branding, report preferences).
        c.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                scope      TEXT NOT NULL,
                key        TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, key)
            )
            """)
        # Saved budget plans (Planner). A plan persists a computed allocation /
        # scenario so it survives the chat turn; ``plan_payload`` holds the rich
        # studio result (geo allocation + flighting calendar) as JSON.
        c.execute("""
            CREATE TABLE IF NOT EXISTS budget_plans (
                id                 TEXT PRIMARY KEY,
                project_id         TEXT,
                org_id             TEXT NOT NULL,
                name               TEXT NOT NULL,
                description        TEXT,
                model_id           TEXT,
                kind               TEXT NOT NULL DEFAULT 'optimization',
                spend_changes      TEXT,
                baseline_outcome   REAL,
                scenario_outcome   REAL,
                outcome_change     REAL,
                outcome_change_pct REAL,
                channel_details    TEXT,
                plan_payload       TEXT,
                created_at         REAL NOT NULL,
                updated_at         REAL NOT NULL
            )
            """)
        for _ddl in (
            "CREATE INDEX IF NOT EXISTS idx_budget_plans_org"
            " ON budget_plans(org_id, updated_at)",
            "CREATE INDEX IF NOT EXISTS idx_budget_plans_project"
            " ON budget_plans(project_id, updated_at)",
        ):
            c.execute(_ddl)

        # Actual in-flight delivery (issue #123): one row per
        # (project, channel, period) so a re-upload of a period overwrites it.
        # The planned side is auto-sourced from the saved budget plan; pacing
        # compares this stored actual against it without passing spend inline.
        c.execute("""
            CREATE TABLE IF NOT EXISTS delivery (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                channel     TEXT NOT NULL,
                period      TEXT NOT NULL DEFAULT '',
                spend       REAL NOT NULL,
                source      TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL,
                UNIQUE(project_id, channel, period)
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_delivery_project"
            " ON delivery(project_id, updated_at)"
        )

        # Continuous-learning programs (model-free geo response-surface bandit).
        # A program is a statused, project-scoped, longitudinal entity; heavy
        # state (posterior draws + the accumulated panel) lives on disk at
        # state_path (.npz), SQLite holds paths + JSON snapshots only —
        # exactly the experiments/run_metrics idiom.
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_programs (
                id            TEXT PRIMARY KEY,
                project_id    TEXT,
                thread_id     TEXT,
                name          TEXT,
                status        TEXT NOT NULL DEFAULT 'active',
                channels_json TEXT NOT NULL,
                config_json   TEXT NOT NULL,
                state_path    TEXT,
                summary_json  TEXT,
                created_at    REAL,
                updated_at    REAL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_learning_programs_project"
            " ON learning_programs(project_id, updated_at)"
        )
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_waves (
                id                  TEXT PRIMARY KEY,
                program_id          TEXT NOT NULL,
                project_id          TEXT,
                wave_index          INTEGER NOT NULL,
                status              TEXT NOT NULL DEFAULT 'designed',
                source              TEXT,
                design_json         TEXT,
                observations_json   TEXT,
                snapshot_json       TEXT,
                experiment_ids_json TEXT,
                created_at          REAL,
                updated_at          REAL
            )
            """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_learning_waves_program"
            " ON learning_waves(program_id, wave_index)"
        )


def list_sessions(project_id: str | None = None) -> list[dict[str, Any]]:
    with _conn() as c:
        _cols = (
            "thread_id, name, created_at, updated_at, project_id,"
            " COALESCE(modeling_mode, 'mmm') AS modeling_mode"
        )
        if project_id is not None:
            rows = c.execute(
                f"SELECT {_cols} FROM sessions"
                " WHERE project_id = ? ORDER BY updated_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = c.execute(
                f"SELECT {_cols} FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_session(thread_id: str) -> dict[str, Any] | None:
    """Return a single session row with artifact_count, or None if not found."""
    with _conn() as c:
        row = c.execute(
            "SELECT thread_id, name, created_at, updated_at, project_id,"
            " COALESCE(modeling_mode, 'mmm') AS modeling_mode"
            " FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        session = dict(row)
        count_row = c.execute(
            "SELECT COUNT(*) AS n FROM artifacts WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        session["artifact_count"] = count_row["n"] if count_row else 0
        return session


def create_session(
    name: str | None = None,
    project_id: str | None = None,
    modeling_mode: str | None = None,
) -> dict[str, Any]:
    thread_id = uuid.uuid4().hex
    now = _now()
    display_name = (
        name or f"Session {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}"
    )
    mode = modeling_mode or "mmm"
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (thread_id, name, created_at, updated_at, project_id,"
            " modeling_mode) VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, display_name, now, now, project_id, mode),
        )
    return {
        "thread_id": thread_id,
        "name": display_name,
        "created_at": now,
        "updated_at": now,
        "project_id": project_id,
        "modeling_mode": mode,
    }


def update_session(
    thread_id: str,
    name: str | None = None,
    project_id: str | None = None,
    modeling_mode: str | None = None,
) -> bool:
    """Update session name, project_id and/or modeling_mode. Returns True if found."""
    updates = []
    params: list[Any] = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if project_id is not None:
        updates.append("project_id = ?")
        params.append(project_id)
    if modeling_mode is not None:
        updates.append("modeling_mode = ?")
        params.append(modeling_mode)
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(_now())
    params.append(thread_id)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE sessions SET {', '.join(updates)} WHERE thread_id = ?",
            params,
        )
        return cur.rowcount > 0


def touch_session(thread_id: str) -> None:
    """Bump updated_at; if the row doesn't exist (legacy thread), insert it."""
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row:
            c.execute(
                "UPDATE sessions SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
        else:
            c.execute(
                "INSERT INTO sessions (thread_id, name, created_at, updated_at, project_id) VALUES (?, ?, ?, ?, NULL)",
                (
                    thread_id,
                    f"Session {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}",
                    now,
                    now,
                ),
            )


def rename_session(thread_id: str, name: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE sessions SET name = ?, updated_at = ? WHERE thread_id = ?",
            (name, _now(), thread_id),
        )
        return cur.rowcount > 0


def delete_session(thread_id: str) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM artifacts WHERE thread_id = ?", (thread_id,))
        cur = c.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        # Note: langgraph checkpoints for this thread_id remain in the same DB.
        # SqliteSaver doesn't expose a public delete-thread API, so we leave
        # them; they're harmless once the session is gone from the listing.
        return cur.rowcount > 0


def add_artifact(thread_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    artifact = {
        "id": uuid.uuid4().hex,
        "thread_id": thread_id,
        "kind": kind,
        "payload": payload,
        "created_at": _now(),
    }
    with _conn() as c:
        c.execute(
            "INSERT INTO artifacts (id, thread_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                artifact["id"],
                thread_id,
                kind,
                json.dumps(payload, default=str),
                artifact["created_at"],
            ),
        )
    return artifact


def list_artifacts(thread_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, thread_id, kind, payload_json, created_at FROM artifacts WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {"raw": r["payload_json"]}
        out.append(
            {
                "id": r["id"],
                "thread_id": r["thread_id"],
                "kind": r["kind"],
                "payload": payload,
                "created_at": r["created_at"],
            }
        )
    return out


def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute(
            "SELECT id, thread_id, kind, payload_json, created_at FROM artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
    if r is None:
        return None
    try:
        payload = json.loads(r["payload_json"])
    except Exception:
        payload = {"raw": r["payload_json"]}
    return {
        "id": r["id"],
        "thread_id": r["thread_id"],
        "kind": r["kind"],
        "payload": payload,
        "created_at": r["created_at"],
    }


def delete_artifact(artifact_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        return cur.rowcount > 0


def update_artifact_payload(
    artifact_id: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Overwrite an artifact's payload (used by the async experiment-simulation
    job to flip status pending → running → done/error in place)."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE artifacts SET payload_json = ? WHERE id = ?",
            (json.dumps(payload, default=str), artifact_id),
        )
        if cur.rowcount == 0:
            return None
    return get_artifact(artifact_id)


# ── Assumptions log ───────────────────────────────────────────────────────────

ASSUMPTION_CATEGORIES = {
    "research_question",  # what we're estimating
    "causal_structure",  # DAG-level claims (e.g. "TV does not confound display")
    "data",  # data assumptions (e.g. "weekly aggregation is appropriate")
    "functional_form",  # adstock/saturation choices
    "prior",  # specific prior choices and their justification
    "identification",  # backdoor/instrument/frontdoor claims
    "external_evidence",  # benchmarks, experiments, prior MMMs
    "other",
}


def record_assumption(
    thread_id: str,
    key: str,
    value: Any,
    rationale: str,
    category: str = "other",
    change_note: str | None = None,
) -> dict[str, Any]:
    """Create or update an assumption. New version always appended; history preserved."""
    if category not in ASSUMPTION_CATEGORIES:
        category = "other"
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM assumptions WHERE thread_id = ? AND key = ?",
            (thread_id, key),
        ).fetchone()
        next_version = (row["v"] or 0) + 1
        aid = uuid.uuid4().hex
        c.execute(
            """
            INSERT INTO assumptions
            (id, thread_id, key, category, value_json, rationale, change_note, version, is_tombstone, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                aid,
                thread_id,
                key,
                category,
                json.dumps(value, default=str),
                rationale,
                change_note,
                next_version,
                now,
            ),
        )
    return {
        "id": aid,
        "thread_id": thread_id,
        "key": key,
        "category": category,
        "value": value,
        "rationale": rationale,
        "change_note": change_note,
        "version": next_version,
        "is_tombstone": False,
        "created_at": now,
    }


def retract_assumption(thread_id: str, key: str, reason: str) -> dict[str, Any] | None:
    """Mark an assumption as retracted (tombstone); preserved in history."""
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(version) AS v FROM assumptions WHERE thread_id = ? AND key = ?",
            (thread_id, key),
        ).fetchone()
        if not row or row["v"] is None:
            return None
        next_version = row["v"] + 1
        aid = uuid.uuid4().hex
        c.execute(
            """
            INSERT INTO assumptions
            (id, thread_id, key, category, value_json, rationale, change_note, version, is_tombstone, created_at)
            VALUES (?, ?, ?, 'other', 'null', '', ?, ?, 1, ?)
            """,
            (aid, thread_id, key, reason, next_version, now),
        )
    return {
        "id": aid,
        "key": key,
        "version": next_version,
        "is_tombstone": True,
        "created_at": now,
    }


def _row_to_assumption(r: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(r["value_json"])
    except Exception:
        value = r["value_json"]
    return {
        "id": r["id"],
        "thread_id": r["thread_id"],
        "key": r["key"],
        "category": r["category"],
        "value": value,
        "rationale": r["rationale"],
        "change_note": r["change_note"],
        "version": r["version"],
        "is_tombstone": bool(r["is_tombstone"]),
        "created_at": r["created_at"],
    }


def list_assumptions(
    thread_id: str, include_history: bool = False
) -> list[dict[str, Any]]:
    """If include_history is False, returns the latest version per key.
    Otherwise returns every row ordered by (key, version).
    """
    with _conn() as c:
        if include_history:
            rows = c.execute(
                "SELECT * FROM assumptions WHERE thread_id = ? ORDER BY key ASC, version ASC",
                (thread_id,),
            ).fetchall()
            return [_row_to_assumption(r) for r in rows]
        # Latest-only: a self-join would work but a simple Python-side reduce is clearer.
        rows = c.execute(
            "SELECT * FROM assumptions WHERE thread_id = ? ORDER BY key ASC, version ASC",
            (thread_id,),
        ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        latest[r["key"]] = _row_to_assumption(r)
    # Drop tombstones from the latest view; their history is still accessible.
    return [a for a in latest.values() if not a["is_tombstone"]]


def get_assumption_history(thread_id: str, key: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM assumptions WHERE thread_id = ? AND key = ? ORDER BY version ASC",
            (thread_id, key),
        ).fetchall()
    return [_row_to_assumption(r) for r in rows]


# ── Workflow status ──────────────────────────────────────────────────────────

WORKFLOW_STEPS = [
    (1, "Define the Question", "Pre-register the causal/business question."),
    (2, "Tell the Story of Your Data", "Generative narrative + DAG."),
    (3, "Build the Model", "Specify components and priors."),
    (
        4,
        "Prior Predictive Check",
        "Simulate from priors; sanity-check implied outcomes.",
    ),
    (5, "Fit the Model", "Run MCMC."),
    (6, "Computational Diagnostics", "R-hat, ESS, divergences."),
    (7, "Posterior Predictive Check", "Compare fit to observed data."),
    (8, "Sensitivity Analysis", "Stress-test conclusions."),
    (9, "Communicate Results", "Honest uncertainty + decision."),
]


def set_workflow_step(
    thread_id: str, step: int, status: str, notes: str | None = None
) -> dict[str, Any]:
    now = _now()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO workflow_status (thread_id, step, status, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, step) DO UPDATE SET
                status = excluded.status,
                notes  = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (thread_id, step, status, notes, now),
        )
    return {
        "thread_id": thread_id,
        "step": step,
        "status": status,
        "notes": notes,
        "updated_at": now,
    }


def get_workflow_overrides(thread_id: str) -> dict[int, dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT step, status, notes, updated_at FROM workflow_status WHERE thread_id = ?",
            (thread_id,),
        ).fetchall()
    return {int(r["step"]): dict(r) for r in rows}


# ── Data files registry ──────────────────────────────────────────────────────


def register_file(
    thread_id: str,
    path: str,
    name: str,
    kind: str,
    size_bytes: int | None = None,
    preview: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fid = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        c.execute(
            """
            INSERT INTO data_files (id, thread_id, path, name, kind, size_bytes, preview, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fid,
                thread_id,
                path,
                name,
                kind,
                size_bytes,
                preview,
                json.dumps(meta or {}, default=str),
                now,
            ),
        )
    return {
        "id": fid,
        "thread_id": thread_id,
        "path": path,
        "name": name,
        "kind": kind,
        "size_bytes": size_bytes,
        "preview": preview,
        "meta": meta or {},
        "created_at": now,
    }


def list_files(thread_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM data_files WHERE thread_id = ? ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        except Exception:
            meta = {}
        out.append(
            {
                "id": r["id"],
                "thread_id": r["thread_id"],
                "path": r["path"],
                "name": r["name"],
                "kind": r["kind"],
                "size_bytes": r["size_bytes"],
                "preview": r["preview"],
                "meta": meta,
                "created_at": r["created_at"],
            }
        )
    return out


def get_file(file_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM data_files WHERE id = ?", (file_id,)).fetchone()
    if r is None:
        return None
    try:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
    except Exception:
        meta = {}
    return {
        "id": r["id"],
        "thread_id": r["thread_id"],
        "path": r["path"],
        "name": r["name"],
        "kind": r["kind"],
        "size_bytes": r["size_bytes"],
        "preview": r["preview"],
        "meta": meta,
        "created_at": r["created_at"],
    }


def delete_file(file_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM data_files WHERE id = ?", (file_id,))
        return cur.rowcount > 0


# ── Analysis plans ────────────────────────────────────────────────────────────


def lock_analysis_plan(
    thread_id: str,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Snapshot the current analysis plan (research question + DAG + assumptions) as a locked record."""
    plan_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO analysis_plans (id, thread_id, name, locked_at, payload_json) VALUES (?, ?, ?, ?, ?)",
            (plan_id, thread_id, name, now, json.dumps(payload, default=str)),
        )
    return {
        "id": plan_id,
        "thread_id": thread_id,
        "name": name,
        "locked_at": now,
        "payload": payload,
    }


def list_analysis_plans(thread_id: str | None = None) -> list[dict[str, Any]]:
    """List analysis plans for a thread (or all plans if thread_id is None)."""
    with _conn() as c:
        if thread_id is not None:
            rows = c.execute(
                "SELECT id, thread_id, name, locked_at, payload_json FROM analysis_plans"
                " WHERE thread_id = ? ORDER BY locked_at DESC",
                (thread_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, thread_id, name, locked_at, payload_json FROM analysis_plans"
                " ORDER BY locked_at DESC"
            ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload_json"])
        except Exception:
            payload = {}
        out.append(
            {
                "id": r["id"],
                "thread_id": r["thread_id"],
                "name": r["name"],
                "locked_at": r["locked_at"],
                "payload": payload,
            }
        )
    return out


def get_analysis_plan(plan_id: str) -> dict[str, Any] | None:
    """Get a single analysis plan by ID."""
    with _conn() as c:
        row = c.execute(
            "SELECT id, thread_id, name, locked_at, payload_json FROM analysis_plans WHERE id = ?",
            (plan_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except Exception:
        payload = {}
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "name": row["name"],
        "locked_at": row["locked_at"],
        "payload": payload,
    }


def delete_analysis_plan(plan_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM analysis_plans WHERE id = ?", (plan_id,))
        return cur.rowcount > 0


# ── Experiments (lift-test registry) ─────────────────────────────────────────

# Lifecycle: draft → planned (pre-registered) → running → completed (measured,
# not yet folded into a fit) → calibrated (closes the loop). "abandoned" exits
# from any active state; "cancelled" is a legacy alias kept readable/writable.
EXPERIMENT_STATUSES = (
    "draft",
    "planned",
    "running",
    "completed",
    "calibrated",
    "abandoned",
    "cancelled",
)

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"planned", "abandoned"},
    "planned": {"running", "abandoned"},
    "running": {"completed", "abandoned"},
    "completed": {"calibrated", "abandoned"},
    "calibrated": set(),
    "abandoned": set(),
    "cancelled": set(),
}


def _transition_path(current: str, target: str) -> list[str] | None:
    """Shortest legal path ``current -> ... -> target`` through
    :data:`ALLOWED_TRANSITIONS` (BFS).

    Returns the hop statuses after ``current`` (last element == ``target``),
    or ``None`` when the target is unreachable.
    """
    seen = {current}
    queue: list[tuple[str, list[str]]] = [(current, [])]
    while queue:
        node, path = queue.pop(0)
        for nxt in sorted(ALLOWED_TRANSITIONS.get(node, set())):
            if nxt in seen:
                continue
            hops = path + [nxt]
            if nxt == target:
                return hops
            seen.add(nxt)
            queue.append((nxt, hops))
    return None


def _json_or_none(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _experiment_row_to_dict(r) -> dict[str, Any]:
    keys = set(r.keys())
    return {
        "id": r["id"],
        "project_id": r["project_id"],
        "thread_id": r["thread_id"],
        "channel": r["channel"],
        "subchannel": r["subchannel"] if "subchannel" in keys else None,
        "design_type": r["design_type"],
        "status": r["status"],
        "start_date": r["start_date"],
        "end_date": r["end_date"],
        "estimand": r["estimand"],
        "value": r["value"],
        "se": r["se"],
        "notes": r["notes"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        # Lifecycle-registry fields (None on rows from pre-migration installs)
        "recommending_run_id": (
            r["recommending_run_id"] if "recommending_run_id" in keys else None
        ),
        "calibrated_run_id": (
            r["calibrated_run_id"] if "calibrated_run_id" in keys else None
        ),
        "design": _json_or_none(r["design_json"] if "design_json" in keys else None),
        "readout": _json_or_none(r["readout_json"] if "readout_json" in keys else None),
        "priority": _json_or_none(
            r["priority_json"] if "priority_json" in keys else None
        ),
        "preregistered_at": (
            r["preregistered_at"] if "preregistered_at" in keys else None
        ),
        "status_history": _json_or_none(
            r["status_history_json"] if "status_history_json" in keys else None
        )
        or [],
    }


def upsert_experiment(
    *,
    experiment_id: str | None = None,
    project_id: str | None = None,
    thread_id: str | None = None,
    channel: str | None = None,
    subchannel: str | None = None,
    design_type: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    estimand: str | None = None,
    value: float | None = None,
    se: float | None = None,
    notes: str | None = None,
    recommending_run_id: str | None = None,
    calibrated_run_id: str | None = None,
    design: dict[str, Any] | None = None,
    readout: dict[str, Any] | None = None,
    priority: dict[str, Any] | None = None,
    allow_calibrated_edit: bool = False,
) -> dict[str, Any]:
    """Create (no ``experiment_id``) or partially update an experiment record.

    On update, only the non-None fields change. Raises ValueError for an
    unknown id, a missing channel on create, or an invalid status.

    Status is state-machine enforced: creating directly in 'calibrated' is
    rejected (calibration happens via :func:`transition_experiment` at fit
    close-out); creating in planned/running/completed stays legal (historical
    import) but the full draft→…→status history chain is backfilled. An update
    may change status to anything REACHABLE through
    :data:`ALLOWED_TRANSITIONS` — multi-hop moves (e.g. planned→completed,
    the one-call results-recording flow) backfill the intermediate hops into
    the history — EXCEPT 'calibrated', which is only ever set by
    :func:`transition_experiment` (fit close-out). Same-status (or
    ``status=None``) updates stay silent no-ops for the other fields.

    A calibrated experiment's measurement fields (value / se / estimand /
    start_date / end_date / readout / channel / subchannel) feed the model's
    calibration likelihood, so CHANGING any of them raises unless
    ``allow_calibrated_edit=True`` (the ``record_experiment_readout`` tool's
    sanctioned, audited overwrite path). Unchanged re-sends and
    notes/design/priority-only updates stay allowed.
    """
    if status is not None and status not in EXPERIMENT_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Valid: {', '.join(EXPERIMENT_STATUSES)}"
        )
    now = _now()
    with _conn() as c:
        if experiment_id is None:
            if not channel:
                raise ValueError("channel is required to create an experiment")
            if status == "calibrated":
                raise ValueError(
                    "Illegal status for create: 'calibrated' — experiments "
                    "become calibrated via transition_experiment (fit "
                    "close-out), not on create."
                )
            experiment_id = uuid.uuid4().hex
            initial_status = status or "planned"
            # Historical import (explicit planned/running/completed): backfill
            # the full legal chain so the audit trail never shows an
            # unreachable state as the first entry.
            _chain = ("draft", "planned", "running", "completed")
            if status is not None and status in _chain[1:]:
                history = [
                    {"status": s, "at": now, "note": "backfilled on create"}
                    for s in _chain[: _chain.index(status) + 1]
                ]
            else:
                history = [{"status": initial_status, "at": now}]
            c.execute(
                "INSERT INTO experiments (id, project_id, thread_id, channel,"
                " subchannel, design_type, status, start_date, end_date, estimand,"
                " value, se, notes, recommending_run_id, calibrated_run_id,"
                " design_json, readout_json, priority_json, status_history_json,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    project_id,
                    thread_id,
                    channel,
                    subchannel,
                    design_type,
                    initial_status,
                    start_date,
                    end_date,
                    estimand,
                    value,
                    se,
                    notes,
                    recommending_run_id,
                    calibrated_run_id,
                    json.dumps(design) if design is not None else None,
                    json.dumps(readout) if readout is not None else None,
                    json.dumps(priority) if priority is not None else None,
                    json.dumps(history),
                    now,
                    now,
                ),
            )
        else:
            row = c.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown experiment id '{experiment_id}'")
            current = row["status"]
            status_changed = status is not None and status != current
            hops: list[str] = []
            if status_changed:
                if status == "calibrated":
                    raise ValueError(
                        f"Illegal transition {current}->calibrated. Experiments "
                        "become calibrated via transition_experiment (fit "
                        "close-out), never via upsert."
                    )
                hops = _transition_path(current, status) or []
                if not hops:
                    allowed = ALLOWED_TRANSITIONS.get(current, set())
                    raise ValueError(
                        f"Illegal transition {current}->{status}. Allowed from "
                        f"'{current}': {', '.join(sorted(allowed)) or '(none)'}"
                    )
            if current == "calibrated" and not allow_calibrated_edit:
                keys = row.keys()
                changing = [
                    name
                    for name, new_val, old_val in (
                        ("value", value, row["value"]),
                        ("se", se, row["se"]),
                        ("estimand", estimand, row["estimand"]),
                        ("start_date", start_date, row["start_date"]),
                        ("end_date", end_date, row["end_date"]),
                        (
                            "readout",
                            readout,
                            _json_or_none(
                                row["readout_json"] if "readout_json" in keys else None
                            ),
                        ),
                        ("channel", channel, row["channel"]),
                        (
                            "subchannel",
                            subchannel,
                            row["subchannel"] if "subchannel" in keys else None,
                        ),
                    )
                    if new_val is not None and new_val != old_val
                ]
                if changing:
                    raise ValueError(
                        f"Illegal update: experiment '{experiment_id}' is "
                        f"calibrated — this would change "
                        f"{', '.join(changing)}, which feed the model's "
                        "calibration likelihood. Readout edits require the "
                        "record_experiment_readout tool "
                        "(overwrite_calibrated=True)."
                    )
            updates = {
                k: v
                for k, v in {
                    "project_id": project_id,
                    "thread_id": thread_id,
                    "channel": channel,
                    "subchannel": subchannel,
                    "design_type": design_type,
                    "status": status,
                    "start_date": start_date,
                    "end_date": end_date,
                    "estimand": estimand,
                    "value": value,
                    "se": se,
                    "notes": notes,
                    "recommending_run_id": recommending_run_id,
                    "calibrated_run_id": calibrated_run_id,
                    "design_json": json.dumps(design) if design is not None else None,
                    "readout_json": (
                        json.dumps(readout) if readout is not None else None
                    ),
                    "priority_json": (
                        json.dumps(priority) if priority is not None else None
                    ),
                }.items()
                if v is not None
            }
            if status_changed:
                history = _json_or_none(
                    row["status_history_json"]
                    if "status_history_json" in row.keys()
                    else None
                ) or [{"status": current, "at": row["created_at"]}]
                for s in hops[:-1]:
                    history.append(
                        {"status": s, "at": now, "note": "backfilled via upsert"}
                    )
                history.append({"status": status, "at": now, "note": "via upsert"})
                updates["status_history_json"] = json.dumps(history)
            updates["updated_at"] = now
            sets = ", ".join(f"{k} = ?" for k in updates)
            c.execute(
                f"UPDATE experiments SET {sets} WHERE id = ?",
                (*updates.values(), experiment_id),
            )
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return _experiment_row_to_dict(row)


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return _experiment_row_to_dict(row) if row else None


def transition_experiment(
    experiment_id: str,
    new_status: str,
    *,
    note: str | None = None,
    value: float | None = None,
    se: float | None = None,
    estimand: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    readout: dict[str, Any] | None = None,
    calibrated_run_id: str | None = None,
) -> dict[str, Any]:
    """Validated lifecycle move with an append-only audit trail.

    Enforces :data:`ALLOWED_TRANSITIONS` (ValueError on an illegal move);
    stamps ``preregistered_at`` on draft→planned; merges the lift readout
    fields on →completed; records ``calibrated_run_id`` on →calibrated.
    """
    if new_status not in EXPERIMENT_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Valid: {', '.join(EXPERIMENT_STATUSES)}"
        )
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown experiment id '{experiment_id}'")
        current = row["status"]
        if new_status not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"Illegal transition {current}->{new_status}. Allowed from "
                f"'{current}': {', '.join(sorted(ALLOWED_TRANSITIONS.get(current, set()))) or '(none)'}"
            )
        history = _json_or_none(
            row["status_history_json"] if "status_history_json" in row.keys() else None
        ) or [{"status": current, "at": row["created_at"]}]
        entry: dict[str, Any] = {"status": new_status, "at": now}
        if note:
            entry["note"] = note
        history.append(entry)

        updates: dict[str, Any] = {
            "status": new_status,
            "status_history_json": json.dumps(history),
            "updated_at": now,
        }
        if current == "draft" and new_status == "planned":
            updates["preregistered_at"] = now
        if new_status == "completed":
            for k, v in (
                ("value", value),
                ("se", se),
                ("estimand", estimand),
                ("start_date", start_date),
                ("end_date", end_date),
            ):
                if v is not None:
                    updates[k] = v
            if readout is not None:
                updates["readout_json"] = json.dumps(readout)
        if new_status == "calibrated" and calibrated_run_id is not None:
            updates["calibrated_run_id"] = calibrated_run_id

        sets = ", ".join(f"{k} = ?" for k in updates)
        c.execute(
            f"UPDATE experiments SET {sets} WHERE id = ?",
            (*updates.values(), experiment_id),
        )
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return _experiment_row_to_dict(row)


def append_experiment_event(
    experiment_id: str, note: str, changed: dict[str, Any] | None = None
) -> None:
    """Append a non-transition audit event to an experiment's history.

    Records an entry ``{status: <current status>, at, note[, changed]}`` in
    ``status_history_json`` without moving the state machine — used to leave a
    trail when a readout is edited in place (e.g. re-recording a value or
    attaching off-panel spend on an already-measured experiment). NULL-tolerant
    like :func:`transition_experiment` (pre-lifecycle rows get their history
    synthesized from the current status). Bumps ``updated_at``.

    Args:
        experiment_id: Registry id of the experiment.
        note: Human-readable description of the event.
        changed: Optional ``{field: [old, new]}`` diff of what changed.

    Raises:
        ValueError: For an unknown experiment id.
    """
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown experiment id '{experiment_id}'")
        current = row["status"]
        history = _json_or_none(
            row["status_history_json"] if "status_history_json" in row.keys() else None
        ) or [{"status": current, "at": row["created_at"]}]
        entry: dict[str, Any] = {"status": current, "at": now, "note": note}
        if changed:
            entry["changed"] = changed
        history.append(entry)
        c.execute(
            "UPDATE experiments SET status_history_json = ?, updated_at = ?"
            " WHERE id = ?",
            (json.dumps(history), now, experiment_id),
        )


def latest_calibrated_evidence(project_id: str | None = None) -> dict[str, dict]:
    """Newest calibrated experiment per channel — the evidence map that feeds
    information decay and the calibration-coverage view.

    Returns ``{channel: {experiment_id, end_date, calibrated_run_id, se,
    estimand, updated_at}}``. Recency is judged by end_date when present,
    falling back to updated_at.
    """
    rows = list_experiments(project_id=project_id, status="calibrated")
    evidence: dict[str, dict] = {}
    for r in rows:
        key = (r.get("end_date") or "", r.get("updated_at") or 0)
        prev = evidence.get(r["channel"])
        if prev is None or key > (
            prev.get("end_date") or "",
            prev.get("updated_at") or 0,
        ):
            evidence[r["channel"]] = {
                "experiment_id": r["id"],
                "end_date": r.get("end_date"),
                "calibrated_run_id": r.get("calibrated_run_id"),
                "se": r.get("se"),
                "estimand": r.get("estimand"),
                "updated_at": r.get("updated_at"),
            }
    return evidence


def list_experiments(
    project_id: str | None = None,
    status: str | None = None,
    channel: str | None = None,
    subchannel: str | None = None,
) -> list[dict[str, Any]]:
    """Experiments, newest-updated first; optionally filtered."""
    q = "SELECT * FROM experiments"
    clauses, params = [], []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if channel is not None:
        clauses.append("channel = ?")
        params.append(channel)
    if subchannel is not None:
        clauses.append("subchannel = ?")
        params.append(subchannel)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY updated_at DESC"
    with _conn() as c:
        rows = c.execute(q, params).fetchall()
    return [_experiment_row_to_dict(r) for r in rows]


def delete_experiment(experiment_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        return cur.rowcount > 0


# ── Continuous-learning programs (model-free geo bandit) ─────────────────────

LEARNING_PROGRAM_STATUSES = ("active", "stopped", "archived")
LEARNING_WAVE_STATUSES = ("designed", "ingested")


def _learning_program_row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    keys = set(r.keys())
    return {
        "id": r["id"],
        "project_id": r["project_id"],
        "thread_id": r["thread_id"],
        "name": r["name"],
        "status": r["status"],
        "channels": _json_or_none(r["channels_json"]) or [],
        "config": _json_or_none(r["config_json"]) or {},
        "state_path": r["state_path"] if "state_path" in keys else None,
        "summary": _json_or_none(r["summary_json"] if "summary_json" in keys else None),
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def create_learning_program(
    *,
    project_id: str | None = None,
    thread_id: str | None = None,
    name: str | None = None,
    channels: list[str],
    config: dict[str, Any],
    state_path: str | None = None,
    status: str = "active",
) -> dict[str, Any]:
    """Insert a learning-program row. ``channels`` is the FLATTENED arm list
    (the surface dimensions); ``config`` is the dollars-at-the-boundary program
    config (wiring contract §3.1)."""
    if status not in LEARNING_PROGRAM_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Valid: {', '.join(LEARNING_PROGRAM_STATUSES)}"
        )
    if not channels:
        raise ValueError("channels is required to create a learning program")
    program_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO learning_programs (id, project_id, thread_id, name, status,"
            " channels_json, config_json, state_path, summary_json, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                program_id,
                project_id,
                thread_id,
                name,
                status,
                json.dumps(list(channels)),
                json.dumps(config, default=str),
                state_path,
                None,
                now,
                now,
            ),
        )
        row = c.execute(
            "SELECT * FROM learning_programs WHERE id = ?", (program_id,)
        ).fetchone()
    return _learning_program_row_to_dict(row)


def get_learning_program(program_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learning_programs WHERE id = ?", (program_id,)
        ).fetchone()
    return _learning_program_row_to_dict(row) if row else None


def list_learning_programs(
    project_id: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    """Learning programs, newest-updated first; optionally filtered."""
    q = "SELECT * FROM learning_programs"
    clauses, params = [], []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY updated_at DESC"
    with _conn() as c:
        rows = c.execute(q, params).fetchall()
    return [_learning_program_row_to_dict(r) for r in rows]


def update_learning_program(
    program_id: str,
    *,
    name: str | None = None,
    status: str | None = None,
    thread_id: str | None = None,
    channels: list[str] | None = None,
    config: dict[str, Any] | None = None,
    state_path: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Partial update; only the non-None fields change. Raises ValueError for
    an unknown id or an invalid status."""
    if status is not None and status not in LEARNING_PROGRAM_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Valid: {', '.join(LEARNING_PROGRAM_STATUSES)}"
        )
    updates = {
        k: v
        for k, v in {
            "name": name,
            "status": status,
            "thread_id": thread_id,
            "channels_json": (
                json.dumps(list(channels)) if channels is not None else None
            ),
            "config_json": (
                json.dumps(config, default=str) if config is not None else None
            ),
            "state_path": state_path,
            "summary_json": (
                json.dumps(summary, default=str) if summary is not None else None
            ),
        }.items()
        if v is not None
    }
    updates["updated_at"] = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM learning_programs WHERE id = ?", (program_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown learning program id '{program_id}'")
        sets = ", ".join(f"{k} = ?" for k in updates)
        c.execute(
            f"UPDATE learning_programs SET {sets} WHERE id = ?",
            (*updates.values(), program_id),
        )
        row = c.execute(
            "SELECT * FROM learning_programs WHERE id = ?", (program_id,)
        ).fetchone()
    return _learning_program_row_to_dict(row)


def delete_learning_program(program_id: str) -> bool:
    """Delete a program and cascade its waves. Returns True if the program
    existed. (The on-disk state.npz is left for the caller to reap.)"""
    with _conn() as c:
        c.execute("DELETE FROM learning_waves WHERE program_id = ?", (program_id,))
        cur = c.execute("DELETE FROM learning_programs WHERE id = ?", (program_id,))
        return cur.rowcount > 0


def _learning_wave_row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "program_id": r["program_id"],
        "project_id": r["project_id"],
        "wave_index": r["wave_index"],
        "status": r["status"],
        "source": r["source"],
        "design": _json_or_none(r["design_json"]),
        "observations": _json_or_none(r["observations_json"]),
        "snapshot": _json_or_none(r["snapshot_json"]),
        "experiment_ids": _json_or_none(r["experiment_ids_json"]),
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def add_learning_wave(
    program_id: str,
    *,
    project_id: str | None = None,
    wave_index: int | None = None,
    status: str = "designed",
    source: str | None = None,
    design: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Append a wave row (``wave_index`` auto-increments per program).

    ``source`` is one of ``wave`` / ``experiment_import`` / ``manual``;
    ``snapshot`` is the pinned fit_and_plan SNAPSHOT (immutable, like
    run_metrics rows)."""
    if status not in LEARNING_WAVE_STATUSES:
        raise ValueError(
            f"Invalid wave status '{status}'. Valid: {', '.join(LEARNING_WAVE_STATUSES)}"
        )
    wave_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        prog = c.execute(
            "SELECT 1 FROM learning_programs WHERE id = ?", (program_id,)
        ).fetchone()
        if prog is None:
            raise ValueError(f"Unknown learning program id '{program_id}'")
        if wave_index is None:
            row = c.execute(
                "SELECT MAX(wave_index) AS mx FROM learning_waves WHERE program_id = ?",
                (program_id,),
            ).fetchone()
            wave_index = 0 if row is None or row["mx"] is None else int(row["mx"]) + 1
        c.execute(
            "INSERT INTO learning_waves (id, program_id, project_id, wave_index,"
            " status, source, design_json, observations_json, snapshot_json,"
            " experiment_ids_json, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                wave_id,
                program_id,
                project_id,
                int(wave_index),
                status,
                source,
                json.dumps(design, default=str) if design is not None else None,
                (
                    json.dumps(observations, default=str)
                    if observations is not None
                    else None
                ),
                json.dumps(snapshot, default=str) if snapshot is not None else None,
                (
                    json.dumps(list(experiment_ids))
                    if experiment_ids is not None
                    else None
                ),
                now,
                now,
            ),
        )
        row = c.execute(
            "SELECT * FROM learning_waves WHERE id = ?", (wave_id,)
        ).fetchone()
    return _learning_wave_row_to_dict(row)


def get_learning_wave(wave_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM learning_waves WHERE id = ?", (wave_id,)
        ).fetchone()
    return _learning_wave_row_to_dict(row) if row else None


def list_learning_waves(program_id: str) -> list[dict[str, Any]]:
    """A program's waves, oldest first (wave_index ascending)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM learning_waves WHERE program_id = ?"
            " ORDER BY wave_index ASC, created_at ASC",
            (program_id,),
        ).fetchall()
    return [_learning_wave_row_to_dict(r) for r in rows]


def update_learning_wave(
    wave_id: str,
    *,
    status: str | None = None,
    source: str | None = None,
    design: dict[str, Any] | None = None,
    observations: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Partial update of a wave row (e.g. designed → ingested with results)."""
    if status is not None and status not in LEARNING_WAVE_STATUSES:
        raise ValueError(
            f"Invalid wave status '{status}'. Valid: {', '.join(LEARNING_WAVE_STATUSES)}"
        )
    updates = {
        k: v
        for k, v in {
            "status": status,
            "source": source,
            "design_json": (
                json.dumps(design, default=str) if design is not None else None
            ),
            "observations_json": (
                json.dumps(observations, default=str)
                if observations is not None
                else None
            ),
            "snapshot_json": (
                json.dumps(snapshot, default=str) if snapshot is not None else None
            ),
            "experiment_ids_json": (
                json.dumps(list(experiment_ids)) if experiment_ids is not None else None
            ),
        }.items()
        if v is not None
    }
    updates["updated_at"] = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM learning_waves WHERE id = ?", (wave_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown learning wave id '{wave_id}'")
        sets = ", ".join(f"{k} = ?" for k in updates)
        c.execute(
            f"UPDATE learning_waves SET {sets} WHERE id = ?",
            (*updates.values(), wave_id),
        )
        row = c.execute(
            "SELECT * FROM learning_waves WHERE id = ?", (wave_id,)
        ).fetchone()
    return _learning_wave_row_to_dict(row)


def record_ingested_wave(
    program_id: str,
    *,
    project_id: str | None = None,
    source: str | None = None,
    observations: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Record ingested evidence on the wave board (designed → ingested).

    A rows-ingest (``source='wave'``) RESOLVES the program's latest wave row
    when that row is still ``'designed'`` — so the timeline shows ONE row per
    real wave instead of a permanently-open 'designed' card plus a duplicate
    'ingested' one. Experiment imports (and ingests with no open design)
    append a new ``'ingested'`` row. Shared by the REST fit worker and the
    agent tools' ``record_learning_wave``/``import_past_experiments`` paths.
    """
    if source == "wave":
        waves = list_learning_waves(program_id)
        if waves and waves[-1].get("status") == "designed":
            return update_learning_wave(
                waves[-1]["id"],
                status="ingested",
                observations=observations,
                snapshot=snapshot,
                experiment_ids=experiment_ids,
            )
    return add_learning_wave(
        program_id,
        project_id=project_id,
        status="ingested",
        source=source,
        observations=observations,
        snapshot=snapshot,
        experiment_ids=experiment_ids,
    )


# ── Budget plans (Planner) ────────────────────────────────────────────────────

_BUDGET_PLAN_JSON_COLS = ("spend_changes", "channel_details", "plan_payload")


def _budget_plan_row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for col in _BUDGET_PLAN_JSON_COLS:
        raw = d.get(col)
        d[col] = json.loads(raw) if raw else None
    # FE expects ``plan_id`` and an ISO ``created_at`` (BudgetPlanInfo).
    d["plan_id"] = d["id"]
    return d


def upsert_budget_plan(
    *,
    plan_id: str | None = None,
    project_id: str | None = None,
    org_id: str,
    name: str,
    description: str | None = None,
    model_id: str | None = None,
    kind: str = "optimization",
    spend_changes: dict[str, Any] | None = None,
    baseline_outcome: float | None = None,
    scenario_outcome: float | None = None,
    outcome_change: float | None = None,
    outcome_change_pct: float | None = None,
    channel_details: dict[str, Any] | None = None,
    plan_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update a budget plan; returns the stored plan dict."""
    now = _now()
    with _conn() as c:
        existing = None
        if plan_id:
            existing = c.execute(
                "SELECT created_at FROM budget_plans WHERE id = ?", (plan_id,)
            ).fetchone()
        pid = plan_id or f"plan_{uuid.uuid4().hex[:12]}"
        created_at = existing["created_at"] if existing else now
        c.execute(
            """
            INSERT INTO budget_plans (
                id, project_id, org_id, name, description, model_id, kind,
                spend_changes, baseline_outcome, scenario_outcome, outcome_change,
                outcome_change_pct, channel_details, plan_payload,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                project_id=excluded.project_id, org_id=excluded.org_id,
                name=excluded.name, description=excluded.description,
                model_id=excluded.model_id, kind=excluded.kind,
                spend_changes=excluded.spend_changes,
                baseline_outcome=excluded.baseline_outcome,
                scenario_outcome=excluded.scenario_outcome,
                outcome_change=excluded.outcome_change,
                outcome_change_pct=excluded.outcome_change_pct,
                channel_details=excluded.channel_details,
                plan_payload=excluded.plan_payload,
                updated_at=excluded.updated_at
            """,
            (
                pid,
                project_id,
                org_id,
                name,
                description,
                model_id,
                kind,
                json.dumps(spend_changes, default=str) if spend_changes else None,
                baseline_outcome,
                scenario_outcome,
                outcome_change,
                outcome_change_pct,
                json.dumps(channel_details, default=str) if channel_details else None,
                json.dumps(plan_payload, default=str) if plan_payload else None,
                created_at,
                now,
            ),
        )
    return get_budget_plan(pid)  # type: ignore[return-value]


def get_budget_plan(plan_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM budget_plans WHERE id = ?", (plan_id,)).fetchone()
    return _budget_plan_row_to_dict(r) if r else None


def list_budget_plans(
    org_id: str,
    project_id: str | None = None,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Plans for an org (required), newest-updated first; optionally filtered."""
    q = "SELECT * FROM budget_plans WHERE org_id = ?"
    params: list[Any] = [org_id]
    if project_id is not None:
        q += " AND project_id = ?"
        params.append(project_id)
    if model_id is not None:
        q += " AND model_id = ?"
        params.append(model_id)
    q += " ORDER BY updated_at DESC"
    with _conn() as c:
        rows = c.execute(q, params).fetchall()
    return [_budget_plan_row_to_dict(r) for r in rows]


def delete_budget_plan(plan_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM budget_plans WHERE id = ?", (plan_id,))
        return cur.rowcount > 0


def latest_budget_plan_for_project(project_id: str) -> dict[str, Any] | None:
    """The most recently updated saved budget plan for a project (project-scoped,
    org-agnostic) — the pacing loop auto-sources its planned series from this
    (issue #123). ``None`` when the project has no saved plan."""
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM budget_plans WHERE project_id = ?"
            " ORDER BY updated_at DESC LIMIT 1",
            (project_id,),
        ).fetchone()
    return _budget_plan_row_to_dict(r) if r else None


# ── In-flight delivery registry (actual spend, issue #123) ───────────────────


def _delivery_row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "project_id": r["project_id"],
        "channel": r["channel"],
        "period": r["period"],
        "spend": r["spend"],
        "source": r["source"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def upsert_delivery(
    project_id: str,
    records: list[dict[str, Any]],
    *,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Upsert actual-delivery rows for a project (issue #123).

    Each record is ``{channel, spend, period?}``; a row is keyed by
    ``(project_id, channel, period)`` so re-uploading a period overwrites it
    (the elapsed window can be re-stated as more actuals land). Records with a
    non-numeric or missing spend/channel are skipped. Returns all of the
    project's delivery rows after the write.
    """
    now = _now()
    with _conn() as c:
        for rec in records or []:
            ch = rec.get("channel")
            if ch is None:
                continue
            try:
                spend = float(rec.get("spend"))
            except (TypeError, ValueError):
                continue
            period = str(rec.get("period", "") or "")
            c.execute(
                """
                INSERT INTO delivery
                    (id, project_id, channel, period, spend, source,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, channel, period) DO UPDATE SET
                    spend = excluded.spend,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    uuid.uuid4().hex,
                    project_id,
                    str(ch),
                    period,
                    spend,
                    source,
                    now,
                    now,
                ),
            )
    return list_delivery(project_id)


def list_delivery(project_id: str) -> list[dict[str, Any]]:
    """A project's actual-delivery rows, oldest-first within a channel so a
    period series reads in order (period is a free-form label — callers align by
    it)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM delivery WHERE project_id = ? ORDER BY channel, period",
            (project_id,),
        ).fetchall()
    return [_delivery_row_to_dict(r) for r in rows]


def delete_delivery(
    project_id: str, *, channel: str | None = None, period: str | None = None
) -> int:
    """Delete delivery rows for a project (optionally one channel / period).
    Returns the number of rows removed."""
    q = "DELETE FROM delivery WHERE project_id = ?"
    params: list[Any] = [project_id]
    if channel is not None:
        q += " AND channel = ?"
        params.append(channel)
    if period is not None:
        q += " AND period = ?"
        params.append(period)
    with _conn() as c:
        cur = c.execute(q, params)
        return cur.rowcount


# ── Model Garden registry ────────────────────────────────────────────────────

#: Lifecycle states for a registered garden model.
GARDEN_STATUSES = ("draft", "tested", "published", "deprecated")

#: Allowed status transitions (mirrors ALLOWED_TRANSITIONS for experiments).
#: ``draft -> tested`` is gated on a passing compatibility report; the human
#: publish gate is ``tested -> published``. Published versions are immutable.
GARDEN_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"tested", "deprecated"},
    "tested": {"published", "draft", "deprecated"},
    "published": {"deprecated"},
    "deprecated": {"draft"},
}


#: Org id used in the single-tenant dev posture (no auth) — MUST match the
#: ``_DEV_PRINCIPAL.org_id`` in api/main.py and ``AuthSettings.dev_org_id`` so the
#: agent (resolves org from the project) and the REST layer (resolves org from
#: the principal) agree on where garden models live.
DEFAULT_ORG_ID = "dev-org"


def resolve_org_id(project_id: str | None) -> str:
    """Owning org for a project — the garden registry's sharing boundary.

    Falls back to :data:`DEFAULT_ORG_ID` in the single-tenant dev posture (or
    when the auth schema's ``projects.org_id`` column isn't present yet)."""
    if not project_id:
        return DEFAULT_ORG_ID
    try:
        with _conn() as c:
            r = c.execute(
                "SELECT org_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        org = (r["org_id"] if r and "org_id" in r.keys() else None) if r else None
        return org or DEFAULT_ORG_ID
    except Exception:  # noqa: BLE001 — org_id column may be absent pre-auth-init
        return DEFAULT_ORG_ID


def _garden_row_to_dict(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    keys = set(r.keys())
    return {
        "id": r["id"],
        "org_id": r["org_id"],
        "name": r["name"],
        "version": r["version"],
        "owner_user_id": r["owner_user_id"] if "owner_user_id" in keys else None,
        "status": r["status"],
        "docs": r["docs"],
        "manifest": _json_or_none(r["manifest_json"]) or {},
        "source_path": r["source_path"],
        "compat_report": _json_or_none(r["compat_report_json"]),
        "base_run_id": r["base_run_id"] if "base_run_id" in keys else None,
        "reference_artifact_path": (
            r["reference_artifact_path"] if "reference_artifact_path" in keys else None
        ),
        "status_history": _json_or_none(r["status_history_json"]) or [],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def next_garden_version(org_id: str, name: str) -> int:
    """Next monotonic version for ``(org, name)`` — 1 when none exist yet."""
    with _conn() as c:
        r = c.execute(
            "SELECT MAX(version) AS v FROM garden_models WHERE org_id = ? AND name = ?",
            (org_id, name),
        ).fetchone()
    return int((r["v"] or 0)) + 1


def upsert_garden_model(
    *,
    org_id: str,
    name: str,
    version: int | None = None,
    model_id: str | None = None,
    owner_user_id: str | None = None,
    docs: str | None = None,
    manifest: dict[str, Any] | None = None,
    source_path: str | None = None,
    base_run_id: str | None = None,
    reference_artifact_path: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Create a new draft (no ``model_id``) or partially update an existing row.

    On create, ``version`` auto-increments per ``(org, name)`` when omitted and
    the row starts as ``draft``. PUBLISHED versions are immutable — editing a
    published row raises ``ValueError`` (re-publish requires a new version).
    """
    now = _now()
    with _conn() as c:
        if model_id is None:
            ver = (
                int(version)
                if version is not None
                else next_garden_version(org_id, name)
            )
            model_id = uuid.uuid4().hex
            init_status = status or "draft"
            if init_status not in GARDEN_STATUSES:
                raise ValueError(f"Invalid status '{init_status}'.")
            try:
                c.execute(
                    "INSERT INTO garden_models (id, org_id, name, version,"
                    " owner_user_id, status, docs, manifest_json, source_path,"
                    " base_run_id, reference_artifact_path, status_history_json,"
                    " created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model_id,
                        org_id,
                        name,
                        ver,
                        owner_user_id,
                        init_status,
                        docs,
                        json.dumps(manifest) if manifest is not None else None,
                        source_path,
                        base_run_id,
                        reference_artifact_path,
                        json.dumps([{"status": init_status, "at": now}]),
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"garden model '{name}' v{ver} already exists for this org"
                ) from exc
        else:
            row = c.execute(
                "SELECT * FROM garden_models WHERE id = ?", (model_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown garden model id '{model_id}'")
            if row["status"] == "published":
                raise ValueError(
                    "published garden versions are immutable — register a new "
                    "version instead of editing"
                )
            updates = {
                k: v
                for k, v in {
                    "owner_user_id": owner_user_id,
                    "docs": docs,
                    "manifest_json": (
                        json.dumps(manifest) if manifest is not None else None
                    ),
                    "source_path": source_path,
                    "base_run_id": base_run_id,
                    "reference_artifact_path": reference_artifact_path,
                }.items()
                if v is not None
            }
            updates["updated_at"] = now
            sets = ", ".join(f"{k} = ?" for k in updates)
            c.execute(
                f"UPDATE garden_models SET {sets} WHERE id = ?",
                (*updates.values(), model_id),
            )
        row = c.execute(
            "SELECT * FROM garden_models WHERE id = ?", (model_id,)
        ).fetchone()
    return _garden_row_to_dict(row)


def get_garden_model(
    *,
    model_id: str | None = None,
    org_id: str | None = None,
    name: str | None = None,
    version: int | None = None,
) -> dict[str, Any] | None:
    """Fetch by id, or by ``(org_id, name, version)``."""
    with _conn() as c:
        if model_id is not None:
            row = c.execute(
                "SELECT * FROM garden_models WHERE id = ?", (model_id,)
            ).fetchone()
        elif org_id is not None and name is not None and version is not None:
            row = c.execute(
                "SELECT * FROM garden_models WHERE org_id = ? AND name = ? AND version = ?",
                (org_id, name, int(version)),
            ).fetchone()
        else:
            raise ValueError(
                "get_garden_model needs model_id or (org_id, name, version)"
            )
    return _garden_row_to_dict(row)


def get_latest_garden_model(
    org_id: str, name: str, *, status: str | None = None
) -> dict[str, Any] | None:
    """Highest-version row for ``(org, name)``, optionally filtered by status
    (e.g. ``"published"`` to resolve what a consumer should load)."""
    q = "SELECT * FROM garden_models WHERE org_id = ? AND name = ?"
    params: list[Any] = [org_id, name]
    if status is not None:
        q += " AND status = ?"
        params.append(status)
    q += " ORDER BY version DESC LIMIT 1"
    with _conn() as c:
        row = c.execute(q, params).fetchone()
    return _garden_row_to_dict(row)


def list_garden_models(
    org_id: str,
    *,
    name: str | None = None,
    status: str | None = None,
    latest_only: bool = False,
) -> list[dict[str, Any]]:
    """Garden models for an org, newest-updated first; optional name/status
    filter. ``latest_only`` collapses to the highest version per name."""
    q = "SELECT * FROM garden_models WHERE org_id = ?"
    params: list[Any] = [org_id]
    if name is not None:
        q += " AND name = ?"
        params.append(name)
    if status is not None:
        q += " AND status = ?"
        params.append(status)
    q += " ORDER BY updated_at DESC"
    with _conn() as c:
        rows = [_garden_row_to_dict(r) for r in c.execute(q, params).fetchall()]
    if latest_only:
        best: dict[str, dict] = {}
        for r in rows:
            cur = best.get(r["name"])
            if cur is None or r["version"] > cur["version"]:
                best[r["name"]] = r
        rows = sorted(best.values(), key=lambda r: r["updated_at"], reverse=True)
    return rows


def list_garden_versions(org_id: str, name: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM garden_models WHERE org_id = ? AND name = ?"
            " ORDER BY version DESC",
            (org_id, name),
        ).fetchall()
    return [_garden_row_to_dict(r) for r in rows]


def set_garden_compat_report(
    model_id: str, report: dict[str, Any]
) -> dict[str, Any] | None:
    """Store the compatibility-suite report for a model (used to gate testing)."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE garden_models SET compat_report_json = ?, updated_at = ?"
            " WHERE id = ?",
            (json.dumps(report), _now(), model_id),
        )
        if cur.rowcount == 0:
            return None
        row = c.execute(
            "SELECT * FROM garden_models WHERE id = ?", (model_id,)
        ).fetchone()
    return _garden_row_to_dict(row)


def transition_garden_model(
    model_id: str,
    new_status: str,
    *,
    note: str | None = None,
    base_run_id: str | None = None,
) -> dict[str, Any]:
    """Validated lifecycle move with an append-only audit trail.

    Enforces :data:`GARDEN_TRANSITIONS`. ``draft -> tested`` additionally
    requires a stored compatibility report whose blocking tiers passed (the
    automated testing gate); ``tested -> published`` is the human publish gate.
    """
    if new_status not in GARDEN_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'.")
    now = _now()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM garden_models WHERE id = ?", (model_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown garden model id '{model_id}'")
        current = row["status"]
        if new_status not in GARDEN_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"Illegal transition {current}->{new_status}. Allowed from "
                f"'{current}': {', '.join(sorted(GARDEN_TRANSITIONS.get(current, set()))) or '(none)'}"
            )
        if current == "draft" and new_status == "tested":
            report = _json_or_none(row["compat_report_json"])
            if not (report and report.get("blocking_passed")):
                raise ValueError(
                    "cannot promote to 'tested': the compatibility suite has not "
                    "passed its blocking tiers (run test_garden_model first)"
                )
        history = _json_or_none(row["status_history_json"]) or [
            {"status": current, "at": row["created_at"]}
        ]
        entry: dict[str, Any] = {"status": new_status, "at": now}
        if note:
            entry["note"] = note
        history.append(entry)
        updates: dict[str, Any] = {
            "status": new_status,
            "status_history_json": json.dumps(history),
            "updated_at": now,
        }
        if base_run_id is not None:
            updates["base_run_id"] = base_run_id
        sets = ", ".join(f"{k} = ?" for k in updates)
        c.execute(
            f"UPDATE garden_models SET {sets} WHERE id = ?",
            (*updates.values(), model_id),
        )
        row = c.execute(
            "SELECT * FROM garden_models WHERE id = ?", (model_id,)
        ).fetchone()
    return _garden_row_to_dict(row)


def delete_garden_model(model_id: str) -> bool:
    """Delete a garden row. Only ``draft`` / ``deprecated`` rows are deletable —
    published history is immutable (deprecate it instead)."""
    with _conn() as c:
        row = c.execute(
            "SELECT status FROM garden_models WHERE id = ?", (model_id,)
        ).fetchone()
        if row is None:
            return False
        if row["status"] not in ("draft", "deprecated"):
            raise ValueError(
                f"cannot delete a '{row['status']}' model; deprecate it first"
            )
        cur = c.execute("DELETE FROM garden_models WHERE id = ?", (model_id,))
        return cur.rowcount > 0


# ── Saved data-source connections ────────────────────────────────────────────


def _data_connection_row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    try:
        d["config"] = json.loads(d.pop("config_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["config"] = {}
        d.pop("config_json", None)
    return d


def create_data_connection(
    project_id: str, name: str, kind: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Persist a reusable data-source connection (no credentials in config)."""
    cid = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        # Stamp the owning org (best-effort) for tenant-level auditability.
        org_id = None
        try:
            r = c.execute(
                "SELECT org_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            org_id = r["org_id"] if r else None
        except sqlite3.OperationalError:
            org_id = None
        c.execute(
            "INSERT INTO data_connections (id, project_id, org_id, name, kind,"
            " config_json, created_at, updated_at, last_synced)"
            " VALUES (?,?,?,?,?,?,?,?, NULL)",
            (cid, project_id, org_id, name, kind, json.dumps(config or {}), now, now),
        )
    return get_data_connection(cid)


def list_data_connections(project_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM data_connections WHERE project_id = ? ORDER BY updated_at DESC",
            (project_id,),
        ).fetchall()
    return [_data_connection_row(r) for r in rows]


def get_data_connection(connection_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM data_connections WHERE id = ?", (connection_id,)
        ).fetchone()
    return _data_connection_row(row)


def get_data_connection_by_name(project_id: str, name: str) -> dict[str, Any] | None:
    """Resolve a connection by name within a project (newest wins)."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM data_connections WHERE project_id = ? AND name = ?"
            " ORDER BY updated_at DESC LIMIT 1",
            (project_id, name),
        ).fetchone()
    return _data_connection_row(row)


def delete_data_connection(connection_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM data_connections WHERE id = ?", (connection_id,))
        return cur.rowcount > 0


def touch_data_connection_synced(connection_id: str) -> None:
    now = _now()
    with _conn() as c:
        c.execute(
            "UPDATE data_connections SET last_synced = ?, updated_at = ? WHERE id = ?",
            (now, now, connection_id),
        )


def set_data_connection_schedule(
    connection_id: str,
    interval_minutes: float | None,
    *,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Set (or clear, with ``None``) a connection's auto-sync interval.

    Setting an interval schedules the first run one interval out; clearing it
    (``None``) stops scheduled syncs (manual/on-demand still works).
    """
    ts = _now() if now is None else now
    next_at = (ts + float(interval_minutes) * 60.0) if interval_minutes else None
    with _conn() as c:
        c.execute(
            "UPDATE data_connections SET sync_interval_minutes = ?, next_sync_at = ?,"
            " updated_at = ? WHERE id = ?",
            (interval_minutes, next_at, ts, connection_id),
        )
    return get_data_connection(connection_id)


def list_due_data_connections(now: float, *, limit: int = 100) -> list[dict[str, Any]]:
    """Scheduled connections whose next_sync_at has arrived (across all projects)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM data_connections WHERE sync_interval_minutes IS NOT NULL"
            " AND next_sync_at IS NOT NULL AND next_sync_at <= ?"
            " ORDER BY next_sync_at ASC LIMIT ?",
            (now, limit),
        ).fetchall()
    return [_data_connection_row(r) for r in rows]


def record_data_connection_sync(
    connection_id: str,
    *,
    status: str,
    row_count: int | None = None,
    error: str | None = None,
    snapshot_path: str | None = None,
    now: float | None = None,
) -> None:
    """Record a sync outcome and advance next_sync_at.

    On ``error`` the next run is backed off (up to 4x the interval, capped so a
    sub-daily schedule waits at most a day) so a permanently broken connection
    isn't retried every interval forever — but never sooner than its own
    interval.
    """
    ts = _now() if now is None else now
    with _conn() as c:
        row = c.execute(
            "SELECT sync_interval_minutes FROM data_connections WHERE id = ?",
            (connection_id,),
        ).fetchone()
        interval = row["sync_interval_minutes"] if row else None
        if interval:
            secs = float(interval) * 60.0
            if status == "error":
                secs = max(secs, min(secs * 4.0, 86400.0))
            next_at = ts + secs
        else:
            next_at = None
        c.execute(
            "UPDATE data_connections SET last_synced = ?, last_sync_status = ?,"
            " last_sync_error = ?, last_row_count = ?, snapshot_path = COALESCE(?, snapshot_path),"
            " next_sync_at = ?, updated_at = ? WHERE id = ?",
            (ts, status, error, row_count, snapshot_path, next_at, ts, connection_id),
        )


# ── Run metrics (per-fit history snapshots) ──────────────────────────────────


def record_run_metrics(
    run_id: str,
    metrics: dict[str, Any],
    *,
    thread_id: str | None = None,
    project_id: str | None = None,
    artifact_id: str | None = None,
    created_at: float | None = None,
) -> None:
    """Upsert the metrics snapshot for a run. ``created_at`` defaults to now;
    backfill passes the original artifact timestamp so series stay ordered."""
    with _conn() as c:
        c.execute(
            "INSERT INTO run_metrics (run_id, artifact_id, thread_id, project_id,"
            " created_at, schema_version, metrics_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(run_id) DO UPDATE SET artifact_id = excluded.artifact_id,"
            " thread_id = excluded.thread_id, project_id = excluded.project_id,"
            " created_at = excluded.created_at,"
            " schema_version = excluded.schema_version,"
            " metrics_json = excluded.metrics_json",
            (
                run_id,
                artifact_id,
                thread_id,
                project_id,
                created_at if created_at is not None else _now(),
                int(metrics.get("schema_version", 1)),
                json.dumps(metrics),
            ),
        )


def _run_metrics_row_to_dict(r) -> dict[str, Any]:
    return {
        "run_id": r["run_id"],
        "artifact_id": r["artifact_id"],
        "thread_id": r["thread_id"],
        "project_id": r["project_id"],
        "created_at": r["created_at"],
        "schema_version": r["schema_version"],
        "metrics": _json_or_none(r["metrics_json"]) or {},
    }


def list_run_metrics(project_id: str | None = None) -> list[dict[str, Any]]:
    """Run metrics, OLDEST first (trajectory order)."""
    q = "SELECT * FROM run_metrics"
    params: list[Any] = []
    if project_id is not None:
        q += " WHERE project_id = ?"
        params.append(project_id)
    q += " ORDER BY created_at ASC"
    with _conn() as c:
        rows = c.execute(q, params).fetchall()
    return [_run_metrics_row_to_dict(r) for r in rows]


def get_run_metrics(run_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM run_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _run_metrics_row_to_dict(row) if row else None


def run_metrics_activity(since_ts: float) -> dict[str, Any]:
    """Deployment-wide fit activity: total fits, fits since ``since_ts``, and the
    most recent fit time — a lightweight reliability/SLA signal."""
    with _conn() as c:
        try:
            total = c.execute("SELECT COUNT(*) AS n FROM run_metrics").fetchone()["n"]
            recent = c.execute(
                "SELECT COUNT(*) AS n FROM run_metrics WHERE created_at >= ?",
                (since_ts,),
            ).fetchone()["n"]
            last = c.execute("SELECT MAX(created_at) AS m FROM run_metrics").fetchone()[
                "m"
            ]
        except sqlite3.OperationalError:
            return {"total": 0, "recent": 0, "last_at": None}
    return {"total": int(total or 0), "recent": int(recent or 0), "last_at": last}


# ── Projects ────────────────────────────────────────────────────────────────

_DEFAULT_PROJECT_ID = "default"


def ensure_default_project() -> str:
    """Create the built-in Default Project if absent; return its id."""
    with _conn() as c:
        row = c.execute(
            "SELECT project_id FROM projects WHERE project_id = ?",
            (_DEFAULT_PROJECT_ID,),
        ).fetchone()
        if row is None:
            now = _now()
            c.execute(
                "INSERT INTO projects (project_id, name, description, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    _DEFAULT_PROJECT_ID,
                    "Default Project",
                    "Auto-created home for sessions without an explicit project.",
                    now,
                    now,
                ),
            )
    return _DEFAULT_PROJECT_ID


def create_project(
    name: str, description: str | None = None, org_id: str | None = None
) -> dict[str, Any]:
    project_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO projects (project_id, name, description, org_id,"
                " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, name or "Untitled Project", description, org_id, now, now),
            )
        except sqlite3.OperationalError:
            # org_id column absent (auth schema not yet initialized) — fall back.
            c.execute(
                "INSERT INTO projects (project_id, name, description, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, name or "Untitled Project", description, now, now),
            )
    return get_project(project_id)  # type: ignore[return-value]


def _project_counts(c: sqlite3.Connection, project_id: str) -> dict[str, int]:
    session_count = c.execute(
        "SELECT COUNT(*) AS n FROM sessions WHERE project_id = ?", (project_id,)
    ).fetchone()["n"]
    doc_count = c.execute(
        "SELECT COUNT(*) AS n FROM kb_documents WHERE project_id = ?", (project_id,)
    ).fetchone()["n"]
    return {"session_count": session_count, "doc_count": doc_count}


def list_projects(org_id: str | None = None) -> list[dict[str, Any]]:
    """List projects. When ``org_id`` is given, scope to that tenant's projects.

    ``org_id=None`` preserves the original behavior (all projects + the built-in
    Default Project) used by single-tenant/dev mode.
    """
    with _conn() as c:
        if org_id is None:
            ensure_default_project()
            rows = c.execute(
                "SELECT project_id, name, description, meta_json, created_at,"
                " updated_at FROM projects"
                " ORDER BY (project_id = 'default') DESC, updated_at DESC"
            ).fetchall()
        else:
            try:
                rows = c.execute(
                    "SELECT project_id, name, description, meta_json, created_at,"
                    " updated_at FROM projects WHERE org_id = ?"
                    " ORDER BY updated_at DESC",
                    (org_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = _json_or_none(d.pop("meta_json", None)) or {}
            d.update(_project_counts(c, r["project_id"]))
            out.append(d)
        return out


def get_project(project_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT project_id, name, description, meta_json, created_at,"
            " updated_at FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["meta"] = _json_or_none(d.pop("meta_json", None)) or {}
        d.update(_project_counts(c, project_id))
        return d


def set_project_meta(project_id: str, meta: dict[str, Any]) -> dict[str, Any] | None:
    """Merge ``meta`` into the project's onboarding profile (None values
    delete keys). Returns the updated project, or None for an unknown id."""
    current = get_project(project_id)
    if current is None:
        return None
    merged = dict(current.get("meta") or {})
    for k, v in meta.items():
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v
    with _conn() as c:
        c.execute(
            "UPDATE projects SET meta_json = ?, updated_at = ? WHERE project_id = ?",
            (json.dumps(merged), _now(), project_id),
        )
    return get_project(project_id)


def update_project(
    project_id: str, name: str | None = None, description: str | None = None
) -> bool:
    sets, params = [], []
    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if not sets:
        return False
    sets.append("updated_at = ?")
    params.append(_now())
    params.append(project_id)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE project_id = ?", params
        )
        return cur.rowcount > 0


def delete_project(project_id: str) -> bool:
    """Delete a project. Its sessions become unassigned (project_id NULL) and
    its KB documents/chunks are removed. The built-in Default Project cannot be
    deleted."""
    if project_id == _DEFAULT_PROJECT_ID:
        return False
    with _conn() as c:
        cur = c.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        if cur.rowcount == 0:
            return False
        c.execute(
            "UPDATE sessions SET project_id = NULL WHERE project_id = ?", (project_id,)
        )
        c.execute("DELETE FROM kb_chunks WHERE project_id = ?", (project_id,))
        c.execute("DELETE FROM kb_documents WHERE project_id = ?", (project_id,))
        return True


def resolve_project_id(thread_id: str | None) -> str:
    """The project a session belongs to, falling back to the Default Project.

    Never returns None — the knowledge base always has a home.
    """
    ensure_default_project()
    if not thread_id:
        return _DEFAULT_PROJECT_ID
    with _conn() as c:
        row = c.execute(
            "SELECT project_id FROM sessions WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    if row and row["project_id"]:
        return row["project_id"]
    return _DEFAULT_PROJECT_ID


# ── Preferences (global defaults + per-project branding) ────────────────────


def set_preference(scope: str, key: str, value: Any) -> dict[str, Any]:
    """Upsert one preference value (stored as JSON) under (scope, key)."""
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO preferences (scope, key, value_json, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(scope, key) DO UPDATE SET value_json = excluded.value_json,"
            " updated_at = excluded.updated_at",
            (scope, key, json.dumps(value), now),
        )
    return {"scope": scope, "key": key, "value": value, "updated_at": now}


def get_preference(scope: str, key: str) -> Any | None:
    with _conn() as c:
        row = c.execute(
            "SELECT value_json FROM preferences WHERE scope = ? AND key = ?",
            (scope, key),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["value_json"])
    except Exception:
        return None


def list_preferences(scope: str) -> dict[str, Any]:
    with _conn() as c:
        rows = c.execute(
            "SELECT key, value_json FROM preferences WHERE scope = ?", (scope,)
        ).fetchall()
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value_json"])
        except Exception:
            continue
    return out


def delete_preference(scope: str, key: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM preferences WHERE scope = ? AND key = ?", (scope, key)
        )
        return cur.rowcount > 0


# ── Users (team roster) ───────────────────────────────────────────────────────

USER_ROLES = ("owner", "analyst", "viewer")


def _user_row_to_dict(r) -> dict[str, Any]:
    return {
        "user_id": r["user_id"],
        "name": r["name"],
        "email": r["email"],
        "role": r["role"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
    }


def create_user(
    name: str, email: str | None = None, role: str = "analyst"
) -> dict[str, Any]:
    if role not in USER_ROLES:
        raise ValueError(f"Invalid role '{role}'. Valid: {', '.join(USER_ROLES)}")
    if not name.strip():
        raise ValueError("name is required")
    user_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        try:
            c.execute(
                "INSERT INTO users (user_id, name, email, role, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name.strip(), (email or "").strip() or None, role, now, now),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"A user with email '{email}' already exists")
        row = c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return _user_row_to_dict(row)


def list_users() -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM users ORDER BY name").fetchall()
    return [_user_row_to_dict(r) for r in rows]


def update_user(
    user_id: str,
    *,
    name: str | None = None,
    email: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    if role is not None and role not in USER_ROLES:
        raise ValueError(f"Invalid role '{role}'. Valid: {', '.join(USER_ROLES)}")
    updates = {
        k: v
        for k, v in {"name": name, "email": email, "role": role}.items()
        if v is not None
    }
    if not updates:
        raise ValueError("nothing to update")
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k} = ?" for k in updates)
    with _conn() as c:
        try:
            cur = c.execute(
                f"UPDATE users SET {sets} WHERE user_id = ?",
                (*updates.values(), user_id),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"A user with email '{email}' already exists")
        if cur.rowcount == 0:
            raise ValueError(f"Unknown user id '{user_id}'")
        row = c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return _user_row_to_dict(row)


def delete_user(user_id: str) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM project_members WHERE user_id = ?", (user_id,))
        cur = c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        return cur.rowcount > 0


def set_project_members(
    project_id: str, members: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Replace the project's member list: ``[{user_id, role?}]``."""
    with _conn() as c:
        known = {
            r["user_id"] for r in c.execute("SELECT user_id FROM users").fetchall()
        }
        unknown = [m["user_id"] for m in members if m["user_id"] not in known]
        if unknown:
            raise ValueError(f"Unknown user id(s): {', '.join(unknown)}")
        c.execute("DELETE FROM project_members WHERE project_id = ?", (project_id,))
        for m in members:
            role = m.get("role", "analyst")
            if role not in USER_ROLES:
                raise ValueError(f"Invalid role '{role}'")
            c.execute(
                "INSERT INTO project_members (project_id, user_id, role)"
                " VALUES (?, ?, ?)",
                (project_id, m["user_id"], role),
            )
    return list_project_members(project_id)


def list_project_members(project_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT u.user_id, u.name, u.email, m.role FROM project_members m"
            " JOIN users u ON u.user_id = m.user_id WHERE m.project_id = ?"
            " ORDER BY u.name",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_project_branding(project_id: str) -> dict[str, Any] | None:
    val = get_preference(project_id, "branding")
    return val if isinstance(val, dict) else None


def set_project_branding(project_id: str, branding: dict[str, Any]) -> dict[str, Any]:
    return set_preference(project_id, "branding", branding)


# ── Knowledge-base documents + chunks ───────────────────────────────────────


def add_kb_document(
    project_id: str,
    name: str,
    path: str,
    kind: str,
    size_bytes: int | None = None,
    status: str = "pending",
    meta: dict | None = None,
) -> dict[str, Any]:
    doc_id = uuid.uuid4().hex
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO kb_documents (id, project_id, name, path, kind, size_bytes,"
            " n_chunks, status, error, meta_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?, ?)",
            (
                doc_id,
                project_id,
                name,
                path,
                kind,
                size_bytes,
                status,
                json.dumps(meta or {}, default=str),
                now,
            ),
        )
    return get_kb_document(doc_id)  # type: ignore[return-value]


def set_kb_document_status(
    doc_id: str, status: str, n_chunks: int | None = None, error: str | None = None
) -> None:
    sets, params = ["status = ?"], [status]
    if n_chunks is not None:
        sets.append("n_chunks = ?")
        params.append(n_chunks)
    if error is not None:
        sets.append("error = ?")
        params.append(error)
    params.append(doc_id)
    with _conn() as c:
        c.execute(f"UPDATE kb_documents SET {', '.join(sets)} WHERE id = ?", params)


def _row_to_kb_doc(r: sqlite3.Row) -> dict[str, Any]:
    try:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
    except Exception:
        meta = {}
    return {
        "id": r["id"],
        "project_id": r["project_id"],
        "name": r["name"],
        "path": r["path"],
        "kind": r["kind"],
        "size_bytes": r["size_bytes"],
        "n_chunks": r["n_chunks"],
        "status": r["status"],
        "error": r["error"],
        "meta": meta,
        "created_at": r["created_at"],
    }


def list_kb_documents(project_id: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM kb_documents WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        return [_row_to_kb_doc(r) for r in rows]


def get_kb_document(doc_id: str) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM kb_documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_kb_doc(row) if row else None


def delete_kb_document(doc_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM kb_documents WHERE id = ?", (doc_id,))
        if cur.rowcount == 0:
            return False
        c.execute("DELETE FROM kb_chunks WHERE document_id = ?", (doc_id,))
        return True


def add_kb_chunks(
    document_id: str, project_id: str, chunks: list[tuple[int, str, bytes, int]]
) -> int:
    """Insert chunk rows. ``chunks`` is a list of (chunk_index, text, embedding_blob, dim)."""
    now = _now()
    rows = [
        (uuid.uuid4().hex, document_id, project_id, idx, text, emb, dim, now)
        for (idx, text, emb, dim) in chunks
    ]
    with _conn() as c:
        c.executemany(
            "INSERT INTO kb_chunks (id, document_id, project_id, chunk_index, text,"
            " embedding, dim, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def iter_kb_chunks(project_id: str) -> list[dict[str, Any]]:
    """All chunks for a project (id, document_id, chunk_index, text, embedding bytes, dim)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, document_id, chunk_index, text, embedding, dim FROM kb_chunks"
            " WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "document_id": r["document_id"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "embedding": r["embedding"],
                "dim": r["dim"],
            }
            for r in rows
        ]

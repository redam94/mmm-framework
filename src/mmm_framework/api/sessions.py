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
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).parent / "sessions.db"


def _now() -> float:
    return time.time()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id  TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id           TEXT PRIMARY KEY,
                thread_id    TEXT NOT NULL,
                kind         TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at   REAL NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_thread ON artifacts(thread_id, created_at)")

        # Modeling assumptions: a key/value log per session with full versioned
        # history. Each row is an immutable snapshot; the "current" value for a
        # key is the row with the highest version. Updates write a new row.
        c.execute(
            """
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
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_assumptions_thread_key ON assumptions(thread_id, key, version)")

        # Workflow status: which of the 9 canonical steps are done/in-progress.
        # We mostly infer status from state, but store manual overrides + notes
        # here. One row per (thread_id, step).
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_status (
                thread_id   TEXT NOT NULL,
                step        INTEGER NOT NULL,
                status      TEXT NOT NULL,
                notes       TEXT,
                updated_at  REAL NOT NULL,
                PRIMARY KEY (thread_id, step)
            )
            """
        )

        # Data files associated with a session: uploads, generated datasets,
        # exports, plots, etc. Distinct from artifacts (which are code/report
        # snippets); this table is the "Files" tab in the UI.
        c.execute(
            """
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
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_data_files_thread ON data_files(thread_id, created_at)")


def list_sessions() -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT thread_id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def create_session(name: str | None = None) -> dict[str, Any]:
    thread_id = uuid.uuid4().hex
    now = _now()
    display_name = name or f"Session {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}"
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (thread_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, display_name, now, now),
        )
    return {"thread_id": thread_id, "name": display_name, "created_at": now, "updated_at": now}


def touch_session(thread_id: str) -> None:
    """Bump updated_at; if the row doesn't exist (legacy thread), insert it."""
    now = _now()
    with _conn() as c:
        row = c.execute("SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)).fetchone()
        if row:
            c.execute("UPDATE sessions SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        else:
            c.execute(
                "INSERT INTO sessions (thread_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, f"Session {time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}", now, now),
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
            (artifact["id"], thread_id, kind, json.dumps(payload, default=str), artifact["created_at"]),
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


def delete_artifact(artifact_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        return cur.rowcount > 0


# ── Assumptions log ───────────────────────────────────────────────────────────

ASSUMPTION_CATEGORIES = {
    "research_question",  # what we're estimating
    "causal_structure",    # DAG-level claims (e.g. "TV does not confound display")
    "data",                # data assumptions (e.g. "weekly aggregation is appropriate")
    "functional_form",     # adstock/saturation choices
    "prior",               # specific prior choices and their justification
    "identification",      # backdoor/instrument/frontdoor claims
    "external_evidence",   # benchmarks, experiments, prior MMMs
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
            (aid, thread_id, key, category, json.dumps(value, default=str), rationale, change_note, next_version, now),
        )
    return {
        "id": aid, "thread_id": thread_id, "key": key, "category": category,
        "value": value, "rationale": rationale, "change_note": change_note,
        "version": next_version, "is_tombstone": False, "created_at": now,
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
    return {"id": aid, "key": key, "version": next_version, "is_tombstone": True, "created_at": now}


def _row_to_assumption(r: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(r["value_json"])
    except Exception:
        value = r["value_json"]
    return {
        "id": r["id"], "thread_id": r["thread_id"], "key": r["key"],
        "category": r["category"], "value": value, "rationale": r["rationale"],
        "change_note": r["change_note"], "version": r["version"],
        "is_tombstone": bool(r["is_tombstone"]), "created_at": r["created_at"],
    }


def list_assumptions(thread_id: str, include_history: bool = False) -> list[dict[str, Any]]:
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
    (1, "Define the Question",            "Pre-register the causal/business question."),
    (2, "Tell the Story of Your Data",    "Generative narrative + DAG."),
    (3, "Build the Model",                "Specify components and priors."),
    (4, "Prior Predictive Check",         "Simulate from priors; sanity-check implied outcomes."),
    (5, "Fit the Model",                  "Run MCMC."),
    (6, "Computational Diagnostics",      "R-hat, ESS, divergences."),
    (7, "Posterior Predictive Check",     "Compare fit to observed data."),
    (8, "Sensitivity Analysis",           "Stress-test conclusions."),
    (9, "Communicate Results",            "Honest uncertainty + decision."),
]


def set_workflow_step(thread_id: str, step: int, status: str, notes: str | None = None) -> dict[str, Any]:
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
    return {"thread_id": thread_id, "step": step, "status": status, "notes": notes, "updated_at": now}


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
            (fid, thread_id, path, name, kind, size_bytes, preview, json.dumps(meta or {}, default=str), now),
        )
    return {
        "id": fid, "thread_id": thread_id, "path": path, "name": name, "kind": kind,
        "size_bytes": size_bytes, "preview": preview, "meta": meta or {}, "created_at": now,
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
        out.append({
            "id": r["id"], "thread_id": r["thread_id"], "path": r["path"],
            "name": r["name"], "kind": r["kind"], "size_bytes": r["size_bytes"],
            "preview": r["preview"], "meta": meta, "created_at": r["created_at"],
        })
    return out


def delete_file(file_id: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM data_files WHERE id = ?", (file_id,))
        return cur.rowcount > 0

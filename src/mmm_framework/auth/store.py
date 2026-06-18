"""Org/user persistence — same SQLite DB the session store already uses.

This module is intentionally self-contained (stdlib ``sqlite3`` only): it points
at ``src/mmm_framework/api/sessions.db`` by default but accepts a ``db_path``
override so it can be unit-tested against a throwaway file without importing the
(heavier) agent app. It *augments* the existing ``users`` / ``projects`` tables
and adds ``organizations`` / ``org_members`` — it never drops anything.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# Default to the same DB file the session/measurement-loop store owns.
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "api" / "sessions.db"


def _now() -> float:
    return time.time()


def _id() -> str:
    return uuid.uuid4().hex


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "org"


@contextmanager
def _conn(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column(c: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    try:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    except sqlite3.OperationalError:
        pass  # already exists


def init_auth_schema(db_path: Path | str | None = None) -> None:
    """Create org tables and add auth columns. Idempotent."""
    with _conn(db_path) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS organizations (
                org_id     TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                slug       TEXT UNIQUE,
                plan       TEXT NOT NULL DEFAULT 'free',
                status     TEXT NOT NULL DEFAULT 'active',
                meta_json  TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS org_members (
                org_id     TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'analyst',
                created_at REAL NOT NULL,
                PRIMARY KEY (org_id, user_id)
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)"
        )

        # The roster tables predate auth; ensure they exist before we ALTER them
        # (the agent app's init_db also creates these — CREATE IF NOT EXISTS keeps
        # us safe whether or not it ran first).
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                email      TEXT UNIQUE,
                role       TEXT NOT NULL DEFAULT 'analyst',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT,
                meta_json   TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """
        )

        _add_column(c, "users", "password_hash", "TEXT")
        _add_column(c, "users", "org_id", "TEXT")
        _add_column(c, "users", "status", "TEXT NOT NULL DEFAULT 'active'")
        _add_column(c, "users", "last_login_at", "REAL")
        _add_column(c, "users", "token_version", "INTEGER NOT NULL DEFAULT 0")
        _add_column(c, "projects", "org_id", "TEXT")
        c.execute("CREATE INDEX IF NOT EXISTS idx_projects_org ON projects(org_id)")

        # Phase 1.4 lifecycle: refresh-token revocation blocklist (logout /
        # rotation / compromise), org invites, and password-reset tokens.
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                user_id    TEXT,
                expires_at REAL NOT NULL,
                revoked_at REAL NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS invites (
                token       TEXT PRIMARY KEY,
                org_id      TEXT NOT NULL,
                email       TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'analyst',
                invited_by  TEXT,
                expires_at  REAL NOT NULL,
                accepted_at REAL,
                created_at  REAL NOT NULL
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_invites_org ON invites(org_id, email)"
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS password_resets (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                expires_at REAL NOT NULL,
                used_at    REAL,
                created_at REAL NOT NULL
            )
            """
        )


# ----- organizations ----------------------------------------------------------


def create_organization(
    name: str, *, plan: str = "free", db_path: Path | str | None = None
) -> dict[str, Any]:
    org_id = _id()
    base = slugify(name)
    now = _now()
    with _conn(db_path) as c:
        # Ensure slug uniqueness by suffixing on collision.
        slug = base
        n = 1
        while c.execute("SELECT 1 FROM organizations WHERE slug=?", (slug,)).fetchone():
            n += 1
            slug = f"{base}-{n}"
        c.execute(
            "INSERT INTO organizations (org_id, name, slug, plan, status,"
            " meta_json, created_at, updated_at) VALUES (?,?,?,?, 'active', NULL,?,?)",
            (org_id, name, slug, plan, now, now),
        )
    return {"org_id": org_id, "name": name, "slug": slug, "plan": plan}


def get_organization(
    org_id: str, db_path: Path | str | None = None
) -> dict[str, Any] | None:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM organizations WHERE org_id=?", (org_id,)
        ).fetchone()
    return dict(row) if row else None


# ----- users ------------------------------------------------------------------


def create_user(
    *,
    email: str,
    password_hash: str,
    org_id: str,
    name: str | None = None,
    role: str = "analyst",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    user_id = _id()
    now = _now()
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO users (user_id, name, email, role, password_hash,"
            " org_id, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?, 'active', ?, ?)",
            (user_id, name or email, email, role, password_hash, org_id, now, now),
        )
    return {"user_id": user_id, "email": email, "org_id": org_id, "name": name}


def get_user_by_email(
    email: str, db_path: Path | str | None = None
) -> dict[str, Any] | None:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def get_user(user_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def set_password_hash(
    user_id: str, password_hash: str, db_path: Path | str | None = None
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "UPDATE users SET password_hash=?, updated_at=? WHERE user_id=?",
            (password_hash, _now(), user_id),
        )


def touch_last_login(user_id: str, db_path: Path | str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute("UPDATE users SET last_login_at=? WHERE user_id=?", (_now(), user_id))


# ----- membership -------------------------------------------------------------


def add_org_member(
    org_id: str, user_id: str, role: str = "analyst", db_path: Path | str | None = None
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO org_members (org_id, user_id, role, created_at)"
            " VALUES (?,?,?,?)",
            (org_id, user_id, role, _now()),
        )


def get_org_role(
    org_id: str, user_id: str, db_path: Path | str | None = None
) -> str | None:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT role FROM org_members WHERE org_id=? AND user_id=?",
            (org_id, user_id),
        ).fetchone()
    return row["role"] if row else None


def list_org_members(
    org_id: str, db_path: Path | str | None = None
) -> list[dict[str, Any]]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT m.user_id, m.role, u.email, u.name FROM org_members m"
            " LEFT JOIN users u ON u.user_id = m.user_id WHERE m.org_id=?",
            (org_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ----- project ↔ org ----------------------------------------------------------


def attach_project_to_org(
    project_id: str, org_id: str, db_path: Path | str | None = None
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "UPDATE projects SET org_id=?, updated_at=? WHERE project_id=?",
            (org_id, _now(), project_id),
        )


def list_org_project_ids(org_id: str, db_path: Path | str | None = None) -> set[str]:
    """Return the set of project ids owned by ``org_id`` (empty if none/no column)."""
    with _conn(db_path) as c:
        try:
            rows = c.execute(
                "SELECT project_id FROM projects WHERE org_id=?", (org_id,)
            ).fetchall()
        except sqlite3.OperationalError:
            return set()
    return {r["project_id"] for r in rows}


def get_project_org(project_id: str, db_path: Path | str | None = None) -> str | None:
    """Return a project's org_id, or ``None`` if the project/column is absent."""
    with _conn(db_path) as c:
        try:
            row = c.execute(
                "SELECT org_id FROM projects WHERE project_id=?", (project_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return row["org_id"] if row and row["org_id"] else None


# ----- default org + one-time backfill ---------------------------------------

# Stable id for the org that owns pre-existing data when a single-tenant install
# turns auth on. Deterministic so repeated startups converge on one org.
DEFAULT_ORG_ID = "org_default"

_KNOWN_ROLES = {"viewer", "analyst", "admin", "owner"}


def ensure_default_organization(
    name: str = "Default Organization", db_path: Path | str | None = None
) -> str:
    """Create (idempotently) the canonical default org; return its id."""
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT org_id FROM organizations WHERE org_id=?", (DEFAULT_ORG_ID,)
        ).fetchone()
        if row is None:
            now = _now()
            c.execute(
                "INSERT INTO organizations (org_id, name, slug, plan, status,"
                " meta_json, created_at, updated_at)"
                " VALUES (?,?,?, 'free','active', NULL, ?, ?)",
                (DEFAULT_ORG_ID, name, "default-org", now, now),
            )
    return DEFAULT_ORG_ID


def attach_orphans_to_org(
    org_id: str, db_path: Path | str | None = None
) -> dict[str, int]:
    """Attach every org-less project and user to ``org_id``. Idempotent.

    Projects with NULL/empty ``org_id`` get stamped; users likewise, and each
    gains an ``org_members`` row (its roster role, clamped to a known role).
    """
    with _conn(db_path) as c:
        proj = c.execute(
            "UPDATE projects SET org_id=?, updated_at=?"
            " WHERE org_id IS NULL OR org_id=''",
            (org_id, _now()),
        ).rowcount
        users = c.execute(
            "SELECT user_id, role FROM users WHERE org_id IS NULL OR org_id=''"
        ).fetchall()
        for u in users:
            role = u["role"] if u["role"] in _KNOWN_ROLES else "analyst"
            c.execute(
                "UPDATE users SET org_id=?, updated_at=? WHERE user_id=?",
                (org_id, _now(), u["user_id"]),
            )
            c.execute(
                "INSERT OR IGNORE INTO org_members (org_id, user_id, role, created_at)"
                " VALUES (?,?,?,?)",
                (org_id, u["user_id"], role, _now()),
            )
    return {"projects": int(proj or 0), "users": len(users)}


# ----- token revocation (refresh-token blocklist) ----------------------------


def revoke_token(
    jti: str,
    expires_at: float,
    user_id: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Add a refresh-token jti to the revocation list until it would expire."""
    if not jti:
        return
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO revoked_tokens (jti, user_id, expires_at, revoked_at)"
            " VALUES (?,?,?,?)",
            (jti, user_id, float(expires_at), _now()),
        )


def is_token_revoked(jti: str, db_path: Path | str | None = None) -> bool:
    if not jti:
        return False
    with _conn(db_path) as c:
        row = c.execute("SELECT 1 FROM revoked_tokens WHERE jti=?", (jti,)).fetchone()
    return row is not None


def purge_expired_revocations(db_path: Path | str | None = None) -> int:
    """Drop revocation rows whose token has expired anyway (housekeeping)."""
    with _conn(db_path) as c:
        n = c.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < ?", (_now(),)
        ).rowcount
    return int(n or 0)


# ----- user status ------------------------------------------------------------


def set_user_status(
    user_id: str, status: str, db_path: Path | str | None = None
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "UPDATE users SET status=?, updated_at=? WHERE user_id=?",
            (status, _now(), user_id),
        )


def get_token_version(user_id: str, db_path: Path | str | None = None) -> int:
    """Current token generation for a user (0 if absent/legacy)."""
    with _conn(db_path) as c:
        try:
            row = c.execute(
                "SELECT token_version FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row["token_version"]) if row and row["token_version"] is not None else 0


def bump_token_version(user_id: str, db_path: Path | str | None = None) -> int:
    """Invalidate all of a user's outstanding tokens; returns the new version."""
    with _conn(db_path) as c:
        c.execute(
            "UPDATE users SET token_version = COALESCE(token_version, 0) + 1,"
            " updated_at=? WHERE user_id=?",
            (_now(), user_id),
        )
        row = c.execute(
            "SELECT token_version FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    return int(row["token_version"]) if row else 0


# ----- invites ----------------------------------------------------------------


def create_invite(
    *,
    token: str,
    org_id: str,
    email: str,
    role: str,
    invited_by: str | None,
    expires_at: float,
    db_path: Path | str | None = None,
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO invites (token, org_id, email, role, invited_by,"
            " expires_at, accepted_at, created_at) VALUES (?,?,?,?,?,?, NULL, ?)",
            (token, org_id, email, role, invited_by, float(expires_at), _now()),
        )


def get_invite(token: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM invites WHERE token=?", (token,)).fetchone()
    return dict(row) if row else None


def mark_invite_accepted(token: str, db_path: Path | str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute("UPDATE invites SET accepted_at=? WHERE token=?", (_now(), token))


# ----- password resets --------------------------------------------------------


def create_password_reset(
    *,
    token: str,
    user_id: str,
    expires_at: float,
    db_path: Path | str | None = None,
) -> None:
    with _conn(db_path) as c:
        c.execute(
            "INSERT INTO password_resets (token, user_id, expires_at, used_at,"
            " created_at) VALUES (?,?,?, NULL, ?)",
            (token, user_id, float(expires_at), _now()),
        )


def get_password_reset(
    token: str, db_path: Path | str | None = None
) -> dict[str, Any] | None:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT * FROM password_resets WHERE token=?", (token,)
        ).fetchone()
    return dict(row) if row else None


def mark_reset_used(token: str, db_path: Path | str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute("UPDATE password_resets SET used_at=? WHERE token=?", (_now(), token))


# ----- org plan + usage metering ---------------------------------------------


def set_org_plan(org_id: str, plan: str, db_path: Path | str | None = None) -> None:
    with _conn(db_path) as c:
        c.execute(
            "UPDATE organizations SET plan=?, updated_at=? WHERE org_id=?",
            (plan, _now(), org_id),
        )


def count_org_members(org_id: str, db_path: Path | str | None = None) -> int:
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM org_members WHERE org_id=?", (org_id,)
        ).fetchone()
    return int(row["n"]) if row else 0


def count_org_projects(org_id: str, db_path: Path | str | None = None) -> int:
    with _conn(db_path) as c:
        try:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM projects WHERE org_id=?", (org_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row["n"]) if row else 0


def count_org_fits_since(
    org_id: str, since_ts: float, db_path: Path | str | None = None
) -> int:
    """Fits (run_metrics rows) for the org's projects since ``since_ts``.

    Reads the agent app's ``run_metrics`` table; returns 0 if it doesn't exist
    (e.g. a fresh DB or the classic-API-only deployment)."""
    with _conn(db_path) as c:
        try:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM run_metrics WHERE created_at >= ?"
                " AND project_id IN (SELECT project_id FROM projects WHERE org_id=?)",
                (since_ts, org_id),
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row["n"]) if row else 0

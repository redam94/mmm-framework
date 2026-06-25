"""Online backup / restore for the SQLite sessions + auth store.

The entire platform state (sessions, auth users/tokens, run_metrics, langgraph
checkpoints) lives in one SQLite file with no backup tooling — a single disk loss
destroys everything. This adds a CONSISTENT online backup (SQLite's backup API,
safe while the app is running, WAL-aware) and a restore, plus a CLI::

    python -m mmm_framework.api.backup backup  /backups/sessions-2026-06-25.db
    python -m mmm_framework.api.backup restore /backups/sessions-2026-06-25.db

This is the stop-gap for the single-SQLite risk until the Postgres migration
(action plan P1a). See technical-docs/disaster-recovery.md.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from . import sessions as sessions_store


def _default_db_path() -> Path:
    return Path(sessions_store.DB_PATH)


def _sqlite_copy(src: Path, dest: Path) -> None:
    """Consistent online copy of ``src`` -> ``dest`` via the SQLite backup API."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    try:
        dst_conn = sqlite3.connect(str(dest))
        try:
            with dst_conn:
                src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def backup_db(dest: str | Path, *, db_path: str | Path | None = None) -> Path:
    """Write a consistent snapshot of the live DB to ``dest``. Returns ``dest``."""
    src = Path(db_path) if db_path else _default_db_path()
    if not src.exists():
        raise FileNotFoundError(f"Source database not found: {src}")
    dest = Path(dest)
    _sqlite_copy(src, dest)
    return dest


def restore_db(src: str | Path, *, db_path: str | Path | None = None) -> Path:
    """Restore the DB from a backup at ``src`` into the live path. Returns target."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Backup not found: {src}")
    target = Path(db_path) if db_path else _default_db_path()
    _sqlite_copy(src, target)
    return target


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Backup/restore the MMM sessions DB")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup", help="snapshot the live DB to a file")
    b.add_argument("dest")
    b.add_argument("--db", default=None, help="source DB path (default: sessions.db)")
    r = sub.add_parser("restore", help="restore the DB from a backup file")
    r.add_argument("src")
    r.add_argument("--db", default=None, help="target DB path (default: sessions.db)")
    args = ap.parse_args(argv)
    if args.cmd == "backup":
        out = backup_db(args.dest, db_path=args.db)
        print(f"Backed up to {out}")
    else:
        out = restore_db(args.src, db_path=args.db)
        print(f"Restored to {out}")


if __name__ == "__main__":  # pragma: no cover
    main()

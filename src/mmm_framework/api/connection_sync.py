"""Scheduled refresh of saved data connections.

A background lifespan tick calls :func:`sync_due_connections`, which pulls every
connection whose ``next_sync_at`` has arrived, writes a per-project snapshot CSV,
and records freshness (last_synced / row count / status). Errors are captured on
the row (scrubbed) rather than raised, so one broken connection never stalls the
batch and shows up in the UI.

``reader`` and ``writer`` are injectable so the scheduling/recording logic is
unit-testable without real cloud credentials.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from . import sessions as store


def sync_interval_seconds() -> float:
    """How often the lifespan tick fires (``MMM_CONNECTION_SYNC_INTERVAL``).

    Default 300s; ``0`` disables the scheduler entirely.
    """
    try:
        return float(os.environ.get("MMM_CONNECTION_SYNC_INTERVAL", "300"))
    except ValueError:
        return 300.0


def max_rows() -> int:
    """Per-sync row cap (disk guard); ``MMM_CONNECTION_SYNC_MAX_ROWS``."""
    try:
        return int(os.environ.get("MMM_CONNECTION_SYNC_MAX_ROWS", "5000000"))
    except ValueError:
        return 5_000_000


def _default_reader(kind: str, config: dict[str, Any], *, max_rows: int | None = None):
    from mmm_framework.integrations import read_connection_dataframe

    return read_connection_dataframe(kind, config, max_rows=max_rows)


def _default_writer(conn: dict[str, Any], df) -> str:
    from mmm_framework.agents import workspace as ws

    pid = conn.get("project_id") or "default"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", conn.get("name") or conn["id"])[:80]
    path = ws.project_data_dir(pid) / f"{safe}.csv"
    df.to_csv(path, index=False)
    return str(path)


def sync_due_connections(
    now: float,
    *,
    reader: Callable[..., Any] | None = None,
    writer: Callable[[dict[str, Any], Any], str] | None = None,
    limit: int = 100,
) -> dict[str, int]:
    """Refresh every due connection; return ``{attempted, ok, failed}``."""
    reader = reader or _default_reader
    writer = writer or _default_writer
    due = store.list_due_data_connections(now, limit=limit)
    ok = failed = 0
    for conn in due:
        cid = conn["id"]
        try:
            df = reader(conn["kind"], conn.get("config") or {}, max_rows=max_rows())
            n = int(len(df))
            try:
                path = writer(conn, df)
            except Exception:  # snapshot write is best-effort
                path = None
            store.record_data_connection_sync(
                cid, status="ok", row_count=n, snapshot_path=path, now=now
            )
            ok += 1
        except Exception as exc:  # noqa: BLE001 - record, never raise
            from mmm_framework.integrations import scrub_cloud_error

            detail = scrub_cloud_error(f"{type(exc).__name__}: {exc}")[:500]
            store.record_data_connection_sync(
                cid, status="error", error=detail, now=now
            )
            failed += 1
    return {"attempted": len(due), "ok": ok, "failed": failed}

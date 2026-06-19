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


def read_timeout_seconds() -> float:
    """Per-connection read timeout so one hung pull can't stall the batch."""
    try:
        return float(os.environ.get("MMM_CONNECTION_SYNC_READ_TIMEOUT", "600"))
    except ValueError:
        return 600.0


def _default_writer(conn: dict[str, Any], df) -> str:
    from mmm_framework.agents import workspace as ws

    pid = conn.get("project_id") or "default"
    # Name is for readability only; the connection id guarantees uniqueness so
    # two connections whose names sanitize to the same string never collide.
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", conn.get("name") or "")[:60]
    cid = re.sub(r"[^A-Za-z0-9]+", "", conn["id"])[:8]
    path = ws.project_data_dir(pid) / f"{base}_{cid}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _record_error(cid: str, message: str, now: float) -> None:
    from mmm_framework.integrations import scrub_cloud_error

    store.record_data_connection_sync(
        cid, status="error", error=scrub_cloud_error(message)[:500], now=now
    )


def sync_due_connections(
    now: float,
    *,
    reader: Callable[..., Any] | None = None,
    writer: Callable[[dict[str, Any], Any], str] | None = None,
    limit: int = 100,
) -> dict[str, int]:
    """Refresh every due connection; return ``{attempted, ok, failed}``.

    Each connection is isolated: a read timeout, a row-cap breach, a read error,
    or a snapshot-write failure records that one connection as ``error`` and the
    batch continues. ``ok`` is recorded only when the snapshot was actually
    written, so the status never lies about cached data.
    """
    import concurrent.futures

    reader = reader or _default_reader
    writer = writer or _default_writer
    cap = max_rows()
    timeout = read_timeout_seconds()
    due = store.list_due_data_connections(now, limit=limit)
    ok = failed = 0
    for conn in due:
        cid = conn["id"]
        # Read with a per-connection timeout (a hung cloud call must not stall
        # the rest of the batch — the worker is abandoned, not awaited).
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(
                reader, conn["kind"], conn.get("config") or {}, max_rows=cap
            )
            df = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            _record_error(cid, f"read timed out after {int(timeout)}s", now)
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001
            _record_error(cid, f"{type(exc).__name__}: {exc}", now)
            failed += 1
            continue
        finally:
            ex.shutdown(wait=False)

        if len(df) > cap:
            _record_error(
                cid,
                f"result exceeds the {cap:,}-row cap; narrow the connection "
                "(add a WHERE/date filter or a LIMIT)",
                now,
            )
            failed += 1
            continue

        try:
            path = writer(conn, df)
        except Exception as exc:  # noqa: BLE001 - a failed write is a failed sync
            _record_error(
                cid, f"snapshot write failed: {type(exc).__name__}: {exc}", now
            )
            failed += 1
            continue

        store.record_data_connection_sync(
            cid, status="ok", row_count=int(len(df)), snapshot_path=path, now=now
        )
        ok += 1
    return {"attempted": len(due), "ok": ok, "failed": failed}

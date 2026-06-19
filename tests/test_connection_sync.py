"""Scheduled data-connection sync: schedule store + sync_due_connections + PATCH."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from fastapi import HTTPException

from mmm_framework.api import connection_sync
from mmm_framework.api import sessions as ss


@pytest.fixture()
def db(tmp_path, monkeypatch):
    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(ss, "DB_PATH", dbp)
    ss.init_db()
    return dbp


def _make(pid, name="weekly"):
    return ss.create_data_connection(
        pid, name, "bigquery", {"dataset": "mmm", "query": "SELECT 1"}
    )


def test_schedule_and_due(db):
    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    # Manual connections are never due.
    assert ss.list_due_data_connections(10_000.0) == []

    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)  # next = 1000 + 3600
    assert ss.get_data_connection(conn["id"])["next_sync_at"] == 4600.0
    assert ss.list_due_data_connections(4000.0) == []  # not yet
    due = ss.list_due_data_connections(5000.0)
    assert [c["id"] for c in due] == [conn["id"]]

    # Clearing the schedule stops it.
    ss.set_data_connection_schedule(conn["id"], None, now=1000.0)
    assert ss.get_data_connection(conn["id"])["next_sync_at"] is None
    assert ss.list_due_data_connections(5000.0) == []


def test_sync_due_records_success_and_advances(db):
    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)  # due at >= 4600

    df = pd.DataFrame({"a": [1, 2, 3]})
    out = connection_sync.sync_due_connections(
        5000.0,
        reader=lambda kind, config, max_rows=None: df,
        writer=lambda c, d: "/tmp/snap.csv",
    )
    assert out == {"attempted": 1, "ok": 1, "failed": 0}

    row = ss.get_data_connection(conn["id"])
    assert row["last_sync_status"] == "ok"
    assert row["last_row_count"] == 3
    assert row["snapshot_path"] == "/tmp/snap.csv"
    assert row["next_sync_at"] == 5000.0 + 3600.0  # advanced from the sync time
    assert ss.list_due_data_connections(5000.0) == []  # no longer due


def test_sync_due_records_error_scrubbed(db):
    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)

    def boom(kind, config, max_rows=None):
        raise RuntimeError("PermissionDenied on projects/secret-proj-99")

    out = connection_sync.sync_due_connections(
        5000.0, reader=boom, writer=lambda c, d: "x"
    )
    assert out == {"attempted": 1, "ok": 0, "failed": 1}
    row = ss.get_data_connection(conn["id"])
    assert row["last_sync_status"] == "error"
    assert "secret-proj-99" not in (row["last_sync_error"] or "")
    assert "projects/***" in row["last_sync_error"]
    # Still rescheduled so a transient failure retries next interval.
    assert row["next_sync_at"] == 5000.0 + 3600.0


def test_schedule_endpoint(db):
    from mmm_framework.api import main as M

    pid = ss.create_project("P")["project_id"]
    conn = ss.create_data_connection(pid, "c", "gcs", {"bucket": "b"})

    resp = asyncio.run(
        M.update_data_connection_schedule_endpoint(
            pid, conn["id"], M.ConnectionScheduleUpdate(sync_interval_minutes=120)
        )
    )
    body = json.loads(resp.body)
    assert body["sync_interval_minutes"] == 120 and body["next_sync_at"] is not None

    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            M.update_data_connection_schedule_endpoint(
                pid, conn["id"], M.ConnectionScheduleUpdate(sync_interval_minutes=0)
            )
        )
    assert ei.value.status_code == 422

    resp = asyncio.run(
        M.update_data_connection_schedule_endpoint(
            pid, conn["id"], M.ConnectionScheduleUpdate(sync_interval_minutes=None)
        )
    )
    body = json.loads(resp.body)
    assert body["sync_interval_minutes"] is None and body["next_sync_at"] is None

    # Cross-project access is a 404.
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            M.update_data_connection_schedule_endpoint(
                "other",
                conn["id"],
                M.ConnectionScheduleUpdate(sync_interval_minutes=60),
            )
        )
    assert ei.value.status_code == 404

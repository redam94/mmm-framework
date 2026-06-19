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
    # Rescheduled with backoff (4x the 60-min interval = 4h) so a broken
    # connection isn't retried every interval forever.
    assert row["next_sync_at"] == 5000.0 + 14400.0


def test_sync_due_enforces_row_cap(db, monkeypatch):
    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)
    monkeypatch.setattr(connection_sync, "max_rows", lambda: 2)

    big = pd.DataFrame({"a": [1, 2, 3]})  # 3 > cap of 2
    out = connection_sync.sync_due_connections(
        5000.0, reader=lambda *a, **k: big, writer=lambda c, d: "x"
    )
    assert out == {"attempted": 1, "ok": 0, "failed": 1}
    row = ss.get_data_connection(conn["id"])
    assert row["last_sync_status"] == "error" and "cap" in row["last_sync_error"]


def test_sync_due_read_timeout(db, monkeypatch):
    import time as _t

    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)
    monkeypatch.setattr(connection_sync, "read_timeout_seconds", lambda: 0.05)

    def slow(*a, **k):
        _t.sleep(0.4)
        return pd.DataFrame({"a": [1]})

    out = connection_sync.sync_due_connections(
        5000.0, reader=slow, writer=lambda c, d: "x"
    )
    assert out["failed"] == 1
    row = ss.get_data_connection(conn["id"])
    assert row["last_sync_status"] == "error" and "timed out" in row["last_sync_error"]


def test_sync_due_writer_failure_is_error_not_ok(db):
    pid = ss.create_project("P")["project_id"]
    conn = _make(pid)
    ss.set_data_connection_schedule(conn["id"], 60, now=1000.0)

    def bad_writer(c, d):
        raise OSError("disk full")

    out = connection_sync.sync_due_connections(
        5000.0, reader=lambda *a, **k: pd.DataFrame({"a": [1]}), writer=bad_writer
    )
    assert out == {"attempted": 1, "ok": 0, "failed": 1}
    row = ss.get_data_connection(conn["id"])
    # A failed snapshot write must NOT be reported as a successful sync.
    assert row["last_sync_status"] == "error"
    assert "snapshot write failed" in row["last_sync_error"]


def test_default_writer_unique_per_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path))
    df = pd.DataFrame({"a": [1]})
    # Two names that sanitize identically must still produce distinct files.
    p1 = connection_sync._default_writer(
        {"id": "aaaaaaaa1111", "project_id": "P", "name": "Daily@Report"}, df
    )
    p2 = connection_sync._default_writer(
        {"id": "bbbbbbbb2222", "project_id": "P", "name": "Daily#Report"}, df
    )
    assert p1 != p2
    assert ".." not in p1 and ".." not in p2  # no dot-traversal tokens in names


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

    # Absurdly large interval is rejected (would overflow next_sync_at -> inf).
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            M.update_data_connection_schedule_endpoint(
                pid, conn["id"], M.ConnectionScheduleUpdate(sync_interval_minutes=1e9)
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

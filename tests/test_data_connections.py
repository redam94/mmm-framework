"""Saved data connections: store CRUD + endpoints + the sync agent tool."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from mmm_framework.api import sessions as ss


@pytest.fixture()
def db(tmp_path, monkeypatch):
    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(ss, "DB_PATH", dbp)
    ss.init_db()
    return dbp


def test_store_crud(db):
    pid = ss.create_project("P")["project_id"]
    conn = ss.create_data_connection(
        pid, "weekly", "bigquery", {"dataset": "mmm", "query": "SELECT 1"}
    )
    assert conn["name"] == "weekly" and conn["kind"] == "bigquery"
    # config is parsed back; the raw json column is not leaked.
    assert conn["config"]["query"] == "SELECT 1" and "config_json" not in conn
    assert conn["last_synced"] is None

    assert [c["id"] for c in ss.list_data_connections(pid)] == [conn["id"]]
    assert ss.get_data_connection(conn["id"])["name"] == "weekly"
    assert ss.get_data_connection_by_name(pid, "weekly")["id"] == conn["id"]
    assert ss.get_data_connection_by_name(pid, "nope") is None

    ss.touch_data_connection_synced(conn["id"])
    assert ss.get_data_connection(conn["id"])["last_synced"] is not None

    assert ss.delete_data_connection(conn["id"]) is True
    assert ss.list_data_connections(pid) == []


def _body_of(resp) -> dict:
    return json.loads(resp.body)


def test_endpoints_create_list_delete(db):
    from mmm_framework.api import main as M

    pid = ss.create_project("P")["project_id"]
    body = M.DataConnectionCreate(
        name="obj", kind="gcs", config={"bucket": "b", "object": "d.csv"}
    )
    created = _body_of(asyncio.run(M.create_data_connection_endpoint(pid, body)))
    cid = created["id"]
    assert created["kind"] == "gcs"

    listed = _body_of(asyncio.run(M.list_data_connections_endpoint(pid)))
    assert [c["id"] for c in listed["connections"]] == [cid]

    asyncio.run(M.delete_data_connection_endpoint(pid, cid))
    after = _body_of(asyncio.run(M.list_data_connections_endpoint(pid)))
    assert after["connections"] == []


def test_endpoint_validation(db):
    from mmm_framework.api import main as M

    pid = ss.create_project("P")["project_id"]

    # Unknown kind -> 422.
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            M.create_data_connection_endpoint(
                pid, M.DataConnectionCreate(name="x", kind="snowflake", config={})
            )
        )
    assert ei.value.status_code == 422

    # Missing project -> 404.
    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            M.create_data_connection_endpoint(
                "nope",
                M.DataConnectionCreate(name="x", kind="gcs", config={"bucket": "b"}),
            )
        )
    assert ei.value.status_code == 404

    # Cross-project access is a 404 (IDOR guard).
    conn = ss.create_data_connection(pid, "c", "gcs", {"bucket": "b"})
    with pytest.raises(HTTPException) as ei:
        asyncio.run(M.delete_data_connection_endpoint("other-project", conn["id"]))
    assert ei.value.status_code == 404


def test_sync_tool_registered_and_needs_project():
    from mmm_framework.agents.tools import TOOLS, sync_data_connection

    assert any(t.name == "sync_data_connection" for t in TOOLS)
    # config=None -> no thread -> no project -> a guided message, never a raise.
    result = sync_data_connection.func(
        state={}, name="weekly", tool_call_id="t", config=None
    )
    msg = result.update["messages"][0].content
    assert "project" in msg.lower()

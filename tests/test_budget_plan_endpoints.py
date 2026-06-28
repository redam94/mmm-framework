"""Budget-plan persistence + CSV export endpoints (B1, B5) on the agent API."""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    import mmm_framework.api.main as main

    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def project(client):
    return client.post("/projects", json={"name": "P"}).json()["project_id"]


_PAYLOAD = {
    "total_budget": 1200.0,
    "expected_uplift": 80.0,
    "uplift_hdi": [20.0, 150.0],
    "prob_positive_uplift": 0.9,
    "allocation": [
        {
            "channel": "TV",
            "current_spend": 600.0,
            "optimal_spend": 700.0,
            "change_pct": 16.7,
        },
        {
            "channel": "Search",
            "current_spend": 600.0,
            "optimal_spend": 500.0,
            "change_pct": -16.7,
        },
    ],
    "flighting": {
        "pattern": "even",
        "channels": ["TV", "Search"],
        "schedule": [{"period": "P1", "TV": 350.0, "Search": 250.0, "total": 600.0}],
    },
}


def _make_plan(client, project, name="Q3 Plan"):
    return client.post(
        "/budget-plans",
        json={
            "name": name,
            "project_id": project,
            "kind": "optimization",
            "plan_payload": _PAYLOAD,
        },
    )


def test_create_list_get_delete_roundtrip(client, project):
    r = _make_plan(client, project)
    assert r.status_code == 201, r.text
    plan = r.json()
    pid = plan["plan_id"]
    assert plan["name"] == "Q3 Plan"
    assert plan["plan_payload"]["allocation"][0]["channel"] == "TV"

    listing = client.get("/budget-plans", params={"project_id": project}).json()
    assert listing["total"] == 1
    assert listing["plans"][0]["plan_id"] == pid

    got = client.get(f"/budget-plans/{pid}").json()
    assert got["plan_id"] == pid

    d = client.delete(f"/budget-plans/{pid}").json()
    assert d["deleted"] is True
    assert (
        client.get("/budget-plans", params={"project_id": project}).json()["total"] == 0
    )


def test_update_in_place(client, project):
    pid = _make_plan(client, project).json()["plan_id"]
    r = client.post(
        "/budget-plans",
        json={
            "plan_id": pid,
            "name": "Renamed",
            "project_id": project,
            "plan_payload": _PAYLOAD,
        },
    )
    assert r.status_code == 201
    assert r.json()["name"] == "Renamed"
    assert (
        client.get("/budget-plans", params={"project_id": project}).json()["total"] == 1
    )


def test_get_missing_is_404(client):
    assert client.get("/budget-plans/nope").status_code == 404


def test_export_csv(client, project):
    pid = _make_plan(client, project).json()["plan_id"]
    r = client.get(f"/budget-plans/{pid}/export.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "attachment" in r.headers.get("content-disposition", "")
    body = r.text
    assert "Allocation" in body and "TV" in body and "Search" in body
    assert "Flighting calendar" in body and "P1" in body


def test_scenario_plan_csv(client, project):
    r = client.post(
        "/budget-plans",
        json={
            "name": "What-if",
            "project_id": project,
            "kind": "scenario",
            "baseline_outcome": 1000.0,
            "scenario_outcome": 1100.0,
            "outcome_change": 100.0,
            "outcome_change_pct": 10.0,
            "channel_details": {"TV": {"change_pct": 20.0}},
            "plan_payload": {},
        },
    )
    pid = r.json()["plan_id"]
    body = client.get(f"/budget-plans/{pid}/export.csv").text
    assert "Scenario" in body and "outcome_change" in body


def test_planner_optimize_job_without_model(client, project):
    """No fitted model → the async job resolves to an error, not a 500."""
    start = client.post(f"/projects/{project}/planner/optimize", json={"by_geo": False})
    assert start.status_code == 202, start.text
    job_id = start.json()["job_id"]
    # poll a few times; with no saved model the job lands on 'error'
    import time

    status = None
    for _ in range(40):
        poll = client.get(f"/projects/{project}/planner/optimize/{job_id}").json()
        status = poll["status"]
        if status in ("done", "error"):
            assert status == "error"
            assert "model" in (poll.get("error") or "").lower()
            break
        time.sleep(0.05)
    assert status in ("done", "error")

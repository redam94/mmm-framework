"""Budget-plan persistence + CSV export endpoints (B1, B5) on the agent API."""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.platform import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    import mmm_framework_server.main as main

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


# ── forecast endpoint (#223) ─────────────────────────────────────────────────


def test_planner_forecast_requires_a_plan_shape(client, project):
    """Neither future_media nor channel_budgets → 400 up front, rather than a
    background job that fails 20 seconds later."""
    r = client.post(f"/projects/{project}/planner/forecast", json={"n_periods": 4})
    assert r.status_code == 400, r.text
    assert "channel_budgets" in r.json()["detail"]


def test_planner_forecast_unknown_project_is_404(client):
    r = client.post(
        "/projects/does-not-exist/planner/forecast",
        json={"channel_budgets": {"TV": 100.0}},
    )
    assert r.status_code == 404


def test_planner_forecast_job_without_model(client, project):
    """No fitted model → the async job resolves to an error, not a 500."""
    import time

    start = client.post(
        f"/projects/{project}/planner/forecast",
        json={"channel_budgets": {"TV": 1000.0}, "n_periods": 4},
    )
    assert start.status_code == 202, start.text
    job_id = start.json()["job_id"]
    status = None
    for _ in range(40):
        poll = client.get(f"/projects/{project}/planner/forecast/{job_id}").json()
        status = poll["status"]
        if status in ("done", "error"):
            assert status == "error"
            assert "model" in (poll.get("error") or "").lower()
            break
        time.sleep(0.05)
    assert status in ("done", "error")


# ── payback horizon route (issue #224) ────────────────────────────────────────


def test_planner_payback_bad_basis_is_400(client, project):
    """An invalid basis fails up front, not as a background job 20s later."""
    r = client.post(f"/projects/{project}/planner/payback", json={"basis": "vibes"})
    assert r.status_code == 400, r.text
    assert "basis" in r.json()["detail"]


def test_planner_payback_unknown_project_is_404(client):
    r = client.post("/projects/does-not-exist/planner/payback", json={})
    assert r.status_code == 404


def test_planner_payback_job_without_model(client, project):
    """No fitted model → the async job resolves to an error, not a 500."""
    import time

    start = client.post(f"/projects/{project}/planner/payback", json={})
    assert start.status_code == 202, start.text
    job_id = start.json()["job_id"]
    status = None
    for _ in range(40):
        poll = client.get(f"/projects/{project}/planner/payback/{job_id}").json()
        status = poll["status"]
        if status in ("done", "error"):
            assert status == "error"
            assert "model" in (poll.get("error") or "").lower()
            break
        time.sleep(0.05)
    assert status in ("done", "error")


# ── plan-of-record routes (issue #225 remainder) ─────────────────────────────


def _committable_forecast():
    return {
        "periods": ["2025-01-06"] * 8,
        "mean": [100.0] * 8,
        "draws_b64": "AAAA",
        "n_draws": 200,
        "random_seed": 42,
        "caveat_fields": {
            "interval_widens_with_horizon": True,
            "trend_extrapolation": {"policy": "linear", "trend_type": "linear"},
            "residual_autocorrelation": {
                "autocorrelated": False,
                "ljung_box_p": 0.42,
            },
            "extrapolated_channels": [],
            "interval_available": True,
        },
    }


def _seed_provenance_and_valuation(client, project):
    """A model run with resolvable provenance + a resolved valuation, so the
    only gates under test are the ones each case manipulates."""
    from mmm_framework.platform import sessions as S

    S.set_preference(project, "economics", {"kind": "revenue", "gross_margin": 0.4})
    tid = S.create_session(name="test", project_id=project)["thread_id"]
    S.add_artifact(
        tid,
        "model_run",
        {
            "run_id": "run-1",
            "spec_hash": "abc123",
            "data_fingerprint": "deadbeef0123",
            "model_path": "/tmp/model",
        },
    )


def test_plan_of_record_unknown_project_is_404(client):
    r = client.post(
        "/projects/nope/plan-of-record",
        json={"forecast": _committable_forecast()},
    )
    assert r.status_code == 404


def test_assess_only_reports_gates_without_writing(client, project):
    _seed_provenance_and_valuation(client, project)
    fc = _committable_forecast()
    fc["caveat_fields"]["residual_autocorrelation"] = {
        "autocorrelated": True,
        "ljung_box_p": 0.001,
    }
    r = client.post(
        f"/projects/{project}/plan-of-record",
        json={"forecast": fc, "assess_only": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committable"] is False
    gates = [x["gate"] for x in body["assessment"]["refusals"]]
    assert "residual_autocorrelation" in gates
    # Nothing was written.
    h = client.get(f"/projects/{project}/plan-of-record/history").json()
    assert h["total"] == 0


def test_refused_commit_is_422_with_the_gate_named(client, project):
    _seed_provenance_and_valuation(client, project)
    fc = _committable_forecast()
    fc["caveat_fields"]["extrapolated_channels"] = [{"channel": "TV", "multiple": 1.6}]
    r = client.post(f"/projects/{project}/plan-of-record", json={"forecast": fc})
    assert r.status_code == 422, r.text
    gates = [x["gate"] for x in r.json()["detail"]["assessment"]["refusals"]]
    assert "spend_support" in gates


def test_commit_then_latest_then_history(client, project):
    _seed_provenance_and_valuation(client, project)
    r = client.post(
        f"/projects/{project}/plan-of-record",
        json={"forecast": _committable_forecast(), "name": "FY25 H1"},
    )
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["version"] == 1
    assert "payload" not in v  # listings/creation responses stay slim

    latest = client.get(f"/projects/{project}/plan-of-record").json()["version"]
    assert latest["id"] == v["id"]
    assert latest["payload"]["forecast"]["n_draws"] == 200

    r2 = client.post(
        f"/projects/{project}/plan-of-record",
        json={"forecast": _committable_forecast()},
    )
    assert r2.json()["version"] == 2

    h = client.get(f"/projects/{project}/plan-of-record/history").json()
    assert h["total"] == 2
    fam = v["plan_family"]
    assert h["chains"][fam]["intact"] is True


def test_override_is_recorded_in_the_committed_payload(client, project):
    _seed_provenance_and_valuation(client, project)
    fc = _committable_forecast()
    fc["caveat_fields"]["extrapolated_channels"] = [{"channel": "TV", "multiple": 1.6}]
    r = client.post(
        f"/projects/{project}/plan-of-record",
        json={
            "forecast": fc,
            "overrides": {"spend_support": "CMO accepted the extrapolation"},
        },
    )
    assert r.status_code == 201, r.text
    latest = client.get(f"/projects/{project}/plan-of-record").json()["version"]
    committability = latest["payload"]["committability"]
    assert "spend_support" in committability["overrides"]


def test_pacing_retargets_to_the_committed_plan(client, project):
    """`latest_budget_plan_for_project` prefers the committed version: pacing
    must grade delivery against what was PROMISED, not the editable draft."""
    from mmm_framework.platform import sessions as S

    _seed_provenance_and_valuation(client, project)
    # A working draft...
    client.post(
        "/budget-plans",
        json={"project_id": project, "name": "draft", "plan_payload": _PAYLOAD},
    )
    before = S.latest_budget_plan_for_project(project)
    assert not before.get("committed")
    # ...then a commit (the committed payload freezes the draft's plan_payload).
    r = client.post(
        f"/projects/{project}/plan-of-record",
        json={"forecast": _committable_forecast(), "name": "committed"},
    )
    assert r.status_code == 201, r.text
    after = S.latest_budget_plan_for_project(project)
    assert after["committed"] is True
    assert after["plan_payload"]["total_budget"] == _PAYLOAD["total_budget"]
    # The draft keeps editing freely; the committed pointer does not move.
    client.post(
        "/budget-plans",
        json={
            "project_id": project,
            "name": "draft2",
            "plan_payload": {**_PAYLOAD, "total_budget": 9999.0},
        },
    )
    still = S.latest_budget_plan_for_project(project)
    assert still["committed"] is True
    assert still["plan_payload"]["total_budget"] == _PAYLOAD["total_budget"]

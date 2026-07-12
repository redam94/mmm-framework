"""Delivery ingestion + in-flight pacing persistence/endpoints (issue #123).

The pacing ENGINE (planning/pacing.py) and report sections shipped in #107; this
covers the persistence + endpoint layer: the delivery registry, the CSV/JSON
upload parser, the plan-auto-sourcing pacing join, and the REST surface.
"""

from __future__ import annotations

import pytest

from mmm_framework.api import pacing as P


# ---------------------------------------------------------------------------
# upload parser (pure)
# ---------------------------------------------------------------------------
class TestParseDelivery:
    def test_wide_csv(self):
        recs = P.parse_delivery_records(
            b"period,TV,Search\nW1,130,48\nW2,120,52\n", "d.csv"
        )
        by = {(r["channel"], r["period"]): r["spend"] for r in recs}
        assert by[("TV", "W1")] == 130
        assert by[("Search", "W2")] == 52
        assert len(recs) == 4

    def test_long_csv(self):
        recs = P.parse_delivery_records(
            b"channel,period,spend\nTV,W1,130\nSearch,W1,48\n", "d.csv"
        )
        assert {r["channel"] for r in recs} == {"TV", "Search"}
        assert next(r for r in recs if r["channel"] == "TV")["spend"] == 130

    def test_json_list(self):
        recs = P.parse_delivery_records(
            b'[{"channel":"TV","period":"W1","spend":130}]', "d.json"
        )
        assert recs == [{"channel": "TV", "period": "W1", "spend": 130}]

    def test_json_nested_map(self):
        recs = P.parse_delivery_records(b'{"TV":{"W1":130,"W2":120}}', "d.json")
        by = {r["period"]: r["spend"] for r in recs}
        assert by == {"W1": 130, "W2": 120}

    def test_json_sniffed_without_extension(self):
        recs = P.parse_delivery_records(b'{"TV":250}', "blob")
        assert recs == [{"channel": "TV", "period": "", "spend": 250}]


# ---------------------------------------------------------------------------
# pacing join (pure)
# ---------------------------------------------------------------------------
_PLAN = {
    "flighting": {
        "schedule": [
            {"period": "W1", "TV": 100, "Search": 50, "total": 150},
            {"period": "W2", "TV": 100, "Search": 50, "total": 150},
        ]
    }
}
_DELIVERY = [
    {"channel": "TV", "period": "W1", "spend": 130},
    {"channel": "TV", "period": "W2", "spend": 120},
    {"channel": "Search", "period": "W1", "spend": 48},
    {"channel": "Search", "period": "W2", "spend": 52},
]


class TestProjectPacing:
    def test_available_and_flags_over_pacing(self):
        out = P.project_pacing(_PLAN, _DELIVERY, threshold=0.10)
        assert out["available"] is True
        assert out["plan_basis"] == "flighting"
        # TV delivered 250 vs 200 planned → +25% over-pace (flagged); Search on track
        assert "TV" in out["flagged"]
        assert out["alert"]["off_pace"] is True
        assert out["alert"]["worst"]["channel"] == "TV"
        tv = next(c for c in out["channels"] if c["channel"] == "TV")
        assert tv["status"] == "over-pacing"

    def test_allocation_fallback_when_no_flighting(self):
        plan = {"allocation": [{"channel": "TV", "optimal_spend": 200}]}
        out = P.project_pacing(plan, [{"channel": "TV", "period": "W1", "spend": 260}])
        assert out["available"] is True and out["plan_basis"] == "allocation"
        assert "TV" in out["flagged"]

    def test_no_plan(self):
        out = P.project_pacing(None, _DELIVERY)
        assert out["available"] is False and out["reason"] == "no_plan"

    def test_no_delivery(self):
        out = P.project_pacing(_PLAN, [])
        assert out["available"] is False and out["reason"] == "no_delivery"

    def test_on_track_has_no_alert(self):
        out = P.project_pacing(
            _PLAN,
            [
                {"channel": "TV", "period": "W1", "spend": 100},
                {"channel": "TV", "period": "W2", "spend": 100},
                {"channel": "Search", "period": "W1", "spend": 50},
                {"channel": "Search", "period": "W2", "spend": 50},
            ],
        )
        assert out["available"] is True
        assert out["flagged"] == [] and out["alert"]["off_pace"] is False


# ---------------------------------------------------------------------------
# store-backed helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


class TestDeliveryStore:
    def test_upsert_overwrites_period(self, store):
        pid = store.create_project("P")["project_id"]
        store.upsert_delivery(pid, [{"channel": "TV", "period": "W1", "spend": 130}])
        store.upsert_delivery(pid, [{"channel": "TV", "period": "W1", "spend": 999}])
        rows = store.list_delivery(pid)
        assert len(rows) == 1 and rows[0]["spend"] == 999

    def test_skips_bad_rows(self, store):
        pid = store.create_project("P")["project_id"]
        rows = store.upsert_delivery(
            pid,
            [
                {"channel": "TV", "period": "W1", "spend": 130},
                {"channel": None, "period": "W1", "spend": 5},  # no channel
                {"channel": "X", "period": "W1", "spend": "abc"},  # bad spend
            ],
        )
        assert len(rows) == 1 and rows[0]["channel"] == "TV"

    def test_delete_by_channel(self, store):
        pid = store.create_project("P")["project_id"]
        store.upsert_delivery(
            pid,
            [
                {"channel": "TV", "period": "W1", "spend": 130},
                {"channel": "Search", "period": "W1", "spend": 48},
            ],
        )
        assert store.delete_delivery(pid, channel="Search") == 1
        assert {r["channel"] for r in store.list_delivery(pid)} == {"TV"}

    def test_latest_budget_plan_for_project(self, store):
        pid = store.create_project("P")["project_id"]
        assert store.latest_budget_plan_for_project(pid) is None
        store.upsert_budget_plan(
            project_id=pid, org_id="o", name="Old", plan_payload={"allocation": []}
        )
        store.upsert_budget_plan(
            project_id=pid, org_id="o", name="New", plan_payload=_PLAN
        )
        assert store.latest_budget_plan_for_project(pid)["name"] == "New"


# ---------------------------------------------------------------------------
# endpoints (TestClient)
# ---------------------------------------------------------------------------
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


def _seed_plan(client, project):
    return client.post(
        "/budget-plans",
        json={"name": "Plan", "project_id": project, "plan_payload": _PLAN},
    )


class TestEndpoints:
    def test_upload_then_pacing(self, client, project):
        # upload a WIDE delivery CSV
        r = client.post(
            f"/projects/{project}/delivery",
            files={
                "file": (
                    "delivery.csv",
                    b"period,TV,Search\nW1,130,48\nW2,120,52\n",
                    "text/csv",
                )
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["ingested"] == 4

        # no plan yet → pacing unavailable
        pac = client.get(f"/projects/{project}/pacing").json()
        assert pac["available"] is False and pac["reason"] == "no_plan"

        # seed a plan → pacing available + TV flagged over-pace
        assert _seed_plan(client, project).status_code == 201
        pac = client.get(f"/projects/{project}/pacing").json()
        assert pac["available"] is True
        assert "TV" in pac["flagged"]
        assert pac["plan_name"] == "Plan"

    def test_list_and_delete(self, client, project):
        client.post(
            f"/projects/{project}/delivery",
            files={
                "file": (
                    "d.csv",
                    b"channel,period,spend\nTV,W1,130\nSearch,W1,48\n",
                    "text/csv",
                )
            },
        )
        assert len(client.get(f"/projects/{project}/delivery").json()["delivery"]) == 2
        assert (
            client.delete(f"/projects/{project}/delivery?channel=Search").json()[
                "deleted"
            ]
            == 1
        )
        assert len(client.get(f"/projects/{project}/delivery").json()["delivery"]) == 1

    def test_bad_upload_400(self, client, project):
        r = client.post(
            f"/projects/{project}/delivery",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code == 400

    def test_calibration_coverage_folds_pacing_alert(self, client, project):
        client.post(
            f"/projects/{project}/delivery",
            files={"file": ("d.csv", b"period,TV,Search\nW1,130,48\n", "text/csv")},
        )
        _seed_plan(client, project)
        body = client.get(f"/projects/{project}/calibration-coverage").json()
        # the pacing signal is folded onto the T5 re-eval surface (issue #123)
        assert body.get("pacing_alert") is not None
        assert body["pacing_alert"]["off_pace"] is True

    def test_pacing_404_unknown_project(self, client):
        assert client.get("/projects/nope/pacing").status_code == 404


# ---------------------------------------------------------------------------
# proactive off-pace alerting sweep (issue #123)
# ---------------------------------------------------------------------------
class TestPacingAlertSweep:
    def _seed_offpace(self, store):
        pid = store.create_project("P")["project_id"]
        store.upsert_budget_plan(
            project_id=pid, org_id="o", name="Plan", plan_payload=_PLAN
        )
        store.upsert_delivery(pid, _DELIVERY)  # TV 250 vs 200 planned → over-pace
        return pid

    def test_sweep_persists_offpace_alert(self, store):
        pid = self._seed_offpace(store)
        digest = P.sweep_pacing_alerts(now=123.0)
        assert digest["off_pace"] >= 1 and digest["persisted"] >= 1

        alert = P.latest_pacing_alert(pid)
        assert alert is not None
        assert alert["project_id"] == pid
        assert alert["alert"]["off_pace"] is True
        assert alert["alert"]["worst"]["channel"] == "TV"
        assert alert["computed_at"] == 123.0

    def test_sweep_upserts_in_place(self, store):
        pid = self._seed_offpace(store)
        P.sweep_pacing_alerts(now=1.0)
        P.sweep_pacing_alerts(now=2.0)
        tid = P._pacing_alert_thread(pid)
        alerts = [a for a in store.list_artifacts(tid) if a["kind"] == "pacing_alert"]
        assert len(alerts) == 1  # upsert, not append
        assert alerts[0]["payload"]["computed_at"] == 2.0

    def test_sweep_clears_recovered_project(self, store):
        pid = self._seed_offpace(store)
        P.sweep_pacing_alerts(now=1.0)
        assert P.latest_pacing_alert(pid)["alert"] is not None
        # deliver on track → the next sweep clears the stale alert
        store.delete_delivery(pid)
        store.upsert_delivery(
            pid,
            [
                {"channel": "TV", "period": "W1", "spend": 100},
                {"channel": "TV", "period": "W2", "spend": 100},
                {"channel": "Search", "period": "W1", "spend": 50},
                {"channel": "Search", "period": "W2", "spend": 50},
            ],
        )
        digest = P.sweep_pacing_alerts(now=2.0)
        assert digest["cleared"] >= 1
        assert P.latest_pacing_alert(pid)["alert"] is None

    def test_sweep_ignores_projects_without_a_plan(self, store):
        # a project with no plan is skipped (no artifact written)
        pid = store.create_project("NoPlan")["project_id"]
        P.sweep_pacing_alerts(now=1.0)
        assert P.latest_pacing_alert(pid) is None


class TestPacingAlertEndpoint:
    def test_alert_endpoint_reads_persisted_sweep(self, client, project):
        client.post(
            f"/projects/{project}/delivery",
            files={
                "file": (
                    "d.csv",
                    b"period,TV,Search\nW1,130,48\nW2,120,52\n",
                    "text/csv",
                )
            },
        )
        _seed_plan(client, project)
        # before the sweep runs → no alert
        assert client.get(f"/projects/{project}/pacing/alert").json()["alert"] is None
        # run the sweep, then the endpoint surfaces the persisted alert
        P.sweep_pacing_alerts(now=5.0)
        body = client.get(f"/projects/{project}/pacing/alert").json()
        assert body["alert"]["off_pace"] is True
        assert body["alert"]["worst"]["channel"] == "TV"

    def test_alert_endpoint_404_unknown_project(self, client):
        assert client.get("/projects/nope/pacing/alert").status_code == 404

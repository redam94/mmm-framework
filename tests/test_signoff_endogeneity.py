"""Sign-off audit workflow + endogeneity diagnostics (issue #110)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.agents import model_ops as M
from mmm_framework.diagnostics.endogeneity import endogeneity_diagnostic


def _demand_chasing_model():
    rng = np.random.RandomState(0)
    T = 120
    demand = np.cumsum(rng.randn(T)) + 10
    tv = np.r_[np.zeros(2), demand[:-2]] * 2 + rng.randn(T)  # spend chases demand
    search = rng.randn(T) * 3 + 20  # exogenous
    y = demand + 0.5 * tv + 0.3 * search + rng.randn(T)

    class M_:
        channel_names = ["TV", "Search"]
        has_geo = False
        n_geos = 1
        X_media_raw = np.column_stack([tv, search])
        y_raw = y

    return M_()


class TestEndogeneity:
    def test_flags_demand_chasing_channel(self):
        d = endogeneity_diagnostic(_demand_chasing_model())
        assert d["available"] is True
        assert "TV" in d["flagged"]  # spend chases demand
        assert "Search" not in d["flagged"]  # exogenous
        by = {r["channel"]: r for r in d["channels"]}
        assert by["TV"]["endogenous"] is True
        assert "experiment" in d["assumption"].lower()

    def test_shape_mismatch_returns_unavailable(self):
        class Bad:
            channel_names = ["TV"]
            X_media_raw = np.zeros((5, 2))  # 2 cols != 1 channel
            y_raw = np.zeros(5)

        assert endogeneity_diagnostic(Bad())["available"] is False

    def test_op_payload(self):
        res = M.OPS["endogeneity"](_demand_chasing_model())
        assert res["error"] is None
        assert "endogeneity" in res["dashboard"]
        assert res["tables"]
        assert "demand-chasing" in res["content"]

    def test_op_in_registry(self):
        assert "endogeneity" in M.OPS


# ---------------------------------------------------------------------------
# sign-off store + hash chain
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


class TestSignoffStore:
    def test_record_and_verify_chain(self, store):
        pid = store.create_project("P")["project_id"]
        r1 = store.record_signoff(
            pid, "Alice", role="owner", note="v1", assumptions=[{"a": 1}]
        )
        r2 = store.record_signoff(
            pid, "Bob", role="analyst", note="v2", assumptions=[{"a": 2}]
        )
        assert r2["prev_hash"] == r1["hash"]  # chained
        assert [s["approver"] for s in store.list_signoffs(pid)] == ["Bob", "Alice"]
        assert store.verify_signoff_chain(pid) == {"intact": True, "n": 2}

    def test_tamper_is_detected(self, store):
        pid = store.create_project("P")["project_id"]
        store.record_signoff(pid, "Alice", note="approved", assumptions=[{"a": 1}])
        store.record_signoff(pid, "Bob", note="approved", assumptions=[{"a": 2}])
        with store._conn() as c:
            c.execute(
                "UPDATE signoffs SET note = ? WHERE approver = ?", ("TAMPERED", "Alice")
            )
        v = store.verify_signoff_chain(pid)
        assert v["intact"] is False and v["broken_at"]

    def test_empty_chain_intact(self, store):
        pid = store.create_project("P")["project_id"]
        assert store.verify_signoff_chain(pid) == {"intact": True, "n": 0}


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


class TestSignoffEndpoints:
    def test_post_and_list(self, client, project):
        r = client.post(
            f"/projects/{project}/signoff",
            json={"approver": "Dana", "role": "owner", "note": "priors approved"},
        )
        assert r.status_code == 201, r.text
        rec = r.json()
        assert rec["approver"] == "Dana" and rec["hash"]

        body = client.get(f"/projects/{project}/signoffs").json()
        assert len(body["signoffs"]) == 1
        assert body["verification"]["intact"] is True

    def test_missing_approver_400(self, client, project):
        r = client.post(f"/projects/{project}/signoff", json={"approver": "  "})
        assert r.status_code == 400

    def test_404_unknown_project(self, client):
        assert client.get("/projects/nope/signoffs").status_code == 404
        assert (
            client.post("/projects/nope/signoff", json={"approver": "X"}).status_code
            == 404
        )


# ---------------------------------------------------------------------------
# prefit readout surfacing
# ---------------------------------------------------------------------------
def test_prefit_readout_renders_endogeneity_and_signoff():
    from mmm_framework.reporting.prefit import PrefitReadoutGenerator

    facts = {
        "meta": {
            "channels": ["TV", "Search"],
            "controls": [],
            "n_obs": 120,
            "kpi": "Sales",
        },
        "assumptions": [],
        "priors": [],
        "ppc": None,
        "curves": {},
        "components": {},
        "estimands": {},
        "densities": [],
        "sbc": None,
        "revisions": [],
        "endogeneity": endogeneity_diagnostic(_demand_chasing_model()),
        "signoffs": [
            {
                "approver": "Alice",
                "role": "owner",
                "note": "approved v1",
                "created_at": 1783000000.0,
                "hash": "abc123def456789",
            }
        ],
    }
    html = PrefitReadoutGenerator(facts=facts).generate_report()
    assert "Endogeneity screen" in html
    assert "demand-chasing" in html
    assert "exogenous" in html and "experiment" in html
    assert "Sign-off audit trail" in html and "Alice" in html
    assert "abc123def456" in html  # audit hash prefix

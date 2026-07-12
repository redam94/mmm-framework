"""Spec-curve Oracle op + async endpoint + pre-registration (issue #118).

The spec-curve ENGINE (validation/spec_curve.py) shipped in #103; this covers the
surfacing: the model-op payload/verdict, the SpecSet pre-registration to the
assumption log, and the non-blocking endpoint. The heavy multi-fit sweep is
stubbed so these stay fast.
"""

from __future__ import annotations

import pytest

from mmm_framework.agents import model_ops as M
from mmm_framework.validation.spec_curve import SpecCurveResult


def _canned() -> SpecCurveResult:
    return SpecCurveResult(
        channels=["TV", "Search"],
        specs=["geometric×hill", "weibull×hill"],
        primary="geometric×hill",
        hdi_prob=0.94,
        fits=[],
        weights={"geometric×hill": 0.6, "weibull×hill": 0.4},
        bma={
            "TV": {"mean": 2.1, "lower": 1.5, "upper": 2.7},
            "Search": {"mean": 0.8, "lower": 0.4, "upper": 1.2},
        },
        robustness={
            # robust: sign holds, tight spread
            "TV": {
                "min": 1.9,
                "max": 2.3,
                "range": 0.4,
                "spread_pct": 18.0,
                "sign_stable": True,
                "profitable_weight": 1.0,
                "primary": 2.1,
                "n_specs": 2,
            },
            # spec-fragile: sign flips across specs
            "Search": {
                "min": 0.6,
                "max": 1.4,
                "range": 0.8,
                "spread_pct": 80.0,
                "sign_stable": False,
                "profitable_weight": 0.4,
                "primary": 0.8,
                "n_specs": 2,
            },
        },
    )


class TestSpecCurveOp:
    def test_guards_missing_inputs(self):
        res = M.OPS["spec_curve"](object())
        assert res["error"] and res["content"] is None and res["dashboard"] == {}

    def test_renders_payload_and_verdicts(self, monkeypatch):
        monkeypatch.setattr(
            "mmm_framework.validation.spec_curve.run_spec_curve",
            lambda *a, **k: _canned(),
        )
        res = M.OPS["spec_curve"](
            object(),
            base_spec={"kpi": "s", "media_channels": [{"name": "TV"}]},
            dataset_path="x.csv",
        )
        assert res["error"] is None
        payload = res["dashboard"]["spec_curve"]
        assert payload["primary"] == "geometric×hill"
        assert res["tables"]
        by = {r["channel"]: r for r in res["tables"][0]["rows"]}
        assert by["TV"]["verdict"] == "robust"
        assert by["Search"]["verdict"] == "spec-fragile"
        # spec-fragile channel called out in the markdown
        assert "Spec-fragile" in res["content"] and "Search" in res["content"]

    def test_verdict_helper(self):
        assert (
            M._spec_curve_verdict({"sign_stable": True, "spread_pct": 20}) == "robust"
        )
        assert (
            M._spec_curve_verdict({"sign_stable": True, "spread_pct": 90})
            == "spec-fragile"
        )
        assert (
            M._spec_curve_verdict({"sign_stable": False, "spread_pct": 5})
            == "spec-fragile"
        )


def test_op_in_registry():
    assert "spec_curve" in M.OPS and getattr(
        M.OPS["spec_curve"], "allow_unfitted", False
    )


# ---------------------------------------------------------------------------
# endpoint (TestClient) — POST spawns a job + pre-registers the spec set
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


class TestEndpoint:
    def test_404_when_no_fitted_run(self, client, project):
        r = client.post(f"/projects/{project}/spec-curve", json={})
        assert r.status_code == 404

    def test_post_spawns_job_and_preregisters(self, client, project, monkeypatch):
        # a model_run with a spec + dataset_path is enough for latest_model_run_payload
        from mmm_framework.api import sessions as S

        tid = S.create_session("s", project_id=project)["thread_id"]
        S.add_artifact(
            tid,
            "model_run",
            {
                "run_id": "r1",
                "run_name": "r1",
                "spec": {"kpi": "Sales", "media_channels": [{"name": "TV"}]},
                "dataset_path": "/tmp/does_not_matter.csv",
                "model_path": "/tmp/nope",
            },
        )
        r = client.post(
            f"/projects/{project}/spec-curve",
            json={"rationale": "adstock × saturation forms", "max_draws": 50},
        )
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "pending" and body["n_specs"] >= 1
        # the declared spec set is pre-registered to the assumption log BEFORE fitting
        assumptions = S.list_assumptions(f"__speccurve__{project}")
        keys = {a["key"] for a in assumptions}
        assert "spec_curve_set" in keys

        # the job artifact is pollable (it may still be pending/errored — we only
        # assert the poll endpoint resolves it, not that the sweep finished)
        got = client.get(f"/projects/{project}/spec-curve/{body['job_id']}")
        assert got.status_code == 200
        assert "status" in got.json()

    def test_poll_404_unknown_job(self, client, project):
        assert client.get(f"/projects/{project}/spec-curve/nope").status_code == 404

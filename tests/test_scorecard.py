"""Recommendation scorecard — predicted vs realized (issue #109)."""

from __future__ import annotations

import pytest

from mmm_framework.api import scorecard as SC


def _estimands(rows_by_run):
    return {
        "groups": [
            {
                "estimand": "contribution_roi",
                "kpi": "sales",
                "models": [
                    {"run_id": rid, "created_at": ca, "rows": rows}
                    for rid, ca, rows in rows_by_run
                ],
            }
        ]
    }


_EST = _estimands(
    [
        (
            "run_A",
            100.0,
            [
                {
                    "channel": "TV",
                    "mean": 2.0,
                    "lower": 1.4,
                    "upper": 2.6,
                    "status": "ok",
                },
                {
                    "channel": "Search",
                    "mean": 4.0,
                    "lower": 3.0,
                    "upper": 5.0,
                    "status": "ok",
                },
            ],
        )
    ]
)


class TestJoin:
    def test_predicted_vs_realized_and_calibration(self):
        exps = [
            {
                "id": "e1",
                "channel": "TV",
                "value": 2.2,
                "se": 0.2,
                "estimand": "roas",
                "status": "calibrated",
                "recommending_run_id": "run_A",
                "end_date": "2026-03-01",
            },
            {
                "id": "e2",
                "channel": "Search",
                "value": 9.0,
                "se": 0.3,
                "estimand": "roas",
                "status": "completed",
                "recommending_run_id": "run_A",
                "end_date": "2026-04-01",
            },
        ]
        out = SC.project_scorecard_rows(_EST, exps)
        by = {r["channel"]: r for r in out["rows"]}
        assert by["TV"]["in_interval"] is True  # 2.2 in [1.4, 2.6]
        assert by["TV"]["error"] == pytest.approx(0.2)
        assert by["Search"]["in_interval"] is False  # 9.0 not in [3, 5]
        assert out["calibration"] == {"n_with_interval": 2, "hits": 1, "coverage": 0.5}
        assert out["n_recommendations"] == 2
        # newest realized first
        assert [r["channel"] for r in out["rows"]] == ["Search", "TV"]

    def test_no_prediction_row(self):
        exps = [
            {
                "id": "e3",
                "channel": "Radio",
                "value": 1.5,
                "se": 0.1,
                "estimand": "roas",
                "status": "calibrated",
                "end_date": "2026-02-01",
            },
        ]
        out = SC.project_scorecard_rows(_EST, exps)
        r = out["rows"][0]
        assert r["channel"] == "Radio" and r["predicted"] is None
        assert r["in_interval"] is None
        assert out["calibration"]["n_with_interval"] == 0

    def test_draft_experiments_dropped(self):
        exps = [
            {
                "id": "d",
                "channel": "TV",
                "value": 5.0,
                "se": 0.1,
                "estimand": "roas",
                "status": "draft",
            },
        ]
        assert SC.project_scorecard_rows(_EST, exps)["n_recommendations"] == 0

    def test_falls_back_to_latest_run_without_recommending_id(self):
        est = _estimands(
            [
                (
                    "old",
                    100.0,
                    [
                        {
                            "channel": "TV",
                            "mean": 1.0,
                            "lower": 0.5,
                            "upper": 1.5,
                            "status": "ok",
                        }
                    ],
                ),
                (
                    "new",
                    200.0,
                    [
                        {
                            "channel": "TV",
                            "mean": 2.0,
                            "lower": 1.5,
                            "upper": 2.5,
                            "status": "ok",
                        }
                    ],
                ),
            ]
        )
        exps = [
            {
                "id": "e",
                "channel": "TV",
                "value": 2.1,
                "se": 0.1,
                "estimand": "roas",
                "status": "calibrated",
                "end_date": "2026-05-01",
            }
        ]
        out = SC.project_scorecard_rows(est, exps)
        # newest run is the prediction (mean 2.0, interval [1.5, 2.5] → hit)
        assert out["rows"][0]["run_id"] == "new"
        assert out["rows"][0]["predicted"] == 2.0
        assert out["rows"][0]["in_interval"] is True


# ---------------------------------------------------------------------------
# endpoint (TestClient)
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


def test_endpoint_joins_store(client, project):
    from mmm_framework.api import sessions as S

    tid = S.create_session("s", project_id=project)["thread_id"]
    S.add_artifact(
        tid,
        "model_run",
        {
            "run_id": "run_A",
            "kpi": "sales",
            "channels": ["TV"],
            "model_kind": "mmm",
            "estimands": [
                {
                    "estimand": "contribution_roi",
                    "channel": "TV",
                    "kind": "roi",
                    "status": "ok",
                    "mean": 2.0,
                    "hdi_low": 1.4,
                    "hdi_high": 2.6,
                    "units": "ROI",
                }
            ],
        },
    )
    # a completed experiment on TV that came true (realized 2.2 in [1.4, 2.6])
    S.upsert_experiment(
        project_id=project,
        channel="TV",
        status="completed",
        estimand="roas",
        value=2.2,
        se=0.2,
        design_type="geo_holdout",
        end_date="2026-03-01",
    )
    out = client.get(f"/projects/{project}/scorecard").json()
    assert out["n_recommendations"] == 1
    assert out["rows"][0]["in_interval"] is True
    assert out["calibration"]["coverage"] == 1.0


def test_endpoint_404_unknown_project(client):
    assert client.get("/projects/nope/scorecard").status_code == 404

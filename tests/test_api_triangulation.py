"""Server-side triangulation join + endpoint (issue #119).

Covers the pure join (persisted MMM contribution_roi estimands × registry
experiment readouts), the calibrated-over-completed per-channel dedup, and the
viewer-gated ``GET /projects/{id}/triangulation`` endpoint — all without loading
a model.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from mmm_framework.api import triangulation as TRI


# ---------------------------------------------------------------------------
# pure join
# ---------------------------------------------------------------------------
def _estimands(*, kpi="sales", run_id="r2", rows=None):
    return {
        "groups": [
            {
                "estimand": "contribution_roi",
                "kpi": kpi,
                "models": [
                    {
                        "run_id": run_id,
                        "created_at": 200.0,
                        "rows": rows
                        or [
                            {
                                "channel": "TV",
                                "mean": 2.1,
                                "lower": 1.6,
                                "upper": 2.7,
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
                    }
                ],
            }
        ]
    }


def test_pure_join_reconciles_mmm_and_experiments():
    out = TRI.project_triangulation_sources(
        _estimands(),
        [
            {
                "channel": "TV",
                "value": 2.0,
                "se": 0.15,
                "estimand": "roas",
                "status": "calibrated",
            }
        ],
        platform={"Search": {"value": 9.0}},
    )
    assert out["kpi"] == "sales" and out["run_id"] == "r2"
    assert out["sources_available"] == {"mmm": 2, "experiment": 1, "platform": 1}
    by = {c["channel"]: c for c in out["channels"]}
    assert by["TV"]["agreement"] == "convergent"
    assert by["Search"]["agreement"] == "platform-inflated"


def test_pure_join_drops_non_readout_statuses():
    out = TRI.project_triangulation_sources(
        _estimands(),
        [
            {
                "channel": "TV",
                "value": 5.0,
                "se": 0.1,
                "estimand": "roas",
                "status": "draft",
            },
            {
                "channel": "TV",
                "value": 2.0,
                "se": 0.1,
                "estimand": "roas",
                "status": "calibrated",
            },
        ],
    )
    assert out["sources_available"]["experiment"] == 1  # draft dropped
    tv = next(c for c in out["channels"] if c["channel"] == "TV")
    exp = next(s for s in tv["sources"] if s["source"] == "experiment")
    assert exp["value"] == 2.0


def test_pure_join_prefers_calibrated_over_completed_per_channel():
    # experiments arrive calibrated-first (as build_project_triangulation orders
    # them); the first per channel wins, so the calibrated readout is used.
    out = TRI.project_triangulation_sources(
        _estimands(),
        [
            {
                "channel": "TV",
                "value": 2.0,
                "se": 0.1,
                "estimand": "roas",
                "status": "calibrated",
            },
            {
                "channel": "TV",
                "value": 3.3,
                "se": 0.1,
                "estimand": "roas",
                "status": "completed",
            },
        ],
    )
    assert out["sources_available"]["experiment"] == 1
    tv = next(c for c in out["channels"] if c["channel"] == "TV")
    exp = next(s for s in tv["sources"] if s["source"] == "experiment")
    assert exp["value"] == 2.0  # calibrated, not the later completed 3.3


def test_pure_join_picks_run_id():
    est = {
        "groups": [
            {
                "estimand": "contribution_roi",
                "kpi": "sales",
                "models": [
                    {
                        "run_id": "new",
                        "created_at": 300.0,
                        "rows": [
                            {
                                "channel": "TV",
                                "mean": 9.9,
                                "lower": 9,
                                "upper": 10,
                                "status": "ok",
                            }
                        ],
                    },
                    {
                        "run_id": "old",
                        "created_at": 100.0,
                        "rows": [
                            {
                                "channel": "TV",
                                "mean": 2.0,
                                "lower": 1.5,
                                "upper": 2.5,
                                "status": "ok",
                            }
                        ],
                    },
                ],
            }
        ]
    }
    out = TRI.project_triangulation_sources(est, [], run_id="old")
    assert out["run_id"] == "old"
    tv = next(c for c in out["channels"] if c["channel"] == "TV")
    assert next(s for s in tv["sources"] if s["source"] == "mmm")["value"] == 2.0


def test_pure_join_no_contribution_roi_group():
    out = TRI.project_triangulation_sources({"groups": []}, [])
    assert out["channels"] == []
    assert out["sources_available"]["mmm"] == 0


# ---------------------------------------------------------------------------
# store-backed build + endpoint
# ---------------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _seed_model_run(store, pid, run_id, kpi, channels, rows):
    tid = store.create_session("s", project_id=pid)["thread_id"]
    store.add_artifact(
        tid,
        "model_run",
        {
            "run_id": run_id,
            "run_name": run_id,
            "kpi": kpi,
            "channels": channels,
            "model_kind": "mmm",
            "estimands": rows,
            "spec": {"kpi": kpi},
        },
    )


def test_build_project_triangulation_reads_store(store):
    pid = store.create_project("P")["project_id"]
    _seed_model_run(
        store,
        pid,
        "r1",
        "sales",
        ["TV", "Search"],
        [
            {
                "estimand": "contribution_roi",
                "channel": "TV",
                "kind": "roi",
                "status": "ok",
                "mean": 2.1,
                "hdi_low": 1.6,
                "hdi_high": 2.7,
                "units": "ROI",
            },
            {
                "estimand": "contribution_roi",
                "channel": "Search",
                "kind": "roi",
                "status": "ok",
                "mean": 4.0,
                "hdi_low": 3.0,
                "hdi_high": 5.0,
                "units": "ROI",
            },
        ],
    )
    # a completed experiment on TV that agrees with the MMM
    store.upsert_experiment(
        project_id=pid,
        channel="TV",
        status="completed",
        estimand="roas",
        value=2.0,
        se=0.15,
        design_type="geo_holdout",
    )
    out = TRI.build_project_triangulation(pid)
    by = {c["channel"]: c for c in out["channels"]}
    assert by["TV"]["agreement"] == "convergent"
    assert by["Search"]["agreement"] == "single-source"
    assert out["sources_available"]["experiment"] == 1


@pytest.mark.asyncio
async def test_endpoint_returns_panel_and_404(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    _seed_model_run(
        store,
        pid,
        "r1",
        "sales",
        ["TV"],
        [
            {
                "estimand": "contribution_roi",
                "channel": "TV",
                "kind": "roi",
                "status": "ok",
                "mean": 2.1,
                "hdi_low": 1.6,
                "hdi_high": 2.7,
                "units": "ROI",
            },
        ],
    )
    resp = await M.project_triangulation_endpoint(pid)
    body = json.loads(resp.body)
    assert body["kpi"] == "sales"
    assert body["channels"][0]["channel"] == "TV"

    with pytest.raises(HTTPException) as ei:
        await M.project_triangulation_endpoint("does-not-exist")
    assert ei.value.status_code == 404

"""Direct-call tests for the experiment lifecycle endpoints in api/main.py:
upsert with lifecycle payloads, single-record GET, the transition endpoint's
409-on-illegal-move contract, and list filtering."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


def _body(resp) -> dict:
    return json.loads(resp.body)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return M


@pytest.mark.asyncio
async def test_lifecycle_via_endpoints(api):
    created = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(
                channel="TV",
                project_id="p1",
                status="draft",
                recommending_run_id="run_1",
                design={"design_type": "geo_holdout", "duration_weeks": 8},
                priority={"eig": 0.4, "quadrant": "test_now"},
            )
        )
    )
    assert created["status"] == "draft"
    assert created["design"]["design_type"] == "geo_holdout"
    eid = created["id"]

    got = _body(await api.get_experiment_endpoint(eid))
    assert got["recommending_run_id"] == "run_1"

    planned = _body(
        await api.transition_experiment_endpoint(
            eid, api.ExperimentTransitionRequest(status="planned")
        )
    )
    assert planned["status"] == "planned" and planned["preregistered_at"]

    # illegal move -> 409 (state conflict, not bad input)
    with pytest.raises(HTTPException) as exc:
        await api.transition_experiment_endpoint(
            eid, api.ExperimentTransitionRequest(status="calibrated")
        )
    assert exc.value.status_code == 409

    # bad status -> 400
    with pytest.raises(HTTPException) as exc:
        await api.transition_experiment_endpoint(
            eid, api.ExperimentTransitionRequest(status="exploded")
        )
    assert exc.value.status_code == 400

    # unknown id -> 404
    with pytest.raises(HTTPException) as exc:
        await api.transition_experiment_endpoint(
            "nope", api.ExperimentTransitionRequest(status="planned")
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_filters_by_channel_and_status(api):
    for ch, status in (("TV", "draft"), ("Digital", "running"), ("TV", "running")):
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(channel=ch, project_id="p1", status=status)
        )
    out = _body(
        await api.list_experiments_endpoint(
            project_id="p1", status="running", channel="TV"
        )
    )
    assert out["total"] == 1
    assert out["experiments"][0]["channel"] == "TV"


@pytest.mark.asyncio
async def test_get_unknown_experiment_404(api):
    with pytest.raises(HTTPException) as exc:
        await api.get_experiment_endpoint("missing")
    assert exc.value.status_code == 404

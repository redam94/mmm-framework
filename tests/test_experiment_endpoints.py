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
async def test_create_in_calibrated_is_409(api):
    # calibration happens via the transition endpoint (fit close-out), never
    # on create — a state conflict, not bad input
    with pytest.raises(HTTPException) as exc:
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(
                channel="TV", project_id="p1", status="calibrated"
            )
        )
    assert exc.value.status_code == 409
    assert "Illegal status" in exc.value.detail


@pytest.mark.asyncio
async def test_upsert_update_multihop_forward_is_legal(api):
    # draft -> completed walks the legal chain and backfills the skipped hops
    created = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(channel="TV", project_id="p1", status="draft")
        )
    )
    updated = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(
                id=created["id"], status="completed", value=1.2, se=0.3
            )
        )
    )
    assert updated["status"] == "completed" and updated["value"] == 1.2
    assert [h["status"] for h in updated["status_history"]] == [
        "draft",
        "planned",
        "running",
        "completed",
    ]


@pytest.mark.asyncio
async def test_upsert_update_across_illegal_edge_is_409(api):
    # 'calibrated' is fit-close-out-only: unreachable via upsert even from
    # 'completed', where the transition endpoint's edge is legal.
    created = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(
                channel="TV", project_id="p1", status="completed", value=2.0, se=0.2
            )
        )
    )
    with pytest.raises(HTTPException) as exc:
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(id=created["id"], status="calibrated")
        )
    assert exc.value.status_code == 409
    assert "Illegal transition" in exc.value.detail
    # the failed update left the row untouched
    got = _body(await api.get_experiment_endpoint(created["id"]))
    assert got["status"] == "completed"


@pytest.mark.asyncio
async def test_upsert_edit_of_calibrated_row_is_409(api):
    # REST clients cannot silently mutate a calibrated experiment's
    # likelihood-feeding fields — the endpoint never exposes
    # allow_calibrated_edit, so the store's guard maps to 409.
    from mmm_framework.api import sessions as S

    created = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(
                channel="TV", project_id="p1", status="completed", value=2.0, se=0.2
            )
        )
    )
    S.transition_experiment(created["id"], "calibrated")
    with pytest.raises(HTTPException) as exc:
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(id=created["id"], value=9.9)
        )
    assert exc.value.status_code == 409
    assert "Illegal update" in exc.value.detail
    got = _body(await api.get_experiment_endpoint(created["id"]))
    assert got["value"] == 2.0 and got["status"] == "calibrated"
    # non-measurement fields stay editable over REST
    ok = _body(
        await api.upsert_experiment_endpoint(
            api.ExperimentUpsertRequest(id=created["id"], notes="context")
        )
    )
    assert ok["notes"] == "context"


@pytest.mark.asyncio
async def test_get_unknown_experiment_404(api):
    with pytest.raises(HTTPException) as exc:
        await api.get_experiment_endpoint("missing")
    assert exc.value.status_code == 404

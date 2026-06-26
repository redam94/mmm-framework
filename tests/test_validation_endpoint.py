"""Non-blocking model-validation runner endpoint (Phase 2).

Fast: endpoint wiring (404/400/202 + job artifact + run-missing error path).
Slow: end-to-end — a real fit saved to the store, the validation job loads it,
runs the battery, and persists content + content-addressed table/plot refs.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _body(resp) -> dict:
    return json.loads(resp.body)


@pytest.mark.asyncio
async def test_validate_endpoint_wiring(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]

    # unknown project -> 404
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await M.start_model_validation("nope", M.ValidationRunRequest(check="validate"))
    assert ei.value.status_code == 404

    # unknown check -> 400
    with pytest.raises(HTTPException) as ei:
        await M.start_model_validation(pid, M.ValidationRunRequest(check="bogus"))
    assert ei.value.status_code == 400

    # valid -> 202 + pending job artifact
    resp = await M.start_model_validation(pid, M.ValidationRunRequest(check="ppc"))
    assert resp.status_code == 202
    job_id = _body(resp)["job_id"]
    art = store.get_artifact(job_id)
    assert art["payload"]["status"] == "pending"
    assert art["payload"]["check"] == "ppc"


@pytest.mark.asyncio
async def test_validation_job_errors_without_model(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    tid = f"__valjobs__{pid}"
    job = store.add_artifact(
        tid,
        "model_validation",
        {"status": "pending", "project_id": pid, "check": "validate"},
    )
    # run=None => no saved model => graceful error, never raises
    await M._run_validation_job(job["id"], tid, None, "validate_model", {})
    payload = store.get_artifact(job["id"])["payload"]
    assert payload["status"] == "error"
    assert "No saved model" in payload["error"]


@pytest.mark.asyncio
async def test_get_validation_job_404_cross_project(store):
    from fastapi import HTTPException

    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    other = store.create_project("Q")["project_id"]
    tid = f"__valjobs__{pid}"
    job = store.add_artifact(
        tid, "model_validation", {"status": "done", "project_id": pid, "check": "ppc"}
    )
    # right project resolves
    resp = await M.get_model_validation(pid, job["id"])
    assert _body(resp)["check"] == "ppc"
    # cross-project access -> 404
    with pytest.raises(HTTPException) as ei:
        await M.get_model_validation(other, job["id"])
    assert ei.value.status_code == 404


@pytest.mark.slow
@pytest.mark.asyncio
async def test_validation_job_end_to_end(store, tmp_path, monkeypatch):
    """A real fit in the store -> the validation job loads it, runs the battery,
    and persists content + table/plot refs that resolve in the plot/table store."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.agents import workspace as ws
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.api import main as M
    from mmm_framework.synth import generate_mff

    df, _ = generate_mff("realistic", seed=5, n_weeks=120)
    path = str(tmp_path / "nat.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": n} for n in ["TV", "Search", "Social"]],
        "control_variables": [],
        "inference": {"draws": 60, "tune": 60, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    _, _, info = build_and_fit(spec, path)
    mr = info["model_run"]

    pid = store.create_project("P")["project_id"]
    sess = store.create_session("s", project_id=pid)
    store.add_artifact(sess["thread_id"], "model_run", mr)

    tid = f"__valjobs__{pid}"
    job = store.add_artifact(
        tid,
        "model_validation",
        {"status": "pending", "project_id": pid, "check": "validate"},
    )
    from mmm_framework.api.history import latest_model_run_payload

    run = latest_model_run_payload(pid)
    await M._run_validation_job(job["id"], tid, run, "validate_model", {})

    payload = store.get_artifact(job["id"])["payload"]
    assert payload["status"] == "done", payload.get("error")
    result = payload["result"]
    assert "Model validation battery" in result["content"]
    assert result["tables"] and "id" in result["tables"][0]
    # a PPC plot ref should resolve on disk
    if result["plots"]:
        assert ws.plot_path(result["plots"][0]["id"]) is not None

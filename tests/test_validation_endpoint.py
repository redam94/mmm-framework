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


@pytest.mark.asyncio
async def test_validations_history_lists_jobs_and_chat_runs(store):
    """`GET /projects/{id}/validations` returns every persisted run, newest
    first — UI jobs and chat-persisted checks alike — so the Validation tab
    keeps track across reloads."""
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]

    # unknown project -> 404
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await M.list_model_validations("nope")
    assert ei.value.status_code == 404

    # empty history
    assert _body(await M.list_model_validations(pid)) == {
        "validations": [],
        "total": 0,
    }

    # a UI job (pending)
    resp = await M.start_model_validation(pid, M.ValidationRunRequest(check="ppc"))
    job_id = _body(resp)["job_id"]

    # a chat-persisted run (same artifact shape the chat tools write)
    store.add_artifact(
        f"__valjobs__{pid}",
        "model_validation",
        {
            "status": "done",
            "project_id": pid,
            "check": "sbc",
            "source": "chat",
            "thread_id": "t1",
            "result": {"content": "# ok", "tables": [], "plots": []},
            "error": None,
        },
    )

    body = _body(await M.list_model_validations(pid))
    assert body["total"] == 2
    checks = {r["check"]: r for r in body["validations"]}
    assert checks["ppc"]["status"] == "pending"
    assert checks["ppc"]["source"] == "job"  # legacy jobs default to "job"
    assert checks["sbc"]["source"] == "chat"
    assert checks["sbc"]["job_id"] != job_id
    # newest first
    created = [r["created_at"] for r in body["validations"]]
    assert created == sorted(created, reverse=True)

    # a past run's full result is retrievable through the existing poll endpoint
    full = _body(await M.get_model_validation(pid, checks["sbc"]["job_id"]))
    assert full["result"]["content"] == "# ok"


@pytest.mark.asyncio
async def test_validations_history_scoped_by_thread_id(store):
    """`?thread_id=` scopes the history to the session that launched each run —
    the Validation tab shows only the CURRENT session's validations. UI jobs
    stamp the launching session from the request body; unstamped legacy rows
    are excluded from a scoped listing."""
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]

    # UI job launched from session t1 (thread_id stamped from the body)
    resp = await M.start_model_validation(
        pid, M.ValidationRunRequest(check="ppc", thread_id="t1")
    )
    t1_job = _body(resp)["job_id"]

    # chat-persisted run from another session t2
    store.add_artifact(
        f"__valjobs__{pid}",
        "model_validation",
        {
            "status": "done",
            "project_id": pid,
            "check": "sbc",
            "source": "chat",
            "thread_id": "t2",
            "result": {"content": "# ok", "tables": [], "plots": []},
            "error": None,
        },
    )
    # legacy row with no thread_id (pre-stamping)
    store.add_artifact(
        f"__valjobs__{pid}",
        "model_validation",
        {"status": "done", "project_id": pid, "check": "residuals"},
    )

    # unscoped: everything, and rows expose their thread_id
    body = _body(await M.list_model_validations(pid))
    assert body["total"] == 3
    by_check = {r["check"]: r for r in body["validations"]}
    assert by_check["ppc"]["thread_id"] == "t1"
    assert by_check["sbc"]["thread_id"] == "t2"
    assert by_check["residuals"]["thread_id"] is None

    # scoped to t1: only t1's run (legacy unstamped rows excluded too)
    body = _body(await M.list_model_validations(pid, thread_id="t1"))
    assert body["total"] == 1
    assert body["validations"][0]["job_id"] == t1_job

    # scoped to an unknown session: empty
    assert _body(await M.list_model_validations(pid, thread_id="t3"))["total"] == 0


def test_persist_chat_validation_writes_project_history(store, monkeypatch):
    """The chat validation tools' persistence helper lands in the SAME
    `__valjobs__<project>` thread the UI jobs use (and silently no-ops for a
    session with no project)."""
    from mmm_framework.agents import runtime as R
    from mmm_framework.agents.tools import _persist_chat_validation

    pid = store.create_project("P")["project_id"]
    tid = store.create_session("s", project_id=pid)["thread_id"]

    tok = R.set_current_thread(tid)
    try:
        _persist_chat_validation(
            "validate", "## verdict", [{"id": "tbl1"}], [{"id": "plt1"}]
        )
    finally:
        R.current_thread_id.reset(tok)

    arts = store.list_artifacts(f"__valjobs__{pid}")
    assert len(arts) == 1
    p = arts[0]["payload"]
    assert p["check"] == "validate" and p["source"] == "chat"
    assert p["thread_id"] == tid and p["status"] == "done"
    assert p["result"]["content"] == "## verdict"
    assert p["result"]["tables"] == [{"id": "tbl1"}]
    assert p["result"]["plots"] == [{"id": "plt1"}]

    # no project on the session -> silent no-op
    tid2 = store.create_session("orphan")["thread_id"]
    tok = R.set_current_thread(tid2)
    try:
        _persist_chat_validation("validate", "x", [], [])
    finally:
        R.current_thread_id.reset(tok)
    assert len(store.list_artifacts(f"__valjobs__{pid}")) == 1

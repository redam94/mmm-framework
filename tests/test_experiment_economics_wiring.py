"""Wiring tests for the experiment-economics model op + the artifact-payload
update used by the async simulate job."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(scope="module")
def geo_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("geo_heterogeneous", seed=3, n_weeks=120)
    path = tmp_path_factory.mktemp("eco") / "geo.csv"
    df.to_csv(path, index=False)
    return str(path), key


def test_update_artifact_payload_roundtrip(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    art = S.add_artifact(
        "__simjobs__p1", "experiment_simulation", {"status": "pending"}
    )
    updated = S.update_artifact_payload(
        art["id"], {"status": "done", "result": {"x": 1}}
    )
    assert updated is not None
    got = S.get_artifact(art["id"])
    assert got["payload"]["status"] == "done"
    assert got["payload"]["result"] == {"x": 1}
    # unknown id -> None
    assert S.update_artifact_payload("nope", {"a": 1}) is None


def test_economics_op_prefit_is_json_safe(geo_csv):
    """allow_unfitted: with no model the op returns the design + the A/A·A/B
    methodology check (which needs no model), fully JSON-serializable."""
    from mmm_framework.agents import model_ops as mo

    path, key = geo_csv
    res = mo.experiment_economics(
        None,
        None,
        design_params={
            "dataset_path": path,
            "kpi": "Sales",
            "channel": key["channels"][0],
            "design_key": "geo_lift",
            "duration": 8,
            "design": "scaling",
            "intensity_pct": 50,
            "seed": 7,
        },
        run_simulation=True,
    )
    assert res["error"] is None
    eco = res["dashboard"]["experiment_economics"]
    assert eco["model_anchored"] is False
    assert eco["anchor"] is None
    assert eco["simulation"] is not None
    # entire payload must serialize (it crosses the artifact/JSON boundary)
    json.dumps(res["dashboard"], default=str)
    json.dumps(res.get("tables") or [], default=str)


def test_safe_json_dumps_strips_native_nan():
    """Review finding 1: native float('nan')/inf must not survive into the JSON
    (they would make JSONResponse(allow_nan=False) 500)."""
    from mmm_framework.api.main import safe_json_dumps

    s = safe_json_dumps(
        {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")}}
    )
    assert "NaN" not in s and "Infinity" not in s
    loaded = json.loads(s)  # strict parse must succeed
    assert loaded["a"] is None and loaded["b"][1] is None and loaded["c"]["d"] is None


def test_insufficient_windows_payload_renders_through_jsonresponse(geo_csv):
    """Review finding 1 end-to-end: a short window forces every estimator into
    the insufficient-windows (NaN) branch; the stored payload must still render
    through the GET endpoint's JSONResponse(safe_json_dumps_load(...))."""
    from fastapi.responses import JSONResponse

    from mmm_framework.agents import model_ops as mo
    from mmm_framework.api.main import safe_json_dumps_load

    path, key = geo_csv
    res = mo.experiment_economics(
        None,
        None,
        design_params={
            "dataset_path": path,
            "kpi": "Sales",
            "channel": key["channels"][0],
            "design_key": "geo_lift",
            "duration": 60,
            "design": "scaling",
            "intensity_pct": 50,
            "seed": 7,
        },
        run_simulation=True,
    )
    eco = res["dashboard"]["experiment_economics"]
    assert eco["simulation"] is not None  # ran, just insufficient windows
    # mimic the artifact storage roundtrip (json.dumps default=str keeps NaN)...
    stored = json.loads(json.dumps(eco, default=str))
    # ...then the GET path must construct without raising (render is in __init__)
    JSONResponse(content=safe_json_dumps_load(stored))


def test_economics_op_bad_params_returns_error():
    from mmm_framework.agents import model_ops as mo

    res = mo.experiment_economics(None, None, design_params={"channel": "TV"})
    assert res["error"] is not None
    assert "dataset_path" in res["error"]


async def _drain_job(M, pid, job_id, *, getter=None, tries=60):
    import asyncio
    import json as _j

    getter = getter or M.get_experiment_simulation
    for _ in range(tries):
        got = await getter(pid, job_id)
        payload = _j.loads(got.body)
        if payload["status"] in ("done", "error"):
            return payload
        await asyncio.sleep(0.05)
    return payload


@pytest.mark.asyncio
async def test_simulate_job_lifecycle_done_and_cross_project_404(
    tmp_path, monkeypatch, geo_csv
):
    import json as _j

    from mmm_framework.api import history as H
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    pid = S.create_project("p")["project_id"]
    path, key = geo_csv
    run = {
        "dataset_path": path,
        "kpi": "Sales",
        "run_name": "run_x",
        "channels": list(key["channels"]),
        "spec": {"kpi": "Sales"},
    }
    monkeypatch.setattr(H, "latest_model_run_payload", lambda p: run)
    # stub the heavy load+op so no fit is needed
    monkeypatch.setattr(
        M,
        "_load_and_run_op",
        lambda *a, **k: {
            "dashboard": {"experiment_economics": {"channel": "TV", "ok": True}}
        },
    )
    resp = await M.start_experiment_simulation(
        pid, M.ExperimentSimulateRequest(channel=key["channels"][0])
    )
    started = _j.loads(resp.body)
    assert started["status"] == "pending"
    job_id = started["job_id"]
    payload = await _drain_job(M, pid, job_id)
    assert payload["status"] == "done"
    assert payload["result"]["channel"] == "TV"
    # cross-project poll must 404 (server-minted, project-scoped job id)
    with pytest.raises(M.HTTPException):
        await M.get_experiment_simulation("someone-else", job_id)


@pytest.mark.asyncio
async def test_simulate_job_writes_error_when_no_saved_model(
    tmp_path, monkeypatch, geo_csv
):
    import json as _j

    from mmm_framework.api import history as H
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    pid = S.create_project("p")["project_id"]
    path, key = geo_csv
    # a run with dataset+kpi (so _design_inputs passes) but NO run_name
    run = {"dataset_path": path, "kpi": "Sales", "channels": list(key["channels"])}
    monkeypatch.setattr(H, "latest_model_run_payload", lambda p: run)
    resp = await M.start_experiment_simulation(
        pid, M.ExperimentSimulateRequest(channel=key["channels"][0])
    )
    job_id = _j.loads(resp.body)["job_id"]
    payload = await _drain_job(M, pid, job_id)
    assert payload["status"] == "error"
    assert payload["error"] and "model" in payload["error"].lower()


def test_optimizer_op_requires_model():
    from mmm_framework.agents import model_ops as mo

    res = mo.experiment_optimizer(
        None, None, dataset_path="x", kpi="Sales", channel="TV"
    )
    assert res["error"] == mo.NO_MODEL_MSG


@pytest.mark.asyncio
async def test_optimize_job_lifecycle_done_and_404(tmp_path, monkeypatch, geo_csv):
    import json as _j

    from mmm_framework.api import history as H
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    pid = S.create_project("p")["project_id"]
    path, key = geo_csv
    run = {
        "dataset_path": path,
        "kpi": "Sales",
        "run_name": "run_x",
        "channels": list(key["channels"]),
        "spec": {"kpi": "Sales"},
    }
    monkeypatch.setattr(H, "latest_model_run_payload", lambda p: run)
    # stub the heavy load+op so no fit is needed; mimic the optimizer dashboard key
    monkeypatch.setattr(
        M,
        "_load_and_run_op",
        lambda *a, **k: {
            "dashboard": {
                "experiment_optimization": {"channel": "TV", "n_candidates": 3}
            }
        },
    )
    resp = await M.start_experiment_optimization(
        pid, M.ExperimentOptimizeRequest(channel=key["channels"][0])
    )
    started = _j.loads(resp.body)
    assert started["status"] == "pending"
    payload = await _drain_job(
        M, pid, started["job_id"], getter=M.get_experiment_optimization
    )
    assert payload["status"] == "done"
    assert payload["result"]["n_candidates"] == 3
    with pytest.raises(M.HTTPException):
        await M.get_experiment_optimization("other-project", started["job_id"])


@pytest.mark.slow
def test_economics_op_full_model_signs_and_loopback(tmp_path):
    """End-to-end with a real (tiny) geo fit: holdout sign conventions hold and
    the EIG/EVOI loopback populates the channel's evoi from the incremental
    estimand."""
    import logging

    logging.disable(logging.CRITICAL)
    from mmm_framework.agents import model_ops as mo
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import dgp_geo
    from mmm_framework.synth.mff import geo_scenario_to_mff

    geos = ["North", "South", "East", "West", "G5", "G6", "G7", "G8"]
    sc = dgp_geo.build("geo_heterogeneous", seed=3, geos=geos, n_weeks=90)
    path = str(tmp_path / "geo.csv")
    geo_scenario_to_mff(sc).to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "kpi_level": "geo",
        "media_channels": [{"name": n} for n in ["TV", "Search", "Social", "Display"]],
        "control_variables": [],
        "inference": {"draws": 60, "tune": 60, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    mmm, results, _info = build_and_fit(spec, path)
    assert mmm.has_geo and mmm.n_geos == 8

    dp = {
        "dataset_path": path,
        "kpi": "Sales",
        "channel": "TV",
        "design_key": "geo_lift",
        "duration": 8,
        "design": "holdout",
        "intensity_pct": 50,
        "seed": 7,
    }
    res = mo.experiment_economics(
        mmm,
        results,
        design_params=dp,
        run_simulation=True,
        margin=0.1,
        kpi_kind="revenue",
        max_draws=60,
    )
    assert res["error"] is None
    eco = res["dashboard"]["experiment_economics"]
    oc = eco["opportunity_cost"]
    anc = eco["anchor"]
    assert oc["spend_delta"] < 0  # holdout saves spend
    assert oc["kpi_delta_median"] <= 0  # holdout loses KPI
    assert anc is not None and anc["verdict"] in (
        "powered",
        "underpowered",
        "overpowered",
        "inconclusive",
    )
    # loopback populated the channel's EIG/EVOI from the incremental estimand
    assert anc.get("evoi") is not None
    json.dumps(eco, default=str)


# ── Structural identification endpoint (non-blocking job) ────────────────────


def test_identify_op_requires_model():
    from mmm_framework.agents import model_ops as mo

    res = mo.identify_structural_parameters(
        None, None, dataset_path="x", kpi="Sales", channel="TV"
    )
    assert res["error"] == mo.NO_MODEL_MSG


@pytest.mark.asyncio
async def test_identify_job_lifecycle_done_and_404(tmp_path, monkeypatch, geo_csv):
    import json as _j

    from mmm_framework.api import history as H
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    pid = S.create_project("p")["project_id"]
    path, key = geo_csv
    run = {
        "dataset_path": path,
        "kpi": "Sales",
        "run_name": "run_x",
        "channels": list(key["channels"]),
        "spec": {"kpi": "Sales"},
    }
    monkeypatch.setattr(H, "latest_model_run_payload", lambda p: run)
    seen_kwargs: dict = {}

    def _fake_op(_tid, _rn, _spec, _dpath, op_name, op_kwargs):
        seen_kwargs.update(op_kwargs, __op__=op_name)
        return {
            "dashboard": {
                "structural_identification": {"channel": "TV", "n_levels": 3}
            }
        }

    monkeypatch.setattr(M, "_load_and_run_op", _fake_op)
    resp = await M.start_structural_identification(
        pid,
        M.ExperimentIdentifyRequest(
            channel=key["channels"][0], levels=[0.5, 1.0, 1.5], block_weeks=3
        ),
    )
    started = _j.loads(resp.body)
    assert started["status"] == "pending"
    payload = await _drain_job(
        M, pid, started["job_id"], getter=M.get_structural_identification
    )
    assert payload["status"] == "done"
    assert payload["result"]["n_levels"] == 3
    # the op saw the request's design knobs and the identify op name
    assert seen_kwargs["__op__"] == "identify_structural_parameters"
    assert seen_kwargs["levels"] == [0.5, 1.0, 1.5]
    assert seen_kwargs["block_weeks"] == 3
    assert seen_kwargs["dataset_path"] == path and seen_kwargs["kpi"] == "Sales"
    # cross-project poll must 404 (server-minted, project-scoped job id)
    with pytest.raises(M.HTTPException):
        await M.get_structural_identification("other-project", started["job_id"])


@pytest.mark.asyncio
async def test_identify_defaults_omit_optional_design_knobs(
    tmp_path, monkeypatch, geo_csv
):
    """No levels/block_weeks in the request ⇒ the op decides (its defaults are
    the documented 0.5/1/1.5 levels and the adstock cool-down block)."""
    import json as _j

    from mmm_framework.api import history as H
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    pid = S.create_project("p")["project_id"]
    path, key = geo_csv
    run = {
        "dataset_path": path,
        "kpi": "Sales",
        "run_name": "run_x",
        "channels": list(key["channels"]),
        "spec": {"kpi": "Sales"},
    }
    monkeypatch.setattr(H, "latest_model_run_payload", lambda p: run)
    seen_kwargs: dict = {}

    def _fake_op(_tid, _rn, _spec, _dpath, op_name, op_kwargs):
        seen_kwargs.update(op_kwargs)
        return {"dashboard": {"structural_identification": {"channel": "TV"}}}

    monkeypatch.setattr(M, "_load_and_run_op", _fake_op)
    resp = await M.start_structural_identification(
        pid, M.ExperimentIdentifyRequest(channel=key["channels"][0])
    )
    await _drain_job(
        M, pid, _j.loads(resp.body)["job_id"], getter=M.get_structural_identification
    )
    assert "levels" not in seen_kwargs and "block_weeks" not in seen_kwargs
    assert seen_kwargs["duration"] == 12

"""Continuous-learning REST endpoints (wiring §3.5): program CRUD shapes, the
sync design-wave endpoint, the non-blocking learning-fit job (mocked-fit fast
path + tiny-NUTS slow paths for rows-ingest and past-experiment import), and
the poll endpoint's project scoping.

Direct-call style (the ``test_validation_endpoint`` idiom): endpoint
coroutines are awaited on pytest-asyncio's loop, so ``_spawn_job_task``'s
``asyncio.create_task`` runs the job in the background while the test polls
the artifact with short sleeps."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from fastapi import HTTPException

#: SNAPSHOT top-level keys pinned in continuous-learning-wiring.md §3.1.
SNAPSHOT_KEYS = {
    "schema_version",
    "fitted_at",
    "evidence",
    "diagnostics",
    "recommendation",
    "recommendation_scaled",
    "allocation_sd",
    "funding",
    "regret",
    "gamma",
    "response_curves",
    "warnings",
}

TINY_FIT = {"num_warmup": 40, "num_samples": 40, "num_chains": 1}

CONFIG = {
    "channels": ["Chatter", "Pulse"],
    "center": {"Chatter": 140000, "Pulse": 140000},
    "budget": 280000,
    "value_per_unit": 5.0,
    "mode": "fixed",
    "kpi": "sales",
    "margin": 1.0,
    "population": 13,
    "wave_cost": 25000,
}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _body(resp) -> dict:
    return json.loads(resp.body)


async def _create_program(M, pid, config=None, name="Prog"):
    resp = await M.create_learning_program_endpoint(
        pid, M.LearningProgramCreateRequest(name=name, config=config or dict(CONFIG))
    )
    assert resp.status_code == 201
    return _body(resp)["program"]


async def _poll_to_terminal(store, job_id, timeout_s=600.0):
    waited = 0.0
    while waited < timeout_s:
        await asyncio.sleep(0.25)
        waited += 0.25
        payload = store.get_artifact(job_id)["payload"]
        if payload["status"] in ("done", "error"):
            return payload
    raise AssertionError(f"job {job_id} never reached a terminal status")


def _wave_rows(n_geo=12, n_periods=2, seed=0):
    """A tiny synthetic CCD-ish panel in DOLLARS (2 channels)."""
    rng = np.random.default_rng(seed)
    mults = [1.0, 1.6, 0.4, 0.0]
    rows = []
    for t in range(n_periods):
        for g in range(n_geo):
            m_c = mults[g % 4]
            m_p = mults[(g // 4) % 4]
            s_c, s_p = m_c, m_p  # scaled by the $140k center
            y = (
                5.0
                + 2.0 * (s_c / (s_c + 0.8))
                + 0.7 * (s_p / (s_p + 0.8))
                + rng.normal(0, 0.3)
            )
            rows.append(
                {
                    "geo": f"g{g:02d}",
                    "week": t,
                    "Chatter": 140000.0 * m_c,
                    "Pulse": 140000.0 * m_p,
                    "y": float(y),
                }
            )
    return rows


# ── wiring (fast) ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_rejects_rows_plus_csv_and_concurrent_fits(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    # rows AND csv_text in one body -> 400 (csv would silently drop the rows)
    with pytest.raises(HTTPException) as ei:
        await M.ingest_learning_wave_endpoint(
            pid,
            prog["id"],
            M.LearningWaveIngestRequest(
                rows=[{"geo": "g", "y": 1.0}], csv_text="geo,y\ng,1\n"
            ),
        )
    assert ei.value.status_code == 400
    assert "exactly one" in ei.value.detail

    # a pending/running fit job for the program blocks new submissions (409)
    store.add_artifact(
        f"__learnjobs__{pid}",
        "learning_fit",
        {
            "status": "pending",
            "project_id": pid,
            "program_id": prog["id"],
            "result": None,
            "error": None,
        },
    )
    with pytest.raises(HTTPException) as ei:
        await M.ingest_learning_wave_endpoint(
            pid,
            prog["id"],
            M.LearningWaveIngestRequest(rows=[{"geo": "g", "y": 1.0}]),
        )
    assert ei.value.status_code == 409
    with pytest.raises(HTTPException) as ei:
        await M.refit_learning_program_endpoint(pid, prog["id"], M.LearningFitRequest())
    assert ei.value.status_code == 409
    # delete is blocked while the fit runs, too
    with pytest.raises(HTTPException) as ei:
        await M.delete_learning_program_endpoint(pid, prog["id"])
    assert ei.value.status_code == 409

    # a DIFFERENT program in the same project is not blocked
    other_prog = await _create_program(M, pid, name="Other")
    resp = await M.refit_learning_program_endpoint(
        pid, other_prog["id"], M.LearningFitRequest()
    )
    assert resp.status_code == 202
    # (drain the spawned job: no evidence yet -> a clean error, not a hang)
    payload = await _poll_to_terminal(store, _body(resp)["job_id"], timeout_s=30)
    assert payload["status"] == "error"


@pytest.mark.asyncio
async def test_delete_program_reaps_state_dir(store):
    from pathlib import Path

    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)
    state_dir = Path(prog["state_path"]).parent
    assert state_dir.exists()

    resp = await M.delete_learning_program_endpoint(pid, prog["id"])
    assert _body(resp) == {"status": "ok"}
    assert store.get_learning_program(prog["id"]) is None
    # the on-disk state (client panel + posterior) is reaped, not orphaned
    assert not state_dir.exists()


@pytest.mark.asyncio
async def test_wave_job_skips_foreign_project_experiments(store, monkeypatch):
    """The ingest worker never folds another project's experiments in."""
    from mmm_framework.api import main as M
    from mmm_framework.continuous_learning import service as cl_service

    monkeypatch.setattr(
        cl_service, "fit_and_plan", lambda state, **kw: {"schema_version": 1}
    )
    pid = store.create_project("P")["project_id"]
    other = store.create_project("Q")["project_id"]
    prog = await _create_program(M, pid)
    foreign = store.upsert_experiment(
        project_id=other,
        channel="Chatter",
        status="completed",
        start_date="2026-01-05",
        end_date="2026-03-02",
        readout={
            "value": 12.0,
            "se": 2.0,
            "estimand": "contribution",
            "spend_per_period": 56000.0,
            "n_treated_units": 4,
        },
    )
    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(experiment_ids=[foreign["id"], "ghost"]),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"], timeout_s=30)
    assert payload["status"] == "error"
    assert "different project" in payload["error"]
    assert "not found in this project" in payload["error"]


@pytest.mark.asyncio
async def test_program_crud_shapes(store):
    from mmm_framework.api import main as M

    with pytest.raises(HTTPException) as ei:
        await _create_program(M, "nope")
    assert ei.value.status_code == 404

    pid = store.create_project("P")["project_id"]

    # bad config -> 400 with the service's message
    with pytest.raises(HTTPException) as ei:
        await M.create_learning_program_endpoint(
            pid,
            M.LearningProgramCreateRequest(name="bad", config={"channels": []}),
        )
    assert ei.value.status_code == 400
    assert "channels" in ei.value.detail

    prog = await _create_program(M, pid)
    # parsed row keys (never raw *_json), state file saved
    assert prog["channels"] == ["Chatter", "Pulse"]
    assert prog["config"]["budget"] == 280000
    assert prog["summary"] is None
    assert prog["state_path"] and prog["state_path"].endswith("state.npz")

    listed = _body(await M.list_learning_programs_endpoint(pid))
    assert [p["id"] for p in listed["programs"]] == [prog["id"]]

    detail = _body(await M.get_learning_program_endpoint(pid, prog["id"]))
    assert detail["program"]["id"] == prog["id"]
    assert detail["waves"] == []

    # cross-project detail -> 404
    other = store.create_project("Q")["project_id"]
    with pytest.raises(HTTPException) as ei:
        await M.get_learning_program_endpoint(other, prog["id"])
    assert ei.value.status_code == 404

    resp = await M.delete_learning_program_endpoint(pid, prog["id"])
    assert _body(resp) == {"status": "ok"}
    assert store.get_learning_program(prog["id"]) is None


@pytest.mark.asyncio
async def test_design_wave_sync_cells_and_wave_row(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    resp = await M.design_learning_wave_endpoint(
        pid, prog["id"], M.LearningDesignWaveRequest(delta=0.5, n_geo=12, n_holdout=2)
    )
    design = _body(resp)
    # K=2, all pairs probed by default -> 1 + 2K + 2*1 + K = 9 cells
    assert design["n_cells"] == 1 + 2 * 2 + 2 * 1 + 2
    assert len(design["cell_labels"]) == design["n_cells"]
    assert len(design["cells_dollars"]) == design["n_cells"]
    assert design["assignment"]["n_holdout"] == 2
    assert len(design["assignment"]["cell_idx"]) == 12

    # probe_pairs=[] -> no off-axis cells (the FE omits the key for "all")
    resp = await M.design_learning_wave_endpoint(
        pid, prog["id"], M.LearningDesignWaveRequest(probe_pairs=[])
    )
    assert _body(resp)["n_cells"] == 1 + 2 * 2 + 2

    waves = store.list_learning_waves(prog["id"])
    assert len(waves) == 2
    assert waves[0]["status"] == "designed" and waves[0]["design"]["n_cells"] == 9


@pytest.mark.asyncio
async def test_design_wave_optimize_and_stratify_passthrough(store):
    """The optimize/stratify request fields reach the service: a fresh program
    (no posterior, no ingested data) degrades gracefully — a knowledge-gradient
    warning instead of a `kg` block, and a round-robin assignment."""
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    resp = await M.design_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningDesignWaveRequest(optimize=True, stratify=True, n_geo=12),
    )
    design = _body(resp)
    assert any("knowledge-gradient" in w for w in design["warnings"])
    assert "kg" not in design
    assert design["assignment"]["stratified_on"] is None
    assert len(design["assignment"]["cell_idx"]) == 12
    # the request model exposes the new knobs with safe defaults
    body = M.LearningDesignWaveRequest()
    assert body.optimize is False and body.stratify is True
    assert body.candidate_deltas is None and body.kg_n_outcomes == 32


@pytest.mark.asyncio
async def test_design_wave_request_bounds_kg_params(store):
    """[12] The Laplace-KG knobs are bounded at the request model (FastAPI maps
    ValidationError to 422): an unbounded kg_n_outcomes/candidate_deltas would
    run hours of SLSQP solves + GB-scale MvNormal draws off one request."""
    from pydantic import ValidationError

    from mmm_framework.api import main as M

    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(kg_n_outcomes=100000)
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(kg_n_outcomes=0)  # np.mean([]) -> NaN scores
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(kg_n_outcomes=7)  # below the floor
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(candidate_deltas=[0.5] * 9)  # too many
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(candidate_deltas=[0.5, 0.0])  # <= 0
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(candidate_deltas=[-0.3])
    with pytest.raises(ValidationError):
        M.LearningDesignWaveRequest(candidate_deltas=[2.0])  # > 1.5
    # in-range values pass
    ok = M.LearningDesignWaveRequest(kg_n_outcomes=64, candidate_deltas=[0.4, 0.8])
    assert ok.kg_n_outcomes == 64 and ok.candidate_deltas == [0.4, 0.8]


@pytest.mark.asyncio
async def test_ingest_requires_evidence_and_poll_scoping(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    with pytest.raises(HTTPException) as ei:
        await M.ingest_learning_wave_endpoint(
            pid, prog["id"], M.LearningWaveIngestRequest()
        )
    assert ei.value.status_code == 400

    # unknown program -> 404
    with pytest.raises(HTTPException) as ei:
        await M.ingest_learning_wave_endpoint(
            pid, "nope", M.LearningWaveIngestRequest(rows=[{"geo": "g", "y": 1}])
        )
    assert ei.value.status_code == 404

    # poll: right project resolves, cross-project 404s
    tid = f"__learnjobs__{pid}"
    job = store.add_artifact(
        tid,
        "learning_fit",
        {"status": "done", "project_id": pid, "program_id": prog["id"], "result": {}},
    )
    payload = _body(await M.get_learning_job_endpoint(pid, prog["id"], job["id"]))
    assert payload["status"] == "done" and payload["project_id"] == pid
    other = store.create_project("Q")["project_id"]
    with pytest.raises(HTTPException) as ei:
        await M.get_learning_job_endpoint(other, prog["id"], job["id"])
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_wave_job_mocked_fit_fast(store, monkeypatch):
    """Full job wiring with the NUTS fit mocked out: pending → running → done,
    result == the snapshot, wave row + program summary written."""
    from mmm_framework.api import main as M
    from mmm_framework.continuous_learning import service as cl_service

    canned = {
        "schema_version": 1,
        "fitted_at": 0.0,
        "evidence": {
            "n_rows": 24,
            "n_summaries": 0,
            "n_waves": 1,
            "shape_identified": {"Chatter": True, "Pulse": True},
        },
        "diagnostics": {"max_rhat": 1.0, "min_ess": 100, "n_draws": 40, "flags": []},
        "recommendation": {"Chatter": 170000.0, "Pulse": 110000.0},
        "recommendation_scaled": {"Chatter": 1.21, "Pulse": 0.79},
        "allocation_sd": {"Chatter": 9000.0, "Pulse": 9000.0},
        "funding": [],
        "regret": {
            "e_regret_kpi": 0.1,
            "e_regret_dollars": 1.3,
            "enbs": -24998.7,
            "stop": True,
            "margin": 1.0,
            "population": 13.0,
            "wave_cost": 25000.0,
        },
        "gamma": [],
        "response_curves": {},
        "warnings": [],
    }
    seen = {}

    def fake_fit_and_plan(state, **kwargs):
        seen["fit_kwargs"] = kwargs.get("fit_kwargs")
        seen["n_rows"] = int(state.data["spend"].shape[0])
        return dict(canned)

    monkeypatch.setattr(cl_service, "fit_and_plan", fake_fit_and_plan)

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)
    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(rows=_wave_rows(), fit_kwargs=TINY_FIT),
    )
    assert resp.status_code == 202
    job_id = _body(resp)["job_id"]
    assert store.get_artifact(job_id)["payload"]["status"] in ("pending", "running")

    payload = await _poll_to_terminal(store, job_id, timeout_s=30)
    assert payload["status"] == "done", payload.get("error")
    assert payload["result"]["recommendation"]["Chatter"] == 170000.0
    assert payload["project_id"] == pid and payload["program_id"] == prog["id"]
    # the request's fit_kwargs reached the service; rows were ingested
    assert seen == {"fit_kwargs": TINY_FIT, "n_rows": 24}

    # wave row (ingested) + program summary persisted
    waves = store.list_learning_waves(prog["id"])
    assert len(waves) == 1 and waves[0]["status"] == "ingested"
    assert waves[0]["source"] == "wave"
    assert waves[0]["observations"] == {"n_rows": 24, "n_geo": 12}
    prog2 = store.get_learning_program(prog["id"])
    assert prog2["summary"]["recommendation"]["Pulse"] == 110000.0

    # a terminal job keeps returning its status on re-fetch (UI contract)
    again = _body(await M.get_learning_job_endpoint(pid, prog["id"], job_id))
    assert again["status"] == "done"


@pytest.mark.asyncio
async def test_wave_job_bad_rows_reports_error(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)
    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(rows=[{"geo": "g0", "y": 1.0}]),  # no spend cols
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"], timeout_s=30)
    assert payload["status"] == "error"
    assert "spend columns" in payload["error"]


# ── tiny-NUTS end-to-end (kept FAST: 40/40/1 on 24 rows ≈ seconds) ──────────


@pytest.mark.asyncio
async def test_wave_ingest_fit_end_to_end(store):
    """create → design-wave → POST rows (+ tiny fit_kwargs) → poll to done →
    SNAPSHOT schema → GET program shows the wave + summary."""
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    design = _body(
        await M.design_learning_wave_endpoint(
            pid, prog["id"], M.LearningDesignWaveRequest(n_geo=12)
        )
    )
    assert design["n_cells"] == 9

    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(rows=_wave_rows(), fit_kwargs=TINY_FIT),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"])
    assert payload["status"] == "done", payload.get("error")
    snap = payload["result"]
    assert SNAPSHOT_KEYS <= set(snap)
    assert snap["schema_version"] == 1
    assert snap["evidence"]["n_rows"] == 24 and snap["evidence"]["n_waves"] == 1
    assert set(snap["recommendation"]) == {"Chatter", "Pulse"}
    # fixed budget: the recommendation sums back to ~the program budget
    assert sum(snap["recommendation"].values()) == pytest.approx(280000, rel=0.02)
    assert {f["channel"] for f in snap["funding"]} == {"Chatter", "Pulse"}
    assert all(f["verdict"] in ("FUND", "HOLD", "CUT") for f in snap["funding"])
    assert snap["regret"]["wave_cost"] == 25000.0
    curve = snap["response_curves"]["Chatter"]
    assert len(curve["spend_dollars"]) == 25 and curve["current"] == 140000.0

    detail = _body(await M.get_learning_program_endpoint(pid, prog["id"]))
    # the design row RESOLVED to ingested (one board row per real wave — no
    # permanently-open 'designed' duplicate)
    assert len(detail["waves"]) == 1
    ingested = [w for w in detail["waves"] if w["status"] == "ingested"]
    assert len(ingested) == 1 and ingested[0]["snapshot"]["schema_version"] == 1
    assert ingested[0]["design"]["n_cells"] == 9  # design joined to its results
    assert detail["program"]["summary"]["evidence"]["n_rows"] == 24


@pytest.mark.asyncio
async def test_wave_ingest_fit_end_to_end_national_time_effect(store):
    """[9]/[13] A time_effect='national' program is usable END-TO-END through
    the REST surface: create with the config knob → POST rows with period_col
    → the job ingests (tau periods indexed off the week column) and the tiny
    fit reaches done with a full SNAPSHOT."""
    from mmm_framework.api import main as M
    from mmm_framework.continuous_learning import service as cl_service

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(
        M, pid, config=dict(CONFIG, time_effect="national"), name="National"
    )

    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(
            rows=_wave_rows(), fit_kwargs=TINY_FIT, period_col="week"
        ),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"])
    assert payload["status"] == "done", payload.get("error")
    snap = payload["result"]
    assert SNAPSHOT_KEYS <= set(snap)
    assert snap["evidence"]["n_rows"] == 24
    # the persisted state accumulated a global period index (2 periods)
    state = cl_service.load_program_state(pid, prog["id"])
    assert state.time_effect == "national"
    assert state.data is not None and "period_idx" in state.data
    np.testing.assert_array_equal(np.unique(state.data["period_idx"]), [0, 1])
    # ... and the fitted posterior carries the national tau sites
    assert state.posterior is not None and "tau" in state.posterior.samples

    # omitting period_col still works: the service auto-detects the 'week'
    # column (a warning note, not a dead-end error)
    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(
            rows=_wave_rows(n_periods=1, seed=1), fit_kwargs=TINY_FIT
        ),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"])
    assert payload["status"] == "done", payload.get("error")
    state = cl_service.load_program_state(pid, prog["id"])
    # wave 2's single period was offset past wave 1's two periods
    np.testing.assert_array_equal(np.unique(state.data["period_idx"]), [0, 1, 2])


@pytest.mark.slow
@pytest.mark.asyncio
async def test_import_experiments_end_to_end(store):
    """A completed registry readout becomes summary evidence: POST waves
    {experiment_ids} → done → evidence.n_summaries == 1 + imported/skipped."""
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    prog = await _create_program(M, pid)

    exp = store.upsert_experiment(
        project_id=pid,
        channel="Chatter",
        status="completed",
        start_date="2026-01-05",
        end_date="2026-03-02",
        estimand="contribution",
        value=12.0,
        se=2.0,
        readout={
            "value": 12.0,
            "se": 2.0,
            "estimand": "contribution",
            "spend_per_period": 56000.0,  # signed +$ per period per unit
            "n_treated_units": 4,
        },
        design={"treatment_geos": ["a", "b", "c", "d"]},
    )
    bogus = store.upsert_experiment(project_id=pid, channel="Nope", status="completed")

    resp = await M.ingest_learning_wave_endpoint(
        pid,
        prog["id"],
        M.LearningWaveIngestRequest(
            experiment_ids=[exp["id"], bogus["id"]], fit_kwargs=TINY_FIT
        ),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"])
    assert payload["status"] == "done", payload.get("error")
    snap = payload["result"]
    assert snap["evidence"]["n_summaries"] == 1
    assert snap["evidence"]["n_rows"] == 0  # summaries-only fit (no panel)
    # Phase C contract: imported/skipped ride ALONGSIDE the snapshot keys
    assert snap["imported"] == 1
    assert [s["id"] for s in snap["skipped"]] == [bogus["id"]]

    waves = store.list_learning_waves(prog["id"])
    assert waves[-1]["source"] == "experiment_import"
    # provenance = only the successfully imported ids; skips ride observations
    assert waves[-1]["experiment_ids"] == [exp["id"]]
    assert [s["id"] for s in waves[-1]["observations"]["skipped"]] == [bogus["id"]]

    # a follow-up PURE refit via /fit adds no wave row but refreshes the summary
    resp = await M.refit_learning_program_endpoint(
        pid,
        prog["id"],
        M.LearningFitRequest(wave_cost=1.0, fit_kwargs=TINY_FIT),
    )
    payload = await _poll_to_terminal(store, _body(resp)["job_id"])
    assert payload["status"] == "done", payload.get("error")
    assert payload["result"]["regret"]["wave_cost"] == 1.0
    assert len(store.list_learning_waves(prog["id"])) == len(waves)

"""Tests for the experiment registry (lift-test log) and the /portfolio
home-page aggregation: model-run history, experiment tracking, and the
next-action flags (calibrate / refresh / next experiment)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


@pytest.fixture()
def app_main(store, monkeypatch):
    from mmm_framework.api import main as M

    # main.py holds its own reference to the sessions module; same module
    # object, so the DB_PATH monkeypatch above already applies.
    return M


def _body(resp) -> dict:
    return json.loads(resp.body)


class TestExperimentStore:
    def test_create_update_lifecycle(self, store):
        e = store.upsert_experiment(
            channel="TV", project_id="p1", design_type="geo holdout"
        )
        assert e["status"] == "planned"
        store.transition_experiment(e["id"], "running")
        e = store.transition_experiment(
            e["id"],
            "completed",
            value=1.4,
            se=0.3,
            estimand="roas",
        )
        # transition merges the readout fields, preserving untouched ones
        assert e["design_type"] == "geo holdout" and e["value"] == 1.4
        assert store.list_experiments(project_id="p1", status="completed")
        store.transition_experiment(e["id"], "calibrated")
        assert store.list_experiments(status="completed") == []

    def test_validation(self, store):
        with pytest.raises(ValueError, match="channel is required"):
            store.upsert_experiment()
        with pytest.raises(ValueError, match="Invalid status"):
            store.upsert_experiment(channel="X", status="nope")
        with pytest.raises(ValueError, match="Unknown experiment id"):
            store.upsert_experiment(experiment_id="missing", status="running")

    def test_delete(self, store):
        e = store.upsert_experiment(channel="Radio")
        assert store.delete_experiment(e["id"]) is True
        assert store.delete_experiment(e["id"]) is False


class TestExperimentEndpoints:
    @pytest.mark.asyncio
    async def test_upsert_list_delete(self, app_main):
        M = app_main
        resp = await M.upsert_experiment_endpoint(
            M.ExperimentUpsertRequest(channel="TV", project_id="p1")
        )
        exp = _body(resp)
        assert exp["channel"] == "TV"

        listed = _body(await M.list_experiments_endpoint(project_id="p1"))
        assert listed["total"] == 1

        resp2 = await M.upsert_experiment_endpoint(
            M.ExperimentUpsertRequest(id=exp["id"], status="running")
        )
        assert _body(resp2)["status"] == "running"

        await M.delete_experiment_endpoint(exp["id"])
        assert _body(await M.list_experiments_endpoint())["total"] == 0

    @pytest.mark.asyncio
    async def test_invalid_upsert_is_400(self, app_main):
        from fastapi import HTTPException

        M = app_main
        with pytest.raises(HTTPException) as exc:
            await M.upsert_experiment_endpoint(
                M.ExperimentUpsertRequest(channel="X", status="bogus")
            )
        assert exc.value.status_code == 400


class TestPortfolio:
    def _seed(self, store):
        tid = store.create_session(name="s1", project_id="p1")["thread_id"]
        store.add_artifact(
            tid,
            "model_run",
            {
                "run_id": "r1",
                "run_name": "run_1",
                "kpi": "Sales",
                "channels": ["TV", "Digital"],
                "trend": "piecewise",
                "n_obs": 104,
                "summary": "fitted",
            },
        )
        store.add_artifact(
            tid,
            "experiment_design",
            {
                "designs": [
                    {"channel": "TV", "why": "uncertain"},
                    {"channel": "Digital", "why": "also uncertain"},
                ],
                "ranking": [],
            },
        )

    @pytest.mark.asyncio
    async def test_aggregates_models_and_recommendation(self, app_main, store):
        self._seed(store)
        body = _body(await app_main.portfolio_endpoint(project_id="p1"))
        assert len(body["model_runs"]) == 1
        assert body["model_runs"][0]["kpi"] == "Sales"
        assert body["latest_experiment_design"]["designs"][0]["channel"] == "TV"
        # fresh model, no experiments -> only the next-experiment action
        types = [a["type"] for a in body["next_actions"]]
        assert types == ["experiment"]
        assert body["next_actions"][0]["design"]["channel"] == "TV"

    @pytest.mark.asyncio
    async def test_active_experiment_channel_is_skipped(self, app_main, store):
        self._seed(store)
        store.upsert_experiment(channel="TV", project_id="p1", status="running")
        body = _body(await app_main.portfolio_endpoint(project_id="p1"))
        exp_actions = [a for a in body["next_actions"] if a["type"] == "experiment"]
        assert exp_actions[0]["design"]["channel"] == "Digital"

    @pytest.mark.asyncio
    async def test_completed_experiment_flags_calibration(self, app_main, store):
        self._seed(store)
        store.upsert_experiment(
            channel="TV",
            project_id="p1",
            status="completed",
            value=1.2,
            se=0.2,
            estimand="roas",
        )
        body = _body(await app_main.portfolio_endpoint(project_id="p1"))
        cal = [a for a in body["next_actions"] if a["type"] == "calibrate"]
        assert len(cal) == 1 and cal[0]["urgency"] == "high"
        assert "TV" in cal[0]["detail"]
        # calibrating clears the flag (completed -> calibrated is legal)
        eid = body["experiments"][0]["id"]
        store.transition_experiment(eid, "calibrated")
        body2 = _body(await app_main.portfolio_endpoint(project_id="p1"))
        assert not [a for a in body2["next_actions"] if a["type"] == "calibrate"]

    @pytest.mark.asyncio
    async def test_staleness_flags(self, app_main, store):
        # no model at all -> 'fit' action
        store.create_session(name="empty", project_id="p9")
        body = _body(await app_main.portfolio_endpoint(project_id="p9"))
        assert [a["type"] for a in body["next_actions"]] == ["fit"]
        # any model is "stale" with a 0-day window -> 'refresh' action
        self._seed(store)
        body2 = _body(
            await app_main.portfolio_endpoint(project_id="p1", stale_after_days=0)
        )
        assert "refresh" in [a["type"] for a in body2["next_actions"]]

"""Tests for the experiment-design endpoints and the design_experiment_plan
agent tool: design options by data granularity, design computation from the
latest run's dataset, and the tool→plan_experiment snapshot handoff."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


@pytest.fixture(scope="module")
def geo_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("geo_heterogeneous", seed=3, n_weeks=80)
    path = tmp_path_factory.mktemp("design_ep") / "geo.csv"
    df.to_csv(path, index=False)
    return str(path), key


def _body(resp) -> dict:
    return json.loads(resp.body)


def _seed_run(store, pid, dataset_path, channels):
    tid = store.create_session("s", project_id=pid)["thread_id"]
    store.add_artifact(
        tid,
        "model_run",
        {
            "run_id": "r1",
            "dataset_path": dataset_path,
            "kpi": "Sales",
            "channels": channels,
            "spec": {"kpi": "Sales"},
        },
    )
    return tid


@pytest.mark.asyncio
class TestDesignEndpoints:
    async def test_options_and_design(self, store, geo_csv):
        from mmm_framework.api import main as M

        path, key = geo_csv
        pid = store.create_project("P")["project_id"]
        _seed_run(store, pid, path, key["channels"])

        opts = _body(
            await M.experiment_design_options_endpoint(pid, channel=key["channels"][0])
        )
        assert opts["recommended"] == "geo_lift"
        assert opts["kpi"] == "Sales"

        design = _body(
            await M.experiment_design_endpoint(
                pid,
                M.ExperimentDesignRequest(
                    channel=key["channels"][0], duration=8, seed=3
                ),
            )
        )
        assert design["design_key"] == "geo_lift"
        assert design["mde_roas"] > 0 and len(design["assignment"]) >= 2

        flight = _body(
            await M.experiment_design_endpoint(
                pid,
                M.ExperimentDesignRequest(
                    channel=key["channels"][0],
                    design_key="national_flighting",
                    amplitude_pct=40,
                    duration=12,
                ),
            )
        )
        assert flight["design_key"] == "national_flighting"
        assert len(flight["schedule"]) >= 12

    async def test_no_runs_404_and_unknown_channel_400(self, store, geo_csv):
        from mmm_framework.api import main as M

        path, key = geo_csv
        pid = store.create_project("P")["project_id"]
        with pytest.raises(HTTPException) as exc:
            await M.experiment_design_options_endpoint(pid, channel="TV")
        assert exc.value.status_code == 404 and "fit a model" in exc.value.detail

        _seed_run(store, pid, path, key["channels"])
        with pytest.raises(HTTPException) as exc:
            await M.experiment_design_endpoint(
                pid, M.ExperimentDesignRequest(channel="NotAChannel")
            )
        assert exc.value.status_code == 400


class TestDesignTool:
    def test_tool_designs_and_plan_experiment_snapshots(self, store, geo_csv):
        from mmm_framework.agents import tools as T

        path, key = geo_csv
        ch = key["channels"][0]
        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s", project_id=pid)["thread_id"]
        cfg = {"configurable": {"thread_id": tid}}
        state = {
            "dataset_path": path,
            "model_spec": {"kpi": "Sales"},
            "dashboard_data": {},
        }

        cmd = T.design_experiment_plan.func(
            channel=ch, state=state, duration=8, seed=5, config=cfg, tool_call_id="t"
        )
        msg = cmd.update["messages"][0].content
        assert "Matched pairs" in msg and "MDE" in msg and "placebo" in msg.lower()
        design = cmd.update["dashboard_data"]["experiment_design_plan"]
        assert design["channel"] == ch and design["design_key"] == "geo_lift"

        # plan_experiment prefers the full studio design as its snapshot
        state2 = {"dashboard_data": cmd.update["dashboard_data"]}
        cmd2 = T.plan_experiment.func(
            channel=ch, state=state2, config=cfg, tool_call_id="t"
        )
        assert "status **draft**" in cmd2.update["messages"][0].content
        exp = store.list_experiments(project_id=pid)[0]
        assert exp["design"]["design_key"] == "geo_lift"
        assert exp["design"]["assignment"]  # full assignment stored
        assert exp["design"]["power_curve"]

    def test_tool_requires_dataset_and_kpi(self, store):
        from mmm_framework.agents import tools as T

        cmd = T.design_experiment_plan.func(
            channel="TV", state={"dashboard_data": {}}, tool_call_id="t"
        )
        assert "No dataset loaded" in cmd.update["messages"][0].content

    def test_flighting_markdown_for_national(self, store, tmp_path):
        from mmm_framework.agents import tools as T
        from mmm_framework.synth import generate_mff

        df, key = generate_mff("realistic", seed=7, n_weeks=80)
        path = str(tmp_path / "nat.csv")
        df.to_csv(path, index=False)
        state = {
            "dataset_path": path,
            "model_spec": {"kpi": "Sales"},
            "dashboard_data": {},
        }
        cmd = T.design_experiment_plan.func(
            channel=key["channels"][0], state=state, tool_call_id="t"
        )
        msg = cmd.update["messages"][0].content
        assert "flighting" in msg and "Exogenous share" in msg
        design = cmd.update["dashboard_data"]["experiment_design_plan"]
        assert design["design_key"] == "national_flighting"

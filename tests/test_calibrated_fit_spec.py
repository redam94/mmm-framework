"""Tests for the calibrated-refit loop: spec["experiments"] →
add_experiment_calibration in build_model, and the agent lifecycle tools
(plan_experiment → preregister_experiment → record_experiment_readout →
apply_experiment_calibration) that stage it."""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _write_synth_mff(tmp_path, n_weeks=30):
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    df = generate_synthetic_mff(n_weeks=n_weeks)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    return path


def _period_bounds(path):
    import pandas as pd

    period = pd.to_datetime(pd.read_csv(path, usecols=["Period"])["Period"])
    return period.min().date().isoformat(), period.max().date().isoformat()


class TestBuildModelCalibration:
    def test_spec_experiments_register_likelihoods(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        lo, hi = _period_bounds(path)
        spec = {
            "kpi": "Sales",
            "kpi_level": "national",
            "time_granularity": "weekly",
            "media_channels": [{"name": "TV"}],
            "control_variables": [],
            "experiments": [
                {
                    "channel": "TV",
                    "test_period": [lo, hi],
                    "value": 2.0,
                    "se": 0.2,
                    "estimand": "roas",
                }
            ],
        }
        mmm = build_model(spec, path)
        assert len(mmm.experiments) == 1
        m = mmm.experiments[0]
        assert m.channel == "TV" and m.value == 2.0 and m.estimand.value == "roas"

    def test_malformed_experiment_entry_clear_error(self, tmp_path):
        from mmm_framework.agents.fitting import build_model

        path = _write_synth_mff(tmp_path)
        spec = {
            "kpi": "Sales",
            "kpi_level": "national",
            "time_granularity": "weekly",
            "media_channels": [{"name": "TV"}],
            "control_variables": [],
            "experiments": [{"channel": "TV", "value": 2.0}],  # missing fields
        }
        with pytest.raises(ValueError, match="spec.experiments"):
            build_model(spec, path)


class TestLifecycleTools:
    def _session(self, store):
        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        return pid, tid, {"configurable": {"thread_id": tid}}

    def test_plan_preregister_readout(self, store):
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        state = {
            "dashboard_data": {
                "model_run": {"run_id": "run_7"},
                "experiment_design": {
                    "designs": [
                        {
                            "channel": "TV",
                            "design_key": "geo_holdout",
                            "design_type": "geo holdout / geo lift test",
                            "target_se": 0.2,
                        }
                    ]
                },
                "experiment_priorities": {
                    "channels": [
                        {
                            "channel": "TV",
                            "eig": 0.8,
                            "evoi": 100.0,
                            "quadrant": "test_now",
                        }
                    ]
                },
            }
        }
        cmd = T.plan_experiment.func(
            channel="TV",
            state=state,
            hypothesis="TV ROAS is overstated",
            config=cfg,
            tool_call_id="t",
        )
        assert "status **draft**" in cmd.update["messages"][0].content

        exps = store.list_experiments(project_id=pid)
        assert len(exps) == 1
        exp = exps[0]
        assert exp["recommending_run_id"] == "run_7"
        assert exp["design"]["design_key"] == "geo_holdout"
        assert exp["design"]["hypothesis"] == "TV ROAS is overstated"
        assert exp["priority"]["quadrant"] == "test_now"

        cmd = T.preregister_experiment.func(
            experiment_id=exp["id"], config=cfg, tool_call_id="t"
        )
        assert "pre-registered" in cmd.update["messages"][0].content
        assert store.get_experiment(exp["id"])["preregistered_at"] is not None

        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=1.4,
            se=0.2,
            estimand="roas",
            start_date="2026-01-05",
            end_date="2026-03-01",
            method="geo holdout DiD",
            config=cfg,
            tool_call_id="t",
        )
        assert "completed" in cmd.update["messages"][0].content
        exp = store.get_experiment(exp["id"])
        assert exp["status"] == "completed" and exp["value"] == 1.4
        assert exp["readout"]["method"] == "geo holdout DiD"
        # planned -> running was auto-advanced
        assert [h["status"] for h in exp["status_history"]] == [
            "draft",
            "planned",
            "running",
            "completed",
        ]

    def test_apply_calibration_stages_spec(self, store, tmp_path):
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.spec_locks import is_spec_patch
        from mmm_framework.agents.state import _merge_spec

        pid, tid, cfg = self._session(store)
        path = _write_synth_mff(tmp_path)
        lo, hi = _period_bounds(path)
        e = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date=lo,
            end_date=hi,
            value=2.0,
            se=0.25,
            estimand="roas",
        )
        spec = {"kpi": "Sales", "media_channels": [{"name": "TV"}]}
        state = {
            "model_spec": spec,
            "dataset_path": path,
            "locked_fields": [],
            "dashboard_data": {},
        }
        cmd = T.apply_experiment_calibration.func(
            state=state, config=cfg, tool_call_id="t"
        )
        msg = cmd.update["messages"][0].content
        assert "fit_mmm_model" in msg
        assert is_spec_patch(cmd.update["model_spec"])
        folded = _merge_spec(spec, cmd.update["model_spec"])
        assert folded["experiment_ids"] == [e["id"]]
        assert folded["experiments"][0]["channel"] == "TV"
        assert folded["experiments"][0]["test_period"] == [lo, hi]
        assert "experiment_id" not in folded["experiments"][0]

    def test_apply_calibration_rejects_out_of_window(self, store, tmp_path):
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        path = _write_synth_mff(tmp_path)
        store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date="2030-01-01",
            end_date="2030-03-01",
            value=2.0,
            se=0.25,
            estimand="roas",
        )
        state = {
            "model_spec": {"kpi": "Sales", "media_channels": [{"name": "TV"}]},
            "dataset_path": path,
            "locked_fields": [],
            "dashboard_data": {},
        }
        cmd = T.apply_experiment_calibration.func(
            state=state, config=cfg, tool_call_id="t"
        )
        msg = cmd.update["messages"][0].content
        assert "outside the dataset's date range" in msg
        assert "model_spec" not in cmd.update

    def test_apply_calibration_requires_completed_and_fields(self, store, tmp_path):
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        # running (not completed) -> nothing to calibrate
        store.upsert_experiment(channel="TV", project_id=pid, status="running")
        state = {
            "model_spec": {"kpi": "Sales", "media_channels": [{"name": "TV"}]},
            "dataset_path": None,
            "locked_fields": [],
            "dashboard_data": {},
        }
        cmd = T.apply_experiment_calibration.func(
            state=state, config=cfg, tool_call_id="t"
        )
        assert "No completed experiments" in cmd.update["messages"][0].content

        # completed but missing the window -> named problem
        store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", value=2.0, se=0.2
        )
        cmd = T.apply_experiment_calibration.func(
            state=state, config=cfg, tool_call_id="t"
        )
        assert "missing test window" in cmd.update["messages"][0].content

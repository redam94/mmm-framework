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

    def test_apply_calibration_out_of_window_without_spend_advises_offpanel(
        self, store, tmp_path
    ):
        # An out-of-window experiment with no recorded spend level can't be
        # placed on the response curve yet — instead of falsely claiming it's
        # impossible, the tool asks for the spend level (off-panel route) and
        # does not commit a (silently no-op) spec.
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
        assert "off-panel" in msg and "spend_per_period" in msg
        assert "model_spec" not in cmd.update

    def test_rerecord_adds_spend_to_completed_experiment(self, store, tmp_path):
        # The off-panel advisory tells the user to re-run record_experiment_readout
        # with a spend level. That must work on an ALREADY-completed experiment
        # (a completed->completed transition is illegal), so re-recording is an
        # idempotent in-place update of the readout.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date="2030-01-06",
            end_date="2030-03-03",
            value=2.0,
            se=0.25,
            estimand="roas",
            readout={"value": 2.0, "se": 0.25, "estimand": "roas", "method": "DiD"},
        )
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=2.0,
            se=0.25,
            estimand="roas",
            spend_per_period=5000.0,
            n_treated_units=2,
            adstock_state="cold_start",
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "Could not record readout" not in msg  # no illegal-transition error
        after = store.get_experiment(exp["id"])
        assert after["status"] == "completed"
        assert after["readout"]["spend_per_period"] == 5000.0
        assert after["readout"]["n_treated_units"] == 2
        assert after["readout"]["adstock_state"] == "cold_start"
        # Pre-existing readout fields are preserved through the merge.
        assert after["readout"]["method"] == "DiD"

    def test_apply_calibration_offpanel_with_spend_stages_eval_fields(
        self, store, tmp_path
    ):
        # With a recorded spend level, an out-of-window experiment stages as an
        # off-panel measurement (eval_spend / eval_periods) and commits.
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.spec_locks import is_spec_patch
        from mmm_framework.agents.state import _merge_spec

        pid, tid, cfg = self._session(store)
        path = _write_synth_mff(tmp_path)
        e = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date="2030-01-06",
            end_date="2030-03-03",  # ~8 weeks, far past the synthetic panel
            value=2.0,
            se=0.25,
            estimand="roas",
            readout={
                "spend_per_period": 5000.0,
                "n_treated_units": 2,
                "adstock_state": "cold_start",
            },
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
        assert "off-panel" in msg and "fit_mmm_model" in msg
        assert is_spec_patch(cmd.update["model_spec"])
        folded = _merge_spec(spec, cmd.update["model_spec"])
        assert folded["experiment_ids"] == [e["id"]]
        staged = folded["experiments"][0]
        assert staged["channel"] == "TV"
        assert staged["eval_spend"] == 5000.0
        assert staged["eval_units"] == 2
        assert staged["adstock_state"] == "cold_start"
        assert staged["eval_periods"] >= 7  # ~8 weeks inferred from the window

    def test_apply_calibration_retains_already_calibrated(self, store, tmp_path):
        # Regression: a refit that adds a new experiment must NOT drop the
        # experiments calibrated in earlier fits. apply stages the full measured
        # set (completed + already-calibrated), since the model is rebuilt from
        # spec["experiments"] each fit.
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.state import _merge_spec

        pid, tid, cfg = self._session(store)
        path = _write_synth_mff(tmp_path)
        lo, hi = _period_bounds(path)
        a = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date=lo,
            end_date=hi,
            value=2.0,
            se=0.2,
            estimand="roas",
        )
        # folded into a previous fit (create-in-'calibrated' is rejected)
        a = store.transition_experiment(a["id"], "calibrated")
        b = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",  # the newly-measured one
            start_date=lo,
            end_date=hi,
            value=1.5,
            se=0.2,
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
        folded = _merge_spec(spec, cmd.update["model_spec"])
        # BOTH experiments staged, not just the freshly-completed one.
        assert set(folded["experiment_ids"]) == {a["id"], b["id"]}
        assert len(folded["experiments"]) == 2

    def _calibrated_exp(self, store, pid, value=2.0):
        e = store.upsert_experiment(
            channel="TV",
            project_id=pid,
            status="completed",
            start_date="2026-01-05",
            end_date="2026-03-01",
            value=value,
            se=0.25,
            estimand="roas",
            readout={"value": value, "se": 0.25, "estimand": "roas"},
        )
        return store.transition_experiment(e["id"], "calibrated")

    def test_rerecord_on_calibrated_requires_overwrite_flag(self, store):
        # Changing value on a CALIBRATED experiment alters the next refit's
        # likelihood — without overwrite_calibrated the tool refuses and the
        # registry is untouched.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = self._calibrated_exp(store, pid, value=2.0)
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=3.0,
            se=0.25,
            estimand="roas",
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "overwrite_calibrated" in msg and "value" in msg
        after = store.get_experiment(exp["id"])
        assert after["value"] == 2.0 and after["status"] == "calibrated"

    def test_rerecord_on_calibrated_with_flag_updates_and_audits(self, store):
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = self._calibrated_exp(store, pid, value=2.0)
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=3.0,
            se=0.25,
            estimand="roas",
            overwrite_calibrated=True,
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "Could not record readout" not in msg
        after = store.get_experiment(exp["id"])
        assert after["value"] == 3.0 and after["status"] == "calibrated"
        last = after["status_history"][-1]
        assert last["status"] == "calibrated"  # an event, not a transition
        assert last["changed"]["value"] == [2.0, 3.0]

    def test_rerecord_additive_offpanel_on_calibrated_needs_no_flag(self, store):
        # Attaching spend_per_period (the off-panel flow) is additive — it must
        # work on a calibrated experiment WITHOUT the flag, and leave an event.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = self._calibrated_exp(store, pid, value=2.0)
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=2.0,
            se=0.25,
            estimand="roas",
            spend_per_period=5000.0,
            n_treated_units=2,
            adstock_state="cold_start",
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "overwrite_calibrated" not in msg
        assert "Could not record readout" not in msg
        after = store.get_experiment(exp["id"])
        assert after["status"] == "calibrated"
        assert after["readout"]["spend_per_period"] == 5000.0
        last = after["status_history"][-1]
        assert last["changed"]["spend_per_period"] == [None, 5000.0]

    def test_rerecord_changing_existing_offpanel_spend_needs_flag(self, store):
        # spend_per_period is an off-panel LIKELIHOOD input (eval_spend): the
        # first attach is additive, but CHANGING an existing value on a
        # calibrated experiment moves the point where the response curve is
        # evaluated — it must demand overwrite_calibrated like a value edit.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = self._calibrated_exp(store, pid, value=2.0)
        # first-time attach: flag-free (covered above; sets the prior value)
        T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=2.0,
            se=0.25,
            estimand="roas",
            spend_per_period=5000.0,
            config=cfg,
            tool_call_id="t",
        )
        # changing the existing spend level without the flag -> refused
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=2.0,
            se=0.25,
            estimand="roas",
            spend_per_period=8000.0,
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "overwrite_calibrated" in msg and "spend_per_period" in msg
        assert "Nothing was changed" in msg
        after = store.get_experiment(exp["id"])
        assert after["readout"]["spend_per_period"] == 5000.0
        # with the flag the sanctioned overwrite goes through and audits
        cmd = T.record_experiment_readout.func(
            experiment_id=exp["id"],
            value=2.0,
            se=0.25,
            estimand="roas",
            spend_per_period=8000.0,
            overwrite_calibrated=True,
            config=cfg,
            tool_call_id="t",
        )
        assert "Could not record readout" not in cmd.update["messages"][0].content
        after = store.get_experiment(exp["id"])
        assert after["readout"]["spend_per_period"] == 8000.0
        last = after["status_history"][-1]
        assert last["changed"]["spend_per_period"] == [5000.0, 8000.0]

    def test_log_experiment_one_call_results_update(self, store):
        # The docstring's advertised flow: an existing planned experiment is
        # updated to completed WITH results in a single log_experiment call;
        # the skipped 'running' hop is backfilled into the audit history.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = store.upsert_experiment(channel="TV", project_id=pid, status="planned")
        cmd = T.log_experiment.func(
            experiment_id=exp["id"],
            status="completed",
            value=1.2,
            se=0.3,
            estimand="roas",
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "Could not log experiment" not in msg
        after = store.get_experiment(exp["id"])
        assert after["status"] == "completed" and after["value"] == 1.2
        assert [h["status"] for h in after["status_history"]] == [
            "draft",
            "planned",
            "running",
            "completed",
        ]

    def test_log_experiment_cannot_edit_calibrated_measurement(self, store):
        # log_experiment does NOT pass allow_calibrated_edit: a raw value edit
        # on a calibrated row is refused by the store and the row is untouched.
        from mmm_framework.agents import tools as T

        pid, tid, cfg = self._session(store)
        exp = self._calibrated_exp(store, pid, value=2.0)
        cmd = T.log_experiment.func(
            experiment_id=exp["id"],
            value=9.9,
            config=cfg,
            tool_call_id="t",
        )
        msg = cmd.update["messages"][0].content
        assert "Could not log experiment" in msg and "Illegal update" in msg
        after = store.get_experiment(exp["id"])
        assert after["value"] == 2.0 and after["status"] == "calibrated"

    def test_apply_calibration_experiment_ids_merge_then_replace(self, store, tmp_path):
        # Explicit experiment_ids MERGE with the current staging by default —
        # "apply just this new experiment" must not un-stage the earlier ones.
        # replace=True narrows to exactly the listed set.
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.state import _merge_spec

        pid, tid, cfg = self._session(store)
        path = _write_synth_mff(tmp_path)
        lo, hi = _period_bounds(path)

        def _mk(value):
            return store.upsert_experiment(
                channel="TV",
                project_id=pid,
                status="completed",
                start_date=lo,
                end_date=hi,
                value=value,
                se=0.2,
                estimand="roas",
            )

        a, b = _mk(2.0), _mk(1.5)
        spec = {"kpi": "Sales", "media_channels": [{"name": "TV"}]}
        state = {
            "model_spec": spec,
            "dataset_path": path,
            "locked_fields": [],
            "dashboard_data": {},
        }
        cmd = T.apply_experiment_calibration.func(
            state=state, experiment_ids=[a["id"]], config=cfg, tool_call_id="t"
        )
        spec = _merge_spec(spec, cmd.update["model_spec"])
        assert spec["experiment_ids"] == [a["id"]]

        # stage B: A stays staged (merge, not replace)
        state["model_spec"] = spec
        cmd = T.apply_experiment_calibration.func(
            state=state, experiment_ids=[b["id"]], config=cfg, tool_call_id="t"
        )
        msg = cmd.update["messages"][0].content
        assert "1 newly staged" in msg and "1 kept" in msg
        spec = _merge_spec(spec, cmd.update["model_spec"])
        assert set(spec["experiment_ids"]) == {a["id"], b["id"]}
        assert len(spec["experiments"]) == 2

        # replace=True narrows to ONLY the listed experiment
        state["model_spec"] = spec
        cmd = T.apply_experiment_calibration.func(
            state=state,
            experiment_ids=[b["id"]],
            replace=True,
            config=cfg,
            tool_call_id="t",
        )
        assert "replace=True" in cmd.update["messages"][0].content
        spec = _merge_spec(spec, cmd.update["model_spec"])
        assert spec["experiment_ids"] == [b["id"]]
        assert len(spec["experiments"]) == 1

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
        assert "No measured experiments" in cmd.update["messages"][0].content

        # completed but missing the window -> named problem
        store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", value=2.0, se=0.2
        )
        cmd = T.apply_experiment_calibration.func(
            state=state, config=cfg, tool_call_id="t"
        )
        assert "missing test window" in cmd.update["messages"][0].content

"""Agent-tool invocation tests for the pre-fit data-quality tools.

Mirrors the pattern of ``tests/test_agent_workspace_kb.py``: a temp workspace +
session store, tools invoked via ``T.<tool>.func(state=..., tool_call_id=...,
config={"configurable": {"thread_id": tid}})``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .conftest import simple_wide, to_mff_long


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


@pytest.fixture()
def session(store):
    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    return sess["thread_id"]


SPEC = {
    "kpi": "Sales",
    "kpi_level": "national",
    "media_channels": [{"name": "TV"}, {"name": "Search"}],
    "control_variables": [{"name": "Price"}],
}


def write_dataset(tid, wide=None, name="data.csv") -> str:
    from mmm_framework.agents import workspace as W

    wide = wide if wide is not None else simple_wide(n=156)
    path = W.thread_dir(tid) / name
    to_mff_long(wide).to_csv(path, index=False)
    return str(path)


def spiked_wide(n=156) -> pd.DataFrame:
    wide = simple_wide(n=n)
    wide.loc[wide.index[60], "TV"] = float(wide["TV"].max() * 15.0)
    return wide


def make_state(tid, ds_path, spec=SPEC):
    return {
        "dataset_path": ds_path,
        "model_spec": dict(spec),
        "locked_fields": [],
        "pending_spec_changes": [],
        "dashboard_data": {},
    }


def cfg_for(tid):
    return {"configurable": {"thread_id": tid}}


class TestValidateData:
    def test_no_dataset_message(self, store, session):
        from mmm_framework.agents import eda_tools as T

        cmd = T.validate_data.func(
            state={"dashboard_data": {}}, tool_call_id="c", config=cfg_for(session)
        )
        assert "No dataset" in cmd.update["messages"][0].content

    def test_validation_with_plots_as_refs(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session)
        cmd = T.validate_data.func(
            state=make_state(session, ds), tool_call_id="c", config=cfg_for(session)
        )
        content = cmd.update["messages"][0].content
        assert "Data validation" in content
        dd = cmd.update["dashboard_data"]
        assert dd["data_quality"]["validation"]["issues"] is not None
        plots = dd["plots"]
        assert plots and all(set(p) == {"id", "title"} for p in plots)

    def test_negative_spend_is_error(self, store, session):
        from mmm_framework.agents import eda_tools as T

        wide = simple_wide()
        wide.loc[wide.index[4], "TV"] = -50.0
        ds = write_dataset(session, wide)
        cmd = T.validate_data.func(
            state=make_state(session, ds), tool_call_id="c", config=cfg_for(session)
        )
        content = cmd.update["messages"][0].content
        assert "FAILED" in content and "negative_spend" in content


class TestRunEDA:
    def test_full_eda_digest_and_sections(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session)
        cmd = T.run_eda.func(
            state=make_state(session, ds), tool_call_id="c", config=cfg_for(session)
        )
        content = cmd.update["messages"][0].content
        for token in ("Profile", "Top correlations", "Spend share", "Stationarity"):
            assert token in content, f"missing {token} in digest"
        eda = cmd.update["dashboard_data"]["data_quality"]["eda"]
        for key in (
            "profile",
            "collinearity",
            "spend_share",
            "seasonality",
            "stationarity",
        ):
            assert key in eda
        assert len(cmd.update["dashboard_data"]["plots"]) >= 6

    def test_analysis_subset(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session)
        cmd = T.run_eda.func(
            state=make_state(session, ds),
            analyses=["correlation"],
            tool_call_id="c",
            config=cfg_for(session),
        )
        eda = cmd.update["dashboard_data"]["data_quality"]["eda"]
        assert "collinearity" in eda and "spend_share" not in eda


class TestDetectOutliers:
    def test_spike_flagged_and_report_persisted(self, store, session):
        from mmm_framework.agents import eda_tools as T
        from mmm_framework.agents import workspace as W

        ds = write_dataset(session, spiked_wide())
        cmd = T.detect_outliers.func(
            state=make_state(session, ds), tool_call_id="c", config=cfg_for(session)
        )
        content = cmd.update["messages"][0].content
        assert "isolated_spike" in content
        assert "winsorize:TV@" in content
        assert "Normalization damage" in content

        report_file = W.thread_dir(session) / "eda" / "outlier_report.json"
        assert report_file.exists()
        payload = json.loads(report_file.read_text())
        assert payload["dataset_path"] == ds
        assert any(f["variable"] == "TV" for f in payload["flags"])
        # plots are refs
        plots = cmd.update["dashboard_data"]["plots"]
        assert plots and all("id" in p and "data" not in p for p in plots)


class TestApplyOutlierTreatment:
    def _detect(self, T, session, ds):
        return T.detect_outliers.func(
            state=make_state(session, ds), tool_call_id="d", config=cfg_for(session)
        )

    def test_requires_detect_first(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session)
        cmd = T.apply_outlier_treatment.func(
            state=make_state(session, ds),
            action_ids=["winsorize:TV@2024-01-01"],
            tool_call_id="c",
            config=cfg_for(session),
        )
        assert "run `detect_outliers` first" in cmd.update["messages"][0].content

    def test_winsorize_writes_treated_dataset(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session, spiked_wide())
        detect_cmd = self._detect(T, session, ds)
        action_id = next(
            a["action_id"]
            for a in detect_cmd.update["dashboard_data"]["data_quality"]["outliers"][
                "actions"
            ]
            if a["strategy"] == "winsorize"
        )
        cmd = T.apply_outlier_treatment.func(
            state=make_state(session, ds),
            action_ids=[action_id],
            tool_call_id="c",
            config=cfg_for(session),
        )
        new_path = cmd.update["dataset_path"]
        assert Path(new_path).name == "treated_data.csv"
        assert Path(new_path).exists()
        # original untouched, spike removed in the treated copy
        orig = pd.read_csv(ds)
        treated = pd.read_csv(new_path)
        tv_orig = orig[orig["VariableName"] == "TV"]["VariableValue"].astype(float)
        tv_new = treated[treated["VariableName"] == "TV"]["VariableValue"].astype(float)
        assert tv_orig.max() > 5 * tv_new.max() / 5  # original still has the spike
        assert tv_new.max() < tv_orig.max()
        # registered for download
        assert "treated_data.csv" in [f["name"] for f in store.list_files(session)]
        # before/after figure published as ref
        assert any(
            p["title"].startswith("Treatment")
            for p in cmd.update["dashboard_data"]["plots"]
        )

    def test_dummy_action_adds_control_to_spec(self, store, session):
        from mmm_framework.agents import eda_tools as T

        # KPI shock: clean world plus a huge one-week sales spike
        wide = simple_wide(n=156)
        wide.loc[wide.index[80], "Sales"] = float(wide["Sales"].mean() * 3.0)
        ds = write_dataset(session, wide)
        detect_cmd = self._detect(T, session, ds)
        actions = detect_cmd.update["dashboard_data"]["data_quality"]["outliers"][
            "actions"
        ]
        dummy_ids = [a["action_id"] for a in actions if a["strategy"] == "dummy"]
        assert dummy_ids, f"no dummy recommended; actions={actions}"

        cmd = T.apply_outlier_treatment.func(
            state=make_state(session, ds),
            action_ids=dummy_ids[:1],
            tool_call_id="c",
            config=cfg_for(session),
        )
        spec = cmd.update["model_spec"]
        names = [c["name"] for c in spec["control_variables"]]
        assert any(n.startswith("outlier_Sales_") for n in names)
        # dummy variable rows exist in the treated dataset
        treated = pd.read_csv(cmd.update["dataset_path"])
        dummy_name = next(n for n in names if n.startswith("outlier_Sales_"))
        rows = treated[treated["VariableName"] == dummy_name]
        assert len(rows) == 156
        assert rows["VariableValue"].sum() == 1.0

    def test_locked_control_defers_to_pending(self, store, session):
        from mmm_framework.agents import eda_tools as T

        wide = simple_wide(n=156)
        wide.loc[wide.index[80], "Sales"] = float(wide["Sales"].mean() * 3.0)
        ds = write_dataset(session, wide)
        detect_cmd = self._detect(T, session, ds)
        actions = detect_cmd.update["dashboard_data"]["data_quality"]["outliers"][
            "actions"
        ]
        dummy_ids = [a["action_id"] for a in actions if a["strategy"] == "dummy"]

        state = make_state(session, ds)
        state["locked_fields"] = ["control_variables"]
        cmd = T.apply_outlier_treatment.func(
            state=state,
            action_ids=dummy_ids[:1],
            tool_call_id="c",
            config=cfg_for(session),
        )
        spec = cmd.update["model_spec"]
        names = [c["name"] for c in spec.get("control_variables", [])]
        assert not any(n.startswith("outlier_Sales_") for n in names)
        assert cmd.update["pending_spec_changes"], "locked change should be deferred"

    def test_stale_report_rejected(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session, spiked_wide())
        detect_cmd = self._detect(T, session, ds)
        action_id = next(
            a["action_id"]
            for a in detect_cmd.update["dashboard_data"]["data_quality"]["outliers"][
                "actions"
            ]
        )
        # dataset changes after detection -> stale
        import os
        import time

        os.utime(ds, (time.time() + 10, time.time() + 10))
        cmd = T.apply_outlier_treatment.func(
            state=make_state(session, ds),
            action_ids=[action_id],
            tool_call_id="c",
            config=cfg_for(session),
        )
        assert "stale" in cmd.update["messages"][0].content

    def test_unknown_action_id_rejected(self, store, session):
        from mmm_framework.agents import eda_tools as T

        ds = write_dataset(session, spiked_wide())
        self._detect(T, session, ds)
        cmd = T.apply_outlier_treatment.func(
            state=make_state(session, ds),
            action_ids=["winsorize:Nope@2020-01-01"],
            tool_call_id="c",
            config=cfg_for(session),
        )
        assert "Unknown action_id" in cmd.update["messages"][0].content

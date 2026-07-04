"""Agent tools for the Data Studio (agents/data_studio_tools.py).

The tools are thin wrappers over data_studio.service sharing the UI's staging
manifest; these tests pin the tool-level contract: status/stage/pipeline
messages, the dashboard_data.data_studio pointer, JSON validation, and that
commit_data_studio returns commit_core's state update PLUS the ToolMessage
(the one thing the REST endpoint must NOT add).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.agents import workspace as W

    return W


def _wide(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "week": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
            "sales": 1000 + np.arange(n) * 5 + rng.normal(0, 20, n),
            "tv_spend": np.abs(rng.normal(50, 10, n)),
            "search_spend": np.abs(rng.normal(30, 8, n)),
        }
    )


def _use_thread(tid: str):
    from mmm_framework.agents.runtime import set_current_thread

    set_current_thread(tid)


def _text(cmd) -> str:
    return cmd.update["messages"][0].content


STATE = {
    "model_spec": {},
    "locked_fields": [],
    "pending_spec_changes": [],
    "dashboard_data": {},
}


def test_tools_are_registered():
    from mmm_framework.agents.data_studio_tools import DATA_STUDIO_TOOLS
    from mmm_framework.agents.tools import TOOLS

    names = {t.name for t in TOOLS}
    for t in DATA_STUDIO_TOOLS:
        assert t.name in names, t.name


def test_status_with_nothing_staged(ws):
    from mmm_framework.agents.data_studio_tools import data_studio_status

    _use_thread("t_ds_none")
    cmd = data_studio_status.func(state=dict(STATE), tool_call_id="t1", config=None)
    assert "No dataset is staged" in _text(cmd)


def test_stage_pipeline_status_roundtrip(ws):
    from mmm_framework.agents.data_studio_tools import (
        data_studio_status,
        set_data_studio_pipeline,
        stage_data_studio_file,
    )

    tid = "t_ds_round"
    _use_thread(tid)
    # a chat-style upload sitting in the thread workspace
    root = ws.thread_dir(tid)
    _wide().to_csv(root / "sales.csv", index=False)

    cmd = stage_data_studio_file.func(
        path="sales.csv", state=dict(STATE), tool_call_id="t1", config=None
    )
    assert "Staged `sales.csv`" in _text(cmd)
    pointer = cmd.update["dashboard_data"]["data_studio"]
    assert pointer["active"] is True and pointer["filename"] == "sales.csv"

    cmd = set_data_studio_pipeline.func(
        steps=json.dumps(
            [
                {"op": "rename", "from": "week", "to": "date"},
                {"op": "parse_date", "column": "date"},
            ]
        ),
        roles=json.dumps({"date": "date", "sales": "kpi"}),
        state=dict(STATE),
        tool_call_id="t2",
        config=None,
    )
    assert "Pipeline set (2 step(s))" in _text(cmd)
    assert cmd.update["dashboard_data"]["data_studio"]["n_steps"] == 2

    cmd = data_studio_status.func(state=dict(STATE), tool_call_id="t3", config=None)
    text = _text(cmd)
    assert "sales.csv" in text and "rename → parse_date" in text


def test_pipeline_validation_errors(ws):
    from mmm_framework.agents.data_studio_tools import set_data_studio_pipeline

    tid = "t_ds_badpipe"
    _use_thread(tid)
    root = ws.thread_dir(tid)
    _wide().to_csv(root / "s.csv", index=False)
    from mmm_framework.agents.data_studio_tools import stage_data_studio_file

    stage_data_studio_file.func(
        path="s.csv", state=dict(STATE), tool_call_id="t0", config=None
    )

    bad_json = set_data_studio_pipeline.func(
        steps="{not json", state=dict(STATE), tool_call_id="t1", config=None
    )
    assert "Could not parse steps JSON" in _text(bad_json)

    unknown_op = set_data_studio_pipeline.func(
        steps=json.dumps([{"op": "explode"}]),
        state=dict(STATE),
        tool_call_id="t2",
        config=None,
    )
    assert "Pipeline rejected" in _text(unknown_op)

    not_a_list = set_data_studio_pipeline.func(
        steps=json.dumps({"op": "rename"}),
        state=dict(STATE),
        tool_call_id="t3",
        config=None,
    )
    assert "JSON LIST" in _text(not_a_list)


def test_stage_rejects_traversal_and_missing(ws):
    from mmm_framework.agents.data_studio_tools import stage_data_studio_file

    tid = "t_ds_paths"
    _use_thread(tid)
    ws.thread_dir(tid)  # materialize
    out = stage_data_studio_file.func(
        path="../../etc/passwd", state=dict(STATE), tool_call_id="t1", config=None
    )
    assert "Error" in _text(out)
    out = stage_data_studio_file.func(
        path="nope.csv", state=dict(STATE), tool_call_id="t2", config=None
    )
    assert "no such file" in _text(out)


def test_commit_returns_state_update_with_toolmessage(ws):
    from mmm_framework.agents.data_studio_tools import (
        commit_data_studio,
        stage_data_studio_file,
    )

    tid = "t_ds_commit"
    _use_thread(tid)
    root = ws.thread_dir(tid)
    _wide().to_csv(root / "sales.csv", index=False)
    stage_data_studio_file.func(
        path="sales.csv", state=dict(STATE), tool_call_id="t0", config=None
    )

    cmd = commit_data_studio.func(
        reason="agent commit", state=dict(STATE), tool_call_id="t1", config=None
    )
    up = cmd.update
    # commit_core's update rides through intact...
    assert up["dataset_path"].endswith("data_studio_dataset.csv")
    assert up["model_spec"]["kpi"] == "sales"
    assert up["dashboard_data"]["data_studio"]["committed"] is True
    # ...plus the ToolMessage the endpoint variant must not have
    assert len(up["messages"]) == 1


def test_commit_with_nothing_staged(ws):
    from mmm_framework.agents.data_studio_tools import commit_data_studio

    _use_thread("t_ds_commit_none")
    cmd = commit_data_studio.func(state=dict(STATE), tool_call_id="t1", config=None)
    assert "Commit failed" in _text(cmd)

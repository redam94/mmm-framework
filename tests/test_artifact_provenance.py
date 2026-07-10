"""Tests for provenance metadata on plot/table artifact refs.

Refs stored in ``dashboard_data["plots"]`` / ``dashboard_data["tables"]`` gain
optional, additive provenance so the UI can group artifacts by the question
(tool call) that produced them:

- plot refs: ``ts`` (epoch float, always), ``call_id`` + ``source`` (when truthy)
- table refs: ``ts`` (always), ``call_id`` (when truthy)

Covers the publishing helpers (``publish_tables``, ``_publish_figures``,
``_publish_modelop_plots``), the model-op dispatch (``_modelop_command``),
``execute_python``, and the state reducer (``_merge_dashboard``) which must
pass the new fields through untouched. No network, no LLM.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the session store + checkpointer + workspace at a temp location."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


def _table(**over):
    base = {
        "title": "T",
        "columns": [{"key": "a", "label": "A", "type": "number"}],
        "rows": [{"a": 1}],
        "total_rows": 1,
        "truncated": False,
        "source": "test",
        "group": "results",
    }
    base.update(over)
    return base


def _fig_dict(x=1.0):
    return {"data": [{"type": "scatter", "x": [x], "y": [x]}], "layout": {}}


# ── publish_tables ───────────────────────────────────────────────────────────


def test_publish_tables_stamps_ts_and_call_id(store):
    from mmm_framework.agents.tables import publish_tables

    dd = {}
    before = time.time()
    refs, dropped = publish_tables([_table()], dd, "threadA", call_id="call-1")
    after = time.time()
    assert dropped == 0 and len(refs) == 1
    ref = refs[0]
    assert ref["call_id"] == "call-1"
    assert isinstance(ref["ts"], float) and before <= ref["ts"] <= after
    # Legacy keys unchanged.
    assert ref["title"] == "T" and ref["source"] == "test" and ref["group"] == "results"
    assert dd["tables"] == refs


def test_publish_tables_omits_call_id_when_absent(store):
    from mmm_framework.agents.tables import publish_tables

    dd = {}
    refs, _ = publish_tables([_table()], dd, "threadA")
    assert "call_id" not in refs[0] and "ts" in refs[0]
    refs2, _ = publish_tables([_table()], dd, "threadA", call_id=None)
    assert "call_id" not in refs2[0]
    refs3, _ = publish_tables([_table()], dd, "threadA", call_id="")
    assert "call_id" not in refs3[0]


# ── _publish_figures (EDA / learning plots) ──────────────────────────────────


def test_publish_figures_stamps_provenance(store):
    import plotly.graph_objects as go

    from mmm_framework.agents.eda_tools import _publish_figures

    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    dd = {}
    before = time.time()
    refs, dropped = _publish_figures(
        [("My chart", fig)], dd, "threadA", call_id="call-2", source="run_eda"
    )
    after = time.time()
    assert dropped == 0 and len(refs) == 1
    ref = refs[0]
    assert ref["title"] == "My chart"
    assert ref["call_id"] == "call-2" and ref["source"] == "run_eda"
    assert isinstance(ref["ts"], float) and before <= ref["ts"] <= after
    assert dd["plots"] == refs


def test_publish_figures_omits_untruthy_provenance(store):
    import plotly.graph_objects as go

    from mmm_framework.agents.eda_tools import _publish_figures

    fig = go.Figure(data=[go.Scatter(x=[1], y=[1])])
    refs, _ = _publish_figures([("t", fig)], {}, "threadA")
    assert "call_id" not in refs[0] and "source" not in refs[0]
    assert "ts" in refs[0]


# ── _publish_modelop_plots ───────────────────────────────────────────────────


def test_publish_modelop_plots_stamps_provenance(store):
    from mmm_framework.agents.tools import _publish_modelop_plots

    dd = {}
    note = _publish_modelop_plots(
        [{"title": "Fit", "figure": _fig_dict()}],
        dd,
        "threadA",
        call_id="call-3",
        source="roi_metrics",
    )
    assert "1 chart(s)" in note
    (ref,) = dd["plots"]
    assert ref["call_id"] == "call-3" and ref["source"] == "roi_metrics"
    assert isinstance(ref["ts"], float)

    # Without provenance the optional keys are omitted.
    dd2 = {}
    _publish_modelop_plots([{"title": "Fit", "figure": _fig_dict()}], dd2, "threadA")
    (ref2,) = dd2["plots"]
    assert "call_id" not in ref2 and "source" not in ref2 and "ts" in ref2


# ── _modelop_command passes call_id through to both publishers ───────────────


def test_modelop_command_passes_call_id_through(store):
    from mmm_framework.agents import runtime as rt
    from mmm_framework.agents.tools import _modelop_command

    rt.set_current_thread("threadP")
    try:
        res = {
            "content": "### x",
            "dashboard": {},
            "error": None,
            "tables": [_table(title="ROI", source="get_roi_metrics")],
            "plots": [{"title": "ROI chart", "figure": _fig_dict()}],
        }
        cmd = _modelop_command(res, {"dashboard_data": {}}, "tc-42")
    finally:
        rt.set_current_thread(None)
    dd = cmd.update["dashboard_data"]
    (tref,) = dd["tables"]
    (pref,) = dd["plots"]
    assert tref["call_id"] == "tc-42" and isinstance(tref["ts"], float)
    assert pref["call_id"] == "tc-42" and isinstance(pref["ts"], float)


# ── execute_python refs carry provenance ─────────────────────────────────────


def test_execute_python_refs_carry_provenance(store):
    from mmm_framework.agents import tools as T

    proj = store.create_project("P")
    sess = store.create_session(name="s", project_id=proj["project_id"])
    cfg = {"configurable": {"thread_id": sess["thread_id"]}}
    code = (
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "go.Figure(data=[go.Scatter(x=[1], y=[1])]).show()\n"
        "show_table(pd.DataFrame({'a': [1]}), title='Tbl')\n"
        "print('done')"
    )
    cmd = T.execute_python.func(
        state={"dashboard_data": {}}, code=code, tool_call_id="tc-exec", config=cfg
    )
    dd = cmd.update["dashboard_data"]
    (pref,) = dd["plots"]
    assert pref["call_id"] == "tc-exec" and pref["source"] == "execute_python"
    assert isinstance(pref["ts"], float)
    (tref,) = dd["tables"]
    assert tref["call_id"] == "tc-exec" and isinstance(tref["ts"], float)


# ── the state reducer preserves the new fields ───────────────────────────────


def test_merge_dashboard_preserves_provenance_fields():
    from mmm_framework.agents.state import _merge_dashboard

    a = {
        "plots": [
            {"id": "p1", "title": "A", "call_id": "c1", "ts": 1.0, "source": "run_eda"}
        ],
        "tables": [{"id": "t1", "title": "T", "call_id": "c1", "ts": 1.0}],
    }
    b = {
        "plots": [
            # id collision: FIRST occurrence (a's ref, with its provenance) wins
            {"id": "p1", "title": "stale", "call_id": "c2", "ts": 2.0},
            {"id": "p2", "title": "B", "call_id": "c2", "ts": 2.0, "source": "s2"},
        ],
        "tables": [{"id": "t2", "title": "T2"}],  # old-style ref, no new fields
    }
    m = _merge_dashboard(a, b)
    assert [p["id"] for p in m["plots"]] == ["p1", "p2"]
    p1, p2 = m["plots"]
    assert p1["call_id"] == "c1" and p1["ts"] == 1.0 and p1["source"] == "run_eda"
    assert p2["call_id"] == "c2" and p2["source"] == "s2"
    t1, t2 = m["tables"]
    assert t1["call_id"] == "c1" and t1["ts"] == 1.0
    # Old refs without provenance stay valid — nothing assumes the fields exist.
    assert "call_id" not in t2 and "ts" not in t2


def test_merge_dashboard_mixes_old_and_new_refs():
    from mmm_framework.agents.state import _merge_dashboard

    old = {"plots": [{"id": "p0", "title": "legacy"}]}
    new = {"plots": [{"id": "p1", "title": "N", "call_id": "c", "ts": 9.9}]}
    m = _merge_dashboard(old, new)
    assert [p["id"] for p in m["plots"]] == ["p0", "p1"]
    assert "call_id" not in m["plots"][0]
    assert m["plots"][1]["call_id"] == "c"

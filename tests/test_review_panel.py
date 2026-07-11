"""Persona review panel (Phase 3): convene_review_panel runs a team of expert
personas (statistician / media planner / CMO), each grounded in the Phase-1
validation/analysis tools, and stitches their feedback into one review.

No network, no LLM: the expert sub-graph is stubbed (mirrors test_delegation).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

import mmm_framework.agents.tools as T

# ── structural ────────────────────────────────────────────────────────────────


def test_panel_is_orchestrator_only():
    orch = {t.name for t in T.ORCHESTRATOR_TOOLS}
    expert = {t.name for t in T.EXPERT_TOOLS}
    assert "convene_review_panel" in orch  # orchestrator can convene
    assert "convene_review_panel" not in expert  # expert must not recurse


def test_panel_available_in_every_mode_for_orchestrator():
    for mode in ("mmm", "causal_inference", "general_bayes", "descriptive"):
        names = {t.name for t in T.get_tools_for_mode(mode, role="orchestrator")}
        assert "convene_review_panel" in names, mode
        exp = {t.name for t in T.get_tools_for_mode(mode, role="expert")}
        assert "convene_review_panel" not in exp, mode


def test_three_personas_registered():
    assert set(T._REVIEW_PERSONAS) == {"statistician", "media_planner", "cmo"}
    for p in T._REVIEW_PERSONAS.values():
        assert p["label"] and p["brief"]


# ── behaviour ─────────────────────────────────────────────────────────────────


class _PanelStub:
    """Persona-aware stub expert graph: tags its reply by which brief it got, so
    we can prove each persona ran with its own instructions."""

    def __init__(self):
        self.tasks: list[str] = []

    def stream(self, init_state, config=None, stream_mode=None):
        task = init_state["messages"][0].content
        self.tasks.append(task)
        tag = (
            "STAT"
            if "statistician" in task
            else "PLAN" if "media planner" in task else "CMO"
        )
        yield {
            "messages": [
                ToolMessage(content="ran a tool", tool_call_id="x"),
                AIMessage(content=f"[{tag}] my review"),
            ],
            "dashboard_data": {"tables": [{"id": f"t-{tag}"}]},
        }


def test_panel_runs_all_personas_and_merges(monkeypatch):
    stub = _PanelStub()
    monkeypatch.setattr(T, "_get_expert_graph", lambda override=None, mode=None: stub)

    cmd = T.convene_review_panel.func(
        state={
            "dataset_path": "/ws/d.csv",
            "model_status": "completed",
            "dashboard_data": {"tables": [{"id": "pre"}]},
        },
        focus="Is this model ready to set next quarter's budget?",
        tool_call_id="call-1",
        config={"configurable": {"thread_id": "thread-z"}},
    )

    assert isinstance(cmd, Command)
    msg = cmd.update["messages"][0]
    assert isinstance(msg, ToolMessage) and msg.tool_call_id == "call-1"
    # all three personas appear, each with its tagged review
    assert "Review panel" in msg.content
    for label in ("🔬 Expert statistician", "📊 Media planner", "🎯 CMO"):
        assert label in msg.content
    assert (
        "[STAT]" in msg.content and "[PLAN]" in msg.content and "[CMO]" in msg.content
    )

    # each persona ran against the SAME thread with its OWN brief + the focus
    assert len(stub.tasks) == 3
    assert any("statistician" in t for t in stub.tasks)
    assert any("media planner" in t for t in stub.tasks)
    assert all("next quarter's budget" in t for t in stub.tasks)

    # dashboards from every persona fold back on top of the pre-existing one
    table_ids = {t["id"] for t in cmd.update["dashboard_data"]["tables"]}
    assert table_ids == {"pre", "t-STAT", "t-PLAN", "t-CMO"}


def test_panel_persists_persona_code_artifacts(monkeypatch, tmp_path):
    """A persona that runs execute_python gets its code recorded as session
    artifacts (via=review_panel:<persona>), same as delegate_to_expert."""
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()

    class _CodeStub:
        def stream(self, init_state, config=None, stream_mode=None):
            task = init_state["messages"][0].content
            call_id = f"exp-{abs(hash(task)) % 1000}"
            yield {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "execute_python",
                                "args": {"code": "1 + 1"},
                                "id": call_id,
                            }
                        ],
                    ),
                    ToolMessage(content="2", tool_call_id=call_id),
                    AIMessage(content="reviewed"),
                ],
                "dashboard_data": {},
            }

    monkeypatch.setattr(
        T, "_get_expert_graph", lambda override=None, mode=None: _CodeStub()
    )
    T.convene_review_panel.func(
        state={},
        focus="check the residuals",
        tool_call_id="c1",
        config={"configurable": {"thread_id": "panel-1"}},
    )

    snippets = [
        a["payload"] for a in S.list_artifacts("panel-1") if a["kind"] == "code_snippet"
    ]
    assert len(snippets) == 3  # one per persona
    vias = {s["via"] for s in snippets}
    assert vias == {
        "review_panel:statistician",
        "review_panel:media_planner",
        "review_panel:cmo",
    }
    assert all(s["question"] == "check the residuals" for s in snippets)


def test_panel_degrades_when_a_reviewer_fails(monkeypatch):
    class _Boom:
        def stream(self, *a, **k):
            raise RuntimeError("kernel exploded")

    monkeypatch.setattr(
        T, "_get_expert_graph", lambda override=None, mode=None: _Boom()
    )
    cmd = T.convene_review_panel.func(
        state={},
        focus="review",
        tool_call_id="c1",
        config={"configurable": {"thread_id": "t1"}},
    )
    # never raises — each section reports the failure, the panel still returns
    content = cmd.update["messages"][0].content
    assert "Review panel" in content
    assert content.count("could not complete") == 3

"""Tests for the two-tier delegation system: a fast chat ORCHESTRATOR that hands
hard work to a strong EXPERT sub-agent via the ``delegate_to_expert`` tool.

Covers the structural tool split (which tools each tier gets) and the delegate
tool's behaviour — that it invokes the expert sub-graph against the SAME
``thread_id`` and folds the expert's summary + session state back into a Command.
No network, no LLM: the expert graph is stubbed.
"""

from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

import mmm_framework.agents.tools as T


# ── Structural tool split ─────────────────────────────────────────────────────


def test_orchestrator_excludes_heavy_tools_and_keeps_delegate():
    orch = {t.name for t in T.ORCHESTRATOR_TOOLS}
    # The heavy / code-gen tools are removed from the fast tier...
    assert T.HEAVY_TOOL_NAMES.isdisjoint(orch)
    assert "fit_mmm_model" not in orch
    assert "execute_python" not in orch
    # ...but it can delegate, and keeps the cheap read/config tools.
    assert "delegate_to_expert" in orch
    assert "inspect_dataset" in orch
    assert "get_current_config" in orch


def test_expert_has_heavy_tools_but_not_delegate():
    expert = {t.name for t in T.EXPERT_TOOLS}
    # Full power for the expert...
    assert T.HEAVY_TOOL_NAMES <= expert
    assert "fit_mmm_model" in expert
    assert "execute_python" in expert
    # ...but it must NOT be able to recurse into itself.
    assert "delegate_to_expert" not in expert


def test_full_tools_escape_hatch(monkeypatch):
    monkeypatch.setenv("MMM_AGENT_ORCHESTRATOR_FULL_TOOLS", "1")
    reloaded = importlib.reload(T)
    try:
        orch = {t.name for t in reloaded.ORCHESTRATOR_TOOLS}
        # With the escape hatch, the orchestrator keeps every tool.
        assert reloaded.HEAVY_TOOL_NAMES <= orch
        assert "fit_mmm_model" in orch
        assert "delegate_to_expert" in orch
    finally:
        # Restore the default module state for other tests.
        monkeypatch.delenv("MMM_AGENT_ORCHESTRATOR_FULL_TOOLS", raising=False)
        importlib.reload(T)


# ── delegate_to_expert behaviour ──────────────────────────────────────────────


class _StubExpertGraph:
    """Stands in for the compiled expert sub-graph. Records the config it was
    streamed with and yields a canned final state.

    ``delegate_to_expert`` drives the expert with ``graph.stream(..., stream_mode=
    "values")`` (keeping the LAST full state so partial progress survives a step-
    limit blowout), so the stub is a generator that yields the canned state once.
    """

    def __init__(self, result):
        self._result = result
        self.invoked_with = None

    def stream(self, init_state, config=None, stream_mode=None):
        self.invoked_with = {
            "init_state": init_state,
            "config": config,
            "stream_mode": stream_mode,
        }
        yield self._result


def test_delegate_returns_summary_and_propagates_state(monkeypatch):
    result_state = {
        "messages": [
            AIMessage(content="", tool_calls=[]),  # placeholder
            ToolMessage(content="tool ran", tool_call_id="x"),
            AIMessage(content="Fit complete. R-hat max 1.01, no divergences."),
        ],
        "dashboard_data": {"plots": [{"id": "p1", "title": "trace"}]},
        "model_spec": {"kpi": "sales", "media_channels": [{"name": "tv"}]},
        "model_status": "completed",
        "fit_results_summary": "ROAS tv 2.3",
        "report_path": "/ws/report.html",
    }
    stub = _StubExpertGraph(result_state)
    monkeypatch.setattr(T, "_get_expert_graph", lambda override=None, mode=None: stub)

    cfg = {"configurable": {"thread_id": "thread-abc"}}
    cmd = T.delegate_to_expert.func(
        state={"dataset_path": "/ws/data.csv", "model_spec": {"kpi": "sales"}},
        task="Fit the configured model and report diagnostics.",
        tool_call_id="call-1",
        config=cfg,
    )

    assert isinstance(cmd, Command)
    upd = cmd.update
    # The expert's final assistant text comes back as the tool message.
    (msg,) = upd["messages"]
    assert isinstance(msg, ToolMessage)
    assert "Fit complete" in msg.content
    assert msg.tool_call_id == "call-1"
    # Session state the expert mutated is folded back.
    assert upd["dashboard_data"]["plots"][0]["id"] == "p1"
    assert upd["model_spec"]["media_channels"][0]["name"] == "tv"
    assert upd["model_status"] == "completed"
    assert upd["fit_results_summary"] == "ROAS tv 2.3"
    assert upd["report_path"] == "/ws/report.html"

    # The expert ran against the SAME thread_id (shared session) and got the task.
    assert stub.invoked_with["config"]["configurable"]["thread_id"] == "thread-abc"
    seeded = stub.invoked_with["init_state"]
    assert seeded["dataset_path"] == "/ws/data.csv"
    assert seeded["messages"][0].content.startswith("Fit the configured model")


def test_delegate_handles_expert_failure_gracefully(monkeypatch):
    class _Boom:
        def stream(self, *a, **k):
            raise RuntimeError("kernel exploded")

    monkeypatch.setattr(
        T, "_get_expert_graph", lambda override=None, mode=None: _Boom()
    )
    cmd = T.delegate_to_expert.func(
        state={},
        task="do something hard",
        tool_call_id="c1",
        config={"configurable": {"thread_id": "t1"}},
    )
    (msg,) = cmd.update["messages"]
    assert msg.status == "error"
    assert "kernel exploded" in msg.content


def test_delegate_empty_summary_falls_back(monkeypatch):
    # Expert ended without any assistant text (e.g. hit the recursion limit).
    stub = _StubExpertGraph({"messages": [ToolMessage(content="x", tool_call_id="y")]})
    monkeypatch.setattr(T, "_get_expert_graph", lambda override=None, mode=None: stub)
    cmd = T.delegate_to_expert.func(
        state={},
        task="t",
        tool_call_id="c1",
        config={"configurable": {"thread_id": "t1"}},
    )
    (msg,) = cmd.update["messages"]
    assert "no summary" in msg.content.lower()


def test_delegate_passes_expert_override_to_build(monkeypatch):
    # The X-Expert-* selection rides in config.configurable; delegate_to_expert
    # must forward it to build_expert_llm (via _get_expert_graph).
    captured = {}

    def fake_build_expert_llm(**kwargs):
        captured.update(kwargs)
        return object()

    fake_graph = _StubExpertGraph({"messages": [AIMessage(content="done")]})

    def fake_create_graph(llm, **kwargs):
        captured["created"] = True
        return fake_graph

    import mmm_framework.agents.graph as G
    import mmm_framework.agents.llm as L

    monkeypatch.setattr(L, "build_expert_llm", fake_build_expert_llm)
    monkeypatch.setattr(G, "create_agent_graph", fake_create_graph)

    cfg = {
        "configurable": {
            "thread_id": "t1",
            "expert_model": "claude-sonnet-4-5",
            "expert_provider": "anthropic",
            "expert_api_key": "sk-expert",
            "expert_base_url": None,
        }
    }
    cmd = T.delegate_to_expert.func(state={}, task="fit", tool_call_id="c1", config=cfg)
    assert "done" in cmd.update["messages"][0].content
    assert captured["model_name"] == "claude-sonnet-4-5"
    assert captured["provider"] == "anthropic"
    assert captured["api_key"] == "sk-expert"


def test_get_expert_graph_no_override_uses_singleton(monkeypatch):
    import mmm_framework.agents.graph as G
    import mmm_framework.agents.llm as L

    calls = {"n": 0}
    monkeypatch.setattr(L, "build_expert_llm", lambda **k: object())
    monkeypatch.setattr(
        G,
        "create_agent_graph",
        lambda llm, **k: calls.update(n=calls["n"] + 1) or object(),
    )
    # Per-mode cache (default mode "mmm"): built once, then served from the cache.
    monkeypatch.setattr(T, "_EXPERT_GRAPHS", {})
    g1 = T._get_expert_graph(None)
    g2 = T._get_expert_graph({})  # all-empty override → still the cached path
    assert g1 is g2
    assert calls["n"] == 1  # built once, cached


def test_final_message_text_handles_block_content():
    msgs = [
        ToolMessage(content="ignored", tool_call_id="a"),
        AIMessage(
            content=[
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ]
        ),
    ]
    assert T._final_message_text(msgs) == "part one\npart two"

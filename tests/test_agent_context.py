"""Tests for the agent context-window manager (mmm_framework.agents.context).

These guard the fix for runaway request size (e.g. gpt-5.5's 1M TPM limit):
summarize-and-compact old turns + a hard trim_messages backstop, with
tool-call/ToolMessage pairing preserved across folds.
"""

import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mmm_framework.agents import context as ctx


class FakeLLM:
    """Records calls and returns a deterministic summary."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=f"SUMMARY#{self.calls}: condensed goals/spec/results")


@pytest.fixture
def small_budgets(monkeypatch):
    """Shrink budgets so the logic triggers on tiny inputs."""
    monkeypatch.setattr(ctx, "MAX_CONTEXT_TOKENS", 400)
    monkeypatch.setattr(ctx, "RECENT_BUDGET", 200)
    monkeypatch.setattr(ctx, "MIN_RECENT_MESSAGES", 2)
    monkeypatch.setattr(ctx, "SUMMARIZE_CHUNK_TOKENS", 150)


def _turns(n, size=200):
    """n turns of human -> ai(tool_call) -> tool."""
    hist = []
    for i in range(n):
        hist.append(HumanMessage(content=f"q{i} " + "x" * size))
        hist.append(
            AIMessage(
                content=f"a{i} " + "y" * size,
                tool_calls=[{"name": "t", "args": {}, "id": f"c{i}"}],
            )
        )
        hist.append(ToolMessage(content=f"r{i} " + "z" * size, tool_call_id=f"c{i}"))
    return hist


def _assert_no_orphan_tools(messages):
    seen = set()
    for m in messages:
        if m.type == "ai":
            for tc in m.tool_calls or []:
                seen.add(tc["id"])
        if m.type == "tool":
            assert m.tool_call_id in seen, f"orphan ToolMessage {m.tool_call_id}"


def test_short_history_is_untouched(small_budgets):
    """Under budget, nothing is summarized or trimmed."""
    sys = SystemMessage(content="SYS")
    hist = [HumanMessage(content="hi"), AIMessage(content="hello")]
    llm = FakeLLM()
    msgs, summary, count = ctx.manage_context(
        hist, system_message=sys, llm=llm, summary=None, summary_count=0
    )
    assert llm.calls == 0
    assert summary is None
    assert count == 0
    assert msgs == [sys, *hist]


def test_long_history_summarized_and_within_cap(small_budgets):
    sys = SystemMessage(content="SYS " + "s" * 100)
    hist = _turns(8)
    llm = FakeLLM()
    msgs, summary, count = ctx.manage_context(
        hist, system_message=sys, llm=llm, summary=None, summary_count=0
    )

    assert llm.calls > 0  # summarization happened
    assert count > 0  # some messages folded
    assert summary  # a running summary now exists
    assert msgs[0] is sys  # system message preserved at the front
    assert ctx._tokens(msgs) <= ctx.MAX_CONTEXT_TOKENS  # hard cap respected
    # The summary rides as a HumanMessage right after the system message.
    assert msgs[1].type == "human" and msgs[1].content.startswith(ctx._SUMMARY_PREFIX)
    _assert_no_orphan_tools(msgs)


def test_summarization_is_incremental(small_budgets):
    """Passing the cache back only summarizes the newly-added turns."""
    sys = SystemMessage(content="SYS " + "s" * 100)
    hist = _turns(8)
    llm = FakeLLM()
    _, summary, count = ctx.manage_context(
        hist, system_message=sys, llm=llm, summary=None, summary_count=0
    )
    calls_after_first = llm.calls

    hist2 = hist + [HumanMessage(content="followup " + "q" * 50)]
    _, _, count2 = ctx.manage_context(
        hist2, system_message=sys, llm=llm, summary=summary, summary_count=count
    )
    # Only a small number of extra summarizer calls for the new content.
    assert llm.calls - calls_after_first <= 2
    assert count2 >= count


def test_hard_trim_when_summarization_disabled(small_budgets):
    """With llm=None, the hard trim alone keeps the request under the cap."""
    sys = SystemMessage(content="SYS " + "s" * 100)
    hist = _turns(10)
    msgs, summary, count = ctx.manage_context(
        hist, system_message=sys, llm=None, summary=None, summary_count=0
    )
    assert summary is None
    assert count == 0
    assert ctx._tokens(msgs) <= ctx.MAX_CONTEXT_TOKENS
    assert msgs[0] is sys
    _assert_no_orphan_tools(msgs)


def test_divergent_cache_resets_safely(small_budgets):
    """A summary_count beyond the history length is reset rather than crashing."""
    sys = SystemMessage(content="SYS")
    hist = _turns(2)
    msgs, summary, count = ctx.manage_context(
        hist, system_message=sys, llm=FakeLLM(), summary="stale", summary_count=999
    )
    assert count <= len(hist)
    assert msgs[0] is sys


def test_cap_text():
    assert ctx.cap_text(None, 10) is None
    assert ctx.cap_text("short", 10) == "short"
    out = ctx.cap_text("x" * 100, 10)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_module_reimport_clean():
    # Ensure the module has no import-time side effects that break reloading.
    importlib.reload(ctx)

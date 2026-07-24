"""Tests for the per-turn workflow-step guard.

Two layers:

* ``TestGuardLogic`` — the pure decision functions in ``agents.workflow_guard``
  (milestone counting, opt-in detection, the pause decision), driven with hand-
  built message lists so they are deterministic and fast.
* ``TestGraphPause`` — the guard wired into a compiled ``create_agent_graph``,
  driven by a scripted fake LLM + stub milestone tools, proving that a second
  workflow step in one turn is NOT executed (the reported bug) while an explicit
  end-to-end request and the expert sub-agent both run through.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from mmm_framework.agents.graph import create_agent_graph
from mmm_framework.agents.workflow_guard import (
    next_paused_step,
    plan_pause,
    turn_milestone_count,
    wants_full_run,
)


def _ai_call(name: str, call_id: str = "c1") -> AIMessage:
    """An AIMessage requesting a single tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": call_id}])


# --------------------------------------------------------------------------- #
# Pure decision logic
# --------------------------------------------------------------------------- #
class TestGuardLogic:
    def test_wants_full_run_matches_opt_in_phrases(self):
        for text in (
            "run the whole pipeline",
            "just do everything",
            "build and fit a model end to end",
            "Run the WHOLE workflow please",
            "go all the way, don't stop",
        ):
            assert wants_full_run(text), text

    def test_wants_full_run_rejects_ordinary_asks(self):
        for text in (
            "I want to understand the total impact of media on my sales",
            "propose a DAG",
            "what is adstock?",
            "continue",
            "",
        ):
            assert not wants_full_run(text), text

    def test_first_milestone_of_turn_is_allowed(self):
        msgs = [
            HumanMessage("I want to understand the total impact of media on sales"),
            _ai_call("define_research_question"),
        ]
        assert turn_milestone_count(msgs) == 0
        assert next_paused_step(msgs, budget=1) is None

    def test_second_milestone_of_turn_pauses(self):
        # define ran, now the model wants to propose the DAG in the SAME turn.
        msgs = [
            HumanMessage("I want to understand the total impact of media on sales"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert turn_milestone_count(msgs) == 1
        label = next_paused_step(msgs, budget=1)
        assert label == "proposing the causal DAG"

    def test_non_milestone_pending_never_pauses(self):
        # Even after a milestone this turn, a read-only tool call is fine.
        msgs = [
            HumanMessage("look at the data"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("inspect_dataset", call_id="c2"),
        ]
        assert next_paused_step(msgs, budget=1) is None

    def test_opt_in_overrides_the_pause(self):
        msgs = [
            HumanMessage("build and fit a model end to end"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert next_paused_step(msgs, budget=1) is None

    def test_budget_zero_disables_guard(self):
        msgs = [
            HumanMessage("understand impact"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert next_paused_step(msgs, budget=0) is None

    def test_higher_budget_allows_more_steps(self):
        msgs = [
            HumanMessage("understand impact"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        # budget 2 → the 2nd milestone still runs; a 3rd would pause.
        assert next_paused_step(msgs, budget=2) is None

    def test_count_resets_across_human_turns(self):
        # A milestone in a PRIOR turn must not count toward this turn's budget.
        msgs = [
            HumanMessage("register the question"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            AIMessage(content="Registered. Next: propose a DAG."),
            HumanMessage("now propose the DAG"),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert turn_milestone_count(msgs) == 0
        assert next_paused_step(msgs, budget=1) is None

    def test_delegate_to_expert_counts_as_a_milestone(self):
        # Two-tier path: heavy steps arrive as delegate_to_expert hand-offs.
        msgs = [
            HumanMessage("configure and fit"),
            _ai_call("build_model_from_dag"),
            ToolMessage(content="ok", tool_call_id="c1", name="build_model_from_dag"),
            _ai_call("delegate_to_expert", call_id="c2"),
        ]
        assert next_paused_step(msgs, budget=1) == "the next analysis step"

    def test_env_budget_default_is_one(self, monkeypatch):
        monkeypatch.delenv("MMM_AGENT_WORKFLOW_STEP_BUDGET", raising=False)
        msgs = [
            HumanMessage("understand impact"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        # No explicit budget → reads the env default (1) → pauses on the 2nd.
        assert next_paused_step(msgs) == "proposing the causal DAG"

    def test_env_budget_override(self, monkeypatch):
        monkeypatch.setenv("MMM_AGENT_WORKFLOW_STEP_BUDGET", "0")
        msgs = [
            HumanMessage("understand impact"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert next_paused_step(msgs) is None

    def test_unparseable_env_budget_keeps_guard_on(self, monkeypatch):
        # A typo must NOT silently disable the safety guard — falls back to 1.
        monkeypatch.setenv("MMM_AGENT_WORKFLOW_STEP_BUDGET", "off")
        msgs = [
            HumanMessage("understand impact"),
            _ai_call("define_research_question"),
            ToolMessage(
                content="ok", tool_call_id="c1", name="define_research_question"
            ),
            _ai_call("propose_dag", call_id="c2"),
        ]
        assert next_paused_step(msgs) == "proposing the causal DAG"

    def test_batched_milestones_in_one_message_trip_the_guard(self):
        # A single AIMessage batching two milestones must NOT slip both through.
        msgs = [
            HumanMessage("I want to understand the total impact of media on sales"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "define_research_question", "args": {}, "id": "a"},
                    {"name": "propose_dag", "args": {}, "id": "b"},
                ],
            ),
        ]
        # committed=0, pending=2, 0+2 > budget(1) → trips; deferred step = the 2nd.
        assert next_paused_step(msgs, budget=1) == "proposing the causal DAG"
        run, defer, label = plan_pause(msgs, budget=1)
        assert [c["name"] for c in run] == ["define_research_question"]
        assert [c["name"] for c in defer] == ["propose_dag"]
        assert label == "proposing the causal DAG"

    def test_rejected_out_of_toolset_milestone_does_not_burn_budget(self):
        # Two-tier: a hallucinated heavy fit is rejected+corrected (no work done),
        # so it must not count toward the budget and falsely pause the recovery.
        msgs = [
            HumanMessage("configure and fit"),
            _ai_call("fit_mmm_model"),  # not in the orchestrator toolset
            ToolMessage(
                content="not available; delegate instead",
                tool_call_id="c1",
                name="fit_mmm_model",
            ),
            _ai_call("delegate_to_expert", call_id="c2"),
        ]
        valid = {"delegate_to_expert", "define_research_question", "propose_dag"}
        assert turn_milestone_count(msgs, valid_tools=valid) == 0
        assert next_paused_step(msgs, budget=1, valid_tools=valid) is None
        # Without the toolset filter the rejected fit would wrongly burn the budget.
        assert next_paused_step(msgs, budget=1) is not None

    def test_wants_full_run_ignores_negated_and_ambiguous(self):
        # Ambiguous bare fragments no longer count as opt-in.
        assert not wants_full_run("Can you build and fit an MMM on my sales data?")
        assert not wants_full_run("walk me through every step")
        assert not wants_full_run("list all the steps involved")
        # A negation before the opt-in phrase is NOT an opt-in.
        assert not wants_full_run(
            "Don't run the whole pipeline — just register the question"
        )
        assert not wants_full_run("no need to do everything, one step is fine")
        # Genuine opt-ins still match.
        assert wants_full_run("run the whole pipeline")
        assert wants_full_run("build and fit a model end to end")
        assert wants_full_run("keep going, don't stop")


# --------------------------------------------------------------------------- #
# The guard inside a compiled graph
# --------------------------------------------------------------------------- #
class _ScriptedLLM:
    """Returns pre-scripted AIMessages, ignoring its input. Enough surface for
    ``create_agent_graph`` (``bind_tools`` + ``invoke``); short histories never
    trigger ``manage_context`` summarization, so ``invoke`` is the only call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _stub_milestone_tools(executed: list[str]):
    """Executable stub tools named for real milestone tools; each records that it
    actually ran so the test can assert a deferred call did NOT execute."""

    def _mk(name: str):
        def _fn() -> str:
            executed.append(name)
            return f"{name}: ok"

        return StructuredTool.from_function(_fn, name=name, description=f"stub {name}")

    return [_mk("define_research_question"), _mk("propose_dag")]


class TestGraphPause:
    def _run(self, responses, user_text, role="orchestrator"):
        executed: list[str] = []
        tools = _stub_milestone_tools(executed)
        graph = create_agent_graph(
            _ScriptedLLM(responses),
            checkpointer=None,
            tools=tools,
            role=role,
            mode="mmm",
        )
        final = graph.invoke({"messages": [HumanMessage(user_text)]})
        return final, executed

    def test_pauses_before_the_second_workflow_step(self):
        # Exactly the reported scenario: a goal statement → define → (would propose).
        responses = [
            _ai_call("define_research_question"),
            _ai_call("propose_dag", call_id="c2"),
            AIMessage(content="unreached"),
        ]
        final, executed = self._run(
            responses, "I want to understand the total impact of media on my sales"
        )

        # The first step ran; the second was deferred, not executed.
        assert executed == ["define_research_question"]
        assert "propose_dag" not in executed

        msgs = final["messages"]
        # The pending propose_dag call got a "deferred" ToolMessage (no orphan
        # tool_call), and the turn ends on the assistant's pause explanation.
        deferred = [
            m for m in msgs if isinstance(m, ToolMessage) and m.name == "propose_dag"
        ]
        assert deferred and "Not executed" in deferred[0].content
        assert isinstance(msgs[-1], AIMessage)
        assert "paused before" in msgs[-1].content.lower()
        assert "proposing the causal DAG" in msgs[-1].content

    def test_opt_in_runs_both_steps(self):
        responses = [
            _ai_call("define_research_question"),
            _ai_call("propose_dag", call_id="c2"),
            AIMessage(content="Done — question registered and DAG proposed."),
        ]
        final, executed = self._run(
            responses, "build and fit a model end to end — do everything"
        )
        assert executed == ["define_research_question", "propose_dag"]
        assert "paused before" not in final["messages"][-1].content.lower()

    def test_expert_subagent_is_never_paused(self):
        # The expert is handed one task and must run it to completion; the guard
        # must not fire for role="expert".
        responses = [
            _ai_call("define_research_question"),
            _ai_call("propose_dag", call_id="c2"),
            AIMessage(content="Task complete."),
        ]
        final, executed = self._run(responses, "do the modeling setup", role="expert")
        assert executed == ["define_research_question", "propose_dag"]
        assert "paused before" not in final["messages"][-1].content.lower()

    def test_batched_first_message_runs_one_and_defers_the_rest(self):
        # A weak model batching two milestones in its FIRST message: exactly one
        # runs, the other is deferred, and the turn pauses.
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "define_research_question", "args": {}, "id": "a"},
                    {"name": "propose_dag", "args": {}, "id": "b"},
                ],
            ),
            AIMessage(content="unreached"),
        ]
        final, executed = self._run(
            responses, "I want to understand the total impact of media on my sales"
        )
        assert executed == ["define_research_question"]
        assert "propose_dag" not in executed

        deferred = [
            m
            for m in final["messages"]
            if isinstance(m, ToolMessage) and m.name == "propose_dag"
        ]
        assert deferred and "Not executed" in deferred[0].content
        assert "paused before" in final["messages"][-1].content.lower()
        assert "proposing the causal DAG" in final["messages"][-1].content

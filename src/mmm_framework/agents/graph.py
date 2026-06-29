import json
import os
from typing import Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from mmm_framework.agents.context import cap_text, manage_context
from mmm_framework.agents.state import AgentState
from mmm_framework.agents.tools import TOOLS

# Per-turn caps on the state blobs re-injected into the system message every
# turn. These don't accumulate, but unbounded they can each dominate a request.
_DATASET_INFO_MAX_CHARS = int(
    os.environ.get("MMM_AGENT_DATASET_INFO_MAX_CHARS", "4000")
)
_MODEL_SPEC_MAX_CHARS = int(os.environ.get("MMM_AGENT_MODEL_SPEC_MAX_CHARS", "16000"))


# Role preambles + the mode/role-aware prompt assembly now live in prompts.py.
# Re-exported here so any external reference to the old names keeps working.
from .prompts import (
    assemble_system_prompt,
)
from .modes import DEFAULT_MODE


def create_agent_graph(
    llm, checkpointer=None, *, tools=None, system_prompt=None, role=None, mode=None
):
    """
    Create and compile the LangGraph for the MMM Agent.

    Args:
        llm: A LangChain chat model instance (e.g. ChatGoogleGenerativeAI)
        checkpointer: Optional LangGraph checkpointer for state memory
        tools: Tool list to bind. Defaults to the full ``TOOLS`` list. The
            orchestrator passes ``ORCHESTRATOR_TOOLS`` (no heavy tools) and the
            expert sub-agent passes ``EXPERT_TOOLS`` (full power, no delegate).
        system_prompt: Explicit system-prompt override. When ``None`` the prompt
            is derived from ``role`` (see below).
        role: ``"orchestrator"`` prepends the delegation preamble, ``"expert"``
            prepends the expert preamble, ``None`` uses the bare shared prompt.
            Ignored when ``system_prompt`` is given.

    Returns:
        Compiled StateGraph
    """

    bound_tools = tools if tools is not None else TOOLS

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(bound_tools)

    # The effective system prompt is assembled PER TURN inside agent_node from
    # the session's modeling_mode (so a mode switch takes effect without a
    # recompile); role + an explicit system_prompt override are fixed per graph.

    def agent_node(state: AgentState):
        """The LLM reasoning node."""
        # State never persists SystemMessages; filter defensively so the running
        # history (and the summary_count index into it) stays consistent.
        history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

        # We want to give the LLM context of the current state so it can pass
        # `dataset_path` and `model_spec` to tools. Large blobs are capped: they
        # are re-injected every turn and otherwise inflate each request.
        state_context = "\n\nCURRENT STATE:\n"
        if state.get("dataset_path"):
            state_context += f"Dataset Path: {state['dataset_path']}\n"
        if state.get("dataset_info"):
            state_context += f"Dataset Info: {cap_text(state['dataset_info'], _DATASET_INFO_MAX_CHARS)}\n"
        if state.get("model_spec"):
            try:
                state_context += (
                    "Model Specification: "
                    f"{cap_text(json.dumps(state['model_spec']), _MODEL_SPEC_MAX_CHARS)}\n"
                )
            except Exception:
                pass
        if state.get("locked_fields"):
            state_context += (
                f"Locked Fields (user-set, do not silently change): "
                f"{', '.join(state['locked_fields'])}\n"
            )
        if state.get("model_status"):
            state_context += f"Model Status: {state['model_status']}\n"

        # State context lives on the system message to avoid "multiple
        # non-consecutive system messages" errors. The mode-specific prompt is
        # resolved from CURRENT STATE (falling back to the graph's default mode).
        effective_mode = state.get("modeling_mode") or mode or DEFAULT_MODE
        effective_system_prompt = assemble_system_prompt(
            mode=effective_mode, role=role, override=system_prompt
        )
        system_message = SystemMessage(content=effective_system_prompt + state_context)

        # Budget the request: summarize old turns + hard-trim to a token cap so a
        # long conversation can't exceed the model's per-request limit.
        messages, summary, summary_count = manage_context(
            history,
            system_message=system_message,
            llm=llm,
            summary=state.get("context_summary"),
            summary_count=state.get("context_summary_count"),
        )

        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "context_summary": summary,
            "context_summary_count": summary_count,
        }

    def should_continue(state: AgentState) -> Literal["tools", END]:
        """Determine if we should call tools or wait for user input."""
        messages = state["messages"]
        # A UI-driven state write (e.g. Data Studio commit, spec edit) can apply
        # as_node="agent" to a thread that has never chatted — no messages yet.
        # Treat that as "nothing to route" rather than indexing off an empty list.
        if not messages:
            return END
        last_message = messages[-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return END

    # Build the graph
    workflow = StateGraph(AgentState)

    # The orchestrator runs a fast/weak model that sometimes ignores the
    # "delegate, don't call heavy tools" instruction and calls an expert-only
    # tool directly (e.g. `execute_python`). The stock ToolNode then answers with
    # a generic "not a valid tool, try one of [60 names]" list, which a weak
    # model flails against. Wrap it so an out-of-toolset call gets a crisp,
    # corrective ToolMessage that names `delegate_to_expert` as the fix.
    _stock_tool_node = ToolNode(bound_tools)
    _valid_tool_names = {getattr(t, "name", None) for t in bound_tools}

    def tools_node(state: AgentState):
        if role != "orchestrator":
            return _stock_tool_node.invoke(state)
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        invalid = [c for c in calls if c.get("name") not in _valid_tool_names]
        if not invalid:
            return _stock_tool_node.invoke(state)

        corrections = [
            ToolMessage(
                tool_call_id=c.get("id", ""),
                name=c.get("name", "unknown"),
                content=(
                    f"`{c.get('name')}` is not available to you (the orchestrator); it is an "
                    f"EXPERT tool. Do NOT retry it. Instead call "
                    f'`delegate_to_expert(task="…")` with a precise, self-contained '
                    f"instruction describing the work you wanted `{c.get('name')}` to do "
                    f"(run code, fit, optimize, etc.). The expert shares this exact session "
                    f"— same dataset, model spec, warm kernel, and fitted model — so describe "
                    f"the task, don't pass data."
                ),
            )
            for c in invalid
        ]

        valid = [c for c in calls if c.get("name") in _valid_tool_names]
        if not valid:
            return {"messages": corrections}

        # Mixed batch: run the valid calls through the stock node on a shimmed
        # message (so it never sees the invalid ones), then append corrections so
        # every original tool_call_id is answered exactly once.
        try:
            shim = last.model_copy(update={"tool_calls": valid})
            sub_state = {**state, "messages": state["messages"][:-1] + [shim]}
            result = _stock_tool_node.invoke(sub_state)
            result_msgs = result["messages"] if isinstance(result, dict) else result
            return {"messages": list(result_msgs) + corrections}
        except Exception:
            # Never break the happy path: fall back to the stock node, which
            # still answers every call (with its generic error for invalid ones).
            return _stock_tool_node.invoke(state)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)

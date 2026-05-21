import json
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from mmm_framework.agents.state import AgentState
from mmm_framework.agents.tools import TOOLS


def create_agent_graph(llm, checkpointer=None):
    """
    Create and compile the LangGraph for the MMM Agent.
    
    Args:
        llm: A LangChain chat model instance (e.g. ChatGoogleGenerativeAI)
        checkpointer: Optional LangGraph checkpointer for state memory
        
    Returns:
        Compiled StateGraph
    """
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # System prompt: enforces the canonical 9-step scientific causal workflow
    # documented in docs/scientific-workflow-demo.html and mmm-methodology.tex.
    system_prompt = """You are an expert Marketing Mix Modeling (MMM) assistant
focused on **causal**, **pre-specified**, **scientifically defensible** modeling.
Your job is to walk the user through the 9-step canonical workflow below — in
order — and to make every modeling decision explicit and revisitable.

## The Canonical Workflow (always follow this order)

**Step 1 — Define the Question (BEFORE looking at data).** Call
`define_research_question` to pre-register the causal question, the business
decision it supports, the treatment variable, and the outcome variable. Do
this before EDA or model configuration. If the user dives straight to
"build me a model", politely ask the framing question first.

**Step 2 — Tell the Story of Your Data (DAG).** Inspect the data
(`inspect_dataset`), then call `propose_dag` to make causal structure
explicit: KPI, media channels (treatments), controls, mediators, and
named confounders. Then call `validate_causal_identification` to check via
the backdoor criterion whether the effect of interest is identified under
this DAG. If it's NOT identified, surface the open backdoor paths to the
user and propose adding the missing confounder as a control before moving
on. Use `record_assumption` for any causal claim that is not obvious from
the DAG (e.g. "no unmeasured macro confounders affecting both TV and
sales").

**Step 3 — Build the Model.** Use `configure_model`/`update_model_setting`.
Every non-default prior, adstock l_max, or saturation choice should be
followed by a `record_assumption` with the rationale.

**Step 4 — Prior Predictive Check.** Call `prior_predictive_check`
BEFORE fitting. If the implied KPI range is implausible (e.g. fraction
negative > 5%), tighten priors and re-check rather than fitting an
absurd model.

**Step 5 — Fit the Model.** `fit_mmm_model`.

**Step 6 — Computational Diagnostics.** `get_model_diagnostics`. If R-hat
or ESS are bad, stop and diagnose; do not move on to interpretation.

**Step 7 — Posterior Predictive Check.** Use `execute_python` to compare
fitted vs observed time series. Don't proceed to ROI claims if the model
clearly does not reproduce the data.

**Step 8 — Sensitivity Analysis.** Use `leave_one_out_decomposition` for
quick "what if this channel weren't there" questions. For genuine prior
sensitivity, RE-FIT with the perturbed spec — do not claim sensitivity has
been tested if you only ran the post-hoc reweighting.

**Step 9 — Communicate Results.** Report ROI with credible intervals,
state the adjustment set you conditioned on, and call `list_assumptions`
to remind the user of the assumption stack underlying the answer.

## Assumptions Discipline

- Every modeling decision that an honest reviewer would want to argue with
  becomes an entry in the assumptions log via `record_assumption`. Update
  (not delete) when revising — the change log is the point.
- Categories: research_question, causal_structure, data, functional_form,
  prior, identification, external_evidence, other.
- At the end of every fit, call `list_assumptions` once so the user sees
  the stack.

## Config / Persistence Tools

| Tool                          | When to use |
|-------------------------------|-------------|
| `save_config <name>`          | After any meaningful configuration is finalised |
| `load_config <name>`          | When user asks to restore a past config |
| `list_configs`                | When user asks what configs are saved |
| `delete_config <name>`        | When user asks to remove a config |
| `get_current_config`          | To verify the active spec |
| `update_model_setting`        | To change ONE setting (record_assumption after) |
| `get_session_status`          | Quick overview of dataset/config/fit state |
| `save_fitted_model <name>`    | After fitting, to keep the model |
| `load_fitted_model <name>`    | To analyse a previously fitted model |
| `list_saved_models`           | When user asks what models are on disk |
| `mark_workflow_step`          | Only to override inferred workflow status (e.g. mark Step 8 'skipped') |

## Visualisation Rules — IMPORTANT
- ALWAYS use Plotly for charts. NEVER matplotlib.
- `px` and `go` are pre-imported inside `execute_python`.
- Call `fig.show()` to render charts in the dashboard.

Be concise, proactive about logging assumptions, and format responses in Markdown.
If a user tries to skip a step (e.g. "just fit it"), do it — but explicitly note
in your reply which steps were skipped and what risk that creates.
"""
    
    def agent_node(state: AgentState):
        """The LLM reasoning node."""
        messages = list(state["messages"])

        # Inject system prompt if not present at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # We want to give the LLM context of the current state so it can pass `dataset_path` and `model_spec` to tools
        state_context = f"\n\nCURRENT STATE:\n"
        if state.get("dataset_path"):
            state_context += f"Dataset Path: {state['dataset_path']}\n"
        if state.get("dataset_info"):
            state_context += f"Dataset Info: {state['dataset_info']}\n"
        if state.get("model_spec"):
            try:
                state_context += f"Model Specification: {json.dumps(state['model_spec'])}\n"
            except Exception:
                pass
        if state.get("model_status"):
            state_context += f"Model Status: {state['model_status']}\n"

        # Append state context to the first SystemMessage to avoid "multiple non-consecutive system messages" errors
        messages[0] = SystemMessage(content=messages[0].content + state_context)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    def should_continue(state: AgentState) -> Literal["tools", END]:
        """Determine if we should call tools or wait for user input."""
        messages = state["messages"]
        last_message = messages[-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return END
        
    # Build the graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)

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
    
    # System prompt to guide the agent
    system_prompt = """You are an expert Marketing Mix Modeling (MMM) assistant.
    Your goal is to help users build, configure, and fit Bayesian MMMs using the mmm-framework.

    ## General Workflow

    1. **Data** — Check if the user has data. If not, offer `generate_synthetic_data`.
       Use `inspect_dataset` to discover column names, date range, and statistics before configuring.
    2. **Configure** — Use `configure_model` to set the KPI, channels, and controls.
       After configuring, use `get_current_config` to confirm the spec is correct.
    3. **Refine settings** — Use `update_model_setting` to change individual settings
       (e.g. inference.draws, trend.type, seasonality.yearly, media_channels.TV.adstock.type)
       without re-running `configure_model` from scratch.
    4. **Save config** — After finalising a configuration, always offer to `save_config` with a
       meaningful name. This lets the user reload it in future sessions.
    5. **Fit** — Once the user confirms, call `fit_mmm_model` with `dataset_path` and `model_spec`
       (JSON string) from state.
    6. **Save model** — After a successful fit, offer to `save_fitted_model` by name.
    7. **Analyse** — Use `get_roi_metrics`, `get_component_decomposition`, `get_model_diagnostics`,
       `get_adstock_weights`, `get_saturation_curves` to interpret results.
    8. **Ad-hoc code** — Use `execute_python` for custom analysis, data exploration, or bespoke plots.

    ## Config Management Tools

    | Tool | When to use |
    |------|-------------|
    | `save_config <name>`   | After any meaningful configuration is finalised |
    | `load_config <name>`   | When user asks to restore or reuse a past config |
    | `list_configs`         | When user asks what configs are saved |
    | `delete_config <name>` | When user asks to remove a saved config |
    | `get_current_config`   | To show or verify the active spec at any time |
    | `update_model_setting` | To change ONE setting without full reconfiguration |
    | `get_session_status`   | Quick overview of dataset/config/fit/saved state |

    ## Model Persistence

    | Tool | When to use |
    |------|-------------|
    | `save_fitted_model <name>`  | After fitting, if user wants to keep the model |
    | `load_fitted_model <name>`  | When user wants to analyse a previously fitted model |
    | `list_saved_models`         | When user asks what models are on disk |

    ## Visualisation Rules — IMPORTANT
    - ALWAYS use Plotly for charts. NEVER matplotlib.
    - `px` and `go` are pre-imported inside `execute_python`.
    - Call `fig.show()` to render charts in the dashboard.

    Always be concise, proactive about saving work, and format responses in Markdown.
    """
    
    def agent_node(state: AgentState):
        """The LLM reasoning node."""
        from langchain_core.messages import AIMessage, ToolMessage as TM

        messages = list(state["messages"])

        # ── Detect and repair broken state ──────────────────────────────────────
        # If the history ends with ToolMessages that aren't followed by an AI
        # response (can happen when the event stream was interrupted mid-graph),
        # trim back to the last valid Human→AI boundary so Anthropic won't 400.
        while messages:
            last = messages[-1]
            # Orphaned ToolMessage at the end: the AI never acknowledged it
            if isinstance(last, TM):
                # Find and remove the preceding orphaned AI(tool_call) + tool results
                # Walk back to remove the tool_call AI message and all its results
                i = len(messages) - 1
                while i >= 0 and isinstance(messages[i], TM):
                    i -= 1
                if i >= 0 and isinstance(messages[i], AIMessage) and messages[i].tool_calls:
                    messages = messages[:i]  # drop the orphaned AI+tool block
                else:
                    break
            else:
                break

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
        
        # If there are tool calls, go to tools
        if last_message.tool_calls:
            return "tools"
            
        # Otherwise, end and return to the user
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

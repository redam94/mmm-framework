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
    
    Follow this general process:
    1. Check if the user has data. If they don't, offer to generate synthetic data using `generate_synthetic_data`.
    2. Understand their dataset and determine the KPI, media channels, and control variables.
    3. Use the `configure_model` tool to lock in the model specification.
    4. Once configured, and the user confirms they want to run the model, use `fit_mmm_model`.
       IMPORTANT: You must pass `dataset_path` and `model_spec` (as a JSON string) from your current state to `fit_mmm_model`.
    5. After the model finishes fitting, report the summary to the user.
    6. When the user asks to interpret the results, explore the model's effectiveness, or view ROI, use the `get_roi_metrics` or `get_component_decomposition` tools to extract the insights and explain them clearly.
    7. If the user asks about model health, convergence, or diagnostics, use `get_model_diagnostics`.
    8. If the user asks about diminishing returns, saturation, half-life, carryover, or adstock, use `get_saturation_curves` and `get_adstock_weights`.
    9. Use `execute_python` to run arbitrary python code for data analysis, exploring the dataframe, or using the mmm_framework library directly.
    
    VISUALIZATION RULES — VERY IMPORTANT:
    - ALWAYS use Plotly for any charts or visualizations. NEVER use matplotlib.
    - The variables `px` (plotly.express) and `go` (plotly.graph_objects) are pre-imported for you in execute_python.
    - To display a chart in the dashboard, call `fig.show()` at the end of your code. This will render it interactively in the UI.
    - Example: `fig = px.line(df, x='date', y='sales', title='Sales Over Time'); fig.show()`
    
    If the user asks questions about MMM concepts (like adstock or saturation), explain them clearly using the tool outputs.
    Always be helpful, keep the user informed about what step you are on, and format your answers beautifully in Markdown.
    """
    
    def agent_node(state: AgentState):
        """The LLM reasoning node."""
        messages = state["messages"]
        
        # Inject system prompt if not present at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
            
        # We want to give the LLM context of the current state so it can pass `dataset_path` and `model_spec` to tools
        state_context = f"\n\nCURRENT STATE:\n"
        if state.get("dataset_path"):
            state_context += f"Dataset Path: {state['dataset_path']}\n"
        if state.get("dataset_info"):
            state_context += f"Dataset Info: {state['dataset_info']}\n"
        if state.get("model_spec"):
            state_context += f"Model Specification: {json.dumps(state['model_spec'])}\n"
        if state.get("model_status"):
            state_context += f"Model Status: {state['model_status']}\n"
            
        # Append state context to the first SystemMessage to avoid "multiple non-consecutive system messages" errors
        messages = list(messages)
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

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class ModelSpec(TypedDict, total=False):
    """
    Intermediate representation of the model specification that the agent builds up.
    """
    kpi: str | None
    kpi_display_name: str | None
    kpi_level: str | None  # 'national' or 'geo'

    media_channels: list[dict[str, Any]]
    control_variables: list[dict[str, Any]]

    # Optional parameters
    time_granularity: str | None  # 'weekly', 'daily'
    model_type: str | None  # 'pymc', 'numpyro'
    hierarchical: bool | None


def _last(a: Any, b: Any) -> Any:
    """Last-writer-wins reducer for scalar/dict state fields."""
    return b


def _merge_dashboard(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge dashboard updates from concurrent tool calls; b's keys take precedence."""
    return {**(a or {}), **(b or {})}


class AgentState(TypedDict):
    """
    The state of the MMM LangGraph agent.

    Every field that a tool can write to must use an Annotated reducer.
    LangGraph's ToolNode runs all tool_calls from one AIMessage concurrently;
    if two tools write to the same key in the same step without a reducer,
    LangGraph raises InvalidUpdateError.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Dataset information
    dataset_path: Annotated[str | None, _last]
    dataset_info: Annotated[str | None, _last]

    # Evolving model specification
    model_spec: Annotated[ModelSpec, _last]

    # Status of the modeling process
    model_status: Annotated[str, _last]

    # Optional generated outputs
    fit_results_summary: Annotated[str | None, _last]
    report_path: Annotated[str | None, _last]

    # Structured data for the frontend dashboard
    dashboard_data: Annotated[dict[str, Any], _merge_dashboard]

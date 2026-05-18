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
    

class AgentState(TypedDict):
    """
    The state of the MMM LangGraph agent.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Dataset information
    dataset_path: str | None
    dataset_info: str | None
    
    # Evolving model specification
    model_spec: ModelSpec
    
    # Status of the modeling process
    model_status: str  # e.g., "unconfigured", "configured", "fitting", "completed", "error"
    
    # Optional generated outputs
    fit_results_summary: str | None
    report_path: str | None
    
    # Structured data for the frontend dashboard
    dashboard_data: dict[str, Any]

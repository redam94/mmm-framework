import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage

from .spec_locks import apply_spec_patch, is_spec_patch


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


def _merge_spec(a: Any, b: Any) -> Any:
    """Reducer for ``model_spec``. Full spec dicts replace (configure_model /
    load_config / UI edits). Patch envelopes from ``update_model_setting`` are
    applied against the LATEST folded value — concurrent single-field updates
    in one ToolNode step compose instead of last-writer-wins clobbering each
    other (each tool only saw the pre-step snapshot). The stored value is
    always a concrete spec, never a patch envelope."""
    if is_spec_patch(b):
        return apply_spec_patch(a, b)
    return b


# Dashboard keys holding accumulating lists of content-addressed refs
# ({"id": ..., "title": ...}). A plain b-wins merge silently drops one side's
# appends when two concurrent tools each copy-append-write the list, so these
# keys are unioned (deduped by ref id). Writing None explicitly clears a list.
_REF_LIST_KEYS = ("plots", "tables")


def _union_refs(a: Any, b: Any) -> list:
    merged: list = []
    seen: set[str] = set()
    for item in list(a or []) + list(b or []):
        rid = item.get("id") if isinstance(item, dict) else None
        if rid is not None:
            if rid in seen:
                continue
            seen.add(rid)
        # Legacy inline figures (no "id") are kept as-is; exact-duplicate
        # detection isn't worth deep comparisons here.
        merged.append(item)
    return merged


def _merge_dashboard(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge dashboard updates from concurrent tool calls; b's keys take
    precedence, except ref-list keys (plots/tables) which are unioned."""
    a = a or {}
    b = b or {}
    merged = {**a, **b}
    for key in _REF_LIST_KEYS:
        if key in b and b[key] is None:
            merged[key] = None  # documented escape hatch: explicit clear
        elif key in a or key in b:
            merged[key] = _union_refs(a.get(key), b.get(key))
    # The dashboard mirror of model_spec composes patches the same way the
    # model_spec channel does (see _merge_spec) — otherwise concurrent
    # update_model_setting calls would clobber each other's fields here too.
    if is_spec_patch(b.get("model_spec")):
        merged["model_spec"] = apply_spec_patch(a.get("model_spec"), b["model_spec"])
    return merged


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
    model_spec: Annotated[ModelSpec, _merge_spec]

    # Dot-paths the user manually set (e.g. "inference.draws",
    # "media_channels.TV.adstock.l_max"). The LLM may not silently overwrite
    # these; a conflicting change is deferred into pending_spec_changes instead.
    locked_fields: Annotated[list[str], _last]

    # LLM-proposed changes to locked fields awaiting user confirmation. Each:
    # {path, current, proposed, reason, tool_call_id}. Mirrored into
    # dashboard_data so the frontend can render a confirmation modal.
    pending_spec_changes: Annotated[list[dict[str, Any]], _last]

    # Status of the modeling process
    model_status: Annotated[str, _last]

    # Optional generated outputs
    fit_results_summary: Annotated[str | None, _last]
    report_path: Annotated[str | None, _last]

    # Structured data for the frontend dashboard
    dashboard_data: Annotated[dict[str, Any], _merge_dashboard]

    # Context-management cache (see mmm_framework.agents.context). The agent
    # folds the oldest turns into ``context_summary`` once and keeps recent turns
    # verbatim, so the per-request token budget is enforced without
    # re-summarizing the whole backlog every turn. ``context_summary_count`` is
    # the number of leading (non-system) messages already folded in.
    context_summary: Annotated[str | None, _last]
    context_summary_count: Annotated[int, _last]

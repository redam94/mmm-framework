"""Agent tools for the Data Studio (staged upload → clean → commit).

Thin wrappers around :mod:`mmm_framework.data_studio.service` so the agent can
drive — and reproduce — the same replayable cleaning pipeline the Data Studio
UI builds. The staging manifest on disk is the single source of truth shared
with the UI: a pipeline the user built interactively is visible to the agent
via ``data_studio_status``, and a pipeline the agent sets shows up in the
studio on reopen. Helper imports from ``agents.tools`` happen inside functions
(the ``learning_tools`` direction) because ``tools.py`` imports
``DATA_STUDIO_TOOLS`` from here at module load.

Contract notes:
- ``set_data_studio_pipeline`` is a FULL-REPLACE of the step list, mirroring
  ``PUT /data-studio/{tid}/pipeline`` (order matters; steps replay from the
  raw file every time).
- ``commit_data_studio`` reuses :func:`data_studio.service.commit_core` — the
  exact code path behind ``POST /data-studio/{tid}/commit`` — so the emitted
  MFF-long dataset, spec roles, and lock reconciliation are identical whether
  the user or the agent commits. The only difference is delivery: the endpoint
  applies the update via ``aupdate_state``; the tool returns it as a
  ``Command`` update (with the ToolMessage the endpoint must not have).
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

InjectedConfig = Annotated[RunnableConfig, InjectedToolArg]


def _tid(config) -> str | None:
    from mmm_framework.agents.runtime import get_current_thread
    from mmm_framework.agents.tools import _activate_thread

    return get_current_thread() if config is None else _activate_thread(config)


def _msg(text: str, tool_call_id) -> Command:
    return Command(
        update={"messages": [ToolMessage(content=text, tool_call_id=tool_call_id)]}
    )


def _preview_markdown(preview: dict) -> str:
    """Compact markdown for a studio preview payload (columns + diff + issues)."""
    lines: list[str] = []
    diff = preview.get("diff") or {}
    if diff:
        lines.append(
            f"- Frame: {diff.get('rows_before', '?')} → "
            f"**{diff.get('rows_after', '?')} rows**, "
            f"{diff.get('cols_before', '?')} → {diff.get('cols_after', '?')} cols"
        )
    cols = preview.get("columns") or []
    if cols:
        names = [str(c.get("name", c)) if isinstance(c, dict) else str(c) for c in cols]
        shown = ", ".join(names[:15]) + ("…" if len(names) > 15 else "")
        lines.append(f"- Columns ({len(names)}): {shown}")
    roles = preview.get("roles") or {}
    if roles:
        by_role: dict[str, list[str]] = {}
        for col, role in roles.items():
            by_role.setdefault(str(role), []).append(str(col))
        lines.append(
            "- Roles: "
            + "; ".join(f"{r}: {', '.join(cs)}" for r, cs in sorted(by_role.items()))
        )
    for w in preview.get("warnings") or []:
        lines.append(f"- ⚠ {w}")
    return "\n".join(lines) if lines else "- (empty preview)"


def _studio_dashboard_patch(state: dict, tid: str) -> dict:
    """A dashboard_data patch carrying the light data_studio pointer (the same
    one the REST endpoints maintain via _studio_set_pointer)."""
    from mmm_framework.data_studio import service as studio

    manifest = studio.read_manifest(tid)
    pointer = None
    if manifest:
        try:
            result = studio.current_result(tid, manifest)
        except Exception:  # noqa: BLE001
            result = None
        pointer = studio.light_summary(tid, manifest, result)
    return {**(state.get("dashboard_data") or {}), "data_studio": pointer}


@tool
def data_studio_status(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Inspect the session's Data Studio staging area: the raw upload, the
    current transform pipeline (ordered steps), inferred/assigned column roles,
    and a preview of the transformed frame.

    Call this FIRST when the user mentions data they staged/cleaned in the
    Data Studio, before proposing pipeline changes — the staged pipeline is
    shared state between the UI and these tools.
    """
    from mmm_framework.data_studio import service as studio

    tid = _tid(config)
    manifest = studio.read_manifest(tid)
    if not manifest:
        return _msg(
            "No dataset is staged in the Data Studio for this session. Stage "
            "one with `stage_data_studio_file` (a workspace file) or ask the "
            "user to upload via the Data tab's 'Upload & clean data'.",
            tool_call_id,
        )
    try:
        result = studio.current_result(tid, manifest)
        preview = studio.preview_payload(tid, manifest, result)
    except Exception as exc:  # noqa: BLE001
        return _msg(
            f"Staged dataset exists but the pipeline fails to replay: {exc}. "
            "Fix or replace the pipeline with `set_data_studio_pipeline`.",
            tool_call_id,
        )
    steps = manifest.get("steps") or []
    lines = [
        "### Data Studio staging",
        f"- Raw upload: `{manifest.get('raw', {}).get('name')}` "
        f"(staging id `{manifest.get('staging_id')}`, "
        f"committed: {bool(manifest.get('committed'))})",
        f"- Pipeline: {len(steps)} step(s)"
        + (": " + " → ".join(str(s.get("op")) for s in steps) if steps else ""),
        _preview_markdown(preview),
    ]
    return _msg("\n".join(lines), tool_call_id)


@tool
def stage_data_studio_file(
    path: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Stage a file already in this session's workspace (e.g. a chat upload)
    into the Data Studio — REPLACING any prior staging. `path` is relative to
    the workspace root (see list_workspace_files). Roles are inferred
    heuristically; adjust them via `set_data_studio_pipeline(roles=...)`.
    Staging does NOT change the working dataset — `commit_data_studio` does.
    """
    from mmm_framework.agents import workspace as ws
    from mmm_framework.data_studio import service as studio

    tid = _tid(config)
    root = ws.thread_dir(tid)
    try:
        src = ws.safe_join(root, path)
    except ValueError as exc:
        return _msg(f"Error: {exc}", tool_call_id)
    if not src.exists() or not src.is_file():
        return _msg(f"Error: no such file in workspace: {path}", tool_call_id)
    if src.suffix.lower() not in (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"):
        return _msg(
            f"Error: `{path}` doesn't look like a dataset "
            "(.csv/.tsv/.txt/.xlsx/.xls/.parquet).",
            tool_call_id,
        )
    try:
        # Copy into the studio's raw dir so a later workspace cleanup can't
        # pull the staged file out from under the replayable pipeline.
        import shutil

        dest = ws.safe_join(studio.raw_dir(tid), src.name)
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        manifest = studio.init_manifest(
            tid, str(dest), src.name, "dataset", dest.stat().st_size
        )
        result = studio.current_result(tid, manifest)
        preview = studio.preview_payload(tid, manifest, result)
    except Exception as exc:  # noqa: BLE001
        return _msg(f"Could not stage `{path}`: {exc}", tool_call_id)
    return Command(
        update={
            "dashboard_data": _studio_dashboard_patch(state, tid),
            "messages": [
                ToolMessage(
                    content=(
                        f"Staged `{src.name}` in the Data Studio (replacing any "
                        "prior staging).\n" + _preview_markdown(preview)
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def set_data_studio_pipeline(
    steps: str,
    roles: str = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Replace the Data Studio's transform pipeline (FULL-REPLACE, ordered)
    and re-preview the frame. `steps` is a JSON list of transform steps —
    the same replayable ops the studio UI builds:

      [{"op": "rename", "from": "dt", "to": "date"},
       {"op": "parse_date", "column": "date"},
       {"op": "drop_columns", "columns": ["notes"]},
       {"op": "cast", "column": "spend", "dtype": "number"},
       {"op": "fill_missing", "columns": ["spend"], "strategy": "zero"},
       {"op": "drop_duplicates"},
       {"op": "filter_rows", "column": "geo", "operator": "!=", "value": "XX"},
       {"op": "date_range", "start": "2024-01-01"},
       {"op": "winsorize", "column": "sales", "periods": ["2024-07-01"], "cap_value": 120000},
       {"op": "impute", "column": "sales", "periods": ["2024-07-08"], "value": 90000},
       {"op": "event_dummy", "name": "promo_week", "periods": ["2024-11-25"]}]

    `roles` (optional JSON object) assigns column roles:
    {"<column>": "kpi"|"media"|"control"|"date"|"group"|"ignore"}.

    An unknown op or missing param is rejected with the reason; a data-level
    failure (e.g. a filter matching nothing) SKIPS that step and reports a
    warning. Nothing touches the working dataset until `commit_data_studio`.
    """
    from mmm_framework.data_studio import service as studio
    from mmm_framework.data_studio.transforms import TransformError

    tid = _tid(config)
    if not studio.read_manifest(tid):
        return _msg(
            "No dataset is staged — stage one first (stage_data_studio_file, "
            "or the Data tab's 'Upload & clean data').",
            tool_call_id,
        )
    try:
        parsed_steps = json.loads(steps) if steps else []
    except json.JSONDecodeError as exc:
        return _msg(f"Could not parse steps JSON: {exc}", tool_call_id)
    if not isinstance(parsed_steps, list):
        return _msg("`steps` must be a JSON LIST of step objects.", tool_call_id)
    parsed_roles = None
    if roles:
        try:
            parsed_roles = json.loads(roles)
        except json.JSONDecodeError as exc:
            return _msg(f"Could not parse roles JSON: {exc}", tool_call_id)
        if not isinstance(parsed_roles, dict):
            return _msg("`roles` must be a JSON object of column: role.", tool_call_id)
    try:
        preview = studio.set_pipeline(tid, parsed_steps, parsed_roles)
    except TransformError as exc:
        return _msg(f"Pipeline rejected: {exc}", tool_call_id)
    except Exception as exc:  # noqa: BLE001
        return _msg(f"Pipeline failed: {exc}", tool_call_id)
    return Command(
        update={
            "dashboard_data": _studio_dashboard_patch(state or {}, tid),
            "messages": [
                ToolMessage(
                    content=(
                        f"Pipeline set ({len(parsed_steps)} step(s)) and replayed.\n"
                        + _preview_markdown(preview)
                        + "\nCommit with `commit_data_studio` when it looks right."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def commit_data_studio(
    reason: str = None,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Promote the staged, cleaned frame to the session's working dataset —
    the exact commit the Data Studio UI performs: writes the MFF-long CSV,
    sets dataset_path + spec roles (kpi/media_channels/control_variables/
    time_granularity) through the same lock-respecting reconciliation, and
    refreshes the dataset dashboard. Confirm with the user before committing
    over an existing working dataset. `reason` is recorded on any spec-lock
    confirmations this triggers.
    """
    from mmm_framework.data_studio import service as studio

    tid = _tid(config)
    error, summary, update = studio.commit_core(dict(state or {}), tid, reason)
    if error:
        return _msg(f"Commit failed: {error}", tool_call_id)
    return Command(
        update={
            **update,
            "messages": [
                ToolMessage(
                    content=summary or "Committed the staged dataset.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


DATA_STUDIO_TOOLS = [
    data_studio_status,
    stage_data_studio_file,
    set_data_studio_pipeline,
    commit_data_studio,
]

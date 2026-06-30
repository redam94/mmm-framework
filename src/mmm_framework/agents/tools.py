import os
import copy
import json
import logging
import importlib
import threading
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import io
import contextlib
import traceback

from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

InjectedConfig = Annotated[RunnableConfig, InjectedToolArg]

from mmm_framework.agents.runtime import (
    MODEL_CACHE as _MODEL_CACHE,
    NAMESPACE_CACHE as _NAMESPACE_CACHE,
    set_current_thread,
    get_current_thread,
)
from mmm_framework.agents import workspace as _ws
from mmm_framework.agents import model_ops as _model_ops
from mmm_framework.agents.fitting import (
    _mff_config_from_spec,
    build_and_fit,
    unconsumed_prior_path,
)
from mmm_framework.agents.spec_locks import (
    get_at,
    make_spec_patch,
    merge_pending,
    reconcile_with_locks,
)
from mmm_framework.agents.kernels import (
    KernelContext,
    ExecuteResult,
    KernelManager,
    SubprocessKernel,
)
from mmm_framework.agents.container_kernel import ContainerKernel


def _activate_thread(config) -> str:
    """Pull the active thread_id from the injected RunnableConfig and mark it
    current (so the thread-scoped model cache + workspace resolve correctly even
    when the tool runs in an executor thread). Returns the thread_id."""
    tid = None
    try:
        tid = (config.get("configurable") or {}).get("thread_id") if config else None
    except Exception:
        tid = None
    set_current_thread(tid)
    return get_current_thread()


from mmm_framework import (
    load_mff,
)

# _build_prior + _mff_config_from_spec moved to agents/fitting.py (kernel-importable,
# no cycle) alongside build_and_fit; imported below.


from mmm_framework.reporting.helpers import (
    compute_saturation_curves_with_uncertainty,
    compute_marginal_roi,
)

# The fitted-model cache is thread-scoped + LRU-bounded; imported from
# agents.runtime as _MODEL_CACHE above so the existing `_MODEL_CACHE[...]` and
# `_MODEL_CACHE.get(...)` call sites (here and in causal_tools) keep working.

_MFF_DIMENSION_COLS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


# Canonical trend-type names live in model/trend_config.py (TrendType). LLMs
# often emit near-miss aliases ("piecewise_linear", "gp"); map them back so the
# fit, the display tools, and the dashboard widgets all see the canonical form.
# Spec-normalization helpers live in spec_normalize.py (extracted from this
# god-module, H4). Re-exported here for backward compatibility (tools.X / tests).
from .spec_normalize import (  # noqa: E402
    _dataset_variable_names,
    _normalize_spec_vars,
    _normalize_trend_type,
    _normalized_spec,
    _partition_latent_controls,
)


def _build_dataset_dashboard(df, ds_path: str) -> tuple[list[str], dict]:
    """Build both the text summary lines and the rich dashboard_data['dataset'] dict."""
    import pandas as pd

    lines = [f"### Dataset: `{ds_path}`\n"]
    lines.append(f"**Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns")

    date_range = None
    date_cols = [
        c
        for c in df.columns
        if any(k in c.lower() for k in ("date", "week", "period", "time"))
    ]
    if date_cols:
        try:
            dates = pd.to_datetime(df[date_cols[0]])
            date_range = {
                "min": str(dates.min().date()),
                "max": str(dates.max().date()),
            }
            lines.append(f"**Date range**: {date_range['min']} → {date_range['max']}")
        except Exception:
            pass

    variable_names: list[str] = []
    if "VariableName" in df.columns:
        variable_names = sorted(df["VariableName"].dropna().unique().tolist())
        lines.append(
            f"\n**Variable Names** ({len(variable_names)}): {', '.join(variable_names)}"
        )

    present_dims = [c for c in _MFF_DIMENSION_COLS if c in df.columns]
    column_stats: dict = {}
    active_dimensions: list[str] = []

    for col in present_dims:
        non_null = df[col].dropna()
        unique_count = int(non_null.nunique())
        if unique_count > 1:
            active_dimensions.append(col)
        counts = non_null.value_counts().head(20)
        column_stats[col] = {
            "unique": unique_count,
            "top_values": [
                {"value": str(v), "count": int(c)} for v, c in counts.items()
            ],
            "truncated": unique_count > 20,
        }
        lines.append(
            f"\n**{col}** ({unique_count} unique): {', '.join(str(v) for v in counts.index[:6])}"
        )

    if "VariableName" not in df.columns:
        num = df.select_dtypes(include="number")
        lines.append(f"\n**Numeric columns** ({len(num.columns)}):")
        for col in num.columns[:30]:
            s = num[col]
            lines.append(
                f"  - `{col}`: mean={s.mean():.3g}, min={s.min():.3g}, max={s.max():.3g}, non-zero={int((s != 0).sum())}"
            )
        if len(num.columns) > 30:
            lines.append(f"  … and {len(num.columns) - 30} more numeric columns")

    geographies: list[str] = []
    if "Geography" in df.columns:
        geographies = sorted(df["Geography"].dropna().unique().tolist())

    dataset_info = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "date_range": date_range,
        "variable_names": variable_names,
        "geographies": geographies if geographies else ["National"],
        "column_stats": column_stats,
        "active_dimensions": active_dimensions,
    }
    return lines, dataset_info


@tool
def generate_synthetic_data(
    state: Annotated[dict, InjectedState],
    n_weeks: Optional[int] = None,
    geographies: Optional[list[str]] = None,
    scenario: str = "realistic",
    seed: Optional[int] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Generate a synthetic Master Flat File (MFF) dataset for testing and demonstration.
    Use this when the user wants to test the framework but doesn't have their own data.

    Data comes from the framework's stress-test worlds (mmm_framework.synth):
    realistic marketing data with confounded budgets, noisy media, many controls,
    and a known causal ground truth written alongside as an answer key.

    Args:
        n_weeks: Number of weeks of data (scenario default if omitted; min 52).
        geographies: Optional list of region names (e.g. ["East", "West"]) for a
            geo panel. With a national scenario name this upgrades the world to
            a panel ("clean" -> "geo_clean", anything else -> "geo_heterogeneous").
        scenario: Which world to generate. Default "realistic" (7 channels, 13
            controls incl. confounders/a mediator/noise, near-collinear Radio+
            Print, low media SNR). Easier: "clean" (model's exact assumptions).
            Single-violation worlds: "unobserved_confounding", "reverse_causality",
            "multicollinearity", "adstock_misspec", "saturation_misspec",
            "time_varying_beta", "heavy_tailed_noise", "synergy", "spend_outliers",
            "mixed_data_errors", "negative_effect", "trend_break",
            "seasonality_misspec", "dense_controls". Panels: "geo_clean",
            "geo_heterogeneous", "geo_product".
        seed: Random seed (scenario default if omitted).

    Returns:
        A Command that updates the dataset_path in the state.
    """
    from ..synth import generate_mff

    try:
        df, truth = generate_mff(
            scenario,
            seed=seed,
            n_weeks=n_weeks,
            geographies=geographies,
        )
    except (KeyError, ValueError, TypeError) as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to generate synthetic data: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Write into the session workspace and expose an ABSOLUTE dataset_path so it
    # is readable by execute_python (which runs in the workspace) AND by tools
    # that run in the server cwd (fit/inspect), and is registered for download.
    tid = _activate_thread(config)
    try:
        out_dir = _ws.thread_dir(tid)
        before = _ws.snapshot_dir(out_dir)
        output_path = str(out_dir / "synthetic_mff_data.csv")
        truth_path = str(out_dir / "synthetic_truth.json")
    except Exception:
        before, output_path = {}, "synthetic_mff_data.csv"
        truth_path = "synthetic_truth.json"
    df.to_csv(output_path, index=False)
    with open(truth_path, "w") as fh:
        json.dump(truth, fh, indent=2)
    try:
        _ws.register_generated_files(tid, before, kind="dataset")
    except Exception:
        pass

    n_periods = df["Period"].nunique()
    info = (
        f"Generated synthetic data from the '{truth['scenario']}' world: "
        f"{len(df)} rows, {n_periods} weeks. Columns: {', '.join(df.columns.tolist())}"
    )
    info += f"\nScenario: {truth['description']}"
    if truth.get("violates"):
        info += f"\nDeliberately violates: {truth['violates']}"
    info += f"\nMedia channels: {', '.join(truth['channels'])}"
    if truth.get("geographies"):
        info += f"\nGeographies: {', '.join(truth['geographies'])}"
        if truth.get("products"):
            info += f"; products: {', '.join(truth['products'])}"
    else:
        info += "\nNational level data (no geographies)."
    info += (
        f"\nCausal ground truth (true per-channel contribution/ROAS) saved to "
        f"{os.path.basename(truth_path)} — consult it only AFTER fitting, to "
        "grade how well the model recovered the known answers."
    )

    dashboard_data = state.get("dashboard_data") or {}
    _, dataset_info = _build_dataset_dashboard(df, output_path)
    dashboard_data["dataset"] = dataset_info

    return Command(
        update={
            "dataset_path": output_path,
            "dataset_info": info,
            "messages": [ToolMessage(content=info, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


#: Refuse to materialize an unbounded external pull into the workspace (a
#: BigQuery query / GCS object can return arbitrarily many rows; this is a
#: disk-exhaustion guard, not a cost guard).
_MAX_EXTERNAL_ROWS = 5_000_000


def _persist_external_dataset(
    df, *, source_label: str, state: dict, config, tool_call_id: str
) -> Command:
    """Write a fetched DataFrame to the session workspace as the active dataset.

    Mirrors generate_synthetic_data: ABSOLUTE dataset_path, file registration,
    and a dashboard summary, so an externally-loaded dataset behaves exactly like
    an uploaded or synthetic one downstream.
    """
    if len(df) > _MAX_EXTERNAL_ROWS:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Refusing to load {len(df):,} rows (limit "
                            f"{_MAX_EXTERNAL_ROWS:,}). Narrow the pull — add a date/"
                            "WHERE filter or a LIMIT — and try again."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    tid = _activate_thread(config)
    try:
        out_dir = _ws.thread_dir(tid)
        before = _ws.snapshot_dir(out_dir)
        output_path = str(out_dir / "external_mff_data.csv")
    except Exception:
        before, output_path = {}, "external_mff_data.csv"
    df.to_csv(output_path, index=False)
    try:
        _ws.register_generated_files(tid, before, kind="dataset")
    except Exception:
        pass

    cols = ", ".join(map(str, df.columns.tolist()))
    info = f"Loaded {len(df)} rows from {source_label}. Columns: {cols}"
    if "VariableName" not in df.columns or "Period" not in df.columns:
        info += (
            "\nNote: this is not MFF long format yet (it needs Period + VariableName "
            "+ VariableValue columns). Use execute_python with "
            "`from mmm_framework.integrations.ad_platforms import spend_to_mff` to "
            "reshape it before configuring or fitting a model."
        )

    dashboard_data = state.get("dashboard_data") or {}
    try:
        _, dataset_info = _build_dataset_dashboard(df, output_path)
        dashboard_data["dataset"] = dataset_info
    except Exception:
        pass

    return Command(
        update={
            "dataset_path": output_path,
            "dataset_info": info,
            "messages": [ToolMessage(content=info, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


def _scrub_gcp_error(text: str) -> str:
    """Redact identifiers GCP SDK errors echo back before they reach the chat."""
    from ..integrations import scrub_cloud_error

    return scrub_cloud_error(text)


def _integration_error_command(exc: Exception, tool_call_id: str) -> Command:
    from ..integrations import IntegrationAuthError, MissingDependencyError

    if isinstance(exc, MissingDependencyError):
        msg = str(exc)
    elif isinstance(exc, IntegrationAuthError):
        # Never echo the credential path / identity back to the chat.
        msg = (
            "Authentication failed. GCS/BigQuery use Application Default "
            "Credentials — run `gcloud auth application-default login`, or set "
            "MMM_GCP_CREDENTIALS_PATH to a valid service-account JSON."
        )
    else:
        msg = _scrub_gcp_error(f"{type(exc).__name__}: {exc}") + (
            ". Check the project/credentials (GCS/BigQuery use Application "
            "Default Credentials)."
        )
    return Command(
        update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
    )


@tool
def load_from_bigquery(
    state: Annotated[dict, InjectedState],
    query: Optional[str] = None,
    table: Optional[str] = None,
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    location: Optional[str] = None,
    max_rows: Optional[int] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load the session dataset from BigQuery — a SQL ``query`` or a ``table`` id.

    The result becomes the active dataset. For modeling it should be (or be
    reshaped to) MFF long format: Period, VariableName, VariableValue (+ optional
    Geography/Product/Campaign). Shape it with SQL aliases, or load raw and use
    ``spend_to_mff`` in execute_python.

    Auth is Application Default Credentials (ADC); requires the optional
    ``mmm-framework[gcp]`` dependency group.

    Args:
        query: A SQL query to run (e.g. "SELECT date AS Period, ...").
        table: A ``dataset.table`` id to read instead of a query (SELECT *).
        project: GCP project (defaults to the ADC project).
        dataset: Default dataset for a bare table id.
        location: BigQuery location (e.g. US, EU).
        max_rows: Optional LIMIT for a table read.

    Returns:
        A Command that sets the dataset_path in state.
    """
    try:
        from ..integrations import BigQueryConfig, BigQueryDataSource

        src = BigQueryDataSource(
            BigQueryConfig(project=project, dataset=dataset, location=location)
        )
        df = src.read_dataframe(table, query=query, max_rows=max_rows)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a ToolMessage
        return _integration_error_command(exc, tool_call_id)
    label = f"BigQuery ({'query' if query else table})"
    return _persist_external_dataset(
        df, source_label=label, state=state, config=config, tool_call_id=tool_call_id
    )


@tool
def load_from_gcs(
    state: Annotated[dict, InjectedState],
    object_path: str,
    bucket: Optional[str] = None,
    project: Optional[str] = None,
    fmt: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load the session dataset from a Google Cloud Storage object (CSV/Parquet).

    The object becomes the active dataset. For modeling it should be (or be
    reshaped to) MFF long format. Auth is Application Default Credentials (ADC);
    requires the optional ``mmm-framework[gcp]`` dependency group.

    Args:
        object_path: Object name within the bucket (e.g. "exports/mmm_2024.csv").
        bucket: GCS bucket (defaults to the MMM_GCS_BUCKET env var).
        project: GCP project (defaults to the ADC project).
        fmt: Force "csv" or "parquet" (otherwise inferred from the extension).

    Returns:
        A Command that sets the dataset_path in state.
    """
    try:
        from ..integrations import GCSConfig, GCSDataSource, IntegrationError

        cfg = GCSConfig.from_env(bucket=bucket, project=project)
        if not cfg.bucket:
            raise IntegrationError(
                "No GCS bucket given; pass bucket=… or set the MMM_GCS_BUCKET env var."
            )
        df = GCSDataSource(cfg).read_dataframe(object_path, fmt=fmt)
    except Exception as exc:  # noqa: BLE001
        return _integration_error_command(exc, tool_call_id)
    return _persist_external_dataset(
        df,
        source_label=f"gs://{cfg.bucket}/{object_path}",
        state=state,
        config=config,
        tool_call_id=tool_call_id,
    )


@tool
def sync_data_connection(
    state: Annotated[dict, InjectedState],
    name: str,
    max_rows: Optional[int] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load a SAVED data connection by name into the session as the active dataset.

    Connections are set up once in the UI (Settings → Data connections) — a named
    GCS object or a BigQuery query/table — so you don't re-type the reference.
    This pulls the latest data and makes it the working dataset. Auth is ADC;
    requires the optional ``mmm-framework[gcp]`` dependency group.

    Args:
        name: The saved connection's name (within the current project).
        max_rows: Optional cap on rows pulled.

    Returns:
        A Command that sets the dataset_path in state.
    """
    from ..api import sessions as _sessions

    def _err(msg: str) -> Command:
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    tid = _activate_thread(config)
    sess = _sessions.get_session(tid) if tid else None
    pid = (sess or {}).get("project_id")
    if not pid:
        return _err(
            "This session isn't attached to a project, so it has no saved data "
            "connections. Use load_from_bigquery / load_from_gcs directly, or open "
            "the session inside a project."
        )
    conn = _sessions.get_data_connection_by_name(pid, name)
    if conn is None:
        avail = [c["name"] for c in _sessions.list_data_connections(pid)]
        return _err(
            f"No saved connection named {name!r}. "
            + (f"Available: {', '.join(avail)}." if avail else "None are saved yet.")
        )
    try:
        from ..integrations import read_connection_dataframe

        df = read_connection_dataframe(
            conn["kind"], conn.get("config") or {}, max_rows=max_rows
        )
    except Exception as exc:  # noqa: BLE001
        return _integration_error_command(exc, tool_call_id)
    try:
        _sessions.touch_data_connection_synced(conn["id"])
    except Exception:
        pass
    return _persist_external_dataset(
        df,
        source_label=f"connection '{name}' ({conn['kind']})",
        state=state,
        config=config,
        tool_call_id=tool_call_id,
    )


def _commit_spec(
    state: dict,
    candidate: dict,
    tool_call_id,
    *,
    success_msg: str,
    reason: str | None = None,
    set_status: str | None = None,
    patch_paths: list[str] | None = None,
) -> Command:
    """Apply an LLM-proposed ``candidate`` spec, honoring user-locked fields.

    Conflicting changes to locked fields are reverted and recorded in
    ``pending_spec_changes`` for the user to confirm; everything else applies.
    ``model_spec``, ``locked_fields`` and ``pending_spec_changes`` are mirrored
    into ``dashboard_data`` so the frontend can render locks + the modal.

    ``patch_paths``: when the candidate differs from the current spec at only
    these dot-paths (single-setting updates), the spec is written as a patch
    envelope instead of a full dict. The state reducer applies patches against
    the latest value, so concurrent ``update_model_setting`` calls in one
    ToolNode step compose instead of overwriting each other with their stale
    snapshots. Full-spec commits (configure_model / load_config) omit it.
    """
    current = state.get("model_spec") or {}
    locked = list(state.get("locked_fields") or [])

    merged, new_pending = reconcile_with_locks(
        candidate, current, locked, reason=reason, tool_call_id=tool_call_id
    )
    pending = merge_pending(state.get("pending_spec_changes"), new_pending)

    if new_pending:
        blocked = ", ".join(f"`{p['path']}`" for p in new_pending)
        msg = (
            f"{success_msg}\n\n⚠️ {len(new_pending)} change(s) touch fields the "
            f"user locked manually ({blocked}). These were NOT applied — they've "
            "been surfaced to the user for confirmation. Do not retry them unless "
            "the user explicitly asks again."
        )
    else:
        msg = success_msg

    # Patch mode: write only the touched paths (post-lock-reconciliation, so a
    # reverted locked value harmlessly re-asserts the current value).
    if patch_paths is not None:
        spec_update: Any = make_spec_patch(
            [{"path": p, "value": get_at(merged, p)} for p in patch_paths]
        )
    else:
        spec_update = merged

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["model_spec"] = spec_update
    dashboard_data["locked_fields"] = locked
    dashboard_data["pending_spec_changes"] = pending

    update: dict[str, Any] = {
        "model_spec": spec_update,
        "pending_spec_changes": pending,
        "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        "dashboard_data": dashboard_data,
    }
    if set_status is not None:
        update["model_status"] = set_status
    return Command(update=update)


@tool
def configure_model(
    state: Annotated[dict, InjectedState],
    kpi: str,
    kpi_level: str,
    media_channels: list[str],
    control_variables: list[str],
    reason: str = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Configure the MMM model specification.
    Call this tool once you have determined the KPI, media channels, and control variables from the user.

    Args:
        kpi: The name of the KPI variable (e.g., "Sales", "Conversions").
        kpi_level: Either "national" or "geo".
        media_channels: List of media channel variable names (e.g., ["TV", "Digital"]).
        control_variables: List of control variable names (e.g., ["Price_Index", "Distribution"]).
        reason: A short explanation of why you are (re)configuring. Shown to the
            user if any change collides with a field they locked manually.

    Latent baseline components: do NOT list "Trend" or "Seasonality" as
    control_variables — they are not dataset variables. They are handled by the
    model's built-in trend/seasonality components (this tool auto-diverts them
    and enables sensible defaults; tune via ``update_model_setting``).

    Returns:
        A Command that updates the model_spec in the state.
    """
    ds_vars = _dataset_variable_names(state.get("dataset_path"))

    def _reject(msg: str) -> Command:
        return Command(
            update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
        )

    if ds_vars is not None:
        if kpi not in ds_vars:
            return _reject(
                f"KPI `{kpi}` not found in the dataset. "
                f"Available variables: {', '.join(sorted(ds_vars))}"
            )
        bad_media = [m for m in media_channels if m not in ds_vars]
        if bad_media:
            return _reject(
                f"Media channel(s) not found in the dataset: {', '.join(bad_media)}. "
                f"Available variables: {', '.join(sorted(ds_vars))}"
            )
    real_controls, latent, missing = _partition_latent_controls(
        control_variables, ds_vars
    )
    if missing:
        return _reject(
            f"Control variable(s) not found in the dataset: {', '.join(missing)}. "
            f"Available variables: {', '.join(sorted(ds_vars or []))}. "
            "Latent components like trend/seasonality belong in the model's "
            "built-in `trend` / `seasonality` settings, not in control_variables."
        )

    model_spec = {
        "kpi": kpi,
        "kpi_level": kpi_level,
        "media_channels": [{"name": ch} for ch in media_channels],
        "control_variables": [{"name": cv} for cv in real_controls],
        "time_granularity": "weekly",
        "model_type": "numpyro",
    }
    notes = []
    for name, comp in latent:
        if comp == "trend":
            model_spec["trend"] = {"type": "linear"}
            notes.append(
                f"- `{name}` is not a dataset variable — mapped to the built-in "
                "trend component (type=linear; adjust via "
                "`update_model_setting('trend.type', ...)`)."
            )
        else:
            model_spec["seasonality"] = {"yearly": 2}
            notes.append(
                f"- `{name}` is not a dataset variable — mapped to the built-in "
                "seasonality component (yearly Fourier order 2; adjust via "
                "`update_model_setting('seasonality.yearly', ...)`)."
            )

    success_msg = "Model configured successfully."
    if notes:
        success_msg += (
            "\n\n**Latent components diverted from controls:**\n" + "\n".join(notes)
        )
    return _commit_spec(
        state,
        model_spec,
        tool_call_id,
        success_msg=success_msg,
        reason=reason,
        set_status="configured",
    )


@tool
def fit_mmm_model(
    state: Annotated[dict, InjectedState],
    dataset_path: str | None = None,
    method: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Build and fit the Bayesian MMM using the active model configuration.

    The model specification and dataset live in the session state — this tool
    reads them directly, so you do NOT need to reconstruct or pass the spec.
    Configure the model first with ``configure_model`` / ``update_model_setting``,
    then simply call this tool.

    Args:
        dataset_path: Optional override for the dataset path. Defaults to the
            dataset already loaded into the session state.
        method: Fit method. Defaults to ``"nuts"`` (full MCMC — use this for
            real inference). Pass an *approximate* method to fit in seconds when
            you just want to check whether a model is sensible (catch bad priors,
            divergent geometry, broken saturation/adstock) before paying for a
            full sample: ``"map"`` (fastest, point estimate), ``"advi"`` /
            ``"fullrank_advi"`` (variational), or ``"pathfinder"`` (needs the
            optional ``pymc_extras`` package). Approximate fits have UNCALIBRATED
            uncertainty — always re-fit with NUTS before reporting intervals or
            making spend decisions.

    Returns:
        A Command that updates the model_status and fit_results_summary in the state.
    """
    _activate_thread(config)
    # Dispatch the build+fit to the active kernel: in-process it fits here and
    # stores the model in MODEL_CACHE (unchanged behavior); in the subprocess it
    # fits IN the kernel so `mmm`/`results` become kernel globals (the model now
    # lives where execute_python + run_model_op run — removing the Phase-1
    # boundary). build_and_fit raises on failure -> InProcessKernel.fit raises ->
    # caught here; SubprocessKernel.fit returns the failure as {error}.
    #
    # The spec is read from state (server-authoritative) rather than passed by
    # the LLM, so manual user edits / locked fields can never be silently
    # reconstructed away. dataset_path falls back to the loaded dataset.
    try:
        spec = state.get("model_spec")
        if not spec or not spec.get("kpi"):
            raise ValueError("No model is configured yet. Call configure_model first.")
        path = dataset_path or state.get("dataset_path")
        if not path:
            raise ValueError("No dataset is loaded. Load a dataset before fitting.")
        spec = copy.deepcopy(dict(spec))
        _normalize_spec_vars(spec)  # accept bare-string channel/control lists
        if method is not None:
            spec.setdefault("inference", {})["method"] = str(method).lower()
        info = _KERNELS.get_or_spawn(get_current_thread()).fit(spec, path)
    except Exception as e:
        info = {"error": f"Error fitting model: {str(e)}"}

    if not isinstance(info, dict) or info.get("error"):
        error_msg = (
            info.get("error")
            if isinstance(info, dict)
            else "Error fitting model: unknown error"
        )
        dashboard_data = dict(state.get("dashboard_data") or {})
        dashboard_data["model_status"] = "error"
        dashboard_data["error"] = error_msg
        return Command(
            update={
                "model_status": "error",
                "fit_results_summary": error_msg,
                "messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )

    summary = info["summary"]
    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data.update(info.get("dashboard") or {})

    # Lineage stamping (host-side; the kernel only knows spec+path): dataset
    # fingerprint, spec hash, parent run, and the assumption stack at fit time.
    # These ride the model_run artifact and feed the /runs MLflow-style
    # timeline (api/runs.py).
    try:
        from mmm_framework.api import runs as _runs
        from mmm_framework.api import sessions as _sessions

        tid = get_current_thread()
        lineage = {
            "data_fingerprint": _runs.data_fingerprint(path),
            "spec_hash": _runs.spec_hash(spec),
            "parent_run_id": next(
                (
                    a["payload"].get("run_id")
                    for a in reversed(_sessions.list_artifacts(tid))
                    if a.get("kind") == "model_run"
                ),
                None,
            ),
            "assumptions": [
                {
                    "key": a.get("key"),
                    "version": a.get("version"),
                    "category": a.get("category"),
                    "rationale": (a.get("rationale") or "")[:300],
                }
                for a in _sessions.list_assumptions(tid)
            ],
        }
        for target in (info.get("model_run"), dashboard_data.get("model_run")):
            if isinstance(target, dict):
                target.update(lineage)
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "run-lineage stamping failed (fit result unaffected)"
        )

    # History metrics persistence (host-side, best-effort): enrich the
    # kernel-computed snapshot with registry calibration status and write the
    # run_metrics row that powers /projects/{id}/history + priorities.
    run_for_metrics = info.get("model_run") or dashboard_data.get("model_run")
    if isinstance(run_for_metrics, dict) and run_for_metrics.get("metrics"):
        from mmm_framework.api.history import persist_run_metrics

        persist_run_metrics(run_for_metrics, get_current_thread())

    update: dict[str, Any] = {
        "model_status": "completed",
        "fit_results_summary": summary,
        "report_path": info.get("report_path"),
        "dashboard_data": dashboard_data,
    }

    # Calibrated-fit close-out (host-side, best-effort): the spec's experiment
    # likelihoods were folded into THIS run, so transition the registry entries
    # completed → calibrated with the run link. The experiments STAY in the spec:
    # each fit rebuilds the model fresh from ``spec["experiments"]``, so clearing
    # them would make the next refit silently drop every prior calibration.
    # (Re-folding the same set is not double-counting — a fit includes each
    # experiment exactly once; double-counting would require a duplicate entry.)
    if spec.get("experiments"):
        try:
            from mmm_framework.api import sessions as _sessions

            run_id = (info.get("model_run") or {}).get("run_id")
            done = 0
            for eid in spec.get("experiment_ids") or []:
                try:
                    _sessions.transition_experiment(
                        eid,
                        "calibrated",
                        calibrated_run_id=run_id,
                        note=f"folded into fit {run_id}",
                    )
                    done += 1
                except ValueError:
                    pass  # already calibrated / deleted — not this fit's problem
            if done:
                summary += f" Registry updated: {done} experiment(s) marked calibrated."
                update["fit_results_summary"] = summary
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "experiment calibration close-out failed (fit result unaffected)"
            )

    update["messages"] = [ToolMessage(content=summary, tool_call_id=tool_call_id)]
    return Command(update=update)


# ── Model-op dispatch (Phase 2) ───────────────────────────────────────────────
# The interpretation tools keep their Command wrapping here (host/state side) and
# delegate the model-touching compute to the active kernel's `run_model_op`
# (PR-B). The kernel resolves the op from the `model_ops` registry and runs it
# where the model lives — in-process today (MODEL_CACHE), in the subprocess kernel
# once fits move there (PR-C). The no-model / unknown-op cases come back as the
# result's `error`, rendered below.
def _modelop_command(res: dict, state: dict, tool_call_id) -> Command:
    """Turn a model_ops result ``{content, dashboard, error, tables?}`` into a
    Command, merging any dashboard payload into ``dashboard_data`` and storing
    any structured table payloads as content-addressed dashboard refs."""
    if res.get("error"):
        return Command(
            update={
                "messages": [
                    ToolMessage(content=res["error"], tool_call_id=tool_call_id)
                ]
            }
        )
    content = res["content"]
    dash = res.get("dashboard") or {}
    tables = res.get("tables") or []
    plots = res.get("plots") or []
    update: dict = {}
    if dash or tables or plots:
        dashboard_data = dict(state.get("dashboard_data") or {})
        dashboard_data.update(dash)
        if tables:
            from mmm_framework.agents.tables import publish_tables, tables_note

            refs, dropped = publish_tables(tables, dashboard_data, get_current_thread())
            content += tables_note(refs, dropped)
        if plots:
            content += _publish_modelop_plots(
                plots, dashboard_data, get_current_thread()
            )
        update["dashboard_data"] = dashboard_data
    update["messages"] = [ToolMessage(content=content, tool_call_id=tool_call_id)]
    return Command(update=update)


def _publish_modelop_plots(plots: list, dashboard_data: dict, thread_id) -> str:
    """Store model-op plot dicts (``[{title, figure}]``) as content-addressed refs
    in ``dashboard_data['plots']`` (branding applied), mirroring the
    ``execute_python`` plot path so a model-op can return themed figures across
    the kernel boundary. Returns a short note for the tool message."""
    from mmm_framework.agents import workspace as _ws
    from mmm_framework.agents.branding import (
        apply_brand_colors,
        is_active as _brand_active,
        resolve_branding,
    )

    existing = dashboard_data.get("plots", [])
    refs: list[dict] = []
    dropped = 0
    branding = resolve_branding(thread_id)
    for p in plots:
        fig = p.get("figure") if isinstance(p, dict) else None
        title = (p.get("title") if isinstance(p, dict) else "") or ""
        if not isinstance(fig, dict):
            dropped += 1
            continue
        try:
            if _brand_active(branding):
                fig = apply_brand_colors(fig, branding)
            pid = _ws.store_plot(fig, thread_id)
        except (
            Exception
        ):  # noqa: BLE001 - oversize/invalid/store fail: drop, don't inline
            dropped += 1
            continue
        refs.append({"id": pid, "title": title})
    dashboard_data["plots"] = existing + refs
    note = ""
    if refs:
        note += f"\n\n*Generated {len(refs)} chart(s). View them in the Plots tab.*"
    if dropped:
        note += f"\n\n*{dropped} chart(s) omitted (too large or invalid).*"
    return note


@tool
def get_roi_metrics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Get the Return on Investment (ROI) and probability of profitability for each media channel.
    Call this tool when the user asks about the efficiency, ROI, ROAS, or cost-effectiveness of their media channels.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op("roi_metrics", {})
    return _modelop_command(res, state, tool_call_id)


@tool
def generate_slide_deck(
    state: Annotated[dict, InjectedState],
    client_name: Optional[str] = None,
    kpi_name: str = "Revenue",
    currency: str = "$",
    margin: Optional[float] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Generate a polished PowerPoint (.pptx) client readout from the fitted model.

    Fills the branded slide template with the model's numbers and charts —
    including each channel's breakthrough / optimal / saturation spend zones
    (defined on ROI and **marginal ROI**, not percent of response) — then writes a
    per-slide AI insight for each channel and a headline synthesized across the
    whole deck. Call this when the user asks for a slide deck, PowerPoint, client
    presentation, or readout deck. Requires a fitted model.

    Args:
        client_name: Client name for the cover and headline.
        kpi_name: KPI label (e.g. "Revenue", "Sales").
        currency: Currency symbol for money formatting.
        margin: Gross margin (0–1); sets a profit-maximizing break-even (1/margin).
    """
    _activate_thread(config)
    tid = get_current_thread()
    kern = _KERNELS.get_or_spawn(tid)
    opts = {
        "client": client_name,
        "kpi_name": kpi_name,
        "currency": currency,
        "margin": margin,
    }
    # 1) deterministic outline (model numbers + facts), 2) AI insights, 3) render
    r1 = kern.run_model_op("slide_deck_notes", opts)
    if r1.get("error"):
        return _modelop_command(r1, state, tool_call_id)
    notes = (r1.get("dashboard") or {}).get("slide_deck_notes") or []
    insights: dict = {}
    try:
        from mmm_framework.agents.deck_insights import generate_deck_insights
        from mmm_framework.agents.llm import build_llm

        insights = generate_deck_insights(notes, build_llm())
    except Exception as e:  # noqa: BLE001 — narrative is best-effort
        logger.warning("Deck insight generation failed: %s", e)
    r2 = kern.run_model_op("render_slide_deck", {**opts, "insights": insights})
    if r2.get("error"):
        return _modelop_command(r2, state, tool_call_id)
    deck = (r2.get("dashboard") or {}).get("slide_deck") or {}
    content = (
        f"Generated a {len(notes)}-slide PowerPoint deck"
        + (f" with {len(insights)} AI-written insights" if insights else "")
        + ". Download it from the Results tab (Slide deck)."
    )
    res = {
        "content": content,
        "dashboard": {
            "slide_deck_path": deck.get("path"),
            "slide_deck_filename": deck.get("filename", "agent_slide_deck.pptx"),
        },
        "error": None,
    }
    return _modelop_command(res, state, tool_call_id)


@tool
def get_estimands(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Compute the model's declarative estimands — the counterfactual causal lens.

    Returns each estimand the model declares (or the capability defaults:
    contribution_roi, marginal_roas, contribution — per channel) as a mean + 94%
    HDI, plus any user-declared estimands (e.g. awareness_lift,
    cost_per_conversion). Call this when the user asks for ROI/ROAS/contribution
    "estimands", a named or custom causal contrast, or the full set of declared
    measures. For the standard ROI table specifically, get_roi_metrics is fine.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "compute_estimands", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def get_component_decomposition(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Decompose the KPI (Sales) into its contributing components (Base/Trend vs Media vs Controls).
    Call this tool when the user asks what drove their sales, what percentage of sales came from media, or wants a decomposition breakdown.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "component_decomposition", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def get_model_diagnostics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Get MCMC convergence diagnostics for the fitted Bayesian model.
    Call this tool when the user asks about model convergence, divergences, R-hat, effective sample size, or diagnostic health.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "model_diagnostics", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def get_adstock_weights(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Get the learned adstock (carryover effect) weights for each media channel.
    Call this tool when the user asks about how long media effects last, decay rates, half-life, or carryover.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "adstock_weights", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def get_saturation_curves(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Get the saturation parameters (diminishing returns) for each media channel.
    Call this tool when the user asks about diminishing returns, saturation, scaling, or which channel to invest more in.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "saturation_curves", {}
    )
    return _modelop_command(res, state, tool_call_id)


# ── Model validation / verification tools (Phase 1) ──────────────────────────


@tool
def validate_model(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run the full model-validation battery and return a single trust verdict.
    Call this when the user asks whether the model is good/trustworthy/valid, or to "validate"/"check"/"verify" the fitted model. Runs convergence, posterior-predictive, residual, channel-identifiability, and unobserved-confounding-robustness checks; returns a consolidated verdict table + a PPC plot. Use the specific tools below for detail.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op("validate_model", {})
    return _modelop_command(res, state, tool_call_id)


@tool
def run_posterior_predictive_checks(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run posterior predictive checks (PPC): do datasets replicated from the posterior look like the observed KPI?
    Call this when the user asks about goodness-of-fit, posterior predictive checks, whether the model reproduces the data, or Bayesian p-values. Returns per-check Bayesian p-values + density-overlay and test-statistic plots.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "posterior_predictive_checks", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def run_residual_diagnostics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run residual diagnostics: autocorrelation (Durbin-Watson / Ljung-Box), heteroscedasticity (Breusch-Pagan), and normality (Shapiro / Jarque-Bera) of the model residuals.
    Call this when the user asks about residuals, autocorrelation, leftover structure, or whether the model is missing something. Returns a test table + residual/ACF/Q-Q plots.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "residual_diagnostics", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def run_channel_diagnostics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run per-channel identifiability diagnostics: VIF / collinearity clusters and per-channel R-hat/ESS.
    Call this when the user asks whether two channels are confounded/collinear, why a channel's ROI is unstable, or about multicollinearity / identifiability. Returns a VIF table + chart.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "channel_diagnostics", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def run_refutation_suite(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Test sensitivity to unobserved confounding: the Robustness Value per channel (how much hidden confounding it would take to nullify each effect).
    Call this when the user asks how robust the results are, whether an omitted variable could overturn a channel's effect, or for a refutation / sensitivity analysis. Returns a per-channel robustness table.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "refutation_suite", {}
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def run_cross_validation(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Run out-of-time cross-validation (rolling-origin backtest): refit on expanding windows and grade genuine out-of-sample forecasts vs naive baselines.
    Call this when the user asks about forecast accuracy, out-of-sample / hold-out performance, MAPE, or backtesting. NOTE: this REFITS the model several times and is slow (a minute or more); only run it when explicitly asked. Returns a model-vs-baseline accuracy table.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "cross_validation", {}
    )
    return _modelop_command(res, state, tool_call_id)


# ── Plot normalization + error formatting (shared with the kernel impls) ──────
# Extracted to module level (Phase 1 of technical-docs/agent-session-kernels.md)
# so BOTH the in-process execute_python path and the future subprocess kernel's
# startup file apply the SAME figure normalization, and so the load-bearing
# "Error executing code" text + NameError hint are formatted identically
# regardless of where the code ran. No behavior change vs. the prior in-function
# definitions — this is a pure extraction.

# Design-consistent palette (indigo / teal / amber / rose / emerald / violet / sky …)
_PALETTE = [
    "#4f46e5",
    "#0d9488",
    "#f59e0b",
    "#e11d48",
    "#059669",
    "#7c3aed",
    "#0284c7",
    "#b45309",
    "#6366f1",
    "#0f766e",
]
# Default Plotly Express / graph_objects colors we want to remap
_PLOTLY_DEFAULTS = {
    "#636efa": 0,
    "#ef553b": 1,
    "#00cc96": 2,
    "#ab63fa": 3,
    "#ffa15a": 4,
    "#19d3f3": 5,
    "#ff6692": 6,
    "#b6e880": 7,
    "#ff97ff": 8,
    "#fecb52": 9,
}


def _normalize_figure(fig):
    """Remap default colors, fix margins and suppress overlapping bar labels."""
    color_map: dict = {}
    next_idx = [0]

    def _remap(c: str) -> str:
        if not isinstance(c, str):
            return c
        key = c.lower()
        if key not in color_map:
            if key in _PLOTLY_DEFAULTS:
                color_map[key] = _PALETTE[_PLOTLY_DEFAULTS[key] % len(_PALETTE)]
            else:
                color_map[key] = _PALETTE[next_idx[0] % len(_PALETTE)]
                next_idx[0] += 1
        return color_map[key]

    for trace in fig.data:
        # Remap solid string colors on the marker
        mc = getattr(getattr(trace, "marker", None), "color", None)
        if isinstance(mc, str):
            trace.marker.color = _remap(mc)
        elif isinstance(mc, (list, tuple)):
            # Array of colors — remap each unique color
            trace.marker.color = [_remap(c) if isinstance(c, str) else c for c in mc]
        # Also remap line color
        lc = getattr(getattr(trace, "line", None), "color", None)
        if isinstance(lc, str):
            trace.line.color = _remap(lc)

    # Fix bar chart text overlap: hide labels that don't fit
    has_bar = any(getattr(t, "type", "") in ("bar",) for t in fig.data)

    fig.update_layout(
        colorway=_PALETTE,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1f2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f9fafb",
        margin=dict(t=90, l=70, r=40, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11, color="#374151"),
        ),
    )
    if has_bar:
        fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))

    return fig


def format_execution_error(
    traceback_str: str,
    *,
    is_name_error: bool = False,
    missing_name: str | None = None,
) -> str:
    """Format an ``execute_python`` failure identically for the in-process and
    (future) subprocess kernels.

    The literal ``Error executing code`` substring is **load-bearing**: the
    ``/chat`` capture loop keys ``is_error`` off it (``api/main.py``) and the
    portable ``.py`` export marks errored cells with it (``session_export.py``).
    When the failure is a ``NameError``, append the self-healing hint (the warm
    namespace persists only within a live session). The caller prepends any
    captured stdout.
    """
    out = f"Error executing code:\n{traceback_str}"
    if is_name_error:
        ref = f"`{missing_name}`" if missing_name else "a variable"
        out += (
            f"\n\nHint: variables persist across execute_python calls only "
            f"within a live session. {ref} from an earlier call is gone — the "
            f"kernel may have been reset (e.g. a server restart). The dataset is "
            f"auto-loaded as `df` and `dataset_path` is set, so reload/rebuild "
            f"what you need; or call `load_result('name')` if you saved it "
            f"earlier with `save_result('name', obj)`."
        )
    return out


class InProcessKernel:
    """Phase-1 default kernel: runs code in-process in the thread-scoped warm
    namespace (NAMESPACE_CACHE). A faithful move of execute_python's execution
    body behind the Kernel seam -- identical behavior. SubprocessKernel (PR3)
    will talk jupyter_client; this one delegates to the in-process namespace,
    whose per-thread state lives in NAMESPACE_CACHE, so a single shared
    instance is correct (per_session=False)."""

    per_session = False

    def __init__(self, thread_id=None):
        # thread_id is accepted (the manager passes it) but unused — in-process
        # per-thread state lives in NAMESPACE_CACHE/MODEL_CACHE via the ContextVar.
        self._thread_id = thread_id

    def execute(self, code, ctx):
        work_dir = Path(ctx.work_dir) if ctx.work_dir else None

        import pandas as pd
        import numpy as np

        # Force non-interactive backend so matplotlib works in server threads
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        # Persistent per-thread namespace ("warm kernel"): variables defined in one
        # execute_python call survive into the next within the same live process, so
        # the agent can build an analysis up incrementally. We compute the reserved
        # SYSTEM bindings into ``env`` below, then re-layer them on top of ``ns`` on
        # EVERY call (so a refit refreshes ``mmm``/``results`` and system names can't
        # be permanently shadowed). User-defined names in ``ns`` are left untouched.
        ns = _NAMESPACE_CACHE.namespace()

        env = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "matplotlib": matplotlib,
            "OUTPUT_DIR": str(work_dir) if work_dir is not None else os.getcwd(),
            "os": os,
            "json": json,
        }

        # Also pre-import plotly so the agent can use it easily
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            env["px"] = px
            env["go"] = go
        except ImportError:
            pass

        # Expose the full framework surface so the agent can reach ALL features
        # (extensions, analysis, calibration, reporting) without import boilerplate.
        try:
            import mmm_framework as mmf

            env["mmf"] = mmf
            env["mmm_framework"] = mmf
            # Eagerly import the key submodules so `mmf.analysis` / `mmf.reporting`
            # / `mmf.mmm_extensions` resolve (importing a submodule registers it as
            # an attribute of the package). Each is guarded + cached in sys.modules,
            # so the cost is paid at most once per process. mmm_extensions is lazy
            # for PyMC, so importing the package itself stays cheap.
            for _sub in ("analysis", "mmm_extensions", "reporting", "diagnostics"):
                try:
                    _mod = importlib.import_module(f"mmm_framework.{_sub}")
                    env[_sub] = _mod
                except Exception:
                    pass
            for _name in (
                "BayesianMMM",
                "ModelConfigBuilder",
                "MediaChannelConfigBuilder",
                "ControlVariableConfigBuilder",
                "KPIConfigBuilder",
                "PriorConfigBuilder",
                "AdstockConfigBuilder",
                "SaturationConfigBuilder",
                "SeasonalityConfigBuilder",
                "MFFConfigBuilder",
                "TrendConfigBuilder",
                "load_mff",
            ):
                if _name in globals():
                    env[_name] = globals()[_name]
        except Exception:
            pass

        mmm = ctx.mmm
        results = ctx.results
        if mmm is not None:
            env["mmm"] = mmm
        if results is not None:
            env["results"] = results

        # ── Durable named results (survive a kernel reset / server restart) ──────
        # The warm namespace is in-process only; these helpers persist *named*
        # objects to the on-disk workspace (parquet for tabular data, cloudpickle
        # otherwise) so the agent can reload them in a later session — the same
        # "disk is the durable fallback" pattern the model cache uses.
        _results_dir = (work_dir / "results") if work_dir is not None else None

        def _result_path(name, ext):
            # Concatenate the extension (do NOT use Path.with_suffix: a name like
            # "q4.2024" would have ".2024" treated as a suffix and be truncated to
            # "q4.parquet", silently colliding with "q4.2023").
            if _results_dir is None:
                raise RuntimeError(
                    "No workspace directory available for saving results."
                )
            _results_dir.mkdir(parents=True, exist_ok=True)
            return _results_dir / f"{_ws._safe_segment(str(name))}{ext}"

        def save_result(name, obj):
            """Persist ``obj`` under ``name`` so it survives a kernel reset / server
            restart. DataFrames/Series -> parquet (fallback pickle); anything else
            -> cloudpickle. Reload later with ``load_result(name)``. Returns the
            file path written."""
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
                try:
                    p = _result_path(name, ".parquet")
                    frame.to_parquet(p)
                    return str(p)
                except Exception:
                    pass  # pyarrow/fastparquet missing -> fall through to pickle
            try:
                import cloudpickle as _pk
            except Exception:
                import pickle as _pk
            p = _result_path(name, ".pkl")
            with open(p, "wb") as _fh:
                _pk.dump(obj, _fh)
            return str(p)

        def load_result(name):
            """Reload an object saved earlier with ``save_result(name)``."""
            pq = _result_path(name, ".parquet")
            if pq.exists():
                return pd.read_parquet(pq)
            pk = _result_path(name, ".pkl")
            if pk.exists():
                try:
                    import cloudpickle as _pk
                except Exception:
                    import pickle as _pk
                with open(pk, "rb") as _fh:
                    return _pk.load(_fh)
            raise FileNotFoundError(
                f"No saved result named {name!r}. Available: {list_saved_results()}"
            )

        def list_saved_results():
            """Names previously persisted with ``save_result`` in this session."""
            if _results_dir is None or not _results_dir.exists():
                return []
            return sorted(
                {
                    p.stem
                    for p in _results_dir.glob("*")
                    if p.suffix in (".parquet", ".pkl")
                }
            )

        env["save_result"] = save_result
        env["load_result"] = load_result
        env["list_saved_results"] = list_saved_results

        # ── Convenience dataset bindings ─────────────────────────────────────────
        # `dataset_path` always reflects the active dataset. `df` is auto-loaded from
        # it so the most common cross-cell reference works even on a cold kernel —
        # (re)loaded only when the active dataset CHANGES (tracked via a private
        # marker), so the analyst can reassign `df` (a filtered view) and have it
        # persist, while a freshly uploaded dataset still refreshes `df`.
        _ds_path = ctx.dataset_path
        if _ds_path:
            env["dataset_path"] = _ds_path
        if _ds_path and ns.get("__mmm_df_source__") != _ds_path:
            try:
                _p = str(_ds_path)
                if not os.path.isabs(_p) and work_dir is not None:
                    _cand = os.path.join(str(work_dir), _p)
                    if os.path.exists(_cand):
                        _p = _cand
                _too_big = (
                    os.path.exists(_p) and os.path.getsize(_p) > 250 * 1024 * 1024
                )
                if not _too_big and _p.lower().endswith(".csv"):
                    env["df"] = pd.read_csv(_p)
                    ns["__mmm_df_source__"] = _ds_path
                elif not _too_big and _p.lower().endswith(".parquet"):
                    env["df"] = pd.read_parquet(_p)
                    ns["__mmm_df_source__"] = _ds_path
            except Exception:
                pass  # auto-load is best-effort; the agent can load explicitly

        stdout_capture = io.StringIO()

        # Intercept Plotly show() calls — both pio.show(fig) and fig.show()
        captured_plots = []
        original_pio_show = None
        original_fig_show = None

        # show_table(df): render a DataFrame as a structured dashboard table
        # instead of printing it (mirrors the subprocess kernel's binding).
        captured_tables = []

        def _show_table(df, title=None, group="repl"):
            """Render a DataFrame as a formatted, sortable table in the
            dashboard (instead of printing it). Returns None."""
            from mmm_framework.agents.tables import df_to_table_json

            captured_tables.append(
                df_to_table_json(
                    df,
                    title=str(title or "Table"),
                    source="execute_python",
                    group=str(group or "repl"),
                )
            )

        env["show_table"] = _show_table

        try:
            import plotly.io as pio
            import plotly.basedatatypes as pbd

            original_pio_show = pio.show
            original_fig_show = pbd.BaseFigure.show

            def custom_show(fig_or_self, *args, **kwargs):
                # Called as pio.show(fig) or fig.show()
                fig = _normalize_figure(fig_or_self)
                captured_plots.append(json.loads(fig.to_json()))

            pio.show = custom_show
            pbd.BaseFigure.show = custom_show
        except ImportError:
            pass

        # Run inside the per-session workspace so EVERY file the agent writes (bare
        # relative name or via OUTPUT_DIR) lands there and becomes downloadable +
        # grep-able. The input-producing tools (generate_synthetic_data, uploads)
        # also write into this same directory and expose absolute dataset paths, so
        # reads by name or by dataset_path resolve correctly. The cwd is restored in
        # the finally block even if the executed code calls os.chdir itself.
        _prev_cwd = os.getcwd()
        try:
            if work_dir is not None:
                os.chdir(work_dir)
            with (
                contextlib.redirect_stdout(stdout_capture),
                contextlib.redirect_stderr(stdout_capture),
            ):
                # Re-layer the reserved system bindings on top of the persistent
                # namespace, then exec against the SINGLE namespace dict (one dict
                # keeps top-level def/class scoping correct — splitting into
                # globals/locals would silently break it).
                ns.update(env)
                exec(code, ns)
            output = stdout_capture.getvalue()
            if not output:
                output = "Code executed successfully with no output."
        except Exception as e:
            captured = stdout_capture.getvalue()
            prefix = (captured + "\n") if captured else ""
            output = prefix + format_execution_error(
                traceback.format_exc(),
                is_name_error=isinstance(e, NameError),
                missing_name=getattr(e, "name", None),
            )
        finally:
            # Always restore the working directory, even on error.
            try:
                os.chdir(_prev_cwd)
            except Exception:
                pass
            # Restore original show methods
            try:
                import plotly.io as pio
                import plotly.basedatatypes as pbd

                if original_pio_show is not None:
                    pio.show = original_pio_show
                if original_fig_show is not None:
                    pbd.BaseFigure.show = original_fig_show
            except Exception:
                pass
        return ExecuteResult(
            stdout=output,
            plots=captured_plots,
            tables=captured_tables,
            is_error="Error executing code" in output,
        )

    def run_model_op(self, op_name, kwargs):
        # In-process: the model lives in MODEL_CACHE; call the op directly (same
        # as the PR-A inline call), JSON-sanitized for parity with the subprocess
        # path. No-model + unknown-op are returned as data (error), never raised.
        from mmm_framework.agents.kernels import _json_safe

        op = _model_ops.OPS.get(op_name)
        if op is None:
            return {
                "content": None,
                "dashboard": {},
                "error": f"Unknown model op: {op_name}",
            }
        mmm = _MODEL_CACHE.get("fitted_model")
        if mmm is None and not getattr(op, "allow_unfitted", False):
            return {"content": None, "dashboard": {}, "error": _model_ops.NO_MODEL_MSG}
        res = op(mmm, _MODEL_CACHE.get("fit_results"), **(kwargs or {}))
        return _json_safe(res)

    def fit(self, model_spec, dataset_path):
        # In-process: fit here and deposit the model in MODEL_CACHE (unchanged
        # behavior). build_and_fit raises on failure -> the tool catches it.
        mmm, results, info = build_and_fit(model_spec, dataset_path)
        _MODEL_CACHE["fitted_model"] = mmm
        _MODEL_CACHE["fit_results"] = results
        return info

    def reset(self):
        _NAMESPACE_CACHE.reset()

    def shutdown(self):
        pass


from mmm_framework.agents.profile import default_kernel_impl as _default_kernel_impl

_KERNELS = KernelManager(
    _default_kernel_impl(),  # hosted -> sandboxed `container`; else in-process
    {
        "inprocess": InProcessKernel,
        "subprocess": SubprocessKernel,
        "container": ContainerKernel,
    },
)

# Reap any subprocess kernels on interpreter exit so child processes/fds aren't
# orphaned (the app lifespan also calls this on graceful shutdown).
import atexit as _atexit

_atexit.register(_KERNELS.shutdown_all)


@tool
def execute_python(
    state: Annotated[dict, InjectedState],
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Execute Python code for ad-hoc analysis or to drive mmm_framework directly.

    Runs inside this session's WORKSPACE directory, so EVERY file you write —
    whether by a bare name (`df.to_csv('result.csv')`) or under `OUTPUT_DIR` —
    is automatically saved, listed, grep-able (`list_workspace_files`,
    `grep_workspace`, `read_workspace_file`) and downloadable from the Files tab.
    Files produced by other tools (e.g. `synthetic_mff_data.csv`) and uploaded
    datasets are placed in this SAME workspace, so read them by their name or via
    `dataset_path` — e.g. `pd.read_csv('synthetic_mff_data.csv')` or
    `pd.read_csv(dataset_path)`.

    STATE PERSISTS across calls (a warm kernel): variables you define in one
    call are available in the next, so you can build an analysis up
    incrementally — exactly like cells in a Jupyter notebook. The dataset is
    auto-loaded as `df` (and its location as `dataset_path`), so you can use
    `df` straight away; reassign it (e.g. a filtered view) and your version
    persists. To keep an object across a server restart, call
    `save_result('name', obj)` and later `load_result('name')`
    (`list_saved_results()` shows what's saved). Call `reset_namespace` to wipe
    all variables for a fresh kernel.

    Pre-bound: `pd`, `np`, `plt`, `matplotlib`, `px`, `go`. The whole framework
    is reachable via `mmf` (the mmm_framework package — e.g. `mmf.analysis`,
    `mmf.mmm_extensions`, `mmf.reporting`) and the convenience names
    `BayesianMMM`, `ModelConfigBuilder`, `MediaChannelConfigBuilder`, etc. If a
    model is fitted, `mmm` (BayesianMMM) and `results` (MMMResults) are bound.
    These framework/system names refresh every call and cannot be permanently
    shadowed; your own variables are never touched. Call `library_reference()`
    to see the full menu of capabilities.

    Always print() what you want to see. Use Plotly + `fig.show()` for charts.
    For tabular results call `show_table(df, title=...)` — it renders a
    formatted, sortable table in the dashboard; don't print full DataFrames.
    """
    _activate_thread(config)
    thread_id = get_current_thread()
    try:
        work_dir = _ws.thread_dir(thread_id)
    except Exception:
        work_dir = None
    before_snapshot = _ws.snapshot_dir(work_dir) if work_dir is not None else {}

    ctx = KernelContext(
        thread_id=thread_id,
        work_dir=str(work_dir) if work_dir is not None else None,
        dataset_path=state.get("dataset_path") if isinstance(state, dict) else None,
        mmm=_MODEL_CACHE.get("fitted_model"),
        results=_MODEL_CACHE.get("fit_results"),
    )
    result = _KERNELS.get_or_spawn(thread_id).execute(code, ctx)
    output = result.stdout
    captured_plots = result.plots

    dashboard_data = dict(state.get("dashboard_data") or {})

    dropped_plots = 0
    if captured_plots:
        # Content-address each figure into the plot store and keep only a
        # lightweight {id, title} ref in state. This stops the full (heavy)
        # Plotly JSON from being re-sent on every turn / re-saved into the
        # LangGraph checkpoint; the frontend fetches each plot once by id and
        # the browser caches it permanently (immutable response). Falls back to
        # an inline figure if the store write fails (back-compat).
        existing_plots = dashboard_data.get("plots", [])
        plot_refs = []
        from mmm_framework.agents.branding import (
            apply_brand_colors,
            is_active as _brand_active,
            resolve_branding,
        )

        _branding = resolve_branding(thread_id)
        for fig in captured_plots:
            try:
                if _brand_active(_branding):
                    fig = apply_brand_colors(fig, _branding)
                pid = _ws.store_plot(fig, thread_id)
            except ValueError as exc:
                # Rejected (oversize / not a figure) — drop it. Inlining would
                # defeat the size cap and re-introduce the untrusted payload.
                dropped_plots += 1
                logging.getLogger("mmm_audit").warning(
                    "plot_rejected thread=%s reason=%s", thread_id, exc
                )
                continue
            except Exception:
                plot_refs.append(fig)  # store write failed — back-compat inline
                continue
            layout = fig.get("layout") or {}
            t = layout.get("title")
            title = (
                t.get("text")
                if isinstance(t, dict)
                else (t if isinstance(t, str) else "")
            )
            plot_refs.append({"id": pid, "title": title or ""})
        dashboard_data["plots"] = existing_plots + plot_refs

    # Structured tables captured via show_table(df) — content-addressed refs,
    # same trust model and streaming behavior as the plots above.
    table_refs, dropped_tables = [], 0
    if result.tables:
        from mmm_framework.agents.tables import publish_tables

        table_refs, dropped_tables = publish_tables(
            result.tables, dashboard_data, thread_id
        )

    # Register any files the code wrote to the workspace so they become listable
    # and downloadable from the frontend. The `results/` subdir (save_result
    # snapshots, reloaded by name) is excluded so it doesn't clutter deliverables.
    new_files = []
    if work_dir is not None:
        try:
            new_files = _ws.register_generated_files(
                thread_id, before_snapshot, kind="export", exclude_dirs=("results",)
            )
        except Exception:
            new_files = []

    content = f"### Python Execution Result\n```text\n{output}\n```"
    if captured_plots and plot_refs:
        content += f"\n\n*Generated {len(plot_refs)} Plotly interactive chart(s). View them in the Plots tab.*"
    if dropped_plots:
        content += (
            f"\n\n*{dropped_plots} chart(s) omitted (too large or not a valid figure).*"
        )
    if table_refs:
        content += (
            f"\n\n*{len(table_refs)} formatted table(s) rendered in the dashboard.*"
        )
    if dropped_tables:
        content += f"\n\n*{dropped_tables} table(s) omitted (too large or invalid).*"
    if new_files:
        names = ", ".join(f"`{f['name']}`" for f in new_files[:8])
        more = "" if len(new_files) <= 8 else f" (+{len(new_files) - 8} more)"
        content += (
            f"\n\n*Saved {len(new_files)} file(s) to your workspace: {names}{more}. "
            f"Download them from the Files tab.*"
        )

    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def reset_namespace(
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Reset the Python kernel: clear every variable defined in previous
    `execute_python` calls, giving a fresh namespace. The system names
    (`pd`, `np`, `mmf`, the builders, `df`, `dataset_path`,
    `save_result`/`load_result`, and `mmm`/`results` if a model is fitted) are
    re-provided automatically on the next `execute_python` call.

    Use this when accumulated variables are confusing the analysis, after a big
    context switch, or to free memory. Files you saved with `save_result` (and
    any workspace files) are on disk and are NOT affected.
    """
    _activate_thread(config)
    _KERNELS.reset(get_current_thread())
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=(
                        "Python kernel reset — all previously defined variables "
                        "were cleared. The dataset (`df`), framework (`mmf`), and "
                        "helpers are restored on the next `execute_python` call. "
                        "Saved results on disk are untouched."
                    ),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


_CONFIGS_DIR = "mmm_configs"
_MODELS_DIR = "mmm_models"


# ── Config management ──────────────────────────────────────────────────────────


@tool
def save_config(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Save the current model configuration to a named JSON file so it can be reloaded later.
    The name should be a short identifier like 'baseline', 'tv_heavy', or 'q4_2024'.
    """
    spec = state.get("model_spec")
    if not spec or not spec.get("kpi"):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No model configuration found in session. Configure a model first, then save it.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    spec = _normalized_spec(spec)  # tolerate bare-string channels/controls

    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(dict(spec), f, indent=2)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Configuration saved as **{name}** (`{path}`).\n\nChannels: {[c['name'] for c in spec.get('media_channels', [])]}",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def load_config(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Load a previously saved model configuration by name and apply it to the current session.
    This replaces the active model_spec but does NOT re-fit the model.
    """
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    if not os.path.exists(path):
        available = (
            sorted(f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".json"))
            if os.path.exists(_CONFIGS_DIR)
            else []
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Config **{name}** not found. Available configs: {available or 'none saved yet'}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    with open(path) as f:
        spec = json.load(f)
    spec = _normalize_spec_vars(spec)  # tolerate bare-string channels/controls

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["model_spec"] = spec

    channels = [c["name"] for c in spec.get("media_channels", [])]
    controls = [c["name"] for c in spec.get("control_variables", [])]
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Loaded config **{name}**.\n- KPI: {spec.get('kpi')}\n- Channels: {channels}\n- Controls: {controls}",
                    tool_call_id=tool_call_id,
                )
            ],
            "model_spec": spec,
            "model_status": "configured",
            "dashboard_data": dashboard_data,
        }
    )


def _scan_saved_configs() -> list[dict]:
    """Summaries of saved model-spec configs in mmm_configs/ (shared by
    list_configs and list_templates)."""
    out: list[dict] = []
    if not os.path.exists(_CONFIGS_DIR):
        return out
    for fname in sorted(os.listdir(_CONFIGS_DIR)):
        if not fname.endswith(".json"):
            continue
        name = fname[:-5]
        try:
            with open(os.path.join(_CONFIGS_DIR, fname)) as f:
                spec = json.load(f)
            out.append(
                {
                    "name": name,
                    "kpi": spec.get("kpi", "?"),
                    "n_channels": len(spec.get("media_channels", [])),
                    "n_controls": len(spec.get("control_variables", [])),
                }
            )
        except Exception:
            out.append({"name": name, "error": "could not read"})
    return out


@tool
def list_configs(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List all saved model configurations with a brief summary of each."""
    configs = _scan_saved_configs()
    rows = []
    for c in configs:
        if c.get("error"):
            rows.append(f"- **{c['name']}**: (could not read)")
        else:
            rows.append(
                f"- **{c['name']}**: KPI=`{c['kpi']}`, {c['n_channels']} channels, "
                f"{c['n_controls']} controls"
            )
    if not rows:
        content = (
            "No saved configurations yet. Use `save_config` after configuring "
            "a model."
        )
    else:
        content = "### Saved Configurations\n\n" + "\n".join(rows)
    return Command(
        update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    )


@tool
def delete_config(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Delete a saved configuration by name."""
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Config **{name}** not found.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    os.remove(path)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Config **{name}** deleted.",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def get_current_config(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Return a human-readable summary of the active model configuration."""
    spec = state.get("model_spec")
    if not spec or not spec.get("kpi"):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No model configuration is active.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    spec = _normalized_spec(spec)  # tolerate bare-string channels/controls

    lines = ["### Active Model Configuration\n"]
    lines.append(
        f"**KPI**: `{spec.get('kpi')}` (level: {spec.get('kpi_level','national')}, granularity: {spec.get('time_granularity','weekly')})"
    )

    channels = spec.get("media_channels", [])
    lines.append(f"\n**Media Channels** ({len(channels)}):")
    for ch in channels:
        ads = ch.get("adstock", {}).get("type", "geometric")
        sat = ch.get("saturation", {}).get("type", "hill")
        l_max = ch.get("adstock", {}).get("l_max", 8)
        lines.append(
            f"  - `{ch['name']}`: adstock={ads}(l_max={l_max}), saturation={sat}"
        )

    controls = spec.get("control_variables", [])
    if controls:
        lines.append(
            f"\n**Controls** ({len(controls)}): {', '.join(c['name'] for c in controls)}"
        )

    inf = spec.get("inference", {})
    if inf:
        lines.append(
            f"\n**Inference**: {inf.get('chains',4)} chains × {inf.get('draws',1000)} draws, tune={inf.get('tune',1000)}, target_accept={inf.get('target_accept',0.85)}"
        )

    trend = spec.get("trend", {})
    if trend:
        t_type = trend.get("type", "linear")
        extras = {k: v for k, v in trend.items() if k != "type"}
        lines.append(f"\n**Trend**: {t_type}" + (f" ({extras})" if extras else ""))

    seas = spec.get("seasonality", {})
    if any(seas.get(k, 0) for k in ("yearly", "monthly", "weekly")):
        lines.append(
            f"\n**Seasonality**: yearly={seas.get('yearly',0)}, monthly={seas.get('monthly',0)}, weekly={seas.get('weekly',0)}"
        )

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ]
        }
    )


@tool
def update_model_setting(
    state: Annotated[dict, InjectedState],
    setting_path: str,
    # A bare ``Any`` produces an untyped JSON schema that Google Gemini's
    # function-declaration validator rejects (it cannot represent "any type" and
    # emits a null property schema, breaking tool-binding for the WHOLE agent).
    # A scalar union is Gemini-valid (rendered as ``anyOf``) and pydantic's
    # smart-union keeps each JSON value's native type, so behavior is unchanged.
    value: Union[str, int, float, bool],
    reason: str = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Update a specific setting in the active model configuration using dot-notation.

    Examples:
      setting_path="inference.draws",           value=2000
      setting_path="inference.chains",          value=4
      setting_path="trend.type",                value="piecewise"
      setting_path="trend.n_changepoints",      value=10
      setting_path="seasonality.yearly",        value=4
      setting_path="kpi",                       value="Revenue"
      setting_path="time_granularity",          value="daily"
      setting_path="media_channels.TV.adstock.type",      value="delayed"
      setting_path="media_channels.TV.adstock.l_max",     value=13
      setting_path="media_channels.TV.saturation.type",   value="logistic"

    Intercept prior (Normal on standardized y — mu is in KPI standard deviations
    from the mean, so values beyond ±2 put the baseline outside the observed KPI
    range; defaults mu=0, sigma=0.5):
      setting_path="priors.intercept.mu",     value=0.0
      setting_path="priors.intercept.sigma",  value=0.3

    Trend priors (standardized-y scale; base slope applies to linear AND piecewise):
      setting_path="priors.trend.growth_prior_mu",          value=0.2
      setting_path="priors.trend.growth_prior_sigma",       value=0.5
      setting_path="priors.trend.changepoint_prior_scale",  value=0.5   (piecewise)
      setting_path="priors.trend.spline_prior_sigma",       value=1.0   (spline)
      setting_path="priors.trend.gp_lengthscale_prior_mu",  value=0.3   (gaussian_process)
      setting_path="priors.trend.gp_amplitude_prior_sigma", value=0.5   (gaussian_process)

    Seasonality amplitude priors (sigma of Normal on Fourier coefficients,
    standardized-y scale; default 0.3 — raise for strongly seasonal KPIs):
      setting_path="priors.seasonality.prior_sigma",         value=0.5  (all components)
      setting_path="priors.seasonality.yearly_prior_sigma",  value=0.8  (per-component override)
      setting_path="priors.seasonality.monthly_prior_sigma", value=0.2
      setting_path="priors.seasonality.weekly_prior_sigma",  value=0.2

    For media_channels and control_variables, use the channel/variable name as the key after the list name.

    If the user manually locked this field, the change is NOT applied silently —
    it is surfaced to the user for confirmation. Pass ``reason`` to explain why
    you want the change; it is shown in the confirmation prompt.
    """
    spec = state.get("model_spec")
    if not spec:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No active model configuration to update. Configure one first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    new_spec = copy.deepcopy(dict(spec))

    def _set(obj: Any, keys: list[str], val: Any) -> None:
        """Recursively walk obj using keys, setting the last key to val."""
        key = keys[0]
        rest = keys[1:]

        # List of dicts (media_channels / control_variables) — key is item name
        if isinstance(obj, list):
            item = next(
                (x for x in obj if isinstance(x, dict) and x.get("name") == key), None
            )
            if item is None:
                raise KeyError(f"No item named '{key}' in list")
            if not rest:
                raise ValueError(
                    "Cannot replace an entire list item; specify a sub-key"
                )
            _set(item, rest, val)
            return

        # Dict
        if not rest:
            obj[key] = val
            return

        # Navigate deeper; auto-create intermediate dicts
        if key not in obj or not isinstance(obj[key], (dict, list)):
            obj[key] = {}
        _set(obj[key], rest, val)

    if setting_path == "trend.type":
        value = _normalize_trend_type(value)

    parts = setting_path.split(".")

    # Refuse priors the model builder would silently drop (writing an unread
    # spec key looks like success but changes nothing — see agents/fitting.py).
    if parts[0] == "priors":
        err = unconsumed_prior_path(parts, value, new_spec)
        if err:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Rejected `{setting_path}`: {err}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

    try:
        _set(new_spec, parts, value)
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not update `{setting_path}`: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    return _commit_spec(
        state,
        new_spec,
        tool_call_id,
        success_msg=f"Updated **{setting_path}** → `{value}`",
        reason=reason,
        patch_paths=[setting_path],
    )


@tool
def get_session_status(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Return a comprehensive status report for the current session:
    dataset, model config, fit status, saved configs, and saved models.
    """
    tid = _activate_thread(config)
    lines = ["### Session Status\n"]

    # Modeling mode (selects the prompt framing + available tools). Surface a mode
    # ↔ loaded-model reconcile suggestion when a non-MMM family is loaded.
    try:
        from mmm_framework.agents.modes import (
            MODE_LABELS,
            normalize_mode,
            reconcile_mode_with_model,
        )
        from mmm_framework.api import sessions as _sessions_store

        _mode = normalize_mode(
            (_sessions_store.get_session(tid) or {}).get("modeling_mode")
        )
        lines.append(f"🧭 **Mode**: {MODE_LABELS.get(_mode, _mode)}")
        _gref = (
            (state.get("model_spec") or {}).get("garden_ref")
            if isinstance(state, dict)
            else None
        )
        if _gref and _gref.get("name"):
            try:
                _pid, _org = _garden_org_for(tid)
                _row = _sessions_store.get_garden_model(
                    org_id=_org, name=_gref["name"], version=_gref.get("version")
                )
                _kind = ((_row or {}).get("manifest") or {}).get("model_kind", "mmm")
                _recon = reconcile_mode_with_model(_mode, {"model_kind": _kind})
                if _recon.get("note"):
                    lines.append(f"   💡 {_recon['note']}")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # Dataset
    ds_path = state.get("dataset_path")
    if ds_path and os.path.exists(ds_path):
        lines.append(f"✅ **Dataset**: `{ds_path}`")
        ds_info = state.get("dataset_info", "")
        if ds_info:
            lines.append(f"   {ds_info.splitlines()[0]}")
    else:
        lines.append("❌ **Dataset**: not loaded")

    # Config
    spec = state.get("model_spec")
    if spec and spec.get("kpi"):
        n_ch = len(spec.get("media_channels", []))
        n_cv = len(spec.get("control_variables", []))
        lines.append(
            f"✅ **Config**: KPI=`{spec['kpi']}`, {n_ch} channels, {n_cv} controls"
        )
    else:
        lines.append("❌ **Config**: not set")

    # Fit
    status = state.get("model_status", "unconfigured")
    fitted = _MODEL_CACHE.get("fitted_model")
    if status == "completed" or fitted is not None:
        lines.append("✅ **Model**: fitted and ready")
    elif status == "fitting":
        lines.append("⏳ **Model**: currently fitting…")
    elif status == "configured":
        lines.append("⚙️  **Model**: configured, not yet fitted")
    else:
        lines.append("❌ **Model**: not configured")

    # Saved configs
    saved_cfgs: list[str] = []
    if os.path.exists(_CONFIGS_DIR):
        saved_cfgs = sorted(
            f[:-5] for f in os.listdir(_CONFIGS_DIR) if f.endswith(".json")
        )
    lines.append(
        f"\n💾 **Saved configs**: {', '.join(saved_cfgs) if saved_cfgs else 'none'}"
    )

    # Saved models
    saved_mdls: list[str] = []
    if os.path.exists(_MODELS_DIR):
        saved_mdls = sorted(
            d
            for d in os.listdir(_MODELS_DIR)
            if os.path.isdir(os.path.join(_MODELS_DIR, d))
        )
    lines.append(
        f"💾 **Saved models**: {', '.join(saved_mdls) if saved_mdls else 'none'}"
    )

    # Report
    rp = state.get("report_path")
    if rp and os.path.exists(rp):
        lines.append(f"\n📊 **Report**: `{rp}`")

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ]
        }
    )


@tool
def inspect_dataset(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Inspect the loaded dataset: show all column names, date range, and basic statistics.
    Use this to discover which columns can be used as media channels, KPI, or control variables.
    """
    import pandas as pd

    ds_path = state.get("dataset_path")
    if not ds_path or not os.path.exists(ds_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No dataset loaded. Generate or upload data first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        df = pd.read_csv(ds_path)
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to read dataset: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    lines, dataset_info = _build_dataset_dashboard(df, ds_path)
    dashboard_data = state.get("dashboard_data") or {}
    dashboard_data["dataset"] = dataset_info

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def save_fitted_model(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Save the currently fitted model to disk under a given name for future sessions.
    The name should be a short identifier like 'v1' or 'baseline_2024'.
    """
    _activate_thread(config)
    # Save runs where the model lives — in-process from MODEL_CACHE (unchanged:
    # API-cwd/mmm_models/<name>), or IN the subprocess kernel (work_dir/mmm_models/
    # <name>, the per-session workspace). No-model comes back as the result error.
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "save_model", {"name": name}
    )
    return _modelop_command(res, {}, tool_call_id)


def load_model_core(
    thread_id: str | None, name: str, spec: dict | None, dataset_path: str | None
) -> dict:
    """Load a saved fitted model into the session's model cache.

    Shared by the ``load_fitted_model`` tool AND the direct
    ``POST /sessions/{tid}/load-model`` endpoint (UI buttons load directly —
    no LLM round-trip). Returns ``{"ok": bool, "message": str}``.
    """
    if thread_id:
        set_current_thread(thread_id)
    save_dir = os.path.join(_MODELS_DIR, name)
    if not os.path.exists(save_dir):
        available = (
            sorted(
                d
                for d in os.listdir(_MODELS_DIR)
                if os.path.isdir(os.path.join(_MODELS_DIR, d))
            )
            if os.path.exists(_MODELS_DIR)
            else []
        )
        return {
            "ok": False,
            "message": f"Model **{name}** not found. Available: {available or 'none'}",
        }

    # The serializer rebuilds the live PyMC model against a COMPATIBLE panel, so
    # loading needs this session's dataset + model spec (it returns a single
    # model, not a (model, results) tuple — the prior `load(save_dir)` omitted the
    # required `panel` and mis-unpacked, so load ALWAYS failed).
    if not spec or not spec.get("kpi") or not dataset_path:
        return {
            "ok": False,
            "message": (
                f"To load **{name}**, this session needs the original dataset and its "
                "model configuration (the saved model is rebuilt against a compatible "
                "panel). Restore the dataset + model spec, then load again."
            ),
        }

    try:
        from mmm_framework.serialization import MMMSerializer

        panel = load_mff(dataset_path, _mff_config_from_spec(_normalized_spec(spec)))
        mmm = MMMSerializer.load(save_dir, panel)
        _MODEL_CACHE["fitted_model"] = mmm
        _MODEL_CACHE["fit_results"] = None
        return {
            "ok": True,
            "message": f"Model **{name}** loaded. You can now run analysis tools.",
        }
    except Exception as exc:
        return {"ok": False, "message": f"Load failed: {exc}"}


@tool
def load_fitted_model(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load a previously saved fitted model from disk by name, making it available for analysis tools."""
    tid = _activate_thread(config)
    res = load_model_core(
        tid,
        name,
        state.get("model_spec") if isinstance(state, dict) else None,
        state.get("dataset_path") if isinstance(state, dict) else None,
    )
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=res["message"], tool_call_id=tool_call_id)]
    }
    if res["ok"]:
        update["model_status"] = "completed"
    return Command(update=update)


@tool
def list_saved_models(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List all fitted models that have been saved to disk."""
    if not os.path.exists(_MODELS_DIR):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No saved models yet.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    models = sorted(
        d
        for d in os.listdir(_MODELS_DIR)
        if os.path.isdir(os.path.join(_MODELS_DIR, d))
    )
    if not models:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No saved models found.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    rows = []
    for m in models:
        meta_path = os.path.join(_MODELS_DIR, m, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                rows.append(
                    f"- **{m}**: saved {meta.get('saved_at','?')}, channels={meta.get('channel_names','?')}"
                )
            except Exception:
                rows.append(f"- **{m}**")
        else:
            rows.append(f"- **{m}**")

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="### Saved Models\n\n" + "\n".join(rows),
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


from mmm_framework.agents.causal_tools import CAUSAL_TOOLS


@tool
def generate_project_report(
    report_title: str,
    state: Annotated[dict, InjectedState] = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Generate a comprehensive self-contained HTML project report AND a Reveal.js HTML
    slideshow covering all findings from this MMM session: research question, data
    overview, model specification, KPI decomposition, ROI by channel, diagnostics,
    all captured charts, and the full assumptions log.

    Use this when the user asks for a report, summary document, presentation, slides,
    or wants to export findings.

    Args:
        report_title: Descriptive title, e.g. "UK Q1 2024 Media Mix Analysis".
    """
    from datetime import datetime, timezone
    from mmm_framework.agents.report_builder import (
        generate_html_report,
        generate_html_slides,
    )
    from mmm_framework.api import sessions as sessions_store_local

    date_str = datetime.now(timezone.utc).strftime("%d %B %Y")
    dashboard = dict((state or {}).get("dashboard_data") or {})

    thread_id = None
    if config and hasattr(config, "get"):
        thread_id = config.get("configurable", {}).get("thread_id")
    elif config and hasattr(config, "configurable"):
        thread_id = getattr(config.configurable, "thread_id", None)

    assumptions: list = []
    if thread_id:
        try:
            assumptions = sessions_store_local.list_assumptions(thread_id)
        except Exception:
            pass

    # Hosted: per-session under the workspace (allowed root); dev: legacy CWD.
    report_path = str(_ws.report_path("agent_project_report.html", thread_id))
    slides_path = str(_ws.report_path("agent_project_slides.html", thread_id))
    errors: list[str] = []

    from mmm_framework.agents.branding import (
        is_active as _brand_active,
        resolve_branding,
    )
    from mmm_framework.agents.report_builder import apply_branding_html

    _branding = resolve_branding(thread_id)
    if not _brand_active(_branding):
        _branding = None

    try:
        html = generate_html_report(report_title, date_str, dashboard, assumptions)
        html = apply_branding_html(html, _branding)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        errors.append(f"Report generation failed: {e}")
        report_path = None

    try:
        slides_html = generate_html_slides(
            report_title, date_str, dashboard, assumptions
        )
        slides_html = apply_branding_html(slides_html, _branding)
        with open(slides_path, "w", encoding="utf-8") as f:
            f.write(slides_html)
    except Exception as e:
        errors.append(f"Slideshow generation failed: {e}")
        slides_path = None

    if errors:
        summary = "Partial generation. Errors:\n" + "\n".join(errors)
    else:
        summary = (
            f"Generated project report **{report_title}** ({date_str}). "
            "View the full report and slideshow in the Artifacts tab."
        )

    dashboard["project_report_path"] = report_path
    dashboard["project_slides_path"] = slides_path

    return Command(
        update={
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard,
        }
    )


@tool
def generate_client_report(
    state: Annotated[dict, InjectedState],
    client_name: str = None,
    report_title: Optional[str] = None,
    analysis_period: Optional[str] = None,
    template: str = "augur",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Generate a clean, client-ready HTML report from the fitted model.

    Uses the project's confirmed client branding automatically (colors, client
    name) — check it first with `get_preferences`.

    template (default "augur"): the editorial "Media Performance Readout" — a
    narrative, evidence-coded CMO/planner deliverable with a masthead, KPI strip
    and recommendations, a Scale/Test/Hold/Reduce channel scorecard, ROI &
    uncertainty, marginal-vs-average return, saturation, reallocation, per-
    channel deep dives, carryover, a posterior-predictive fit-over-time + checks
    block, recommended tests and next steps, with AI-written insights woven
    through. Other templates produce the classic technical report: "full" (every
    section — the technical readout), "client", "presentation", "minimal". See
    `list_templates`.

    Call this after `fit_mmm_model` when the user wants to share results externally.

    Args:
        client_name: Client/company name (defaults to the branded client name).
        report_title: Optional report title (defaults to the readout title).
        analysis_period: Optional period string, e.g. "Q1–Q2 2024".
        template: Report template name ("augur" default).
    """
    _activate_thread(config)
    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found. Please fit a model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    from mmm_framework.agents.branding import (
        branding_to_channel_colors,
        branding_to_color_scheme,
        is_active as _brand_active,
        resolve_branding,
    )

    branding = resolve_branding(get_current_thread())
    if not client_name:
        client_name = (branding or {}).get("client_name") or "Client"

    tpl = (template or "augur").strip().lower()
    _AUGUR_ALIASES = {"augur", "media-readout", "media_readout", "readout"}
    is_augur = tpl in _AUGUR_ALIASES
    default_title = (
        "Media Performance Readout" if is_augur else "Marketing Mix Model Results"
    )
    title = report_title or default_title
    report_path = str(_ws.report_path("agent_client_report.html"))

    try:
        from mmm_framework.reporting.generator import ReportBuilder

        builder = (
            ReportBuilder()
            .with_model(mmm, results)
            .with_title(title)
            .with_client(client_name)
        )
        if is_augur:
            tpl = "augur"
            builder = builder.augur_readout()
            # CMO/planner narrative is enriched by the LLM when available; the
            # report falls back to grounded templated insights otherwise.
            try:
                from mmm_framework.agents.llm import build_llm

                builder = builder.with_llm(build_llm())
            except Exception:
                pass
        elif tpl == "minimal":
            builder = builder.minimal_report()
        elif tpl in ("full", "technical"):
            tpl = "full"
            builder = builder.enable_all_sections()
        elif tpl == "presentation":
            builder = builder.client_report()
            for section in ("saturation", "methodology"):
                builder = builder.disable_section(section)
        else:
            tpl = "client"
            builder = builder.client_report()
        if analysis_period:
            builder = builder.with_analysis_period(analysis_period)
        branded = False
        if _brand_active(branding):
            # The Augur readout keeps its editorial cream/ink/evidence palette;
            # branding only recolors the per-channel chart hues. The classic
            # templates take the full branded color scheme.
            if not is_augur:
                builder = builder.with_color_scheme(branding_to_color_scheme(branding))
            channels = [
                m.get("name")
                for m in _normalized_spec(state.get("model_spec")).get(
                    "media_channels", []
                )
                if m.get("name")
            ]
            if channels:
                builder = builder.with_channel_colors(
                    branding_to_channel_colors(branding, channels)
                )
            branded = True

        # Surface a conservative *default reallocation* in the report: reallocate
        # the current budget with each channel held within ±20% of today's spend
        # and inside the model's evidence range (so no channel is switched off and
        # nothing extrapolates). Best-effort — never block the report on it.
        try:
            from mmm_framework.planning import default_reallocation

            builder = builder.with_allocation(default_reallocation(mmm))
        except Exception:
            pass

        report = builder.build()
        report.to_html(report_path)
        summary = (
            f"Client report (template: {tpl}"
            + (", branded" if branded else "")
            + f") generated at `{report_path}`."
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to generate client report: {e}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["client_report_path"] = report_path

    return Command(
        update={
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def generate_model_defense_report(
    state: Annotated[dict, InjectedState],
    report_title: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Generate a one-click "model defense" report — the causal-rigor evidence
    behind the fitted model, as a CFO-readable "why you can trust this number"
    HTML artifact.

    Runs the causal refutation suite (placebo / negative-control / random-common-
    cause / data-subset), reads sampler convergence, counts any calibrated
    experiments, and renders a verdict (Robust / Qualified / Needs scrutiny) with
    plain-English per-test explanations and honest caveats. Uses confirmed client
    branding automatically. Call after `fit_mmm_model`.

    EXPENSIVE: the refutation suite REFITS the model once per test.
    """
    from dataclasses import replace

    _activate_thread(config)
    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found. Please fit a model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        from mmm_framework.agents.branding import is_active as _brand_active
        from mmm_framework.agents.branding import resolve_branding
        from mmm_framework.reporting import build_model_defense, model_defense_report
        from mmm_framework.validation.config import ValidationConfig
        from mmm_framework.validation.validator import ModelValidator

        cfg = replace(ValidationConfig.standard(), run_causal_refutation=True)
        summary = ModelValidator(mmm, results).validate(cfg)
        refutation = summary.causal_refutation  # may be None if it couldn't run
        convergence = summary.convergence
        n_cal = len(
            (_normalized_spec(state.get("model_spec")) or {}).get("experiments", [])
            or []
        )
        branding = resolve_branding(get_current_thread())
        payload = build_model_defense(
            refutation, convergence=convergence, n_calibrated_experiments=n_cal
        )
        html = model_defense_report(
            refutation,
            convergence=convergence,
            n_calibrated_experiments=n_cal,
            title=report_title or "Model Defense",
            brand=branding if _brand_active(branding) else None,
        )
        report_path = str(_ws.report_path("agent_model_defense.html"))
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to generate model-defense report: {e}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    msg = (
        f"Model-defense report generated at `{report_path}`. "
        f"Verdict: {payload['verdict']} "
        f"({payload['n_passed']}/{payload['n_tests']} refutation tests passed)."
    )
    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["model_defense_path"] = report_path
    dashboard_data["model_defense"] = payload
    return Command(
        update={
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def generate_client_slides(
    state: Annotated[dict, InjectedState],
    client_name: str,
    report_title: Optional[str] = None,
    analysis_period: Optional[str] = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Generate a clean, client-ready Reveal.js HTML slideshow.

    Compared to the internal project slides this version:
    • Omits MCMC parameters and diagnostic statistics (R̂, ESS, divergences)
    • Replaces "Model Diagnostics" with a plain "Model Validated" confirmation slide
    • Skips internal analysis charts (residuals, posterior predictive checks)
    • Formats channel names (underscores → spaces, title-case)
    • Shows analysis period weeks instead of raw data row count
    • Adds a confidentiality footer with the client name

    Call this when the user wants presentation-ready slides to share with a client.

    Args:
        client_name: Client/company name for the title slide and confidentiality footer.
        report_title: Optional slide deck title (defaults to "Marketing Mix Model Results").
        analysis_period: Optional period string e.g. "Q1–Q2 2024" (informational).
    """
    from datetime import datetime, timezone
    from mmm_framework.agents.report_builder import generate_html_slides
    from mmm_framework.api import sessions as sessions_store_local

    date_str = datetime.now(timezone.utc).strftime("%d %B %Y")
    dashboard = dict((state or {}).get("dashboard_data") or {})

    title = report_title or "Marketing Mix Model Results"
    slides_path = str(_ws.report_path("agent_client_slides.html"))

    thread_id = None
    if config and hasattr(config, "get"):
        thread_id = config.get("configurable", {}).get("thread_id")
    elif config and hasattr(config, "configurable"):
        thread_id = getattr(config.configurable, "thread_id", None)

    assumptions: list = []
    if thread_id:
        try:
            assumptions = sessions_store_local.list_assumptions(thread_id)
        except Exception:
            pass

    # Enrich dashboard with saturation curves and marginal ROI if model is available
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is not None:
        try:
            curves_result = compute_saturation_curves_with_uncertainty(mmm)
            dashboard["saturation_curves"] = {
                ch: r.to_dict() for ch, r in curves_result.items()
            }
        except Exception:
            pass

        roi_list = dashboard.get("roi_metrics") or []
        if roi_list:
            mroi_map = {}
            for r in roi_list:
                ch = r["channel"]
                try:
                    mroi_map[ch] = compute_marginal_roi(mmm, ch)
                except Exception:
                    pass
            if mroi_map:
                dashboard["marginal_roi"] = mroi_map

    try:
        slides_html = generate_html_slides(
            title,
            date_str,
            dashboard,
            assumptions,
            client_mode=True,
            client_name=client_name,
        )
        from mmm_framework.agents.branding import (
            is_active as _brand_active,
            resolve_branding,
        )
        from mmm_framework.agents.report_builder import apply_branding_html

        _branding = resolve_branding(thread_id)
        if _brand_active(_branding):
            slides_html = apply_branding_html(slides_html, _branding)
        with open(slides_path, "w", encoding="utf-8") as f:
            f.write(slides_html)

        has_curves = bool(dashboard.get("saturation_curves"))
        has_mroi = bool(dashboard.get("marginal_roi"))
        extras = []
        if has_curves:
            extras.append("S-curves")
        if has_mroi:
            extras.append("mROI vs avg ROI")
        if dashboard.get("roi_metrics"):
            extras.append("channel performance")
        extras_str = (", ".join(extras) + " slides added; ") if extras else ""
        summary = (
            f"Client slides generated at `{slides_path}`. "
            f"{extras_str}"
            f"MCMC diagnostics and internal charts excluded; "
            f"channel names formatted; confidentiality footer added for **{client_name}**."
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to generate client slides: {e}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["client_slides_path"] = slides_path

    return Command(
        update={
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data,
        }
    )


# ══════════════════════════════════════════════════════════════════════════
#  Workspace filesystem tools (req 7 — see & grep output files)
# ══════════════════════════════════════════════════════════════════════════


@tool
def list_workspace_files(
    config: InjectedConfig = None,
    subdir: str = "",
) -> str:
    """List the files in this session's workspace directory (where execute_python
    saves output: reports, CSVs, plots). Optionally restrict to a subdirectory.
    Returns a tree with sizes."""
    tid = _activate_thread(config)
    root = _ws.thread_dir(tid)
    try:
        base = _ws.safe_join(root, subdir) if subdir else root
    except ValueError as exc:
        return f"Error: {exc}"
    if not base.exists():
        return f"(workspace empty — no files yet under {subdir or '.'})"
    lines = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            lines.append(f"  {rel}  ({size:,} bytes)")
    if not lines:
        return "(workspace empty — no files yet)"
    return f"Workspace files for this session ({len(lines)}):\n" + "\n".join(lines)


@tool
def read_workspace_file(
    path: str,
    config: InjectedConfig = None,
    max_bytes: int = 20000,
) -> str:
    """Read a text file from this session's workspace directory. `path` is
    relative to the workspace root. Truncates to max_bytes."""
    tid = _activate_thread(config)
    root = _ws.thread_dir(tid)
    try:
        target = _ws.safe_join(root, path)
    except ValueError as exc:
        return f"Error: {exc}"
    if not target.exists() or not target.is_file():
        return f"Error: no such file in workspace: {path}"
    try:
        data = target.read_bytes()[: max(1, max_bytes)]
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {path}: {exc}"
    suffix = "\n…(truncated)…" if target.stat().st_size > max_bytes else ""
    return f"### {path}\n```\n{text}{suffix}\n```"


@tool
def grep_workspace(
    pattern: str,
    config: InjectedConfig = None,
    glob: str = "*",
    max_results: int = 100,
) -> str:
    """Search this session's workspace files for a regex `pattern` (like grep).
    Optionally restrict to files matching `glob` (e.g. '*.csv'). Returns
    file:line: matched-line hits."""
    import re

    tid = _activate_thread(config)
    root = _ws.thread_dir(tid)
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"
    hits = []
    for p in sorted(root.rglob(glob)):
        if not p.is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if rx.search(line):
                        rel = p.relative_to(root)
                        hits.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                        if len(hits) >= max_results:
                            break
        except (OSError, UnicodeError):
            continue
        if len(hits) >= max_results:
            break
    if not hits:
        return f"No matches for /{pattern}/ in workspace (glob={glob})."
    more = "" if len(hits) < max_results else f"\n…(capped at {max_results})"
    return f"{len(hits)} match(es):\n" + "\n".join(hits) + more


# ══════════════════════════════════════════════════════════════════════════
#  Knowledge-base tools (req 2/3 — project-level RAG)
# ══════════════════════════════════════════════════════════════════════════


@tool
def search_knowledge_base(
    query: str,
    config: InjectedConfig = None,
    top_k: int = 6,
) -> str:
    """Search the PROJECT knowledge base (documents the user uploaded for
    context) for passages relevant to `query`. Use this whenever the user refers
    to their own data dictionary, brief, prior analysis, or domain docs.
    Returns the top matching snippets with their source document."""
    from mmm_framework.api import sessions as sessions_store
    from mmm_framework.agents import knowledge_base as kb

    tid = _activate_thread(config)
    project_id = sessions_store.resolve_project_id(tid)
    try:
        results = kb.search(project_id, query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return f"Knowledge base search failed: {exc}"
    if not results:
        return (
            "No relevant passages found in the project knowledge base "
            "(it may be empty — the user can add documents in the Knowledge tab)."
        )
    out = [f"Top {len(results)} knowledge-base passages for: {query!r}\n"]
    for r in results:
        out.append(
            f"— **{r['document']}** (chunk {r['chunk_index']}, score {r['score']}):\n"
            f"  {r['text'].strip()[:800]}"
        )
    return "\n\n".join(out)


@tool
def list_knowledge_base(config: InjectedConfig = None) -> str:
    """List the documents in the current project's knowledge base, with their
    ingest status and chunk counts."""
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    project_id = sessions_store.resolve_project_id(tid)
    docs = sessions_store.list_kb_documents(project_id)
    if not docs:
        return (
            "The project knowledge base is empty. Add documents in the Knowledge tab."
        )
    lines = [f"Project knowledge base ({len(docs)} document(s)):"]
    for d in docs:
        status = d["status"]
        extra = f", {d['n_chunks']} chunks" if status == "ready" else ""
        err = f" — {d['error']}" if d.get("error") else ""
        lines.append(f"  • {d['name']} [{d['kind']}, {status}{extra}]{err}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  Preferences, branding & templates
# ══════════════════════════════════════════════════════════════════════════


@tool
def get_preferences(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Recall the user's saved preferences and the project's client branding
    (color palette, logo, fonts, footer). Call this BEFORE producing
    client-facing charts, reports, or slides so output matches the client's
    brand. Plots automatically use confirmed branding; this tool tells you
    what is set."""
    from mmm_framework.agents.branding import brand_palette, is_active
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    project_id = sessions_store.resolve_project_id(tid)
    branding = sessions_store.get_project_branding(project_id)
    global_prefs = sessions_store.list_preferences("global")

    lines = ["### Preferences & branding"]
    if branding:
        colors = branding.get("colors") or {}
        status = (
            "confirmed" if branding.get("confirmed") else "PROPOSED — not confirmed"
        )
        lines.append(
            f"**Project branding** ({status}, source: {branding.get('source','manual')}):"
        )
        if branding.get("client_name"):
            lines.append(f"- Client: {branding['client_name']}")
        if brand_palette(branding):
            lines.append(f"- Palette: {', '.join(brand_palette(branding))}")
        for k in ("primary", "secondary", "accent"):
            if colors.get(k):
                lines.append(f"- {k.title()}: {colors[k]}")
        if branding.get("logo_url"):
            lines.append(f"- Logo: {branding['logo_url']}")
        fonts = branding.get("fonts") or {}
        if fonts.get("heading") or fonts.get("body"):
            lines.append(
                f"- Fonts: heading={fonts.get('heading') or '—'}, "
                f"body={fonts.get('body') or '—'}"
            )
        if branding.get("footer_text"):
            lines.append(f"- Footer: {branding['footer_text']}")
        if is_active(branding):
            lines.append("\nPlots and client reports automatically use this palette.")
        elif not branding.get("confirmed"):
            lines.append(
                "\n⚠️ This branding was extracted but NOT confirmed — ask the "
                "user to confirm before styling deliverables with it."
            )
    else:
        lines.append("**Project branding:** none set.")
    if global_prefs:
        lines.append("\n**Global preferences:**")
        for k, v in global_prefs.items():
            lines.append(f"- {k}: {json.dumps(v, default=str)[:200]}")
    else:
        lines.append("\n**Global preferences:** none set.")

    dashboard_data = dict(state.get("dashboard_data") or {})
    if branding:
        dashboard_data["branding"] = branding
    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def save_preference(
    key: str,
    value: str,
    scope: str = "global",
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Persist a lasting user preference so future sessions recall it.

    Use when the user states a durable preference ("always use the corporate
    palette", "reports should be in EUR"). scope: "global" (all projects) or
    "project" (this project — e.g. client branding; key "branding" expects the
    branding JSON shape). value may be a JSON string or plain text."""
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    try:
        parsed = json.loads(value)
    except Exception:
        parsed = value

    if scope == "project":
        target = sessions_store.resolve_project_id(tid)
    elif scope == "global":
        from mmm_framework.agents.profile import is_hosted

        if is_hosted():
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Global preferences are disabled in the hosted "
                            'profile — save with scope="project" instead.',
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        target = "global"
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f'Unknown scope {scope!r} — use "global" or "project".',
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    if key == "branding" and isinstance(parsed, dict):
        from mmm_framework.agents.branding import Branding

        try:
            parsed = Branding.model_validate(parsed).model_dump()
        except Exception as exc:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Invalid branding payload: {exc}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

    sessions_store.set_preference(target, key, parsed)
    dashboard_data = dict(state.get("dashboard_data") or {}) if state else {}
    if key == "branding" and isinstance(parsed, dict):
        dashboard_data["branding"] = parsed
    update: dict = {
        "messages": [
            ToolMessage(
                content=f"Saved preference `{key}` (scope: {scope}).",
                tool_call_id=tool_call_id,
            )
        ]
    }
    if dashboard_data:
        update["dashboard_data"] = dashboard_data
    return Command(update=update)


@tool
def list_templates(
    kind: str = None,
    config: InjectedConfig = None,
) -> str:
    """Discover available templates: report layouts, color palettes, saved
    model-spec configs, and knowledge-base template documents.

    kind: "report" | "palette" | "model_config" | "kb" (default: all). Load a
    model config with `load_config`; pick a report template by passing
    template=<name> to `generate_client_report`."""
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    kinds = [kind] if kind else ["report", "palette", "model_config", "kb"]
    sections: list[str] = []

    if "report" in kinds:
        sections.append(
            "**Report templates** (use with `generate_client_report(template=...)`):\n"
            "- `augur` — **default**: the editorial *Media Performance Readout* — "
            "narrative, evidence-coded (Scale/Test/Hold/Reduce), with a KPI strip, "
            "channel scorecard, ROI & uncertainty, marginal return, saturation, "
            "reallocation, per-channel deep dives, carryover, a posterior-predictive "
            "fit-over-time + checks block, recommended tests, and AI-written CMO/"
            "planner insights\n"
            "- `full` — every section incl. diagnostics & methodology (the technical "
            "readout the Augur report links to)\n"
            "- `client` — clean classic client report (no MCMC internals)\n"
            "- `presentation` — summary, ROI, decomposition (deck-friendly)\n"
            "- `minimal` — executive summary + channel ROI only"
        )

    if "palette" in kinds:
        from mmm_framework.reporting.config import ColorPalette, ColorScheme

        rows = []
        for p in ColorPalette:
            cs = ColorScheme.from_palette(p)
            rows.append(f"- `{p.value}` — primary {cs.primary}, accent {cs.accent}")
        branding = sessions_store.get_project_branding(
            sessions_store.resolve_project_id(tid)
        )
        if branding:
            from mmm_framework.agents.branding import brand_palette

            pal = brand_palette(branding)
            if pal:
                status = "confirmed" if branding.get("confirmed") else "unconfirmed"
                rows.append(
                    f"- `brand` — project branding ({status}): {', '.join(pal)}"
                )
        sections.append("**Color palettes:**\n" + "\n".join(rows))

    if "model_config" in kinds:
        configs = _scan_saved_configs()
        if configs:
            rows = [
                f"- `{c['name']}` — KPI={c.get('kpi','?')}, "
                f"{c.get('n_channels',0)} channels, {c.get('n_controls',0)} controls"
                for c in configs
            ]
            sections.append(
                "**Saved model configs** (load with `load_config`):\n" + "\n".join(rows)
            )
        else:
            sections.append(
                "**Saved model configs:** none yet (`save_config` creates them)."
            )

    if "kb" in kinds:
        project_id = sessions_store.resolve_project_id(tid)
        docs = sessions_store.list_kb_documents(project_id)
        tpl_docs = [
            d
            for d in docs
            if (d.get("meta") or {}).get("template")
            or "template" in (d.get("name") or "").lower()
        ]
        if tpl_docs:
            rows = [f"- {d['name']} [{d['status']}]" for d in tpl_docs]
            sections.append(
                "**Knowledge-base templates** (search with `search_knowledge_base`):\n"
                + "\n".join(rows)
            )
        else:
            sections.append("**Knowledge-base templates:** none found.")

    return "### Available templates\n\n" + "\n\n".join(sections)


@tool
def extract_brand_from_website(
    url: str,
    save: bool = True,
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Extract client branding (colors, logo, fonts, company name) from a
    website and save it as the project's PROPOSED branding (confirmed=false).

    Show the user the extracted swatches and ask them to confirm; once they
    approve, save it via `save_preference(scope="project", key="branding",
    value=<branding json with "confirmed": true>)` — unconfirmed branding never
    styles deliverables. Runs server-side with a strict public-URL guard."""
    from mmm_framework.agents.brand_extract import (
        BrandExtractError,
        extract_brand_from_url,
    )
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    try:
        proposal = extract_brand_from_url(url)
    except BrandExtractError as exc:
        return Command(
            update={
                "messages": [ToolMessage(content=str(exc), tool_call_id=tool_call_id)]
            }
        )
    except Exception as exc:  # noqa: BLE001
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Brand extraction failed: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    if save:
        project_id = sessions_store.resolve_project_id(tid)
        sessions_store.set_project_branding(project_id, proposal)

    colors = proposal.get("colors") or {}
    fonts = proposal.get("fonts") or {}
    lines = [f"### Brand proposal extracted from {url}", ""]
    if proposal.get("client_name"):
        lines.append(f"- **Client:** {proposal['client_name']}")
    lines.append("\n| Role | Color |\n|---|---|")
    for k in ("primary", "secondary", "accent"):
        if colors.get(k):
            lines.append(f"| {k.title()} | `{colors[k]}` |")
    if colors.get("palette"):
        lines.append(f"| Palette | {', '.join(f'`{c}`' for c in colors['palette'])} |")
    if proposal.get("logo_url"):
        lines.append(f"\n- **Logo:** {proposal['logo_url']}")
    if fonts.get("heading"):
        lines.append(
            f"- **Fonts:** {fonts.get('heading')} / {fonts.get('body') or '—'}"
        )
    lines.append(
        "\n⚠️ Saved as **proposed** branding (unconfirmed). Ask the user to "
        "review these colors; on approval, re-save with `confirmed: true`."
        if save
        else "\n(Not saved — pass save=true to store it as the project proposal.)"
    )

    dashboard_data = dict(state.get("dashboard_data") or {}) if state else {}
    dashboard_data["branding"] = proposal
    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard_data,
        }
    )


# ══════════════════════════════════════════════════════════════════════════
#  Reusable past results (req 6)
# ══════════════════════════════════════════════════════════════════════════


@tool
def query_past_results(
    config: InjectedConfig = None,
    kind: str = None,
) -> str:
    """List prior results/artifacts saved in THIS session so you can reuse them:
    fitted model runs, generated reports, code snippets, and python text outputs.
    Optionally filter by kind (model_run | report | code_snippet | text_output |
    project_report). Returns artifact ids that can be downloaded from the
    frontend."""
    from mmm_framework.api import sessions as sessions_store

    tid = _activate_thread(config)
    arts = sessions_store.list_artifacts(tid)
    if kind:
        arts = [a for a in arts if a.get("kind") == kind]
    if not arts:
        return "No saved results yet in this session."
    lines = [f"Saved results in this session ({len(arts)}):"]
    for a in arts:
        p = a.get("payload", {})
        if a["kind"] == "model_run":
            desc = f"model_run '{p.get('run_name','?')}' kpi={p.get('kpi','?')} channels={p.get('channels')}"
        elif a["kind"] == "text_output":
            snip = (p.get("stdout", "") or "")[:80].replace("\n", " ")
            desc = f"python output ({'error' if p.get('is_error') else 'ok'}): {snip}…"
        elif a["kind"] in (
            "report",
            "project_report",
            "project_slides",
            "client_report",
            "client_slides",
        ):
            desc = f"{a['kind']}: {p.get('path','?')}"
        elif a["kind"] == "code_snippet":
            snip = (p.get("code", "") or "")[:80].replace("\n", " ")
            desc = f"code: {snip}…"
        else:
            desc = a["kind"]
        lines.append(f"  • [{a['id'][:8]}] {desc}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  Library discovery + power analysis (req 1)
# ══════════════════════════════════════════════════════════════════════════

_LIBRARY_MENU = """\
# mmm_framework capability menu (reach all of this via `execute_python`, using `mmf`)

## Data loading
- `mmf.MFFLoader(config).load(df_or_path)` → PanelDataset (.y, .X_media, .X_controls, .coords)
- `mmf.load_mff(...)`, `mmf.mff_from_wide_format(...)`, `mmf.load_ragged_mff(...)`

## Pre-fit data quality (EDA / validation / outliers)
- `from mmm_framework.eda import load_eda_panel, validate_dataset, detect_outliers,
  recommend_treatments, apply_treatments, profile_panel, collinearity_analysis,
  decompose_series, stationarity_tests, spend_share`
- Chart builders (return go.Figure — call `.show()`): `from mmm_framework.eda.charts import ...`
  (dedicated tools cover the common paths: `validate_data`, `run_eda`,
  `detect_outliers`, `apply_outlier_treatment`)

## Build & fit the standard model
- Builders (`mmf.ModelConfigBuilder`, `mmf.MediaChannelConfigBuilder`, …) → ModelConfig
- `mmf.BayesianMMM(panel, model_config, trend_config=None)` then `.fit(draws=, tune=, chains=)`
  NOTE: BayesianMMM takes a **PanelDataset** (from a loader), not raw arrays.

## Analysis on a FITTED model  (the cached one is bound as `mmm`/`results`)
- `mmm.compute_counterfactual_contributions(...)`, `mmm.compute_marginal_contributions(spend_increase_pct=10)`
- `mmm.what_if_scenario({'TV': +0.1})`, `mmm.compute_component_decomposition()`
- `from mmm_framework.analysis import MMMAnalyzer; MMMAnalyzer(mmm).compute_channel_roi()`
  (also: dedicated tools `run_budget_scenario`, `run_marginal_analysis`)

## Extended models  (mediation / multi-outcome / combined)
- `from mmm_framework.mmm_extensions import NestedMMM, MultivariateMMM, CombinedMMM`
  These take **raw arrays**: `NestedMMM(X_media: np.ndarray, y, channel_names, index=None)`.
- The DAG→model-type bridge auto-selects the subclass:
  `from mmm_framework.dag_model_builder import DAGModelBuilder, create_mediation_dag`
  `model = DAGModelBuilder().with_dag(dag).with_mff_data(df).bayesian_numpyro().build(); model.fit(...)`
- Factory mediators: `from mmm_framework.mmm_extensions import awareness_mediator, cross_effect, ...`

## Experiment / lift-test calibration  (fold a measured lift into the prior)
- `from mmm_framework.calibration.likelihood import ExperimentMeasurement, ExperimentEstimand`
- Pass `experiments=[...]` to BayesianMMM, OR `mmm.add_experiment_calibration([...])` **before** `.fit()`.

## Diagnostics & reporting
- `from mmm_framework.diagnostics import parameter_learning` (prior→posterior contraction)
- `from mmm_framework.reporting import MMMReportGenerator, ReportBuilder` (HTML reports)
- Standalone Plotly charts: `from mmm_framework.reporting import create_roi_forest_plot, ...`

## Serialization
- `from mmm_framework import MMMSerializer` → `.save(model, results, path)` /
  `.load(path, panel, rebuild_model=True)` (load needs a compatible PanelDataset).

Tip: write outputs to `OUTPUT_DIR` so they become downloadable; use Plotly `fig.show()` for charts.
"""


def _filter_reference(menu: str, topic: str | None) -> str:
    """Return the full menu, or only the ``## `` sections matching ``topic``."""
    if not topic:
        return menu
    blocks = menu.split("\n## ")
    matched = [blocks[0]] + [b for b in blocks[1:] if topic.lower() in b.lower()]
    return "\n## ".join(matched) if len(matched) > 1 else menu


@tool
def library_reference(topic: str = None) -> str:
    """Return a menu of EVERY mmm_framework capability the agent can use (data
    loading, standard & extended/mediation/multivariate models, counterfactual
    & budget analysis, lift-test calibration, diagnostics, reporting,
    serialization) with exact import paths and the input-shape/ordering traps.
    Consult this before hand-writing complex code in execute_python. Optionally
    pass a topic substring to filter."""
    return _filter_reference(_LIBRARY_MENU, topic)


@tool
def bayesian_workflow_reference(topic: str = None) -> str:
    """Return the Bayesian-workflow methodology reference: why each workflow
    step exists, what to inspect, the decision thresholds (R-hat/ESS/divergence
    gates, prior-predictive flags), what to do when a check FAILS, and the
    MMM-specific silent failure modes (confounding, collinearity, saturation
    form, prior-dominated posteriors, average-vs-marginal ROAS).

    Consult this when a diagnostic fails and you need the remedy, when choosing
    or revising priors, before claiming sensitivity was tested, when explaining
    methodology to the user, or whenever unsure WHY a workflow step matters.
    Optionally pass a topic substring (e.g. "prior", "diagnostics",
    "sensitivity", "failure modes") to filter sections."""
    from mmm_framework.agents.workflow_reference import BAYESIAN_WORKFLOW_REFERENCE

    return _filter_reference(BAYESIAN_WORKFLOW_REFERENCE, topic)


@tool
def run_budget_scenario(
    spend_changes: str,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Run a what-if budget scenario on the fitted model. `spend_changes` is a
    JSON object mapping channel name → fractional spend change (e.g.
    {"TV": 0.2, "Search": -0.1} for +20% TV, -10% Search). Returns the predicted
    KPI change vs baseline."""
    _activate_thread(config)
    try:
        changes = json.loads(spend_changes)
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not parse spend_changes JSON: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "budget_scenario", {"spend_changes": changes}
    )
    return _modelop_command(res, {}, tool_call_id)


@tool
def run_marginal_analysis(
    config: InjectedConfig = None,
    spend_increase_pct: float = 10.0,
    channels: str = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Compute marginal contributions / marginal ROAS for a `spend_increase_pct`
    bump (default +10%) on the fitted model — i.e. the incremental return of the
    next dollar per channel. `channels` is an optional JSON list to restrict to."""
    _activate_thread(config)
    chans = None
    if channels:
        try:
            chans = json.loads(channels)
        except Exception:
            chans = None
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "marginal_analysis",
        {"spend_increase_pct": spend_increase_pct, "channels": chans},
    )
    return _modelop_command(res, {}, tool_call_id)


@tool
def run_budget_optimizer(
    config: InjectedConfig = None,
    total_budget: float = None,
    budget_change_pct: float = None,
    min_multiplier: float = 0.0,
    max_multiplier: float = 2.0,
    channel_bounds: dict = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Find the budget allocation that maximizes expected KPI, using the fitted
    model's posterior response curves (saturation + adstock respected).

    Defaults to REALLOCATING the current total spend; pass `total_budget` (an
    absolute amount) or `budget_change_pct` (e.g. -10 or 15) to size the budget.
    `min_multiplier`/`max_multiplier` bound each channel's spend as multiples of
    its current spend (default 0–2x — beyond observed spend the curves are
    extrapolation, so recommendations stay inside the evidence).

    `channel_bounds` sets PER-CHANNEL spend limits that override the global
    bounds — `{"TV": [1.0, 1.0], "Social": [0.0, 1.2]}` means "freeze TV at its
    current spend, cap Social at +20%". Use this to encode real plan constraints
    (partner-committed caps, contractual floors, a locked line). Each value is
    `[low, high]` multipliers of that channel's current spend; an unknown channel
    name is rejected (so a constraint is never silently ignored).

    The result includes DECISION uncertainty: the optimizer re-runs under each
    posterior draw, so each channel gets a 90% range of its optimal share. Wide
    ranges mean the data does not pin down the optimum — follow up with
    `recommend_lift_experiments`. Requires a fitted model.
    """
    _activate_thread(config)
    kwargs = {
        "min_multiplier": min_multiplier,
        "max_multiplier": max_multiplier,
    }
    if total_budget is not None:
        kwargs["total_budget"] = float(total_budget)
    if budget_change_pct is not None:
        kwargs["budget_change_pct"] = float(budget_change_pct)
    if channel_bounds:
        kwargs["bounds"] = channel_bounds
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "optimize_budget", kwargs
    )
    return _modelop_command(res, {}, tool_call_id)


@tool
def log_experiment(
    channel: str = None,
    status: str = None,
    experiment_id: str = None,
    design_type: str = None,
    start_date: str = None,
    end_date: str = None,
    estimand: str = None,
    value: float = None,
    se: float = None,
    notes: str = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Create or update an entry in the project's experiment registry — the
    log the home page tracks and the calibration loop reads.

    PREFER the lifecycle tools for the standard flow: plan_experiment (create
    with design+priority snapshots) → preregister_experiment →
    record_experiment_readout → apply_experiment_calibration + fit_mmm_model
    (auto-marks calibrated). Use log_experiment for ad-hoc edits: importing a
    historical experiment, fixing a field, marking a test 'running' at launch,
    or abandoning one (status='abandoned').

    Create: pass `channel` (+ optional design fields), status defaults to
    'planned'. Update: pass `experiment_id` plus only the fields that changed.
    Lifecycle: draft → planned → running → completed → calibrated.
    Results: status='completed' with `value`, `se`, `estimand`
    ('roas' | 'contribution' | 'mroas'). Dates are ISO strings.
    """
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    project_id = None
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
    except Exception:
        pass
    try:
        exp = sessions_store.upsert_experiment(
            experiment_id=experiment_id,
            project_id=project_id,
            thread_id=tid,
            channel=channel,
            design_type=design_type,
            status=status,
            start_date=start_date,
            end_date=end_date,
            estimand=estimand,
            value=value,
            se=se,
            notes=notes,
        )
    except ValueError as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not log experiment: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    msg = (
        f"Experiment **{exp['channel']}** logged (id `{exp['id']}`, "
        f"status **{exp['status']}**"
        + (
            f", result {exp['value']:g} ± {exp['se']:g} {exp['estimand'] or ''}"
            if exp.get("value") is not None
            else ""
        )
        + ")."
    )
    if exp["status"] == "completed":
        msg += (
            " This result is not yet calibrated into a fit — when you refit "
            "with it via add_experiment_calibration, update the entry to "
            "status='calibrated'."
        )
    return Command(
        update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
    )


@tool
def list_experiment_log(
    status: str = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List the project's experiment registry (planned / running / completed /
    calibrated lift tests), optionally filtered by status. Check this before
    recommending new experiments (don't re-propose a channel already being
    tested) and when deciding whether a refresh should calibrate in completed
    results."""
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    project_id = None
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
    except Exception:
        pass
    exps = sessions_store.list_experiments(project_id=project_id, status=status)
    if not exps:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No experiments logged for this project yet.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    lines = ["### Experiment Log", ""]
    lines.append("| Channel | Status | Window | Result | id |")
    lines.append("|---|---|---|---|---|")
    for e in exps:
        window = (
            f"{e['start_date'] or '?'} → {e['end_date'] or '?'}"
            if e.get("start_date") or e.get("end_date")
            else "—"
        )
        result = (
            f"{e['value']:g} ± {e['se']:g} {e['estimand'] or ''}"
            if e.get("value") is not None
            else "—"
        )
        lines.append(
            f"| {e['channel']} | {e['status']} | {window} | {result} | `{e['id'][:8]}` |"
        )
    completed = [e for e in exps if e["status"] == "completed"]
    if completed:
        lines.append(
            f"\n⚠️ {len(completed)} completed result(s) not yet calibrated into a "
            "fit — call apply_experiment_calibration, then fit_mmm_model."
        )
    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ]
        }
    )


def _simple_msg(text: str, tool_call_id) -> Command:
    return Command(
        update={"messages": [ToolMessage(content=text, tool_call_id=tool_call_id)]}
    )


@tool
def plan_experiment(
    channel: str,
    state: Annotated[dict, InjectedState],
    hypothesis: str = None,
    start_date: str = None,
    end_date: str = None,
    preregister: bool = False,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Create a registry entry (status 'draft') for an experiment on `channel`,
    snapshotting the latest recommendation: the concrete design from
    recommend_lift_experiments / compute_experiment_priorities (design type,
    duration, target SE) plus the EIG/EVOI priority row, and the recommending
    model run. Set preregister=true to lock it straight to 'planned' (the
    pre-registration step) — or call preregister_experiment after review.

    Lifecycle: draft → planned → running → completed → calibrated. Dates are
    ISO strings for the intended test window.
    """
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    project_id = None
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
    except Exception:
        pass

    dashboard = state.get("dashboard_data") or {}
    # design snapshot: a full design-studio plan for this channel wins
    # (assignment, power curve, schedule); else the latest recommendation row.
    design = None
    plan = dashboard.get("experiment_design_plan")
    if isinstance(plan, dict) and plan.get("channel") == channel:
        design = dict(plan)
    if design is None:
        for d in (dashboard.get("experiment_design") or {}).get("designs", []):
            if d.get("channel") == channel:
                design = dict(d)
                break
    if hypothesis:
        design = design or {"channel": channel}
        design["hypothesis"] = hypothesis
    # priority snapshot from the latest grid (if any)
    priority = next(
        (
            dict(r)
            for r in (dashboard.get("experiment_priorities") or {}).get("channels", [])
            if r.get("channel") == channel
        ),
        None,
    )
    recommending_run_id = (dashboard.get("model_run") or {}).get("run_id")

    try:
        exp = sessions_store.upsert_experiment(
            project_id=project_id,
            thread_id=tid,
            channel=channel,
            design_type=(design or {}).get("design_key")
            or (design or {}).get("design_type"),
            status="draft",
            start_date=start_date,
            end_date=end_date,
            recommending_run_id=recommending_run_id,
            design=design,
            priority=priority,
        )
        if preregister:
            exp = sessions_store.transition_experiment(
                exp["id"], "planned", note="pre-registered at planning time"
            )
    except ValueError as exc:
        return _simple_msg(f"Could not plan experiment: {exc}", tool_call_id)

    msg = (
        f"Experiment on **{channel}** created (id `{exp['id']}`, status "
        f"**{exp['status']}**"
        + (
            f", recommended by run `{recommending_run_id}`"
            if recommending_run_id
            else ""
        )
        + ")."
    )
    if design:
        msg += f" Design snapshot: {design.get('design_type', 'n/a')}."
    if exp["status"] == "draft":
        msg += " Pre-register it with preregister_experiment before launch."
    return _simple_msg(msg, tool_call_id)


@tool
def preregister_experiment(
    experiment_id: str,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Pre-register a drafted experiment (draft → planned): locks the design
    BEFORE the experiment runs, stamping the pre-registration time. Run this
    only when the design (channel, window, intensity, target precision) is
    final — analysis choices after seeing results are exactly what
    pre-registration protects against."""
    from mmm_framework.api import sessions as sessions_store

    try:
        exp = sessions_store.transition_experiment(
            experiment_id, "planned", note="pre-registered"
        )
    except ValueError as exc:
        return _simple_msg(f"Could not pre-register: {exc}", tool_call_id)
    return _simple_msg(
        f"Experiment `{exp['id']}` on **{exp['channel']}** pre-registered "
        "(status **planned**). Move it to 'running' with log_experiment when "
        "the test launches, and record results with record_experiment_readout.",
        tool_call_id,
    )


@tool
def record_experiment_readout(
    experiment_id: str,
    value: float,
    se: float,
    estimand: str = "roas",
    start_date: str = None,
    end_date: str = None,
    method: str = None,
    notes: str = None,
    spend_per_period: float = None,
    n_treated_units: int = 1,
    adstock_state: str = "steady_state",
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Record an experiment's measured result (→ 'completed'): the lift point
    estimate, its standard error, and the estimand ('roas' | 'contribution' |
    'mroas'), plus the actual test window (ISO dates) and the measurement
    method (e.g. 'geo holdout DiD', 'synthetic control'). A planned experiment
    is moved through 'running' automatically.

    Off-panel calibration (experiment ran OUTSIDE the model's fitted date
    range): also pass `spend_per_period` — the channel's spend per period per
    treated unit during the test, on the same scale as the dataset's spend
    column — plus `n_treated_units` (number of treated geos/units, default 1)
    and `adstock_state` ('steady_state' for an always-on/sustained test,
    'cold_start' for a burst launched from dark). With these, the calibration
    evaluates the channel's response curve at the test's spend level instead of
    requiring the window to overlap the data, so an experiment from a different
    period can still be folded in (assuming the response curve is stable across
    the two periods). Not needed when the window falls inside the data.

    'completed' means measured but NOT yet in the model — follow with
    apply_experiment_calibration + fit_mmm_model to close the loop.
    """
    from mmm_framework.api import sessions as sessions_store

    exp = sessions_store.get_experiment(experiment_id)
    if exp is None:
        return _simple_msg(f"Unknown experiment id '{experiment_id}'.", tool_call_id)
    if adstock_state not in ("steady_state", "cold_start"):
        return _simple_msg(
            "adstock_state must be 'steady_state' or 'cold_start'.", tool_call_id
        )
    try:
        # Merge onto any existing readout so re-recording (e.g. adding a spend
        # level for off-panel calibration) only adds fields rather than wiping
        # method/notes that weren't re-passed.
        readout = dict(exp.get("readout") or {})
        readout.update({"value": value, "se": se, "estimand": estimand})
        if method is not None:
            readout["method"] = method
        if notes is not None:
            readout["notes"] = notes
        # Off-panel calibration inputs (used when the test window is outside the
        # dataset): the response curve is evaluated at this spend level.
        if spend_per_period is not None:
            readout["spend_per_period"] = float(spend_per_period)
            readout["n_treated_units"] = int(n_treated_units or 1)
            readout["adstock_state"] = adstock_state

        if exp["status"] in ("completed", "calibrated"):
            # Already measured: update the readout in place. A completed->completed
            # transition is illegal (and we must not bounce a calibrated experiment
            # back), so re-recording is an idempotent self-update — this is exactly
            # the path the off-panel advisory tells the user to take to attach a
            # spend level after the fact.
            exp = sessions_store.upsert_experiment(
                experiment_id=experiment_id,
                value=float(value),
                se=float(se),
                estimand=estimand,
                start_date=start_date,
                end_date=end_date,
                readout=readout,
            )
        else:
            if exp["status"] == "planned":
                exp = sessions_store.transition_experiment(
                    experiment_id, "running", note="auto-advanced at readout"
                )
            exp = sessions_store.transition_experiment(
                experiment_id,
                "completed",
                value=float(value),
                se=float(se),
                estimand=estimand,
                start_date=start_date,
                end_date=end_date,
                readout=readout,
                note=method,
            )
    except ValueError as exc:
        return _simple_msg(f"Could not record readout: {exc}", tool_call_id)
    return _simple_msg(
        f"Readout recorded for **{exp['channel']}**: {value:g} ± {se:g} "
        f"({estimand}). Status **{exp['status']}** — not yet in the model. Next: "
        "apply_experiment_calibration, then fit_mmm_model for the calibrated "
        "refit.",
        tool_call_id,
    )


@tool
def apply_experiment_calibration(
    state: Annotated[dict, InjectedState],
    experiment_ids: list[str] = None,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Stage completed experiment readouts as calibration likelihoods for the
    NEXT fit: writes `experiments` (ExperimentMeasurement payloads) into the
    model spec. The next fit_mmm_model folds them into the model as in-graph
    likelihood terms and marks the registry entries 'calibrated'.

    Uses every measured experiment in the project — 'completed' AND already
    'calibrated' (from earlier fits) — unless `experiment_ids` narrows it, so a
    refit that adds one experiment keeps all the prior calibrations. Each
    experiment needs value, se, estimand, and a test window (start/end dates);
    out-of-window experiments calibrate off-panel from their recorded spend.
    """
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    spec = state.get("model_spec")
    if not spec or not spec.get("kpi"):
        return _simple_msg(
            "No model is configured yet — configure_model first, then stage "
            "the calibration.",
            tool_call_id,
        )
    project_id = None
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
    except Exception:
        pass

    # Stage the FULL measured set — newly 'completed' AND already 'calibrated'
    # (from prior fits). The spec is rebuilt from scratch each fit, so staging
    # only the new batch would drop every previously-calibrated experiment and
    # silently lose its calibration on the next refit.
    completed = [
        e
        for e in sessions_store.list_experiments(project_id=project_id)
        if e.get("status") in ("completed", "calibrated")
    ]
    if experiment_ids:
        wanted = set(experiment_ids)
        missing = wanted - {e["id"] for e in completed}
        if missing:
            return _simple_msg(
                f"Not measured yet (or unknown): {', '.join(sorted(missing))}. "
                "Only completed or already-calibrated experiments can be staged.",
                tool_call_id,
            )
        completed = [e for e in completed if e["id"] in wanted]
    if not completed:
        return _simple_msg(
            "No measured experiments to calibrate. Record results first with "
            "record_experiment_readout.",
            tool_call_id,
        )

    channels = {m.get("name") for m in spec.get("media_channels", [])}

    # Dataset date range + period cadence: an experiment whose window lies inside
    # the range calibrates in-panel (sum the contribution over the fitted rows);
    # one outside calibrates *off-panel* (evaluate the response curve at the
    # test's spend level). Best-effort — None when the file can't be read.
    dataset_path = state.get("dataset_path")
    bounds = _dataset_date_bounds(dataset_path)
    freq_days = _dataset_period_freq_days(dataset_path)

    measurements, problems, needs_spend = [], [], []
    for e in completed:
        errs = []
        if e["channel"] not in channels:
            errs.append(f"channel '{e['channel']}' is not in the model spec")
        if e.get("value") is None or e.get("se") is None:
            errs.append("missing value/se")
        if not e.get("start_date") or not e.get("end_date"):
            errs.append("missing test window (start_date/end_date)")
        if errs:
            problems.append(f"`{e['id'][:8]}` ({e['channel']}): {'; '.join(errs)}")
            continue

        start, end = e["start_date"], e["end_date"]
        estimand = (e.get("estimand") or "roas").lower()
        m = {
            "experiment_id": e["id"],
            "channel": e["channel"],
            "test_period": [start, end],
            "value": float(e["value"]),
            "se": float(e["se"]),
            "estimand": estimand,
        }

        if not _window_within_bounds(start, end, bounds):
            # Off-panel: needs the test's spend level (recorded on the readout).
            readout = e.get("readout") or {}
            spend_pp = readout.get("spend_per_period")
            if spend_pp is None:
                needs_spend.append(
                    f"`{e['id'][:8]}` ({e['channel']}) [{start} → {end}]"
                )
                continue
            if estimand == "mroas":
                problems.append(
                    f"`{e['id'][:8]}` ({e['channel']}): off-panel calibration "
                    "supports 'contribution'/'roas' only (not 'mroas') — re-run "
                    "the test inside the data window for an mROAS estimand."
                )
                continue
            m["eval_spend"] = float(spend_pp)
            m["eval_periods"] = _periods_in_window(start, end, freq_days)
            m["eval_units"] = int(readout.get("n_treated_units") or 1)
            m["adstock_state"] = readout.get("adstock_state") or "steady_state"
        measurements.append(m)

    if problems:
        return _simple_msg(
            "Cannot stage calibration — fix these first:\n- " + "\n- ".join(problems),
            tool_call_id,
        )

    if needs_spend and not measurements:
        rng = f"[{bounds[0]} → {bounds[1]}]" if bounds else "the fitted window"
        return _simple_msg(
            f"These experiments ran outside the dataset's date range {rng}, so "
            "they calibrate **off-panel** — evaluating the channel's response "
            "curve at the test's spend level (valid as long as that curve is "
            "stable between the test period and your data). I just need the spend "
            "level: re-run `record_experiment_readout` with `spend_per_period` "
            "(the channel's spend per period per treated unit during the test), "
            "plus `n_treated_units` / `adstock_state` if relevant, then call "
            "apply_experiment_calibration again.\n- " + "\n- ".join(needs_spend),
            tool_call_id,
        )

    new_spec = copy.deepcopy(dict(spec))
    new_spec["experiments"] = [
        {k: v for k, v in m.items() if k != "experiment_id"} for m in measurements
    ]
    new_spec["experiment_ids"] = [m["experiment_id"] for m in measurements]
    chs = ", ".join(sorted({m["channel"] for m in measurements}))
    n_off = sum(1 for m in measurements if "eval_spend" in m)
    detail = f" ({n_off} off-panel)" if n_off else ""
    tail = ""
    if needs_spend:
        tail = (
            f" Skipped {len(needs_spend)} out-of-window experiment(s) missing a "
            "test spend level — add `spend_per_period` via record_experiment_readout "
            "to calibrate those off-panel."
        )
    return _commit_spec(
        state,
        new_spec,
        tool_call_id,
        success_msg=(
            f"Staged {len(measurements)} experiment likelihood(s){detail} ({chs}) "
            "into the spec. Now call fit_mmm_model — the refit folds them into the "
            "model and marks the registry entries calibrated." + tail
        ),
        patch_paths=["experiments", "experiment_ids"],
    )


def _dataset_date_bounds(dataset_path: str | None) -> tuple[str, str] | None:
    """(min, max) ISO dates of the dataset's Period column, or None when the
    file/column can't be read cheaply."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        import pandas as pd

        df = pd.read_csv(dataset_path, usecols=["Period"])
        period = pd.to_datetime(df["Period"], errors="coerce").dropna()
        if period.empty:
            return None
        return (
            period.min().date().isoformat(),
            period.max().date().isoformat(),
        )
    except Exception:
        return None


def _dataset_period_freq_days(dataset_path: str | None) -> float | None:
    """Median spacing (in days) between the dataset's unique periods, used to
    size an off-panel experiment window into a number of periods. None when the
    file/column can't be read or has fewer than two periods."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    try:
        import pandas as pd

        df = pd.read_csv(dataset_path, usecols=["Period"])
        period = pd.to_datetime(df["Period"], errors="coerce").dropna().unique()
        if len(period) < 2:
            return None
        diffs = pd.Series(pd.to_datetime(period)).sort_values().diff().dropna()
        days = float(diffs.dt.days.median())
        return days if days > 0 else None
    except Exception:
        return None


def _window_within_bounds(start: str, end: str, bounds: tuple[str, str] | None) -> bool:
    """Whether ``[start, end]`` lies inside the dataset's date range.

    Parses with ``pd.to_datetime`` (matching the model's own window resolver in
    ``BayesianMMM._period_to_indices``) so unpadded / non-ISO dates compare
    correctly rather than by lexicographic string order. ``None`` bounds (date
    range unreadable) is treated as in-window — the model layer is the backstop.
    """
    if not bounds:
        return True
    try:
        import pandas as pd

        s, e = pd.to_datetime(start), pd.to_datetime(end)
        lo, hi = pd.to_datetime(bounds[0]), pd.to_datetime(bounds[1])
        return bool(s >= lo and e <= hi)
    except Exception:
        # Unparseable dates: fall back to lexicographic ISO comparison.
        return start >= bounds[0] and end <= bounds[1]


def _periods_in_window(start: str, end: str, freq_days: float | None) -> int:
    """Number of periods spanned by ``[start, end]`` at the dataset's cadence.

    Inclusive of both endpoints. Falls back to a weekly cadence when the
    dataset frequency can't be inferred, and never returns less than 1. A
    reversed or zero-length window collapses to a single period.
    """
    try:
        import pandas as pd

        span_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
    except Exception:
        return 1
    if span_days <= 0:
        return 1
    step = freq_days if (freq_days and freq_days > 0) else 7.0
    return max(1, int(round(span_days / step)) + 1)


@tool
def get_run_history(
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Return the project's model-run lineage: every fit with its dataset
    fingerprint, what changed in the spec vs the previous run, and which
    assumptions were added/revised (the rationale for each change).

    Use this to audit the modeling process, explain why the current model
    differs from earlier ones, and as the PROVENANCE section when writing a
    final report — it is the versioned record of data, model, and rationale.
    """
    from mmm_framework.api import runs as _runs
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    project_id = None
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
    except Exception:
        pass
    # Cap how many runs land in the LLM's context; this ToolMessage rides along
    # in history for the rest of the conversation.
    import os as _os

    _max_runs = int(_os.environ.get("MMM_AGENT_MAX_HISTORY_RUNS", "20"))
    md = _runs.run_timeline_markdown(project_id, max_runs=_max_runs)
    return Command(
        update={"messages": [ToolMessage(content=md, tool_call_id=tool_call_id)]}
    )


@tool
def recommend_lift_experiments(
    config: InjectedConfig = None,
    top_k: int = 3,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Recommend which lift experiments to run next, based on the fitted model.
    Channels are ranked by EIG × EVOI — what the experiment would teach
    (expected information gain over the channel's ROI posterior) times what
    that learning is worth to the budget decision (expected value of
    information) — falling back to the transparent heuristic (spend share ×
    ROAS uncertainty × allocation instability) if the EIG/EVOI pass fails.

    Each recommendation includes a concrete design (geo holdout vs national
    spend pulse, minimum duration from the channel's adstock window, target
    standard error) and the `ExperimentMeasurement` snippet that calibrates the
    result into the next fit. Requires a fitted model. Use
    `compute_experiment_priorities` for the full grid with quadrants and
    re-test triggers.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "experiment_design", {"top_k": int(top_k)}
    )
    return _modelop_command(res, {}, tool_call_id)


def _project_evidence(config) -> tuple[str | None, dict]:
    """(project_id, latest calibrated evidence per channel) for the active
    session — registry reads stay host-side; the evidence dict crosses the
    kernel boundary as plain JSON."""
    from mmm_framework.api import sessions as sessions_store

    tid = get_current_thread() if config is None else _activate_thread(config)
    try:
        sess = sessions_store.get_session(tid) if tid else None
        project_id = (sess or {}).get("project_id")
        if project_id:
            return project_id, sessions_store.latest_calibrated_evidence(project_id)
    except Exception:
        pass
    return None, {}


@tool
def compute_experiment_priorities(
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """The full EIG/EVOI experiment-priority grid from the fitted model: per
    channel, the expected information gain of an experiment (nats), the
    expected value of that information for the budget decision (KPI units),
    the composite priority, and the 2×2 quadrant — test_now / learn_cheaply /
    monitor / deprioritize. Channels with calibrated experiments in the
    project registry also get information-decay status: how stale the evidence
    is and whether a re-test is due. Requires a fitted model.

    Use this at step T₁ of the measurement cycle (after a fit, before
    committing to the next experiment portfolio); `plan_experiment` turns a
    grid row into a registry entry.
    """
    from datetime import date as _date

    _activate_thread(config)
    _project_id, evidence = _project_evidence(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "experiment_priorities",
        {"evidence": evidence, "as_of": _date.today().isoformat()},
    )
    return _modelop_command(res, {}, tool_call_id)


@tool
def design_experiment_plan(
    channel: str,
    state: Annotated[dict, InjectedState],
    design_key: str = None,
    duration: int = 8,
    intensity_pct: float = 50.0,
    geo_design: str = "scaling",
    amplitude_pct: float = 50.0,
    block_weeks: int = 2,
    n_pairs: int = None,
    levels: list[float] = None,
    seed: int = 42,
    margin: float = None,
    kpi_kind: str = "revenue",
    include_economics: bool = True,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Design a runnable experiment for `channel` from the loaded dataset —
    the randomization IS the product:

    - Geo/DMA panels (>= 4 geos): `design_key='geo_lift'` builds matched pairs
      from pre-period KPI co-movement and RANDOMIZES treatment within each
      pair, with a DiD power analysis (SE/MDE on the ROAS scale vs duration)
      and a placebo falsification bar. `geo_design='holdout'` goes dark in the
      treated geos; `'scaling'` lifts spend by `intensity_pct`.
    - `design_key='matched_market_did'`: the pseudo-experimental variant when
      the business dictates treatment geos — same matching + power math, with
      the parallel-trends caveat made explicit.
    - National data: `design_key='national_flighting'` builds a budget-neutral
      block-randomized on/off spend schedule (±`amplitude_pct`%, blocks of
      `block_weeks` weeks) that manufactures the exogenous variance the spend
      history lacks, with the identification gain quantified.

    Omit design_key to auto-pick from the data. The design itself is pure data
    (works PRE-FIT). When a model IS fitted and `include_economics=True`, the
    plan is enriched with the model's expected-effect anchor (is the test
    powered to detect the effect we EXPECT?) and the short-term opportunity cost
    of deviating from business-as-usual (forgone KPI, spend at risk, net $ when
    `margin` is given, learning-vs-cost). For the full A/A·A/B methodology
    comparison use `simulate_experiment`. Follow with `plan_experiment` to
    register + pre-register the design.
    """
    import os as _os

    from mmm_framework.planning.design import design_experiment, design_options

    _activate_thread(config)
    dataset_path = state.get("dataset_path")
    if not dataset_path or not _os.path.exists(dataset_path):
        return _simple_msg(
            "No dataset loaded — load or generate one before designing.",
            tool_call_id,
        )
    kpi = (state.get("model_spec") or {}).get("kpi")
    if not kpi:
        return _simple_msg(
            "No KPI configured — configure_model first so the designer knows "
            "the outcome series.",
            tool_call_id,
        )
    try:
        key = design_key or design_options(dataset_path, kpi, channel)["recommended"]
        if key == "national_flighting":
            _fl_kw: dict = dict(
                duration=int(duration),
                amplitude_pct=float(amplitude_pct),
                block_weeks=int(block_weeks),
                seed=int(seed),
            )
            if levels:
                # multi-level spend schedule (>=3 distinct levels traces the curve)
                _fl_kw["levels"] = tuple(float(m) for m in levels)
            design = design_experiment(
                dataset_path, kpi, channel, design_key=key, **_fl_kw
            )
        else:
            kwargs: dict = dict(
                duration=int(duration),
                design=geo_design,
                intensity_pct=float(intensity_pct),
                seed=int(seed),
            )
            if n_pairs is not None:
                kwargs["n_pairs"] = int(n_pairs)
            design = design_experiment(
                dataset_path, kpi, channel, design_key=key, **kwargs
            )
    except ValueError as exc:
        return _simple_msg(f"Could not design the experiment: {exc}", tool_call_id)

    lines = [f"### Experiment design — `{channel}` ({design['design_type']})", ""]
    if design["design_key"] in ("geo_lift", "matched_market_did"):
        pairs = ", ".join(
            f"{p['treatment']}→T / {p['control']}→C (r={p['correlation']:.2f})"
            for p in design["assignment"]
        )
        lines += [
            f"- Matched pairs ({design['n_pairs']}): {pairs}",
            f"- Treated-cell spend change: {design['intensity_pct']:+.0f}% "
            f"(≈ {design['weekly_spend_delta']:,.0f}/week)",
            f"- {design['duration']}-week test → SE(ROAS) ≈ {design['se_roas']:.2f}, "
            f"MDE ≈ {design['mde_roas']:.2f} (80% power)",
            f"- Placebo bar (pre-period chance 'lift', 95%): "
            f"±{design['placebo'].get('p95_abs') or 0:,.0f} KPI units",
        ]
        diag = design["diagnostics"]
        lines.append(
            f"- Matching: {diag.get('matching', 'residual matching')}; power "
            f"{design.get('se_source', 'analytic')}; worst covariate imbalance "
            f"{diag.get('max_balance_abs_std_diff', 0):.2f} std (< 0.25 is balanced)"
        )
        if diag["parallel_trends_warning"]:
            lines.append(
                "- ⚠️ Weakest pair RESIDUAL correlation "
                f"{diag.get('min_residual_correlation', 0):.2f} — after removing "
                "shared trend/seasonality the pairs barely co-move, so parallel "
                "trends is shaky; prefer fewer, better-matched pairs."
            )
        if not design["randomized"]:
            lines.append(
                "- ⚠️ Observational (no randomization) — report the placebo band "
                "prominently; this is a pseudo-experiment."
            )
    else:
        ident = design["identification"]
        on_off = "".join(
            "▮" if s["multiplier"] > 1 else "▯" for s in design["schedule"]
        )
        lines += [
            f"- Schedule ({design['duration']} weeks, ±{design['amplitude_pct']:.0f}%, "
            f"{design['block_weeks']}-week blocks, budget-neutral): `{on_off}`",
            f"- Exogenous share of test-window spend variance: "
            f"{ident['exogenous_share']:.0%} (historical spend CV "
            f"{ident['historical_spend_cv']:.2f} is demand-confounded; the "
            "schedule's variance is randomized, i.e. clean)",
            f"- On/off contrast → SE(ROAS) ≈ {design['se_roas']:.2f}, "
            f"MDE ≈ {design['mde_roas']:.2f} (80% power)",
        ]
    lines += [
        "",
        design["analysis_plan"],
        "",
        "Next: `plan_experiment` to register and pre-register this design.",
    ]

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["experiment_design_plan"] = design

    # Model-anchored enrichment: when a model is fitted, add the expected-effect
    # anchor + short-term opportunity cost via the kernel op (no-op pre-fit, so
    # this only costs posterior passes when there is a model to anchor to).
    if include_economics:
        try:
            design_params = {
                "dataset_path": dataset_path,
                "kpi": kpi,
                "channel": channel,
                "design_key": key,
                "duration": int(duration),
                "design": geo_design,
                "intensity_pct": float(intensity_pct),
                "amplitude_pct": float(amplitude_pct),
                "block_weeks": int(block_weeks),
                "seed": int(seed),
            }
            if n_pairs is not None:
                design_params["n_pairs"] = int(n_pairs)
            if levels:
                design_params["levels"] = [float(m) for m in levels]
            eco_res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
                "experiment_economics",
                {
                    "design_params": design_params,
                    "run_simulation": False,
                    "margin": margin,
                    "kpi_kind": kpi_kind,
                },
            )
            eco = (eco_res.get("dashboard") or {}).get("experiment_economics")
            if eco and eco.get("model_anchored"):
                dashboard_data["experiment_economics"] = eco
                anc = eco.get("anchor") or {}
                oc = eco.get("opportunity_cost") or {}
                if anc.get("verdict"):
                    design["model_anchor"] = (eco.get("design") or {}).get(
                        "model_anchor"
                    )
                    lines += [
                        "",
                        f"**Model anchor** — expected incremental ROAS ≈ "
                        f"{anc.get('incremental_roas_median', 0):.2f}; design is "
                        f"**{str(anc['verdict']).upper()}** "
                        f"(assurance {(anc.get('assurance') or 0):.0%}).",
                    ]
                if oc:
                    net = oc.get("net_profit_impact_median")
                    lines.append(
                        f"- Short-term cost of running it: forgo ≈ "
                        f"{oc.get('forgone_kpi_median', 0):,.0f} KPI, spend Δ "
                        f"{oc.get('spend_delta', 0):+,.0f}"
                        + (
                            f", net ${net:+,.0f}."
                            if net is not None
                            else " (give a margin for net-$ impact)."
                        )
                    )
                lines.append(
                    "\nRun `simulate_experiment` for the full A/A·A/B methodology "
                    "comparison (power, MDE, false-positive rate by method)."
                )
        except Exception:  # noqa: BLE001 — enrichment is additive, never blocks
            logger.exception("design_experiment_plan economics enrichment failed")

    return Command(
        update={
            "messages": [
                ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def simulate_experiment(
    channel: str,
    state: Annotated[dict, InjectedState],
    design_key: str = None,
    duration: int = 8,
    intensity_pct: float = 50.0,
    geo_design: str = "scaling",
    amplitude_pct: float = 50.0,
    block_weeks: int = 2,
    n_pairs: int = None,
    margin: float = None,
    kpi_kind: str = "revenue",
    seed: int = 42,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Run A/A and A/B simulations on the project's HISTORICAL data to compare
    experiment methodologies for `channel` — the rigorous "will this test
    actually work, and which estimator should we trust?" check.

    For each candidate methodology (pooled DiD, per-pair DiD, synthetic-control
    geo, or national on/off) it reports:
    - **A/A false-positive rate** — slide the estimator over no-treatment
      windows; an estimator whose FPR far exceeds 5% is INVALID for this data
      (its analytic SE is fooled by autocorrelation), no matter its nominal power.
    - **A/B empirical power & MDE** — inject the model's predicted lift (or a
      fixed lift pre-fit) onto real history and measure detection at the
      size-calibrated threshold.
    - the design's **opportunity cost** and the model's **expected-effect
      anchor**, when a model is fitted.

    It then recommends the methodology that is valid AND powered AND cheapest.
    Heavier than `design_experiment_plan` (many estimator passes over history +
    posterior passes); use it once a design is worth committing to.
    """
    import os as _os

    _activate_thread(config)
    dataset_path = state.get("dataset_path")
    if not dataset_path or not _os.path.exists(dataset_path):
        return _simple_msg(
            "No dataset loaded — load or generate one before simulating.",
            tool_call_id,
        )
    kpi = (state.get("model_spec") or {}).get("kpi")
    if not kpi:
        return _simple_msg(
            "No KPI configured — configure_model first so the simulator knows "
            "the outcome series.",
            tool_call_id,
        )
    design_params = {
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channel": channel,
        "design_key": design_key,
        "duration": int(duration),
        "design": geo_design,
        "intensity_pct": float(intensity_pct),
        "amplitude_pct": float(amplitude_pct),
        "block_weeks": int(block_weeks),
        "seed": int(seed),
    }
    if n_pairs is not None:
        design_params["n_pairs"] = int(n_pairs)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "experiment_economics",
        {
            "design_params": design_params,
            "run_simulation": True,
            "margin": margin,
            "kpi_kind": kpi_kind,
        },
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def suggest_experiment(
    channel: str,
    state: Annotated[dict, InjectedState],
    margin: float = None,
    price: float = None,
    kpi_kind: str = "revenue",
    duration_min: int = 4,
    duration_max: int = 12,
    intensity_min: float = 50.0,
    intensity_max: float = 100.0,
    include_holdout: bool = True,
    durations: list[int] = None,
    scaling_intensities: list[float] = None,
    footprints: list[str] = None,
    max_draws: int = 80,
    seed: int = 42,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Suggest a runnable experiment SETUP for `channel` from the fitted model,
    and compute the PARETO FRONT of designs.

    Explores a grid of designs (holdout vs scaling, footprint, duration — or
    national flighting) bounded by the caller's RANGES and ranks them on four
    objectives the client trades off: **lowest MDE** (precision), **highest
    statistical power** (target 80%), **smallest short-term cost** (opportunity
    cost of deviating from business-as-usual), and **shortest duration**. It
    returns the non-dominated Pareto front plus a single recommended setup —
    test/control groups (or a flighting schedule), intensity, duration, and a
    **cool-down period** derived from the channel's fitted adstock (the washout
    before the treated cells return to BAU).

    Bound the search with `duration_min`/`duration_max` (weeks) and
    `intensity_min`/`intensity_max` (signed spend-variation %, e.g. -100 go dark
    … +150 scale up; the optimizer auto-samples a few points in each range).
    `include_holdout` adds a go-dark baseline. Pass a `margin` (profit per KPI
    unit) for a complete net-$ cost comparison. Requires a fitted model.

    Heavy (a posterior pass per candidate); use it to choose a design to commit
    to. Follow with `plan_experiment` / `preregister_experiment`.
    """
    import os as _os

    _activate_thread(config)
    dataset_path = state.get("dataset_path")
    if not dataset_path or not _os.path.exists(dataset_path):
        return _simple_msg(
            "No dataset loaded — load or generate one before optimizing.",
            tool_call_id,
        )
    kpi = (state.get("model_spec") or {}).get("kpi")
    if not kpi:
        return _simple_msg(
            "No KPI configured — configure_model first so the optimizer knows "
            "the outcome series.",
            tool_call_id,
        )
    op_kwargs: dict = {
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channel": channel,
        "margin": margin,
        "price": price,
        "kpi_kind": kpi_kind,
        "duration_min": int(duration_min),
        "duration_max": int(duration_max),
        "intensity_min": float(intensity_min),
        "intensity_max": float(intensity_max),
        "include_holdout": bool(include_holdout),
        "max_draws": int(max_draws),
        "random_seed": int(seed),
    }
    if durations:
        op_kwargs["durations"] = [int(d) for d in durations]
    if scaling_intensities:
        op_kwargs["scaling_intensities"] = [float(x) for x in scaling_intensities]
    if footprints:
        op_kwargs["footprints"] = [str(f) for f in footprints]
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "experiment_optimizer", op_kwargs
    )
    return _modelop_command(res, state, tool_call_id)


@tool
def identify_structural_parameters(
    channel: str,
    state: Annotated[dict, InjectedState],
    levels: list[float] = None,
    block_weeks: int = None,
    duration: int = 12,
    max_draws: int = 200,
    seed: int = 42,
    config: InjectedConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """Design a MULTI-LEVEL flighting test for `channel` and report how well its
    refit would identify the channel's STRUCTURAL model parameters — the
    saturation curve (ψ), the adstock carryover (α), and the coefficient (β) —
    using the fitted model as the anchor.

    Pass `levels` as ≥3 distinct spend multipliers (e.g. `[0.5, 1.0, 1.5]`) so
    the schedule spans the response curve (curvature → ψ); the block length
    defaults to the channel's adstock washout so sharp pulses identify the
    carryover (α). Returns per-parameter identification **power**, **MDE**, and
    posterior **contraction**, plus a binding (worst-parameter) power.

    This is an OPTIMISTIC UPPER BOUND on what the next refit achieves (a local
    design calculation), not a guarantee — the recommended readout is a full
    structural refit with the experiment weeks appended. Requires a parametric
    geometric-adstock + logistic-saturation national fit; otherwise it returns
    the reduced-form curve/marginal-ROAS identification only. Single-call kernel
    op — follow with `plan_experiment` / `preregister_experiment` to lock it.
    """
    import os as _os

    _activate_thread(config)
    dataset_path = state.get("dataset_path")
    if not dataset_path or not _os.path.exists(dataset_path):
        return _simple_msg(
            "No dataset loaded — load or generate one before designing a test.",
            tool_call_id,
        )
    kpi = (state.get("model_spec") or {}).get("kpi")
    if not kpi:
        return _simple_msg(
            "No KPI configured — configure_model first so the design knows the "
            "outcome series.",
            tool_call_id,
        )
    op_kwargs: dict = {
        "dataset_path": dataset_path,
        "kpi": kpi,
        "channel": channel,
        "duration": int(duration),
        "max_draws": int(max_draws),
        "random_seed": int(seed),
    }
    if levels:
        op_kwargs["levels"] = [float(m) for m in levels]
    if block_weeks is not None:
        op_kwargs["block_weeks"] = int(block_weeks)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "identify_structural_parameters", op_kwargs
    )
    return _modelop_command(res, state, tool_call_id)


from mmm_framework.agents.eda_tools import EDA_TOOLS

# ── Two-tier delegation: orchestrator (fast) → expert (strong) ──────────────

# Compute/code-gen-heavy tools removed from the fast chat tier so it must
# delegate them to the expert. Tunable; the escape hatch
# MMM_AGENT_ORCHESTRATOR_FULL_TOOLS=1 restores them on the orchestrator.
#
# NB: the experiment-design tools (design_experiment_plan / simulate_experiment /
# suggest_experiment) are deliberately NOT here — they are single-call kernel ops
# that return a result in one shot (the heavy compute runs in the kernel either
# way), so they don't need the expert's iterative tool loop the way fitting and
# code-gen do. Keeping them on the orchestrator means the chat tier can actually
# run the experiment-planning flow (design → plan → preregister) directly instead
# of failing when a weak orchestrator model doesn't reliably delegate.
HEAVY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "fit_mmm_model",
        "prior_predictive_check",
        "execute_python",
        "run_marginal_analysis",
        "run_budget_optimizer",
        "run_budget_scenario",
        "test_garden_model",  # fits the candidate on synthetic worlds
        "run_cross_validation",  # refits the model once per rolling origin
    }
)

# Lazily-built, cached expert sub-agent graphs (strong model, mode-gated toolset,
# NO checkpointer), keyed by modeling mode. Building reads the server model config
# once; the server config is static for the process, so a per-mode cached instance
# is correct. The mode also drives the prompt (read from state in agent_node).
_EXPERT_GRAPHS: dict[str, Any] = {}
_EXPERT_GRAPH_LOCK = threading.Lock()


def _get_expert_graph(override: dict | None = None, mode: str | None = None):
    """Build (and, for the server default, cache) the expert sub-agent graph.

    Imports are function-local to break the ``graph`` ↔ ``tools`` module cycle
    (``graph`` imports ``TOOLS`` from here at import time). The expert graph is
    compiled WITHOUT a checkpointer: it shares the live session via the same
    ``thread_id`` (kernel/workspace/model cache), but must not write to the
    orchestrator's conversation checkpoint.

    ``override`` carries the per-request ``X-Expert-*`` selection (model/provider/
    api_key/base_url). With no override we build once and cache the server-default
    graph as a singleton. With an override present we build a FRESH graph each call
    and do NOT cache it — a delegation always precedes seconds-to-minutes of heavy
    work, so the ~ms build is negligible, and not caching per-key avoids any
    cross-user key bleed in hosted mode.
    """
    from mmm_framework.agents.graph import create_agent_graph
    from mmm_framework.agents.llm import build_expert_llm
    from mmm_framework.agents.modes import normalize_mode

    mode = normalize_mode(mode)
    expert_tools = get_tools_for_mode(mode, role="expert")

    override = {k: v for k, v in (override or {}).items() if v}
    if override:
        expert_llm = build_expert_llm(
            provider=override.get("provider"),
            model_name=override.get("model"),
            api_key=override.get("api_key"),
            base_url=override.get("base_url"),
        )
        return create_agent_graph(
            expert_llm,
            checkpointer=None,
            tools=expert_tools,
            role="expert",
            mode=mode,
        )

    cached = _EXPERT_GRAPHS.get(mode)
    if cached is not None:
        return cached
    with _EXPERT_GRAPH_LOCK:
        if mode not in _EXPERT_GRAPHS:
            expert_llm = build_expert_llm()
            _EXPERT_GRAPHS[mode] = create_agent_graph(
                expert_llm,
                checkpointer=None,
                tools=expert_tools,
                role="expert",
                mode=mode,
            )
    return _EXPERT_GRAPHS[mode]


def _final_message_text(messages: list) -> str:
    """Extract the expert's final assistant text (skipping trailing tool msgs).

    Handles both plain-string content and the list-of-content-blocks shape that
    Anthropic/Vertex return.
    """
    for m in reversed(messages or []):
        if isinstance(m, ToolMessage):
            continue
        content = getattr(m, "content", None)
        if isinstance(content, str):
            if content.strip():
                return content
            continue
        if isinstance(content, list):
            parts = [
                (blk.get("text", "") if isinstance(blk, dict) else str(blk))
                for blk in content
                if (isinstance(blk, dict) and blk.get("type") == "text")
                or isinstance(blk, str)
            ]
            text = "\n".join(p for p in parts if p).strip()
            if text:
                return text
    return ""


@tool
def delegate_to_expert(
    state: Annotated[dict, InjectedState],
    task: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Delegate a hard task to the expert sub-agent (a stronger model).

    Use this for model fitting, prior/posterior predictive checks, writing and
    running custom analysis code, budget optimization, marginal analysis,
    experiment design, or any multi-step quantitative reasoning the fast chat
    tier should not attempt itself. The expert shares THIS session — the same
    dataset, model specification, warm ``execute_python`` kernel, fitted model,
    and workspace — so pass only a clear, self-contained description of what to
    do, never the data or the spec JSON. The expert runs its own tool loop and
    returns a summary; relay that summary to the user rather than redoing it.

    Args:
        task: A single, precise, self-contained instruction for the expert
            (e.g. "Fit the configured model, then report R-hat, ESS and
            divergences and flag any convergence problems").

    Returns:
        A Command whose message is the expert's summary, with any model or
        dashboard state the expert produced folded back into the session.
    """
    from langchain_core.messages import HumanMessage

    thread_id = _activate_thread(config)
    # The frontend's X-Expert-* selection rides in the injected RunnableConfig's
    # `configurable` (set by /chat). An empty/absent override falls back to the
    # server-configured expert (or the chat model) inside _get_expert_graph.
    configurable = (config or {}).get("configurable") or {}
    expert_override = {
        "model": configurable.get("expert_model"),
        "provider": configurable.get("expert_provider"),
        "api_key": configurable.get("expert_api_key"),
        "base_url": configurable.get("expert_base_url"),
    }
    # The modeling mode rides in `configurable` (set by /chat) and on the
    # orchestrator state; the expert inherits it so its prompt + tool set match.
    expert_mode = configurable.get("modeling_mode") or state.get("modeling_mode")
    try:
        recursion_limit = int(os.environ.get("MMM_AGENT_EXPERT_RECURSION_LIMIT", "60"))
    except ValueError:
        recursion_limit = 60

    # Seed the expert with a full AgentState mirror of the current session, so it
    # can read the dataset/spec/status. No checkpointer => these values are the
    # starting state (not loaded from a checkpoint).
    init_state = {
        "messages": [HumanMessage(content=task)],
        "dataset_path": state.get("dataset_path"),
        "dataset_info": state.get("dataset_info"),
        "model_spec": state.get("model_spec") or {},
        "locked_fields": state.get("locked_fields") or [],
        "pending_spec_changes": state.get("pending_spec_changes") or [],
        "model_status": state.get("model_status") or "not_started",
        "fit_results_summary": state.get("fit_results_summary"),
        "report_path": state.get("report_path"),
        "dashboard_data": {},
        "context_summary": None,
        "context_summary_count": 0,
        "modeling_mode": expert_mode or "mmm",
    }

    # Stream the expert (values mode) so we keep the LAST full state even if it
    # blows the step budget — that lets us salvage partial progress and a useful
    # steer instead of discarding everything as a bare "delegation failed".
    from langgraph.errors import GraphRecursionError

    last_state: dict | None = None
    hit_limit = False
    try:
        expert_graph = _get_expert_graph(expert_override, mode=expert_mode)
        for chunk in expert_graph.stream(
            init_state,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            },
            stream_mode="values",
        ):
            last_state = chunk
    except GraphRecursionError:
        logger.warning(
            "delegate_to_expert hit recursion limit (%s) for thread %s",
            recursion_limit,
            thread_id,
        )
        hit_limit = True
    except Exception as e:
        logger.exception("delegate_to_expert failed for thread %s", thread_id)
        err = f"Expert delegation failed: {e}"
        return Command(
            update={
                "messages": [
                    ToolMessage(content=err, tool_call_id=tool_call_id, status="error")
                ],
            }
        )

    result = last_state or {}
    summary = _final_message_text(result.get("messages") or [])
    if hit_limit:
        note = (
            "⚠️ The expert reached its step limit before producing a final answer "
            "(it most likely looped on a tool). Hand it a narrower, single-step "
            "task, or call the dedicated tool instead of free-form `execute_python` "
            "— e.g. `prior_predictive_check` for a prior predictive trace, "
            "`fit_mmm_model` to fit, `get_model_diagnostics` for R-hat/ESS."
        )
        summary = f"{note}\n\nPartial progress:\n{summary}".strip() if summary else note
    elif not summary:
        summary = "The expert completed the task but returned no summary text."

    msg_kwargs: dict[str, Any] = {"content": summary, "tool_call_id": tool_call_id}
    if hit_limit:
        msg_kwargs["status"] = "error"
    update: dict[str, Any] = {"messages": [ToolMessage(**msg_kwargs)]}
    # Fold back the session-level state the expert may have mutated. model_spec is
    # a full dict (not a patch envelope), so _merge_spec replaces it; dashboard
    # plot/table refs union via _merge_dashboard.
    if result.get("dashboard_data"):
        update["dashboard_data"] = result["dashboard_data"]
    if result.get("model_spec"):
        update["model_spec"] = result["model_spec"]
    for key in ("model_status", "fit_results_summary", "report_path"):
        if result.get(key) is not None:
            update[key] = result[key]
    return Command(update=update)


# ── Review panel: a team of expert personas (Phase 3) ────────────────────────
# Each persona is the SAME expert sub-agent (strong model + full tool set, incl.
# the Phase-1 validation tools) driven by a persona-specific brief, so its
# feedback is GROUNDED in real tool output rather than vibes. The personas run
# sequentially against the shared session; their plots/tables fold back into the
# dashboard and their write-ups are stitched into one panel review.

_REVIEW_PERSONAS: dict[str, dict[str, str]] = {
    "statistician": {
        "label": "🔬 Expert statistician",
        "brief": (
            "You are an expert Bayesian statistician reviewing a fitted marketing "
            "mix model. Be skeptical and rigorous. Interrogate whether the model is "
            "TRUSTWORTHY: run `validate_model` first, then dig in with "
            "`run_posterior_predictive_checks`, `run_residual_diagnostics`, "
            "`run_channel_diagnostics` (collinearity/identifiability) and "
            "`run_refutation_suite` (robustness to unobserved confounding) as "
            "needed. Report, in a few tight paragraphs: what the diagnostics say, "
            "which estimates are reliable, which are fragile or weakly identified, "
            "and the single most important statistical caveat. Cite the numbers you "
            "saw. Do NOT propose budget changes — that is the planner's job."
        ),
    },
    "media_planner": {
        "label": "📊 Media planner",
        "brief": (
            "You are a seasoned media planner reviewing a fitted marketing mix "
            "model. Focus on PLANNING implications. Use `get_roi_metrics`, "
            "`get_saturation_curves`, `run_marginal_analysis` and "
            "`run_budget_optimizer` to see which channels are saturated, where the "
            "next dollar works hardest (marginal ROAS), and how budget should "
            "shift. Recommend, concretely: which channels to scale up or down, "
            "what to hold, and which one experiment would most reduce the risk in "
            "the plan (consider `recommend_lift_experiments`). Keep it practical."
        ),
    },
    "cmo": {
        "label": "🎯 CMO",
        "brief": (
            "You are a CMO reviewing a fitted marketing mix model. Translate it "
            "into the BUSINESS story for an executive audience. Use "
            "`get_roi_metrics`, `get_estimands` and `get_component_decomposition` "
            "to ground yourself. In plain language: what is media actually driving, "
            "how confident should we be, what decision does this support, and what "
            "is the one risk to watch. No jargon, no MCMC internals — outcomes, "
            "confidence, and the call to make."
        ),
    },
}


@tool
def convene_review_panel(
    state: Annotated[dict, InjectedState],
    focus: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Convene a panel of expert personas to review the fitted model and give
    feedback from their different lenses.

    Calls three expert sub-agents in turn — an expert **statistician** (is the
    model trustworthy? — runs the validation suite), a **media planner** (what
    should we do with the budget? — saturation/marginal/optimizer), and a **CMO**
    (what's the business story and the risk?) — each grounding its feedback in the
    real analysis/validation tools. Use this when the user wants a rounded
    review, a second opinion, multi-perspective feedback, or to "ask the team".

    Args:
        focus: What the panel should review or the question to weigh in on
            (e.g. "Is this model ready to set next quarter's budget?").

    Returns:
        A Command whose message stitches together each persona's write-up, with
        any plots/tables they produced folded back into the dashboard.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    thread_id = _activate_thread(config)
    configurable = (config or {}).get("configurable") or {}
    expert_override = {
        "model": configurable.get("expert_model"),
        "provider": configurable.get("expert_provider"),
        "api_key": configurable.get("expert_api_key"),
        "base_url": configurable.get("expert_base_url"),
    }
    expert_mode = configurable.get("modeling_mode") or state.get("modeling_mode")
    try:
        recursion_limit = int(os.environ.get("MMM_AGENT_EXPERT_RECURSION_LIMIT", "60"))
    except ValueError:
        recursion_limit = 60

    base_state = {
        "dataset_path": state.get("dataset_path"),
        "dataset_info": state.get("dataset_info"),
        "model_spec": state.get("model_spec") or {},
        "locked_fields": state.get("locked_fields") or [],
        "pending_spec_changes": state.get("pending_spec_changes") or [],
        "model_status": state.get("model_status") or "not_started",
        "fit_results_summary": state.get("fit_results_summary"),
        "report_path": state.get("report_path"),
        "context_summary": None,
        "context_summary_count": 0,
        "modeling_mode": expert_mode or "mmm",
    }

    sections: list[str] = []
    merged_dashboard = dict(state.get("dashboard_data") or {})

    def _merge(into: dict, extra: dict) -> None:
        for k, v in (extra or {}).items():
            if isinstance(v, list) and isinstance(into.get(k), list):
                into[k] = into[k] + v
            elif isinstance(v, list):
                into[k] = list(v)
            else:
                into[k] = v

    for persona in _REVIEW_PERSONAS.values():
        task = (
            f"{persona['brief']}\n\n## Focus of this review\n{focus}\n\n"
            "Report ONLY your own perspective, concisely (a few short paragraphs)."
        )
        init_state = {
            **base_state,
            "messages": [HumanMessage(content=task)],
            "dashboard_data": {},
        }
        text = ""
        try:
            graph = _get_expert_graph(expert_override, mode=expert_mode)
            last_state: dict | None = None
            for chunk in graph.stream(
                init_state,
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": recursion_limit,
                },
                stream_mode="values",
            ):
                last_state = chunk
            result = last_state or {}
            text = _final_message_text(result.get("messages") or [])
            _merge(merged_dashboard, result.get("dashboard_data") or {})
        except GraphRecursionError:
            text = "(reached the step limit before finishing this review)"
        except Exception as e:  # noqa: BLE001
            logger.exception("review panel persona failed for thread %s", thread_id)
            text = f"(this reviewer could not complete: {e})"
        sections.append(f"### {persona['label']}\n\n{text or '(no response)'}")

    summary = "## Review panel\n\n" + "\n\n".join(sections)
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]
    }
    if merged_dashboard:
        update["dashboard_data"] = merged_dashboard
    return Command(update=update)


# ── Model Garden: author / version / test / share bespoke models ─────────────


# Static (AST-only) source validation + the register core are shared with the
# REST `POST /model-garden` endpoint; re-exported here for the tool + tests.
from mmm_framework.agents.garden_registry import (  # noqa: E402
    register_garden_model_core as _register_garden_model_core,
    static_class_name as _garden_static_class_name,  # noqa: F401 - re-exported for tests
)


def _garden_org_for(tid: str | None) -> tuple[str, str]:
    """(project_id, org_id) for the active session."""
    from mmm_framework.api import sessions as sessions_store

    project_id = sessions_store.resolve_project_id(tid)
    return project_id, sessions_store.resolve_org_id(project_id)


def _garden_copy_source_to_session(row: dict, tid: str | None) -> str:
    """Copy a registered model's source into the session workspace so the kernel
    (incl. the container kernel, which only mounts the thread dir) can import it.
    Returns the thread-local source path."""
    import shutil

    dest_dir = _ws.garden_loaded_dir(row["name"], row["version"], tid)
    dest = dest_dir / "model.py"
    shutil.copyfile(row["source_path"], dest)
    return str(dest)


@tool
def register_garden_model(
    state: Annotated[dict, InjectedState],
    source_code: str,
    name: str,
    docs: str = "",
    version: int | None = None,
    tags: list[str] | None = None,
    dataset_schema: dict | None = None,
    recommended_fit: dict | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Register a bespoke MMM model into the org's Model Garden as a DRAFT.

    ``source_code`` is the full Python source defining a `BayesianMMM` subclass
    (recommended: subclass `mmm_framework.garden.CustomMMM`). The source is
    validated statically (parsed, the model class located) and saved to the
    org's garden store — it is NOT executed here. Call `test_garden_model` next
    to run the compatibility suite (which fits it in the sandbox) and, on pass,
    promote it to `tested`; then `publish_garden_model` shares it org-wide.

    Use this when an expert has authored or finalized a custom model they want
    versioned, documented, and reusable across projects. ``name`` is the stable
    model name (versions auto-increment); ``docs`` documents what it does and
    when to use it; ``dataset_schema`` (optional) declares data requirements
    (e.g. {"requires_geo": true, "min_channels": 2}); ``recommended_fit``
    (optional) default fit knobs (e.g. {"method": "nuts", "draws": 2000}).
    """
    tid = _activate_thread(config)
    _project_id, org_id = _garden_org_for(tid)
    try:
        row = _register_garden_model_core(
            org_id=org_id,
            source_code=source_code,
            name=name,
            docs=docs,
            version=version,
            tags=tags,
            dataset_schema=dataset_schema,
            recommended_fit=recommended_fit,
        )
    except ValueError as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not register model: {e}",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )
    msg = (
        f"Registered garden model **{name}** v{row['version']} "
        f"(class `{(row.get('manifest') or {}).get('class_name')}`) as a draft. Run "
        f"`test_garden_model('{name}')` to check compatibility before publishing."
    )
    return Command(
        update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
    )


@tool
def list_garden_models(
    state: Annotated[dict, InjectedState],
    status: str | None = None,
    name: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """List Model Garden models available to this project's organization.

    Shows bespoke models authored anywhere in the org that can be loaded and
    re-fit here. Optionally filter by ``status`` ('draft'|'tested'|'published'|
    'deprecated') or ``name`` (to see every version of one model). Use this to
    discover reusable models before loading one with `load_garden_model`.
    """
    tid = _activate_thread(config)
    from mmm_framework.api import sessions as sessions_store

    _project_id, org_id = _garden_org_for(tid)
    rows = sessions_store.list_garden_models(
        org_id, name=name, status=status, latest_only=(name is None)
    )
    if not rows:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No garden models found for this organization yet. "
                        "Author one with `register_garden_model`.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    lines = [
        "### Model Garden",
        "",
        "| Model | Version | Status | Score | Docs |",
        "|---|---|---|---|---|",
    ]
    records = []
    for r in rows:
        report = r.get("compat_report") or {}
        score = report.get("score")
        score_s = "—" if score is None else f"{score}"
        docs = (r.get("docs") or "").replace("\n", " ")[:60]
        lines.append(
            f"| {r['name']} | v{r['version']} | {r['status']} | {score_s} | {docs} |"
        )
        records.append(
            {
                "name": r["name"],
                "version": r["version"],
                "status": r["status"],
                "score": score,
                "docs": docs,
            }
        )
    update: dict[str, Any] = {
        "messages": [ToolMessage(content="\n".join(lines), tool_call_id=tool_call_id)]
    }
    return Command(update=update)


def _garden_schema_warnings(schema: dict, spec: dict, dataset_path: str | None) -> str:
    """Advisory check that the consuming project's data can satisfy a model's
    declared requirements. Channels are project-specific (the model is
    channel-agnostic), so this checks SHAPE not exact names."""
    notes: list[str] = []
    if not isinstance(schema, dict) or not schema:
        return ""
    n_channels = len(spec.get("media_channels") or [])
    min_ch = int(schema.get("min_channels", 0) or 0)
    if min_ch and n_channels < min_ch:
        notes.append(
            f"this model expects ≥{min_ch} media channels but the current spec has {n_channels}"
        )
    if schema.get("requires_geo"):
        has_geo = bool(spec.get("geo") or spec.get("has_geo"))
        if not has_geo:
            notes.append(
                "this model expects a geo panel — confirm the dataset has a Geography dimension"
            )
    if schema.get("expects_controls") and not (spec.get("control_variables") or []):
        notes.append("this model expects control variables but none are configured")

    # Flexible data layer: a model may declare role requirements (DatasetRole) and
    # data capabilities. These are read from the manifest's dataset_schema, merged
    # AST-side at registration. Advisory shape checks only.
    required_roles = schema.get("required_roles") or []
    if "indicator" in required_roles:
        notes.append(
            "this model reads its inputs as latent-measurement INDICATORS (every "
            "measured column), not as media channels — so ROI/budget/experiment "
            "tools do not apply to it"
        )
    if "predictor" in required_roles and n_channels == 0:
        notes.append(
            "this model needs at least one predictor/media column but the spec has none"
        )
    req_caps = schema.get("required_capabilities") or []
    if "GEO_PANEL" in req_caps and not (spec.get("geo") or spec.get("has_geo")):
        notes.append(
            "this model expects a geo panel — confirm the dataset has a Geography dimension"
        )
    if "HAS_TRIALS" in req_caps:
        notes.append(
            "this model needs a trials/denominator column (a binomial-count outcome)"
        )
    return (" ⚠️ Data-fit notes: " + "; ".join(notes) + ".") if notes else ""


@tool
def load_garden_model(
    state: Annotated[dict, InjectedState],
    name: str,
    version: int | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load a garden model into this session so the NEXT fit re-fits it on this
    project's data.

    Resolves the model (a specific ``version`` or the latest published), copies
    its source into the session workspace, and stages it into the model spec
    (``garden_ref`` + the model's recommended fit settings). After loading, call
    `fit_mmm_model` to re-fit the bespoke model on the current dataset, then the
    usual analysis tools (`get_roi_metrics`, etc.) work as normal. This is how a
    model authored in one project is reused in another.
    """
    tid = _activate_thread(config)
    from mmm_framework.api import sessions as sessions_store

    _project_id, org_id = _garden_org_for(tid)
    if version is not None:
        row = sessions_store.get_garden_model(
            org_id=org_id, name=name, version=int(version)
        )
    else:
        row = sessions_store.get_latest_garden_model(
            org_id, name, status="published"
        ) or sessions_store.get_latest_garden_model(org_id, name)
    if not row:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Garden model **{name}** not found in this org. "
                        "Use `list_garden_models` to see what's available.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )

    try:
        dest = _garden_copy_source_to_session(row, tid)
    except Exception as e:  # noqa: BLE001
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not stage model source: {e}",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )

    manifest = row.get("manifest") or {}
    garden_ref = {
        "name": name,
        "version": row["version"],
        "source_path": dest,
        "class_name": manifest.get("class_name"),
        "contract_version": manifest.get("contract_version"),
    }
    spec = dict(state.get("model_spec") or {}) if isinstance(state, dict) else {}
    spec["garden_ref"] = garden_ref
    rec = manifest.get("recommended_fit") or {}
    if rec:
        inf = dict(spec.get("inference") or {})
        for k in ("method", "draws", "tune", "chains", "target_accept", "random_seed"):
            if k in rec:
                inf[k] = rec[k]
        spec["inference"] = inf

    warn = _garden_schema_warnings(
        manifest.get("dataset_schema") or {},
        spec,
        state.get("dataset_path") if isinstance(state, dict) else None,
    )
    status_note = (
        ""
        if row["status"] == "published"
        else f" (note: this model is '{row['status']}', not yet published)"
    )
    # Mode reconcile (auto-SUGGEST, never auto-switch): if the loaded family's kind
    # doesn't fit the session's modeling mode, surface a one-line switch suggestion.
    from mmm_framework.agents.modes import reconcile_mode_with_model

    _sess = sessions_store.get_session(tid) or {}
    _recon = reconcile_mode_with_model(
        _sess.get("modeling_mode"), {"model_kind": manifest.get("model_kind")}
    )
    mode_note = f"\n\n💡 {_recon['note']}" if _recon.get("note") else ""
    msg = (
        f"Loaded garden model **{name}** v{row['version']}{status_note}. The next "
        "`fit_mmm_model` will re-fit it on this project's data." + warn + mode_note
    )
    return Command(
        update={
            "model_spec": spec,
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


@tool
def test_garden_model(
    state: Annotated[dict, InjectedState],
    name: str,
    version: int | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Run the compatibility suite on a registered garden model (in the sandbox)
    and, if its blocking tiers pass, promote it from `draft` to `tested`.

    The suite fits the model on synthetic worlds with known causal truth and
    checks the contract the oracle relies on (build, fit, scaling, trace naming,
    every read-op, and accuracy vs ground truth). Call this after
    `register_garden_model`. Heavy (it fits models) — runs in the session kernel.
    """
    tid = _activate_thread(config)
    from mmm_framework.api import sessions as sessions_store

    _project_id, org_id = _garden_org_for(tid)
    if version is not None:
        row = sessions_store.get_garden_model(
            org_id=org_id, name=name, version=int(version)
        )
    else:
        row = sessions_store.get_latest_garden_model(org_id, name)
    if not row:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Garden model **{name}** not found.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )

    try:
        dest = _garden_copy_source_to_session(row, tid)
    except Exception as e:  # noqa: BLE001
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not stage model source for testing: {e}",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )

    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "garden_compat",
        {
            "source_path": dest,
            "class_name": (row.get("manifest") or {}).get("class_name"),
        },
    )
    if res.get("error"):
        return _modelop_command(res, state, tool_call_id)

    report = res.get("compat_report") or (res.get("dashboard") or {}).get(
        "garden_compat"
    )
    if report:
        sessions_store.set_garden_compat_report(row["id"], report)

    note = ""
    if report and report.get("blocking_passed") and row["status"] == "draft":
        try:
            sessions_store.transition_garden_model(row["id"], "tested")
            note = (
                f"\n\n✅ Promoted **{name}** v{row['version']} to **tested**. "
                "Use `publish_garden_model` to share it across the org."
            )
        except ValueError:
            pass
    elif report and not report.get("blocking_passed"):
        note = "\n\n❌ Blocking tiers failed — fix the issues above, re-register, and test again."

    if res.get("content"):
        res = dict(res)
        res["content"] = res["content"] + note
    return _modelop_command(res, state, tool_call_id)


@tool
def publish_garden_model(
    state: Annotated[dict, InjectedState],
    name: str,
    version: int,
    note: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Publish a TESTED garden model (`tested` -> `published`) so every project
    in the organization can load it.

    This is the human approval gate: only call it when the user explicitly asks
    to publish/share a model. The model must already be `tested` (its
    compatibility suite passed). Published versions are immutable — to change a
    published model, register a new version.
    """
    tid = _activate_thread(config)
    from mmm_framework.api import sessions as sessions_store

    _project_id, org_id = _garden_org_for(tid)
    row = sessions_store.get_garden_model(
        org_id=org_id, name=name, version=int(version)
    )
    if not row:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Garden model **{name}** v{version} not found.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )
    try:
        updated = sessions_store.transition_garden_model(
            row["id"], "published", note=note or None
        )
    except ValueError as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Could not publish: {e}",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                ]
            }
        )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"📦 Published **{name}** v{updated['version']} — it is "
                    "now available to every project in the org via `load_garden_model`.",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    )


@tool
def suggest_model_improvements(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Analyze the currently fitted model's diagnostics and suggest concrete
    changes to improve FITTING TIME and ACCURACY.

    Reads convergence (divergences / R-hat / ESS) and parameter-learning
    (prior-dominated parameters) signals and returns ranked, actionable advice
    (raise target_accept, add tune/draws, switch sampler, tighten priors, add
    calibrating experiments, etc.). Call after a fit, especially when the model
    sampled poorly or the user asks how to make it better/faster.
    """
    _activate_thread(config)
    res = _KERNELS.get_or_spawn(get_current_thread()).run_model_op(
        "garden_tune_suggestions", {}
    )
    return _modelop_command(res, state, tool_call_id)


# List of all tools
TOOLS = [
    # Step 1 — Define the question (pre-registration)
    *[t for t in CAUSAL_TOOLS if t.name == "define_research_question"],
    # Data
    generate_synthetic_data,
    load_from_bigquery,
    load_from_gcs,
    sync_data_connection,
    inspect_dataset,
    # Step 2 — Data quality (pre-fit): validate_data, run_eda, detect_outliers,
    # apply_outlier_treatment
    *EDA_TOOLS,
    # Step 2 — Tell the story / DAG
    *[
        t
        for t in CAUSAL_TOOLS
        if t.name
        in ("propose_dag", "validate_causal_identification", "build_model_from_dag")
    ],
    # Config management
    configure_model,
    get_current_config,
    update_model_setting,
    save_config,
    load_config,
    list_configs,
    delete_config,
    # Step 4 — Prior predictive (before fitting)
    *[t for t in CAUSAL_TOOLS if t.name == "prior_predictive_check"],
    # Model fitting
    fit_mmm_model,
    save_fitted_model,
    load_fitted_model,
    list_saved_models,
    # Analysis
    get_roi_metrics,
    get_estimands,
    get_component_decomposition,
    get_model_diagnostics,
    get_adstock_weights,
    get_saturation_curves,
    # Validation / verification (Phase 1)
    validate_model,
    run_posterior_predictive_checks,
    run_residual_diagnostics,
    run_channel_diagnostics,
    run_refutation_suite,
    run_cross_validation,
    # Simulation-based calibration (inference-engine calibration)
    *[t for t in CAUSAL_TOOLS if t.name == "run_calibration_check"],
    run_budget_scenario,
    run_marginal_analysis,
    # Decision layer: learnings -> budget + next experiment
    run_budget_optimizer,
    recommend_lift_experiments,
    compute_experiment_priorities,
    design_experiment_plan,
    simulate_experiment,
    suggest_experiment,
    identify_structural_parameters,
    plan_experiment,
    preregister_experiment,
    record_experiment_readout,
    apply_experiment_calibration,
    log_experiment,
    list_experiment_log,
    get_run_history,
    # Step 8 — Sensitivity (post-fit)
    *[t for t in CAUSAL_TOOLS if t.name == "leave_one_out_decomposition"],
    # Pre-registration: lock the plan + check divergence (were previously unregistered)
    *[
        t
        for t in CAUSAL_TOOLS
        if t.name in ("define_analysis_plan", "check_spec_divergence")
    ],
    # Cross-cutting — assumptions + workflow tracking
    *[
        t
        for t in CAUSAL_TOOLS
        if t.name in ("record_assumption", "list_assumptions", "mark_workflow_step")
    ],
    # Model Garden — author / version / test / share bespoke models
    register_garden_model,
    list_garden_models,
    load_garden_model,
    test_garden_model,
    publish_garden_model,
    suggest_model_improvements,
    # Session
    get_session_status,
    # Library discovery (reach ALL features)
    library_reference,
    bayesian_workflow_reference,
    # Preferences, branding & templates
    get_preferences,
    save_preference,
    list_templates,
    extract_brand_from_website,
    # Knowledge base (project-level RAG)
    search_knowledge_base,
    list_knowledge_base,
    # Workspace filesystem (see & grep output)
    list_workspace_files,
    read_workspace_file,
    grep_workspace,
    # Reusable past results
    query_past_results,
    # Ad-hoc
    execute_python,
    reset_namespace,
    # Reporting
    generate_project_report,
    generate_slide_deck,
    generate_client_report,
    generate_model_defense_report,
    generate_client_slides,
]


# Two-tier toolsets derived from TOOLS:
#   EXPERT_TOOLS       — full power for the strong sub-agent, minus delegate
#                        (the expert must not recurse into itself).
#   ORCHESTRATOR_TOOLS — the fast chat tier: TOOLS + delegate_to_expert, with the
#                        heavy/code-gen tools removed so it must delegate them.
#                        MMM_AGENT_ORCHESTRATOR_FULL_TOOLS=1 keeps every tool on
#                        the orchestrator (prompt-driven delegation instead).
EXPERT_TOOLS = list(TOOLS)
# Orchestrator-only tools that spawn expert sub-agents — appended AFTER the
# EXPERT_TOOLS snapshot so the expert can't recurse into them.
TOOLS = TOOLS + [delegate_to_expert, convene_review_panel]
if os.environ.get("MMM_AGENT_ORCHESTRATOR_FULL_TOOLS") == "1":
    ORCHESTRATOR_TOOLS = list(TOOLS)
else:
    ORCHESTRATOR_TOOLS = [t for t in TOOLS if t.name not in HEAVY_TOOL_NAMES]


# ===========================================================================
# Mode-aware tool gating (the oracle now spans more than MMM)
# ===========================================================================
#
# The modeling mode (see ``agents.modes``) selects which tools are bound. MMM is
# the full surface (today's behavior); the other modes drop the MMM-specific
# ROI/budget/experiment tools, keep the causal-identification tools where they
# apply, and always keep the generalizable spine (data/EDA, config/fit/diagnostics,
# estimands, garden, kernel/workspace/KB, reporting). Composed with the existing
# role filter (orchestrator drops HEAVY tools + keeps delegate; expert keeps the
# heavy tools, no delegate). ``mmm`` reproduces ORCHESTRATOR_TOOLS / EXPERT_TOOLS
# exactly (golden-tested).

#: MMM-only — meaningful only with media channels / spend.
_MMM_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_roi_metrics",
        "get_adstock_weights",
        "get_saturation_curves",
        "run_budget_scenario",
        "run_marginal_analysis",
        "run_budget_optimizer",
        "recommend_lift_experiments",
        "compute_experiment_priorities",
        "design_experiment_plan",
        "simulate_experiment",
        "suggest_experiment",
        "identify_structural_parameters",
        "plan_experiment",
        "preregister_experiment",
        "record_experiment_readout",
        "apply_experiment_calibration",
        "log_experiment",
        "list_experiment_log",
        # validation tools that need media channels / the MMM forward pass
        "run_channel_diagnostics",
        "run_refutation_suite",
        "run_cross_validation",
    }
)

#: Causal-identification tools — central in mmm / causal, available in general,
#: dropped only in the purely descriptive (measurement) mode.
_CAUSAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "propose_dag",
        "validate_causal_identification",
        "build_model_from_dag",
        "leave_one_out_decomposition",
    }
)

#: Every non-delegate tool that is NOT MMM-only and NOT causal — the spine present
#: in EVERY mode (data/EDA, pre-registration, config/fit/diagnostics, estimands,
#: garden, kernel/workspace/KB, reporting).
_ALL_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in EXPERT_TOOLS)
_SPINE_TOOL_NAMES: frozenset[str] = (
    _ALL_TOOL_NAMES - _MMM_ONLY_TOOL_NAMES - _CAUSAL_TOOL_NAMES
)

#: Tools that spawn expert sub-agents — only the orchestrator gets them (the
#: expert must not recurse). They are NOT in EXPERT_TOOLS / _ALL_TOOL_NAMES, so
#: get_tools_for_mode special-cases them in (orchestrator) / out (expert).
_ORCHESTRATOR_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {"delegate_to_expert", "convene_review_panel"}
)

_MODE_TOOL_NAMES: dict[str, frozenset[str]] = {
    "mmm": _ALL_TOOL_NAMES,  # full surface — identical to today
    "causal_inference": _SPINE_TOOL_NAMES | _CAUSAL_TOOL_NAMES,
    "general_bayes": _SPINE_TOOL_NAMES | _CAUSAL_TOOL_NAMES,
    "descriptive": _SPINE_TOOL_NAMES,
}


def get_tools_for_mode(
    mode: str = "mmm",
    role: str | None = None,
    *,
    full_orchestrator: bool | None = None,
) -> list:
    """The tools to bind for ``(mode, role)``.

    ``mode='mmm'`` + ``role='orchestrator'`` reproduces ``ORCHESTRATOR_TOOLS``
    exactly, and ``mode='mmm'`` + ``role='expert'`` reproduces ``EXPERT_TOOLS``
    (golden-tested). Non-MMM modes drop the MMM-only tools (and, for descriptive,
    the causal tools); the orchestrator additionally drops ``HEAVY_TOOL_NAMES``
    (unless ``MMM_AGENT_ORCHESTRATOR_FULL_TOOLS`` / ``full_orchestrator``) and keeps
    ``delegate_to_expert``; the expert keeps the heavy tools and never gets delegate.
    """
    allowed = _MODE_TOOL_NAMES.get(mode, _MODE_TOOL_NAMES["mmm"])
    pool = EXPERT_TOOLS if role == "expert" else TOOLS  # TOOLS carries orch-only tools
    base = [
        t for t in pool if t.name in allowed or t.name in _ORCHESTRATOR_ONLY_TOOL_NAMES
    ]
    if role == "expert":
        return [t for t in base if t.name not in _ORCHESTRATOR_ONLY_TOOL_NAMES]
    if role == "orchestrator":
        full = (
            full_orchestrator
            if full_orchestrator is not None
            else os.environ.get("MMM_AGENT_ORCHESTRATOR_FULL_TOOLS") == "1"
        )
        if not full:
            base = [t for t in base if t.name not in HEAVY_TOOL_NAMES]
        return base  # delegate_to_expert / convene_review_panel retained
    return [t for t in base if t.name not in _ORCHESTRATOR_ONLY_TOOL_NAMES]

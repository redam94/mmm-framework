import os
import json
import importlib
from typing import Annotated, Any, Optional
import io
import contextlib
import traceback

from langchain_core.tools import tool, InjectedToolCallId, InjectedToolArg
from langchain_core.messages import ToolMessage
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
    MFFConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    PriorConfigBuilder,
    BayesianMMM,
    load_mff,
)
from mmm_framework.builders.model import SeasonalityConfigBuilder
from mmm_framework.builders.prior import (
    PriorConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
)


def _build_prior(p: dict):
    """Convert a {distribution, params} dict into a PriorConfig."""
    dist = p.get("distribution", "half_normal")
    params = p.get("params", {})
    b = PriorConfigBuilder()
    if dist == "normal":
        b.normal(mu=float(params.get("mu", 0.0)), sigma=float(params.get("sigma", 1.0)))
    elif dist == "log_normal":
        b.log_normal(
            mu=float(params.get("mu", 0.0)), sigma=float(params.get("sigma", 1.0))
        )
    elif dist == "gamma":
        b.gamma(
            alpha=float(params.get("alpha", 2.0)), beta=float(params.get("beta", 1.0))
        )
    elif dist == "beta":
        b.beta(
            alpha=float(params.get("alpha", 2.0)), beta=float(params.get("beta", 2.0))
        )
    elif dist == "truncated_normal":
        b.truncated_normal(
            mu=float(params.get("mu", 0.0)),
            sigma=float(params.get("sigma", 1.0)),
            lower=float(params.get("lower", 0.0)),
        )
    elif dist == "half_student_t":
        b.half_student_t(
            nu=float(params.get("nu", 3.0)), sigma=float(params.get("sigma", 1.0))
        )
    else:  # half_normal (default)
        b.half_normal(sigma=float(params.get("sigma", 1.0)))
    return b.build()


from mmm_framework.reporting.helpers import (
    generate_model_summary,
    compute_roi_with_uncertainty,
    compute_component_decomposition,
    _get_diagnostics,
    compute_adstock_weights,
    compute_saturation_curves_with_uncertainty,
    compute_marginal_roi,
)

# The fitted-model cache is thread-scoped + LRU-bounded; imported from
# agents.runtime as _MODEL_CACHE above so the existing `_MODEL_CACHE[...]` and
# `_MODEL_CACHE.get(...)` call sites (here and in causal_tools) keep working.

_MFF_DIMENSION_COLS = ["Geography", "Product", "Campaign", "Outlet", "Creative"]


def _normalize_spec_vars(spec: dict) -> dict:
    """Tolerate ``media_channels`` / ``control_variables`` given as bare name
    strings (``["TV", "Digital"]``) OR as dicts (``[{"name": "TV", ...}]``).

    Weaker models often emit the simpler string form, which previously crashed
    fit with "string indices must be integers". Normalises every entry to a dict
    with at least a ``name`` key, in place, and returns the spec.
    """
    for key in ("media_channels", "control_variables"):
        items = spec.get(key)
        if isinstance(items, list):
            normalized = []
            for it in items:
                if isinstance(it, str):
                    normalized.append({"name": it})
                elif isinstance(it, dict) and "name" in it:
                    normalized.append(it)
                # silently drop malformed entries (None, numbers, dict w/o name)
            spec[key] = normalized
    return spec


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
    n_weeks: int = 104,
    geographies: Optional[list[str]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Generate a synthetic Master Flat File (MFF) dataset for testing and demonstration.
    Use this when the user wants to test the framework but doesn't have their own data.

    Args:
        n_weeks: Number of weeks of data to generate (default 104)
        geographies: Optional list of geographic regions (e.g., ["East", "West"]). If None, national data is generated.

    Returns:
        A Command that updates the dataset_path in the state.
    """
    # Import the synthetic data generator from our example
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples"))
    )
    try:
        from ex_model_workflow import generate_synthetic_mff
    except ImportError:
        return (
            "Failed to import generate_synthetic_mff from examples.ex_model_workflow."
        )

    df = generate_synthetic_mff(n_weeks=n_weeks, geographies=geographies)

    # Write into the session workspace and expose an ABSOLUTE dataset_path so it
    # is readable by execute_python (which runs in the workspace) AND by tools
    # that run in the server cwd (fit/inspect), and is registered for download.
    tid = _activate_thread(config)
    try:
        out_dir = _ws.thread_dir(tid)
        before = _ws.snapshot_dir(out_dir)
        output_path = str(out_dir / "synthetic_mff_data.csv")
    except Exception:
        before, output_path = {}, "synthetic_mff_data.csv"
    df.to_csv(output_path, index=False)
    try:
        _ws.register_generated_files(tid, before, kind="dataset")
    except Exception:
        pass

    info = f"Generated synthetic data with {len(df)} rows. Columns: {', '.join(df.columns.tolist())}"
    if geographies:
        info += f"\nGeographies included: {', '.join(geographies)}"
    else:
        info += "\nNational level data (no geographies)."

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


@tool
def configure_model(
    state: Annotated[dict, InjectedState],
    kpi: str,
    kpi_level: str,
    media_channels: list[str],
    control_variables: list[str],
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

    Returns:
        A Command that updates the model_spec in the state.
    """
    model_spec = {
        "kpi": kpi,
        "kpi_level": kpi_level,
        "media_channels": [{"name": ch} for ch in media_channels],
        "control_variables": [{"name": cv} for cv in control_variables],
        "time_granularity": "weekly",
        "model_type": "numpyro",
    }

    dashboard_data = state.get("dashboard_data", {})
    if dashboard_data is None:
        dashboard_data = {}
    dashboard_data["model_spec"] = model_spec

    return Command(
        update={
            "model_spec": model_spec,
            "model_status": "configured",
            "messages": [
                ToolMessage(
                    content="Model configured successfully.", tool_call_id=tool_call_id
                )
            ],
            "dashboard_data": dashboard_data,
        }
    )


@tool
def fit_mmm_model(
    state: Annotated[dict, InjectedState],
    dataset_path: str,
    model_spec: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """
    Build and fit the Bayesian MMM using the dataset and the configured model specification.
    You must pass the dataset_path and the JSON string representation of the model_spec from the state.

    Args:
        dataset_path: The path to the CSV dataset.
        model_spec: A JSON string representation of the model_spec dictionary.

    Returns:
        A Command that updates the model_status and fit_results_summary in the state.
    """
    _activate_thread(config)
    try:
        spec = json.loads(model_spec)
        _normalize_spec_vars(spec)  # accept bare-string channel/control lists

        # 1. Build MFFConfig
        mff_builder = MFFConfigBuilder()

        # KPI
        kpi_builder = KPIConfigBuilder(spec["kpi"])
        if spec.get("kpi_level") == "geo":
            kpi_builder.by_geo()
        else:
            kpi_builder.national()
        mff_builder.with_kpi_builder(kpi_builder)

        # Per-channel prior overrides keyed by channel name
        media_priors = spec.get("priors", {}).get("media", {})
        control_priors_cfg = spec.get("priors", {}).get("controls", {})

        # Media — read adstock/saturation type + priors from spec
        for media in spec.get("media_channels", []):
            ch_name = media["name"]
            ch_priors = media_priors.get(ch_name, {})
            ch_builder = MediaChannelConfigBuilder(ch_name).national()

            # Adstock
            adstock_cfg = media.get("adstock", {})
            adstock_type = adstock_cfg.get("type", "geometric").lower()
            l_max = int(adstock_cfg.get("l_max", 8))
            ab = AdstockConfigBuilder()
            if adstock_type == "weibull":
                ab.weibull().with_max_lag(l_max)
            elif adstock_type == "delayed":
                ab.delayed().with_max_lag(l_max)
            else:
                ab.geometric().with_max_lag(l_max)
            if "adstock_alpha" in ch_priors:
                ab.with_alpha_prior(_build_prior(ch_priors["adstock_alpha"]))
            if "adstock_theta" in ch_priors:
                ab.with_theta_prior(_build_prior(ch_priors["adstock_theta"]))
            ch_builder.with_adstock_builder(ab)

            # Saturation
            sat_type = media.get("saturation", {}).get("type", "hill").lower()
            sb = SaturationConfigBuilder()
            if sat_type == "logistic":
                sb.logistic()
            elif sat_type in ("michaelis_menten", "michaelis-menten"):
                sb.michaelis_menten()
            elif sat_type == "tanh":
                sb.tanh()
            else:
                sb.hill()
            if "saturation_kappa" in ch_priors:
                sb.with_kappa_prior(_build_prior(ch_priors["saturation_kappa"]))
            if "saturation_slope" in ch_priors:
                sb.with_slope_prior(_build_prior(ch_priors["saturation_slope"]))
            ch_builder.with_saturation_builder(sb)

            # Coefficient prior
            if "coefficient" in ch_priors:
                ch_builder.with_coefficient_prior(
                    _build_prior(ch_priors["coefficient"])
                )

            mff_builder.add_media_builder(ch_builder)

        # Controls — apply coefficient priors from spec when provided
        for control in spec.get("control_variables", []):
            cv_name = control["name"]
            cv_builder = ControlVariableConfigBuilder(cv_name).national()
            cv_priors = control_priors_cfg.get(cv_name, {})
            if cv_priors.get("allow_negative", True) is False:
                cv_builder.positive_only()
            if "coefficient" in cv_priors:
                cv_builder.with_coefficient_prior(
                    _build_prior(cv_priors["coefficient"])
                )
            mff_builder.add_control_builder(cv_builder)

        granularity = spec.get("time_granularity", "weekly").lower()
        if granularity == "daily":
            mff_builder.daily()
        elif granularity == "monthly":
            mff_builder.monthly()
        else:
            mff_builder.weekly()
        mff_builder.with_date_format("%Y-%m-%d")
        mff_config = mff_builder.build()

        # 2. Load Data
        panel = load_mff(dataset_path, mff_config)

        # 3. Model Config — read inference settings from spec when provided
        inf = spec.get("inference", {})
        chains = int(inf.get("chains", 4))
        draws = int(inf.get("draws", 1000))
        tune = int(inf.get("tune", 1000))
        target_accept = float(inf.get("target_accept", 0.85))
        random_seed = int(inf.get("random_seed", 42))

        model_config_builder = (
            ModelConfigBuilder()
            .bayesian_numpyro()
            .with_chains(chains)
            .with_draws(draws)
            .with_tune(tune)
            .with_target_accept(target_accept)
        )

        # Seasonality
        season = spec.get("seasonality", {})
        yearly = int(season.get("yearly", 0))
        monthly = int(season.get("monthly", 0))
        weekly = int(season.get("weekly", 0))
        if yearly > 0 or monthly > 0 or weekly > 0:
            sb = SeasonalityConfigBuilder()
            if yearly > 0:
                sb.with_yearly(order=yearly)
            if monthly > 0:
                sb.with_monthly(order=monthly)
            if weekly > 0:
                sb.with_weekly(order=weekly)
            model_config_builder.with_seasonality_builder(sb)

        model_config = model_config_builder.build()

        # Trend config — read type, structural parameters, and priors from spec
        trend_spec = spec.get("trend", {})
        trend_type = trend_spec.get("type", "linear").lower()
        trend_prior_cfg = spec.get("priors", {}).get("trend", {})
        tb = TrendConfigBuilder()
        if trend_type == "piecewise":
            tb.piecewise()
            if "n_changepoints" in trend_spec:
                tb.with_n_changepoints(int(trend_spec["n_changepoints"]))
            if "changepoint_range" in trend_spec:
                tb.with_changepoint_range(float(trend_spec["changepoint_range"]))
            if "changepoint_prior_scale" in trend_prior_cfg:
                tb.with_changepoint_prior_scale(
                    float(trend_prior_cfg["changepoint_prior_scale"])
                )
        elif trend_type == "spline":
            tb.spline()
            if "n_knots" in trend_spec:
                tb.with_n_knots(int(trend_spec["n_knots"]))
            if "spline_degree" in trend_spec:
                tb.with_spline_degree(int(trend_spec["spline_degree"]))
            if "spline_prior_sigma" in trend_prior_cfg:
                tb.with_spline_prior_sigma(float(trend_prior_cfg["spline_prior_sigma"]))
        elif trend_type == "gaussian_process":
            tb.gaussian_process()
            if "gp_lengthscale_prior_mu" in trend_prior_cfg:
                tb.with_gp_lengthscale(
                    mu=float(trend_prior_cfg["gp_lengthscale_prior_mu"]),
                    sigma=float(trend_prior_cfg.get("gp_lengthscale_prior_sigma", 0.2)),
                )
            if "gp_amplitude_prior_sigma" in trend_prior_cfg:
                tb.with_gp_amplitude(
                    sigma=float(trend_prior_cfg["gp_amplitude_prior_sigma"])
                )
        elif trend_type == "none":
            pass
        else:  # linear
            tb.linear()
            if (
                "growth_prior_mu" in trend_prior_cfg
                or "growth_prior_sigma" in trend_prior_cfg
            ):
                tb.with_growth_prior(
                    mu=float(trend_prior_cfg.get("growth_prior_mu", 0.0)),
                    sigma=float(trend_prior_cfg.get("growth_prior_sigma", 0.1)),
                )
        trend_config = tb.build()

        # 4. Fit Model
        mmm = BayesianMMM(panel, model_config, trend_config)
        results = mmm.fit(random_seed=random_seed)

        # 5. Generate a brief summary
        summary = (
            f"Model fitted successfully! "
            f"Observations: {mmm.n_obs}, Channels: {mmm.n_channels}. "
            f"Trend: {trend_type}, Seasonality: yearly={yearly}/monthly={monthly}/weekly={weekly}, "
            f"Inference: {chains} chains × {draws} draws."
        )

        # Generate full report immediately for convenience
        report_path = "agent_mmm_report.html"
        try:
            from mmm_framework.reporting.generator import ReportBuilder

            report = (
                ReportBuilder().with_model(mmm, results).enable_all_sections().build()
            )
            report.to_html(report_path)
            summary += f" Full HTML report generated at {report_path}."
        except Exception as e:
            summary += f" Note: Report generation failed: {str(e)}"
            report_path = None

        # Cache the model globally so interpretation tools can use it
        _MODEL_CACHE["fitted_model"] = mmm
        _MODEL_CACHE["fit_results"] = results

        # Auto-save the model to disk with a timestamped run name
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{run_id}"
        model_path = os.path.join(_MODELS_DIR, run_name)
        model_saved = False
        try:
            os.makedirs(model_path, exist_ok=True)
            from mmm_framework.serialization import MMMSerializer

            MMMSerializer().save(mmm, results, model_path)
            model_saved = True
            summary += f" Auto-saved as **{run_name}**."
        except Exception as save_err:
            summary += f" (Auto-save failed: {save_err})"

        # Build a structured run record for the artifact log
        channel_names = [m["name"] for m in spec.get("media_channels", [])]
        control_names = [c["name"] for c in spec.get("control_variables", [])]
        model_run = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
            "kpi": spec.get("kpi", ""),
            "channels": channel_names,
            "controls": control_names,
            "trend": trend_type,
            "seasonality": {"yearly": yearly, "monthly": monthly, "weekly": weekly},
            "inference": {
                "chains": chains,
                "draws": draws,
                "tune": tune,
                "target_accept": target_accept,
            },
            "model_path": model_path if model_saved else None,
            "report_path": report_path,
            "summary": summary,
            "n_obs": int(mmm.n_obs),
            "n_channels": int(mmm.n_channels),
        }

        # Write metadata.json alongside the saved model
        if model_saved:
            try:
                with open(os.path.join(model_path, "run_metadata.json"), "w") as f:
                    json.dump(model_run, f, indent=2, default=str)
            except Exception:
                pass

        dashboard_data = state.get("dashboard_data") or {}
        dashboard_data["model_status"] = "completed"
        dashboard_data["summary"] = summary
        dashboard_data["model_run"] = model_run
        if report_path:
            dashboard_data["report_path"] = report_path

        # Auto-populate ROI and decomposition so Results tab fills immediately
        try:
            roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)
            dashboard_data["roi_metrics"] = roi_df.to_dict(orient="records")
        except Exception:
            pass

        try:
            decomp_list = compute_component_decomposition(
                mmm, include_time_series=False
            )
            dashboard_data["decomposition"] = [
                {
                    "component": d.component,
                    "total_contribution": d.total_contribution,
                    "pct_of_total": d.pct_of_total,
                }
                for d in decomp_list
            ]
        except Exception:
            pass

        return Command(
            update={
                "model_status": "completed",
                "fit_results_summary": summary,
                "report_path": report_path,
                "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        error_msg = f"Error fitting model: {str(e)}"

        dashboard_data = state.get("dashboard_data", {}) if "state" in locals() else {}
        if dashboard_data is None:
            dashboard_data = {}
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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found in state. Please fit the model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        roi_df = compute_roi_with_uncertainty(mmm, hdi_prob=0.94)

        # Format as markdown table
        content = "### ROI Analysis\n\n| Channel | Mean ROI | 94% HDI | Prob Profitable |\n|---|---|---|---|\n"
        for _, row in roi_df.iterrows():
            ci = f"[{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}]"
            content += f"| {row['channel']} | {row['roi_mean']:.2f} | {ci} | {row['prob_profitable']:.1%} |\n"

        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["roi_metrics"] = roi_df.to_dict(orient="records")

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error computing ROI: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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
    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if mmm is None or results is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found in state. Please fit the model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        # Don't pass results as second arg, it expects include_time_series bool
        decomp_list = compute_component_decomposition(mmm, include_time_series=False)

        content = "### Component Decomposition\n\n"
        content += "| Component | Contribution | Percentage |\n|---|---|---|\n"

        decomp_json = []
        for d in decomp_list:
            content += f"| {d.component} | {d.total_contribution:,.0f} | {d.pct_of_total:.1%} |\n"
            decomp_json.append(
                {
                    "component": d.component,
                    "total_contribution": d.total_contribution,
                    "pct_of_total": d.pct_of_total,
                }
            )

        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["decomposition"] = decomp_json

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error computing decomposition: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found in state. Please fit the model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        diag = _get_diagnostics(mmm)

        if not diag:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Diagnostics could not be extracted. Make sure ArviZ is installed and the model sampled correctly.",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        content = "### Model Diagnostics\n\n"
        content += f"**Converged:** {'✅ Yes' if diag.get('converged') else '⚠️ No'}\n"
        content += f"**Divergences:** {diag.get('divergences', 0)} (Should be 0)\n"
        content += f"**Max R-hat:** {diag.get('rhat_max', 'N/A')} (Should be < 1.01)\n"
        content += (
            f"**Min Bulk ESS:** {diag.get('ess_bulk_min', 'N/A')} (Should be > 400)\n"
        )
        content += (
            f"**Min Tail ESS:** {diag.get('ess_tail_min', 'N/A')} (Should be > 400)\n"
        )

        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["diagnostics"] = diag

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error computing diagnostics: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found in state. Please fit the model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        adstock = compute_adstock_weights(mmm)

        content = "### Adstock (Carryover) Effects\n\n"
        content += "| Channel | Half-life (Periods) | Total Carryover % | Alpha (Decay Rate) |\n|---|---|---|---|\n"

        adstock_json = {}
        for ch, result in adstock.items():
            content += f"| {ch} | {result.half_life:.1f} | {result.total_carryover:.1%} | {result.alpha_mean:.3f} |\n"
            adstock_json[ch] = {
                "half_life": result.half_life,
                "total_carryover": result.total_carryover,
                "alpha_mean": result.alpha_mean,
            }

        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["adstock"] = adstock_json

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error computing adstock weights: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model found in state. Please fit the model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        curves = compute_saturation_curves_with_uncertainty(mmm)

        content = "### Saturation (Diminishing Returns) Analysis\n\n"
        content += "| Channel | Current Saturation Level | Marginal Response (Next $1) |\n|---|---|---|\n"

        saturation_json = {}
        for ch, curve in curves.items():
            sat_pct = curve.saturation_level
            # Determine if highly saturated
            status = (
                "🔴 High"
                if sat_pct > 0.8
                else "🟡 Medium" if sat_pct > 0.5 else "🟢 Low"
            )
            content += f"| {ch} | {sat_pct:.1%} ({status}) | {curve.marginal_response_at_current:.3f} |\n"
            saturation_json[ch] = {
                "saturation_level": sat_pct,
                "marginal_response_at_current": curve.marginal_response_at_current,
                "status": status.split(" ")[1],
            }

        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["saturation"] = saturation_json

        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data,
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error computing saturation: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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
    """
    _activate_thread(config)
    thread_id = get_current_thread()
    try:
        work_dir = _ws.thread_dir(thread_id)
    except Exception:
        work_dir = None
    before_snapshot = _ws.snapshot_dir(work_dir) if work_dir is not None else {}

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

    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
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
            raise RuntimeError("No workspace directory available for saving results.")
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
            {p.stem for p in _results_dir.glob("*") if p.suffix in (".parquet", ".pkl")}
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
    _ds_path = state.get("dataset_path") if isinstance(state, dict) else None
    if _ds_path:
        env["dataset_path"] = _ds_path
    if _ds_path and ns.get("__mmm_df_source__") != _ds_path:
        try:
            _p = str(_ds_path)
            if not os.path.isabs(_p) and work_dir is not None:
                _cand = os.path.join(str(work_dir), _p)
                if os.path.exists(_cand):
                    _p = _cand
            _too_big = os.path.exists(_p) and os.path.getsize(_p) > 250 * 1024 * 1024
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

    dashboard_data = dict(state.get("dashboard_data") or {})

    if captured_plots:
        # Content-address each figure into the plot store and keep only a
        # lightweight {id, title} ref in state. This stops the full (heavy)
        # Plotly JSON from being re-sent on every turn / re-saved into the
        # LangGraph checkpoint; the frontend fetches each plot once by id and
        # the browser caches it permanently (immutable response). Falls back to
        # an inline figure if the store write fails (back-compat).
        existing_plots = dashboard_data.get("plots", [])
        plot_refs = []
        for fig in captured_plots:
            try:
                pid = _ws.store_plot(fig)
                layout = fig.get("layout") or {}
                t = layout.get("title")
                title = (
                    t.get("text")
                    if isinstance(t, dict)
                    else (t if isinstance(t, str) else "")
                )
                plot_refs.append({"id": pid, "title": title or ""})
            except Exception:
                plot_refs.append(fig)
        dashboard_data["plots"] = existing_plots + plot_refs

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
    if captured_plots:
        content += f"\n\n*Generated {len(captured_plots)} Plotly interactive chart(s). View them in the Plots tab.*"
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
    _NAMESPACE_CACHE.reset()
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
    import copy

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

    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = os.path.join(_CONFIGS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(dict(copy.deepcopy(spec)), f, indent=2)

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


@tool
def list_configs(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """List all saved model configurations with a brief summary of each."""
    if not os.path.exists(_CONFIGS_DIR):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No saved configurations yet. Use `save_config` after configuring a model.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    rows = []
    for fname in sorted(os.listdir(_CONFIGS_DIR)):
        if not fname.endswith(".json"):
            continue
        name = fname[:-5]
        try:
            with open(os.path.join(_CONFIGS_DIR, fname)) as f:
                spec = json.load(f)
            channels = len(spec.get("media_channels", []))
            controls = len(spec.get("control_variables", []))
            rows.append(
                f"- **{name}**: KPI=`{spec.get('kpi','?')}`, {channels} channels, {controls} controls"
            )
        except Exception:
            rows.append(f"- **{name}**: (could not read)")

    content = (
        "### Saved Configurations\n\n" + "\n".join(rows)
        if rows
        else "No saved configurations found."
    )
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
    value: Any,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Update a specific setting in the active model configuration using dot-notation.

    Examples:
      setting_path="inference.draws",           value=2000
      setting_path="inference.chains",          value=4
      setting_path="trend.type",                value="piecewise_linear"
      setting_path="trend.n_changepoints",      value=10
      setting_path="seasonality.yearly",        value=4
      setting_path="kpi",                       value="Revenue"
      setting_path="time_granularity",          value="daily"
      setting_path="media_channels.TV.adstock.type",      value="delayed"
      setting_path="media_channels.TV.adstock.l_max",     value=13
      setting_path="media_channels.TV.saturation.type",   value="logistic"

    For media_channels and control_variables, use the channel/variable name as the key after the list name.
    """
    import copy

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

    try:
        parts = setting_path.split(".")
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

    dashboard_data = dict(state.get("dashboard_data") or {})
    dashboard_data["model_spec"] = new_spec

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Updated **{setting_path}** → `{value}`",
                    tool_call_id=tool_call_id,
                )
            ],
            "model_spec": new_spec,
            "dashboard_data": dashboard_data,
        }
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
    _activate_thread(config)
    lines = ["### Session Status\n"]

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
    fitted = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if fitted is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model in session. Fit a model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    save_dir = os.path.join(_MODELS_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    try:
        from mmm_framework.serialization import MMMSerializer

        MMMSerializer().save(fitted, results, save_dir)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Model saved as **{name}** at `{save_dir}/`.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Save failed: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def load_fitted_model(
    state: Annotated[dict, InjectedState],
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
    config: InjectedConfig = None,
) -> Command:
    """Load a previously saved fitted model from disk by name, making it available for analysis tools."""
    _activate_thread(config)
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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Model **{name}** not found. Available: {available or 'none'}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    try:
        from mmm_framework.serialization import MMMSerializer

        mmm, results = MMMSerializer().load(save_dir)
        _MODEL_CACHE["fitted_model"] = mmm
        _MODEL_CACHE["fit_results"] = results
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Model **{name}** loaded. You can now run analysis tools.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "model_status": "completed",
            }
        )
    except Exception as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Load failed: {exc}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


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

    report_path = "agent_project_report.html"
    slides_path = "agent_project_slides.html"
    errors: list[str] = []

    try:
        html = generate_html_report(report_title, date_str, dashboard, assumptions)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        errors.append(f"Report generation failed: {e}")
        report_path = None

    try:
        slides_html = generate_html_slides(
            report_title, date_str, dashboard, assumptions
        )
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
    client_name: str,
    report_title: Optional[str] = None,
    analysis_period: Optional[str] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:
    """
    Generate a clean, client-ready HTML report from the fitted model.

    Compared to the internal report this version:
    • Omits MCMC diagnostics and trace plots (not relevant to clients)
    • Adds a sticky navigation sidebar for easy section jumping
    • Adds a confidentiality notice in the footer
    • Formats channel names (underscores → spaces, title-case)
    • Uses plain-language methodology description

    Call this after `fit_mmm_model` when the user wants to share results externally.

    Args:
        client_name: Client/company name shown in the header and confidentiality notice.
        report_title: Optional report title (defaults to "Marketing Mix Model Results").
        analysis_period: Optional period string, e.g. "Q1–Q2 2024".
    """
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

    title = report_title or "Marketing Mix Model Results"
    report_path = "agent_client_report.html"

    try:
        from mmm_framework.reporting.generator import ReportBuilder

        builder = (
            ReportBuilder()
            .with_model(mmm, results)
            .with_title(title)
            .with_client(client_name)
            .client_report()
        )
        if analysis_period:
            builder = builder.with_analysis_period(analysis_period)

        report = builder.build()
        report.to_html(report_path)
        summary = (
            f"Client report generated at `{report_path}`. "
            f"Diagnostics and trace plots excluded; navigation sidebar and "
            f"confidentiality notice added."
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
    slides_path = "agent_client_slides.html"

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


@tool
def library_reference(topic: str = None) -> str:
    """Return a menu of EVERY mmm_framework capability the agent can use (data
    loading, standard & extended/mediation/multivariate models, counterfactual
    & budget analysis, lift-test calibration, diagnostics, reporting,
    serialization) with exact import paths and the input-shape/ordering traps.
    Consult this before hand-writing complex code in execute_python. Optionally
    pass a topic substring to filter."""
    if not topic:
        return _LIBRARY_MENU
    blocks = _LIBRARY_MENU.split("\n## ")
    matched = [blocks[0]] + [b for b in blocks[1:] if topic.lower() in b.lower()]
    return "\n## ".join(matched) if len(matched) > 1 else _LIBRARY_MENU


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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model in this session. Fit or load a model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
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
    try:
        result = mmm.what_if_scenario(changes)
        text = json.dumps(result, indent=2, default=str)
        content = (
            f"### Budget scenario\nApplied: {changes}\n```json\n{text[:4000]}\n```"
        )
    except Exception as exc:  # noqa: BLE001
        content = f"Scenario failed: {exc}"
    return Command(
        update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    )


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
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No fitted model in this session. Fit or load a model first.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    chans = None
    if channels:
        try:
            chans = json.loads(channels)
        except Exception:
            chans = None
    try:
        df = mmm.compute_marginal_contributions(
            spend_increase_pct=spend_increase_pct, channels=chans
        )
        content = (
            f"### Marginal analysis (+{spend_increase_pct}% spend)\n```\n"
            f"{df.to_string()[:4000]}\n```"
        )
    except Exception as exc:  # noqa: BLE001
        content = f"Marginal analysis failed: {exc}"
    return Command(
        update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]}
    )


# List of all tools
TOOLS = [
    # Step 1 — Define the question (pre-registration)
    *[t for t in CAUSAL_TOOLS if t.name == "define_research_question"],
    # Data
    generate_synthetic_data,
    inspect_dataset,
    # Step 2 — Tell the story / DAG
    *[
        t
        for t in CAUSAL_TOOLS
        if t.name in ("propose_dag", "validate_causal_identification")
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
    get_component_decomposition,
    get_model_diagnostics,
    get_adstock_weights,
    get_saturation_curves,
    run_budget_scenario,
    run_marginal_analysis,
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
    # Session
    get_session_status,
    # Library discovery (reach ALL features)
    library_reference,
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
    generate_client_report,
    generate_client_slides,
]

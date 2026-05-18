import os
import json
from typing import Annotated, Any, Optional
import io
import contextlib
import traceback

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

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

from mmm_framework.reporting.helpers import (
    generate_model_summary,
    compute_roi_with_uncertainty,
    compute_component_decomposition,
    _get_diagnostics,
    compute_adstock_weights,
    compute_saturation_curves_with_uncertainty
)

# Global cache to store the fitted model and avoid LangGraph msgpack serialization errors
_MODEL_CACHE = {}


@tool
def generate_synthetic_data(
    state: Annotated[dict, InjectedState],
    n_weeks: int = 104, 
    geographies: Optional[list[str]] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
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
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples")))
    try:
        from ex_model_workflow import generate_synthetic_mff
    except ImportError:
        return "Failed to import generate_synthetic_mff from examples.ex_model_workflow."
    
    df = generate_synthetic_mff(n_weeks=n_weeks, geographies=geographies)
    
    output_path = "synthetic_mff_data.csv"
    df.to_csv(output_path, index=False)
    
    info = f"Generated synthetic data with {len(df)} rows. Columns: {', '.join(df.columns.tolist())}"
    if geographies:
        info += f"\nGeographies included: {', '.join(geographies)}"
    else:
        info += "\nNational level data (no geographies)."
        
    dashboard_data = state.get("dashboard_data", {})
    if dashboard_data is None:
        dashboard_data = {}
    dashboard_data["dataset"] = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "geographies": geographies if geographies else ["National"]
    }
        
    return Command(
        update={
            "dataset_path": output_path,
            "dataset_info": info,
            "messages": [ToolMessage(content=info, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data
        }
    )


@tool
def configure_model(
    state: Annotated[dict, InjectedState],
    kpi: str,
    kpi_level: str,
    media_channels: list[str],
    control_variables: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
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
        "model_type": "numpyro"
    }
    
    dashboard_data = state.get("dashboard_data", {})
    if dashboard_data is None:
        dashboard_data = {}
    dashboard_data["model_spec"] = model_spec
    
    return Command(
        update={
            "model_spec": model_spec,
            "model_status": "configured",
            "messages": [ToolMessage(content="Model configured successfully.", tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data
        }
    )


@tool
def fit_mmm_model(
    state: Annotated[dict, InjectedState],
    dataset_path: str, 
    model_spec: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
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
    try:
        spec = json.loads(model_spec)
        
        # 1. Build MFFConfig
        mff_builder = MFFConfigBuilder()
        
        # KPI
        kpi_builder = KPIConfigBuilder(spec["kpi"])
        if spec.get("kpi_level") == "geo":
            kpi_builder.by_geo()
        else:
            kpi_builder.national()
        mff_builder.with_kpi_builder(kpi_builder)
        
        # Media
        for media in spec.get("media_channels", []):
            ch_builder = MediaChannelConfigBuilder(media["name"]).national()
            ch_builder.with_geometric_adstock()
            ch_builder.with_hill_saturation()
            mff_builder.add_media_builder(ch_builder)
            
        # Controls
        for control in spec.get("control_variables", []):
            cv_builder = ControlVariableConfigBuilder(control["name"]).national()
            mff_builder.add_control_builder(cv_builder)
            
        mff_builder.weekly().with_date_format("%Y-%m-%d")
        mff_config = mff_builder.build()
        
        # 2. Load Data
        panel = load_mff(dataset_path, mff_config)
        
        # 3. Model Config
        model_config = ModelConfigBuilder().bayesian_numpyro().with_chains(4).with_draws(1000).with_tune(1000).build()
        trend_config = TrendConfigBuilder().linear().build()
        
        # 4. Fit Model
        mmm = BayesianMMM(panel, model_config, trend_config)
        results = mmm.fit(random_seed=42)
        
        # 5. Generate a brief summary
        summary = f"Model fitted successfully! Observations: {mmm.n_obs}, Channels: {mmm.n_channels}."
        
        # Generate full report immediately for convenience
        report_path = "agent_mmm_report.html"
        try:
            from mmm_framework.reporting.generator import ReportBuilder
            report = (
                ReportBuilder()
                .with_model(mmm, results)
                .enable_all_sections()
                .build()
            )
            report.to_html(report_path)
            summary += f" Full HTML report generated at {report_path}."
        except Exception as e:
            summary += f" Note: Report generation failed: {str(e)}"
            report_path = None
        
        # Cache the model globally so interpretation tools can use it
        _MODEL_CACHE["fitted_model"] = mmm
        _MODEL_CACHE["fit_results"] = results
        
        dashboard_data = state.get("dashboard_data", {}) if "state" in locals() else {}
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["model_status"] = "completed"
        dashboard_data["summary"] = summary
        if report_path:
            dashboard_data["report_path"] = report_path
        
        return Command(
            update={
                "model_status": "completed",
                "fit_results_summary": summary,
                "report_path": report_path,
                "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data
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
                "dashboard_data": dashboard_data
            }
        )

@tool
def get_roi_metrics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Get the Return on Investment (ROI) and probability of profitability for each media channel.
    Call this tool when the user asks about the efficiency, ROI, ROAS, or cost-effectiveness of their media channels.
    """
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(content="No fitted model found in state. Please fit the model first.", tool_call_id=tool_call_id)]})
        
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
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"Error computing ROI: {str(e)}", tool_call_id=tool_call_id)]})


@tool
def get_component_decomposition(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Decompose the KPI (Sales) into its contributing components (Base/Trend vs Media vs Controls).
    Call this tool when the user asks what drove their sales, what percentage of sales came from media, or wants a decomposition breakdown.
    """
    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if mmm is None or results is None:
        return Command(update={"messages": [ToolMessage(content="No fitted model found in state. Please fit the model first.", tool_call_id=tool_call_id)]})
        
    try:
        # Don't pass results as second arg, it expects include_time_series bool
        decomp_list = compute_component_decomposition(mmm, include_time_series=False)
        
        content = "### Component Decomposition\n\n"
        content += "| Component | Contribution | Percentage |\n|---|---|---|\n"
        
        decomp_json = []
        for d in decomp_list:
            content += f"| {d.component} | {d.total_contribution:,.0f} | {d.pct_of_total:.1%} |\n"
            decomp_json.append({
                "component": d.component,
                "total_contribution": d.total_contribution,
                "pct_of_total": d.pct_of_total
            })
            
        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["decomposition"] = decomp_json
            
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"Error computing decomposition: {str(e)}", tool_call_id=tool_call_id)]})


@tool
def get_model_diagnostics(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Get MCMC convergence diagnostics for the fitted Bayesian model.
    Call this tool when the user asks about model convergence, divergences, R-hat, effective sample size, or diagnostic health.
    """
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(content="No fitted model found in state. Please fit the model first.", tool_call_id=tool_call_id)]})
        
    try:
        diag = _get_diagnostics(mmm)
        
        if not diag:
            return Command(update={"messages": [ToolMessage(content="Diagnostics could not be extracted. Make sure ArviZ is installed and the model sampled correctly.", tool_call_id=tool_call_id)]})
            
        content = "### Model Diagnostics\n\n"
        content += f"**Converged:** {'✅ Yes' if diag.get('converged') else '⚠️ No'}\n"
        content += f"**Divergences:** {diag.get('divergences', 0)} (Should be 0)\n"
        content += f"**Max R-hat:** {diag.get('rhat_max', 'N/A')} (Should be < 1.01)\n"
        content += f"**Min Bulk ESS:** {diag.get('ess_bulk_min', 'N/A')} (Should be > 400)\n"
        content += f"**Min Tail ESS:** {diag.get('ess_tail_min', 'N/A')} (Should be > 400)\n"
            
        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["diagnostics"] = diag
            
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"Error computing diagnostics: {str(e)}", tool_call_id=tool_call_id)]})


@tool
def get_adstock_weights(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Get the learned adstock (carryover effect) weights for each media channel.
    Call this tool when the user asks about how long media effects last, decay rates, half-life, or carryover.
    """
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(content="No fitted model found in state. Please fit the model first.", tool_call_id=tool_call_id)]})
        
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
                "alpha_mean": result.alpha_mean
            }
            
        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["adstock"] = adstock_json
            
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"Error computing adstock weights: {str(e)}", tool_call_id=tool_call_id)]})


@tool
def get_saturation_curves(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Get the saturation parameters (diminishing returns) for each media channel.
    Call this tool when the user asks about diminishing returns, saturation, scaling, or which channel to invest more in.
    """
    mmm = _MODEL_CACHE.get("fitted_model")
    if mmm is None:
        return Command(update={"messages": [ToolMessage(content="No fitted model found in state. Please fit the model first.", tool_call_id=tool_call_id)]})
        
    try:
        curves = compute_saturation_curves_with_uncertainty(mmm)
        
        content = "### Saturation (Diminishing Returns) Analysis\n\n"
        content += "| Channel | Current Saturation Level | Marginal Response (Next $1) |\n|---|---|---|\n"
        
        saturation_json = {}
        for ch, curve in curves.items():
            sat_pct = curve.saturation_level
            # Determine if highly saturated
            status = "🔴 High" if sat_pct > 0.8 else "🟡 Medium" if sat_pct > 0.5 else "🟢 Low"
            content += f"| {ch} | {sat_pct:.1%} ({status}) | {curve.marginal_response_at_current:.3f} |\n"
            saturation_json[ch] = {
                "saturation_level": sat_pct,
                "marginal_response_at_current": curve.marginal_response_at_current,
                "status": status.split(" ")[1]
            }
            
        dashboard_data = state.get("dashboard_data", {})
        if dashboard_data is None:
            dashboard_data = {}
        dashboard_data["saturation"] = saturation_json
            
        return Command(
            update={
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
                "dashboard_data": dashboard_data
            }
        )
    except Exception as e:
        return Command(update={"messages": [ToolMessage(content=f"Error computing saturation: {str(e)}", tool_call_id=tool_call_id)]})


@tool
def execute_python(
    state: Annotated[dict, InjectedState],
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = None
) -> Command:
    """
    Execute Python code in a stateful shell environment. 
    Use this to perform ad-hoc data analysis, examine datasets, or use the mmm_framework directly.
    You have access to the pandas library (as pd) and numpy (as np).
    If a model has been fitted, you also have access to `mmm` (the BayesianMMM object) and `results` (the MMMResults object).
    Always print() the output you want to see.
    """
    import pandas as pd
    import numpy as np
    
    # Force non-interactive backend so matplotlib works in server threads
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
    env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
    }
    
    # Also pre-import plotly so the agent can use it easily
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        env["px"] = px
        env["go"] = go
    except ImportError:
        pass
    
    mmm = _MODEL_CACHE.get("fitted_model")
    results = _MODEL_CACHE.get("fit_results")
    if mmm is not None:
        env["mmm"] = mmm
    if results is not None:
        env["results"] = results
        
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
            fig = fig_or_self
            captured_plots.append(json.loads(fig.to_json()))
            
        pio.show = custom_show
        pbd.BaseFigure.show = custom_show
    except ImportError:
        pass
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, env)
        output = stdout_capture.getvalue()
        if not output:
            output = "Code executed successfully with no output."
    except Exception as e:
        output = f"Error executing code:\n{traceback.format_exc()}"
    finally:
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
        # Append to existing plots list rather than replacing
        existing_plots = dashboard_data.get("plots", [])
        dashboard_data["plots"] = existing_plots + captured_plots
        
    content = f"### Python Execution Result\n```text\n{output}\n```"
    if captured_plots:
        content += f"\n\n*Generated {len(captured_plots)} Plotly interactive chart(s). View them in the Dashboard pane.*"
        
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            "dashboard_data": dashboard_data
        }
    )


# List of all tools
TOOLS = [
    generate_synthetic_data, 
    configure_model, 
    fit_mmm_model, 
    get_roi_metrics, 
    get_component_decomposition, 
    get_model_diagnostics,
    get_adstock_weights,
    get_saturation_curves,
    execute_python
]

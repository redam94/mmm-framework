"""
Chart generation functions for MMM reports.

All charts use Plotly for interactive visualization and can be embedded
in portable HTML reports.
"""

from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd

from .config import ChartConfig, ColorScheme, ChannelColors, ReportConfig


def _to_json(data: Any) -> str:
    """Convert data to JSON string for Plotly."""
    return json.dumps(data, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


def create_plotly_div(
    traces: list[dict],
    layout: dict,
    div_id: str,
    config: dict | None = None
) -> str:
    """Create an HTML div with embedded Plotly chart."""
    config = config or {"displayModeBar": False, "responsive": True}
    
    return f'''
    <div id="{div_id}" class="chart-container"></div>
    <script>
        Plotly.newPlot(
            "{div_id}",
            {_to_json(traces)},
            {_to_json(layout)},
            {_to_json(config)}
        );
    </script>
    '''


def create_model_fit_chart(
    dates: list | np.ndarray | pd.DatetimeIndex,
    actual: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_lower: np.ndarray,
    predicted_upper: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "modelFitChart",
) -> str:
    """
    Create model fit visualization showing actual vs predicted with uncertainty.
    
    Parameters
    ----------
    dates : array-like
        Time index for observations
    actual : ndarray
        Observed KPI values
    predicted_mean : ndarray
        Posterior mean predictions
    predicted_lower : ndarray
        Lower bound of credible interval
    predicted_upper : ndarray
        Upper bound of credible interval
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID for the chart
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(
        height=400,
        y_title="Revenue",
        show_credible_intervals=True,
    )
    
    colors = config.color_scheme
    
    # Convert dates to strings for JSON serialization
    date_strings = [str(d) for d in dates]
    
    traces = [
        # Credible interval band
        {
            "type": "scatter",
            "x": date_strings + date_strings[::-1],
            "y": list(predicted_upper) + list(predicted_lower[::-1]),
            "fill": "toself",
            "fillcolor": f"rgba({_hex_to_rgb(colors.accent)}, {chart_config.ci_alpha})",
            "line": {"color": "transparent"},
            "name": f"{int(chart_config.ci_level * 100)}% CI",
            "hoverinfo": "skip",
        },
        # Predicted mean line
        {
            "type": "scatter",
            "x": date_strings,
            "y": list(predicted_mean),
            "mode": "lines",
            "name": "Predicted",
            "line": {"color": colors.accent, "width": 2},
            "hovertemplate": "Predicted: %{y:,.0f}<extra></extra>",
        },
        # Actual values
        {
            "type": "scatter",
            "x": date_strings,
            "y": list(actual),
            "mode": "markers",
            "name": "Actual",
            "marker": {"color": colors.primary_dark, "size": 6},
            "hovertemplate": "Actual: %{y:,.0f}<extra></extra>",
        },
    ]
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Model Fit: Actual vs Predicted", "font": {"size": 16}}
    
    return create_plotly_div(traces, layout, div_id)


def create_roi_forest_plot(
    channels: list[str],
    roi_mean: np.ndarray,
    roi_lower: np.ndarray,
    roi_upper: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "roiForestPlot",
    reference_line: float = 1.0,
) -> str:
    """
    Create forest plot showing channel ROI with credible intervals.
    
    Parameters
    ----------
    channels : list[str]
        Channel names
    roi_mean : ndarray
        Posterior mean ROI for each channel
    roi_lower : ndarray
        Lower bound of credible interval
    roi_upper : ndarray
        Upper bound of credible interval
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
    reference_line : float
        Value for vertical reference line (default 1.0 for break-even)
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(
        height=max(250, 50 * len(channels)),
        x_title="ROI",
    )
    
    colors = config.color_scheme
    channel_colors = config.channel_colors
    
    # Sort by mean ROI descending
    sort_idx = np.argsort(roi_mean)[::-1]
    channels_sorted = [channels[i] for i in sort_idx]
    roi_mean_sorted = roi_mean[sort_idx]
    roi_lower_sorted = roi_lower[sort_idx]
    roi_upper_sorted = roi_upper[sort_idx]
    
    # Create colors list
    marker_colors = [channel_colors.get(ch) for ch in channels_sorted]
    
    traces = [
        # Error bars
        {
            "type": "scatter",
            "x": list(roi_mean_sorted),
            "y": channels_sorted,
            "error_x": {
                "type": "data",
                "symmetric": False,
                "array": list(roi_upper_sorted - roi_mean_sorted),
                "arrayminus": list(roi_mean_sorted - roi_lower_sorted),
                "color": colors.text_muted,
                "thickness": 2,
                "width": 8,
            },
            "mode": "markers",
            "marker": {"color": marker_colors, "size": 14},
            "name": "ROI",
            "hovertemplate": "%{y}<br>ROI: %{x:.2f}<br>CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>",
            "customdata": [[l, u] for l, u in zip(roi_lower_sorted, roi_upper_sorted)],
        },
    ]
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Channel ROI with Uncertainty", "font": {"size": 16}}
    layout["yaxis"]["autorange"] = "reversed"  # Highest ROI at top
    
    # Add reference line at break-even
    layout["shapes"] = [
        {
            "type": "line",
            "x0": reference_line,
            "x1": reference_line,
            "y0": -0.5,
            "y1": len(channels) - 0.5,
            "line": {"color": colors.text_muted, "width": 1, "dash": "dash"},
        }
    ]
    layout["annotations"] = [
        {
            "x": reference_line,
            "y": -0.3,
            "xref": "x",
            "yref": "y",
            "text": "Break-even",
            "showarrow": False,
            "font": {"size": 10, "color": colors.text_muted},
        }
    ]
    
    return create_plotly_div(traces, layout, div_id)


def create_waterfall_chart(
    categories: list[str],
    values: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "waterfallChart",
    total_label: str = "Total",
) -> str:
    """
    Create waterfall chart for revenue decomposition.
    
    Parameters
    ----------
    categories : list[str]
        Component names (e.g., ["Baseline", "TV", "Search", ...])
    values : ndarray
        Contribution values for each component
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
    total_label : str
        Label for the total bar
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(height=400, y_title="Revenue")
    
    colors = config.color_scheme
    channel_colors = config.channel_colors
    
    # Waterfall uses measure to distinguish increase/decrease/total
    measures = ["relative"] * len(categories) + ["total"]
    x_labels = list(categories) + [total_label]
    y_values = list(values) + [sum(values)]
    
    # Colors based on component type
    marker_colors = []
    for cat in categories:
        if cat.lower() == "baseline":
            marker_colors.append(colors.text_muted)
        else:
            marker_colors.append(channel_colors.get(cat))
    marker_colors.append(colors.primary_dark)  # Total
    
    traces = [
        {
            "type": "waterfall",
            "x": x_labels,
            "y": y_values,
            "measure": measures,
            "connector": {"line": {"color": colors.border}},
            "increasing": {"marker": {"color": colors.success}},
            "decreasing": {"marker": {"color": colors.danger}},
            "totals": {"marker": {"color": colors.primary_dark}},
            "texttemplate": "%{y:,.0f}",
            "textposition": "outside",
            "hovertemplate": "%{x}<br>%{y:,.0f}<extra></extra>",
        }
    ]
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Revenue Decomposition", "font": {"size": 16}}
    layout["showlegend"] = False
    
    return create_plotly_div(traces, layout, div_id)


def create_decomposition_chart(
    dates: list | np.ndarray | pd.DatetimeIndex,
    components: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "decompositionChart",
    chart_type: str = "stacked_area",
) -> str:
    """
    Create time series decomposition visualization.
    
    Parameters
    ----------
    dates : array-like
        Time index
    components : dict
        Mapping of component name to time series values
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
    chart_type : str
        Either "stacked_area" or "stacked_bar"
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(height=400, y_title="Revenue")
    
    colors = config.color_scheme
    channel_colors = config.channel_colors
    
    date_strings = [str(d) for d in dates]
    
    traces = []
    for name, values in components.items():
        color = channel_colors.get(name)
        trace = {
            "x": date_strings,
            "y": list(values),
            "name": name,
            "hovertemplate": f"{name}: %{{y:,.0f}}<extra></extra>",
        }
        
        if chart_type == "stacked_area":
            trace["type"] = "scatter"
            trace["mode"] = "lines"
            trace["stackgroup"] = "one"
            trace["fillcolor"] = color
            trace["line"] = {"color": color, "width": 0.5}
        else:
            trace["type"] = "bar"
            trace["marker"] = {"color": color}
        
        traces.append(trace)
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Revenue Components Over Time", "font": {"size": 16}}
    
    if chart_type == "stacked_bar":
        layout["barmode"] = "stack"
    
    return create_plotly_div(traces, layout, div_id)


def create_stacked_area_chart(
    dates: list | np.ndarray | pd.DatetimeIndex,
    components: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "stackedAreaChart",
) -> str:
    """Convenience wrapper for stacked area decomposition chart."""
    return create_decomposition_chart(
        dates, components, config, chart_config, div_id, chart_type="stacked_area"
    )


def create_saturation_curves(
    channels: list[str],
    spend_ranges: dict[str, np.ndarray],
    response_curves: dict[str, np.ndarray],
    current_spend: dict[str, float],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "saturationCharts",
    ci_bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> str:
    """
    Create saturation curve visualizations for each channel.
    
    Parameters
    ----------
    channels : list[str]
        Channel names
    spend_ranges : dict
        Mapping of channel to spend value array for x-axis
    response_curves : dict
        Mapping of channel to response curve values
    current_spend : dict
        Mapping of channel to current spend level
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
    ci_bands : dict, optional
        Mapping of channel to (lower, upper) CI arrays
        
    Returns
    -------
    str
        HTML string with embedded Plotly charts in a grid
    """
    chart_config = chart_config or ChartConfig(height=280)
    
    colors = config.color_scheme
    channel_colors = config.channel_colors
    
    # Create a grid of charts
    n_cols = 2
    n_rows = (len(channels) + 1) // 2
    
    html_parts = [f'<div class="chart-grid" style="display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 1.5rem;">']
    
    for i, channel in enumerate(channels):
        sub_div_id = f"{div_id}_{i}"
        spend = spend_ranges.get(channel, np.linspace(0, 1, 100))
        response = response_curves.get(channel, np.zeros_like(spend))
        current = current_spend.get(channel, 0)
        ch_color = channel_colors.get(channel)
        
        traces = []
        
        # Add CI band if available
        if ci_bands and channel in ci_bands:
            lower, upper = ci_bands[channel]
            traces.append({
                "type": "scatter",
                "x": list(spend) + list(spend[::-1]),
                "y": list(upper) + list(lower[::-1]),
                "fill": "toself",
                "fillcolor": f"rgba({_hex_to_rgb(ch_color)}, 0.2)",
                "line": {"color": "transparent"},
                "hoverinfo": "skip",
                "showlegend": False,
            })
        
        # Main response curve
        traces.append({
            "type": "scatter",
            "x": list(spend),
            "y": list(response),
            "mode": "lines",
            "name": channel,
            "line": {"color": ch_color, "width": 2.5},
            "hovertemplate": f"Spend: {config.currency_symbol}%{{x:,.0f}}<br>Response: %{{y:,.0f}}<extra></extra>",
        })
        
        # Current spend marker
        if current > 0:
            current_response = np.interp(current, spend, response)
            traces.append({
                "type": "scatter",
                "x": [current],
                "y": [current_response],
                "mode": "markers",
                "name": "Current",
                "marker": {"color": colors.warning, "size": 10, "symbol": "diamond"},
                "hovertemplate": f"Current Spend: {config.currency_symbol}%{{x:,.0f}}<br>Response: %{{y:,.0f}}<extra></extra>",
            })
        
        layout = {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 11},
            "margin": {"t": 35, "r": 15, "b": 45, "l": 55},
            "height": chart_config.height,
            "title": {"text": channel, "font": {"size": 14}},
            "showlegend": False,
            "xaxis": {
                "title": "Spend",
                "gridcolor": colors.border,
                "tickformat": ",.0s",
            },
            "yaxis": {
                "title": "Response",
                "gridcolor": colors.border,
                "tickformat": ",.0s",
            },
        }
        
        html_parts.append(f'<div class="chart-box">')
        html_parts.append(create_plotly_div(traces, layout, sub_div_id))
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def create_adstock_chart(
    channels: list[str],
    lag_weights: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "adstockChart",
) -> str:
    """
    Create adstock/carryover decay visualization.
    
    Parameters
    ----------
    channels : list[str]
        Channel names
    lag_weights : dict
        Mapping of channel to decay weight arrays
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(
        height=350,
        x_title="Weeks Since Exposure",
        y_title="Effect Weight",
    )
    
    colors = config.color_scheme
    channel_colors = config.channel_colors
    
    traces = []
    for channel in channels:
        weights = lag_weights.get(channel, np.array([1.0]))
        ch_color = channel_colors.get(channel)
        
        traces.append({
            "type": "scatter",
            "x": list(range(len(weights))),
            "y": list(weights),
            "mode": "lines+markers",
            "name": channel,
            "line": {"color": ch_color, "width": 2},
            "marker": {"color": ch_color, "size": 6},
            "hovertemplate": f"{channel}<br>Lag %{{x}}: %{{y:.3f}}<extra></extra>",
        })
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Adstock Decay Curves", "font": {"size": 16}}
    
    return create_plotly_div(traces, layout, div_id)


def create_prior_posterior_chart(
    parameter_names: list[str],
    prior_samples: dict[str, np.ndarray],
    posterior_samples: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorPosteriorChart",
) -> str:
    """
    Create prior vs posterior comparison visualization.
    
    Parameters
    ----------
    parameter_names : list[str]
        Names of parameters to plot
    prior_samples : dict
        Mapping of parameter name to prior samples
    posterior_samples : dict
        Mapping of parameter name to posterior samples
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
        
    Returns
    -------
    str
        HTML string with embedded Plotly charts
    """
    chart_config = chart_config or ChartConfig(height=280)
    
    colors = config.color_scheme
    
    n_cols = min(3, len(parameter_names))
    
    html_parts = [f'<div class="chart-grid" style="display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 1.5rem;">']
    
    for i, param in enumerate(parameter_names):
        sub_div_id = f"{div_id}_{i}"
        
        prior = prior_samples.get(param, np.array([]))
        posterior = posterior_samples.get(param, np.array([]))
        
        traces = []
        
        if len(prior) > 0:
            traces.append({
                "type": "histogram",
                "x": list(prior),
                "name": "Prior",
                "opacity": 0.5,
                "marker": {"color": colors.text_muted},
                "histnorm": "probability density",
                "nbinsx": 50,
            })
        
        if len(posterior) > 0:
            traces.append({
                "type": "histogram",
                "x": list(posterior),
                "name": "Posterior",
                "opacity": 0.7,
                "marker": {"color": colors.primary},
                "histnorm": "probability density",
                "nbinsx": 50,
            })
        
        layout = {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 11},
            "margin": {"t": 35, "r": 15, "b": 45, "l": 55},
            "height": chart_config.height,
            "title": {"text": param, "font": {"size": 14}},
            "barmode": "overlay",
            "showlegend": i == 0,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
            "xaxis": {"gridcolor": colors.border},
            "yaxis": {"gridcolor": colors.border, "title": "Density"},
        }
        
        html_parts.append('<div class="chart-box">')
        html_parts.append(create_plotly_div(traces, layout, sub_div_id))
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def create_trace_plot(
    parameter_names: list[str],
    traces_data: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "tracePlot",
    n_chains: int = 4,
) -> str:
    """
    Create MCMC trace plots for diagnostics.
    
    Parameters
    ----------
    parameter_names : list[str]
        Parameters to visualize
    traces_data : dict
        Mapping of parameter name to samples array (chains x draws)
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
    n_chains : int
        Number of MCMC chains
        
    Returns
    -------
    str
        HTML string with embedded Plotly charts
    """
    chart_config = chart_config or ChartConfig(height=200)
    
    colors = config.color_scheme
    chain_colors = ["#6a8fa8", "#8fa86a", "#a88f6a", "#8f6aa8"]
    
    html_parts = ['<div style="display: flex; flex-direction: column; gap: 1rem;">']
    
    for i, param in enumerate(parameter_names):
        sub_div_id = f"{div_id}_{i}"
        
        data = traces_data.get(param, np.array([]))
        
        traces = []
        if len(data.shape) == 2:
            # Shape is (chains, draws)
            for chain_idx in range(min(n_chains, data.shape[0])):
                traces.append({
                    "type": "scatter",
                    "y": list(data[chain_idx]),
                    "mode": "lines",
                    "name": f"Chain {chain_idx + 1}",
                    "line": {"color": chain_colors[chain_idx % len(chain_colors)], "width": 0.5},
                    "opacity": 0.7,
                })
        elif len(data.shape) == 1:
            traces.append({
                "type": "scatter",
                "y": list(data),
                "mode": "lines",
                "name": "Samples",
                "line": {"color": colors.primary, "width": 0.5},
            })
        
        layout = {
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 11},
            "margin": {"t": 30, "r": 15, "b": 30, "l": 55},
            "height": chart_config.height,
            "title": {"text": param, "font": {"size": 13}},
            "showlegend": i == 0,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
            "xaxis": {"title": "Iteration", "gridcolor": colors.border},
            "yaxis": {"gridcolor": colors.border},
        }
        
        html_parts.append(create_plotly_div(traces, layout, sub_div_id))
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def create_sensitivity_chart(
    scenarios: list[str],
    base_values: np.ndarray,
    alternative_values: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "sensitivityChart",
) -> str:
    """
    Create sensitivity analysis comparison chart.
    
    Parameters
    ----------
    scenarios : list[str]
        Names of sensitivity scenarios
    base_values : ndarray
        Values from base model specification
    alternative_values : dict
        Mapping of scenario name to alternative values
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID
        
    Returns
    -------
    str
        HTML string with embedded Plotly chart
    """
    chart_config = chart_config or ChartConfig(height=400)
    
    colors = config.color_scheme
    
    traces = []
    
    # Base model
    traces.append({
        "type": "bar",
        "x": scenarios,
        "y": list(base_values),
        "name": "Base Model",
        "marker": {"color": colors.primary},
    })
    
    # Alternative specifications
    alt_colors = [colors.accent, colors.warning, colors.danger]
    for i, (name, values) in enumerate(alternative_values.items()):
        traces.append({
            "type": "bar",
            "x": scenarios,
            "y": list(values),
            "name": name,
            "marker": {"color": alt_colors[i % len(alt_colors)]},
        })
    
    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Sensitivity Analysis", "font": {"size": 16}}
    layout["barmode"] = "group"
    
    return create_plotly_div(traces, layout, div_id)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to RGB string for Plotly rgba()."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"{r}, {g}, {b}"
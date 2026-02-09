"""
Diagnostic chart functions for MMM reporting.

Contains saturation curves, adstock decay, prior/posterior comparison,
trace plots, and sensitivity analysis charts.
"""

from __future__ import annotations

import numpy as np

from ..config import ChartConfig, ReportConfig
from .base import _hex_to_rgb, create_plotly_div


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

    html_parts = [
        f'<div class="chart-grid" style="display: grid; '
        f'grid-template-columns: repeat({n_cols}, 1fr); gap: 1.5rem;">'
    ]

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
            traces.append(
                {
                    "type": "scatter",
                    "x": list(spend) + list(spend[::-1]),
                    "y": list(upper) + list(lower[::-1]),
                    "fill": "toself",
                    "fillcolor": f"rgba({_hex_to_rgb(ch_color)}, 0.2)",
                    "line": {"color": "transparent"},
                    "hoverinfo": "skip",
                    "showlegend": False,
                }
            )

        # Main response curve
        traces.append(
            {
                "type": "scatter",
                "x": list(spend),
                "y": list(response),
                "mode": "lines",
                "name": channel,
                "line": {"color": ch_color, "width": 2.5},
                "hovertemplate": (
                    f"Spend: {config.currency_symbol}%{{x:,.0f}}<br>"
                    f"Response: %{{y:,.0f}}<extra></extra>"
                ),
            }
        )

        # Current spend marker
        if current > 0:
            current_response = np.interp(current, spend, response)
            traces.append(
                {
                    "type": "scatter",
                    "x": [current],
                    "y": [current_response],
                    "mode": "markers",
                    "name": "Current",
                    "marker": {
                        "color": colors.warning,
                        "size": 10,
                        "symbol": "diamond",
                    },
                    "hovertemplate": (
                        f"Current Spend: {config.currency_symbol}%{{x:,.0f}}<br>"
                        f"Response: %{{y:,.0f}}<extra></extra>"
                    ),
                }
            )

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

        html_parts.append('<div class="chart-box">')
        html_parts.append(create_plotly_div(traces, layout, sub_div_id))
        html_parts.append("</div>")

    html_parts.append("</div>")
    return "\n".join(html_parts)


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

        traces.append(
            {
                "type": "scatter",
                "x": list(range(len(weights))),
                "y": list(weights),
                "mode": "lines+markers",
                "name": channel,
                "line": {"color": ch_color, "width": 2},
                "marker": {"color": ch_color, "size": 6},
                "hovertemplate": f"{channel}<br>Lag %{{x}}: %{{y:.3f}}<extra></extra>",
            }
        )

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

    html_parts = [
        f'<div class="chart-grid" style="display: grid; '
        f'grid-template-columns: repeat({n_cols}, 1fr); gap: 1.5rem;">'
    ]

    for i, param in enumerate(parameter_names):
        sub_div_id = f"{div_id}_{i}"

        prior = prior_samples.get(param, np.array([]))
        posterior = posterior_samples.get(param, np.array([]))

        traces = []

        if len(prior) > 0:
            traces.append(
                {
                    "type": "histogram",
                    "x": list(prior),
                    "name": "Prior",
                    "opacity": 0.5,
                    "marker": {"color": colors.text_muted},
                    "histnorm": "probability density",
                    "nbinsx": 50,
                }
            )

        if len(posterior) > 0:
            traces.append(
                {
                    "type": "histogram",
                    "x": list(posterior),
                    "name": "Posterior",
                    "opacity": 0.7,
                    "marker": {"color": colors.primary},
                    "histnorm": "probability density",
                    "nbinsx": 50,
                }
            )

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
        html_parts.append("</div>")

    html_parts.append("</div>")
    return "\n".join(html_parts)


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
                traces.append(
                    {
                        "type": "scatter",
                        "y": list(data[chain_idx]),
                        "mode": "lines",
                        "name": f"Chain {chain_idx + 1}",
                        "line": {
                            "color": chain_colors[chain_idx % len(chain_colors)],
                            "width": 0.5,
                        },
                        "opacity": 0.7,
                    }
                )
        elif len(data.shape) == 1:
            traces.append(
                {
                    "type": "scatter",
                    "y": list(data),
                    "mode": "lines",
                    "name": "Samples",
                    "line": {"color": colors.primary, "width": 0.5},
                }
            )

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

    html_parts.append("</div>")
    return "\n".join(html_parts)


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
    traces.append(
        {
            "type": "bar",
            "x": scenarios,
            "y": list(base_values),
            "name": "Base Model",
            "marker": {"color": colors.primary},
        }
    )

    # Alternative specifications
    alt_colors = [colors.accent, colors.warning, colors.danger]
    for i, (name, values) in enumerate(alternative_values.items()):
        traces.append(
            {
                "type": "bar",
                "x": scenarios,
                "y": list(values),
                "name": name,
                "marker": {"color": alt_colors[i % len(alt_colors)]},
            }
        )

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Sensitivity Analysis", "font": {"size": 16}}
    layout["barmode"] = "group"

    return create_plotly_div(traces, layout, div_id)


__all__ = [
    "create_saturation_curves",
    "create_adstock_chart",
    "create_prior_posterior_chart",
    "create_trace_plot",
    "create_sensitivity_chart",
]

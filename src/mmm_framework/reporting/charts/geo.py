"""
Geographic chart functions for MMM reporting.

Contains heatmaps and charts for geographic analysis and comparison.
"""

from __future__ import annotations

from ..config import ChartConfig, ReportConfig
from .base import create_plotly_div


def create_geo_roi_heatmap(
    geo_names: list[str],
    channel_names: list[str],
    geo_roi: dict[str, dict[str, dict[str, float]]],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "geoRoiHeatmap",
) -> str:
    """
    Create geographic ROI heatmap showing channel performance across regions.

    Parameters
    ----------
    geo_names : list[str]
        Names of geographies (y-axis)
    channel_names : list[str]
        Names of channels (x-axis)
    geo_roi : dict
        Nested dict: {geo: {channel: {"mean", "lower", "upper"}}}
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID

    Returns
    -------
    str
        HTML string with embedded Plotly heatmap
    """
    chart_config = chart_config or ChartConfig(height=400)
    colors = config.color_scheme

    # Build matrix
    z_values = []
    hover_text = []

    for geo in geo_names:
        row_vals = []
        row_text = []
        for ch in channel_names:
            roi_data = geo_roi.get(geo, {}).get(ch, {})
            mean = roi_data.get("mean", 0)
            lower = roi_data.get("lower", 0)
            upper = roi_data.get("upper", 0)
            row_vals.append(mean)
            row_text.append(
                f"{geo} × {ch}<br>ROI: {mean:.2f}x<br>CI: [{lower:.2f}, {upper:.2f}]"
            )
        z_values.append(row_vals)
        hover_text.append(row_text)

    traces = [
        {
            "type": "heatmap",
            "x": channel_names,
            "y": geo_names,
            "z": z_values,
            "text": hover_text,
            "hoverinfo": "text",
            "colorscale": [
                [0, "#c97067"],  # Low ROI (danger)
                [0.5, "#fafbf9"],  # Neutral
                [1, colors.primary],  # High ROI
            ],
            "colorbar": {
                "title": "ROI",
                "thickness": 15,
            },
            "zmin": 0,
            "zmid": 1.0,
        }
    ]

    layout = chart_config.to_plotly_layout(colors)
    layout["xaxis"] = {"title": "Channel", "side": "top"}
    layout["yaxis"] = {"title": "Geography", "autorange": "reversed"}

    return create_plotly_div(traces, layout, div_id)


def create_geo_decomposition_chart(
    geo_names: list[str],
    geo_contribution: dict[str, dict[str, float]],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "geoDecompChart",
) -> str:
    """
    Create stacked bar chart showing contribution decomposition by geography.

    Parameters
    ----------
    geo_names : list[str]
        Names of geographies
    geo_contribution : dict
        Nested dict: {geo: {component: contribution}}
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
    channel_colors = config.channel_colors

    # Get all components across geos
    all_components = set()
    for geo_data in geo_contribution.values():
        all_components.update(geo_data.keys())
    all_components = sorted(all_components)

    traces = []
    for comp in all_components:
        values = [geo_contribution.get(geo, {}).get(comp, 0) for geo in geo_names]
        traces.append(
            {
                "type": "bar",
                "x": geo_names,
                "y": values,
                "name": comp,
                "marker": {"color": channel_colors.get(comp)},
            }
        )

    layout = chart_config.to_plotly_layout(colors)
    layout["barmode"] = "stack"
    layout["xaxis"] = {"title": "Geography"}
    layout["yaxis"] = {"title": "Contribution", "tickformat": "$,.0f"}
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "center",
        "x": 0.5,
    }

    return create_plotly_div(traces, layout, div_id)


__all__ = [
    "create_geo_roi_heatmap",
    "create_geo_decomposition_chart",
]

"""
Decomposition chart functions for MMM reporting.

Contains waterfall, stacked area, and time series decomposition charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ChartConfig, ColorScheme, ReportConfig
from .base import (
    _dates_to_strings,
    _hex_to_rgb,
    create_plotly_div,
)


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


def create_stacked_area_chart_with_geo_selector(
    dates: list | np.ndarray | pd.DatetimeIndex,
    components_agg: dict[str, np.ndarray],  # {component_name: time_series}
    components_by_geo: dict[str, dict[str, np.ndarray]] | None = None,
    geo_names: list[str] | None = None,
    config: ReportConfig = None,
    chart_config: ChartConfig | None = None,
    div_id: str = "decompositionStackedArea",
) -> str:
    """
    Create stacked area decomposition chart with geo selector dropdown.

    Parameters
    ----------
    dates : array-like
        Time index for x-axis
    components_agg : dict
        Aggregated component time series: {component_name: ndarray}
    components_by_geo : dict, optional
        Per-geo components: {geo_name: {component_name: ndarray}}
    geo_names : list, optional
        List of geography names
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID for the chart

    Returns
    -------
    str
        HTML string with embedded Plotly chart and dropdown
    """
    chart_config = chart_config or ChartConfig(height=450)
    colors = config.color_scheme if config else ColorScheme()
    channel_colors = config.channel_colors if config else {}

    # Convert dates to string
    dates_str = _dates_to_strings(dates)

    # Component ordering and colors
    component_names = list(components_agg.keys())
    n_components = len(component_names)

    # Default color palette for components
    default_colors = [
        "#5A6B5A",  # Baseline - muted green
        "#8FA86A",  # Trend - sage green
        "#C9A227",  # Seasonality - gold
        "#4285F4",  # Media 1 - blue
        "#EA4335",  # Media 2 - red
        "#FBBC04",  # Media 3 - yellow
        "#34A853",  # Media 4 - green
        "#FF6D01",  # Media 5 - orange
        "#9334E6",  # Control 1 - purple
        "#E91E63",  # Control 2 - pink
    ]

    def get_component_color(comp_name: str, idx: int) -> str:
        """Get color for a component."""
        if channel_colors is not None:
            if hasattr(channel_colors, "get"):
                # ChannelColors dataclass with .get() method
                color = channel_colors.get(comp_name)
                if color:
                    return color
            elif isinstance(channel_colors, dict) and comp_name in channel_colors:
                return channel_colors[comp_name]
        if comp_name == "Baseline":
            return default_colors[0]
        if comp_name == "Trend":
            return default_colors[1]
        if comp_name == "Seasonality":
            return default_colors[2]
        return default_colors[min(idx, len(default_colors) - 1)]

    traces = []

    # =========================================================================
    # AGGREGATED TRACES (visible by default)
    # =========================================================================

    for i, comp_name in enumerate(component_names):
        comp_values = components_agg[comp_name]
        comp_color = get_component_color(comp_name, i)

        traces.append(
            {
                "type": "scatter",
                "x": dates_str,
                "y": list(comp_values),
                "mode": "lines",
                "name": comp_name,
                "stackgroup": "agg",
                "fillcolor": comp_color,
                "line": {"width": 0.5, "color": comp_color},
                "hovertemplate": f"{comp_name}: %{{y:,.0f}}<extra></extra>",
                "visible": True,
            }
        )

    n_agg_traces = n_components

    # =========================================================================
    # GEO-LEVEL TRACES (hidden by default)
    # =========================================================================

    has_geo = (
        geo_names is not None and len(geo_names) > 1 and components_by_geo is not None
    )

    n_geos = len(geo_names) if has_geo else 0
    n_geo_traces_per_geo = n_components

    if has_geo:
        for geo in geo_names:
            geo_components = components_by_geo.get(geo, {})

            for i, comp_name in enumerate(component_names):
                comp_values = geo_components.get(comp_name, np.zeros(len(dates_str)))
                comp_color = get_component_color(comp_name, i)

                traces.append(
                    {
                        "type": "scatter",
                        "x": dates_str,
                        "y": list(comp_values),
                        "mode": "lines",
                        "name": comp_name,
                        "stackgroup": f"geo_{geo}",
                        "fillcolor": comp_color,
                        "line": {"width": 0.5, "color": comp_color},
                        "hovertemplate": f"{comp_name}: %{{y:,.0f}}<extra></extra>",
                        "visible": False,
                    }
                )

    # =========================================================================
    # BUILD DROPDOWN MENU
    # =========================================================================

    buttons = []

    # Button: "Aggregated (Total)"
    visible_agg = [True] * n_agg_traces + [False] * (n_geo_traces_per_geo * n_geos)
    buttons.append(
        {
            "label": "Aggregated (Total)",
            "method": "update",
            "args": [
                {"visible": visible_agg},
                {"title": {"text": "Revenue Decomposition: Aggregated (Total)"}},
            ],
        }
    )

    # Buttons for each geo
    if has_geo:
        for i, geo in enumerate(geo_names):
            visible_geo = [False] * n_agg_traces
            for j in range(n_geos):
                visible_geo.extend([j == i] * n_geo_traces_per_geo)

            buttons.append(
                {
                    "label": geo,
                    "method": "update",
                    "args": [
                        {"visible": visible_geo},
                        {"title": {"text": f"Revenue Decomposition: {geo}"}},
                    ],
                }
            )

    # =========================================================================
    # LAYOUT
    # =========================================================================

    layout = {
        "title": {
            "text": "Revenue Decomposition: Aggregated (Total)",
            "font": {"size": 16},
        },
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 12},
        "margin": {"t": 80, "r": 30, "b": 60, "l": 70},
        "height": chart_config.height,
        "xaxis": {
            "title": "Period",
            "gridcolor": colors.border,
            "showgrid": True,
        },
        "yaxis": {
            "title": "Revenue Contribution",
            "gridcolor": colors.border,
            "showgrid": True,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
        "hovermode": "x unified",
    }

    # Add dropdown if we have geo data
    if has_geo and len(buttons) > 1:
        layout["updatemenus"] = [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": colors.surface,
                "bordercolor": colors.border,
                "borderwidth": 1,
                "font": {"size": 11},
            }
        ]

    return create_plotly_div(traces, layout, div_id)


def create_waterfall_chart_with_geo_selector(
    component_totals_agg: dict[str, float],
    component_totals_by_geo: dict[str, dict[str, float]] | None = None,
    geo_names: list[str] | None = None,
    config: ReportConfig = None,
    chart_config: ChartConfig | None = None,
    div_id: str = "decompositionWaterfall",
) -> str:
    """
    Create waterfall chart for contribution breakdown with geo selector.

    Parameters
    ----------
    component_totals_agg : dict
        Aggregated component totals: {component_name: total_contribution}
    component_totals_by_geo : dict, optional
        Per-geo totals: {geo_name: {component_name: total}}
    geo_names : list, optional
        List of geography names
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID

    Returns
    -------
    str
        HTML string with embedded Plotly waterfall chart
    """
    chart_config = chart_config or ChartConfig(height=400)
    colors = config.color_scheme if config else ColorScheme()

    def create_waterfall_trace(
        totals: dict[str, float],
        name: str,
        visible: bool = True,
    ) -> dict:
        """Create a single waterfall trace."""
        component_names = list(totals.keys())
        values = list(totals.values())

        # Waterfall measure types
        measures = ["relative"] * len(values) + ["total"]
        x_labels = component_names + ["Total"]
        y_values = values + [None]  # None for total (calculated automatically)

        return {
            "type": "waterfall",
            "x": x_labels,
            "y": y_values,
            "measure": measures,
            "name": name,
            "textposition": "outside",
            "text": [f"{v:,.0f}" if v is not None else "" for v in y_values],
            "connector": {"line": {"color": colors.border, "width": 1}},
            "increasing": {"marker": {"color": colors.success}},
            "decreasing": {"marker": {"color": colors.danger}},
            "totals": {"marker": {"color": colors.primary}},
            "visible": visible,
        }

    traces = []

    # Aggregated waterfall (visible)
    traces.append(
        create_waterfall_trace(component_totals_agg, "Aggregated", visible=True)
    )

    # Geo-level waterfalls (hidden)
    has_geo = (
        geo_names is not None
        and len(geo_names) > 1
        and component_totals_by_geo is not None
    )

    n_geos = len(geo_names) if has_geo else 0

    if has_geo:
        for geo in geo_names:
            geo_totals = component_totals_by_geo.get(geo, {})
            traces.append(create_waterfall_trace(geo_totals, geo, visible=False))

    # Build dropdown
    buttons = []

    # Aggregated button
    visible_agg = [True] + [False] * n_geos
    buttons.append(
        {
            "label": "Aggregated (Total)",
            "method": "update",
            "args": [
                {"visible": visible_agg},
                {"title": {"text": "Contribution Breakdown: Aggregated (Total)"}},
            ],
        }
    )

    # Geo buttons
    if has_geo:
        for i, geo in enumerate(geo_names):
            visible_geo = [False] + [j == i for j in range(n_geos)]
            buttons.append(
                {
                    "label": geo,
                    "method": "update",
                    "args": [
                        {"visible": visible_geo},
                        {"title": {"text": f"Contribution Breakdown: {geo}"}},
                    ],
                }
            )

    layout = {
        "title": {
            "text": "Contribution Breakdown: Aggregated (Total)",
            "font": {"size": 16},
        },
        "paper_bgcolor": "transparent",
        "plot_bgcolor": "transparent",
        "font": {"family": "Inter, sans-serif", "color": colors.text, "size": 12},
        "margin": {"t": 80, "r": 30, "b": 100, "l": 70},
        "height": chart_config.height,
        "xaxis": {
            "title": "",
            "tickangle": -45,
        },
        "yaxis": {
            "title": "Contribution",
            "gridcolor": colors.border,
        },
        "showlegend": False,
    }

    if has_geo and len(buttons) > 1:
        layout["updatemenus"] = [
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": colors.surface,
                "bordercolor": colors.border,
                "borderwidth": 1,
                "font": {"size": 11},
            }
        ]

    return create_plotly_div(traces, layout, div_id)


__all__ = [
    "create_decomposition_chart",
    "create_stacked_area_chart",
    "create_waterfall_chart",
    "create_stacked_area_chart_with_geo_selector",
    "create_waterfall_chart_with_geo_selector",
]

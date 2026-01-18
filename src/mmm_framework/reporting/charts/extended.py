"""
Extended model chart functions for MMM reporting.

Contains charts for nested MMM (mediators), multivariate MMM,
and cannibalization analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import ChartConfig, ReportConfig
from .base import _hex_to_rgb, create_plotly_div


def create_mediator_pathway_chart(
    channel_names: list[str],
    mediator_names: list[str],
    mediator_pathways: dict[str, dict[str, Any]],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "mediatorPathwayChart",
) -> str:
    """
    Create Sankey-style diagram showing effect pathways through mediators.

    Parameters
    ----------
    channel_names : list[str]
        Names of media channels
    mediator_names : list[str]
        Names of mediators (e.g., awareness, consideration)
    mediator_pathways : dict
        Pathway data: {channel: {mediator: effect, "_direct": effect, "_total": effect}}
    config : ReportConfig
        Report configuration
    chart_config : ChartConfig, optional
        Chart-specific configuration
    div_id : str
        HTML div ID

    Returns
    -------
    str
        HTML string with embedded Plotly Sankey diagram
    """
    chart_config = chart_config or ChartConfig(height=500)
    colors = config.color_scheme
    channel_colors = config.channel_colors

    # Build Sankey diagram
    # Nodes: channels, mediators, "Sales"
    node_labels = list(channel_names) + list(mediator_names) + ["Sales"]
    node_colors = [channel_colors.get(ch) for ch in channel_names]
    node_colors += [colors.accent for _ in mediator_names]
    node_colors += [colors.primary]

    n_channels = len(channel_names)
    n_mediators = len(mediator_names)

    sources = []
    targets = []
    values = []
    link_colors = []

    for i, ch in enumerate(channel_names):
        pathways = mediator_pathways.get(ch, {})

        # Direct effects: channel -> Sales
        direct = pathways.get("_direct", {})
        direct_val = (
            abs(direct.get("mean", 0)) if isinstance(direct, dict) else abs(direct)
        )
        if direct_val > 0.001:
            sources.append(i)
            targets.append(n_channels + n_mediators)  # Sales node
            values.append(direct_val)
            link_colors.append(
                f"rgba({_hex_to_rgb(channel_colors.get(ch))}, 0.4)"
            )

        # Indirect effects: channel -> mediator -> Sales
        for j, med in enumerate(mediator_names):
            med_effect = pathways.get(med, {})
            effect_val = (
                abs(med_effect.get("mean", 0))
                if isinstance(med_effect, dict)
                else abs(med_effect)
            )
            if effect_val > 0.001:
                # Channel -> Mediator
                sources.append(i)
                targets.append(n_channels + j)
                values.append(effect_val)
                link_colors.append(
                    f"rgba({_hex_to_rgb(channel_colors.get(ch))}, 0.3)"
                )

    # Mediator -> Sales (aggregate)
    for j, med in enumerate(mediator_names):
        total_through_med = 0
        for ch in channel_names:
            pathways = mediator_pathways.get(ch, {})
            med_effect = pathways.get(med, {})
            effect_val = (
                abs(med_effect.get("mean", 0))
                if isinstance(med_effect, dict)
                else abs(med_effect)
            )
            total_through_med += effect_val

        if total_through_med > 0.001:
            sources.append(n_channels + j)
            targets.append(n_channels + n_mediators)
            values.append(total_through_med)
            link_colors.append(f"rgba({_hex_to_rgb(colors.accent)}, 0.4)")

    traces = [
        {
            "type": "sankey",
            "orientation": "h",
            "node": {
                "label": node_labels,
                "color": node_colors,
                "pad": 20,
                "thickness": 20,
                "line": {"color": colors.border, "width": 0.5},
            },
            "link": {
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
            },
        }
    ]

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Effect Pathways: Media → Mediators → Sales",
        "font": {"size": 14},
    }

    return create_plotly_div(traces, layout, div_id)


def create_mediator_time_series(
    dates: np.ndarray | pd.DatetimeIndex | list,
    mediator_names: list[str],
    mediator_time_series: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "mediatorTsChart",
) -> str:
    """
    Create time series chart for mediator values.

    Parameters
    ----------
    dates : array-like
        Date index
    mediator_names : list[str]
        Names of mediators
    mediator_time_series : dict
        Time series data: {mediator: values}
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
    chart_config = chart_config or ChartConfig(height=350)
    colors = config.color_scheme

    # Convert dates
    if isinstance(dates, pd.DatetimeIndex):
        x_vals = dates.strftime("%Y-%m-%d").tolist()
    else:
        x_vals = [str(d) for d in dates]

    mediator_colors = [colors.accent, colors.primary, colors.warning, colors.success]

    traces = []
    for i, med in enumerate(mediator_names):
        ts = mediator_time_series.get(med)
        if ts is not None:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": x_vals,
                    "y": ts.tolist() if hasattr(ts, "tolist") else list(ts),
                    "name": med,
                    "line": {
                        "color": mediator_colors[i % len(mediator_colors)],
                        "width": 2,
                    },
                }
            )

    layout = chart_config.to_plotly_layout(colors)
    layout["xaxis"] = {"title": "Date", "gridcolor": colors.border}
    layout["yaxis"] = {"title": "Value (normalized)", "gridcolor": colors.border}
    layout["legend"] = {"orientation": "h", "yanchor": "bottom", "y": 1.02}
    layout["hovermode"] = "x unified"

    return create_plotly_div(traces, layout, div_id)


def create_cannibalization_heatmap(
    product_names: list[str],
    cannibalization_matrix: dict[str, dict[str, dict[str, float]]],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "cannibHeatmap",
) -> str:
    """
    Create heatmap showing cross-product cannibalization effects.

    Parameters
    ----------
    product_names : list[str]
        Names of products
    cannibalization_matrix : dict
        Nested dict: {source: {target: {"mean", "lower", "upper"}}}
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
    annotations = []

    for i, source in enumerate(product_names):
        row_vals = []
        row_text = []
        for j, target in enumerate(product_names):
            if source == target:
                row_vals.append(None)  # Diagonal = N/A
                row_text.append(f"{source} (self)")
            else:
                effect = cannibalization_matrix.get(source, {}).get(target, {})
                mean = effect.get("mean", 0)
                lower = effect.get("lower", 0)
                upper = effect.get("upper", 0)
                row_vals.append(mean)

                effect_type = "Cannibalization" if mean < 0 else "Synergy"
                row_text.append(
                    f"{source} → {target}<br>"
                    f"{effect_type}: {mean:.1%}<br>"
                    f"CI: [{lower:.1%}, {upper:.1%}]"
                )

                # Add text annotation
                annotations.append(
                    {
                        "x": target,
                        "y": source,
                        "text": f"{mean:.1%}",
                        "showarrow": False,
                        "font": {
                            "size": 10,
                            "color": "#ffffff" if abs(mean) > 0.05 else colors.text,
                        },
                    }
                )

        z_values.append(row_vals)
        hover_text.append(row_text)

    traces = [
        {
            "type": "heatmap",
            "x": product_names,
            "y": product_names,
            "z": z_values,
            "text": hover_text,
            "hoverinfo": "text",
            "colorscale": [
                [0, "#c97067"],  # Cannibalization (negative)
                [0.5, "#fafbf9"],  # Neutral
                [1, colors.success],  # Synergy (positive)
            ],
            "colorbar": {
                "title": "Effect",
                "tickformat": ".0%",
                "thickness": 15,
            },
            "zmid": 0,
            "zmin": -0.15,
            "zmax": 0.15,
        }
    ]

    layout = chart_config.to_plotly_layout(colors)
    layout["xaxis"] = {"title": "Target Product", "side": "top", "tickangle": -45}
    layout["yaxis"] = {"title": "Source Product's Marketing", "autorange": "reversed"}
    layout["annotations"] = annotations

    return create_plotly_div(traces, layout, div_id)


__all__ = [
    "create_mediator_pathway_chart",
    "create_mediator_time_series",
    "create_cannibalization_heatmap",
]

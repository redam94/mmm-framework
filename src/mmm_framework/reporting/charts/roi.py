"""
ROI chart functions for MMM reporting.

Contains ROI forest plots and channel performance visualizations.
"""

from __future__ import annotations

import numpy as np

from ..config import ChartConfig, ReportConfig
from .base import create_plotly_div


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
            "hovertemplate": (
                "%{y}<br>ROI: %{x:.2f}<br>"
                "CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
            ),
            "customdata": [
                [l, u] for l, u in zip(roi_lower_sorted, roi_upper_sorted)
            ],
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


__all__ = [
    "create_roi_forest_plot",
]

"""
Chart generation functions for MMM reports.

All charts use Plotly for interactive visualization and can be embedded
in portable HTML reports.

This module provides comprehensive charting capabilities organized by domain:
- base: Common utilities and Plotly div creation
- fit: Model fit visualizations (actual vs predicted)
- decomposition: Waterfall and stacked area charts
- roi: ROI forest plots
- diagnostic: Saturation curves, adstock, prior/posterior, trace plots
- geo: Geographic heatmaps and comparisons
- extended: Mediator pathways, cannibalization heatmaps
"""

# Base utilities
from .base import (
    NumpyEncoder,
    _to_json,
    _hex_to_rgb,
    _dates_to_strings,
    create_plotly_div,
    _generate_dimension_colors,
    _build_dimension_filter_html,
    _build_dimension_filter_js,
)

# Model fit charts
from .fit import (
    create_model_fit_chart,
    create_model_fit_chart_with_geo_selector,
    create_model_fit_chart_with_dimension_filter,
    create_fit_statistics_with_geo_selector,
)

# Decomposition charts
from .decomposition import (
    create_decomposition_chart,
    create_stacked_area_chart,
    create_waterfall_chart,
    create_stacked_area_chart_with_geo_selector,
    create_waterfall_chart_with_geo_selector,
)

# ROI charts
from .roi import (
    create_roi_forest_plot,
)

# Diagnostic charts
from .diagnostic import (
    create_saturation_curves,
    create_adstock_chart,
    create_prior_posterior_chart,
    create_trace_plot,
    create_sensitivity_chart,
)

# Geographic charts
from .geo import (
    create_geo_roi_heatmap,
    create_geo_decomposition_chart,
)

# Extended model charts
from .extended import (
    create_mediator_pathway_chart,
    create_mediator_time_series,
    create_cannibalization_heatmap,
)

# Also import ChartConfig for convenience
from ..config import ChartConfig


__all__ = [
    # Base utilities
    "NumpyEncoder",
    "_to_json",
    "_hex_to_rgb",
    "_dates_to_strings",
    "create_plotly_div",
    "_generate_dimension_colors",
    "_build_dimension_filter_html",
    "_build_dimension_filter_js",
    # Configuration
    "ChartConfig",
    # Model fit charts
    "create_model_fit_chart",
    "create_model_fit_chart_with_geo_selector",
    "create_model_fit_chart_with_dimension_filter",
    "create_fit_statistics_with_geo_selector",
    # Decomposition charts
    "create_decomposition_chart",
    "create_stacked_area_chart",
    "create_waterfall_chart",
    "create_stacked_area_chart_with_geo_selector",
    "create_waterfall_chart_with_geo_selector",
    # ROI charts
    "create_roi_forest_plot",
    # Diagnostic charts
    "create_saturation_curves",
    "create_adstock_chart",
    "create_prior_posterior_chart",
    "create_trace_plot",
    "create_sensitivity_chart",
    # Geographic charts
    "create_geo_roi_heatmap",
    "create_geo_decomposition_chart",
    # Extended model charts
    "create_mediator_pathway_chart",
    "create_mediator_time_series",
    "create_cannibalization_heatmap",
]

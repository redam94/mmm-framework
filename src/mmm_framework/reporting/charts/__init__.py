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

# Spec-curve / model-averaging robustness chart (issue #103)
from .spec_curve import (
    create_spec_curve_plot,
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

# Posterior-predictive check (goodness-of-fit) charts
from .ppc import (
    create_ppc_observed_vs_predicted,
    create_ppc_density_overlay,
    create_ppc_interval_calibration,
    create_ppc_residual_plot,
)

# Pre-fit charts (prior densities, prior predictive, SBC)
from .prior import (
    create_prior_predictive_fan,
    create_prior_stat_distribution,
    create_prior_density_chart,
    create_prior_component_chart,
    create_prior_saturation_band,
    create_prior_adstock_band,
    create_sbc_rank_histogram,
    create_sbc_ecdf_diff,
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
    "create_spec_curve_plot",
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
    # Posterior-predictive check charts
    "create_ppc_observed_vs_predicted",
    "create_ppc_density_overlay",
    "create_ppc_interval_calibration",
    "create_ppc_residual_plot",
    # Pre-fit charts (prior densities, prior predictive, SBC)
    "create_prior_predictive_fan",
    "create_prior_stat_distribution",
    "create_prior_density_chart",
    "create_prior_component_chart",
    "create_prior_saturation_band",
    "create_prior_adstock_band",
    "create_sbc_rank_histogram",
    "create_sbc_ecdf_diff",
]

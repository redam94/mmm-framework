"""
MMM Reporting Module

A modular, portable HTML report generator for Bayesian Marketing Mix Models.
Integrates with mmm-framework's BayesianMMM and ExtendedMMM modules.

Features:
- Portable single-file HTML reports with embedded Plotly charts
- Customizable section-based architecture
- Honest uncertainty quantification throughout
- Support for nested, multivariate, and combined MMM models
- Interactive visualizations with credible intervals
"""

from .generator import MMMReportGenerator, ReportBuilder
from .config import ReportConfig, SectionConfig, ColorScheme, ColorPalette
from .sections import (
    Section,
    ExecutiveSummarySection,
    ModelFitSection,
    ChannelROISection,
    DecompositionSection,
    SaturationSection,
    SensitivitySection,
    MethodologySection,
    DiagnosticsSection,
)
from .charts import (
    ChartConfig,
    create_model_fit_chart,
    create_roi_forest_plot,
    create_waterfall_chart,
    create_decomposition_chart,
    create_stacked_area_chart,
    create_saturation_curves,
    create_adstock_chart,
    create_prior_posterior_chart,
    create_trace_plot,
)
from .data_extractors import MMMDataBundle

__version__ = "1.0.0"
__all__ = [
    # Main generator
    "MMMReportGenerator",
    "ReportBuilder",
    # Configuration
    "ReportConfig",
    "SectionConfig",
    "ColorScheme",
    "ColorPalette",
    # Sections
    "Section",
    "ExecutiveSummarySection",
    "ModelFitSection",
    "ChannelROISection",
    "DecompositionSection",
    "SaturationSection",
    "SensitivitySection",
    "MethodologySection",
    "DiagnosticsSection",
    # Charts
    "ChartConfig",
    "create_model_fit_chart",
    "create_roi_forest_plot",
    "create_waterfall_chart",
    "create_decomposition_chart",
    "create_stacked_area_chart",
    "create_saturation_curves",
    "create_adstock_chart",
    "create_prior_posterior_chart",
    "create_trace_plot",
    # Data Extractors
    "MMMDataBundle",
]
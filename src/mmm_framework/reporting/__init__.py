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
from .prefit import (
    PrefitReadoutGenerator,
    build_prefit_insights,
    prefit_facts,
    PREFIT_INSIGHT_SLOTS,
)
from .consultant_artifacts import ARTIFACTS, ArtifactSpec, render_artifact, write_all
from .data_extractors import (
    MMMDataBundle,
    DataExtractor,
    AggregationMixin,
    BayesianMMMExtractor,
    ExtendedMMMExtractor,
    PyMCMarketingExtractor,
    create_extractor,
)
from .sections import (
    Section,
    ExecutiveSummarySection,
    FactorAnalysisSection,
    ModelFitSection,
    PosteriorPredictiveSection,
    EstimandsSection,
    ChannelROISection,
    DecompositionSection,
    SaturationSection,
    SensitivitySection,
    MethodologySection,
    DiagnosticsSection,
    GeographicSection,
    MediatorSection,
    CannibalizationSection,
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
    create_geo_roi_heatmap,
    create_geo_decomposition_chart,
    create_mediator_pathway_chart,
    create_mediator_time_series,
    create_cannibalization_heatmap,
    create_ppc_observed_vs_predicted,
    create_ppc_density_overlay,
    create_ppc_interval_calibration,
    create_ppc_residual_plot,
)
from .helpers import (
    compute_roi_with_uncertainty,
    compute_marginal_roi,
    get_prior_posterior_comparison,
    compute_shrinkage_summary,
    compute_saturation_curves_with_uncertainty,
    compute_adstock_weights,
    compute_component_decomposition,
    compute_decomposition_waterfall,
    compute_mediated_effects,
    compute_cross_effects,
    generate_model_summary,
)
from .model_defense import (
    build_model_defense,
    render_model_defense_html,
    model_defense_report,
)

__version__ = "1.0.0"
__all__ = [
    # Main generator
    "MMMReportGenerator",
    "ReportBuilder",
    # Pre-fit Model Design Readout
    "PrefitReadoutGenerator",
    "build_prefit_insights",
    "prefit_facts",
    "PREFIT_INSIGHT_SLOTS",
    # Model-defense (causal-rigor) report
    "build_model_defense",
    "render_model_defense_html",
    "model_defense_report",
    # Consultant artifacts
    "ARTIFACTS",
    "ArtifactSpec",
    "render_artifact",
    "write_all",
    # Data & Extractors
    "MMMDataBundle",
    "DataExtractor",
    "AggregationMixin",
    "BayesianMMMExtractor",
    "ExtendedMMMExtractor",
    "PyMCMarketingExtractor",
    "create_extractor",
    # Configuration
    "ReportConfig",
    "SectionConfig",
    "ColorScheme",
    "ColorPalette",
    # Sections
    "Section",
    "ExecutiveSummarySection",
    "FactorAnalysisSection",
    "ModelFitSection",
    "PosteriorPredictiveSection",
    "EstimandsSection",
    "ChannelROISection",
    "DecompositionSection",
    "SaturationSection",
    "SensitivitySection",
    "MethodologySection",
    "DiagnosticsSection",
    "GeographicSection",
    "MediatorSection",
    "CannibalizationSection",
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
    "create_geo_roi_heatmap",
    "create_geo_decomposition_chart",
    "create_mediator_pathway_chart",
    "create_mediator_time_series",
    "create_cannibalization_heatmap",
    "create_ppc_observed_vs_predicted",
    "create_ppc_density_overlay",
    "create_ppc_interval_calibration",
    "create_ppc_residual_plot",
]

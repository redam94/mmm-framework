"""
UI Components Package.

Provides reusable UI components for the MMM Streamlit application.
"""

from .common import (
    # Color utilities
    rgb_to_rgba,
    CHART_COLORS,
    COMPONENT_COLORS,
    
    # Status helpers
    status_badge,
    status_icon,
    
    # Formatting
    format_bytes,
    format_duration,
    format_datetime,
    format_percentage,
    format_number,
    
    # Session state
    init_session_state,
    get_session,
    set_session,
    
    # UI components
    metric_card,
    info_box,
    confirm_dialog,
    loading_placeholder,
    data_preview_table,
    summary_statistics,
    job_progress_display,
    page_header,
    sidebar_status,
    
    # Error handling
    display_api_error,
    handle_api_call,
    
    # CSS
    CUSTOM_CSS,
    apply_custom_css,
)

from .charts import (
    # Model fit
    plot_model_fit,
    plot_residuals,
    
    # Contributions
    plot_channel_contributions,
    plot_contribution_waterfall,
    plot_contribution_pie,
    plot_contribution_timeseries,
    
    # Response curves
    plot_response_curves,
    plot_marginal_roas,
    
    # Decomposition
    plot_component_decomposition,
    
    # Posteriors
    plot_posterior_distributions,
    plot_trace,
    
    # Scenarios
    plot_scenario_comparison,
    plot_budget_optimization,
)


__all__ = [
    # Common
    "rgb_to_rgba",
    "CHART_COLORS",
    "COMPONENT_COLORS",
    "status_badge",
    "status_icon",
    "format_bytes",
    "format_duration",
    "format_datetime",
    "format_percentage",
    "format_number",
    "init_session_state",
    "get_session",
    "set_session",
    "metric_card",
    "info_box",
    "confirm_dialog",
    "loading_placeholder",
    "data_preview_table",
    "summary_statistics",
    "job_progress_display",
    "page_header",
    "sidebar_status",
    "display_api_error",
    "handle_api_call",
    "CUSTOM_CSS",
    "apply_custom_css",
    
    # Charts
    "plot_model_fit",
    "plot_residuals",
    "plot_channel_contributions",
    "plot_contribution_waterfall",
    "plot_contribution_pie",
    "plot_contribution_timeseries",
    "plot_response_curves",
    "plot_marginal_roas",
    "plot_component_decomposition",
    "plot_posterior_distributions",
    "plot_trace",
    "plot_scenario_comparison",
    "plot_budget_optimization",
]
"""
Common UI Components and Utilities.

Provides reusable components, formatters, and session state helpers.
"""

import streamlit as st
from datetime import datetime
from typing import Any


# =============================================================================
# Custom CSS
# =============================================================================


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown(
        """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Page Header
# =============================================================================


def page_header(title: str, description: str = ""):
    """Render a consistent page header."""
    st.title(title)
    if description:
        st.markdown(description)
    st.markdown("---")


# =============================================================================
# Session State
# =============================================================================


def init_session_state(**defaults):
    """Initialize session state with default values."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session_value(key: str, default: Any = None) -> Any:
    """Get value from session state with default."""
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """Set value in session state."""
    st.session_state[key] = value


# =============================================================================
# Formatters
# =============================================================================


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M") -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime(format_str)


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:,.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:,.{decimals}f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$", decimals: int = 0) -> str:
    """Format value as currency."""
    if value is None:
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"


# =============================================================================
# Error Handling
# =============================================================================


def display_api_error(error: Exception, context: str = ""):
    """Display API error with context."""
    from api_client import APIError

    if isinstance(error, APIError):
        st.error(f"❌ {context}: {error.message}")
        if error.details:
            with st.expander("Error Details"):
                st.json(error.details)
    else:
        st.error(f"❌ {context}: {str(error)}")


def display_validation_error(errors: list[str]):
    """Display validation errors."""
    if errors:
        st.error("⚠️ Please fix the following errors:")
        for error in errors:
            st.markdown(f"- {error}")


# =============================================================================
# Status Indicators
# =============================================================================


def status_badge(status: str) -> str:
    """Return emoji badge for status."""
    badges = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
        "queued": "📋",
    }
    return badges.get(status.lower(), "❓")


def status_color(status: str) -> str:
    """Return color for status."""
    colors = {
        "pending": "orange",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "gray",
        "queued": "purple",
    }
    return colors.get(status.lower(), "gray")


# =============================================================================
# Progress Display
# =============================================================================


def render_progress(progress: float, message: str = ""):
    """Render progress bar with message."""
    st.progress(progress)
    if message:
        st.caption(message)


# =============================================================================
# Confirmation Dialog
# =============================================================================


def confirm_action(key: str, message: str = "Are you sure?") -> bool:
    """Simple confirmation using checkbox."""
    return st.checkbox(message, key=f"confirm_{key}")


from .common import (
    # Color utilities
    rgb_to_rgba,
    CHART_COLORS,
    COMPONENT_COLORS,
    # # Status helpers
    # status_badge,
    status_icon,
    # # Formatting
    format_bytes,
    format_duration,
    # format_datetime,
    # format_percentage,
    # format_number,
    # # Session state
    # init_session_state,
    # get_session,
    # set_session,
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
    # # Error handling
    # display_api_error,
    # handle_api_call,
    # CSS
    CUSTOM_CSS,
    # apply_custom_css,
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

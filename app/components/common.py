"""
Common UI Components and Utilities.

Shared components used across multiple pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any


# =============================================================================
# Color Utilities
# =============================================================================

def rgb_to_rgba(rgb: str, alpha: float = 1.0) -> str:
    """Convert RGB color to RGBA."""
    r, g, b = rgb.strip("rgb(").strip(")").split(",")
    return f"rgba({r},{g},{b},{alpha})"


# Default color palette for charts
CHART_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

COMPONENT_COLORS = {
    "Base": "#1f77b4",
    "Trend": "#ff7f0e",
    "Seasonality": "#2ca02c",
    "Media": "#d62728",
    "Controls": "#9467bd",
    "Observed": "#17becf",
    "Predicted": "#e377c2",
    "Residual": "#7f7f7f",
}


# =============================================================================
# Status Indicators
# =============================================================================

def status_badge(status: str) -> str:
    """Get status badge HTML."""
    status_colors = {
        "pending": ("⏳", "#FFA500"),
        "queued": ("📋", "#6495ED"),
        "running": ("🔄", "#32CD32"),
        "completed": ("✅", "#228B22"),
        "failed": ("❌", "#DC143C"),
        "cancelled": ("🚫", "#808080"),
    }
    emoji, color = status_colors.get(status.lower(), ("❓", "#808080"))
    return f'<span style="color: {color}; font-weight: bold;">{emoji} {status.capitalize()}</span>'


def status_icon(status: str) -> str:
    """Get status icon emoji."""
    icons = {
        "pending": "⏳",
        "queued": "📋",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }
    return icons.get(status.lower(), "❓")


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float | None) -> str:
    """Format duration to human readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_datetime(dt: datetime | str | None) -> str:
    """Format datetime to human readable string."""
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M")


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


# =============================================================================
# Session State Helpers
# =============================================================================

def init_session_state(**defaults):
    """Initialize session state with defaults."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session(key: str, default: Any = None) -> Any:
    """Get session state value with default."""
    return st.session_state.get(key, default)


def set_session(key: str, value: Any):
    """Set session state value."""
    st.session_state[key] = value


# =============================================================================
# UI Components
# =============================================================================

def metric_card(label: str, value: str | float, delta: str | None = None, delta_color: str = "normal"):
    """Display a metric in a styled card."""
    with st.container():
        st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def info_box(message: str, type: str = "info"):
    """Display an info box."""
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)


def confirm_dialog(message: str, key: str) -> bool:
    """Display a confirmation dialog."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.warning(message)
    with col2:
        return st.button("Confirm", key=key, type="primary")


def loading_placeholder(message: str = "Loading..."):
    """Display a loading placeholder."""
    return st.empty()


# =============================================================================
# Data Display Components
# =============================================================================

@st.fragment
def data_preview_table(df: pd.DataFrame, max_rows: int = 10):
    """Display a data preview table (fragment for isolation)."""
    st.dataframe(df.head(max_rows), use_container_width=True)


@st.fragment
def summary_statistics(df: pd.DataFrame):
    """Display summary statistics for a dataframe (fragment for isolation)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
    else:
        st.info("No numeric columns found.")


# =============================================================================
# Progress Components
# =============================================================================

@st.fragment
def job_progress_display(
    status: str,
    progress: float,
    message: str | None = None,
    show_spinner: bool = True
):
    """Display job progress (fragment for isolated updates)."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"### {status_icon(status)}")
    
    with col2:
        st.markdown(f"**Status:** {status.capitalize()}")
        
        if status == "running":
            st.progress(progress / 100, text=message or f"{progress:.0f}%")
        elif status == "completed":
            st.progress(1.0, text="Complete!")
        elif status == "failed":
            st.error(message or "Job failed")
        else:
            st.progress(0, text=message or "Waiting...")


# =============================================================================
# Navigation Helpers
# =============================================================================

def page_header(title: str, description: str | None = None):
    """Display a page header."""
    st.title(title)
    if description:
        st.markdown(description)
    st.markdown("---")


def sidebar_status(
    data_loaded: bool,
    config_loaded: bool,
    model_fitted: bool,
    results_loaded: bool
):
    """Display sidebar status indicators."""
    st.sidebar.markdown("### Status")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if data_loaded:
            st.success("✓ Data")
        else:
            st.warning("○ Data")
    with col2:
        if config_loaded:
            st.success("✓ Config")
        else:
            st.warning("○ Config")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if model_fitted:
            st.success("✓ Model")
        else:
            st.warning("○ Model")
    with col2:
        if results_loaded:
            st.success("✓ Results")
        else:
            st.warning("○ Results")


# =============================================================================
# Error Handling
# =============================================================================

def display_api_error(error: Exception):
    """Display API error in a user-friendly way."""
    error_msg = str(error)
    
    if "Connection refused" in error_msg or "ConnectError" in error_msg:
        st.error("🔌 **Connection Error**: Cannot connect to the API server. Please ensure the backend is running.")
        with st.expander("Technical Details"):
            st.code(error_msg)
    elif "404" in error_msg:
        st.error("🔍 **Not Found**: The requested resource was not found.")
        with st.expander("Technical Details"):
            st.code(error_msg)
    elif "500" in error_msg:
        st.error("🔥 **Server Error**: An internal server error occurred.")
        with st.expander("Technical Details"):
            st.code(error_msg)
    else:
        st.error(f"❌ **Error**: {error_msg}")


def handle_api_call(func, *args, error_message: str = "Operation failed", **kwargs):
    """Execute API call with error handling."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, e


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .job-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
    }
    .job-card.running {
        border-left-color: #f39c12;
        animation: pulse 2s infinite;
    }
    .job-card.completed {
        border-left-color: #27ae60;
    }
    .job-card.failed {
        border-left-color: #e74c3c;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
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
</style>
"""


def apply_custom_css():
    """Apply custom CSS styles."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
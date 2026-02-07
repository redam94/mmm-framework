"""
Common UI Components for MMM Framework.

Provides reusable UI utilities and styling.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Any

# =============================================================================
# Color Utilities
# =============================================================================

CHART_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

COMPONENT_COLORS = {
    "baseline": "#636EFA",
    "trend": "#EF553B",
    "seasonality": "#00CC96",
    "media": "#AB63FA",
    "control": "#FFA15A",
    "residual": "#19D3F3",
}


def rgb_to_rgba(color: str, alpha: float = 0.3) -> str:
    """Convert hex or rgb color to rgba with specified alpha."""
    if color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    elif color.startswith("rgb"):
        return color.replace("rgb", "rgba").replace(")", f",{alpha})")
    return color


# =============================================================================
# Status Helpers
# =============================================================================


def status_icon(status: str) -> str:
    """Get status icon for job status."""
    icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }
    return icons.get(status.lower(), "❓")


def status_badge(status: str) -> str:
    """Get styled status badge."""
    colors = {
        "pending": "gray",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "cancelled": "orange",
    }
    color = colors.get(status.lower(), "gray")
    return f":{color}[{status_icon(status)} {status.title()}]"


# =============================================================================
# Formatting
# =============================================================================


def format_bytes(size_bytes: int | None) -> str:
    """Format bytes to human readable string."""
    if size_bytes is None:
        return "N/A"

    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds to human readable string."""
    if seconds is None:
        return "N/A"

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_datetime(dt: datetime | str | None) -> str:
    """Format datetime to readable string."""
    if dt is None:
        return "N/A"

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt

    return dt.strftime("%Y-%m-%d %H:%M")


def format_percentage(value: float | None, decimals: int = 1) -> str:
    """Format float as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float | int | None, decimals: int = 2) -> str:
    """Format number with appropriate suffix."""
    if value is None:
        return "N/A"

    if abs(value) >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


# =============================================================================
# Session State
# =============================================================================


def init_session_state(**defaults):
    """Initialize session state with defaults."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_session(key: str, default: Any = None) -> Any:
    """Get value from session state."""
    return st.session_state.get(key, default)


def set_session(key: str, value: Any):
    """Set value in session state."""
    st.session_state[key] = value


# =============================================================================
# UI Components
# =============================================================================


def metric_card(
    label: str, value: Any, delta: Any = None, help_text: str | None = None
):
    """Display a styled metric card."""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def info_box(title: str, content: str, type: str = "info"):
    """Display an info box with title and content."""
    if type == "info":
        st.info(f"**{title}**\n\n{content}")
    elif type == "warning":
        st.warning(f"**{title}**\n\n{content}")
    elif type == "error":
        st.error(f"**{title}**\n\n{content}")
    elif type == "success":
        st.success(f"**{title}**\n\n{content}")


def confirm_dialog(title: str, message: str, confirm_label: str = "Confirm") -> bool:
    """Display a confirmation dialog."""
    with st.expander(title, expanded=True):
        st.warning(message)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(confirm_label, type="primary", use_container_width=True):
                return True
        with col2:
            if st.button("Cancel", use_container_width=True):
                return False
    return False


def loading_placeholder(message: str = "Loading..."):
    """Display a loading placeholder."""
    return st.empty()


def data_preview_table(data: list[dict], max_rows: int = 10):
    """Display a preview table of data."""
    import pandas as pd

    df = pd.DataFrame(data[:max_rows])
    st.dataframe(df, use_container_width=True)


def summary_statistics(data: dict[str, Any]):
    """Display summary statistics in columns."""
    n_cols = min(4, len(data))
    cols = st.columns(n_cols)

    for i, (label, value) in enumerate(data.items()):
        with cols[i % n_cols]:
            st.metric(label, value)


def job_progress_display(progress: float, message: str | None = None):
    """Display job progress with optional message."""
    st.progress(progress / 100)
    if message:
        st.caption(message)


def page_header(title: str, subtitle: str | None = None):
    """Display a consistent page header."""
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")


def sidebar_status():
    """Display API connection status in sidebar."""
    from api_client import check_api_connection

    with st.sidebar:
        if check_api_connection():
            st.success("✅ API Connected")
            st.session_state.api_connected = True
        else:
            st.error("❌ API Disconnected")
            st.session_state.api_connected = False


# =============================================================================
# Error Handling
# =============================================================================


def display_api_error(error: Exception):
    """Display API error with details."""
    from api_client import APIError, ConnectionError, NotFoundError, ValidationError

    if isinstance(error, ConnectionError):
        st.error(f"🔌 Connection Error: {error.message}")
        st.info("Please ensure the API server is running.")
    elif isinstance(error, NotFoundError):
        st.warning(f"🔍 Not Found: {error.detail}")
    elif isinstance(error, ValidationError):
        st.error(f"⚠️ Validation Error: {error.detail}")
    elif isinstance(error, APIError):
        st.error(f"❌ API Error ({error.status_code}): {error.message}")
    else:
        st.error(f"❌ Error: {str(error)}")


def handle_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        display_api_error(e)
        return None


# =============================================================================
# CSS Styling
# =============================================================================

CUSTOM_CSS = """
<style>
    /* Reduce padding for denser layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style metric cards */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Style expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: #f0f2f6;
        border-radius: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e0e2e6;
    }
    
    /* Improve table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Style buttons */
    .stButton > button {
        border-radius: 4px;
    }
    
    /* Style download buttons */
    .stDownloadButton > button {
        width: 100%;
    }
</style>
"""


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

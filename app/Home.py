"""
MMM Framework - Streamlit Web Application

Main entry point for the multipage application.
Uses FastAPI backend for all data operations.
"""

import streamlit as st

from api_client import get_api_client, APIError
from components.common import (
    apply_custom_css,
    page_header,
    sidebar_status,
    format_datetime,
    format_bytes,
    display_api_error,
    init_session_state,
    get_session,
    status_icon,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="MMM Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()


# =============================================================================
# Session State Initialization
# =============================================================================

init_session_state(
    selected_data_id=None,
    selected_config_id=None,
    selected_model_id=None,
    api_connected=False,
)


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar():
    """Render the sidebar with navigation and status."""
    with st.sidebar:
        st.title("📊 MMM Framework")
        st.markdown("---")

        # API Connection Status
        st.subheader("API Status")

        try:
            client = get_api_client()
            health = client.health()
            st.session_state.api_connected = True

            col1, col2 = st.columns(2)
            with col1:
                if health.get("status") == "healthy":
                    st.success("✓ API")
                else:
                    st.warning("⚠ API")
            with col2:
                if health.get("redis_connected"):
                    st.success("✓ Redis")
                else:
                    st.warning("○ Redis")

            col1, col2 = st.columns(2)
            with col1:
                if health.get("worker_active"):
                    st.success("✓ Worker")
                else:
                    st.warning("○ Worker")
            with col2:
                st.caption(f"v{health.get('version')}")

        except Exception as e:
            st.session_state.api_connected = False
            st.error("✗ Disconnected")
            with st.expander("Details"):
                st.caption(str(e))

        st.markdown("---")

        # Current selections
        st.subheader("Current Selection")

        if st.session_state.selected_data_id:
            st.info(f"📁 Data: {st.session_state.selected_data_id[:8]}...")
        else:
            st.caption("No data selected")

        if st.session_state.selected_config_id:
            st.info(f"⚙️ Config: {st.session_state.selected_config_id[:8]}...")
        else:
            st.caption("No config selected")

        if st.session_state.selected_model_id:
            st.info(f"🔬 Model: {st.session_state.selected_model_id[:8]}...")
        else:
            st.caption("No model selected")

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_data_id = None
            st.session_state.selected_config_id = None
            st.session_state.selected_model_id = None
            st.rerun()

        st.markdown("---")

        # Info
        st.caption("Built with PyMC & FastAPI")
        st.caption("v0.5.0 - Multipage App")


# =============================================================================
# Main Content
# =============================================================================


def main():
    """Main application entry point."""
    render_sidebar()

    # Page header
    page_header(
        "Marketing Mix Model Framework",
        "A comprehensive platform for building, fitting, and analyzing Marketing Mix Models.",
    )

    # Check API connection
    if not st.session_state.api_connected:
        st.error(
            "⚠️ Cannot connect to the API server. Please ensure the backend is running."
        )
        st.markdown(
            """
        ### Getting Started
        
        1. Start the FastAPI backend:
           ```bash
           cd api && uvicorn main:app --reload
           ```
        
        2. Start the ARQ worker:
           ```bash
           cd api && arq worker.WorkerSettings
           ```
        
        3. Ensure Redis is running:
           ```bash
           redis-server
           ```
        """
        )
        return

    # Dashboard overview
    st.markdown("## Dashboard")

    try:
        client = get_api_client()
        health_detailed = client.health_detailed()

        # System metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            storage_info = health_detailed.get("storage", {})
            st.metric("Datasets", storage_info.get("datasets", 0))

        with col2:
            st.metric("Configurations", storage_info.get("configs", 0))

        with col3:
            st.metric("Models", storage_info.get("models", 0))

        with col4:
            queue_info = health_detailed.get("queue", {})
            active = queue_info.get("active", 0)
            pending = queue_info.get("pending", 0)
            st.metric(
                "Active Jobs",
                active,
                delta=f"{pending} pending" if pending > 0 else None,
            )

    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error fetching dashboard data: {e}")

    st.markdown("---")

    # Quick start guide
    st.markdown("## Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### Workflow
        
        1. **📁 Data Management** - Upload your MFF data
        2. **⚙️ Configuration** - Create model configuration
        3. **🔬 Model Fitting** - Start Bayesian model fitting
        4. **📈 Results** - View model diagnostics and contributions
        5. **🔮 Scenarios** - Run what-if scenarios
        """
        )

    with col2:
        st.markdown(
            """
        ### Features
        
        - **Async Model Fitting** - Non-blocking background jobs
        - **Real-time Progress** - Track fitting progress
        - **Counterfactual Analysis** - Channel contribution decomposition
        - **Scenario Planning** - Budget optimization
        - **Export Results** - Download models and reports
        """
        )

    st.markdown("---")

    # Recent activity
    st.markdown("## Recent Activity")

    try:
        client = get_api_client()
        models = client.list_models(limit=5)

        if models:
            for model in models:
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

                with col1:
                    st.markdown(f"**{getattr(model, 'name', None) or model.model_id}**")

                with col2:
                    st.caption(format_datetime(model.created_at))

                with col3:
                    st.markdown(
                        f"{status_icon(model.status)} {model.status.capitalize()}"
                    )

                with col4:
                    if model.status == "active":
                        st.progress(model.progress / 100)
                    elif model.status == "completed":
                        st.success("Complete")
                    elif model.status == "failed":
                        st.error("Failed")

                with col5:
                    if st.button("View", key=f"view_{model.model_id}"):
                        st.session_state.selected_model_id = model.model_id
                        st.switch_page("pages/4_📈_Results.py")
        else:
            st.info(
                "No models found. Get started by uploading data and creating a configuration."
            )

    except Exception as e:
        st.warning(f"Could not load recent activity: {e}")

    st.markdown("---")

    # Help section
    with st.expander("ℹ️ Help & Documentation"):
        st.markdown(
            """
        ### Data Format
        
        The framework expects data in **Master Flat File (MFF)** format with columns:
        - `Period` - Time period identifier
        - `Geography` - Geographic dimension (optional)
        - `Product` - Product dimension (optional)
        - `Campaign`, `Outlet`, `Creative` - Media dimensions (optional)
        - `VariableName` - Variable identifier
        - `VariableValue` - Numeric value
        
        ### Model Configuration
        
        Configure your model with:
        - **KPI**: Target variable to model
        - **Media Channels**: Marketing spend variables with adstock/saturation
        - **Controls**: Non-media variables (price, distribution, etc.)
        - **Trend**: Time trend (none, linear, piecewise, spline, GP)
        - **Seasonality**: Fourier seasonality components
        
        ### API Endpoints
        
        For programmatic access, see the API documentation at:
        - Swagger UI: http://localhost:8000/docs
        - ReDoc: http://localhost:8000/redoc
        """
        )


if __name__ == "__main__":
    main()

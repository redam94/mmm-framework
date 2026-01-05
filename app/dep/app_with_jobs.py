"""
MMM Framework - Streamlit Web Application with Job Management

A user-friendly interface for:
- Uploading MFF data
- Configuring model settings
- Running Bayesian MMM in background processes
- Managing multiple model fitting jobs
- Visualizing results with counterfactual contributions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")


def rgb_to_rgba(rgb: str, alpha: float = 1.0) -> str:
    """Convert RGB color to RGBA."""
    r, g, b = rgb.strip("rgb(").strip(")").split(",")
    return f"rgba({r},{g},{b},{alpha})"


# Set page config
st.set_page_config(
    page_title="Marketing Mix Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "data": None,
        "data_filename": None,
        "panel": None,
        "mff_config": None,
        "model_config": None,
        # Job-based results
        "active_job_id": None,
        "loaded_results": None,  # Results loaded from a job
        "loaded_mmm": None,  # MMM loaded from a job
        "loaded_contributions": None,
        "component_decomposition": None,
        "prior_samples": None,
        # Config state
        "kpi_name": "Sales",
        "kpi_dimensions": "National",
        "media_channels": [],
        "control_variables": [],
        "geo_allocation": "equal",
        # Job refresh timestamp
        "last_job_refresh": time.time(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================


@st.cache_data
def load_data(uploaded_file):
    """Load uploaded CSV or Excel file."""
    filename = uploaded_file.name
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    return df


def detect_mff_columns(df):
    """Detect MFF column structure."""
    expected_cols = [
        "Period",
        "Geography",
        "Product",
        "Campaign",
        "Outlet",
        "Creative",
        "VariableName",
        "VariableValue",
    ]

    found = [col for col in expected_cols if col in df.columns]
    missing = [col for col in expected_cols if col not in df.columns]

    return found, missing


def get_variable_names(df):
    """Extract unique variable names from MFF data."""
    if "VariableName" in df.columns:
        return sorted(df["VariableName"].unique().tolist())
    return []


def get_dimension_values(df, dim_col):
    """Get unique values for a dimension column."""
    if dim_col in df.columns:
        values = df[dim_col].dropna().unique()
        values = [v for v in values if v != "" and pd.notna(v)]
        return sorted(values)
    return []


def create_sample_mff():
    """Create sample MFF data for demo."""
    from datetime import datetime, timedelta

    np.random.seed(42)
    n_weeks = 104
    start_date = datetime(2022, 1, 3)

    records = []
    for i in range(n_weeks):
        date = start_date + timedelta(weeks=i)
        date_str = date.strftime("%Y-%m-%d")

        # Sales KPI
        trend = 1000 + 2 * i
        seasonality = 200 * np.sin(2 * np.pi * i / 52)
        sales = trend + seasonality + np.random.normal(0, 100)

        records.append(
            {
                "Period": date_str,
                "Geography": "",
                "Product": "",
                "Campaign": "",
                "Outlet": "",
                "Creative": "",
                "VariableName": "Sales",
                "VariableValue": max(0, sales),
            }
        )

        # Media channels
        for channel, spend_range in [
            ("TV", (30000, 80000)),
            ("Digital", (20000, 50000)),
            ("Radio", (10000, 30000)),
        ]:
            records.append(
                {
                    "Period": date_str,
                    "Geography": "",
                    "Product": "",
                    "Campaign": "",
                    "Outlet": "",
                    "Creative": "",
                    "VariableName": channel,
                    "VariableValue": np.random.uniform(*spend_range),
                }
            )

        # Controls
        records.append(
            {
                "Period": date_str,
                "Geography": "",
                "Product": "",
                "Campaign": "",
                "Outlet": "",
                "Creative": "",
                "VariableName": "Price",
                "VariableValue": 100 + np.random.normal(0, 5),
            }
        )
        records.append(
            {
                "Period": date_str,
                "Geography": "",
                "Product": "",
                "Campaign": "",
                "Outlet": "",
                "Creative": "",
                "VariableName": "Distribution",
                "VariableValue": 0.8 + np.random.uniform(-0.1, 0.1),
            }
        )

    return pd.DataFrame(records)


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_status_color(status) -> str:
    """Get color for job status."""
    from mmm_framework import JobStatus

    return {
        JobStatus.PENDING: "#3498db",
        JobStatus.RUNNING: "#f39c12",
        JobStatus.COMPLETED: "#27ae60",
        JobStatus.FAILED: "#e74c3c",
        JobStatus.CANCELLED: "#95a5a6",
    }.get(status, "#7f8c8d")


def get_status_icon(status) -> str:
    """Get icon for job status."""
    from mmm_framework import JobStatus

    return {
        JobStatus.PENDING: "⏳",
        JobStatus.RUNNING: "🔄",
        JobStatus.COMPLETED: "✅",
        JobStatus.FAILED: "❌",
        JobStatus.CANCELLED: "🚫",
    }.get(status, "❓")


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar():
    """Render the sidebar with navigation and status."""
    with st.sidebar:
        st.title("📊 MMM Framework")
        st.markdown("---")

        # Status indicators
        st.subheader("Status")

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.data is not None:
                st.success("✓ Data")
            else:
                st.warning("○ Data")
        with col2:
            if st.session_state.mff_config is not None:
                st.success("✓ Config")
            else:
                st.warning("○ Config")

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.panel is not None:
                st.success("✓ Panel")
            else:
                st.warning("○ Panel")
        with col2:
            if st.session_state.loaded_results is not None:
                st.success("✓ Results")
            else:
                st.warning("○ Results")

        st.markdown("---")

        # Active Jobs Summary
        try:
            from mmm_framework import get_job_manager, JobStatus

            manager = get_job_manager()

            active_jobs = manager.get_active_jobs()
            completed_jobs = manager.list_jobs(
                status_filter=[JobStatus.COMPLETED], limit=5
            )

            st.subheader("Jobs")

            if active_jobs:
                st.info(f"🔄 {len(active_jobs)} active job(s)")
            else:
                st.caption("No active jobs")

            if completed_jobs:
                st.success(f"✅ {len(completed_jobs)} recent completed")

        except Exception as e:
            st.caption(f"Jobs unavailable: {e}")

        st.markdown("---")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()

        if st.button("📥 Load Demo Data", use_container_width=True):
            st.session_state.data = create_sample_mff()
            st.session_state.data_filename = "demo_data.csv"
            st.rerun()

        st.markdown("---")

        # Info
        st.caption("Built with PyMC & Streamlit")
        st.caption("v0.4.0 - Job Management")


# =============================================================================
# Tab 1: Data Upload
# =============================================================================


def render_data_tab():
    """Render the data upload and preview tab."""
    st.header("📁 Data Upload")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload MFF File",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Upload a Master Flat File (MFF) with columns: Period, Geography, Product, Campaign, Outlet, Creative, VariableName, VariableValue",
        )

        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.session_state.data = df
                st.session_state.data_filename = uploaded_file.name
                st.success(f"✓ Loaded {len(df):,} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col2:
        st.markdown("**Expected MFF Format:**")
        st.code(
            """
Period,Geography,Product,...,VariableName,VariableValue
2022-01-03,,,,...,Sales,1234.56
2022-01-03,,,,...,TV,50000.00
        """
        )

    # Data preview
    if st.session_state.data is not None:
        df = st.session_state.data

        st.markdown("---")
        st.subheader("Data Preview")

        # Column detection
        found_cols, missing_cols = detect_mff_columns(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Variables", len(get_variable_names(df)))

        if missing_cols:
            st.warning(f"⚠️ Missing expected columns: {', '.join(missing_cols)}")

        # Show data
        st.dataframe(df.head(100), use_container_width=True)

        # Variable summary
        st.subheader("Variable Summary")

        if "VariableName" in df.columns and "VariableValue" in df.columns:
            var_summary = df.groupby("VariableName")["VariableValue"].agg(
                ["count", "mean", "std", "min", "max"]
            )
            var_summary = var_summary.round(2)
            st.dataframe(var_summary, use_container_width=True)

        # Dimension analysis
        st.subheader("Dimensions")

        dim_cols = ["Geography", "Product", "Campaign", "Outlet", "Creative"]
        dim_info = []
        for col in dim_cols:
            values = get_dimension_values(df, col)
            dim_info.append(
                {
                    "Dimension": col,
                    "Unique Values": len(values),
                    "Values": (
                        ", ".join(values[:5]) + ("..." if len(values) > 5 else "")
                        if values
                        else "(empty)"
                    ),
                }
            )

        st.dataframe(pd.DataFrame(dim_info), use_container_width=True)


# =============================================================================
# Tab 2: Configuration
# =============================================================================


def render_config_tab():
    """Render the model configuration tab."""
    st.header("⚙️ Model Configuration")

    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return

    df = st.session_state.data
    variables = get_variable_names(df)
    geographies = get_dimension_values(df, "Geography")
    products = get_dimension_values(df, "Product")

    # Configuration sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 KPI Configuration")

        kpi_name = st.selectbox(
            "KPI Variable",
            options=variables,
            index=variables.index("Sales") if "Sales" in variables else 0,
            help="Select the dependent variable (e.g., Sales, Revenue)",
        )
        st.session_state.kpi_name = kpi_name

        kpi_dim_options = ["National"]
        if geographies:
            kpi_dim_options.append("By Geography")
        if products:
            kpi_dim_options.append("By Product")
        if geographies and products:
            kpi_dim_options.append("By Geography & Product")

        kpi_dimensions = st.selectbox(
            "KPI Granularity",
            options=kpi_dim_options,
            help="At what level is the KPI measured?",
        )
        st.session_state.kpi_dimensions = kpi_dimensions

        st.markdown("---")

        st.subheader("📺 Media Channels")

        # Filter out KPI and common control names
        potential_media = [
            v
            for v in variables
            if v not in [kpi_name, "Price", "Distribution", "Temperature", "Promotion"]
        ]

        media_channels = st.multiselect(
            "Select Media Channels",
            options=potential_media,
            default=[
                v
                for v in ["TV", "Digital", "Radio", "Social", "Print"]
                if v in potential_media
            ],
            help="Select variables that represent media spend",
        )

        # Media configuration
        media_configs = []
        if media_channels:
            st.markdown("**Channel Settings:**")

            for channel in media_channels:
                with st.expander(f"📺 {channel}", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        adstock_max = st.slider(
                            f"Max Adstock Lag (weeks)",
                            min_value=1,
                            max_value=12,
                            value=8,
                            key=f"adstock_{channel}",
                        )
                    with col_b:
                        channel_level = st.selectbox(
                            "Data Level",
                            options=["National", "By Geography"],
                            key=f"level_{channel}",
                        )

                    media_configs.append(
                        {
                            "name": channel,
                            "adstock_lmax": adstock_max,
                            "level": channel_level,
                        }
                    )

        st.session_state.media_channels = media_configs

    with col2:
        st.subheader("📊 Control Variables")

        potential_controls = [
            v
            for v in variables
            if v not in [kpi_name] + [m["name"] for m in media_configs]
        ]

        control_vars = st.multiselect(
            "Select Control Variables",
            options=potential_controls,
            default=[v for v in ["Price", "Distribution"] if v in potential_controls],
            help="Select variables that control for external factors",
        )

        control_configs = []
        if control_vars:
            st.markdown("**Control Settings:**")

            for control in control_vars:
                with st.expander(f"📊 {control}", expanded=False):
                    allow_neg = st.checkbox(
                        "Allow Negative Effect",
                        value=control.lower() in ["price", "competition"],
                        key=f"neg_{control}",
                        help="Check if this variable can have negative impact on KPI",
                    )
                    control_configs.append(
                        {"name": control, "allow_negative": allow_neg}
                    )

        st.session_state.control_variables = control_configs

        st.markdown("---")

        st.subheader("🗺️ Dimension Alignment")

        if kpi_dimensions != "National" and any(
            m.get("level") == "National" for m in media_configs
        ):
            st.info(
                "ℹ️ National media will be allocated to geographic/product dimensions"
            )

            allocation_method = st.selectbox(
                "Allocation Method",
                options=["Equal", "By Population", "By Sales", "Custom"],
                help="How to distribute national media to sub-national levels",
            )
            st.session_state.geo_allocation = (
                allocation_method.lower().replace(" ", "_").replace("by_", "")
            )

    st.markdown("---")

    # Build configuration button
    if st.button("🔧 Build Configuration", type="primary", use_container_width=True):
        try:
            config = build_mff_config()
            st.session_state.mff_config = config

            # Also build panel
            from mmm_framework import load_mff

            panel = load_mff(df, config)
            st.session_state.panel = panel

            st.success(f"✓ Configuration built successfully!")
            st.info(
                f"Panel: {panel.n_obs} observations, {panel.n_channels} channels, {panel.n_controls} controls"
            )

        except Exception as e:
            st.error(f"Error building configuration: {e}")
            import traceback

            st.code(traceback.format_exc())


def build_mff_config():
    """Build MFFConfig from session state."""
    from mmm_framework import (
        MFFConfigBuilder,
        KPIConfigBuilder,
        MediaChannelConfigBuilder,
        ControlVariableConfigBuilder,
        DimensionAlignmentConfigBuilder,
    )

    builder = MFFConfigBuilder()

    # KPI
    kpi_builder = KPIConfigBuilder(st.session_state.kpi_name)
    if st.session_state.kpi_dimensions == "National":
        kpi_builder.national()
    elif st.session_state.kpi_dimensions == "By Geography":
        kpi_builder.by_geo()
    elif st.session_state.kpi_dimensions == "By Product":
        kpi_builder.by_product()
    else:
        kpi_builder.by_geo_and_product()

    builder.with_kpi_builder(kpi_builder)

    # Media channels
    for media in st.session_state.media_channels:
        media_builder = MediaChannelConfigBuilder(media["name"])
        if media.get("level", "National") == "National":
            media_builder.national()
        else:
            media_builder.by_geo()
        media_builder.with_geometric_adstock(media.get("adstock_lmax", 8))
        media_builder.with_hill_saturation()
        builder.add_media_builder(media_builder)

    # Control variables
    for control in st.session_state.control_variables:
        control_builder = ControlVariableConfigBuilder(control["name"]).national()
        if control.get("allow_negative", False):
            control_builder.allow_negative()
        else:
            control_builder.positive_only()
        builder.add_control_builder(control_builder)

    # Alignment
    if st.session_state.kpi_dimensions != "National":
        align_builder = DimensionAlignmentConfigBuilder()
        alloc = st.session_state.geo_allocation
        if alloc == "equal":
            align_builder.geo_equal()
        elif alloc == "population":
            align_builder.geo_by_population()
        elif alloc == "sales":
            align_builder.geo_by_sales()
        builder.with_alignment_builder(align_builder)

    return builder.build()


# =============================================================================
# Tab 3: Model Settings & Job Submission
# =============================================================================


def render_model_tab():
    """Render the model settings and job submission tab."""
    st.header("🔬 Model Settings & Job Submission")

    if st.session_state.panel is None:
        st.warning("⚠️ Please configure the model first")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎛️ MCMC Settings")

        n_chains = st.slider(
            "Number of Chains",
            1,
            8,
            4,
            help="More chains = better convergence diagnostics",
        )
        n_draws = st.slider(
            "Draws per Chain",
            500,
            4000,
            1000,
            step=500,
            help="More draws = more precise estimates",
        )
        n_tune = st.slider(
            "Tuning Samples",
            500,
            2000,
            1000,
            step=500,
            help="Samples used for adaptation",
        )
        target_accept = st.slider(
            "Target Accept Rate",
            0.8,
            0.99,
            0.95,
            step=0.01,
            help="Higher = fewer divergences but slower",
        )

        st.markdown("---")

        st.subheader("📈 Trend")

        trend_type = st.selectbox(
            "Trend Type",
            options=[
                "None",
                "Linear",
                "Piecewise (Prophet-style)",
                "Spline",
                "Gaussian Process",
            ],
            index=1,
            help="How to model the underlying trend",
        )

        # Trend-specific settings
        trend_settings = {}

        if trend_type == "Piecewise (Prophet-style)":
            with st.expander("Piecewise Trend Settings", expanded=True):
                trend_settings["n_changepoints"] = st.slider(
                    "Number of Changepoints",
                    5,
                    30,
                    10,
                    help="More changepoints = more flexible trend",
                )
                trend_settings["changepoint_range"] = st.slider(
                    "Changepoint Range",
                    0.5,
                    0.95,
                    0.8,
                    0.05,
                    help="Fraction of time series where changepoints can occur",
                )
                trend_settings["changepoint_prior_scale"] = st.slider(
                    "Changepoint Prior Scale",
                    0.001,
                    0.5,
                    0.05,
                    0.001,
                    help="Smaller = smoother trend, larger = more flexible",
                )

        elif trend_type == "Spline":
            with st.expander("Spline Trend Settings", expanded=True):
                trend_settings["n_knots"] = st.slider(
                    "Number of Knots",
                    5,
                    30,
                    10,
                    help="More knots = more flexible trend",
                )
                trend_settings["spline_degree"] = st.selectbox(
                    "Spline Degree",
                    options=[1, 2, 3],
                    index=2,
                    format_func=lambda x: {
                        1: "Linear (1)",
                        2: "Quadratic (2)",
                        3: "Cubic (3)",
                    }[x],
                    help="Higher degree = smoother curves",
                )
                trend_settings["spline_prior_sigma"] = st.slider(
                    "Spline Prior Sigma",
                    0.1,
                    3.0,
                    1.0,
                    0.1,
                    help="Prior standard deviation for spline coefficients",
                )

        elif trend_type == "Gaussian Process":
            with st.expander("GP Trend Settings", expanded=True):
                trend_settings["gp_lengthscale_mu"] = st.slider(
                    "Lengthscale Prior Mean",
                    0.1,
                    0.7,
                    0.3,
                    0.05,
                    help="Expected smoothness (fraction of time series)",
                )
                trend_settings["gp_lengthscale_sigma"] = st.slider(
                    "Lengthscale Prior Sigma",
                    0.05,
                    0.5,
                    0.2,
                    0.05,
                    help="Uncertainty in smoothness",
                )
                trend_settings["gp_amplitude_sigma"] = st.slider(
                    "Amplitude Prior Sigma",
                    0.1,
                    1.5,
                    0.5,
                    0.1,
                    help="Prior on trend magnitude",
                )
                trend_settings["gp_n_basis"] = st.slider(
                    "Number of Basis Functions",
                    10,
                    40,
                    20,
                    help="More = better approximation but slower",
                )

        elif trend_type == "Linear":
            with st.expander("Linear Trend Settings", expanded=False):
                trend_settings["growth_prior_mu"] = st.slider(
                    "Growth Prior Mean",
                    -0.5,
                    0.5,
                    0.0,
                    0.05,
                    help="Expected growth rate direction",
                )
                trend_settings["growth_prior_sigma"] = st.slider(
                    "Growth Prior Sigma",
                    0.01,
                    0.5,
                    0.1,
                    0.01,
                    help="Uncertainty in growth rate",
                )

    with col2:
        st.subheader("🌊 Seasonality")

        yearly_order = st.slider(
            "Yearly Seasonality Order",
            0,
            5,
            2,
            help="Number of Fourier terms for yearly seasonality (0 = disabled)",
        )

        st.markdown("---")

        st.subheader("🏗️ Hierarchical Structure")

        if st.session_state.kpi_dimensions != "National":
            pool_geo = st.checkbox(
                "Pool Across Geographies",
                value=True,
                help="Share information between geographic units",
            )
        else:
            pool_geo = False

        st.markdown("---")

        st.subheader("⚡ Performance")

        use_numpyro = st.checkbox(
            "Use NumPyro (JAX)",
            value=False,
            help="Faster sampling with JAX backend (requires numpyro installed)",
        )

        st.markdown("---")

        st.subheader("📝 Job Info")

        job_name = st.text_input(
            "Job Name",
            value=f"MMM_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Name for this model fitting job",
        )

        job_description = st.text_area(
            "Description (optional)",
            placeholder="Notes about this model run...",
            height=80,
        )

    st.markdown("---")

    # Estimated time
    n_params = (
        len(st.session_state.media_channels) * 3
        + len(st.session_state.control_variables)
        + 5
    )
    est_time = (n_chains * (n_draws + n_tune) * n_params) / 5000
    st.info(f"⏱️ Estimated fitting time: {est_time:.0f} - {est_time*2:.0f} seconds")

    # Submit button
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "🚀 Submit Job (Background)", type="primary", use_container_width=True
        ):
            submit_background_job(
                job_name=job_name,
                job_description=job_description,
                n_chains=n_chains,
                n_draws=n_draws,
                n_tune=n_tune,
                target_accept=target_accept,
                trend_type=trend_type,
                trend_settings=trend_settings,
                yearly_order=yearly_order,
                pool_geo=pool_geo,
                use_numpyro=use_numpyro,
            )

    with col2:
        if st.button("⚡ Fit Inline (Blocking)", use_container_width=True):
            fit_model_inline(
                n_chains=n_chains,
                n_draws=n_draws,
                n_tune=n_tune,
                target_accept=target_accept,
                trend_type=trend_type,
                trend_settings=trend_settings,
                yearly_order=yearly_order,
                pool_geo=pool_geo,
                use_numpyro=use_numpyro,
            )


def submit_background_job(
    job_name,
    job_description,
    n_chains,
    n_draws,
    n_tune,
    target_accept,
    trend_type,
    trend_settings,
    yearly_order,
    pool_geo,
    use_numpyro,
):
    """Submit a model fitting job to run in background."""
    from mmm_framework import JobConfig, get_job_manager

    # Map trend type to internal name
    trend_map = {
        "None": "none",
        "Linear": "linear",
        "Piecewise (Prophet-style)": "piecewise",
        "Spline": "spline",
        "Gaussian Process": "gaussian_process",
    }

    config = JobConfig(
        name=job_name,
        description=job_description,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        target_accept=target_accept,
        use_numpyro=use_numpyro,
        trend_type=trend_map.get(trend_type, "linear"),
        trend_settings=trend_settings,
        yearly_order=yearly_order,
        pool_geo=pool_geo,
        random_seed=42,
    )

    try:
        manager = get_job_manager()
        job = manager.submit_job(st.session_state.panel, config)

        st.session_state.active_job_id = job.id
        st.success(f"✅ Job submitted: {job.display_name} (ID: {job.id[:8]}...)")
        st.info("Go to the **Jobs** tab to monitor progress")

    except Exception as e:
        st.error(f"Error submitting job: {e}")
        import traceback

        st.code(traceback.format_exc())


def fit_model_inline(
    n_chains,
    n_draws,
    n_tune,
    target_accept,
    trend_type,
    trend_settings,
    yearly_order,
    pool_geo,
    use_numpyro,
):
    """Fit the model inline (blocking) - for quick tests."""
    from mmm_framework import (
        BayesianMMM,
        TrendConfig,
        TrendType,
        ModelConfigBuilder,
        HierarchicalConfigBuilder,
        SeasonalityConfigBuilder,
        TrendConfigBuilder,
    )

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    try:
        status_text.text("Building model configuration...")
        progress_bar.progress(10, text="Building configuration...")

        model_builder = ModelConfigBuilder()

        if use_numpyro:
            model_builder.bayesian_numpyro()
        else:
            model_builder.bayesian_pymc()

        model_builder.with_chains(n_chains)
        model_builder.with_draws(n_draws)
        model_builder.with_tune(n_tune)
        model_builder.with_target_accept(target_accept)

        # Seasonality
        season_builder = SeasonalityConfigBuilder()
        if yearly_order > 0:
            season_builder.with_yearly(yearly_order)
        model_builder.with_seasonality_builder(season_builder)

        # Hierarchical
        if pool_geo:
            hier_builder = HierarchicalConfigBuilder().enabled().pool_across_geo()
            model_builder.with_hierarchical_builder(hier_builder)

        model_config = model_builder.build()

        # Build trend config
        trend_builder = TrendConfigBuilder()

        if trend_type == "None":
            trend_builder.none()
        elif trend_type == "Linear":
            trend_builder.linear()
            if "growth_prior_mu" in trend_settings:
                trend_builder.with_growth_prior(
                    mu=trend_settings.get("growth_prior_mu", 0.0),
                    sigma=trend_settings.get("growth_prior_sigma", 0.1),
                )
        elif trend_type == "Piecewise (Prophet-style)":
            trend_builder.piecewise()
            for key, setter in [
                ("n_changepoints", trend_builder.with_n_changepoints),
                ("changepoint_range", trend_builder.with_changepoint_range),
                ("changepoint_prior_scale", trend_builder.with_changepoint_prior_scale),
            ]:
                if key in trend_settings:
                    setter(trend_settings[key])
        elif trend_type == "Spline":
            trend_builder.spline()
            for key, setter in [
                ("n_knots", trend_builder.with_n_knots),
                ("spline_degree", trend_builder.with_spline_degree),
                ("spline_prior_sigma", trend_builder.with_spline_prior_sigma),
            ]:
                if key in trend_settings:
                    setter(trend_settings[key])
        elif trend_type == "Gaussian Process":
            trend_builder.gaussian_process()
            if "gp_lengthscale_mu" in trend_settings:
                trend_builder.with_gp_lengthscale(
                    mu=trend_settings.get("gp_lengthscale_mu", 0.3),
                    sigma=trend_settings.get("gp_lengthscale_sigma", 0.2),
                )
            if "gp_amplitude_sigma" in trend_settings:
                trend_builder.with_gp_amplitude(trend_settings["gp_amplitude_sigma"])
            if "gp_n_basis" in trend_settings:
                trend_builder.with_gp_n_basis(trend_settings["gp_n_basis"])

        trend_config = trend_builder.build()

        progress_bar.progress(20, text="Building model...")
        status_text.text("Building PyMC model...")

        # Create model
        mmm = BayesianMMM(st.session_state.panel, model_config, trend_config)

        progress_bar.progress(30, text="Fitting model (this may take a few minutes)...")
        status_text.text(f"Sampling {n_chains} chains × {n_draws} draws...")

        # Fit model
        results = mmm.fit(random_seed=42)

        progress_bar.progress(100, text="Complete!")
        status_text.text("Model fitted successfully!")

        st.session_state.loaded_mmm = mmm
        st.session_state.loaded_results = results
        st.session_state.loaded_contributions = None
        st.session_state.component_decomposition = None

        # Show quick diagnostics
        st.success("✓ Model fitted successfully!")

        col1, col2, col3 = st.columns(3)
        with col1:
            div_count = results.diagnostics["divergences"]
            if div_count == 0:
                st.metric(
                    "Divergences", div_count, delta="Perfect", delta_color="normal"
                )
            elif div_count < n_chains * n_draws * 0.01:
                st.metric(
                    "Divergences", div_count, delta="Acceptable", delta_color="normal"
                )
            else:
                st.metric("Divergences", div_count, delta="High", delta_color="inverse")

        with col2:
            rhat = results.diagnostics["rhat_max"]
            if rhat < 1.01:
                st.metric(
                    "R-hat Max", f"{rhat:.4f}", delta="Good", delta_color="normal"
                )
            else:
                st.metric(
                    "R-hat Max", f"{rhat:.4f}", delta="Check", delta_color="inverse"
                )

        with col3:
            ess = results.diagnostics["ess_bulk_min"]
            if ess > 400:
                st.metric("ESS Min", f"{ess:.0f}", delta="Good", delta_color="normal")
            else:
                st.metric("ESS Min", f"{ess:.0f}", delta="Low", delta_color="inverse")

    except Exception as e:
        progress_bar.progress(0, text="Error")
        st.error(f"Error fitting model: {e}")
        import traceback

        st.code(traceback.format_exc())


# =============================================================================
# Tab 4: Jobs
# =============================================================================


def render_jobs_tab():
    """Render the job management tab."""
    st.header("📋 Job Management")

    try:
        from mmm_framework import get_job_manager, JobStatus

        manager = get_job_manager()
    except Exception as e:
        st.error(f"Error initializing job manager: {e}")
        return

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.last_job_refresh = time.time()
            st.rerun()

    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if auto_refresh:
            time.sleep(2)
            st.rerun()

    with col3:
        st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("---")

    # Get all jobs
    all_jobs = manager.list_jobs(limit=50)
    active_jobs = [j for j in all_jobs if j.is_active]
    completed_jobs = [j for j in all_jobs if j.status == JobStatus.COMPLETED]
    failed_jobs = [
        j for j in all_jobs if j.status in [JobStatus.FAILED, JobStatus.CANCELLED]
    ]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", len(all_jobs))
    with col2:
        st.metric("Active", len(active_jobs), delta="🔄" if active_jobs else None)
    with col3:
        st.metric(
            "Completed", len(completed_jobs), delta="✅" if completed_jobs else None
        )
    with col4:
        st.metric(
            "Failed",
            len(failed_jobs),
            delta="❌" if failed_jobs else None,
            delta_color="inverse",
        )

    st.markdown("---")

    # Tabs for different job states
    job_tabs = st.tabs(
        ["🔄 Active Jobs", "✅ Completed", "❌ Failed/Cancelled", "📋 All Jobs"]
    )

    with job_tabs[0]:
        render_job_list(active_jobs, manager, show_progress=True, key_prefix="active")

    with job_tabs[1]:
        render_job_list(
            completed_jobs, manager, allow_load=True, key_prefix="completed"
        )

    with job_tabs[2]:
        render_job_list(failed_jobs, manager, show_error=True, key_prefix="failed")

    with job_tabs[3]:
        render_job_list(all_jobs, manager, show_all_columns=True, key_prefix="all")


def render_job_list(
    jobs,
    manager,
    show_progress=False,
    allow_load=False,
    show_error=False,
    show_all_columns=False,
    key_prefix="",
):
    """Render a list of jobs."""
    if not jobs:
        st.info("No jobs in this category")
        return

    for job in jobs:
        with st.container():
            render_job_card(
                job, manager, show_progress, allow_load, show_error, key_prefix
            )
            st.markdown("---")


def render_job_card(
    job, manager, show_progress=False, allow_load=False, show_error=False, key_prefix=""
):
    """Render a single job card."""
    from mmm_framework import JobStatus

    status_icon = get_status_icon(job.status)
    status_color = get_status_color(job.status)

    # Create unique key prefix for this job in this context
    unique_prefix = f"{key_prefix}_{job.id}" if key_prefix else job.id

    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        st.markdown(f"### {status_icon} {job.display_name}")
        st.caption(f"ID: `{job.id[:8]}...` | Created: {job.created_at[:19]}")

        if job.config.description:
            st.caption(job.config.description)

    with col2:
        st.markdown(f"**Status:** {job.status.value.title()}")

        if job.duration_seconds:
            st.caption(f"Duration: {format_duration(job.duration_seconds)}")

        if job.result and job.status == JobStatus.COMPLETED:
            st.caption(
                f"R²: {job.result.r_squared:.3f} | Divergences: {job.result.divergences}"
            )

    with col3:
        if job.status == JobStatus.RUNNING:
            if st.button("⏹️ Cancel", key=f"cancel_{unique_prefix}"):
                manager.cancel_job(job.id)
                st.rerun()

        elif job.status == JobStatus.COMPLETED and allow_load:
            if st.button("📂 Load", key=f"load_{unique_prefix}", type="primary"):
                load_job_results(job.id, manager)

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            if st.button("🗑️ Delete", key=f"delete_{unique_prefix}"):
                manager.delete_job(job.id)
                st.rerun()

    # Progress bar for running jobs
    if show_progress and job.status == JobStatus.RUNNING:
        progress = job.progress
        st.progress(
            progress.percent_complete / 100,
            text=f"{progress.stage}: {progress.message}",
        )

    # Error details for failed jobs
    if show_error and job.result and job.result.error_message:
        with st.expander("Error Details"):
            st.error(job.result.error_message)
            if job.result.error_traceback:
                st.code(job.result.error_traceback)


def load_job_results(job_id: str, manager):
    """Load results from a completed job into session state."""
    try:
        data = manager.load_job_results(job_id)

        if data is None:
            st.error("Could not load job results")
            return

        st.session_state.loaded_mmm = data.get("mmm")
        st.session_state.loaded_results = data.get("results")
        st.session_state.loaded_contributions = data.get("contributions")
        st.session_state.panel = data.get("panel")
        st.session_state.component_decomposition = None
        st.session_state.prior_samples = None
        st.session_state.active_job_id = job_id

        st.success(f"✅ Loaded results from job {job_id[:8]}...")
        st.info("Go to the **Results** tab to analyze")

    except Exception as e:
        st.error(f"Error loading results: {e}")
        import traceback

        st.code(traceback.format_exc())


# =============================================================================
# Tab 5: Results
# =============================================================================


@st.fragment
def render_results_tab():
    """Render the results and visualization tab."""
    st.header("📈 Results & Analysis")

    if st.session_state.loaded_results is None:
        st.warning(
            "⚠️ No results loaded. Either fit a model inline or load results from a completed job."
        )

        # Show recent completed jobs that can be loaded
        try:
            from mmm_framework import get_job_manager, JobStatus

            manager = get_job_manager()
            completed_jobs = manager.list_jobs(
                status_filter=[JobStatus.COMPLETED], limit=5
            )

            if completed_jobs:
                st.markdown("### Recent Completed Jobs")
                for job in completed_jobs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{job.display_name}** - {job.created_at[:10]}")
                    with col2:
                        if st.button("Load", key=f"quickload_{job.id}"):
                            load_job_results(job.id, manager)
                            st.rerun()
        except Exception:
            pass

        return

    results = st.session_state.loaded_results
    mmm = st.session_state.loaded_mmm

    if mmm is None:
        st.error("Model object not available. Please load from a job or fit inline.")
        return

    # Results tabs
    result_tabs = st.tabs(
        [
            "📊 Diagnostics",
            "🎯 Model Fit",
            "📉 Posteriors",
            "📈 Response Curves",
            "💰 Contributions",
            "📋 Summary",
        ]
    )

    with result_tabs[0]:
        render_diagnostics(results)

    with result_tabs[1]:
        render_model_fit(results, mmm)

    with result_tabs[2]:
        render_posteriors(results, mmm)

    with result_tabs[3]:
        render_response_curves(results, mmm)

    with result_tabs[4]:
        render_contributions(results, mmm)

    with result_tabs[5]:
        render_summary(results, mmm)


@st.fragment
def render_diagnostics(results):
    """Render model diagnostics."""
    st.subheader("🔍 MCMC Diagnostics")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Divergences", results.diagnostics["divergences"])
    with col2:
        st.metric("R-hat Max", f"{results.diagnostics['rhat_max']:.4f}")
    with col3:
        st.metric("ESS Bulk Min", f"{results.diagnostics['ess_bulk_min']:.0f}")
    with col4:
        total_samples = (
            results.trace.posterior.dims["chain"] * results.trace.posterior.dims["draw"]
        )
        st.metric("Total Samples", f"{total_samples:,}")

    # Interpretation
    st.markdown("---")

    div_pct = results.diagnostics["divergences"] / total_samples * 100
    if div_pct == 0:
        st.success("✓ No divergences - excellent convergence!")
    elif div_pct < 1:
        st.info(
            f"ℹ️ {div_pct:.2f}% divergences - acceptable, but consider increasing target_accept"
        )
    else:
        st.warning(
            f"⚠️ {div_pct:.2f}% divergences - consider reparameterization or more tuning"
        )

    if results.diagnostics["rhat_max"] < 1.01:
        st.success("✓ All R-hat values < 1.01 - chains have converged")
    else:
        st.warning("⚠️ Some R-hat values > 1.01 - chains may not have converged")


@st.fragment
def render_model_fit(results, mmm):
    """Render posterior predictive model fit."""
    st.subheader("🎯 Model Fit")

    panel = st.session_state.panel

    # Get predictions
    with st.spinner("Computing predictions..."):
        try:
            pred_results = mmm.predict(
                return_original_scale=True, hdi_prob=0.94, random_seed=42
            )
        except Exception as e:
            st.error(f"Error computing predictions: {e}")
            return

    # Build plot data
    periods = list(panel.coords.periods)

    if mmm.has_geo or mmm.has_product:
        y_obs_by_period = []
        y_pred_by_period = []
        y_hdi_low_by_period = []
        y_hdi_high_by_period = []

        for t, period in enumerate(periods):
            mask = mmm.time_idx == t
            y_obs_by_period.append(mmm.y_raw[mask].sum())
            y_pred_by_period.append(pred_results.y_pred_mean[mask].sum())
            y_hdi_low_by_period.append(pred_results.y_pred_hdi_low[mask].sum())
            y_hdi_high_by_period.append(pred_results.y_pred_hdi_high[mask].sum())

        plot_df = pd.DataFrame(
            {
                "Period": periods,
                "y_obs": y_obs_by_period,
                "y_pred_mean": y_pred_by_period,
                "y_pred_hdi_low": y_hdi_low_by_period,
                "y_pred_hdi_high": y_hdi_high_by_period,
            }
        )
    else:
        plot_df = pd.DataFrame(
            {
                "Period": periods[: len(pred_results.y_pred_mean)],
                "y_obs": mmm.y_raw,
                "y_pred_mean": pred_results.y_pred_mean,
                "y_pred_hdi_low": pred_results.y_pred_hdi_low,
                "y_pred_hdi_high": pred_results.y_pred_hdi_high,
            }
        )

    # Plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(plot_df["Period"]) + list(plot_df["Period"])[::-1],
            y=list(plot_df["y_pred_hdi_high"]) + list(plot_df["y_pred_hdi_low"])[::-1],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="94% HDI",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Period"],
            y=plot_df["y_pred_mean"],
            mode="lines",
            name="Predicted",
            line=dict(color="rgb(99, 110, 250)", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Period"],
            y=plot_df["y_obs"],
            mode="markers",
            name="Observed",
            marker=dict(color="rgb(239, 85, 59)", size=6),
        )
    )

    fig.update_layout(
        title="Posterior Predictive Fit",
        xaxis_title="Period",
        yaxis_title=mmm.mff_config.kpi.name,
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Fit statistics
    y_obs = plot_df["y_obs"].values
    y_pred = plot_df["y_pred_mean"].values

    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    mape = np.mean(np.abs((y_obs - y_pred) / (y_obs + 1e-8))) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{r_squared:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:,.1f}")
    with col3:
        st.metric("MAPE", f"{mape:.1f}%")


@st.fragment
def render_posteriors(results, mmm):
    """Render posterior distributions."""
    st.subheader("📉 Posterior Distributions")

    channel_names = mmm.channel_names
    beta_vars = [f"beta_{ch}" for ch in channel_names]

    posterior = results.trace.posterior

    st.markdown("### Media Coefficients (β)")

    beta_data = []
    for var in beta_vars:
        if var in posterior:
            samples = posterior[var].values.flatten()
            beta_data.append(
                {
                    "Channel": var.replace("beta_", ""),
                    "Mean": samples.mean(),
                    "Std": samples.std(),
                    "HDI 3%": np.percentile(samples, 3),
                    "HDI 97%": np.percentile(samples, 97),
                    "samples": samples,
                }
            )

    if beta_data:
        fig = go.Figure()

        for i, d in enumerate(beta_data):
            fig.add_trace(
                go.Box(
                    x=d["samples"],
                    name=d["Channel"],
                    orientation="h",
                    boxpoints=False,
                    marker_color=px.colors.qualitative.Set2[
                        i % len(px.colors.qualitative.Set2)
                    ],
                )
            )

        fig.update_layout(
            title="Media Coefficient Posteriors",
            xaxis_title="Coefficient Value",
            height=300 + len(beta_data) * 50,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        summary_df = pd.DataFrame(
            [{k: v for k, v in d.items() if k != "samples"} for d in beta_data]
        )
        st.dataframe(summary_df.round(4), use_container_width=True)


@st.fragment
def render_response_curves(results, mmm):
    """Render response curves."""
    st.subheader("📈 Response Curves")

    channel_names = mmm.channel_names
    posterior = results.trace.posterior
    panel = st.session_state.panel

    X_media_raw = panel.X_media.values

    response_data = []

    for c, channel in enumerate(channel_names):
        sat_lam_var = f"sat_lam_{channel}"
        beta_var = f"beta_{channel}"

        if sat_lam_var not in posterior or beta_var not in posterior:
            continue

        sat_lam_samples = posterior[sat_lam_var].values.flatten()
        beta_samples = posterior[beta_var].values.flatten()

        spend_raw = X_media_raw[:, c]
        spend_max = spend_raw.max()

        n_points = 100
        x_original = np.linspace(0, spend_max * 1.2, n_points)
        x_scaled = x_original / (spend_max + 1e-8)

        n_samples = min(500, len(sat_lam_samples))
        idx = np.random.choice(len(sat_lam_samples), n_samples, replace=False)

        curves = np.zeros((n_samples, n_points))
        for i, j in enumerate(idx):
            saturated = 1 - np.exp(-sat_lam_samples[j] * x_scaled)
            curves[i, :] = beta_samples[j] * saturated

        curves_original = curves * mmm.y_std

        response_data.append(
            {
                "channel": channel,
                "x": x_original,
                "mean": curves_original.mean(axis=0),
                "hdi_low": np.percentile(curves_original, 3, axis=0),
                "hdi_high": np.percentile(curves_original, 97, axis=0),
                "spend_max": spend_max,
            }
        )

    if not response_data:
        st.warning("No response curve data available")
        return

    colors = px.colors.qualitative.Set2

    fig = go.Figure()

    for i, data in enumerate(response_data):
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([data["x"], data["x"][::-1]]),
                y=np.concatenate([data["hdi_high"], data["hdi_low"][::-1]]),
                fill="toself",
                fillcolor=rgb_to_rgba(color, 0.2),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["mean"],
                mode="lines",
                name=data["channel"],
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title="Response Curves by Channel",
        xaxis_title="Media Spend",
        yaxis_title="Contribution to Sales",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_contributions(results, mmm):
    """Render channel contributions."""
    st.subheader("💰 Channel Contributions")

    panel = st.session_state.panel
    channel_names = mmm.channel_names

    # Check if we have pre-computed contributions
    if st.session_state.loaded_contributions is not None:
        contrib = st.session_state.loaded_contributions
    else:
        if st.button("🔄 Compute Contributions", type="primary"):
            with st.spinner("Computing counterfactual contributions..."):
                try:
                    contrib = mmm.compute_counterfactual_contributions(
                        compute_uncertainty=True, random_seed=42
                    )
                    st.session_state.loaded_contributions = contrib
                except Exception as e:
                    st.error(f"Error computing contributions: {e}")
                    return
        else:
            st.info("Click the button above to compute counterfactual contributions")
            return
        contrib = st.session_state.loaded_contributions

    if contrib is None:
        return

    # Pie chart
    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = px.pie(
            values=contrib.total_contributions.values,
            names=contrib.total_contributions.index,
            title="Share of Total Media Contribution",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        summary_df = contrib.summary()
        summary_df["Total Contribution"] = summary_df["Total Contribution"].apply(
            lambda x: f"{x:,.0f}"
        )
        summary_df["Contribution %"] = summary_df["Contribution %"].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ROAS
    st.markdown("### ROAS Analysis")

    X_media_raw = panel.X_media.values

    roas_data = []
    for i, ch in enumerate(channel_names):
        total_spend = X_media_raw[:, i].sum()
        total_contrib = contrib.total_contributions[ch]
        roas = total_contrib / total_spend if total_spend > 0 else 0

        roas_data.append(
            {
                "Channel": ch,
                "Total Spend": f"${total_spend:,.0f}",
                "Total Contribution": f"{total_contrib:,.0f}",
                "ROAS": f"{roas:.3f}",
            }
        )

    st.dataframe(pd.DataFrame(roas_data), use_container_width=True, hide_index=True)


@st.fragment
def render_summary(results, mmm):
    """Render model summary."""
    st.subheader("📋 Full Model Summary")

    summary = results.summary()

    key_params = [
        p
        for p in summary.index
        if any(x in p for x in ["beta", "sigma", "adstock", "sat_lam"])
    ]

    if key_params:
        st.markdown("### Key Parameters")
        st.dataframe(summary.loc[key_params].round(4), use_container_width=True)

    st.markdown("### Full Parameter Summary")
    st.dataframe(summary.round(4), use_container_width=True)

    st.markdown("---")
    st.markdown("### Export Results")

    col1, col2 = st.columns(2)

    with col1:
        csv_summary = summary.to_csv()
        st.download_button(
            "📥 Download Summary (CSV)",
            csv_summary,
            "mmm_summary.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        if st.session_state.loaded_contributions is not None:
            csv_contrib = (
                st.session_state.loaded_contributions.channel_contributions.to_csv()
            )
            st.download_button(
                "📥 Download Contributions (CSV)",
                csv_contrib,
                "mmm_contributions.csv",
                "text/csv",
                use_container_width=True,
            )


# =============================================================================
# Main App
# =============================================================================


def main():
    """Main application entry point."""
    render_sidebar()

    tabs = st.tabs(["📁 Data", "⚙️ Configure", "🔬 Model", "📋 Jobs", "📈 Results"])

    with tabs[0]:
        render_data_tab()

    with tabs[1]:
        render_config_tab()

    with tabs[2]:
        render_model_tab()

    with tabs[3]:
        render_jobs_tab()

    with tabs[4]:
        render_results_tab()


if __name__ == "__main__":
    main()

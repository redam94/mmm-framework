"""
MMM Framework - Streamlit Web Application (Enhanced v2)

A user-friendly interface for:
- Uploading MFF data
- Configuring model settings
- Running Bayesian MMM
- Visualizing results with counterfactual contributions
- Prior vs Posterior analysis
- Component decomposition (base, trend, seasonality, media, controls)
- Trend analysis
- Scenario planning and what-if analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
import hashlib
warnings.filterwarnings('ignore')

def rgb_to_rgba(rgb:str, alpha:float=1.0) -> str:
    """Convert RGB color to RGBA."""
    r, g, b = rgb.strip("rgb(").strip(")").split(",")
    return f"rgba({r},{g},{b},{alpha})"

# Set page config
st.set_page_config(
    page_title="Marketing Mix Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Caching Functions for Expensive Computations
# =============================================================================

@st.cache_data(ttl=3600)
def compute_posterior_predictive(_model, _trace, n_obs, y_std, y_mean, random_seed=42):
    """Cache posterior predictive samples."""
    import pymc as pm
    
    with _model:
        ppc = pm.sample_posterior_predictive(
            _trace,
            var_names=["y_obs"],
            random_seed=random_seed
        )
    
    y_pred_samples = ppc.posterior_predictive["y_obs"].values
    y_pred_samples = y_pred_samples.reshape(-1, n_obs)
    y_pred_original = y_pred_samples * y_std + y_mean
    
    return {
        'mean': y_pred_original.mean(axis=0),
        'std': y_pred_original.std(axis=0),
        'samples': y_pred_original
    }


@st.cache_data(ttl=3600)
def compute_prior_samples(_model, n_samples=1000):
    """Cache prior predictive samples."""
    import pymc as pm
    
    with _model:
        prior = pm.sample_prior_predictive(samples=n_samples)
    
    return prior


@st.cache_data(ttl=3600)
def compute_response_curves(
    posterior_sat_lam,
    posterior_beta,
    spend_max,
    y_std,
    n_points=100
):
    """Cache response curve computations."""
    x_original = np.linspace(0, spend_max * 1.2, n_points)
    x_scaled = x_original / (spend_max + 1e-8)
    
    n_samples = len(posterior_sat_lam)
    curves = np.zeros((n_samples, n_points))
    
    for i in range(n_samples):
        saturated = 1 - np.exp(-posterior_sat_lam[i] * x_scaled)
        curves[i, :] = posterior_beta[i] * saturated
    
    curves_original = curves * y_std
    
    return {
        'x': x_original,
        'mean': curves_original.mean(axis=0),
        'std': curves_original.std(axis=0),
        'samples': curves_original
    }


@st.cache_data(ttl=3600)
def aggregate_by_period(data_df, period_col, value_cols, agg_func='sum'):
    """Aggregate data by period (sum over geos/products)."""
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    
    return data_df.groupby(period_col)[value_cols].agg(agg_func).reset_index()


def get_dimension_info(mmm, panel):
    """Extract dimension information for aggregation."""
    info = {
        'has_geo': mmm.has_geo,
        'has_product': mmm.has_product,
        'n_geos': mmm.n_geos if mmm.has_geo else 1,
        'n_products': mmm.n_products if mmm.has_product else 1,
        'geo_names': mmm.geo_names if mmm.has_geo else ['National'],
        'product_names': mmm.product_names if mmm.has_product else ['All Products'],
        'periods': list(panel.coords.periods) if hasattr(panel.coords, 'periods') else list(range(mmm.n_periods)),
        'n_periods': mmm.n_periods
    }
    return info


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data': None,
        'data_filename': None,
        'panel': None,
        'mff_config': None,
        'model_config': None,
        'mmm': None,
        'results': None,
        'fitted': False,
        'contributions': None,
        'component_decomposition': None,
        'prior_samples': None,
        # Config state
        'kpi_name': 'Sales',
        'kpi_dimensions': 'National',
        'media_channels': [],
        'control_variables': [],
        'geo_allocation': 'equal',
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
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    return df


def detect_mff_columns(df):
    """Detect MFF column structure."""
    expected_cols = ['Period', 'Geography', 'Product', 'Campaign', 
                     'Outlet', 'Creative', 'VariableName', 'VariableValue']
    
    found = [col for col in expected_cols if col in df.columns]
    missing = [col for col in expected_cols if col not in df.columns]
    
    return found, missing


def get_variable_names(df):
    """Extract unique variable names from MFF data."""
    if 'VariableName' in df.columns:
        return sorted(df['VariableName'].unique().tolist())
    return []


def get_dimension_values(df, dim_col):
    """Get unique values for a dimension column."""
    if dim_col in df.columns:
        values = df[dim_col].dropna().unique()
        values = [v for v in values if v != '' and pd.notna(v)]
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
        
        records.append({
            "Period": date_str, "Geography": "", "Product": "",
            "Campaign": "", "Outlet": "", "Creative": "",
            "VariableName": "Sales", "VariableValue": max(0, sales)
        })
        
        # Media channels
        for channel, spend_range in [("TV", (30000, 80000)), 
                                      ("Digital", (20000, 50000)),
                                      ("Radio", (10000, 30000))]:
            records.append({
                "Period": date_str, "Geography": "", "Product": "",
                "Campaign": "", "Outlet": "", "Creative": "",
                "VariableName": channel, 
                "VariableValue": np.random.uniform(*spend_range)
            })
        
        # Controls
        records.append({
            "Period": date_str, "Geography": "", "Product": "",
            "Campaign": "", "Outlet": "", "Creative": "",
            "VariableName": "Price", "VariableValue": 100 + np.random.normal(0, 5)
        })
        records.append({
            "Period": date_str, "Geography": "", "Product": "",
            "Campaign": "", "Outlet": "", "Creative": "",
            "VariableName": "Distribution", "VariableValue": 0.8 + np.random.uniform(-0.1, 0.1)
        })
    
    return pd.DataFrame(records)


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
            if st.session_state.fitted:
                st.success("✓ Fitted")
            else:
                st.warning("○ Fitted")
        with col2:
            if st.session_state.results is not None:
                st.success("✓ Results")
            else:
                st.warning("○ Results")
        
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
        st.caption("v0.3.0 - Component Decomposition")


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
            type=['csv', 'xlsx', 'xls', 'parquet'],
            help="Upload a Master Flat File (MFF) with columns: Period, Geography, Product, Campaign, Outlet, Creative, VariableName, VariableValue"
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
        st.code("""
Period,Geography,Product,...,VariableName,VariableValue
2022-01-03,,,,...,Sales,1234.56
2022-01-03,,,,...,TV,50000.00
        """)
    
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
        
        if 'VariableName' in df.columns and 'VariableValue' in df.columns:
            var_summary = df.groupby('VariableName')['VariableValue'].agg(['count', 'mean', 'std', 'min', 'max'])
            var_summary = var_summary.round(2)
            st.dataframe(var_summary, use_container_width=True)
        
        # Dimension analysis
        st.subheader("Dimensions")
        
        dim_cols = ['Geography', 'Product', 'Campaign', 'Outlet', 'Creative']
        dim_info = []
        for col in dim_cols:
            values = get_dimension_values(df, col)
            dim_info.append({
                'Dimension': col,
                'Unique Values': len(values),
                'Values': ', '.join(values[:5]) + ('...' if len(values) > 5 else '') if values else '(empty)'
            })
        
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
    geographies = get_dimension_values(df, 'Geography')
    products = get_dimension_values(df, 'Product')
    
    # Configuration sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 KPI Configuration")
        
        kpi_name = st.selectbox(
            "KPI Variable",
            options=variables,
            index=variables.index('Sales') if 'Sales' in variables else 0,
            help="Select the dependent variable (e.g., Sales, Revenue)"
        )
        st.session_state.kpi_name = kpi_name
        
        kpi_dim_options = ['National']
        if geographies:
            kpi_dim_options.append('By Geography')
        if products:
            kpi_dim_options.append('By Product')
        if geographies and products:
            kpi_dim_options.append('By Geography & Product')
        
        kpi_dimensions = st.selectbox(
            "KPI Granularity",
            options=kpi_dim_options,
            help="At what level is the KPI measured?"
        )
        st.session_state.kpi_dimensions = kpi_dimensions
        
        st.markdown("---")
        
        st.subheader("📺 Media Channels")
        
        # Filter out KPI and common control names
        potential_media = [v for v in variables if v not in [kpi_name, 'Price', 'Distribution', 'Temperature', 'Promotion']]
        
        media_channels = st.multiselect(
            "Select Media Channels",
            options=potential_media,
            default=[v for v in ['TV', 'Digital', 'Radio', 'Social', 'Print'] if v in potential_media],
            help="Select variables that represent media spend"
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
                            min_value=1, max_value=12, value=8,
                            key=f"adstock_{channel}"
                        )
                    with col_b:
                        channel_level = st.selectbox(
                            "Data Level",
                            options=['National', 'By Geography'],
                            key=f"level_{channel}"
                        )
                    
                    media_configs.append({
                        'name': channel,
                        'adstock_lmax': adstock_max,
                        'level': channel_level
                    })
        
        st.session_state.media_channels = media_configs
    
    with col2:
        st.subheader("📊 Control Variables")
        
        potential_controls = [v for v in variables if v not in [kpi_name] + [m['name'] for m in media_configs]]
        
        control_vars = st.multiselect(
            "Select Control Variables",
            options=potential_controls,
            default=[v for v in ['Price', 'Distribution'] if v in potential_controls],
            help="Select variables that control for external factors"
        )
        
        control_configs = []
        if control_vars:
            st.markdown("**Control Settings:**")
            
            for control in control_vars:
                with st.expander(f"📊 {control}", expanded=False):
                    allow_neg = st.checkbox(
                        "Allow Negative Effect",
                        value=control.lower() in ['price', 'competition'],
                        key=f"neg_{control}",
                        help="Check if this variable can have negative impact on KPI"
                    )
                    control_configs.append({
                        'name': control,
                        'allow_negative': allow_neg
                    })
        
        st.session_state.control_variables = control_configs
        
        st.markdown("---")
        
        st.subheader("🗺️ Dimension Alignment")
        
        if kpi_dimensions != 'National' and any(m.get('level') == 'National' for m in media_configs):
            st.info("ℹ️ National media will be allocated to geographic/product dimensions")
            
            allocation_method = st.selectbox(
                "Allocation Method",
                options=['Equal', 'By Population', 'By Sales', 'Custom'],
                help="How to distribute national media to sub-national levels"
            )
            st.session_state.geo_allocation = allocation_method.lower().replace(' ', '_').replace('by_', '')
    
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
            st.info(f"Panel: {panel.n_obs} observations, {panel.n_channels} channels, {panel.n_controls} controls")
            
        except Exception as e:
            st.error(f"Error building configuration: {e}")
            import traceback
            st.code(traceback.format_exc())


def build_mff_config():
    """Build MFFConfig from session state."""
    from mmm_framework import (
        MFFConfigBuilder, KPIConfigBuilder, MediaChannelConfigBuilder,
        ControlVariableConfigBuilder, DimensionAlignmentConfigBuilder
    )
    
    builder = MFFConfigBuilder()
    
    # KPI
    kpi_builder = KPIConfigBuilder(st.session_state.kpi_name)
    if st.session_state.kpi_dimensions == 'National':
        kpi_builder.national()
    elif st.session_state.kpi_dimensions == 'By Geography':
        kpi_builder.by_geo()
    elif st.session_state.kpi_dimensions == 'By Product':
        kpi_builder.by_product()
    else:
        kpi_builder.by_geo_and_product()
    
    builder.with_kpi_builder(kpi_builder)
    
    # Media channels
    for media in st.session_state.media_channels:
        media_builder = MediaChannelConfigBuilder(media['name'])
        if media.get('level', 'National') == 'National':
            media_builder.national()
        else:
            media_builder.by_geo()
        media_builder.with_geometric_adstock(media.get('adstock_lmax', 8))
        media_builder.with_hill_saturation()
        builder.add_media_builder(media_builder)
    
    # Control variables
    for control in st.session_state.control_variables:
        control_builder = ControlVariableConfigBuilder(control['name']).national()
        if control.get('allow_negative', False):
            control_builder.allow_negative()
        else:
            control_builder.positive_only()
        builder.add_control_builder(control_builder)
    
    # Alignment
    if st.session_state.kpi_dimensions != 'National':
        align_builder = DimensionAlignmentConfigBuilder()
        alloc = st.session_state.geo_allocation
        if alloc == 'equal':
            align_builder.geo_equal()
        elif alloc == 'population':
            align_builder.geo_by_population()
        elif alloc == 'sales':
            align_builder.geo_by_sales()
        builder.with_alignment_builder(align_builder)
    
    return builder.build()


# =============================================================================
# Tab 3: Model Settings
# =============================================================================

def render_model_tab():
    """Render the model settings and fitting tab."""
    st.header("🔬 Model Settings & Fitting")
    
    if st.session_state.panel is None:
        st.warning("⚠️ Please configure the model first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ MCMC Settings")
        
        n_chains = st.slider("Number of Chains", 1, 8, 4, help="More chains = better convergence diagnostics")
        n_draws = st.slider("Draws per Chain", 500, 4000, 1000, step=500, help="More draws = more precise estimates")
        n_tune = st.slider("Tuning Samples", 500, 2000, 1000, step=500, help="Samples used for adaptation")
        target_accept = st.slider("Target Accept Rate", 0.8, 0.99, 0.95, step=0.01, help="Higher = fewer divergences but slower")
        
        st.markdown("---")
        
        st.subheader("📈 Trend")
        
        trend_type = st.selectbox(
            "Trend Type",
            options=['None', 'Linear', 'Piecewise (Prophet-style)', 'Spline', 'Gaussian Process'],
            index=1,
            help="How to model the underlying trend"
        )
        
        # Trend-specific settings
        trend_settings = {}
        
        if trend_type == 'Piecewise (Prophet-style)':
            with st.expander("Piecewise Trend Settings", expanded=True):
                trend_settings['n_changepoints'] = st.slider(
                    "Number of Changepoints", 5, 30, 10,
                    help="More changepoints = more flexible trend"
                )
                trend_settings['changepoint_range'] = st.slider(
                    "Changepoint Range", 0.5, 0.95, 0.8, 0.05,
                    help="Fraction of time series where changepoints can occur"
                )
                trend_settings['changepoint_prior_scale'] = st.slider(
                    "Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001,
                    help="Smaller = smoother trend, larger = more flexible"
                )
        
        elif trend_type == 'Spline':
            with st.expander("Spline Trend Settings", expanded=True):
                trend_settings['n_knots'] = st.slider(
                    "Number of Knots", 5, 30, 10,
                    help="More knots = more flexible trend"
                )
                trend_settings['spline_degree'] = st.selectbox(
                    "Spline Degree",
                    options=[1, 2, 3],
                    index=2,
                    format_func=lambda x: {1: "Linear (1)", 2: "Quadratic (2)", 3: "Cubic (3)"}[x],
                    help="Higher degree = smoother curves"
                )
                trend_settings['spline_prior_sigma'] = st.slider(
                    "Spline Prior Sigma", 0.1, 3.0, 1.0, 0.1,
                    help="Prior standard deviation for spline coefficients"
                )
        
        elif trend_type == 'Gaussian Process':
            with st.expander("GP Trend Settings", expanded=True):
                trend_settings['gp_lengthscale_mu'] = st.slider(
                    "Lengthscale Prior Mean", 0.1, 0.7, 0.3, 0.05,
                    help="Expected smoothness (fraction of time series)"
                )
                trend_settings['gp_lengthscale_sigma'] = st.slider(
                    "Lengthscale Prior Sigma", 0.05, 0.5, 0.2, 0.05,
                    help="Uncertainty in smoothness"
                )
                trend_settings['gp_amplitude_sigma'] = st.slider(
                    "Amplitude Prior Sigma", 0.1, 1.5, 0.5, 0.1,
                    help="Prior on trend magnitude"
                )
                trend_settings['gp_n_basis'] = st.slider(
                    "Number of Basis Functions", 10, 40, 20,
                    help="More = better approximation but slower"
                )
        
        elif trend_type == 'Linear':
            with st.expander("Linear Trend Settings", expanded=False):
                trend_settings['growth_prior_mu'] = st.slider(
                    "Growth Prior Mean", -0.5, 0.5, 0.0, 0.05,
                    help="Expected growth rate direction"
                )
                trend_settings['growth_prior_sigma'] = st.slider(
                    "Growth Prior Sigma", 0.01, 0.5, 0.1, 0.01,
                    help="Uncertainty in growth rate"
                )
        
        st.session_state.trend_type = trend_type
        st.session_state.trend_settings = trend_settings
    
    with col2:
        st.subheader("🌊 Seasonality")
        
        yearly_order = st.slider(
            "Yearly Seasonality Order",
            0, 5, 2,
            help="Number of Fourier terms for yearly seasonality (0 = disabled)"
        )
        st.session_state.yearly_order = yearly_order
        
        st.markdown("---")
        
        st.subheader("🏗️ Hierarchical Structure")
        
        if st.session_state.kpi_dimensions != 'National':
            pool_geo = st.checkbox("Pool Across Geographies", value=True, help="Share information between geographic units")
        else:
            pool_geo = False
        
        st.markdown("---")
        
        st.subheader("⚡ Performance")
        
        use_numpyro = st.checkbox(
            "Use NumPyro (JAX)",
            value=False,
            help="Faster sampling with JAX backend (requires numpyro installed)"
        )
    
    st.markdown("---")
    
    # Estimated time
    n_params = len(st.session_state.media_channels) * 3 + len(st.session_state.control_variables) + 5
    est_time = (n_chains * (n_draws + n_tune) * n_params) / 5000
    st.info(f"⏱️ Estimated fitting time: {est_time:.0f} - {est_time*2:.0f} seconds")
    
    # Fit button
    if st.button("🚀 Fit Model", type="primary", use_container_width=True):
        fit_model(
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=target_accept,
            trend_type=trend_type,
            trend_settings=trend_settings,
            yearly_order=yearly_order,
            pool_geo=pool_geo,
            use_numpyro=use_numpyro
        )


def fit_model(n_chains, n_draws, n_tune, target_accept, trend_type, trend_settings, yearly_order, pool_geo, use_numpyro):
    """Fit the Bayesian MMM model."""
    from mmm_framework import (
        BayesianMMM, TrendConfig, TrendType,
        ModelConfigBuilder, HierarchicalConfigBuilder, SeasonalityConfigBuilder,
        TrendConfigBuilder
    )
    
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    try:
        # Build model config
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
        
        # Build trend config using TrendConfigBuilder
        trend_builder = TrendConfigBuilder()
        
        if trend_type == 'None':
            trend_builder.none()
        elif trend_type == 'Linear':
            trend_builder.linear()
            if 'growth_prior_mu' in trend_settings:
                trend_builder.with_growth_prior(
                    mu=trend_settings.get('growth_prior_mu', 0.0),
                    sigma=trend_settings.get('growth_prior_sigma', 0.1)
                )
        elif trend_type == 'Piecewise (Prophet-style)':
            trend_builder.piecewise()
            if 'n_changepoints' in trend_settings:
                trend_builder.with_n_changepoints(trend_settings['n_changepoints'])
            if 'changepoint_range' in trend_settings:
                trend_builder.with_changepoint_range(trend_settings['changepoint_range'])
            if 'changepoint_prior_scale' in trend_settings:
                trend_builder.with_changepoint_prior_scale(trend_settings['changepoint_prior_scale'])
        elif trend_type == 'Spline':
            trend_builder.spline()
            if 'n_knots' in trend_settings:
                trend_builder.with_n_knots(trend_settings['n_knots'])
            if 'spline_degree' in trend_settings:
                trend_builder.with_spline_degree(trend_settings['spline_degree'])
            if 'spline_prior_sigma' in trend_settings:
                trend_builder.with_spline_prior_sigma(trend_settings['spline_prior_sigma'])
        elif trend_type == 'Gaussian Process':
            trend_builder.gaussian_process()
            if 'gp_lengthscale_mu' in trend_settings:
                trend_builder.with_gp_lengthscale(
                    mu=trend_settings.get('gp_lengthscale_mu', 0.3),
                    sigma=trend_settings.get('gp_lengthscale_sigma', 0.2)
                )
            if 'gp_amplitude_sigma' in trend_settings:
                trend_builder.with_gp_amplitude(trend_settings['gp_amplitude_sigma'])
            if 'gp_n_basis' in trend_settings:
                trend_builder.with_gp_n_basis(trend_settings['gp_n_basis'])
        
        trend_config = trend_builder.build()
        
        progress_bar.progress(20, text="Building model...")
        status_text.text("Building PyMC model...")
        
        # Create model
        mmm = BayesianMMM(
            st.session_state.panel,
            model_config,
            trend_config
        )
        st.session_state.mmm = mmm
        
        progress_bar.progress(30, text="Fitting model (this may take a few minutes)...")
        status_text.text(f"Sampling {n_chains} chains × {n_draws} draws...")
        
        # Fit model
        results = mmm.fit(random_seed=42)
        
        progress_bar.progress(100, text="Complete!")
        status_text.text("Model fitted successfully!")
        
        st.session_state.results = results
        st.session_state.fitted = True
        st.session_state.contributions = None
        st.session_state.component_decomposition = None
        
        # Show quick diagnostics
        st.success("✓ Model fitted successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            div_count = results.diagnostics['divergences']
            if div_count == 0:
                st.metric("Divergences", div_count, delta="Perfect", delta_color="normal")
            elif div_count < n_chains * n_draws * 0.01:
                st.metric("Divergences", div_count, delta="Acceptable", delta_color="normal")
            else:
                st.metric("Divergences", div_count, delta="High", delta_color="inverse")
        
        with col2:
            rhat = results.diagnostics['rhat_max']
            if rhat < 1.01:
                st.metric("R-hat Max", f"{rhat:.4f}", delta="Good", delta_color="normal")
            else:
                st.metric("R-hat Max", f"{rhat:.4f}", delta="Check", delta_color="inverse")
        
        with col3:
            ess = results.diagnostics['ess_bulk_min']
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
# Tab 4: Results
# =============================================================================
@st.fragment
def render_results_tab():
    """Render the results and visualization tab."""
    st.header("📈 Results & Analysis")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please fit the model first")
        return
    
    results = st.session_state.results
    mmm = st.session_state.mmm
    
    # Determine which tabs to show based on model configuration
    tab_list = [
        "📊 Diagnostics", 
        "🎯 Model Fit",
        "🔄 Prior vs Posterior",
        "📉 Posteriors", 
        "📈 Response Curves", 
        "🧩 Component Breakdown",
        "💰 Channel Contributions",
    ]
    
    # Add Trend Analysis tab if trend/seasonality is enabled
    trend_type = st.session_state.get('trend_type', 'None')
    yearly_order = st.session_state.get('yearly_order', 0)
    if trend_type != 'None' or yearly_order > 0:
        tab_list.append("📈 Trend Analysis")
    
    tab_list.extend([
        "🔮 Scenario Planning",
        "📋 Summary"
    ])
    
    result_tabs = st.tabs(tab_list)
    
    tab_idx = 0
    
    with result_tabs[tab_idx]:
        render_diagnostics(results)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_model_fit(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_prior_vs_posterior(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_posteriors(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_response_curves(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_component_breakdown(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_contributions(results, mmm)
    tab_idx += 1
    
    # Trend Analysis tab (conditional)
    if trend_type != 'None' or yearly_order > 0:
        with result_tabs[tab_idx]:
            render_trend_analysis(results, mmm)
        tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_scenario_planning(results, mmm)
    tab_idx += 1
    
    with result_tabs[tab_idx]:
        render_summary(results, mmm)


@st.fragment
def render_diagnostics(results):
    """Render model diagnostics."""
    st.subheader("🔍 MCMC Diagnostics")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Divergences", results.diagnostics['divergences'])
    with col2:
        st.metric("R-hat Max", f"{results.diagnostics['rhat_max']:.4f}")
    with col3:
        st.metric("ESS Bulk Min", f"{results.diagnostics['ess_bulk_min']:.0f}")
    with col4:
        total_samples = results.trace.posterior.dims['chain'] * results.trace.posterior.dims['draw']
        st.metric("Total Samples", f"{total_samples:,}")
    
    # Interpretation
    st.markdown("---")
    
    div_pct = results.diagnostics['divergences'] / total_samples * 100
    if div_pct == 0:
        st.success("✓ No divergences - excellent convergence!")
    elif div_pct < 1:
        st.info(f"ℹ️ {div_pct:.2f}% divergences - acceptable, but consider increasing target_accept")
    else:
        st.warning(f"⚠️ {div_pct:.2f}% divergences - consider reparameterization or more tuning")
    
    if results.diagnostics['rhat_max'] < 1.01:
        st.success("✓ All R-hat values < 1.01 - chains have converged")
    else:
        st.warning("⚠️ Some R-hat values > 1.01 - chains may not have converged")
    
    # R-hat by parameter
    st.markdown("---")
    st.subheader("R-hat by Parameter")
    
    import arviz as az
    
    try:
        rhat_data = az.rhat(results.trace)
        
        rhat_records = []
        for var_name in rhat_data.data_vars:
            values = rhat_data[var_name].values
            if np.ndim(values) == 0:
                rhat_records.append({'Parameter': var_name, 'R-hat': float(values)})
            else:
                flat_values = np.atleast_1d(values).flatten()
                for i, val in enumerate(flat_values):
                    param_name = f"{var_name}[{i}]" if len(flat_values) > 1 else var_name
                    rhat_records.append({'Parameter': param_name, 'R-hat': float(val)})
        
        rhat_df = pd.DataFrame(rhat_records)
        rhat_df = rhat_df.sort_values('R-hat', ascending=False)
        
        fig = px.bar(
            rhat_df.head(15),
            x='Parameter',
            y='R-hat',
            color='R-hat',
            color_continuous_scale=['green', 'yellow', 'red'],
            range_color=[1.0, 1.05]
        )
        fig.add_hline(y=1.01, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate R-hat plot: {e}")

@st.fragment
def render_model_fit(results, mmm):
    """Render posterior predictive model fit."""
    st.subheader("🎯 Model Fit")
    
    st.markdown("""
    This plot shows the observed data against the model's posterior predictions.
    The shaded band represents the prediction uncertainty (HDI).
    """)
    
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    
    # Settings
    col_settings1, col_settings2 = st.columns([1, 1])
    
    with col_settings1:
        hdi_prob = st.slider("HDI Width", 0.5, 0.99, 0.94, 0.01, key="fit_hdi")
    
    with col_settings2:
        show_residuals = st.checkbox("Show Residuals", value=False, key="fit_residuals")
    
    # Get predictions using the model's predict method
    with st.spinner("Computing posterior predictions..."):
        try:
            pred_results = mmm.predict(
                return_original_scale=True,
                hdi_prob=hdi_prob,
                random_seed=42
            )
        except Exception as e:
            st.error(f"Error computing predictions: {e}")
            return
    
    # Build dataframe with predictions and observed
    periods = dim_info['periods']
    
    # For panel data, aggregate by period
    if dim_info['has_geo'] or dim_info['has_product']:
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
        
        plot_df = pd.DataFrame({
            'Period': periods,
            'y_obs': y_obs_by_period,
            'y_pred_mean': y_pred_by_period,
            'y_pred_hdi_low': y_hdi_low_by_period,
            'y_pred_hdi_high': y_hdi_high_by_period
        })
    else:
        plot_df = pd.DataFrame({
            'Period': periods[:len(pred_results.y_pred_mean)],
            'y_obs': mmm.y_raw,
            'y_pred_mean': pred_results.y_pred_mean,
            'y_pred_hdi_low': pred_results.y_pred_hdi_low,
            'y_pred_hdi_high': pred_results.y_pred_hdi_high
        })
    
    # Plot
    fig = go.Figure()
    
    # HDI band
    fig.add_trace(go.Scatter(
        x=list(plot_df['Period']) + list(plot_df['Period'])[::-1],
        y=list(plot_df['y_pred_hdi_high']) + list(plot_df['y_pred_hdi_low'])[::-1],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'{int(hdi_prob*100)}% HDI',
        hoverinfo='skip'
    ))
    
    # Predicted mean
    fig.add_trace(go.Scatter(
        x=plot_df['Period'],
        y=plot_df['y_pred_mean'],
        mode='lines',
        name='Predicted (Mean)',
        line=dict(color='rgb(99, 110, 250)', width=2)
    ))
    
    # Observed
    fig.add_trace(go.Scatter(
        x=plot_df['Period'],
        y=plot_df['y_obs'],
        mode='markers',
        name='Observed',
        marker=dict(color='rgb(239, 85, 59)', size=6)
    ))
    
    fig.update_layout(
        title="Posterior Predictive Fit",
        xaxis_title="Period",
        yaxis_title=mmm.mff_config.kpi.name,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Fit statistics
    y_obs = plot_df['y_obs'].values
    y_pred = plot_df['y_pred_mean'].values
    
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
    
    # Residuals
    if show_residuals:
        residuals = y_obs - y_pred
        
        fig_resid = make_subplots(rows=1, cols=2, 
                                   subplot_titles=["Residuals Over Time", "Residual Distribution"])
        
        fig_resid.add_trace(go.Scatter(
            x=list(plot_df['Period']),
            y=residuals,
            mode='markers+lines',
            marker=dict(size=4),
            line=dict(width=1),
            name='Residuals'
        ), row=1, col=1)
        fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        fig_resid.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Distribution'
        ), row=1, col=2)
        
        fig_resid.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_resid, use_container_width=True)

@st.fragment
def render_prior_vs_posterior(results, mmm):
    """Render prior vs posterior comparison plots."""
    st.subheader("🔄 Prior vs Posterior")
    
    st.markdown("""
    Compare how the data updated our prior beliefs. 
    Significant differences indicate the data is informative about the parameter.
    """)
    
    posterior = results.trace.posterior
    channel_names = mmm.channel_names
    
    # Sample from prior if not already done
    if st.session_state.prior_samples is None:
        with st.spinner("Sampling from prior distribution..."):
            try:
                st.session_state.prior_samples = mmm.sample_prior_predictive(samples=1000)
            except Exception as e:
                st.error(f"Error sampling prior: {e}")
                return
    
    prior = st.session_state.prior_samples
    
    # Parameter selection
    param_categories = {
        'Media Coefficients (β)': [f"beta_{ch}" for ch in channel_names],
        'Adstock Parameters': [f"adstock_{ch}" for ch in channel_names],
        'Saturation Parameters (λ)': [f"sat_lam_{ch}" for ch in channel_names],
        'Other': ['intercept', 'sigma']
    }
    
    # Add trend parameters if present
    trend_params = []
    for var in posterior.data_vars:
        if 'trend' in var or 'spline' in var or 'gp' in var:
            trend_params.append(var)
    if trend_params:
        param_categories['Trend Parameters'] = trend_params
    
    # Add seasonality if present
    season_params = [v for v in posterior.data_vars if 'season' in v]
    if season_params:
        param_categories['Seasonality'] = season_params
    
    selected_category = st.selectbox(
        "Parameter Category",
        options=list(param_categories.keys())
    )
    
    params_to_plot = param_categories[selected_category]
    
    # Filter to available parameters
    available_params = [p for p in params_to_plot if p in posterior and p in prior.prior]
    
    if not available_params:
        st.info("No parameters available for comparison in this category.")
        return
    
    # Create comparison plots
    n_params = len(available_params)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[p.replace('_', ' ').title() for p in available_params]
    )
    
    for idx, param in enumerate(available_params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Get prior samples
        prior_samples = prior.prior[param].values.flatten()
        
        # Get posterior samples
        posterior_samples = posterior[param].values.flatten()
        
        # Add prior histogram
        fig.add_trace(
            go.Histogram(
                x=prior_samples,
                name='Prior',
                opacity=0.5,
                marker_color='blue',
                nbinsx=50,
                histnorm='probability density',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Add posterior histogram
        fig.add_trace(
            go.Histogram(
                x=posterior_samples,
                name='Posterior',
                opacity=0.5,
                marker_color='red',
                nbinsx=50,
                histnorm='probability density',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=300 * n_rows,
        barmode='overlay',
        title="Prior (blue) vs Posterior (red) Distributions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    st.markdown("### Prior vs Posterior Summary")
    
    summary_data = []
    for param in available_params:
        prior_samples = prior.prior[param].values.flatten()
        posterior_samples = posterior[param].values.flatten()
        
        summary_data.append({
            'Parameter': param.replace('_', ' ').title(),
            'Prior Mean': prior_samples.mean(),
            'Prior Std': prior_samples.std(),
            'Posterior Mean': posterior_samples.mean(),
            'Posterior Std': posterior_samples.std(),
            'Std Reduction (%)': (1 - posterior_samples.std() / (prior_samples.std() + 1e-8)) * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **Interpretation:**
    - **Std Reduction %**: Higher values indicate the data is more informative for that parameter.
    - If prior and posterior are similar, the data provides little information about that parameter.
    - Large shifts from prior to posterior mean indicate strong evidence from the data.
    """)


@st.fragment
def render_posteriors(results, mmm):
    """Render posterior distributions."""
    st.subheader("📉 Posterior Distributions")
    
    # Get channel betas
    channel_names = mmm.channel_names
    beta_vars = [f"beta_{ch}" for ch in channel_names]
    
    # Extract posterior samples
    posterior = results.trace.posterior
    
    # Beta coefficients
    st.markdown("### Media Coefficients (β)")
    
    beta_data = []
    for var in beta_vars:
        if var in posterior:
            samples = posterior[var].values.flatten()
            beta_data.append({
                'Channel': var.replace('beta_', ''),
                'Mean': samples.mean(),
                'Std': samples.std(),
                'HDI 3%': np.percentile(samples, 3),
                'HDI 97%': np.percentile(samples, 97),
                'samples': samples
            })
    
    if beta_data:
        # Forest plot
        fig = go.Figure()
        
        for i, d in enumerate(beta_data):
            fig.add_trace(go.Box(
                x=d['samples'],
                name=d['Channel'],
                orientation='h',
                boxpoints=False,
                marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            ))
        
        fig.update_layout(
            title="Media Coefficient Posteriors",
            xaxis_title="Coefficient Value (standardized)",
            height=300 + len(beta_data) * 50,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        summary_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'samples'} for d in beta_data])
        summary_df = summary_df.round(4)
        st.dataframe(summary_df, use_container_width=True)
    
    # Adstock parameters
    st.markdown("### Adstock Parameters")
    
    adstock_data = []
    for ch in channel_names:
        var = f"adstock_{ch}"
        if var in posterior:
            samples = posterior[var].values.flatten()
            adstock_data.append({
                'Channel': ch,
                'Mean': samples.mean(),
                'Std': samples.std(),
                'samples': samples
            })
    
    if adstock_data:
        fig = go.Figure()
        for d in adstock_data:
            fig.add_trace(go.Histogram(
                x=d['samples'],
                name=d['Channel'],
                opacity=0.7,
                nbinsx=50
            ))
        fig.update_layout(
            title="Adstock Mix Parameter Posteriors",
            xaxis_title="Adstock Mix (0=instant, 1=high carryover)",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Saturation parameters
    st.markdown("### Saturation Parameters")
    
    sat_data = []
    for ch in channel_names:
        var = f"sat_lam_{ch}"
        if var in posterior:
            samples = posterior[var].values.flatten()
            sat_data.append({
                'Channel': ch,
                'Mean': samples.mean(),
                'Std': samples.std(),
                'samples': samples
            })
    
    if sat_data:
        fig = go.Figure()
        for d in sat_data:
            fig.add_trace(go.Histogram(
                x=d['samples'],
                name=d['Channel'],
                opacity=0.7,
                nbinsx=50
            ))
        fig.update_layout(
            title="Saturation Rate (λ) Posteriors",
            xaxis_title="Saturation Rate (higher = faster saturation)",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_response_curves(results, mmm):
    """Render response curves with HDI bands."""
    st.subheader("📈 Response Curves")
    
    st.markdown("""
    Response curves show how media spend translates to sales effect after accounting for 
    saturation (diminishing returns). The shaded bands represent the HDI (Highest Density Interval).
    """)
    
    channel_names = mmm.channel_names
    posterior = results.trace.posterior
    panel = st.session_state.panel
    
    # Get original media spend ranges
    X_media_raw = panel.X_media.values
    
    # Settings
    col1, col2 = st.columns([1, 3])
    with col1:
        hdi_prob = st.slider("HDI Probability", 0.5, 0.99, 0.94, 0.01, 
                             help="Width of the credible interval", key="rc_hdi")
        n_points = st.slider("Curve Resolution", 50, 200, 100,
                            help="Number of points on the curve", key="rc_points")
        show_observed = st.checkbox("Show Observed Spend", value=True,
                                   help="Mark the range of observed spend values", key="rc_observed")
    
    # Compute response curves for each channel
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
        spend_mean = spend_raw.mean()
        
        # Use cached computation
        curve_data = compute_response_curves(
            sat_lam_samples,
            beta_samples,
            spend_max,
            mmm.y_std,
            n_points
        )
        
        # Compute HDI at selected probability
        hdi_low_pct = (1 - hdi_prob) / 2 * 100
        hdi_high_pct = (1 + hdi_prob) / 2 * 100
        
        response_data.append({
            'channel': channel,
            'x': curve_data['x'],
            'mean': curve_data['mean'],
            'hdi_low': np.percentile(curve_data['samples'], hdi_low_pct, axis=0),
            'hdi_high': np.percentile(curve_data['samples'], hdi_high_pct, axis=0),
            'spend_min': 0,
            'spend_max': spend_max,
            'spend_mean': spend_mean,
            'observed_spend': spend_raw
        })
    
    if not response_data:
        st.warning("No response curve data available")
        return
    
    # Plot
    with col2:
        colors = px.colors.qualitative.Set2
        
        fig = go.Figure()
        
        for i, data in enumerate(response_data):
            color = colors[i % len(colors)]
            color_rgba = rgb_to_rgba(color, 0.2)
            
            # HDI band
            fig.add_trace(go.Scatter(
                x=np.concatenate([data['x'], data['x'][::-1]]),
                y=np.concatenate([data['hdi_high'], data['hdi_low'][::-1]]),
                fill='toself',
                fillcolor=color_rgba,
                line=dict(color='rgba(0,0,0,0)'),
                name=f"{data['channel']} HDI",
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Mean curve
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['mean'],
                mode='lines',
                name=data['channel'],
                line=dict(color=color, width=2)
            ))
            
            # Observed spend range
            if show_observed:
                fig.add_vline(
                    x=data['spend_max'],
                    line_dash="dash",
                    line_color=color,
                    opacity=0.5,
                    annotation_text=f"{data['channel']} max",
                    annotation_position="top"
                )
        
        fig.update_layout(
            title="Response Curves by Channel",
            xaxis_title="Media Spend",
            yaxis_title="Contribution to Sales",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Diminishing returns analysis
    st.markdown("---")
    st.markdown("### 📊 Diminishing Returns Analysis")
    
    analysis_data = []
    for data in response_data:
        spend_mean = data['spend_mean']
        spend_max = data['spend_max']
        
        mean_idx = np.argmin(np.abs(data['x'] - spend_mean))
        max_idx = np.argmin(np.abs(data['x'] - spend_max))
        
        effect_at_mean = data['mean'][mean_idx]
        effect_at_max = data['mean'][max_idx]
        max_possible = data['mean'][-1]
        
        saturation_mean = effect_at_mean / max_possible * 100 if max_possible > 0 else 0
        saturation_max = effect_at_max / max_possible * 100 if max_possible > 0 else 0
        
        if mean_idx > 0 and mean_idx < len(data['x']) - 1:
            dx = data['x'][mean_idx + 1] - data['x'][mean_idx - 1]
            dy = data['mean'][mean_idx + 1] - data['mean'][mean_idx - 1]
            marginal_mean = dy / dx if dx > 0 else 0
        else:
            marginal_mean = 0
        
        analysis_data.append({
            'Channel': data['channel'],
            'Mean Spend': f"${spend_mean:,.0f}",
            'Max Spend': f"${spend_max:,.0f}",
            'Saturation @ Mean': f"{saturation_mean:.0f}%",
            'Saturation @ Max': f"{saturation_max:.0f}%",
            'Marginal Return @ Mean': f"{marginal_mean:.4f}",
        })
    
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)

@st.fragment
def render_component_breakdown(results, mmm):
    """Render full component decomposition of the model."""
    st.subheader("🧩 Component Breakdown")
    
    st.markdown("""
    This analysis breaks down the model's predictions into individual components:
    base (intercept), trend, seasonality, media effects, and control variables.
    """)
    
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    
    # Compute component decomposition if not already done
    if st.session_state.component_decomposition is None:
        with st.spinner("Computing component decomposition..."):
            try:
                st.session_state.component_decomposition = mmm.compute_component_decomposition()
            except Exception as e:
                st.error(f"Error computing decomposition: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    decomp = st.session_state.component_decomposition
    
    # Component summary pie chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Component Summary")
        
        summary_df = decomp.summary()
        
        # Filter to non-zero components
        summary_df = summary_df[summary_df['Total Contribution'].abs() > 1e-6]
        
        fig_pie = px.pie(
            summary_df,
            values='Total Contribution',
            names='Component',
            title="Share of Total Predicted Outcome",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Component Table")
        
        display_df = summary_df.copy()
        display_df['Total Contribution'] = display_df['Total Contribution'].apply(lambda x: f"{x:,.0f}")
        display_df['Contribution %'] = display_df['Contribution %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Media breakdown
        st.markdown("### Media Channel Breakdown")
        media_summary = decomp.media_summary()
        media_summary['Total Contribution'] = media_summary['Total Contribution'].apply(lambda x: f"{x:,.0f}")
        media_summary['Share of Media %'] = media_summary['Share of Media %'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(media_summary, use_container_width=True, hide_index=True)
    
    # Stacked area chart over time
    st.markdown("---")
    st.markdown("### Component Contributions Over Time")
    
    # Build time series of components
    periods = dim_info['periods']
    
    # Aggregate by period if panel data
    if dim_info['has_geo'] or dim_info['has_product']:
        # Sum over geos/products
        component_ts = {'Period': periods}
        
        for t in range(len(periods)):
            mask = mmm.time_idx == t
            
            if t == 0:
                component_ts['Base'] = [decomp.intercept[mask].sum()]
                component_ts['Trend'] = [decomp.trend[mask].sum()]
                component_ts['Seasonality'] = [decomp.seasonality[mask].sum()]
                component_ts['Media'] = [decomp.media_total[mask].sum()]
                component_ts['Controls'] = [decomp.controls_total[mask].sum()]
            else:
                component_ts['Base'].append(decomp.intercept[mask].sum())
                component_ts['Trend'].append(decomp.trend[mask].sum())
                component_ts['Seasonality'].append(decomp.seasonality[mask].sum())
                component_ts['Media'].append(decomp.media_total[mask].sum())
                component_ts['Controls'].append(decomp.controls_total[mask].sum())
        
        ts_df = pd.DataFrame(component_ts)
    else:
        ts_df = pd.DataFrame({
            'Period': periods[:len(decomp.intercept)],
            'Base': decomp.intercept,
            'Trend': decomp.trend,
            'Seasonality': decomp.seasonality,
            'Media': decomp.media_total,
            'Controls': decomp.controls_total
        })
    
    # Stacked area chart
    fig_stack = go.Figure()
    
    colors = {
        'Base': '#3498db',
        'Trend': '#e74c3c',
        'Seasonality': '#2ecc71',
        'Media': '#9b59b6',
        'Controls': '#f39c12'
    }
    
    for component in ['Base', 'Trend', 'Seasonality', 'Media', 'Controls']:
        if component in ts_df.columns:
            fig_stack.add_trace(go.Scatter(
                x=ts_df['Period'],
                y=ts_df[component],
                name=component,
                mode='lines',
                stackgroup='one',
                line=dict(color=colors.get(component, 'gray'))
            ))
    
    # Add observed data line
    if dim_info['has_geo'] or dim_info['has_product']:
        y_obs_by_period = []
        for t in range(len(periods)):
            mask = mmm.time_idx == t
            y_obs_by_period.append(mmm.y_raw[mask].sum())
        observed = y_obs_by_period
    else:
        observed = mmm.y_raw
    
    fig_stack.add_trace(go.Scatter(
        x=ts_df['Period'],
        y=observed[:len(ts_df)],
        name='Observed',
        mode='markers',
        marker=dict(color='black', size=4)
    ))
    
    fig_stack.update_layout(
        title="Stacked Component Contributions vs Observed",
        xaxis_title="Period",
        yaxis_title=mmm.mff_config.kpi.name,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_stack, use_container_width=True)
    
    # Individual component time series
    st.markdown("### Individual Component Time Series")
    
    selected_components = st.multiselect(
        "Select components to visualize",
        options=['Base', 'Trend', 'Seasonality', 'Media', 'Controls'],
        default=['Trend', 'Seasonality', 'Media']
    )
    
    if selected_components:
        fig_individual = go.Figure()
        
        for component in selected_components:
            if component in ts_df.columns:
                fig_individual.add_trace(go.Scatter(
                    x=ts_df['Period'],
                    y=ts_df[component],
                    name=component,
                    mode='lines',
                    line=dict(color=colors.get(component, 'gray'), width=2)
                ))
        
        fig_individual.update_layout(
            title="Selected Component Contributions",
            xaxis_title="Period",
            yaxis_title="Contribution",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_individual, use_container_width=True)

@st.fragment
def render_trend_analysis(results, mmm):
    """Render detailed trend and seasonality analysis."""
    st.subheader("📈 Trend & Seasonality Analysis")
    
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    posterior = results.trace.posterior
    
    # Get decomposition
    if st.session_state.component_decomposition is None:
        with st.spinner("Computing component decomposition..."):
            st.session_state.component_decomposition = mmm.compute_component_decomposition()
    
    decomp = st.session_state.component_decomposition
    
    # Tab for different analyses
    trend_tabs = st.tabs(["🔄 Trend Component", "🌊 Seasonality", "📊 Combined"])
    
    with trend_tabs[0]:
        render_trend_tab(results, mmm, decomp, dim_info, posterior)
    
    with trend_tabs[1]:
        render_seasonality_tab(results, mmm, decomp, dim_info, posterior)
    
    with trend_tabs[2]:
        render_combined_trend_seasonality(results, mmm, decomp, dim_info)

@st.fragment
def render_trend_tab(results, mmm, decomp, dim_info, posterior):
    """Render trend component analysis."""
    st.markdown("### Trend Component Analysis")
    
    trend_type = st.session_state.get('trend_type', 'Linear')
    
    st.info(f"**Trend Type:** {trend_type}")
    
    # Build trend time series
    periods = dim_info['periods']
    
    if dim_info['has_geo'] or dim_info['has_product']:
        trend_by_period = []
        for t in range(len(periods)):
            mask = mmm.time_idx == t
            trend_by_period.append(decomp.trend[mask].mean())  # Average across groups
        trend_values = trend_by_period
    else:
        trend_values = decomp.trend
    
    # Plot trend
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=periods[:len(trend_values)],
        y=trend_values,
        name='Trend',
        mode='lines',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig_trend.update_layout(
        title="Estimated Trend Over Time",
        xaxis_title="Period",
        yaxis_title="Trend Contribution",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Trend statistics
    col1, col2, col3, col4 = st.columns(4)
    
    trend_arr = np.array(trend_values)
    
    with col1:
        st.metric("Mean Trend", f"{trend_arr.mean():,.1f}")
    with col2:
        st.metric("Trend Std", f"{trend_arr.std():,.1f}")
    with col3:
        # Overall slope (linear approximation)
        if len(trend_arr) > 1:
            slope = (trend_arr[-1] - trend_arr[0]) / len(trend_arr)
            st.metric("Avg Weekly Change", f"{slope:+.1f}")
        else:
            st.metric("Avg Weekly Change", "N/A")
    with col4:
        st.metric("Total Trend Effect", f"{decomp.total_trend:,.0f}")
    
    # Trend-specific parameter analysis
    st.markdown("---")
    st.markdown("### Trend Parameters")
    
    if trend_type == 'Linear':
        if 'trend_slope' in posterior:
            slope_samples = posterior['trend_slope'].values.flatten()
            
            fig_slope = go.Figure()
            fig_slope.add_trace(go.Histogram(
                x=slope_samples * mmm.y_std,  # Original scale
                nbinsx=50,
                name='Trend Slope'
            ))
            fig_slope.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_slope.update_layout(
                title="Posterior Distribution of Trend Slope",
                xaxis_title="Weekly Growth Rate (original scale)",
                height=300
            )
            st.plotly_chart(fig_slope, use_container_width=True)
            
            st.markdown(f"""
            **Interpretation:**
            - Mean slope: {slope_samples.mean() * mmm.y_std:+.2f} per week
            - {(slope_samples > 0).mean() * 100:.1f}% probability of positive trend
            """)
    
    elif trend_type == 'Piecewise (Prophet-style)':
        if 'trend_delta' in posterior:
            delta_samples = posterior['trend_delta'].values
            delta_mean = delta_samples.mean(axis=(0, 1))
            
            # Changepoint magnitudes
            changepoints = mmm.trend_features.get('changepoints', [])
            
            if len(changepoints) > 0:
                cp_df = pd.DataFrame({
                    'Changepoint': [f"CP {i+1}" for i in range(len(changepoints))],
                    'Location (%)': changepoints * 100,
                    'Magnitude': delta_mean * mmm.y_std
                })
                
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Bar(
                    x=cp_df['Changepoint'],
                    y=cp_df['Magnitude'],
                    marker_color=['green' if m > 0 else 'red' for m in cp_df['Magnitude']]
                ))
                fig_cp.update_layout(
                    title="Changepoint Magnitudes",
                    xaxis_title="Changepoint",
                    yaxis_title="Rate Change",
                    height=350
                )
                st.plotly_chart(fig_cp, use_container_width=True)
                
                st.dataframe(cp_df.round(3), use_container_width=True, hide_index=True)
    
    elif trend_type == 'Gaussian Process':
        if 'gp_lengthscale' in posterior:
            ls_samples = posterior['gp_lengthscale'].values.flatten()
            amp_samples = posterior['gp_amplitude'].values.flatten() if 'gp_amplitude' in posterior else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_ls = go.Figure()
                fig_ls.add_trace(go.Histogram(x=ls_samples, nbinsx=50))
                fig_ls.update_layout(
                    title="GP Lengthscale Posterior",
                    xaxis_title="Lengthscale (proportion of time series)",
                    height=300
                )
                st.plotly_chart(fig_ls, use_container_width=True)
            
            with col2:
                if amp_samples is not None:
                    fig_amp = go.Figure()
                    fig_amp.add_trace(go.Histogram(x=amp_samples, nbinsx=50))
                    fig_amp.update_layout(
                        title="GP Amplitude Posterior",
                        xaxis_title="Amplitude",
                        height=300
                    )
                    st.plotly_chart(fig_amp, use_container_width=True)
            
            st.markdown(f"""
            **GP Interpretation:**
            - Mean lengthscale: {ls_samples.mean():.3f} (≈ {ls_samples.mean() * mmm.n_periods:.1f} weeks)
            - Longer lengthscale = smoother trend
            """)

@st.fragment
def render_seasonality_tab(results, mmm, decomp, dim_info, posterior):
    """Render seasonality analysis."""
    st.markdown("### Seasonality Analysis")
    
    yearly_order = st.session_state.get('yearly_order', 0)
    
    if yearly_order == 0:
        st.info("No yearly seasonality was included in this model.")
        return
    
    periods = dim_info['periods']
    
    # Build seasonality time series
    if dim_info['has_geo'] or dim_info['has_product']:
        season_by_period = []
        for t in range(len(periods)):
            mask = mmm.time_idx == t
            season_by_period.append(decomp.seasonality[mask].mean())
        season_values = season_by_period
    else:
        season_values = decomp.seasonality
    
    # Plot seasonality
    fig_season = go.Figure()
    
    fig_season.add_trace(go.Scatter(
        x=periods[:len(season_values)],
        y=season_values,
        name='Seasonality',
        mode='lines',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig_season.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig_season.update_layout(
        title="Seasonal Pattern Over Time",
        xaxis_title="Period",
        yaxis_title="Seasonal Effect",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    
    # Extract average seasonal pattern by week of year
    if len(periods) >= 52:
        st.markdown("### Average Weekly Seasonal Pattern")
        
        # Calculate week of year for each period
        period_dates = pd.to_datetime(periods[:len(season_values)])
        weeks = period_dates.isocalendar().week.values
        
        season_df = pd.DataFrame({
            'week': weeks,
            'seasonality': season_values
        })
        
        weekly_pattern = season_df.groupby('week')['seasonality'].mean().reset_index()
        
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Bar(
            x=weekly_pattern['week'],
            y=weekly_pattern['seasonality'],
            marker_color=['green' if s > 0 else 'red' for s in weekly_pattern['seasonality']]
        ))
        fig_weekly.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_weekly.update_layout(
            title="Average Seasonal Effect by Week of Year",
            xaxis_title="Week of Year",
            yaxis_title="Average Seasonal Effect",
            height=350
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Peak/trough analysis
        peak_week = weekly_pattern.loc[weekly_pattern['seasonality'].idxmax(), 'week']
        trough_week = weekly_pattern.loc[weekly_pattern['seasonality'].idxmin(), 'week']
        peak_value = weekly_pattern['seasonality'].max()
        trough_value = weekly_pattern['seasonality'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Peak Week", f"Week {int(peak_week)}")
        with col2:
            st.metric("Peak Effect", f"{peak_value:+,.0f}")
        with col3:
            st.metric("Trough Week", f"Week {int(trough_week)}")
        with col4:
            st.metric("Trough Effect", f"{trough_value:+,.0f}")
    
    # Fourier coefficient analysis
    st.markdown("---")
    st.markdown("### Seasonality Coefficients")
    
    season_coefs = []
    for var in posterior.data_vars:
        if 'season_yearly' in var:
            samples = posterior[var].values.flatten()
            season_coefs.append({
                'Coefficient': var,
                'Mean': samples.mean() * mmm.y_std,
                'Std': samples.std() * mmm.y_std,
                'HDI 3%': np.percentile(samples, 3) * mmm.y_std,
                'HDI 97%': np.percentile(samples, 97) * mmm.y_std
            })
    
    if season_coefs:
        coef_df = pd.DataFrame(season_coefs).round(3)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

@st.fragment
def render_combined_trend_seasonality(results, mmm, decomp, dim_info):
    """Render combined trend and seasonality analysis."""
    st.markdown("### Combined Trend & Seasonality")
    
    periods = dim_info['periods']
    
    # Build time series
    if dim_info['has_geo'] or dim_info['has_product']:
        trend_by_period = []
        season_by_period = []
        for t in range(len(periods)):
            mask = mmm.time_idx == t
            trend_by_period.append(decomp.trend[mask].mean())
            season_by_period.append(decomp.seasonality[mask].mean())
        trend_values = trend_by_period
        season_values = season_by_period
    else:
        trend_values = decomp.trend
        season_values = decomp.seasonality
    
    combined = np.array(trend_values) + np.array(season_values)
    
    fig_combined = go.Figure()
    
    fig_combined.add_trace(go.Scatter(
        x=periods[:len(trend_values)],
        y=trend_values,
        name='Trend',
        mode='lines',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=periods[:len(season_values)],
        y=season_values,
        name='Seasonality',
        mode='lines',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=periods[:len(combined)],
        y=combined,
        name='Trend + Seasonality',
        mode='lines',
        line=dict(color='#3498db', width=3)
    ))
    
    fig_combined.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig_combined.update_layout(
        title="Trend and Seasonality Components",
        xaxis_title="Period",
        yaxis_title="Effect",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Variance decomposition
    st.markdown("### Variance Decomposition")
    
    trend_var = np.var(trend_values)
    season_var = np.var(season_values)
    total_var = trend_var + season_var
    
    if total_var > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Variance", f"{trend_var:,.1f}")
        with col2:
            st.metric("Seasonality Variance", f"{season_var:,.1f}")
        with col3:
            trend_pct = trend_var / total_var * 100
            st.metric("Trend % of Variation", f"{trend_pct:.1f}%")
        
        # Pie chart
        fig_var = px.pie(
            values=[trend_var, season_var],
            names=['Trend', 'Seasonality'],
            title="Variance Attribution",
            color_discrete_sequence=['#e74c3c', '#2ecc71']
        )
        st.plotly_chart(fig_var, use_container_width=True)

@st.fragment
def render_contributions(results, mmm):
    """Render channel contributions using counterfactual analysis."""
    st.subheader("💰 Channel Contributions (Counterfactual)")
    
    st.markdown("""
    **Counterfactual Contribution Analysis**: For each channel, we compare the baseline 
    prediction (with all channels) to a counterfactual scenario (with that channel zeroed out).
    The difference represents the channel's true contribution, properly accounting for 
    saturation and adstock effects.
    """)
    
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    channel_names = mmm.channel_names
    
    # Settings
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Settings")
        
        compute_uncertainty = st.checkbox(
            "Compute Uncertainty (HDI)",
            value=True,
            help="Compute confidence intervals for contributions (slower)"
        )
        
        hdi_prob = st.slider(
            "HDI Probability",
            0.5, 0.99, 0.94, 0.01,
            help="Width of the credible interval"
        ) if compute_uncertainty else 0.94
        
        use_time_period = st.checkbox("Filter by Time Period", value=False)
        
        if use_time_period:
            start_week = st.number_input(
                "Start Week",
                min_value=0,
                max_value=dim_info['n_periods'] - 1,
                value=0
            )
            end_week = st.number_input(
                "End Week",
                min_value=0,
                max_value=dim_info['n_periods'] - 1,
                value=dim_info['n_periods'] - 1
            )
            time_period = (int(start_week), int(end_week))
        else:
            time_period = None
    
    with col2:
        if st.button("🔄 Compute Contributions", type="primary", use_container_width=True):
            with st.spinner("Computing counterfactual contributions..."):
                try:
                    contrib_results = mmm.compute_counterfactual_contributions(
                        time_period=time_period,
                        compute_uncertainty=compute_uncertainty,
                        hdi_prob=hdi_prob,
                        random_seed=42
                    )
                    st.session_state.contributions = contrib_results
                    st.success("✓ Contributions computed successfully!")
                except Exception as e:
                    st.error(f"Error computing contributions: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
    
    if st.session_state.contributions is not None:
        contrib = st.session_state.contributions
        
        st.markdown("---")
        
        st.markdown("### Contribution Summary")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_pie = px.pie(
                values=contrib.total_contributions.values,
                names=contrib.total_contributions.index,
                title="Share of Total Media Contribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            summary_df = contrib.summary()
            summary_df['Total Contribution'] = summary_df['Total Contribution'].apply(lambda x: f"{x:,.0f}")
            summary_df['Contribution %'] = summary_df['Contribution %'].apply(lambda x: f"{x:.1f}%")
            
            if 'HDI 3%' in summary_df.columns:
                summary_df['HDI 3%'] = summary_df['HDI 3%'].apply(lambda x: f"{x:,.0f}")
                summary_df['HDI 97%'] = summary_df['HDI 97%'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Time series of contributions
        st.markdown("### Contributions Over Time")
        
        if dim_info['has_geo'] or dim_info['has_product']:
            contrib_by_period = []
            for t, period in enumerate(dim_info['periods']):
                mask = mmm.time_idx == t
                row = {'Period': period}
                for ch in channel_names:
                    row[ch] = contrib.channel_contributions[ch].values[mask].sum()
                contrib_by_period.append(row)
            period_contrib = pd.DataFrame(contrib_by_period)
        else:
            period_contrib = contrib.channel_contributions.copy()
            period_contrib['Period'] = dim_info['periods'][:len(period_contrib)]
        
        fig_ts = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for i, col in enumerate(channel_names):
            fig_ts.add_trace(go.Scatter(
                x=period_contrib['Period'],
                y=period_contrib[col],
                name=col,
                mode='lines',
                stackgroup='one',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig_ts.update_layout(
            title="Stacked Channel Contributions Over Time",
            xaxis_title="Period",
            yaxis_title="Contribution",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode='x unified'
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # ROAS Analysis
        st.markdown("### ROAS Analysis")
        
        X_media_raw = panel.X_media.values
        
        roas_data = []
        for i, ch in enumerate(channel_names):
            total_spend = X_media_raw[:, i].sum()
            total_contrib = contrib.total_contributions[ch]
            roas = total_contrib / total_spend if total_spend > 0 else 0
            
            row = {
                'Channel': ch,
                'Total Spend': total_spend,
                'Total Contribution': total_contrib,
                'ROAS': roas
            }
            
            if contrib.contribution_hdi_low is not None:
                row['ROAS HDI Low'] = contrib.contribution_hdi_low[ch] / total_spend if total_spend > 0 else 0
                row['ROAS HDI High'] = contrib.contribution_hdi_high[ch] / total_spend if total_spend > 0 else 0
            
            roas_data.append(row)
        
        roas_df = pd.DataFrame(roas_data)
        
        fig_roas = go.Figure()
        
        if 'ROAS HDI Low' in roas_df.columns:
            error_minus = roas_df['ROAS'] - roas_df['ROAS HDI Low']
            error_plus = roas_df['ROAS HDI High'] - roas_df['ROAS']
            
            fig_roas.add_trace(go.Bar(
                x=roas_df['Channel'],
                y=roas_df['ROAS'],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=error_plus.values,
                    arrayminus=error_minus.values,
                    color='rgba(0,0,0,0.5)',
                    thickness=2,
                    width=6
                ),
                marker_color=[colors[i % len(colors)] for i in range(len(roas_df))],
                name='ROAS'
            ))
        else:
            fig_roas.add_trace(go.Bar(
                x=roas_df['Channel'],
                y=roas_df['ROAS'],
                marker_color=[colors[i % len(colors)] for i in range(len(roas_df))],
                name='ROAS'
            ))
        
        fig_roas.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                         annotation_text="Break-even", annotation_position="right")
        
        fig_roas.update_layout(
            title="Return on Ad Spend (ROAS) by Channel",
            xaxis_title="Channel",
            yaxis_title="ROAS",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_roas, use_container_width=True)
        
        display_df = roas_df.copy()
        display_df['Total Spend'] = display_df['Total Spend'].apply(lambda x: f"${x:,.0f}")
        display_df['Total Contribution'] = display_df['Total Contribution'].apply(lambda x: f"{x:,.0f}")
        display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.3f}")
        if 'ROAS HDI Low' in display_df.columns:
            display_df['ROAS HDI Low'] = display_df['ROAS HDI Low'].apply(lambda x: f"{x:.3f}")
            display_df['ROAS HDI High'] = display_df['ROAS HDI High'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

@st.fragment
def render_scenario_planning(results, mmm):
    """Render scenario planning and what-if analysis."""
    st.subheader("🔮 Scenario Planning")
    
    st.markdown("""
    Simulate how changes in media spend would affect outcomes. This uses the fitted model
    to predict outcomes under different budget scenarios.
    """)
    
    panel = st.session_state.panel
    channel_names = mmm.channel_names
    dim_info = get_dimension_info(mmm, panel)
    
    scenario_tabs = st.tabs(["📊 Marginal Analysis", "🎯 What-If Scenario", "📈 Budget Optimization"])
    
    with scenario_tabs[0]:
        render_marginal_analysis(mmm, channel_names, dim_info)
    
    with scenario_tabs[1]:
        render_what_if_scenario(mmm, channel_names, dim_info)
    
    with scenario_tabs[2]:
        render_budget_optimization(mmm, channel_names, dim_info)

@st.fragment
def render_marginal_analysis(mmm, channel_names, dim_info):
    """Render marginal contribution analysis."""
    st.markdown("### Marginal Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        spend_increase_pct = st.slider(
            "Spend Increase %",
            1, 50, 10,
            help="Simulate this percentage increase in spend"
        )
        
        compute_marginal = st.button("Compute Marginal Returns", type="primary", use_container_width=True)
    
    with col2:
        if compute_marginal:
            with st.spinner("Computing marginal contributions..."):
                try:
                    marginal_df = mmm.compute_marginal_contributions(
                        spend_increase_pct=float(spend_increase_pct),
                        random_seed=42
                    )
                    st.session_state.marginal_results = marginal_df
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
    
    if 'marginal_results' in st.session_state and st.session_state.marginal_results is not None:
        marginal_df = st.session_state.marginal_results
        
        fig = go.Figure()
        
        colors = ['#2ecc71' if x >= 1 else '#e74c3c' for x in marginal_df['Marginal ROAS']]
        
        fig.add_trace(go.Bar(
            x=marginal_df['Channel'],
            y=marginal_df['Marginal ROAS'],
            marker_color=colors,
            text=marginal_df['Marginal ROAS'].apply(lambda x: f"{x:.3f}"),
            textposition='outside'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="Break-even")
        
        fig.update_layout(
            title=f"Marginal ROAS for {spend_increase_pct}% Spend Increase",
            xaxis_title="Channel",
            yaxis_title="Marginal ROAS",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        display_df = marginal_df.copy()
        for col in ['Current Spend', f'Spend Increase ({spend_increase_pct}%)', 'Marginal Contribution']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        display_df['Marginal ROAS'] = display_df['Marginal ROAS'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

@st.fragment
def render_what_if_scenario(mmm, channel_names, dim_info):
    """Render what-if scenario analysis."""
    st.markdown("### What-If Scenario")
    
    st.markdown("**Set spend multipliers for each channel:**")
    st.caption("1.0 = no change, 1.2 = +20%, 0.8 = -20%")
    
    spend_changes = {}
    cols = st.columns(min(3, len(channel_names)))
    
    for i, ch in enumerate(channel_names):
        with cols[i % 3]:
            multiplier = st.slider(
                f"{ch}",
                0.0, 2.0, 1.0, 0.1,
                key=f"what_if_{ch}"
            )
            if multiplier != 1.0:
                spend_changes[ch] = multiplier
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        run_scenario = st.button("Run Scenario", type="primary", use_container_width=True)
    
    if run_scenario and spend_changes:
        with st.spinner("Running scenario..."):
            try:
                scenario_results = mmm.what_if_scenario(
                    spend_changes=spend_changes,
                    random_seed=42
                )
                st.session_state.scenario_results = scenario_results
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    if 'scenario_results' in st.session_state and st.session_state.scenario_results is not None:
        results = st.session_state.scenario_results
        
        st.markdown("---")
        st.markdown("### Scenario Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Baseline Outcome", f"{results['baseline_outcome']:,.0f}")
        
        with col2:
            st.metric("Scenario Outcome", f"{results['scenario_outcome']:,.0f}")
        
        with col3:
            change = results['outcome_change']
            delta_color = "normal" if change >= 0 else "inverse"
            st.metric(
                "Change",
                f"{change:+,.0f}",
                delta=f"{results['outcome_change_pct']:+.1f}%",
                delta_color=delta_color
            )
        
        spend_data = []
        for ch, details in results['spend_changes'].items():
            spend_data.append({
                'Channel': ch,
                'Original Spend': f"${details['original']:,.0f}",
                'Scenario Spend': f"${details['scenario']:,.0f}",
                'Change': f"${details['change']:+,.0f}",
                'Change %': f"{details['change_pct']:+.1f}%"
            })
        
        if spend_data:
            st.markdown("**Spend Changes:**")
            st.dataframe(pd.DataFrame(spend_data), use_container_width=True, hide_index=True)
    
    elif run_scenario and not spend_changes:
        st.info("No changes specified. Adjust the sliders to create a scenario.")

@st.fragment
def render_budget_optimization(mmm, channel_names, dim_info):
    """Render budget optimization analysis."""
    st.markdown("### Budget Optimization")
    
    X_media_raw = mmm.X_media_raw
    current_spend = {ch: X_media_raw[:, i].sum() for i, ch in enumerate(channel_names)}
    total_budget = sum(current_spend.values())
    
    st.markdown(f"**Current Total Budget:** ${total_budget:,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Allocation:**")
        for ch, spend in current_spend.items():
            pct = spend / total_budget * 100
            st.write(f"- {ch}: ${spend:,.0f} ({pct:.1f}%)")
    
    with col2:
        budget_change = st.slider(
            "Total Budget Change %",
            -50, 100, 0, 5,
            help="Adjust total budget by this percentage"
        )
        
        new_total = total_budget * (1 + budget_change / 100)
        st.write(f"**New Total Budget:** ${new_total:,.0f}")
    
    if st.button("Suggest Optimal Allocation", type="primary"):
        with st.spinner("Computing optimal allocation..."):
            try:
                marginal_10 = mmm.compute_marginal_contributions(spend_increase_pct=10.0, random_seed=42)
                
                marginal_roas = marginal_10.set_index('Channel')['Marginal ROAS']
                
                weights = marginal_roas.clip(lower=0.1)
                weights = weights / weights.sum()
                
                suggested_allocation = {}
                for ch in channel_names:
                    suggested_allocation[ch] = {
                        'current': current_spend[ch],
                        'suggested': new_total * weights[ch],
                        'change': new_total * weights[ch] - current_spend[ch],
                        'marginal_roas': marginal_roas[ch]
                    }
                
                st.session_state.suggested_allocation = suggested_allocation
                
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    if 'suggested_allocation' in st.session_state and st.session_state.suggested_allocation:
        alloc = st.session_state.suggested_allocation
        
        st.markdown("---")
        st.markdown("### Suggested Allocation")
        
        alloc_data = []
        for ch, details in alloc.items():
            alloc_data.append({
                'Channel': ch,
                'Current': f"${details['current']:,.0f}",
                'Suggested': f"${details['suggested']:,.0f}",
                'Change': f"${details['change']:+,.0f}",
                'Marginal ROAS': f"{details['marginal_roas']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(alloc_data), use_container_width=True, hide_index=True)
        
        fig = go.Figure()
        
        current_vals = [alloc[ch]['current'] for ch in channel_names]
        suggested_vals = [alloc[ch]['suggested'] for ch in channel_names]
        
        fig.add_trace(go.Bar(
            name='Current',
            x=channel_names,
            y=current_vals,
            marker_color='rgba(99, 110, 250, 0.6)'
        ))
        
        fig.add_trace(go.Bar(
            name='Suggested',
            x=channel_names,
            y=suggested_vals,
            marker_color='rgba(0, 204, 150, 0.6)'
        ))
        
        fig.update_layout(
            title="Current vs Suggested Budget Allocation",
            xaxis_title="Channel",
            yaxis_title="Budget ($)",
            barmode='group',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)

@st.fragment
def render_summary(results, mmm):
    """Render model summary."""
    st.subheader("📋 Full Model Summary")
    
    summary = results.summary()
    
    key_params = [p for p in summary.index if any(x in p for x in ['beta', 'sigma', 'adstock', 'sat_lam', 'trend', 'geo'])]
    
    if key_params:
        st.markdown("### Key Parameters")
        st.dataframe(summary.loc[key_params].round(4), use_container_width=True)
    
    st.markdown("### Full Parameter Summary")
    st.dataframe(summary.round(4), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_summary = summary.to_csv()
        st.download_button(
            "📥 Download Summary (CSV)",
            csv_summary,
            "mmm_summary.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.contributions is not None:
            csv_contrib = st.session_state.contributions.channel_contributions.to_csv()
            st.download_button(
                "📥 Download Contributions (CSV)",
                csv_contrib,
                "mmm_contributions.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.session_state.component_decomposition is not None:
            decomp = st.session_state.component_decomposition
            summary_csv = decomp.summary().to_csv(index=False)
            st.download_button(
                "📥 Download Decomposition (CSV)",
                summary_csv,
                "mmm_decomposition.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    
    tabs = st.tabs(["📁 Data", "⚙️ Configure", "🔬 Model", "📈 Results"])
    
    with tabs[0]:
        render_data_tab()
    
    with tabs[1]:
        render_config_tab()
    
    with tabs[2]:
        render_model_tab()
    
    with tabs[3]:
        render_results_tab()


if __name__ == "__main__":
    main()
"""
MMM Framework - Streamlit Web Application

A user-friendly interface for:
- Uploading MFF data
- Configuring model settings
- Running Bayesian MMM
- Visualizing results
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
        st.caption("v0.1.0")


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
            options=['Linear', 'None'],
            help="How to model the underlying trend"
        )
    
    with col2:
        st.subheader("🌊 Seasonality")
        
        yearly_order = st.slider(
            "Yearly Seasonality Order",
            0, 5, 2,
            help="Number of Fourier terms for yearly seasonality (0 = disabled)"
        )
        
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
            yearly_order=yearly_order,
            pool_geo=pool_geo,
            use_numpyro=use_numpyro
        )


def fit_model(n_chains, n_draws, n_tune, target_accept, trend_type, yearly_order, pool_geo, use_numpyro):
    """Fit the Bayesian MMM model."""
    from mmm_framework import (
        BayesianMMM, TrendConfig, TrendType,
        ModelConfigBuilder, HierarchicalConfigBuilder, SeasonalityConfigBuilder
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
        
        # Trend config
        if trend_type == 'Linear':
            trend_config = TrendConfig(type=TrendType.LINEAR)
        else:
            trend_config = TrendConfig(type=TrendType.NONE)
        
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

def render_results_tab():
    """Render the results and visualization tab."""
    st.header("📈 Results & Analysis")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please fit the model first")
        return
    
    results = st.session_state.results
    mmm = st.session_state.mmm
    
    # Sub-tabs for different result views
    result_tabs = st.tabs([
        "📊 Diagnostics", 
        "🎯 Model Fit",
        "📉 Posteriors", 
        "🔄 Prior vs Posterior",
        "📈 Response Curves", 
        "🌊 Decomposition",
        "💰 Contributions", 
        "📋 Summary"
    ])
    
    with result_tabs[0]:
        render_diagnostics(results)
    
    with result_tabs[1]:
        render_model_fit(results, mmm)
    
    with result_tabs[2]:
        render_posteriors(results, mmm)
    
    with result_tabs[3]:
        render_prior_vs_posterior(results, mmm)
    
    with result_tabs[4]:
        render_response_curves(results, mmm)
    
    with result_tabs[5]:
        render_decomposition(results, mmm)
    
    with result_tabs[6]:
        render_contributions(results, mmm)
    
    with result_tabs[7]:
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
        # Get rhat as xarray Dataset and convert properly
        rhat_data = az.rhat(results.trace)
        
        # Build a list of (parameter, rhat_value) tuples
        rhat_records = []
        for var_name in rhat_data.data_vars:
            values = rhat_data[var_name].values
            # Handle scalar vs array values
            if np.ndim(values) == 0:
                rhat_records.append({'Parameter': var_name, 'R-hat': float(values)})
            else:
                # For array variables, flatten and create indexed names
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


def render_model_fit(results, mmm):
    """Render posterior predictive model fit."""
    st.subheader("🎯 Model Fit")
    
    st.markdown("""
    This plot shows the observed data against the model's posterior predictions.
    The shaded band represents the prediction uncertainty (HDI).
    """)
    
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    
    # Dimension selection for multi-geo/product data
    col_settings1, col_settings2 = st.columns([1, 1])
    
    with col_settings1:
        if dim_info['has_geo'] or dim_info['has_product']:
            view_level = st.radio(
                "Aggregation Level",
                ["Total (Sum over all)", "By Geography", "By Product"] if dim_info['has_geo'] and dim_info['has_product']
                else ["Total (Sum over all)", "By Geography"] if dim_info['has_geo']
                else ["Total (Sum over all)", "By Product"] if dim_info['has_product']
                else ["Total"],
                horizontal=True,
                key="fit_view_level"
            )
        else:
            view_level = "Total"
    
    with col_settings2:
        hdi_prob = st.slider("HDI Width", 0.5, 0.99, 0.94, 0.01, key="fit_hdi")
        show_residuals = st.checkbox("Show Residuals", value=False, key="fit_residuals")
    
    # Get time periods for x-axis
    periods = dim_info['periods']
    
    # Generate posterior predictive with caching
    with st.spinner("Computing posterior predictions..."):
        try:
            ppc_results = compute_posterior_predictive(
                mmm.model, 
                results.trace,
                mmm.n_obs,
                mmm.y_std,
                mmm.y_mean
            )
        except Exception as e:
            st.error(f"Error computing predictions: {e}")
            return
    
    # Build dataframe with predictions and observed
    pred_df = pd.DataFrame({
        'y_obs': mmm.y_raw,
        'y_pred_mean': ppc_results['mean'],
        'y_pred_std': ppc_results['std']
    })
    
    # Add period and dimension columns (not using panel index to avoid ambiguity)
    if isinstance(panel.index, pd.MultiIndex):
        period_col = mmm.mff_config.columns.period
        pred_df['Period'] = panel.index.get_level_values(period_col).values
        if dim_info['has_geo']:
            geo_col = mmm.mff_config.columns.geography
            pred_df['Geography'] = panel.index.get_level_values(geo_col).values
        if dim_info['has_product']:
            prod_col = mmm.mff_config.columns.product
            pred_df['Product'] = panel.index.get_level_values(prod_col).values
    else:
        pred_df['Period'] = periods[:len(pred_df)]
    
    # Compute HDI from cached samples
    hdi_low_pct = (1 - hdi_prob) / 2 * 100
    hdi_high_pct = (1 + hdi_prob) / 2 * 100
    
    # Aggregate based on view level
    if view_level == "Total (Sum over all)" or view_level == "Total":
        # Sum over all dimensions, group by period
        agg_df = pred_df.groupby('Period').agg({
            'y_obs': 'sum',
            'y_pred_mean': 'sum',
            'y_pred_std': lambda x: np.sqrt((x**2).sum())  # Sum of variances for independent
        }).reset_index()
        
        # Recompute HDI for aggregated data
        y_pred_samples_agg = ppc_results['samples'].copy()
        
        # Aggregate samples by period
        if dim_info['has_geo'] or dim_info['has_product']:
            n_per_period = mmm.n_obs // dim_info['n_periods']
            y_pred_agg = np.zeros((y_pred_samples_agg.shape[0], dim_info['n_periods']))
            y_obs_agg = np.zeros(dim_info['n_periods'])
            
            for t in range(dim_info['n_periods']):
                mask = pred_df['Period'] == periods[t]
                y_pred_agg[:, t] = y_pred_samples_agg[:, mask.values].sum(axis=1)
                y_obs_agg[t] = pred_df.loc[mask, 'y_obs'].sum()
            
            agg_df = pd.DataFrame({
                'Period': periods,
                'y_obs': y_obs_agg,
                'y_pred_mean': y_pred_agg.mean(axis=0),
                'y_pred_hdi_low': np.percentile(y_pred_agg, hdi_low_pct, axis=0),
                'y_pred_hdi_high': np.percentile(y_pred_agg, hdi_high_pct, axis=0)
            })
        else:
            agg_df['y_pred_hdi_low'] = np.percentile(ppc_results['samples'], hdi_low_pct, axis=0)
            agg_df['y_pred_hdi_high'] = np.percentile(ppc_results['samples'], hdi_high_pct, axis=0)
        
        _plot_model_fit_single(agg_df, "Total", hdi_prob, show_residuals, mmm)
    
    elif view_level == "By Geography" and dim_info['has_geo']:
        selected_geo = st.selectbox(
            "Select Geography",
            ["All Geographies (Faceted)"] + dim_info['geo_names'],
            key="fit_geo_select"
        )
        
        if selected_geo == "All Geographies (Faceted)":
            _plot_model_fit_faceted(pred_df, ppc_results, dim_info, 'Geography', 
                                    hdi_low_pct, hdi_high_pct, periods, mmm)
        else:
            geo_mask = pred_df['Geography'] == selected_geo
            geo_df = pred_df[geo_mask].copy()
            
            # Get samples for this geo
            geo_samples = ppc_results['samples'][:, geo_mask.values]
            geo_df['y_pred_hdi_low'] = np.percentile(geo_samples, hdi_low_pct, axis=0)
            geo_df['y_pred_hdi_high'] = np.percentile(geo_samples, hdi_high_pct, axis=0)
            
            _plot_model_fit_single(geo_df, selected_geo, hdi_prob, show_residuals, mmm)
    
    elif view_level == "By Product" and dim_info['has_product']:
        selected_prod = st.selectbox(
            "Select Product",
            ["All Products (Faceted)"] + dim_info['product_names'],
            key="fit_prod_select"
        )
        
        if selected_prod == "All Products (Faceted)":
            _plot_model_fit_faceted(pred_df, ppc_results, dim_info, 'Product',
                                    hdi_low_pct, hdi_high_pct, periods, mmm)
        else:
            prod_mask = pred_df['Product'] == selected_prod
            prod_df = pred_df[prod_mask].copy()
            
            prod_samples = ppc_results['samples'][:, prod_mask.values]
            prod_df['y_pred_hdi_low'] = np.percentile(prod_samples, hdi_low_pct, axis=0)
            prod_df['y_pred_hdi_high'] = np.percentile(prod_samples, hdi_high_pct, axis=0)
            
            _plot_model_fit_single(prod_df, selected_prod, hdi_prob, show_residuals, mmm)


def _plot_model_fit_single(df, title_suffix, hdi_prob, show_residuals, mmm):
    """Plot model fit for a single aggregation level."""
    fig = go.Figure()
    
    x_axis = df['Period'].tolist()
    
    # HDI band
    if 'y_pred_hdi_low' in df.columns and 'y_pred_hdi_high' in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=df['y_pred_hdi_high'].tolist() + df['y_pred_hdi_low'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{int(hdi_prob*100)}% HDI',
            hoverinfo='skip'
        ))
    
    # Predicted mean
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df['y_pred_mean'],
        mode='lines',
        name='Predicted (Mean)',
        line=dict(color='rgb(99, 110, 250)', width=2)
    ))
    
    # Observed
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df['y_obs'],
        mode='markers',
        name='Observed',
        marker=dict(color='rgb(239, 85, 59)', size=6)
    ))
    
    fig.update_layout(
        title=f"Posterior Predictive Fit - {title_suffix}",
        xaxis_title="Period",
        yaxis_title=mmm.mff_config.kpi.name,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals
    if show_residuals:
        residuals = df['y_obs'].values - df['y_pred_mean'].values
        
        fig_resid = make_subplots(rows=1, cols=2, 
                                   subplot_titles=["Residuals Over Time", "Residual Distribution"])
        
        fig_resid.add_trace(go.Scatter(
            x=x_axis,
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
    
    # Fit statistics
    _display_fit_statistics(df['y_obs'].values, df['y_pred_mean'].values,
                           df.get('y_pred_hdi_low', pd.Series()).values if 'y_pred_hdi_low' in df else None,
                           df.get('y_pred_hdi_high', pd.Series()).values if 'y_pred_hdi_high' in df else None,
                           hdi_prob)


def _plot_model_fit_faceted(pred_df, ppc_results, dim_info, facet_col, 
                            hdi_low_pct, hdi_high_pct, periods, mmm):
    """Plot model fit faceted by geography or product."""
    facet_values = dim_info['geo_names'] if facet_col == 'Geography' else dim_info['product_names']
    n_facets = len(facet_values)
    n_cols = min(3, n_facets)
    n_rows = (n_facets + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=facet_values,
        horizontal_spacing=0.08,
        vertical_spacing=0.1
    )
    
    for i, facet_val in enumerate(facet_values):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        mask = pred_df[facet_col] == facet_val
        facet_df = pred_df[mask]
        
        x_axis = facet_df['Period'].tolist()
        
        # Predicted
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=facet_df['y_pred_mean'],
            mode='lines',
            name='Predicted',
            line=dict(color='rgb(99, 110, 250)', width=2),
            showlegend=(i == 0)
        ), row=row, col=col)
        
        # Observed
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=facet_df['y_obs'],
            mode='markers',
            name='Observed',
            marker=dict(color='rgb(239, 85, 59)', size=4),
            showlegend=(i == 0)
        ), row=row, col=col)
    
    fig.update_layout(
        title=f"Model Fit by {facet_col}",
        height=300 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_fit_statistics(y_obs, y_pred, hdi_low, hdi_high, hdi_prob):
    """Display model fit statistics."""
    st.markdown("### Model Fit Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # R-squared
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    
    # MAPE
    mape = np.mean(np.abs((y_obs - y_pred) / (y_obs + 1e-8))) * 100
    
    with col1:
        st.metric("R²", f"{r_squared:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:,.1f}")
    with col3:
        st.metric("MAPE", f"{mape:.1f}%")
    
    # Coverage
    if hdi_low is not None and hdi_high is not None and len(hdi_low) > 0:
        in_hdi = np.sum((y_obs >= hdi_low) & (y_obs <= hdi_high))
        coverage = in_hdi / len(y_obs) * 100
        expected_coverage = hdi_prob * 100
        delta = coverage - expected_coverage
        
        with col4:
            st.metric(f"HDI Coverage", f"{coverage:.0f}%",
                      delta=f"{delta:+.0f}% vs expected {expected_coverage:.0f}%",
                      delta_color="normal" if abs(delta) < 10 else "inverse")
    else:
        with col4:
            st.metric("HDI Coverage", "N/A")


def render_prior_vs_posterior(results, mmm):
    """Render prior vs posterior comparison plots."""
    st.subheader("🔄 Prior vs Posterior")
    
    st.markdown("""
    Compare the prior distributions (what we believed before seeing data) with the 
    posterior distributions (what we learned from the data). Priors are shown as 
    **dashed lines** with transparency.
    """)
    
    posterior = results.trace.posterior
    channel_names = mmm.channel_names
    
    # Sample from prior with caching
    with st.spinner("Loading prior samples..."):
        try:
            prior_samples = compute_prior_samples(mmm.model, n_samples=1000)
            prior = prior_samples.prior
        except Exception as e:
            st.error(f"Error sampling prior: {e}")
            return
    
    # Parameter selection
    param_categories = {
        "Media Coefficients (β)": [f"beta_{ch}" for ch in channel_names],
        "Adstock Parameters": [f"adstock_{ch}" for ch in channel_names],
        "Saturation Parameters (λ)": [f"sat_lam_{ch}" for ch in channel_names],
        "Other Parameters": ["intercept", "trend_slope", "sigma", "geo_sigma", "product_sigma"]
    }
    
    selected_category = st.selectbox(
        "Parameter Category",
        list(param_categories.keys())
    )
    
    params_to_plot = [p for p in param_categories[selected_category] if p in posterior]
    
    if not params_to_plot:
        st.info("No parameters available in this category")
        return
    
    # Plot settings
    col1, col2 = st.columns([1, 3])
    with col1:
        n_bins = st.slider("Histogram Bins", 20, 100, 50)
        show_stats = st.checkbox("Show Statistics", value=True)
    
    # Create plots
    n_params = len(params_to_plot)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=params_to_plot,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    stats_data = []
    
    for i, param in enumerate(params_to_plot):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Get posterior samples
        post_samples = posterior[param].values.flatten()
        
        # Get prior samples
        if param in prior:
            prior_samps = prior[param].values.flatten()
        else:
            prior_samps = None
        
        # Determine x range
        if prior_samps is not None:
            all_samples = np.concatenate([post_samples, prior_samps])
            x_min = np.percentile(all_samples, 0.5)
            x_max = np.percentile(all_samples, 99.5)
        else:
            x_min = np.percentile(post_samples, 0.5)
            x_max = np.percentile(post_samples, 99.5)
        
        # Create histogram bins
        bins = np.linspace(x_min, x_max, n_bins)
        
        # Prior histogram (dashed, translucent)
        if prior_samps is not None:
            prior_hist, bin_edges = np.histogram(prior_samps, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=prior_hist,
                mode='lines',
                name='Prior' if i == 0 else None,
                line=dict(color='rgba(150, 150, 150, 0.6)', width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(150, 150, 150, 0.15)',
                showlegend=(i == 0)
            ), row=row, col=col)
        
        # Posterior histogram (solid)
        post_hist, bin_edges = np.histogram(post_samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=post_hist,
            mode='lines',
            name='Posterior' if i == 0 else None,
            line=dict(color='rgb(99, 110, 250)', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.3)',
            showlegend=(i == 0)
        ), row=row, col=col)
        
        # Collect statistics
        stats_row = {
            'Parameter': param,
            'Prior Mean': f"{prior_samps.mean():.4f}" if prior_samps is not None else "N/A",
            'Prior Std': f"{prior_samps.std():.4f}" if prior_samps is not None else "N/A",
            'Posterior Mean': f"{post_samples.mean():.4f}",
            'Posterior Std': f"{post_samples.std():.4f}",
            'HDI 3%': f"{np.percentile(post_samples, 3):.4f}",
            'HDI 97%': f"{np.percentile(post_samples, 97):.4f}",
        }
        
        # Compute shrinkage (how much did we learn from data)
        if prior_samps is not None and prior_samps.std() > 0:
            shrinkage = 1 - (post_samples.std() / prior_samps.std())
            stats_row['Shrinkage'] = f"{shrinkage:.1%}"
        else:
            stats_row['Shrinkage'] = "N/A"
        
        stats_data.append(stats_row)
    
    fig.update_layout(
        height=300 * n_rows,
        title_text=f"Prior vs Posterior: {selected_category}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    if show_stats:
        st.markdown("### Parameter Statistics")
        st.markdown("""
        **Shrinkage** indicates how much the posterior narrowed compared to the prior.
        Higher shrinkage = more information gained from the data.
        """)
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Individual detailed comparison
    st.markdown("---")
    st.markdown("### Detailed Parameter Comparison")
    
    selected_param = st.selectbox("Select Parameter for Detailed View", params_to_plot)
    
    if selected_param:
        post_samples = posterior[selected_param].values.flatten()
        prior_samps = prior[selected_param].values.flatten() if selected_param in prior else None
        
        fig_detail = go.Figure()
        
        # Prior
        if prior_samps is not None:
            fig_detail.add_trace(go.Histogram(
                x=prior_samps,
                name='Prior',
                opacity=0.4,
                nbinsx=60,
                histnorm='probability density',
                marker_color='gray',
                marker_line=dict(color='gray', width=1)
            ))
            
            # Prior mean line
            fig_detail.add_vline(
                x=prior_samps.mean(),
                line_dash="dash",
                line_color="gray",
                opacity=0.7,
                annotation_text=f"Prior μ={prior_samps.mean():.3f}"
            )
        
        # Posterior
        fig_detail.add_trace(go.Histogram(
            x=post_samples,
            name='Posterior',
            opacity=0.7,
            nbinsx=60,
            histnorm='probability density',
            marker_color='rgb(99, 110, 250)'
        ))
        
        # Posterior mean and HDI
        post_mean = post_samples.mean()
        hdi_low = np.percentile(post_samples, 3)
        hdi_high = np.percentile(post_samples, 97)
        
        fig_detail.add_vline(
            x=post_mean,
            line_color="rgb(99, 110, 250)",
            line_width=2,
            annotation_text=f"Posterior μ={post_mean:.3f}"
        )
        
        fig_detail.add_vrect(
            x0=hdi_low, x1=hdi_high,
            fillcolor="rgba(99, 110, 250, 0.1)",
            line_width=0,
            annotation_text="94% HDI"
        )
        
        fig_detail.update_layout(
            title=f"Prior vs Posterior: {selected_param}",
            xaxis_title="Parameter Value",
            yaxis_title="Density",
            barmode='overlay',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)


def render_decomposition(results, mmm):
    """Render model decomposition: trend, seasonality, controls."""
    st.subheader("🌊 Model Decomposition")
    
    st.markdown("""
    Decompose the model into its components: trend, seasonality, media effects, 
    and control variables.
    """)
    
    posterior = results.trace.posterior
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    n_obs = mmm.n_obs
    
    # View level selection for multi-dimensional data
    if dim_info['has_geo'] or dim_info['has_product']:
        view_level = st.radio(
            "Aggregation Level",
            ["Total (Sum over all)", "Individual Observation Level"],
            horizontal=True,
            key="decomp_view_level"
        )
    else:
        view_level = "Total (Sum over all)"
    
    # Build period index
    periods = dim_info['periods']
    
    # Extract components at observation level first
    components = _extract_model_components(posterior, mmm, results, dim_info)
    
    # Aggregate if needed
    if view_level == "Total (Sum over all)" and (dim_info['has_geo'] or dim_info['has_product']):
        components = _aggregate_components_by_period(components, mmm, dim_info, periods, panel)
        x_axis = periods
        y_obs_agg = _aggregate_observed_by_period(mmm.y_raw, mmm, dim_info, periods, panel)
    else:
        x_axis = list(range(n_obs))
        y_obs_agg = mmm.y_raw
    
    # Plot selection
    plot_type = st.radio(
        "View",
        ["Stacked Decomposition", "Individual Components", "Control Analysis"],
        horizontal=True,
        key="decomp_plot_type"
    )
    
    if plot_type == "Stacked Decomposition":
        _plot_stacked_decomposition(components, x_axis, y_obs_agg, mmm)
    
    elif plot_type == "Individual Components":
        _plot_individual_component(components, x_axis, mmm)
    
    else:  # Control Analysis
        _plot_control_analysis(posterior, mmm, dim_info)


def _extract_model_components(posterior, mmm, results, dim_info):
    """Extract all model components from posterior.
    
    The model is fit on standardized data: y_std = (y - y_mean) / y_std
    To convert back to original scale:
    - Intercept: intercept * y_std + y_mean (includes the baseline)
    - Other components: component * y_std (deviations from baseline)
    """
    components = {}
    n_obs = mmm.n_obs
    
    # Intercept - add y_mean to get the baseline in original units
    # This is the "base sales" before any media/trend/seasonality effects
    if 'intercept' in posterior:
        intercept = posterior['intercept'].values.flatten().mean()
        # Convert: intercept_original = intercept * y_std + y_mean
        intercept_original = intercept * mmm.y_std + mmm.y_mean
        components['Baseline'] = {
            'mean': np.full(n_obs, intercept_original),
            'is_constant': True,
            'is_baseline': True  # Flag for stacking behavior
        }
    
    # Trend (deviation from baseline, scaled by y_std only)
    if 'trend_slope' in posterior:
        trend_slope = posterior['trend_slope'].values.flatten()
        t_scaled = mmm.t_scaled[mmm.time_idx]
        
        trend_samples = np.outer(trend_slope, t_scaled)
        components['Trend'] = {
            'mean': trend_samples.mean(axis=0) * mmm.y_std,
            'hdi_low': np.percentile(trend_samples, 3, axis=0) * mmm.y_std,
            'hdi_high': np.percentile(trend_samples, 97, axis=0) * mmm.y_std,
            'is_constant': False
        }
    
    # Seasonality (deviation from baseline)
    if hasattr(mmm, 'seasonality_features') and mmm.seasonality_features:
        for name, features in mmm.seasonality_features.items():
            season_var = f"season_{name}"
            if season_var in posterior:
                season_coef = posterior[season_var].values
                season_coef = season_coef.reshape(-1, features.shape[1])
                
                season_effects = np.dot(season_coef, features.T)
                season_at_obs = season_effects[:, mmm.time_idx]
                
                components[f'Seasonality ({name})'] = {
                    'mean': season_at_obs.mean(axis=0) * mmm.y_std,
                    'hdi_low': np.percentile(season_at_obs, 3, axis=0) * mmm.y_std,
                    'hdi_high': np.percentile(season_at_obs, 97, axis=0) * mmm.y_std,
                    'is_constant': False
                }
    
    # Media contributions (already in original scale from model)
    if results.channel_contributions is not None:
        for channel in mmm.channel_names:
            if channel in results.channel_contributions.columns:
                components[f'Media: {channel}'] = {
                    'mean': results.channel_contributions[channel].values,
                    'is_constant': False
                }
    
    # Control effects (deviation from baseline)
    if mmm.n_controls > 0 and 'beta_controls' in posterior:
        beta_controls = posterior['beta_controls'].values.reshape(-1, mmm.n_controls)
        
        for c, control_name in enumerate(mmm.control_names):
            control_data = mmm.X_controls[:, c]
            control_effects = np.outer(beta_controls[:, c], control_data)
            
            components[f'Control: {control_name}'] = {
                'mean': control_effects.mean(axis=0) * mmm.y_std,
                'hdi_low': np.percentile(control_effects, 3, axis=0) * mmm.y_std,
                'hdi_high': np.percentile(control_effects, 97, axis=0) * mmm.y_std,
                'is_constant': False
            }
    
    return components


def _aggregate_components_by_period(components, mmm, dim_info, periods, panel):
    """Aggregate components by period (sum over geos/products).
    
    For the baseline (intercept + y_mean), we need to multiply by the number
    of units being aggregated since it represents per-unit baseline sales.
    """
    
    # Build period mapping
    if isinstance(panel.index, pd.MultiIndex):
        period_col = mmm.mff_config.columns.period
        obs_periods = panel.index.get_level_values(period_col)
    else:
        obs_periods = periods[:mmm.n_obs]
    
    # Count observations per period (for scaling baseline)
    obs_per_period = {}
    for p in periods:
        obs_per_period[p] = (obs_periods == p).sum()
    
    agg_components = {}
    
    for name, data in components.items():
        is_baseline = data.get('is_baseline', False)
        
        if data.get('is_constant', False):
            # For baseline, multiply by number of units per period
            if is_baseline:
                baseline_per_obs = data['mean'][0]
                agg_values = [baseline_per_obs * obs_per_period[p] for p in periods]
                agg_components[name] = {
                    'mean': np.array(agg_values),
                    'is_constant': False,  # No longer constant after aggregation
                    'is_baseline': True
                }
            else:
                # Other constants - just repeat
                agg_components[name] = {
                    'mean': np.full(len(periods), data['mean'][0]),
                    'is_constant': True
                }
        else:
            # Aggregate by period (sum)
            mean_by_period = []
            hdi_low_by_period = [] if 'hdi_low' in data else None
            hdi_high_by_period = [] if 'hdi_high' in data else None
            
            for p in periods:
                mask = obs_periods == p
                mean_by_period.append(data['mean'][mask].sum())
                if hdi_low_by_period is not None:
                    hdi_low_by_period.append(data['hdi_low'][mask].sum())
                    hdi_high_by_period.append(data['hdi_high'][mask].sum())
            
            agg_data = {
                'mean': np.array(mean_by_period),
                'is_constant': False
            }
            if hdi_low_by_period is not None:
                agg_data['hdi_low'] = np.array(hdi_low_by_period)
                agg_data['hdi_high'] = np.array(hdi_high_by_period)
            
            agg_components[name] = agg_data
    
    return agg_components


def _aggregate_observed_by_period(y_raw, mmm, dim_info, periods, panel):
    """Aggregate observed y by period."""
    
    if isinstance(panel.index, pd.MultiIndex):
        period_col = mmm.mff_config.columns.period
        obs_periods = panel.index.get_level_values(period_col)
        
        y_agg = []
        for p in periods:
            mask = obs_periods == p
            y_agg.append(y_raw[mask].sum())
        return np.array(y_agg)
    
    return y_raw


def _plot_stacked_decomposition(components, x_axis, y_obs, mmm):
    """Plot stacked decomposition chart with baseline at bottom.
    
    The decomposition should sum to approximately the observed values:
    y = Baseline + Trend + Seasonality + Media + Controls + Residual
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    # Separate components by type for proper stacking order
    baseline_component = None
    trend_component = None
    seasonality_components = {}
    media_components = {}
    control_components = {}
    
    for name, data in components.items():
        if data.get('is_baseline', False) or name == 'Baseline':
            baseline_component = (name, data)
        elif name == 'Trend':
            trend_component = (name, data)
        elif 'Seasonality' in name:
            seasonality_components[name] = data
        elif 'Media:' in name:
            media_components[name] = data
        elif 'Control:' in name:
            control_components[name] = data
    
    # Build stacking order: Baseline -> Trend -> Seasonality -> Media -> Controls
    stack_order = []
    if baseline_component:
        stack_order.append(baseline_component)
    if trend_component:
        stack_order.append(trend_component)
    for name, data in seasonality_components.items():
        stack_order.append((name, data))
    for name, data in media_components.items():
        stack_order.append((name, data))
    for name, data in control_components.items():
        stack_order.append((name, data))
    
    # Create cumulative stacked areas
    cumulative = np.zeros(len(x_axis))
    
    for i, (name, data) in enumerate(stack_order):
        values = data['mean']
        color = colors[i % len(colors)]
        
        # For proper stacking, we add each component to the cumulative
        new_cumulative = cumulative + values
        
        fig.add_trace(go.Scatter(
            x=list(x_axis) + list(x_axis)[::-1],
            y=list(new_cumulative) + list(cumulative)[::-1],
            fill='toself',
            fillcolor=color,
            line=dict(width=0.5, color=color),
            name=name,
            hovertemplate=f'{name}<br>Value: %{{y:,.0f}}<extra></extra>'
        ))
        
        cumulative = new_cumulative
    
    # Add observed as line on top
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_obs,
        name='Observed',
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=4, color='black')
    ))
    
    # Add predicted (sum of components) as dashed line
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=cumulative,
        name='Predicted (Sum)',
        mode='lines',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Model Decomposition (Stacked Components)",
        xaxis_title="Period",
        yaxis_title=mmm.mff_config.kpi.name,
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show component totals
    st.markdown("### Component Contribution Summary")
    
    summary_data = []
    total_predicted = cumulative.sum()
    total_observed = y_obs.sum()
    
    for name, data in stack_order:
        comp_total = data['mean'].sum()
        pct_of_predicted = (comp_total / total_predicted * 100) if total_predicted != 0 else 0
        summary_data.append({
            'Component': name,
            'Total': comp_total,
            '% of Predicted': pct_of_predicted,
            'Mean per Period': data['mean'].mean()
        })
    
    summary_data.append({
        'Component': '**Predicted Total**',
        'Total': total_predicted,
        '% of Predicted': 100.0,
        'Mean per Period': cumulative.mean()
    })
    summary_data.append({
        'Component': '**Observed Total**',
        'Total': total_observed,
        '% of Predicted': (total_observed / total_predicted * 100) if total_predicted != 0 else 0,
        'Mean per Period': y_obs.mean()
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Total'] = summary_df['Total'].apply(lambda x: f"{x:,.0f}")
    summary_df['% of Predicted'] = summary_df['% of Predicted'].apply(lambda x: f"{x:.1f}%")
    summary_df['Mean per Period'] = summary_df['Mean per Period'].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _plot_individual_component(components, x_axis, mmm):
    """Plot individual component with HDI."""
    component_names = list(components.keys())
    selected_component = st.selectbox("Select Component", component_names, key="decomp_component")
    
    data = components[selected_component]
    is_baseline = data.get('is_baseline', False)
    
    fig = go.Figure()
    
    # HDI band if available
    if 'hdi_low' in data and 'hdi_high' in data:
        fig.add_trace(go.Scatter(
            x=list(x_axis) + list(x_axis)[::-1],
            y=list(data['hdi_high']) + list(data['hdi_low'])[::-1],
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='94% HDI',
            hoverinfo='skip'
        ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=data['mean'],
        mode='lines',
        name='Mean',
        line=dict(color='rgb(99, 110, 250)', width=2),
        fill='tozeroy' if 'hdi_low' not in data and not is_baseline else None,
        fillcolor='rgba(99, 110, 250, 0.3)' if 'hdi_low' not in data and not is_baseline else None
    ))
    
    # Only add zero line for non-baseline components
    if not is_baseline:
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    title_suffix = " (includes mean of target variable)" if is_baseline else ""
    fig.update_layout(
        title=f"Component: {selected_component}{title_suffix}",
        xaxis_title="Period",
        yaxis_title=f"Effect on {mmm.mff_config.kpi.name}" if not is_baseline else mmm.mff_config.kpi.name,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    mean_effect = np.mean(data['mean'])
    total_effect = np.sum(data['mean'])
    
    with col1:
        label = "Mean per Period" if not is_baseline else "Baseline per Period"
        st.metric(label, f"{mean_effect:,.1f}")
    with col2:
        st.metric("Total", f"{total_effect:,.1f}")
    with col3:
        st.metric("Std Dev", f"{np.std(data['mean']):,.1f}")
    
    if is_baseline:
        st.info("""
        **Baseline** represents the expected sales when all other effects (media, trend, 
        seasonality, controls) are zero. It includes the mean of the target variable 
        that was subtracted during standardization.
        """)


def _plot_control_analysis(posterior, mmm, dim_info):
    """Plot control variable analysis."""
    if mmm.n_controls == 0:
        st.info("No control variables in the model")
        return
    
    st.markdown("### Control Variable Effects")
    
    if 'beta_controls' in posterior:
        beta_controls = posterior['beta_controls'].values.reshape(-1, mmm.n_controls)
        
        # Box plot of coefficients
        fig = go.Figure()
        
        for c, control_name in enumerate(mmm.control_names):
            samples = beta_controls[:, c] * mmm.y_std
            
            fig.add_trace(go.Box(
                y=samples,
                name=control_name,
                boxpoints='outliers'
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Control Variable Coefficients (Original Scale)",
            yaxis_title="Effect per Unit Change",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        control_stats = []
        for c, control_name in enumerate(mmm.control_names):
            samples = beta_controls[:, c] * mmm.y_std
            
            control_raw = mmm.X_controls_raw[:, c] if hasattr(mmm, 'X_controls_raw') else mmm.X_controls[:, c]
            
            hdi_low = np.percentile(samples, 3)
            hdi_high = np.percentile(samples, 97)
            
            control_stats.append({
                'Control': control_name,
                'Coefficient Mean': samples.mean(),
                'Coefficient Std': samples.std(),
                'HDI 3%': hdi_low,
                'HDI 97%': hdi_high,
                'Data Mean': control_raw.mean(),
                'Significant': "Yes" if (hdi_low > 0 or hdi_high < 0) else "No"
            })
        
        st.dataframe(pd.DataFrame(control_stats).round(4), use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **Coefficient**: Change in KPI per unit change in control
        - **Significant**: "Yes" if 94% HDI excludes zero
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
@st.cache_resource
def plot_all_channels(response_data, colors, show_observed):
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
    return fig

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
    
    # Compute response curves for each channel with caching
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
    
    # Plot options
    with col2:
        plot_type = st.radio(
            "Plot Type",
            ["All Channels (Overlay)", "Individual Channel Plots", "Side by Side"],
            horizontal=True
        )
    
    colors = px.colors.qualitative.Set2
    
    if plot_type == "All Channels (Overlay)":
        # Single plot with all channels
        
        fig = plot_all_channels(response_data, colors, show_observed)
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Individual Channel Plots":
        # Separate plot for each channel
        selected_channel = st.selectbox("Select Channel", channel_names)
        
        data = next((d for d in response_data if d['channel'] == selected_channel), None)
        if data:
            fig = go.Figure()
            
            color = colors[channel_names.index(selected_channel) % len(colors)]
            color_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)"
            
            # HDI band
            fig.add_trace(go.Scatter(
                x=np.concatenate([data['x'], data['x'][::-1]]),
                y=np.concatenate([data['hdi_high'], data['hdi_low'][::-1]]),
                fill='toself',
                fillcolor=color_rgba,
                line=dict(color='rgba(0,0,0,0)'),
                name=f"{int(hdi_prob*100)}% HDI",
                hoverinfo='skip'
            ))
            
            # Mean curve
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['mean'],
                mode='lines',
                name='Mean',
                line=dict(color=color, width=3)
            ))
            
            # Add observed spend distribution as rug plot
            if show_observed:
                fig.add_trace(go.Scatter(
                    x=data['observed_spend'],
                    y=np.zeros_like(data['observed_spend']),
                    mode='markers',
                    marker=dict(symbol='line-ns', size=10, color=color, opacity=0.5),
                    name='Observed Spend',
                    hoverinfo='x'
                ))
                
                # Vertical lines for spend range
                fig.add_vline(x=data['spend_mean'], line_dash="dash", line_color="gray",
                             annotation_text="Mean Spend")
                fig.add_vrect(x0=np.percentile(data['observed_spend'], 25),
                             x1=np.percentile(data['observed_spend'], 75),
                             fillcolor="gray", opacity=0.1, line_width=0,
                             annotation_text="IQR")
            
            fig.update_layout(
                title=f"Response Curve: {selected_channel}",
                xaxis_title="Media Spend",
                yaxis_title="Contribution to Sales",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics for this channel
            st.markdown(f"### {selected_channel} Response Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate marginal effect at different spend levels
            with col1:
                # Effect at mean spend
                mean_idx = np.argmin(np.abs(data['x'] - data['spend_mean']))
                effect_at_mean = data['mean'][mean_idx]
                st.metric("Effect at Mean Spend", f"{effect_at_mean:,.0f}")
            
            with col2:
                # Effect at max observed spend
                max_idx = np.argmin(np.abs(data['x'] - data['spend_max']))
                effect_at_max = data['mean'][max_idx]
                st.metric("Effect at Max Spend", f"{effect_at_max:,.0f}")
            
            with col3:
                # Saturation level (what % of max effect are we at mean spend)
                max_effect = data['mean'][-1]
                saturation_pct = effect_at_mean / max_effect * 100 if max_effect > 0 else 0
                st.metric("Saturation at Mean", f"{saturation_pct:.0f}%")
            
            with col4:
                # Marginal effect (slope at mean spend)
                if mean_idx > 0 and mean_idx < len(data['x']) - 1:
                    dx = data['x'][mean_idx + 1] - data['x'][mean_idx - 1]
                    dy = data['mean'][mean_idx + 1] - data['mean'][mean_idx - 1]
                    marginal = dy / dx if dx > 0 else 0
                    st.metric("Marginal Effect", f"{marginal:.4f}")
    
    else:  # Side by Side
        # Grid of subplots
        n_channels = len(response_data)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[d['channel'] for d in response_data],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
        for i, data in enumerate(response_data):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            color = colors[i % len(colors)]
            color_rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)"
            
            # HDI band
            fig.add_trace(go.Scatter(
                x=np.concatenate([data['x'], data['x'][::-1]]),
                y=np.concatenate([data['hdi_high'], data['hdi_low'][::-1]]),
                fill='toself',
                fillcolor=color_rgba,
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)
            
            # Mean curve
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['mean'],
                mode='lines',
                name=data['channel'],
                line=dict(color=color, width=2),
                showlegend=False
            ), row=row, col=col)
            
            # Observed spend marker
            if show_observed:
                fig.add_vline(
                    x=data['spend_mean'],
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Response Curves by Channel",
            height=300 * n_rows,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(n_channels):
            row = i // n_cols + 1
            col = i % n_cols + 1
            if row == n_rows:
                fig.update_xaxes(title_text="Spend", row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="Effect", row=row, col=col)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimal spend analysis
    st.markdown("---")
    st.markdown("### 📊 Diminishing Returns Analysis")
    
    st.markdown("""
    This table shows the saturation level at current spend and the marginal return 
    (additional sales per additional dollar spent).
    """)
    
    analysis_data = []
    for data in response_data:
        spend_mean = data['spend_mean']
        spend_max = data['spend_max']
        
        # Find indices
        mean_idx = np.argmin(np.abs(data['x'] - spend_mean))
        max_idx = np.argmin(np.abs(data['x'] - spend_max))
        
        # Compute metrics
        effect_at_mean = data['mean'][mean_idx]
        effect_at_max = data['mean'][max_idx]
        max_possible = data['mean'][-1]
        
        # Saturation %
        saturation_mean = effect_at_mean / max_possible * 100 if max_possible > 0 else 0
        saturation_max = effect_at_max / max_possible * 100 if max_possible > 0 else 0
        
        # Marginal effect at mean (derivative)
        if mean_idx > 0 and mean_idx < len(data['x']) - 1:
            dx = data['x'][mean_idx + 1] - data['x'][mean_idx - 1]
            dy = data['mean'][mean_idx + 1] - data['mean'][mean_idx - 1]
            marginal_mean = dy / dx if dx > 0 else 0
        else:
            marginal_mean = 0
        
        # Marginal at max
        if max_idx > 0 and max_idx < len(data['x']) - 1:
            dx = data['x'][max_idx + 1] - data['x'][max_idx - 1]
            dy = data['mean'][max_idx + 1] - data['mean'][max_idx - 1]
            marginal_max = dy / dx if dx > 0 else 0
        else:
            marginal_max = 0
        
        analysis_data.append({
            'Channel': data['channel'],
            'Mean Spend': f"${spend_mean:,.0f}",
            'Max Spend': f"${spend_max:,.0f}",
            'Saturation @ Mean': f"{saturation_mean:.0f}%",
            'Saturation @ Max': f"{saturation_max:.0f}%",
            'Marginal Return @ Mean': f"{marginal_mean:.4f}",
            'Marginal Return @ Max': f"{marginal_max:.4f}",
        })
    
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    
    st.info("""
    💡 **Interpretation Guide:**
    - **Saturation %**: How much of the maximum possible effect you're achieving (higher = more saturated)
    - **Marginal Return**: Additional sales per additional dollar (higher = more room to grow)
    - Channels with low saturation and high marginal returns may benefit from increased investment
    - Channels with high saturation (>80%) may be at diminishing returns
    """)


def render_contributions(results, mmm):
    """Render channel contributions."""
    st.subheader("💰 Channel Contributions")
    
    if results.channel_contributions is None:
        st.warning("Channel contributions not available")
        return
    
    contrib = results.channel_contributions.copy()
    panel = st.session_state.panel
    dim_info = get_dimension_info(mmm, panel)
    
    # Add period and dimension columns (reset index to avoid ambiguity)
    contrib = contrib.reset_index(drop=True)
    
    if isinstance(panel.index, pd.MultiIndex):
        period_col = mmm.mff_config.columns.period
        contrib['Period'] = panel.index.get_level_values(period_col).values
        if dim_info['has_geo']:
            geo_col = mmm.mff_config.columns.geography
            contrib['Geography'] = panel.index.get_level_values(geo_col).values
        if dim_info['has_product']:
            prod_col = mmm.mff_config.columns.product
            contrib['Product'] = panel.index.get_level_values(prod_col).values
    else:
        contrib['Period'] = dim_info['periods'][:len(contrib)]
    
    # View level selection
    if dim_info['has_geo'] or dim_info['has_product']:
        view_options = ["Total (Sum over all)"]
        if dim_info['has_geo']:
            view_options.append("By Geography")
        if dim_info['has_product']:
            view_options.append("By Product")
        
        view_level = st.radio(
            "Aggregation Level",
            view_options,
            horizontal=True,
            key="contrib_view_level"
        )
    else:
        view_level = "Total (Sum over all)"
    
    # Channel selection for detailed views
    channel_names = mmm.channel_names
    
    if view_level == "Total (Sum over all)":
        _render_contributions_total(contrib, channel_names, dim_info, mmm, panel)
    
    elif view_level == "By Geography":
        selected_geo = st.selectbox(
            "Select Geography",
            ["All Geographies"] + dim_info['geo_names'],
            key="contrib_geo_select"
        )
        
        if selected_geo == "All Geographies":
            _render_contributions_by_dimension(contrib, channel_names, 'Geography', 
                                               dim_info['geo_names'], dim_info, mmm, panel)
        else:
            geo_contrib = contrib[contrib['Geography'] == selected_geo]
            _render_contributions_single(geo_contrib, channel_names, selected_geo, dim_info, mmm, panel)
    
    elif view_level == "By Product":
        selected_prod = st.selectbox(
            "Select Product",
            ["All Products"] + dim_info['product_names'],
            key="contrib_prod_select"
        )
        
        if selected_prod == "All Products":
            _render_contributions_by_dimension(contrib, channel_names, 'Product',
                                               dim_info['product_names'], dim_info, mmm, panel)
        else:
            prod_contrib = contrib[contrib['Product'] == selected_prod]
            _render_contributions_single(prod_contrib, channel_names, selected_prod, dim_info, mmm, panel)


def _render_contributions_total(contrib, channel_names, dim_info, mmm, panel):
    """Render total contributions aggregated over all dimensions."""
    
    # Aggregate by period
    period_contrib = contrib.groupby('Period')[channel_names].sum().reset_index()
    
    # Total contributions pie chart
    st.markdown("### Total Contributions by Channel")
    
    totals = contrib[channel_names].sum()
    total_contrib = totals.sum()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.pie(
            values=totals.values,
            names=totals.index,
            title="Share of Total Media Contribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Contribution Totals:**")
        for ch, val in totals.items():
            pct = val / total_contrib * 100 if total_contrib > 0 else 0
            st.metric(ch, f"{val:,.0f}", f"{pct:.1f}%")
    
    # Time series - stacked area
    st.markdown("### Contributions Over Time")
    
    fig = go.Figure()
    for col in channel_names:
        fig.add_trace(go.Scatter(
            x=period_contrib['Period'],
            y=period_contrib[col],
            name=col,
            mode='lines',
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Stacked Channel Contributions Over Time",
        xaxis_title="Period",
        yaxis_title="Contribution",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROAS estimation
    _render_roas_analysis(totals, channel_names, mmm, panel)


def _render_contributions_by_dimension(contrib, channel_names, dim_col, dim_values, dim_info, mmm, panel):
    """Render contributions comparison across a dimension."""
    
    # Aggregate by dimension
    dim_totals = contrib.groupby(dim_col)[channel_names].sum()
    
    st.markdown(f"### Contributions by {dim_col}")
    
    # Grouped bar chart
    fig = go.Figure()
    
    for ch in channel_names:
        fig.add_trace(go.Bar(
            x=dim_totals.index,
            y=dim_totals[ch],
            name=ch
        ))
    
    fig.update_layout(
        title=f"Channel Contributions by {dim_col}",
        xaxis_title=dim_col,
        yaxis_title="Total Contribution",
        barmode='group',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Contribution share by dimension
    st.markdown(f"### Contribution Share by {dim_col}")
    
    dim_shares = dim_totals.div(dim_totals.sum(axis=1), axis=0) * 100
    
    fig2 = go.Figure()
    for ch in channel_names:
        fig2.add_trace(go.Bar(
            x=dim_shares.index,
            y=dim_shares[ch],
            name=ch
        ))
    
    fig2.update_layout(
        title=f"Channel Mix by {dim_col}",
        xaxis_title=dim_col,
        yaxis_title="Share (%)",
        barmode='stack',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary table
    st.markdown("### Summary Table")
    summary_df = dim_totals.copy()
    summary_df['Total'] = summary_df.sum(axis=1)
    summary_df = summary_df.round(0)
    st.dataframe(summary_df, use_container_width=True)


def _render_contributions_single(contrib_df, channel_names, title, dim_info, mmm, panel):
    """Render contributions for a single dimension value."""
    
    period_contrib = contrib_df.groupby('Period')[channel_names].sum().reset_index()
    
    st.markdown(f"### Contributions for {title}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series
        fig = go.Figure()
        for col in channel_names:
            fig.add_trace(go.Scatter(
                x=period_contrib['Period'],
                y=period_contrib[col],
                name=col,
                mode='lines',
                stackgroup='one'
            ))
        
        fig.update_layout(
            title=f"Contributions Over Time - {title}",
            xaxis_title="Period",
            yaxis_title="Contribution",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Totals
        totals = contrib_df[channel_names].sum()
        total = totals.sum()
        
        st.markdown("**Totals:**")
        for ch, val in totals.items():
            pct = val / total * 100 if total > 0 else 0
            st.metric(ch, f"{val:,.0f}", f"{pct:.1f}%")


def _render_roas_analysis(totals, channel_names, mmm, panel):
    """Render ROAS analysis section."""
    st.markdown("### ROAS Estimates")
    
    if panel.X_media is not None:
        spend_totals = panel.X_media.sum()
        
        roas_data = []
        for ch in channel_names:
            if ch in totals.index and ch in spend_totals.index:
                contrib_val = totals[ch]
                spend_val = spend_totals[ch]
                roas = contrib_val / spend_val if spend_val > 0 else 0
                roas_data.append({
                    'Channel': ch,
                    'Total Contribution': contrib_val,
                    'Total Spend': spend_val,
                    'ROAS': roas
                })
        
        if roas_data:
            roas_df = pd.DataFrame(roas_data)
            
            fig = px.bar(
                roas_df,
                x='Channel',
                y='ROAS',
                color='Channel',
                title="Return on Ad Spend (ROAS) by Channel",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(roas_df.round(2), use_container_width=True)


def render_summary(results, mmm):
    """Render model summary."""
    st.subheader("📋 Full Model Summary")
    
    # ArviZ summary
    summary = results.summary()
    
    # Filter to key parameters
    key_params = [p for p in summary.index if any(x in p for x in ['beta', 'sigma', 'adstock', 'sat_lam', 'trend', 'geo'])]
    
    if key_params:
        st.markdown("### Key Parameters")
        st.dataframe(summary.loc[key_params].round(4), use_container_width=True)
    
    st.markdown("### Full Parameter Summary")
    st.dataframe(summary.round(4), use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Summary CSV
        csv_summary = summary.to_csv()
        st.download_button(
            "📥 Download Summary (CSV)",
            csv_summary,
            "mmm_summary.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Contributions CSV
        if results.channel_contributions is not None:
            csv_contrib = results.channel_contributions.to_csv()
            st.download_button(
                "📥 Download Contributions (CSV)",
                csv_contrib,
                "mmm_contributions.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col3:
        # Full trace (NetCDF)
        st.info("💡 Full trace can be saved with arviz.to_netcdf()")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    
    # Main content tabs
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
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
    result_tabs = st.tabs(["📊 Diagnostics", "📉 Posteriors", "📈 Response Curves", "💰 Contributions", "📋 Summary"])
    
    with result_tabs[0]:
        render_diagnostics(results)
    
    with result_tabs[1]:
        render_posteriors(results, mmm)
    
    with result_tabs[2]:
        render_response_curves(results, mmm)
    
    with result_tabs[3]:
        render_contributions(results, mmm)
    
    with result_tabs[4]:
        render_summary(results, mmm)


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


def render_response_curves(results, mmm):
    """Render response curves with HDI bands."""
    st.subheader("📈 Response Curves")
    
    st.markdown("""
    Response curves show how media spend translates to sales effect after accounting for 
    saturation (diminishing returns). The shaded bands represent the 94% HDI (Highest Density Interval).
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
                             help="Width of the credible interval")
        n_points = st.slider("Curve Resolution", 50, 200, 100,
                            help="Number of points on the curve")
        show_observed = st.checkbox("Show Observed Spend", value=True,
                                   help="Mark the range of observed spend values")
    
    # Compute response curves for each channel
    response_data = []
    
    for c, channel in enumerate(channel_names):
        # Get posterior samples
        sat_lam_var = f"sat_lam_{channel}"
        beta_var = f"beta_{channel}"
        
        if sat_lam_var not in posterior or beta_var not in posterior:
            continue
        
        sat_lam_samples = posterior[sat_lam_var].values.flatten()
        beta_samples = posterior[beta_var].values.flatten()
        
        # Get spend range for this channel
        spend_raw = X_media_raw[:, c]
        spend_min, spend_max = 0, spend_raw.max()
        spend_mean = spend_raw.mean()
        
        # Scale factor (same as used in model)
        # The model scales media to [0, 1] by dividing by max after adstock
        # For simplicity, we'll show the curve in original spend units
        
        # Generate x values (original spend scale)
        x_original = np.linspace(0, spend_max * 1.2, n_points)
        
        # Scale x to [0, 1] as the model does (approximately)
        x_scaled = x_original / (spend_max + 1e-8)
        
        # Compute curves for each posterior sample
        n_samples = len(sat_lam_samples)
        curves = np.zeros((n_samples, n_points))
        
        for i in range(n_samples):
            # Logistic saturation: 1 - exp(-lambda * x)
            saturated = 1 - np.exp(-sat_lam_samples[i] * x_scaled)
            # Scale by beta (this gives effect in standardized y units)
            curves[i, :] = beta_samples[i] * saturated
        
        # Convert to original y scale
        curves_original = curves * mmm.y_std
        
        # Compute statistics
        curve_mean = curves_original.mean(axis=0)
        hdi_low = np.percentile(curves_original, (1 - hdi_prob) / 2 * 100, axis=0)
        hdi_high = np.percentile(curves_original, (1 + hdi_prob) / 2 * 100, axis=0)
        
        response_data.append({
            'channel': channel,
            'x': x_original,
            'mean': curve_mean,
            'hdi_low': hdi_low,
            'hdi_high': hdi_high,
            'spend_min': spend_min,
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
        fig = go.Figure()
        
        for i, data in enumerate(response_data):
            color = colors[i % len(colors)]
            color_rgba = color.replace("rgb", "").replace("(", "").replace(")", "")
            r, g, b = map(lambda x: int(x), color_rgba.split(","))
            color_rgba = f"rgba({r}, {g}, {b}, 0.2)"
            print(color_rgba)
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
    
    elif plot_type == "Individual Channel Plots":
        # Separate plot for each channel
        selected_channel = st.selectbox("Select Channel", channel_names)
        
        data = next((d for d in response_data if d['channel'] == selected_channel), None)
        if data:
            fig = go.Figure()
            
            color = colors[channel_names.index(selected_channel) % len(colors)]
            color_rgb = color.replace("rgb", "").replace("(", "").replace(")", "")
            r, g, b = map(lambda x: int(x), color_rgb.split(","))
            color_rgba = f"rgba({r}, {g}, {b}, 0.3)"
            
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
            color_rgb = color.replace("rgb", "").replace("(", "").replace(")", "")
            r, g, b = map(lambda x: int(x), color_rgb.split(","))
            color_rgba = f"rgba({r}, {g}, {b}, 0.3)"
            
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
    
    contrib = results.channel_contributions
    
    # Total contributions
    st.markdown("### Total Contributions by Channel")
    
    totals = contrib.sum()
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
            pct = val / total_contrib * 100
            st.metric(ch, f"{val:,.0f}", f"{pct:.1f}%")
    
    # Time series
    st.markdown("### Contributions Over Time")
    
    # Reset index for plotting
    if isinstance(contrib.index, pd.MultiIndex):
        contrib_plot = contrib.reset_index()
        period_col = [c for c in contrib_plot.columns if 'period' in c.lower() or 'date' in c.lower()]
        if period_col:
            contrib_plot = contrib_plot.groupby(period_col[0])[mmm.channel_names].sum()
    else:
        contrib_plot = contrib
    
    fig = go.Figure()
    for col in contrib_plot.columns:
        fig.add_trace(go.Scatter(
            x=contrib_plot.index,
            y=contrib_plot[col],
            name=col,
            mode='lines',
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Stacked Channel Contributions Over Time",
        xaxis_title="Period",
        yaxis_title="Contribution (original scale)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROAS estimation
    st.markdown("### ROAS Estimates")
    
    panel = st.session_state.panel
    if panel.X_media is not None:
        spend_totals = panel.X_media.sum()
        
        roas_data = []
        for ch in mmm.channel_names:
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
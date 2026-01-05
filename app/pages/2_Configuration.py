"""
Configuration Page.

Create and manage model configurations with dataset-aware variable selection.
Single-column layout with expandable sections for detailed settings.
"""

import streamlit as st
from typing import Any

from api_client import (
    get_api_client,
    fetch_configs,
    fetch_datasets,
    clear_config_cache,
    APIError,
    ConfigInfo,
)
from components import (
    apply_custom_css,
    page_header,
    format_datetime,
    display_api_error,
    init_session_state,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Configuration | MMM Framework",
    page_icon="⚙️",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_config_id=None,
    config_dataset_id=None,
    config_dataset_variables=[],
    # KPI settings
    kpi_name="",
    kpi_dimensions="National",
    # Media channels
    media_channels=[],
    media_configs=[],
    # Control variables
    control_variables=[],
    control_configs=[],
    # Dimension alignment
    geo_allocation="equal",
    # Model settings
    n_chains=4,
    n_draws=1000,
    n_tune=1000,
    target_accept=0.95,
    # Trend settings
    trend_type="Linear",
    trend_settings={},
    # Seasonality
    yearly_order=2,
    # Hierarchical
    pool_geo=True,
    # Performance
    use_numpyro=False,
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_dataset_variables(client, data_id: str) -> list[str]:
    """Fetch variable names from a dataset."""
    try:
        dataset = client.get_dataset(data_id, include_preview=False)
        return dataset.variables
    except Exception:
        return []


def get_dataset_dimensions(client, data_id: str) -> dict:
    """Fetch dimension values from a dataset."""
    try:
        dataset = client.get_dataset(data_id, include_preview=False)
        return {
            "geographies": dataset.get_geographies(),  # Method call
            "products": dataset.get_products(),  # Method call
        }
    except Exception:
        return {"geographies": [], "products": []}


# =============================================================================
# Configuration Form - Single Column Layout
# =============================================================================


@st.fragment
def render_config_form():
    """Render the configuration creation form in single-column layout."""

    # Get available datasets
    try:
        client = get_api_client()
        datasets = fetch_datasets(client)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return

    if not datasets:
        st.warning("⚠️ Please upload a dataset first in the Data Management page.")
        return

    dataset_options = {d.filename: d.data_id for d in datasets}

    # -------------------------------------------------------------------------
    # Dataset Selection
    # -------------------------------------------------------------------------
    st.subheader("📁 Dataset Selection")

    selected_dataset = st.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        help="Choose the dataset to use for model configuration",
    )

    if selected_dataset:
        data_id = dataset_options[selected_dataset]

        # Fetch variables if dataset changed
        if st.session_state.config_dataset_id != data_id:
            variables = get_dataset_variables(client, data_id)
            dimensions = get_dataset_dimensions(client, data_id)
            st.session_state.config_dataset_id = data_id
            st.session_state.config_dataset_variables = variables
            st.session_state.config_dataset_dimensions = dimensions

        variables = st.session_state.config_dataset_variables
        dimensions = st.session_state.get("config_dataset_dimensions", {})
        geographies = dimensions.get("geographies", [])
        products = dimensions.get("products", [])

        if not variables:
            st.warning("⚠️ No variables found in dataset.")
            return

        st.info(f"📊 Dataset contains {len(variables)} variables")
    else:
        return

    st.markdown("---")

    # -------------------------------------------------------------------------
    # KPI Configuration
    # -------------------------------------------------------------------------
    st.subheader("🎯 KPI Configuration")

    kpi_name = st.selectbox(
        "KPI Variable",
        options=variables,
        index=variables.index("Sales") if "Sales" in variables else 0,
        help="Select the dependent variable (e.g., Sales, Revenue)",
    )
    st.session_state.kpi_name = kpi_name

    # KPI Granularity options
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

    # -------------------------------------------------------------------------
    # Media Channels
    # -------------------------------------------------------------------------
    st.subheader("📺 Media Channels")

    # Filter out KPI and common control names for media channel options
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

    # Media channel configuration
    media_configs = []
    if media_channels:
        st.markdown("**Channel Settings:**")

        for channel in media_channels:
            with st.expander(f"📺 {channel}", expanded=False):
                adstock_max = st.slider(
                    f"Max Adstock Lag (weeks)",
                    min_value=1,
                    max_value=12,
                    value=8,
                    key=f"adstock_{channel}",
                    help="Maximum lag for carryover effects",
                )

                channel_level = st.selectbox(
                    "Data Level",
                    options=["National", "By Geography"],
                    key=f"level_{channel}",
                    help="At what level is the media data recorded?",
                )

                saturation_type = st.selectbox(
                    "Saturation Function",
                    options=["Logistic", "Hill"],
                    index=0,
                    key=f"sat_{channel}",
                    help="How media response diminishes at high spend levels",
                )

                media_configs.append(
                    {
                        "name": channel,
                        "adstock_lmax": adstock_max,
                        "level": channel_level,
                        "saturation": saturation_type,
                    }
                )

    st.session_state.media_channels = media_channels
    st.session_state.media_configs = media_configs

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Control Variables
    # -------------------------------------------------------------------------
    st.subheader("📊 Control Variables")

    # Filter out KPI and media channels for control variable options
    potential_controls = [v for v in variables if v not in [kpi_name] + media_channels]

    control_vars = st.multiselect(
        "Select Control Variables",
        options=potential_controls,
        default=[v for v in ["Price", "Distribution"] if v in potential_controls],
        help="Select variables that control for external factors",
    )

    # Control variable configuration
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

                control_level = st.selectbox(
                    "Data Level",
                    options=["National", "By Geography"],
                    index=0,
                    key=f"ctrl_level_{control}",
                    help="At what level is the control data recorded?",
                )

                control_configs.append(
                    {
                        "name": control,
                        "allow_negative": allow_neg,
                        "level": control_level,
                    }
                )

    st.session_state.control_variables = control_vars
    st.session_state.control_configs = control_configs

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Dimension Alignment
    # -------------------------------------------------------------------------
    if kpi_dimensions != "National" and any(
        m.get("level") == "National" for m in media_configs
    ):
        st.subheader("🗺️ Dimension Alignment")

        st.info("ℹ️ National media will be allocated to geographic/product dimensions")

        allocation_method = st.selectbox(
            "Allocation Method",
            options=["Equal", "By Population", "By Sales", "Custom"],
            help="How to distribute national media to sub-national levels",
        )
        st.session_state.geo_allocation = (
            allocation_method.lower().replace(" ", "_").replace("by_", "")
        )

        st.markdown("---")

    # -------------------------------------------------------------------------
    # MCMC Settings
    # -------------------------------------------------------------------------
    st.subheader("🎛️ MCMC Settings")

    n_chains = st.slider(
        "Number of Chains",
        min_value=1,
        max_value=8,
        value=4,
        help="More chains = better convergence diagnostics",
    )
    st.session_state.n_chains = n_chains

    n_draws = st.slider(
        "Draws per Chain",
        min_value=500,
        max_value=4000,
        value=1000,
        step=500,
        help="More draws = more precise estimates",
    )
    st.session_state.n_draws = n_draws

    n_tune = st.slider(
        "Tuning Samples",
        min_value=500,
        max_value=2000,
        value=1000,
        step=500,
        help="Samples used for adaptation",
    )
    st.session_state.n_tune = n_tune

    target_accept = st.slider(
        "Target Accept Rate",
        min_value=0.8,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Higher = fewer divergences but slower",
    )
    st.session_state.target_accept = target_accept

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Trend Configuration
    # -------------------------------------------------------------------------
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
    st.session_state.trend_type = trend_type

    # Trend-specific settings
    trend_settings = {}

    if trend_type == "Linear":
        with st.expander("Linear Trend Settings", expanded=False):
            trend_settings["growth_prior_mu"] = st.slider(
                "Growth Prior Mean",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.05,
                help="Expected growth rate direction",
            )
            trend_settings["growth_prior_sigma"] = st.slider(
                "Growth Prior Sigma",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Uncertainty in growth rate",
            )

    elif trend_type == "Piecewise (Prophet-style)":
        with st.expander("Piecewise Trend Settings", expanded=True):
            trend_settings["n_changepoints"] = st.slider(
                "Number of Changepoints",
                min_value=5,
                max_value=30,
                value=10,
                help="More changepoints = more flexible trend",
            )
            trend_settings["changepoint_range"] = st.slider(
                "Changepoint Range",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                help="Fraction of time series where changepoints can occur",
            )
            trend_settings["changepoint_prior_scale"] = st.slider(
                "Changepoint Prior Scale",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Smaller = smoother trend, larger = more flexible",
            )

    elif trend_type == "Spline":
        with st.expander("Spline Trend Settings", expanded=True):
            trend_settings["n_knots"] = st.slider(
                "Number of Knots",
                min_value=5,
                max_value=30,
                value=10,
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
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Prior standard deviation for spline coefficients",
            )

    elif trend_type == "Gaussian Process":
        with st.expander("GP Trend Settings", expanded=True):
            trend_settings["gp_lengthscale_mu"] = st.slider(
                "Lengthscale Prior Mean",
                min_value=0.1,
                max_value=0.7,
                value=0.3,
                step=0.05,
                help="Expected smoothness (fraction of time series)",
            )
            trend_settings["gp_lengthscale_sigma"] = st.slider(
                "Lengthscale Prior Sigma",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Uncertainty in smoothness",
            )
            trend_settings["gp_amplitude_sigma"] = st.slider(
                "Amplitude Prior Sigma",
                min_value=0.1,
                max_value=1.5,
                value=0.5,
                step=0.1,
                help="Prior on trend magnitude",
            )
            trend_settings["gp_n_basis"] = st.slider(
                "Number of Basis Functions",
                min_value=10,
                max_value=40,
                value=20,
                help="More = better approximation but slower",
            )

    st.session_state.trend_settings = trend_settings

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Seasonality
    # -------------------------------------------------------------------------
    st.subheader("🌊 Seasonality")

    yearly_order = st.slider(
        "Yearly Seasonality Order",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of Fourier terms for yearly seasonality (0 = disabled)",
    )
    st.session_state.yearly_order = yearly_order

    if yearly_order > 0:
        with st.expander("Seasonality Details", expanded=False):
            st.markdown(
                f"""
            **Fourier Terms:** {yearly_order}
            
            This will create {yearly_order * 2} seasonality features 
            (sin and cos terms for each order).
            
            - Order 1: Annual cycle
            - Order 2: Semi-annual patterns  
            - Order 3+: Finer seasonal variations
            """
            )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Hierarchical Structure
    # -------------------------------------------------------------------------
    st.subheader("🏗️ Hierarchical Structure")

    if kpi_dimensions != "National":
        pool_geo = st.checkbox(
            "Pool Across Geographies",
            value=True,
            help="Share information between geographic units for more stable estimates",
        )
        st.session_state.pool_geo = pool_geo

        if pool_geo:
            with st.expander("Hierarchical Details", expanded=False):
                st.markdown(
                    """
                **Partial Pooling Benefits:**
                
                - Borrows strength from other geographies
                - Reduces overfitting for small markets
                - Provides uncertainty-aware estimates
                - Better handles sparse data
                """
                )
    else:
        st.info("ℹ️ Hierarchical pooling is only available for multi-geography models")
        st.session_state.pool_geo = False

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Performance Options
    # -------------------------------------------------------------------------
    st.subheader("⚡ Performance")

    use_numpyro = st.checkbox(
        "Use NumPyro (JAX)",
        value=False,
        help="Faster sampling with JAX backend (requires numpyro installed)",
    )
    st.session_state.use_numpyro = use_numpyro

    if use_numpyro:
        st.info("🚀 NumPyro will use JAX for faster GPU/TPU sampling if available")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Configuration Summary & Estimated Time
    # -------------------------------------------------------------------------
    st.subheader("📋 Configuration Summary")

    n_params = len(media_channels) * 3 + len(control_vars) + 5
    est_time = (n_chains * (n_draws + n_tune) * n_params) / 5000

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Media Channels", len(media_channels))
    with summary_cols[1]:
        st.metric("Controls", len(control_vars))
    with summary_cols[2]:
        st.metric("Est. Parameters", n_params)
    with summary_cols[3]:
        st.metric("Est. Time", f"{est_time:.0f}-{est_time*2:.0f}s")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Build Configuration Button
    # -------------------------------------------------------------------------
    config_name = st.text_input(
        "Configuration Name",
        value=f"config_{selected_dataset.replace('.', '_')}",
        help="Give your configuration a descriptive name",
    )

    if st.button("🔧 Save Configuration", type="primary", use_container_width=True):
        save_configuration(client, data_id, config_name)


def save_configuration(client, data_id: str, config_name: str):
    """Save the configuration to the backend."""
    try:
        # Convert KPI dimensions string to list format
        kpi_dims_map = {
            "National": [],
            "By Geography": ["Geography"],
            "By Product": ["Product"],
            "By Geography & Product": ["Geography", "Product"],
        }
        kpi_dimensions = kpi_dims_map.get(st.session_state.kpi_dimensions, [])

        # Convert trend type to lowercase API format
        trend_type_map = {
            "None": "none",
            "Linear": "linear",
            "Piecewise (Prophet-style)": "piecewise",
            "Spline": "spline",
            "Gaussian Process": "gaussian_process",
        }
        trend_type = trend_type_map.get(st.session_state.trend_type, "linear")

        # Convert saturation type to dict format
        def get_saturation_config(sat_type: str) -> dict:
            if sat_type == "Hill":
                return {"type": "hill"}
            return {"type": "logistic"}

        # Build mff_config structure
        mff_config = {
            "kpi": {
                "name": st.session_state.kpi_name,
                "dimensions": kpi_dimensions,
            },
            "media_channels": [
                {
                    "name": ch["name"],
                    "adstock_lmax": ch.get("adstock_lmax", 8),
                    "level": ch.get("level", "National")
                    .lower()
                    .replace(" ", "_")
                    .replace("by_", ""),
                    "saturation": get_saturation_config(
                        ch.get("saturation", "Logistic")
                    ),
                }
                for ch in st.session_state.media_configs
            ],
            "controls": [
                {
                    "name": ctrl["name"],
                    "allow_negative": ctrl.get("allow_negative", False),
                    "level": ctrl.get("level", "National")
                    .lower()
                    .replace(" ", "_")
                    .replace("by_", ""),
                }
                for ctrl in st.session_state.control_configs
            ],
            "alignment": {
                "method": st.session_state.geo_allocation,
            },
        }

        # Build model_settings structure
        model_settings = {
            "inference_method": (
                "bayesian_numpyro" if st.session_state.use_numpyro else "bayesian_pymc"
            ),
            "n_chains": st.session_state.n_chains,
            "n_draws": st.session_state.n_draws,
            "n_tune": st.session_state.n_tune,
            "target_accept": st.session_state.target_accept,
            "trend": {
                "type": trend_type,
                **st.session_state.trend_settings,
            },
            "seasonality": {
                "yearly_order": st.session_state.yearly_order,
            },
            "hierarchical": {
                "pool_geo": st.session_state.pool_geo,
            },
        }

        # Build the configuration request
        config_data = {
            "name": config_name,
            "description": f"Configuration for dataset {data_id}",
            "mff_config": mff_config,
            "model_settings": model_settings,
        }

        # Submit to API
        result = client.create_config(config_data)

        st.success(f"✅ Configuration '{config_name}' saved successfully!")

        # Clear cache to refresh config list
        clear_config_cache()

        # Store the config ID
        if isinstance(result, dict):
            st.session_state.selected_config_id = result.get("config_id")

    except APIError as e:
        display_api_error(e, "Failed to save configuration")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")


# =============================================================================
# Configuration List
# =============================================================================


@st.fragment
def render_config_list():
    """Render the list of existing configurations."""
    st.subheader("📑 Saved Configurations")

    try:
        client = get_api_client()
        configs = fetch_configs(client)
    except Exception as e:
        st.error(f"Error loading configurations: {e}")
        return

    if not configs:
        st.info("No configurations found. Create one above.")
        return

    for config in configs:
        with st.expander(f"📄 {config.name}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**ID:** `{config.config_id}`")
                if config.description:
                    st.markdown(f"**Description:** {config.description}")
                st.markdown(f"**Created:** {format_datetime(config.created_at)}")
                if config.updated_at:
                    st.markdown(f"**Updated:** {format_datetime(config.updated_at)}")

                # Display config summary from mff_config using methods
                st.markdown(f"**KPI:** {config.get_kpi_name()}")
                st.markdown(f"**Media Channels:** {len(config.get_media_channels())}")
                st.markdown(f"**Controls:** {len(config.get_controls())}")
                st.markdown(f"**Inference:** {config.get_inference_method()}")

            with col2:
                if st.button("🗑️ Delete", key=f"del_{config.config_id}"):
                    try:
                        client.delete_config(config.config_id)
                        clear_config_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

                if st.button("📋 Load", key=f"load_{config.config_id}"):
                    st.session_state.selected_config_id = config.config_id
                    st.info(f"Configuration '{config.name}' selected")


# =============================================================================
# Main Page
# =============================================================================


def main():
    """Main page entry point."""
    page_header(
        "⚙️ Model Configuration",
        "Configure your Marketing Mix Model with dataset-aware variable selection.",
    )

    # Single column layout with expandable form
    with st.expander(
        "➕ Create New Configuration", expanded=not st.session_state.selected_config_id
    ):
        render_config_form()

    st.markdown("---")

    render_config_list()


if __name__ == "__main__":
    main()

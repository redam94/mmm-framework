"""
Configuration Page.

Create and manage model configurations with dataset-aware variable selection.
Single-column layout with expandable sections for detailed settings.
Provides flexible model configuration matching the full API capability.
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
    kpi_log_transform=False,
    # Media channels
    media_channels=[],
    media_configs=[],
    # Control variables
    control_variables=[],
    control_configs=[],
    # Dimension alignment
    geo_allocation="equal",
    product_allocation="sales",
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
    monthly_order=0,
    weekly_order=0,
    # Hierarchical
    pool_geo=True,
    pool_product=True,
    use_non_centered=True,
    # Control selection
    control_selection_method="none",
    # Performance
    use_numpyro=True,
)


# =============================================================================
# Constants
# =============================================================================

ADSTOCK_TYPES = ["Geometric", "Weibull", "Delayed", "None"]
SATURATION_TYPES = ["Hill", "Logistic", "Michaelis-Menten", "Tanh", "None"]
PRIOR_DISTRIBUTIONS = [
    "HalfNormal",
    "Normal",
    "LogNormal",
    "Gamma",
    "Beta",
    "TruncatedNormal",
    "HalfStudentT",
]
TREND_TYPES = [
    "None",
    "Linear",
    "Piecewise (Prophet-style)",
    "Spline",
    "Gaussian Process",
]
CONTROL_SELECTION_METHODS = ["None", "Horseshoe", "Spike-Slab", "LASSO"]


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
            "geographies": dataset.get_geographies(),
            "products": dataset.get_products(),
        }
    except Exception:
        return {"geographies": [], "products": []}


def render_prior_config(
    prefix: str,
    label: str,
    default_dist: str = "HalfNormal",
    default_params: dict = None,
    help_text: str = None,
) -> dict:
    """Render a configurable prior distribution UI and return the config dict."""
    default_params = default_params or {}

    col1, col2 = st.columns([1, 2])

    with col1:
        dist = st.selectbox(
            f"{label} Distribution",
            options=PRIOR_DISTRIBUTIONS,
            index=(
                PRIOR_DISTRIBUTIONS.index(default_dist)
                if default_dist in PRIOR_DISTRIBUTIONS
                else 0
            ),
            key=f"{prefix}_dist",
            help=help_text,
        )

    with col2:
        params = {}
        if dist == "HalfNormal":
            params["sigma"] = st.number_input(
                "Sigma",
                value=default_params.get("sigma", 1.0),
                min_value=0.01,
                step=0.1,
                key=f"{prefix}_sigma",
            )
        elif dist == "Normal":
            c1, c2 = st.columns(2)
            with c1:
                params["mu"] = st.number_input(
                    "Mu",
                    value=default_params.get("mu", 0.0),
                    step=0.1,
                    key=f"{prefix}_mu",
                )
            with c2:
                params["sigma"] = st.number_input(
                    "Sigma",
                    value=default_params.get("sigma", 1.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_sigma",
                )
        elif dist == "LogNormal":
            c1, c2 = st.columns(2)
            with c1:
                params["mu"] = st.number_input(
                    "Mu",
                    value=default_params.get("mu", 0.0),
                    step=0.1,
                    key=f"{prefix}_mu",
                )
            with c2:
                params["sigma"] = st.number_input(
                    "Sigma",
                    value=default_params.get("sigma", 1.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_sigma",
                )
        elif dist == "Gamma":
            c1, c2 = st.columns(2)
            with c1:
                params["alpha"] = st.number_input(
                    "Alpha",
                    value=default_params.get("alpha", 2.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_alpha",
                )
            with c2:
                params["beta"] = st.number_input(
                    "Beta",
                    value=default_params.get("beta", 1.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_beta",
                )
        elif dist == "Beta":
            c1, c2 = st.columns(2)
            with c1:
                params["alpha"] = st.number_input(
                    "Alpha",
                    value=default_params.get("alpha", 2.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_alpha",
                )
            with c2:
                params["beta"] = st.number_input(
                    "Beta",
                    value=default_params.get("beta", 2.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_beta",
                )
        elif dist == "TruncatedNormal":
            c1, c2 = st.columns(2)
            with c1:
                params["mu"] = st.number_input(
                    "Mu",
                    value=default_params.get("mu", 0.0),
                    step=0.1,
                    key=f"{prefix}_mu",
                )
                params["lower"] = st.number_input(
                    "Lower",
                    value=default_params.get("lower", 0.0),
                    step=0.1,
                    key=f"{prefix}_lower",
                )
            with c2:
                params["sigma"] = st.number_input(
                    "Sigma",
                    value=default_params.get("sigma", 1.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_sigma",
                )
                params["upper"] = st.number_input(
                    "Upper (optional)",
                    value=default_params.get("upper", 10.0),
                    step=0.1,
                    key=f"{prefix}_upper",
                )
        elif dist == "HalfStudentT":
            c1, c2 = st.columns(2)
            with c1:
                params["nu"] = st.number_input(
                    "Nu (df)",
                    value=default_params.get("nu", 3.0),
                    min_value=1.0,
                    step=0.5,
                    key=f"{prefix}_nu",
                )
            with c2:
                params["sigma"] = st.number_input(
                    "Sigma",
                    value=default_params.get("sigma", 1.0),
                    min_value=0.01,
                    step=0.1,
                    key=f"{prefix}_sigma",
                )

    return {"distribution": dist, "params": params}


def render_adstock_config(channel: str) -> dict:
    """Render adstock configuration UI for a channel."""

    adstock_type = st.selectbox(
        "Adstock Type",
        options=ADSTOCK_TYPES,
        index=0,
        key=f"adstock_type_{channel}",
        help="How media effects persist over time",
    )

    config = {
        "type": adstock_type.lower().replace("-", "_"),
    }

    if adstock_type != "None":
        config["l_max"] = st.slider(
            "Max Lag (weeks)",
            min_value=1,
            max_value=26,
            value=8,
            key=f"adstock_lmax_{channel}",
            help="Maximum lag for carryover effects",
        )

        config["normalize"] = st.checkbox(
            "Normalize weights",
            value=True,
            key=f"adstock_normalize_{channel}",
            help="Normalize adstock weights to sum to 1",
        )

        # Show relevant prior based on adstock type
        if adstock_type == "Geometric":
            with st.expander("Alpha Prior (Decay Rate)", expanded=False):
                st.caption(
                    "Controls how quickly the effect decays. Higher alpha = slower decay."
                )
                config["alpha_prior"] = render_prior_config(
                    f"adstock_alpha_{channel}",
                    "Alpha",
                    default_dist="Beta",
                    default_params={"alpha": 1.0, "beta": 3.0},
                    help_text="Prior for geometric decay rate (0-1)",
                )
        elif adstock_type == "Weibull":
            with st.expander("Weibull Parameters", expanded=False):
                st.caption("Weibull allows for delayed peak effects.")
                config["alpha_prior"] = render_prior_config(
                    f"adstock_alpha_{channel}",
                    "Shape (Alpha)",
                    default_dist="Gamma",
                    default_params={"alpha": 2.0, "beta": 1.0},
                    help_text="Shape parameter controls curve steepness",
                )
                config["theta_prior"] = render_prior_config(
                    f"adstock_theta_{channel}",
                    "Scale (Theta)",
                    default_dist="Gamma",
                    default_params={"alpha": 2.0, "beta": 1.0},
                    help_text="Scale parameter controls when peak occurs",
                )
        elif adstock_type == "Delayed":
            with st.expander("Delayed Adstock Parameters", expanded=False):
                config["alpha_prior"] = render_prior_config(
                    f"adstock_alpha_{channel}",
                    "Decay Rate",
                    default_dist="Beta",
                    default_params={"alpha": 2.0, "beta": 2.0},
                )
                config["theta_prior"] = render_prior_config(
                    f"adstock_theta_{channel}",
                    "Delay",
                    default_dist="Gamma",
                    default_params={"alpha": 2.0, "beta": 2.0},
                )

    return config


def render_saturation_config(channel: str) -> dict:
    """Render saturation configuration UI for a channel."""

    sat_type = st.selectbox(
        "Saturation Type",
        options=SATURATION_TYPES,
        index=0,
        key=f"sat_type_{channel}",
        help="How diminishing returns are modeled",
    )

    config = {
        "type": sat_type.lower().replace("-", "_"),
    }

    if sat_type != "None":
        with st.expander("Saturation Priors", expanded=False):
            st.caption("Configure the prior distributions for saturation parameters.")

            # Kappa bounds (data-driven)
            st.markdown("**Half-Saturation Point (Kappa) Bounds**")
            c1, c2 = st.columns(2)
            with c1:
                lower_pct = st.number_input(
                    "Lower Percentile",
                    value=0.1,
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                    key=f"sat_kappa_lower_{channel}",
                    help="Data percentile for lower kappa bound",
                )
            with c2:
                upper_pct = st.number_input(
                    "Upper Percentile",
                    value=0.9,
                    min_value=0.5,
                    max_value=1.0,
                    step=0.05,
                    key=f"sat_kappa_upper_{channel}",
                    help="Data percentile for upper kappa bound",
                )
            config["kappa_bounds_percentiles"] = (lower_pct, upper_pct)

            # Kappa prior
            st.markdown("**Kappa Prior (EC50)**")
            config["kappa_prior"] = render_prior_config(
                f"sat_kappa_{channel}",
                "Kappa",
                default_dist="Beta",
                default_params={"alpha": 2.0, "beta": 2.0},
                help_text="Half-saturation point (scaled by data bounds)",
            )

            # Slope prior
            st.markdown("**Slope Prior (Steepness)**")
            config["slope_prior"] = render_prior_config(
                f"sat_slope_{channel}",
                "Slope",
                default_dist="HalfNormal",
                default_params={"sigma": 1.5},
                help_text="Curve steepness - higher = sharper saturation",
            )

            # Beta prior (max effect)
            st.markdown("**Beta Prior (Max Effect)**")
            config["beta_prior"] = render_prior_config(
                f"sat_beta_{channel}",
                "Beta",
                default_dist="HalfNormal",
                default_params={"sigma": 1.5},
                help_text="Maximum effect scaling factor",
            )

    return config


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
        st.warning("Please upload a dataset first in the Data Management page.")
        return

    dataset_options = {d.filename: d.data_id for d in datasets}

    # -------------------------------------------------------------------------
    # Dataset Selection
    # -------------------------------------------------------------------------
    st.subheader("Dataset Selection")

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
            st.warning("No variables found in dataset.")
            return

        st.info(f"Dataset contains {len(variables)} variables")
    else:
        return

    st.markdown("---")

    # -------------------------------------------------------------------------
    # KPI Configuration
    # -------------------------------------------------------------------------
    st.subheader("KPI Configuration")

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

    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        kpi_dimensions = st.selectbox(
            "KPI Granularity",
            options=kpi_dim_options,
            help="At what level is the KPI measured?",
        )
        st.session_state.kpi_dimensions = kpi_dimensions

    with kpi_col2:
        kpi_log_transform = st.checkbox(
            "Use Multiplicative Model (Log Transform)",
            value=False,
            key="kpi_log_transform_checkbox",
            help="Transform to log scale for multiplicative effects",
        )
        st.session_state.kpi_log_transform = kpi_log_transform

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Media Channels
    # -------------------------------------------------------------------------
    st.subheader("Media Channels")

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
            with st.expander(f"**{channel}**", expanded=False):
                # Data level
                channel_level = st.selectbox(
                    "Data Level",
                    options=["National", "By Geography"],
                    key=f"level_{channel}",
                    help="At what level is the media data recorded?",
                )

                # Parent channel for hierarchical grouping
                other_channels = [c for c in media_channels if c != channel]
                parent_options = ["None"] + other_channels
                parent_channel = st.selectbox(
                    "Parent Channel (for hierarchical grouping)",
                    options=parent_options,
                    key=f"parent_{channel}",
                    help="Group this channel under a parent (e.g., Meta under Social)",
                )

                st.markdown("---")

                # Adstock configuration
                st.markdown("**Adstock Configuration**")
                adstock_config = render_adstock_config(channel)

                st.markdown("---")

                # Saturation configuration
                st.markdown("**Saturation Configuration**")
                saturation_config = render_saturation_config(channel)

                st.markdown("---")

                # Coefficient prior
                st.markdown("**Coefficient Prior**")
                st.caption(
                    "Prior for the media effect coefficient (constrained positive)."
                )
                coef_prior = render_prior_config(
                    f"coef_{channel}",
                    "Coefficient",
                    default_dist="HalfNormal",
                    default_params={"sigma": 2.0},
                    help_text="Prior for positive media effect",
                )

                media_configs.append(
                    {
                        "name": channel,
                        "level": channel_level,
                        "parent_channel": (
                            None if parent_channel == "None" else parent_channel
                        ),
                        "adstock": adstock_config,
                        "saturation": saturation_config,
                        "coefficient_prior": coef_prior,
                    }
                )

    st.session_state.media_channels = media_channels
    st.session_state.media_configs = media_configs

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Control Variables
    # -------------------------------------------------------------------------
    st.subheader("Control Variables")

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
            with st.expander(f"**{control}**", expanded=False):
                ctrl_col1, ctrl_col2 = st.columns(2)

                with ctrl_col1:
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

                with ctrl_col2:
                    use_shrinkage = st.checkbox(
                        "Use Shrinkage Prior",
                        value=False,
                        key=f"shrinkage_{control}",
                        help="Apply shrinkage for variable selection",
                    )

                # Coefficient prior
                st.markdown("**Coefficient Prior**")
                if allow_neg:
                    coef_prior = render_prior_config(
                        f"ctrl_coef_{control}",
                        "Coefficient",
                        default_dist="Normal",
                        default_params={"mu": 0.0, "sigma": 1.0},
                        help_text="Prior for effect (can be negative)",
                    )
                else:
                    coef_prior = render_prior_config(
                        f"ctrl_coef_{control}",
                        "Coefficient",
                        default_dist="HalfNormal",
                        default_params={"sigma": 1.0},
                        help_text="Prior for positive effect",
                    )

                control_configs.append(
                    {
                        "name": control,
                        "allow_negative": allow_neg,
                        "level": control_level,
                        "use_shrinkage": use_shrinkage,
                        "coefficient_prior": coef_prior,
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
        st.subheader("Dimension Alignment")

        st.info("National media will be allocated to geographic/product dimensions")

        align_col1, align_col2 = st.columns(2)

        with align_col1:
            geo_allocation = st.selectbox(
                "Geographic Allocation",
                options=["Equal", "By Population", "By Sales", "Custom"],
                help="How to distribute national media to geographic levels",
            )
            st.session_state.geo_allocation = (
                geo_allocation.lower().replace(" ", "_").replace("by_", "")
            )

        with align_col2:
            product_allocation = st.selectbox(
                "Product Allocation",
                options=["Equal", "By Sales", "Custom"],
                index=1,
                help="How to distribute to product levels",
            )
            st.session_state.product_allocation = (
                product_allocation.lower().replace(" ", "_").replace("by_", "")
            )

        st.markdown("---")

    # -------------------------------------------------------------------------
    # Inference Method
    # -------------------------------------------------------------------------
    st.subheader("Inference Settings")

    inf_col1, inf_col2 = st.columns(2)

    with inf_col1:
        inference_method = st.selectbox(
            "Inference Method",
            options=[
                "Bayesian (NumPyro/JAX)",
                "Bayesian (PyMC)",
                "Frequentist (Ridge)",
                "Frequentist (CVXPY)",
            ],
            index=0,
            help="Sampling/fitting method",
        )
        st.session_state.use_numpyro = inference_method == "Bayesian (NumPyro/JAX)"
        st.session_state.inference_method = inference_method

    with inf_col2:
        if "Bayesian" in inference_method:
            random_seed = st.number_input(
                "Random Seed",
                value=42,
                min_value=0,
                key="random_seed",
                help="For reproducibility",
            )
        else:
            random_seed = 42

    if "Bayesian" in inference_method:
        st.markdown("**MCMC Settings**")
        mcmc_col1, mcmc_col2, mcmc_col3, mcmc_col4 = st.columns(4)

        with mcmc_col1:
            n_chains = st.slider(
                "Chains",
                min_value=1,
                max_value=8,
                value=4,
                help="More chains = better convergence diagnostics",
            )
            st.session_state.n_chains = n_chains

        with mcmc_col2:
            n_draws = st.slider(
                "Draws per Chain",
                min_value=500,
                max_value=4000,
                value=1000,
                step=500,
                help="More draws = more precise estimates",
            )
            st.session_state.n_draws = n_draws

        with mcmc_col3:
            n_tune = st.slider(
                "Tuning Samples",
                min_value=500,
                max_value=2000,
                value=1000,
                step=500,
                help="Samples used for adaptation",
            )
            st.session_state.n_tune = n_tune

        with mcmc_col4:
            target_accept = st.slider(
                "Target Accept",
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
    st.subheader("Trend")

    trend_type = st.selectbox(
        "Trend Type",
        options=TREND_TYPES,
        index=1,
        help="How to model the underlying trend",
    )
    st.session_state.trend_type = trend_type

    # Trend-specific settings
    trend_settings = {"type": trend_type}

    if trend_type == "Linear":
        with st.expander("Linear Trend Settings", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                trend_settings["growth_prior_mu"] = st.slider(
                    "Growth Prior Mean",
                    min_value=-0.5,
                    max_value=0.5,
                    value=0.0,
                    step=0.05,
                    help="Expected growth rate direction",
                )
            with t_col2:
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
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                trend_settings["n_changepoints"] = st.slider(
                    "Number of Changepoints",
                    min_value=5,
                    max_value=30,
                    value=10,
                    help="More changepoints = more flexible trend",
                )
            with p_col2:
                trend_settings["changepoint_range"] = st.slider(
                    "Changepoint Range",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.8,
                    step=0.05,
                    help="Fraction of time series where changepoints can occur",
                )
            with p_col3:
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
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                trend_settings["n_knots"] = st.slider(
                    "Number of Knots",
                    min_value=5,
                    max_value=30,
                    value=10,
                    help="More knots = more flexible trend",
                )
            with s_col2:
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
            with s_col3:
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
            gp_col1, gp_col2 = st.columns(2)
            with gp_col1:
                trend_settings["gp_lengthscale_prior_mu"] = st.slider(
                    "Lengthscale Prior Mean",
                    min_value=0.1,
                    max_value=0.7,
                    value=0.3,
                    step=0.05,
                    help="Expected smoothness (fraction of time series)",
                )
                trend_settings["gp_lengthscale_prior_sigma"] = st.slider(
                    "Lengthscale Prior Sigma",
                    min_value=0.05,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Uncertainty in smoothness",
                )
            with gp_col2:
                trend_settings["gp_amplitude_prior_sigma"] = st.slider(
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
    st.subheader("Seasonality")

    seas_col1, seas_col2, seas_col3 = st.columns(3)

    with seas_col1:
        yearly_order = st.slider(
            "Yearly Seasonality Order",
            min_value=0,
            max_value=5,
            value=2,
            help="Fourier terms for yearly seasonality (0 = disabled)",
        )
        st.session_state.yearly_order = yearly_order

    with seas_col2:
        monthly_order = st.slider(
            "Monthly Seasonality Order",
            min_value=0,
            max_value=5,
            value=0,
            help="Fourier terms for monthly seasonality (0 = disabled)",
        )
        st.session_state.monthly_order = monthly_order

    with seas_col3:
        weekly_order = st.slider(
            "Weekly Seasonality Order",
            min_value=0,
            max_value=5,
            value=0,
            help="Fourier terms for weekly seasonality (0 = disabled)",
        )
        st.session_state.weekly_order = weekly_order

    if yearly_order > 0:
        with st.expander("Seasonality Details", expanded=False):
            total_terms = (yearly_order + monthly_order + weekly_order) * 2
            st.markdown(f"""
            **Total Fourier Terms:** {total_terms}

            This will create {total_terms} seasonality features
            (sin and cos terms for each order).

            - Order 1: Annual/Monthly/Weekly cycle
            - Order 2: Semi-annual/bi-weekly patterns
            - Order 3+: Finer seasonal variations
            """)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Hierarchical Structure
    # -------------------------------------------------------------------------
    st.subheader("Hierarchical Structure")

    if kpi_dimensions != "National":
        hier_col1, hier_col2 = st.columns(2)

        with hier_col1:
            pool_geo = st.checkbox(
                "Pool Across Geographies",
                value=True,
                help="Share information between geographic units",
            )
            st.session_state.pool_geo = pool_geo

        with hier_col2:
            pool_product = st.checkbox(
                "Pool Across Products",
                value=True,
                help="Share information between products",
            )
            st.session_state.pool_product = pool_product

        if pool_geo or pool_product:
            with st.expander("Hierarchical Settings", expanded=False):
                use_non_centered = st.checkbox(
                    "Use Non-Centered Parameterization",
                    value=True,
                    help="Better for sparse groups, recommended for most cases",
                )
                st.session_state.use_non_centered = use_non_centered

                if use_non_centered:
                    non_centered_threshold = st.slider(
                        "Non-Centered Threshold",
                        min_value=5,
                        max_value=50,
                        value=20,
                        help="Min observations per group to use centered parameterization",
                    )
                    st.session_state.non_centered_threshold = non_centered_threshold

                st.markdown("**Hyperpriors**")
                st.caption("Priors for the group-level parameters")

                st.markdown("*Group Mean Prior*")
                hier_mu_prior = render_prior_config(
                    "hier_mu",
                    "Group Mean",
                    default_dist="Normal",
                    default_params={"mu": 0.0, "sigma": 1.0},
                )
                st.session_state.hier_mu_prior = hier_mu_prior

                st.markdown("*Group SD Prior*")
                hier_sigma_prior = render_prior_config(
                    "hier_sigma",
                    "Group SD",
                    default_dist="HalfNormal",
                    default_params={"sigma": 0.5},
                )
                st.session_state.hier_sigma_prior = hier_sigma_prior
    else:
        st.info(
            "Hierarchical pooling is only available for multi-geography/product models"
        )
        st.session_state.pool_geo = False
        st.session_state.pool_product = False

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Control Selection
    # -------------------------------------------------------------------------
    if control_vars:
        st.subheader("Control Variable Selection")

        ctrl_sel_method = st.selectbox(
            "Selection Method",
            options=CONTROL_SELECTION_METHODS,
            index=0,
            help="Method for automatic control variable selection",
        )
        st.session_state.control_selection_method = ctrl_sel_method.lower().replace(
            "-", "_"
        )

        if ctrl_sel_method != "None":
            with st.expander("Selection Settings", expanded=False):
                if ctrl_sel_method == "Horseshoe":
                    expected_nonzero = st.slider(
                        "Expected Non-Zero Controls",
                        min_value=1,
                        max_value=len(control_vars),
                        value=min(3, len(control_vars)),
                        help="Expected number of relevant controls",
                    )
                    st.session_state.expected_nonzero = expected_nonzero

                elif ctrl_sel_method == "LASSO":
                    regularization = st.slider(
                        "Regularization Strength",
                        min_value=0.1,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        help="Higher = more sparsity",
                    )
                    st.session_state.regularization = regularization

        st.markdown("---")

    # -------------------------------------------------------------------------
    # Configuration Summary
    # -------------------------------------------------------------------------
    st.subheader("Configuration Summary")

    n_params = len(media_channels) * 3 + len(control_vars) + 5
    est_time = (
        st.session_state.n_chains
        * (st.session_state.n_draws + st.session_state.n_tune)
        * n_params
    ) / 5000

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

    config_description = st.text_area(
        "Description (optional)",
        value="",
        help="Add notes about this configuration",
    )

    if st.button("Save Configuration", type="primary", use_container_width=True):
        save_configuration(client, data_id, config_name, config_description)


def save_configuration(
    client, data_id: str, config_name: str, config_description: str = ""
):
    """Save the configuration to the backend."""
    try:
        # Convert KPI dimensions string to list format
        kpi_dims_map = {
            "National": ["Period"],
            "By Geography": ["Period", "Geography"],
            "By Product": ["Period", "Product"],
            "By Geography & Product": ["Period", "Geography", "Product"],
        }
        kpi_dimensions = kpi_dims_map.get(st.session_state.kpi_dimensions, ["Period"])

        # Convert trend type to API format
        trend_type_map = {
            "None": "none",
            "Linear": "linear",
            "Piecewise (Prophet-style)": "piecewise",
            "Spline": "spline",
            "Gaussian Process": "gaussian_process",
        }
        trend_type = trend_type_map.get(st.session_state.trend_type, "linear")

        # Convert inference method
        inference_method_map = {
            "Bayesian (NumPyro/JAX)": "bayesian_numpyro",
            "Bayesian (PyMC)": "bayesian_pymc",
            "Frequentist (Ridge)": "frequentist_ridge",
            "Frequentist (CVXPY)": "frequentist_cvxpy",
        }
        inference_method = inference_method_map.get(
            st.session_state.get("inference_method", "Bayesian (NumPyro/JAX)"),
            "bayesian_numpyro",
        )

        def level_to_dimensions(level: str) -> list[str]:
            """Convert level string to dimensions list."""
            if level == "By Geography":
                return ["Period", "Geography"]
            return ["Period"]

        def build_prior_schema(prior_config: dict) -> dict:
            """Convert prior config dict to API schema format."""
            return {
                "distribution": prior_config["distribution"],
                "params": prior_config["params"],
            }

        def build_adstock_schema(adstock_config: dict) -> dict:
            """Convert adstock config dict to API schema format."""
            schema = {
                "type": adstock_config["type"],
                "l_max": adstock_config.get("l_max", 8),
                "normalize": adstock_config.get("normalize", True),
            }
            if "alpha_prior" in adstock_config and adstock_config["alpha_prior"]:
                schema["alpha_prior"] = build_prior_schema(
                    adstock_config["alpha_prior"]
                )
            if "theta_prior" in adstock_config and adstock_config["theta_prior"]:
                schema["theta_prior"] = build_prior_schema(
                    adstock_config["theta_prior"]
                )
            return schema

        def build_saturation_schema(sat_config: dict) -> dict:
            """Convert saturation config dict to API schema format."""
            schema = {
                "type": sat_config["type"],
            }
            if "kappa_bounds_percentiles" in sat_config:
                schema["kappa_bounds_percentiles"] = list(
                    sat_config["kappa_bounds_percentiles"]
                )
            if "kappa_prior" in sat_config and sat_config["kappa_prior"]:
                schema["kappa_prior"] = build_prior_schema(sat_config["kappa_prior"])
            if "slope_prior" in sat_config and sat_config["slope_prior"]:
                schema["slope_prior"] = build_prior_schema(sat_config["slope_prior"])
            if "beta_prior" in sat_config and sat_config["beta_prior"]:
                schema["beta_prior"] = build_prior_schema(sat_config["beta_prior"])
            return schema

        # Build media channel schemas
        media_channel_schemas = []
        for ch in st.session_state.media_configs:
            channel_schema = {
                "name": ch["name"],
                "dimensions": level_to_dimensions(ch.get("level", "National")),
                "adstock": build_adstock_schema(
                    ch.get("adstock", {"type": "geometric", "l_max": 8})
                ),
                "saturation": build_saturation_schema(
                    ch.get("saturation", {"type": "hill"})
                ),
            }
            if ch.get("coefficient_prior"):
                channel_schema["coefficient_prior"] = build_prior_schema(
                    ch["coefficient_prior"]
                )
            if ch.get("parent_channel"):
                channel_schema["parent_channel"] = ch["parent_channel"]
            media_channel_schemas.append(channel_schema)

        # Build control variable schemas
        control_schemas = []
        for ctrl in st.session_state.control_configs:
            ctrl_schema = {
                "name": ctrl["name"],
                "dimensions": level_to_dimensions(ctrl.get("level", "National")),
                "allow_negative": ctrl.get("allow_negative", True),
                "use_shrinkage": ctrl.get("use_shrinkage", False),
            }
            if ctrl.get("coefficient_prior"):
                ctrl_schema["coefficient_prior"] = build_prior_schema(
                    ctrl["coefficient_prior"]
                )
            control_schemas.append(ctrl_schema)

        # Build mff_config structure
        mff_config = {
            "kpi": {
                "name": st.session_state.kpi_name,
                "dimensions": kpi_dimensions,
                "log_transform": st.session_state.get("kpi_log_transform", False),
            },
            "media_channels": media_channel_schemas,
            "controls": control_schemas,
            "alignment": {
                "geo_allocation": st.session_state.get("geo_allocation", "equal"),
                "product_allocation": st.session_state.get(
                    "product_allocation", "sales"
                ),
            },
        }

        # Build trend config
        trend_settings = st.session_state.get("trend_settings", {})
        trend_config = {
            "type": trend_type,
        }
        # Add trend-specific settings
        for key in [
            "n_changepoints",
            "changepoint_range",
            "changepoint_prior_scale",
            "n_knots",
            "spline_degree",
            "spline_prior_sigma",
            "gp_lengthscale_prior_mu",
            "gp_lengthscale_prior_sigma",
            "gp_amplitude_prior_sigma",
            "gp_n_basis",
            "growth_prior_mu",
            "growth_prior_sigma",
        ]:
            if key in trend_settings:
                trend_config[key] = trend_settings[key]

        # Build hierarchical config
        hierarchical_config = {
            "enabled": st.session_state.get("pool_geo", False)
            or st.session_state.get("pool_product", False),
            "pool_across_geo": st.session_state.get("pool_geo", False),
            "pool_across_product": st.session_state.get("pool_product", False),
            "use_non_centered": st.session_state.get("use_non_centered", True),
        }

        # Build seasonality config
        seasonality_config = {
            "yearly": (
                st.session_state.get("yearly_order", 2)
                if st.session_state.get("yearly_order", 2) > 0
                else None
            ),
            "monthly": (
                st.session_state.get("monthly_order", 0)
                if st.session_state.get("monthly_order", 0) > 0
                else None
            ),
            "weekly": (
                st.session_state.get("weekly_order", 0)
                if st.session_state.get("weekly_order", 0) > 0
                else None
            ),
        }

        # Build model_settings structure
        model_settings = {
            "inference_method": inference_method,
            "n_chains": st.session_state.n_chains,
            "n_draws": st.session_state.n_draws,
            "n_tune": st.session_state.n_tune,
            "target_accept": st.session_state.target_accept,
            "trend": trend_config,
            "seasonality": seasonality_config,
            "hierarchical": hierarchical_config,
            "random_seed": st.session_state.get("random_seed", 42),
        }

        # Build the configuration request
        config_data = {
            "name": config_name,
            "description": config_description or f"Configuration for dataset {data_id}",
            "mff_config": mff_config,
            "model_settings": model_settings,
        }

        # Submit to API
        result = client.create_config(config_data)

        st.success(f"Configuration '{config_name}' saved successfully!")

        # Clear cache to refresh config list
        clear_config_cache()

        # Store the config ID
        if isinstance(result, dict):
            st.session_state.selected_config_id = result.get("config_id")

    except APIError as e:
        display_api_error(e, "Failed to save configuration")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        import traceback

        st.code(traceback.format_exc())


# =============================================================================
# Configuration List
# =============================================================================


@st.fragment
def render_config_list():
    """Render the list of existing configurations."""
    st.subheader("Saved Configurations")

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
        with st.expander(f"**{config.name}**", expanded=False):
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

                # Show more details if available
                if config.mff_config:
                    media_names = [
                        ch.get("name", "?") for ch in config.get_media_channels()
                    ]
                    if media_names:
                        st.markdown(f"**Channels:** {', '.join(media_names)}")

            with col2:
                if st.button("Delete", key=f"del_{config.config_id}"):
                    try:
                        client.delete_config(config.config_id)
                        clear_config_cache()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

                if st.button("Load", key=f"load_{config.config_id}"):
                    st.session_state.selected_config_id = config.config_id
                    st.info(f"Configuration '{config.name}' selected")


# =============================================================================
# Main Page
# =============================================================================


def main():
    """Main page entry point."""
    page_header(
        "Model Configuration",
        "Configure your Marketing Mix Model with flexible options for adstock, saturation, priors, and more.",
    )

    # Single column layout with expandable form
    with st.expander(
        "Create New Configuration", expanded=not st.session_state.selected_config_id
    ):
        render_config_form()

    st.markdown("---")

    render_config_list()


if __name__ == "__main__":
    main()

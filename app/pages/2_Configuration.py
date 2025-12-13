"""
Configuration Page.

Create and manage model configurations with dataset-aware variable selection.
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


# =============================================================================
# Configuration Form
# =============================================================================

@st.fragment
def render_config_form():
    """Render the configuration creation form."""
    st.markdown("### Create New Configuration")
    
    # Get available datasets
    try:
        client = get_api_client()
        datasets = fetch_datasets(client)
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return
    
    if not datasets:
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Management"):
            st.switch_page("pages/1_📁_Data_Management.py")
        return
    
    # Dataset selection (outside form for dynamic updates)
    dataset_options = {d.filename: d.data_id for d in datasets}
    
    selected_dataset_name = st.selectbox(
        "📊 Select Dataset",
        options=list(dataset_options.keys()),
        index=0,
        help="Choose the dataset containing your marketing and KPI data",
    )
    
    selected_data_id = dataset_options[selected_dataset_name]
    
    # Fetch variables for selected dataset
    if selected_data_id != st.session_state.config_dataset_id:
        st.session_state.config_dataset_id = selected_data_id
        st.session_state.config_dataset_variables = get_dataset_variables(client, selected_data_id)
    
    variables = st.session_state.config_dataset_variables
    
    if not variables:
        st.warning("Could not load variables from dataset. Please check the data format.")
        return
    
    st.caption(f"Found {len(variables)} variables in dataset")
    
    # Main configuration form
    with st.form("config_form"):
        # Basic info
        st.markdown("#### Basic Information")
        
        col1, col2 = st.columns(2)
        with col1:
            config_name = st.text_input(
                "Configuration Name",
                placeholder="My MMM Config",
                help="A descriptive name for this configuration",
            )
        with col2:
            config_description = st.text_area(
                "Description (optional)",
                height=68,
                placeholder="Describe the purpose of this model configuration...",
            )
        
        st.markdown("---")
        
        # Variable Selection
        st.markdown("#### Variable Mapping")
        
        # KPI Selection
        col1, col2 = st.columns([2, 1])
        with col1:
            kpi_variable = st.selectbox(
                "🎯 KPI (Target Variable)",
                options=[""] + variables,
                index=0,
                help="Select the variable representing your key performance indicator (e.g., Sales, Revenue, Conversions)",
            )
        with col2:
            kpi_dimensions = st.selectbox(
                "KPI Dimensions",
                options=["National", "By Geography", "By Product", "By Geography and Product"],
                help="How is your KPI data structured?",
            )
        
        # Media Channels Selection
        st.markdown("##### 📺 Media Channels")
        
        # Filter out KPI from available variables for media
        media_available = [v for v in variables if v != kpi_variable]
        
        media_channels = st.multiselect(
            "Select Media/Marketing Variables",
            options=media_available,
            help="Select all variables representing marketing spend or impressions (e.g., TV_Spend, Digital_Spend, Radio_GRPs)",
        )
        
        # Media channel settings (if any selected)
        if media_channels:
            st.caption(f"Configure settings for {len(media_channels)} media channels:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                default_adstock = st.number_input(
                    "Default Adstock L-max",
                    min_value=1,
                    max_value=52,
                    value=8,
                    help="Maximum lag for adstock transformation (weeks)",
                )
            with col2:
                media_level = st.selectbox(
                    "Media Data Level",
                    options=["National", "By Geography"],
                    help="Are media variables at national or geo level?",
                )
            with col3:
                saturation_type = st.selectbox(
                    "Saturation Function",
                    options=["logistic", "hill"],
                    help="Function to model diminishing returns",
                )
        else:
            default_adstock = 8
            media_level = "National"
            saturation_type = "logistic"
        
        # Control Variables Selection
        st.markdown("##### 🎛️ Control Variables")
        
        # Filter out KPI and media from available variables for controls
        control_available = [v for v in variables if v != kpi_variable and v not in media_channels]
        
        control_variables = st.multiselect(
            "Select Control Variables (Optional)",
            options=control_available,
            help="Select variables to control for external factors (e.g., Price, Promotions, Seasonality_Index, Competitor_Activity)",
        )
        
        # Control settings
        if control_variables:
            allow_negative_controls = st.checkbox(
                "Allow negative coefficients for controls",
                value=True,
                help="If unchecked, control variables will be constrained to positive effects only",
            )
        else:
            allow_negative_controls = True
        
        st.markdown("---")
        
        # Model Settings
        st.markdown("#### Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trend_type = st.selectbox(
                "Trend Type",
                options=["none", "linear", "piecewise", "spline", "gaussian_process"],
                index=0,
                help="Type of trend component to include in the model",
            )
            yearly_seasonality = st.number_input(
                "Yearly Seasonality Order",
                min_value=0,
                max_value=10,
                value=2,
                help="Number of Fourier terms for yearly seasonality (0 = none)",
            )
        
        with col2:
            n_chains = st.number_input(
                "MCMC Chains",
                min_value=1,
                max_value=8,
                value=4,
                help="Number of parallel chains for sampling",
            )
            n_draws = st.number_input(
                "MCMC Draws",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Number of posterior samples per chain",
            )
            n_tune = st.number_input(
                "MCMC Tune Steps",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Number of tuning/warmup steps",
            )
        
        # Dimension Alignment (if applicable)
        if kpi_dimensions != "National":
            st.markdown("#### Dimension Alignment")
            geo_allocation = st.selectbox(
                "Geographic Allocation Method",
                options=["equal", "population", "sales"],
                help="How to allocate national media spend across geographies",
            )
        else:
            geo_allocation = "equal"
        
        st.markdown("---")
        
        # Summary before submission
        if kpi_variable and media_channels:
            st.markdown("#### Configuration Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("KPI", kpi_variable)
            with summary_col2:
                st.metric("Media Channels", len(media_channels))
            with summary_col3:
                st.metric("Control Variables", len(control_variables))
        
        # Submit
        submitted = st.form_submit_button(
            "Create Configuration",
            type="primary",
            use_container_width=True,
        )
        
        if submitted:
            # Validation
            if not config_name:
                st.error("Please provide a configuration name.")
                return
            if not kpi_variable:
                st.error("Please select a KPI variable.")
                return
            if not media_channels:
                st.error("Please select at least one media channel.")
                return
            
            # Build MFF config
            kpi_dim_mapping = {
                "National": ["Period"],
                "By Geography": ["Period", "Geography"],
                "By Product": ["Period", "Product"],
                "By Geography and Product": ["Period", "Geography", "Product"],
            }
            
            media_dim_mapping = {
                "National": ["Period"],
                "By Geography": ["Period", "Geography"],
            }
            
            mff_config = {
                "kpi": {
                    "name": kpi_variable,
                    "dimensions": kpi_dim_mapping[kpi_dimensions],
                },
                "media_channels": [
                    {
                        "name": channel,
                        "dimensions": media_dim_mapping[media_level],
                        "adstock": {
                            "type": "geometric",
                            "l_max": default_adstock,
                        },
                        "saturation": {
                            "type": saturation_type,
                        },
                    }
                    for channel in media_channels
                ],
                "control_variables": [
                    {
                        "name": ctrl,
                        "allow_negative": allow_negative_controls,
                    }
                    for ctrl in control_variables
                ],
                "dimension_alignment": {
                    "geographic_allocation": geo_allocation,
                },
            }
            
            # Model settings
            model_settings = {
                "trend_type": trend_type,
                "yearly_seasonality_order": yearly_seasonality,
                "n_chains": n_chains,
                "n_draws": n_draws,
                "n_tune": n_tune,
            }
            
            # Save to API
            with st.spinner("Creating configuration..."):
                try:
                    result = client.create_config(
                        name=config_name,
                        description=config_description or None,
                        mff_config=mff_config,
                        model_settings=model_settings,
                    )
                    
                    st.success(f"✅ Configuration '{config_name}' created successfully!")
                    st.session_state.selected_config_id = result.config_id
                    st.session_state.selected_data_id = selected_data_id
                    
                    clear_config_cache()
                    
                except APIError as e:
                    display_api_error(e)
                except Exception as e:
                    st.error(f"Failed to create configuration: {e}")


# =============================================================================
# Configuration List
# =============================================================================

@st.fragment
def render_config_list():
    """Render the list of existing configurations."""
    st.markdown("### Existing Configurations")
    
    try:
        client = get_api_client()
        configs = fetch_configs(client)
        
        if not configs:
            st.info("No configurations found. Create one above.")
            return
        
        for config in configs:
            with st.container():
                col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                
                with col1:
                    is_selected = st.session_state.selected_config_id == config.config_id
                    icon = "✅" if is_selected else "⚙️"
                    st.markdown(f"**{icon} {config.name}**")
                    if config.description:
                        st.caption(config.description[:50] + "..." if len(config.description) > 50 else config.description)
                
                with col2:
                    st.caption(f"ID: {config.config_id[:8]}...")
                
                with col3:
                    st.caption(format_datetime(config.created_at))
                
                with col4:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📋", key=f"select_cfg_{config.config_id}", help="Select"):
                            st.session_state.selected_config_id = config.config_id
                            st.rerun()
                    with col_b:
                        if st.button("🗑️", key=f"delete_cfg_{config.config_id}", help="Delete"):
                            st.session_state[f"confirm_delete_cfg_{config.config_id}"] = True
                            st.rerun()
                
                # Confirm delete
                if st.session_state.get(f"confirm_delete_cfg_{config.config_id}", False):
                    st.warning(f"⚠️ Delete '{config.name}'?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", key=f"confirm_yes_cfg_{config.config_id}", type="primary"):
                            try:
                                client.delete_config(config.config_id)
                                st.success("Deleted!")
                                if st.session_state.selected_config_id == config.config_id:
                                    st.session_state.selected_config_id = None
                                clear_config_cache()
                                del st.session_state[f"confirm_delete_cfg_{config.config_id}"]
                                st.rerun()
                            except APIError as e:
                                display_api_error(e)
                    with col_no:
                        if st.button("Cancel", key=f"confirm_no_cfg_{config.config_id}"):
                            del st.session_state[f"confirm_delete_cfg_{config.config_id}"]
                            st.rerun()
                
                st.markdown("---")
                
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading configurations: {e}")


# =============================================================================
# Configuration Details
# =============================================================================

@st.fragment
def render_config_details():
    """Render details for the selected configuration."""
    st.markdown("### Configuration Details")
    
    config_id = st.session_state.selected_config_id
    
    if not config_id:
        st.info("Select a configuration to view details.")
        return
    
    try:
        client = get_api_client()
        config = client.get_config(config_id)
        
        # Basic info
        st.markdown(f"**{config.name}**")
        if config.description:
            st.caption(config.description)
        
        st.caption(f"Created: {format_datetime(config.created_at)} | Updated: {format_datetime(config.updated_at)}")
        
        # Quick summary
        mff = config.mff_config
        kpi_name = mff.get("kpi", {}).get("name", "Unknown")
        n_media = len(mff.get("media_channels", []))
        n_controls = len(mff.get("control_variables", []))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("KPI", kpi_name)
        with col2:
            st.metric("Media", n_media)
        with col3:
            st.metric("Controls", n_controls)
        
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["🎯 MFF Config", "⚙️ Model Settings", "📄 Raw JSON"])
        
        with tab1:
            # KPI
            st.markdown("##### KPI")
            kpi = mff.get("kpi", {})
            st.markdown(f"**{kpi.get('name')}** — Dimensions: {', '.join(kpi.get('dimensions', []))}")
            
            # Media Channels
            st.markdown("##### Media Channels")
            media_names = [ch.get("name") for ch in mff.get("media_channels", [])]
            st.markdown(", ".join(media_names) if media_names else "None")
            
            with st.expander("Channel Details"):
                for channel in mff.get("media_channels", []):
                    st.markdown(f"**{channel.get('name')}**")
                    adstock = channel.get("adstock", {})
                    saturation = channel.get("saturation", {})
                    st.caption(f"Adstock: {adstock.get('type', 'geometric')} (L-max: {adstock.get('l_max', 8)}) | Saturation: {saturation.get('type', 'logistic')}")
            
            # Controls
            st.markdown("##### Control Variables")
            control_names = [ctrl.get("name") for ctrl in mff.get("control_variables", [])]
            st.markdown(", ".join(control_names) if control_names else "None")
        
        with tab2:
            settings = config.model_settings
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Trend Type:** {settings.get('trend_type', 'none')}")
                st.markdown(f"**Seasonality Order:** {settings.get('yearly_seasonality_order', 0)}")
            
            with col2:
                st.markdown(f"**Chains:** {settings.get('n_chains', 4)}")
                st.markdown(f"**Draws:** {settings.get('n_draws', 1000)}")
                st.markdown(f"**Tune:** {settings.get('n_tune', 1000)}")
        
        with tab3:
            st.json({
                "mff_config": mff,
                "model_settings": config.model_settings,
            })
        
        # Actions
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔬 Fit Model", use_container_width=True, type="primary"):
                st.switch_page("pages/3_Model_Fitting.py")
        
        with col2:
            if st.button("📋 Duplicate", use_container_width=True):
                st.info("Duplicate functionality coming soon")
        
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main page function."""
    page_header(
        "⚙️ Configuration",
        "Create and manage model configurations for your MMM analysis."
    )
    
    # Two-column layout
    col_form, col_details = st.columns([3, 2])
    
    with col_form:
        with st.expander("➕ Create New Configuration", expanded=not st.session_state.selected_config_id):
            render_config_form()
    
    with col_details:
        render_config_details()
    
    st.markdown("---")
    render_config_list()


if __name__ == "__main__":
    main()
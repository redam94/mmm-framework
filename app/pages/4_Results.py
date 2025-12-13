"""
Results Page.

View model results, diagnostics, and contributions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any

from api_client import (
    get_api_client,
    fetch_models,
    fetch_model_results,
    fetch_contributions,
    APIError,
    JobStatus,
)
from components import (
    apply_custom_css,
    page_header,
    format_datetime,
    format_number,
    format_percentage,
    display_api_error,
    init_session_state,
    status_icon,
    CHART_COLORS,
    COMPONENT_COLORS,
    plot_model_fit,
    plot_residuals,
    plot_channel_contributions,
    plot_contribution_waterfall,
    plot_contribution_pie,
    plot_contribution_timeseries,
    plot_response_curves,
    plot_marginal_roas,
    plot_posterior_distributions,
    plot_component_decomposition,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Results | MMM Framework",
    page_icon="📈",
    layout="wide",
)

apply_custom_css()

init_session_state(
    selected_model_id=None,
    results_cache={},
)


# =============================================================================
# Model Selector
# =============================================================================

@st.fragment
def render_model_selector():
    """Render the model selection dropdown."""
    try:
        client = get_api_client()
        models = fetch_models(client, status_filter="completed")
        
        if not models:
            st.warning("No completed models found. Fit a model first.")
            if st.button("Go to Model Fitting"):
                st.switch_page("pages/3_🔬_Model_Fitting.py")
            return
        
        # Build options
        model_options = {f"{m.name or m.model_id[:8]} ({format_datetime(m.created_at)})": m.model_id for m in models}
        
        # Find current selection
        current_idx = None
        if st.session_state.selected_model_id:
            for i, (name, mid) in enumerate(model_options.items()):
                if mid == st.session_state.selected_model_id:
                    current_idx = i
                    break
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            selected_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                index=current_idx,
                key="model_selector",
            )
        
        with col2:
            if st.button("🔄 Refresh", key="refresh_selector"):
                st.cache_data.clear()
                st.rerun()
        
        if selected_name:
            new_model_id = model_options[selected_name]
            if new_model_id != st.session_state.selected_model_id:
                st.session_state.selected_model_id = new_model_id
                st.rerun()
                
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading models: {e}")


# =============================================================================
# Diagnostics Tab
# =============================================================================

@st.fragment
def render_diagnostics_tab(results: dict[str, Any]):
    """Render model diagnostics."""
    st.markdown("### MCMC Diagnostics")
    
    diagnostics = results.get("diagnostics", {})
    
    if not diagnostics:
        st.warning("No diagnostics available.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        div = diagnostics.get("divergences", 0)
        delta = "Good" if div == 0 else ("Warning" if div < 10 else "Bad")
        st.metric("Divergences", div, delta=delta, delta_color="normal" if div == 0 else "inverse")
    
    with col2:
        rhat = diagnostics.get("rhat_max", None)
        if rhat:
            delta = "Good" if rhat < 1.01 else "Check"
            st.metric("R-hat Max", f"{rhat:.4f}", delta=delta, delta_color="normal" if rhat < 1.01 else "inverse")
        else:
            st.metric("R-hat Max", "N/A")
    
    with col3:
        ess = diagnostics.get("ess_bulk_min", None)
        if ess:
            delta = "Good" if ess > 400 else "Low"
            st.metric("ESS Bulk Min", f"{ess:.0f}", delta=delta, delta_color="normal" if ess > 400 else "inverse")
        else:
            st.metric("ESS Bulk Min", "N/A")
    
    with col4:
        ess_tail = diagnostics.get("ess_tail_min", None)
        if ess_tail:
            delta = "Good" if ess_tail > 400 else "Low"
            st.metric("ESS Tail Min", f"{ess_tail:.0f}", delta=delta, delta_color="normal" if ess_tail > 400 else "inverse")
        else:
            st.metric("ESS Tail Min", "N/A")
    
    # Interpretation
    st.markdown("---")
    
    div_count = diagnostics.get("divergences", 0)
    rhat_max = diagnostics.get("rhat_max", 1.0)
    ess_min = diagnostics.get("ess_bulk_min", 0)
    
    if div_count == 0 and rhat_max < 1.01 and ess_min > 400:
        st.success("✅ All diagnostics look good! The model has converged well.")
    elif div_count > 0 or rhat_max >= 1.1:
        st.error("⚠️ Some convergence issues detected. Consider re-running with more samples or adjusting priors.")
    else:
        st.warning("⚠️ Some diagnostics are borderline. Results may still be usable but should be interpreted with caution.")
    
    # Detailed diagnostics table
    with st.expander("📋 Detailed Diagnostics"):
        if diagnostics:
            diag_df = pd.DataFrame([
                {"Metric": k, "Value": v}
                for k, v in diagnostics.items()
            ])
            st.dataframe(diag_df, use_container_width=True, hide_index=True)


# =============================================================================
# Model Fit Tab
# =============================================================================

@st.fragment
def render_model_fit_tab(results: dict[str, Any]):
    """Render model fit visualization."""
    st.markdown("### Model Fit")
    
    # Check for required data
    fit_data = results.get("fit", {})
    
    if not fit_data:
        st.info("Model fit data not available. This may require computing posterior predictive samples.")
        return
    
    periods = fit_data.get("periods", [])
    observed = fit_data.get("observed", [])
    predicted_mean = fit_data.get("predicted_mean", [])
    predicted_std = fit_data.get("predicted_std", [])
    
    if not periods or not observed:
        st.warning("Insufficient data for model fit visualization.")
        return
    
    # Plot
    plot_model_fit(
        periods=periods,
        observed=observed,
        predicted_mean=predicted_mean,
        predicted_std=predicted_std if predicted_std else None,
        y_label=results.get("kpi_name", "Value"),
    )
    
    # Fit metrics
    col1, col2, col3 = st.columns(3)
    
    r2 = fit_data.get("r2", None)
    rmse = fit_data.get("rmse", None)
    mape = fit_data.get("mape", None)
    
    with col1:
        if r2 is not None:
            st.metric("R²", f"{r2:.4f}")
    with col2:
        if rmse is not None:
            st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        if mape is not None:
            st.metric("MAPE", f"{mape:.2%}")
    
    # Residual analysis
    if predicted_mean:
        residuals = [o - p for o, p in zip(observed, predicted_mean)]
        
        with st.expander("📊 Residual Analysis", expanded=False):
            plot_residuals(periods, residuals)


# =============================================================================
# Posterior Tab
# =============================================================================

@st.fragment
def render_posteriors_tab(results: dict[str, Any]):
    """Render posterior distributions."""
    st.markdown("### Posterior Distributions")
    
    posteriors = results.get("posteriors", {})
    
    if not posteriors:
        st.info("Posterior samples not available in API response.")
        return
    
    # Parameter selection
    all_params = list(posteriors.keys())
    
    # Group parameters
    beta_params = [p for p in all_params if "beta" in p.lower()]
    adstock_params = [p for p in all_params if "adstock" in p.lower() or "alpha" in p.lower()]
    saturation_params = [p for p in all_params if "sat" in p.lower() or "lam" in p.lower()]
    other_params = [p for p in all_params if p not in beta_params + adstock_params + saturation_params]
    
    tabs = st.tabs(["📊 All", "β Coefficients", "Adstock", "Saturation", "Other"])
    
    with tabs[0]:
        selected_params = st.multiselect(
            "Select Parameters",
            options=all_params,
            default=all_params[:8] if len(all_params) > 8 else all_params,
        )
        if selected_params:
            filtered = {k: posteriors[k] for k in selected_params}
            plot_posterior_distributions(filtered)
    
    with tabs[1]:
        if beta_params:
            plot_posterior_distributions({k: posteriors[k] for k in beta_params})
        else:
            st.info("No beta parameters found.")
    
    with tabs[2]:
        if adstock_params:
            plot_posterior_distributions({k: posteriors[k] for k in adstock_params})
        else:
            st.info("No adstock parameters found.")
    
    with tabs[3]:
        if saturation_params:
            plot_posterior_distributions({k: posteriors[k] for k in saturation_params})
        else:
            st.info("No saturation parameters found.")
    
    with tabs[4]:
        if other_params:
            plot_posterior_distributions({k: posteriors[k] for k in other_params})
        else:
            st.info("No other parameters found.")


# =============================================================================
# Response Curves Tab
# =============================================================================

@st.fragment
def render_response_curves_tab(results: dict[str, Any]):
    """Render response curves."""
    st.markdown("### Response Curves")
    
    response_curves = results.get("response_curves", {})
    
    if not response_curves:
        st.info("Response curve data not available.")
        return
    
    plot_response_curves(response_curves, title="Channel Response Curves")
    
    # ROAS analysis
    st.markdown("---")
    st.markdown("### Marginal ROAS")
    
    roas_data = results.get("marginal_roas", {})
    
    if roas_data:
        plot_marginal_roas(roas_data)
    else:
        st.info("ROAS data not available.")


# =============================================================================
# Contributions Tab
# =============================================================================

@st.fragment
def render_contributions_tab(model_id: str):
    """Render channel contributions."""
    st.markdown("### Channel Contributions")
    
    # Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        hdi_prob = st.slider("HDI Probability", min_value=0.5, max_value=0.99, value=0.94, step=0.01)
    
    with col2:
        compute_btn = st.button("🔄 Compute", type="primary")
    
    # Fetch/compute contributions
    try:
        client = get_api_client()
        
        if compute_btn:
            with st.spinner("Computing contributions..."):
                contributions = client.compute_contributions(model_id, hdi_prob=hdi_prob)
        else:
            contributions = fetch_contributions(client, model_id, hdi_prob=hdi_prob)
        
        if not contributions:
            st.info("No contributions available. Click Compute to generate.")
            return
        
        # Parse contributions
        total_contrib = contributions.get("total_contributions", {})
        contrib_pct = contributions.get("contribution_pct", {})
        
        if not total_contrib:
            st.warning("Contribution data not available.")
            return
        
        # Visualization tabs
        viz_tabs = st.tabs(["📊 Bar Chart", "📈 Waterfall", "🥧 Pie Chart", "📋 Table"])
        
        with viz_tabs[0]:
            plot_channel_contributions(total_contrib, show_percentage=True)
        
        with viz_tabs[1]:
            baseline = contributions.get("baseline", 0)
            plot_contribution_waterfall(total_contrib, baseline=baseline)
        
        with viz_tabs[2]:
            plot_contribution_pie(total_contrib)
        
        with viz_tabs[3]:
            contrib_df = pd.DataFrame([
                {
                    "Channel": k,
                    "Contribution": v,
                    "Percentage": contrib_pct.get(k, v / sum(total_contrib.values()) if sum(total_contrib.values()) > 0 else 0),
                }
                for k, v in total_contrib.items()
            ])
            contrib_df["Contribution"] = contrib_df["Contribution"].apply(lambda x: f"{x:,.0f}")
            contrib_df["Percentage"] = contrib_df["Percentage"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error computing contributions: {e}")


# =============================================================================
# Component Decomposition Tab
# =============================================================================

@st.fragment
def render_decomposition_tab(results: dict[str, Any]):
    """Render component decomposition."""
    st.markdown("### Component Decomposition")
    
    decomposition = results.get("decomposition", {})
    
    if not decomposition:
        st.info("Component decomposition data not available.")
        return
    
    periods = decomposition.get("periods", [])
    components = decomposition.get("components", {})
    observed = decomposition.get("observed", [])
    
    if not periods or not components:
        st.warning("Insufficient data for decomposition visualization.")
        return
    
    # Component selection
    available_components = list(components.keys())
    selected_components = st.multiselect(
        "Select Components",
        options=available_components,
        default=available_components,
    )
    
    if selected_components:
        filtered_components = {k: components[k] for k in selected_components}
        plot_component_decomposition(
            periods=periods,
            components=filtered_components,
            observed=observed if observed else None,
        )


# =============================================================================
# Summary Tab
# =============================================================================

@st.fragment
def render_summary_tab(results: dict[str, Any]):
    """Render model summary."""
    st.markdown("### Model Summary")
    
    summary = results.get("summary", {})
    
    if not summary:
        st.info("Model summary not available.")
        return
    
    # Display summary table
    if isinstance(summary, dict):
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df.round(4), use_container_width=True)
    else:
        st.json(summary)
    
    # Export options
    st.markdown("---")
    st.markdown("### Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if isinstance(summary, dict):
            csv = pd.DataFrame(summary).to_csv()
            st.download_button(
                "📥 Download Summary (CSV)",
                csv,
                "model_summary.csv",
                "text/csv",
                use_container_width=True,
            )
    
    with col2:
        st.download_button(
            "📥 Download Results (JSON)",
            str(results),
            "model_results.json",
            "application/json",
            use_container_width=True,
        )
    
    with col3:
        if st.button("📥 Download Model", use_container_width=True):
            st.info("Model download functionality would go here")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main page function."""
    page_header(
        "📈 Results",
        "View model diagnostics, contributions, and analysis results."
    )
    
    # Model selector
    render_model_selector()
    
    if not st.session_state.selected_model_id:
        st.info("Select a model to view results.")
        return
    
    model_id = st.session_state.selected_model_id
    
    # Fetch results
    try:
        client = get_api_client()
        results = fetch_model_results(client, model_id)
        
    except APIError as e:
        display_api_error(e)
        return
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return
    
    st.markdown("---")
    
    # Results tabs
    tabs = st.tabs([
        "📊 Diagnostics",
        "🎯 Model Fit",
        "📉 Posteriors",
        "📈 Response Curves",
        "💰 Contributions",
        "🧩 Decomposition",
        "📋 Summary",
    ])
    
    with tabs[0]:
        render_diagnostics_tab(results)
    
    with tabs[1]:
        render_model_fit_tab(results)
    
    with tabs[2]:
        render_posteriors_tab(results)
    
    with tabs[3]:
        render_response_curves_tab(results)
    
    with tabs[4]:
        render_contributions_tab(model_id)
    
    with tabs[5]:
        render_decomposition_tab(results)
    
    with tabs[6]:
        render_summary_tab(results)


if __name__ == "__main__":
    main()
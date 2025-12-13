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
    fetch_model_fit,
    fetch_posteriors,
    fetch_response_curves,
    fetch_decomposition,
    fetch_marginal_roas,
    fetch_contributions,
    fetch_model_summary,
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
            return
        
        model_options = {
            f"{m.name or m.model_id[:8]} ({format_datetime(m.completed_at)})": m.model_id
            for m in models
        }
        
        selected_label = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            key="model_selector",
        )
        
        if selected_label:
            st.session_state.selected_model_id = model_options[selected_label]
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading models: {e}")


# =============================================================================
# Diagnostics Tab
# =============================================================================

@st.fragment
def render_diagnostics_tab(model_id: str):
    """Render model diagnostics."""
    st.markdown("### Model Diagnostics")
    
    try:
        client = get_api_client()
        results = fetch_model_results(client, model_id)
        
        diagnostics = results.get("diagnostics", {})
        
        if not diagnostics:
            st.info("Diagnostics not available.")
            return
        
        # Diagnostic metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            divergences = diagnostics.get("divergences", 0)
            status = "✅" if divergences == 0 else "⚠️" if divergences < 10 else "❌"
            st.metric("Divergences", f"{status} {divergences}")
        
        with col2:
            rhat = diagnostics.get("rhat_max", 0)
            status = "✅" if rhat < 1.01 else "⚠️" if rhat < 1.05 else "❌"
            st.metric("Max R-hat", f"{status} {rhat:.4f}")
        
        with col3:
            ess = diagnostics.get("ess_bulk_min", 0)
            status = "✅" if ess > 400 else "⚠️" if ess > 100 else "❌"
            st.metric("Min ESS (bulk)", f"{status} {ess:.0f}")
        
        # Interpretation
        st.markdown("---")
        st.markdown("### Interpretation")
        
        issues = []
        if divergences > 0:
            issues.append(f"⚠️ {divergences} divergences detected. Consider increasing `target_accept` or reparameterizing the model.")
        if rhat > 1.01:
            issues.append(f"⚠️ R-hat ({rhat:.3f}) exceeds 1.01. Chains may not have converged. Consider running more iterations.")
        if ess < 400:
            issues.append(f"⚠️ Low ESS ({ess:.0f}). Effective sample size is low. Consider running more iterations.")
        
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("✅ All diagnostics look good! The model appears to have converged properly.")
        
        # Parameter summary table
        st.markdown("---")
        st.markdown("### Parameter Summary")
        
        param_summary = results.get("parameter_summary", [])
        
        if param_summary:
            df = pd.DataFrame(param_summary)
            
            # Format columns
            if "mean" in df.columns:
                df["mean"] = df["mean"].apply(lambda x: f"{x:.4f}")
            if "sd" in df.columns:
                df["sd"] = df["sd"].apply(lambda x: f"{x:.4f}")
            if "hdi_3%" in df.columns:
                df["hdi_3%"] = df["hdi_3%"].apply(lambda x: f"{x:.4f}")
            if "hdi_97%" in df.columns:
                df["hdi_97%"] = df["hdi_97%"].apply(lambda x: f"{x:.4f}")
            if "r_hat" in df.columns:
                df["r_hat"] = df["r_hat"].apply(lambda x: f"{x:.4f}" if x else "N/A")
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Parameter summary not available.")
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading diagnostics: {e}")


# =============================================================================
# Geo Comparison Helper
# =============================================================================

def render_geo_comparison(fit_data: dict, periods: list, geographies: list):
    """Render a comparison chart of R² across geographies."""
    import plotly.graph_objects as go
    
    by_geo = fit_data.get("by_geography", {})
    
    if not by_geo:
        st.info("No geography-level data available.")
        return
    
    # Create metrics table
    geo_metrics = []
    for geo in geographies:
        geo_data = by_geo.get(geo, {})
        geo_metrics.append({
            "Geography": geo,
            "R²": geo_data.get("r2"),
            "RMSE": geo_data.get("rmse"),
            "MAPE (%)": geo_data.get("mape"),
        })
    
    df_metrics = pd.DataFrame(geo_metrics)
    
    # Format for display
    if not df_metrics.empty:
        df_display = df_metrics.copy()
        if "R²" in df_display.columns:
            df_display["R²"] = df_display["R²"].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
        if "RMSE" in df_display.columns:
            df_display["RMSE"] = df_display["RMSE"].apply(lambda x: f"{x:,.2f}" if x is not None else "N/A")
        if "MAPE (%)" in df_display.columns:
            df_display["MAPE (%)"] = df_display["MAPE (%)"].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # R² bar chart
    if df_metrics["R²"].notna().any():
        fig = go.Figure()
        
        colors = ['#4285f4', '#ea4335', '#fbbc04', '#34a853', '#ff6d01', '#46bdc6', '#9334e6', '#e91e63']
        
        for i, geo in enumerate(geographies):
            r2_val = by_geo.get(geo, {}).get("r2", 0)
            fig.add_trace(go.Bar(
                x=[geo],
                y=[r2_val],
                name=geo,
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ))
        
        fig.update_layout(
            title="R² by Geography",
            xaxis_title="Geography",
            yaxis_title="R²",
            yaxis_range=[0, 1],
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Model Fit Tab
# =============================================================================

@st.fragment
def render_model_fit_tab(model_id: str):
    """Render model fit visualization with aggregation controls."""
    st.markdown("### Model Fit")
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading model fit data..."):
            fit_data = fetch_model_fit(client, model_id)
        
        if not fit_data:
            st.info("Model fit data not available.")
            return
        
        periods = fit_data.get("periods", [])
        has_geo = fit_data.get("has_geo", False)
        geographies = fit_data.get("geographies", [])
        
        if not periods:
            st.warning("Insufficient data for model fit visualization.")
            return
        
        # View selection
        view_options = ["Aggregated (Total)"]
        if has_geo and geographies:
            view_options.extend([f"📍 {geo}" for geo in geographies])
        
        col_view, col_spacer = st.columns([2, 3])
        with col_view:
            selected_view = st.selectbox(
                "View",
                options=view_options,
                index=0,
                key=f"fit_view_{model_id}",
                help="View aggregated total or individual geography",
            )
        
        # Get data based on selection
        if selected_view == "Aggregated (Total)":
            # Use aggregated data
            agg_data = fit_data.get("aggregated", {})
            observed = agg_data.get("observed", fit_data.get("observed", []))
            predicted_mean = agg_data.get("predicted_mean", fit_data.get("predicted_mean", []))
            predicted_std = agg_data.get("predicted_std", fit_data.get("predicted_std"))
            r2 = agg_data.get("r2", fit_data.get("r2"))
            rmse = agg_data.get("rmse", fit_data.get("rmse"))
            mape = agg_data.get("mape", fit_data.get("mape"))
            view_title = "Aggregated Model Fit (Sum Across Geographies)"
        else:
            # Extract geo name (remove emoji prefix)
            geo_name = selected_view.replace("📍 ", "")
            geo_data = fit_data.get("by_geography", {}).get(geo_name, {})
            
            if not geo_data:
                st.warning(f"No data available for geography: {geo_name}")
                return
            
            observed = geo_data.get("observed", [])
            predicted_mean = geo_data.get("predicted_mean", [])
            predicted_std = None  # Not typically available at geo level
            r2 = geo_data.get("r2")
            rmse = geo_data.get("rmse")
            mape = geo_data.get("mape")
            view_title = f"Model Fit: {geo_name}"
        
        if not observed:
            st.warning("No data available for selected view.")
            return
        
        # Display subtitle for context
        if has_geo:
            st.caption(view_title)
        
        # Fit metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if r2 is not None:
                st.metric("R²", f"{r2:.4f}")
        with col2:
            if rmse is not None:
                st.metric("RMSE", f"{rmse:,.2f}")
        with col3:
            if mape is not None:
                st.metric("MAPE", f"{mape:.2f}%")
        
        # Plot
        plot_model_fit(
            periods=periods,
            observed=observed,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std if predicted_std else None,
            y_label="Sales" if selected_view == "Aggregated (Total)" else f"Sales ({geo_name})" if has_geo else "Sales",
        )
        
        # Residual analysis
        if predicted_mean:
            residuals = [o - p for o, p in zip(observed, predicted_mean)]
            
            with st.expander("📊 Residual Analysis", expanded=False):
                plot_residuals(periods, residuals)
        
        # Show geo comparison chart if aggregated view
        if selected_view == "Aggregated (Total)" and has_geo and geographies:
            with st.expander("🗺️ Compare Geographies", expanded=False):
                render_geo_comparison(fit_data, periods, geographies)
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading model fit: {e}")


# =============================================================================
# Posterior Tab
# =============================================================================

@st.fragment
def render_posteriors_tab(model_id: str):
    """Render posterior distributions."""
    st.markdown("### Posterior Distributions")
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading posterior samples..."):
            posteriors = fetch_posteriors(client, model_id)
        
        if not posteriors or len(posteriors) <= 1:  # Only model_id key
            st.info("Posterior samples not available.")
            return
        
        # Remove model_id from posteriors dict
        posteriors_data = {k: v for k, v in posteriors.items() if k != "model_id"}
        
        # Get all parameter names
        all_params = list(posteriors_data.keys())
        
        if not all_params:
            st.info("No parameters found in posteriors.")
            return
        
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
                key="posteriors_all_select",
            )
            if selected_params:
                filtered = {k: posteriors_data[k] for k in selected_params}
                plot_posterior_distributions(filtered)
        
        with tabs[1]:
            if beta_params:
                plot_posterior_distributions({k: posteriors_data[k] for k in beta_params})
            else:
                st.info("No beta parameters found.")
        
        with tabs[2]:
            if adstock_params:
                plot_posterior_distributions({k: posteriors_data[k] for k in adstock_params})
            else:
                st.info("No adstock parameters found.")
        
        with tabs[3]:
            if saturation_params:
                plot_posterior_distributions({k: posteriors_data[k] for k in saturation_params})
            else:
                st.info("No saturation parameters found.")
        
        with tabs[4]:
            if other_params:
                plot_posterior_distributions({k: posteriors_data[k] for k in other_params})
            else:
                st.info("No other parameters found.")
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading posteriors: {e}")


# =============================================================================
# Response Curves Tab
# =============================================================================

@st.fragment
def render_response_curves_tab(model_id: str):
    """Render response curves."""
    st.markdown("### Response Curves")
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading response curves..."):
            curves_data = fetch_response_curves(client, model_id)
        
        if not curves_data or not curves_data.get("channels"):
            st.info("Response curve data not available.")
            return
        
        channels = curves_data.get("channels", {})
        
        # Transform to expected format for plot function
        curves_for_plot = {}
        for channel, data in channels.items():
            curves_for_plot[channel] = {
                "spend": data.get("spend", []),
                "response": data.get("response", []),
                "current_spend": data.get("current_spend", 0),
            }
        
        plot_response_curves(curves_for_plot, title="Channel Response Curves")
        
        # ROAS analysis
        st.markdown("---")
        st.markdown("### Marginal ROAS")
        
        with st.spinner("Loading ROAS data..."):
            roas_data = fetch_marginal_roas(client, model_id)
        
        if roas_data and roas_data.get("channels"):
            plot_marginal_roas(roas_data.get("channels", {}))
        else:
            st.info("ROAS data not available.")
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading response curves: {e}")


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
        hdi_prob = st.slider(
            "HDI Probability",
            min_value=0.5,
            max_value=0.99,
            value=0.94,
            step=0.01,
            key="contributions_hdi_slider",
        )
    
    with col2:
        compute_btn = st.button("🔄 Compute", type="primary", key="compute_contributions_btn")
    
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
def render_decomposition_tab(model_id: str):
    """Render component decomposition."""
    st.markdown("### Component Decomposition")
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading decomposition data..."):
            decomposition = fetch_decomposition(client, model_id)
        
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
            key="decomposition_component_select",
        )
        
        if selected_components:
            filtered_components = {k: components[k] for k in selected_components}
            plot_component_decomposition(
                periods=periods,
                components=filtered_components,
                observed=observed if observed else None,
            )
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading decomposition: {e}")


# =============================================================================
# Summary Tab
# =============================================================================

@st.fragment
def render_summary_tab(model_id: str):
    """Render model summary."""
    st.markdown("### Model Summary")
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading model summary..."):
            summary = fetch_model_summary(client, model_id)
        
        if not summary:
            st.info("Model summary not available.")
            return
        
        # Model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Information")
            st.write(f"**Name:** {summary.get('name', 'N/A')}")
            st.write(f"**Description:** {summary.get('description', 'N/A')}")
            st.write(f"**Created:** {summary.get('created_at', 'N/A')}")
            st.write(f"**Completed:** {summary.get('completed_at', 'N/A')}")
        
        with col2:
            st.markdown("#### Data Dimensions")
            st.write(f"**Observations:** {summary.get('n_obs', 'N/A')}")
            st.write(f"**Channels:** {summary.get('n_channels', 'N/A')}")
            st.write(f"**Controls:** {summary.get('n_controls', 'N/A')}")
        
        # Channel and control names
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Media Channels")
            channels = summary.get("channel_names", [])
            if channels:
                for ch in channels:
                    st.write(f"• {ch}")
            else:
                st.write("N/A")
        
        with col2:
            st.markdown("#### Control Variables")
            controls = summary.get("control_names", [])
            if controls:
                for ctrl in controls:
                    st.write(f"• {ctrl}")
            else:
                st.write("None")
        
        # Export options
        st.markdown("---")
        st.markdown("### Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export parameter summary
            param_summary = summary.get("parameter_summary", [])
            if param_summary:
                csv = pd.DataFrame(param_summary).to_csv(index=False)
                st.download_button(
                    "📥 Download Parameters (CSV)",
                    csv,
                    "parameter_summary.csv",
                    "text/csv",
                    use_container_width=True,
                )
        
        with col2:
            # Export summary as JSON
            import json
            st.download_button(
                "📥 Download Summary (JSON)",
                json.dumps(summary, indent=2, default=str),
                "model_summary.json",
                "application/json",
                use_container_width=True,
            )
        
        with col3:
            if st.button("📥 Download Model", use_container_width=True, key="download_model_btn"):
                st.info("Model download will be available in a future update.")
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading summary: {e}")


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
    
    st.markdown("---")
    
    # Results tabs - each tab now fetches its own data from the API
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
        render_diagnostics_tab(model_id)
    
    with tabs[1]:
        render_model_fit_tab(model_id)
    
    with tabs[2]:
        render_posteriors_tab(model_id)
    
    with tabs[3]:
        render_response_curves_tab(model_id)
    
    with tabs[4]:
        render_contributions_tab(model_id)
    
    with tabs[5]:
        render_decomposition_tab(model_id)
    
    with tabs[6]:
        render_summary_tab(model_id)


if __name__ == "__main__":
    main()
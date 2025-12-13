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
    fetch_prior_posterior,
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### Model Fit")
    
    st.markdown("""
    The model fit shows observed data against posterior predictions. The **shaded band** represents 
    the prediction uncertainty (±1 std), capturing both parameter uncertainty and residual variance.
    """)
    
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
            predicted_std = geo_data.get("predicted_std")  # Now available at geo level
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
        
        # Fit metrics with context
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if r2 is not None:
                # Color code R² based on quality
                if r2 >= 0.9:
                    st.metric("R²", f"{r2:.4f}", delta="Excellent", delta_color="normal")
                elif r2 >= 0.7:
                    st.metric("R²", f"{r2:.4f}", delta="Good", delta_color="normal")
                else:
                    st.metric("R²", f"{r2:.4f}", delta="Check", delta_color="inverse")
        with col2:
            if rmse is not None:
                st.metric("RMSE", f"{rmse:,.2f}")
        with col3:
            if mape is not None:
                if mape < 10:
                    st.metric("MAPE", f"{mape:.2f}%", delta="Excellent", delta_color="normal")
                elif mape < 20:
                    st.metric("MAPE", f"{mape:.2f}%", delta="Good", delta_color="normal")
                else:
                    st.metric("MAPE", f"{mape:.2f}%", delta="High", delta_color="inverse")
        with col4:
            # Coverage metric if std available
            if predicted_std:
                # Calculate how many observations fall within prediction interval
                within_band = sum(
                    1 for o, m, s in zip(observed, predicted_mean, predicted_std)
                    if m - 1.96 * s <= o <= m + 1.96 * s
                )
                coverage = within_band / len(observed) * 100
                st.metric("95% Coverage", f"{coverage:.0f}%")
        
        # Create model fit plot with uncertainty
        fig = go.Figure()
        
        # Add uncertainty band if std is available
        if predicted_std and len(predicted_std) == len(predicted_mean):
            upper = [m + 1.96 * s for m, s in zip(predicted_mean, predicted_std)]
            lower = [m - 1.96 * s for m, s in zip(predicted_mean, predicted_std)]
            
            fig.add_trace(go.Scatter(
                x=periods + periods[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% Prediction Interval',
                hoverinfo='skip',
            ))
        
        # Predicted mean
        fig.add_trace(go.Scatter(
            x=periods,
            y=predicted_mean,
            mode='lines',
            name='Predicted (Mean)',
            line=dict(color='rgb(99, 110, 250)', width=2),
            hovertemplate="Predicted: %{y:,.0f}<extra></extra>",
        ))
        
        # Observed data
        fig.add_trace(go.Scatter(
            x=periods,
            y=observed,
            mode='markers',
            name='Observed',
            marker=dict(color='rgb(239, 85, 59)', size=6),
            hovertemplate="Observed: %{y:,.0f}<extra></extra>",
        ))
        
        fig.update_layout(
            title="Posterior Predictive Fit",
            xaxis_title="Period",
            yaxis_title="Sales" if selected_view == "Aggregated (Total)" else f"Sales ({geo_name})" if has_geo else "Sales",
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode='x unified',
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"fit_plot_{selected_view}")
        
        # Residual analysis
        if predicted_mean:
            residuals = [o - p for o, p in zip(observed, predicted_mean)]
            
            with st.expander("📊 Residual Analysis", expanded=False):
                # Create subplot with residuals and histogram
                fig_resid = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Residuals Over Time", "Residual Distribution"],
                    column_widths=[0.65, 0.35],
                )
                
                # Residuals over time
                fig_resid.add_trace(go.Scatter(
                    x=periods,
                    y=residuals,
                    mode='markers+lines',
                    marker=dict(size=4, color='rgb(99, 110, 250)'),
                    line=dict(width=1, color='rgb(99, 110, 250)'),
                    name='Residuals',
                    showlegend=False,
                ), row=1, col=1)
                
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                
                # Residual histogram
                fig_resid.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=25,
                    marker_color='rgb(99, 110, 250)',
                    name='Distribution',
                    showlegend=False,
                ), row=1, col=2)
                
                fig_resid.update_layout(height=300)
                st.plotly_chart(fig_resid, use_container_width=True, key=f"resid_{selected_view}")
                
                # Residual statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Residual", f"{np.mean(residuals):,.1f}")
                with col2:
                    st.metric("Std Residual", f"{np.std(residuals):,.1f}")
                with col3:
                    st.metric("Min", f"{min(residuals):,.1f}")
                with col4:
                    st.metric("Max", f"{max(residuals):,.1f}")
        
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
# Prior vs Posterior Tab
# =============================================================================

def rgb_to_rgba(rgb: str, alpha: float = 1.0) -> str:
    """Convert RGB color to RGBA format."""
    # Handle different color formats
    if rgb.startswith("rgba"):
        return rgb
    if rgb.startswith("rgb("):
        r, g, b = rgb.strip("rgb(").strip(")").split(",")
        return f"rgba({r},{g},{b},{alpha})"
    if rgb.startswith("#"):
        # Hex to RGBA
        hex_color = rgb.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return rgb


@st.fragment
def render_prior_posterior_tab(model_id: str):
    """Render prior vs posterior comparison with shrinkage metrics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### Prior vs Posterior Analysis")
    
    st.markdown("""
    Compare how the data updated our prior beliefs. **Shrinkage** measures how much the posterior 
    standard deviation has reduced compared to the prior—higher shrinkage indicates the data is 
    more informative about that parameter.
    """)
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading prior vs posterior data..."):
            pp_data = fetch_prior_posterior(client, model_id)
        
        if not pp_data or "parameters" not in pp_data:
            st.info("Prior vs posterior data not available.")
            return
        
        parameters = pp_data.get("parameters", {})
        channel_names = pp_data.get("channel_names", [])
        
        if not parameters:
            st.info("No parameters available for comparison.")
            return
        
        # Organize parameters by category
        param_categories = {
            "Media Coefficients (β)": [],
            "Adstock Parameters": [],
            "Saturation Parameters (λ)": [],
            "Trend & Seasonality": [],
            "Other": [],
        }
        
        for param in parameters.keys():
            if "beta" in param.lower():
                param_categories["Media Coefficients (β)"].append(param)
            elif "adstock" in param.lower() or "alpha" in param.lower():
                param_categories["Adstock Parameters"].append(param)
            elif "sat" in param.lower() or "lam" in param.lower():
                param_categories["Saturation Parameters (λ)"].append(param)
            elif any(x in param.lower() for x in ["trend", "season", "gp_", "spline"]):
                param_categories["Trend & Seasonality"].append(param)
            else:
                param_categories["Other"].append(param)
        
        # Remove empty categories
        param_categories = {k: v for k, v in param_categories.items() if v}
        
        # Category selector
        selected_category = st.selectbox(
            "Parameter Category",
            options=list(param_categories.keys()),
            key="pp_category_select",
        )
        
        params_to_plot = param_categories[selected_category]
        
        # Filter to params that have both prior and posterior
        params_with_prior = [p for p in params_to_plot 
                           if parameters.get(p, {}).get("prior_samples") is not None]
        
        if not params_with_prior:
            st.warning(f"No parameters in '{selected_category}' have prior samples available.")
            # Show posteriors only
            params_with_prior = params_to_plot
        
        # Create comparison plots
        n_params = len(params_with_prior)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Create subplot titles
        subplot_titles = [p.replace('_', ' ').title() for p in params_with_prior]
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )
        
        for idx, param in enumerate(params_with_prior):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            param_data = parameters.get(param, {})
            
            # Get posterior samples
            post_samples = param_data.get("posterior_samples", [])
            
            # Get prior samples (if available)
            prior_samples = param_data.get("prior_samples", [])
            
            # Add prior histogram (if available)
            if prior_samples:
                fig.add_trace(
                    go.Histogram(
                        x=prior_samples,
                        name='Prior',
                        opacity=0.5,
                        marker_color='rgba(99, 110, 250, 0.6)',
                        nbinsx=40,
                        histnorm='probability density',
                        showlegend=(idx == 0),
                        legendgroup='prior',
                    ),
                    row=row, col=col
                )
            
            # Add posterior histogram
            if post_samples:
                fig.add_trace(
                    go.Histogram(
                        x=post_samples,
                        name='Posterior',
                        opacity=0.6,
                        marker_color='rgba(239, 85, 59, 0.7)',
                        nbinsx=40,
                        histnorm='probability density',
                        showlegend=(idx == 0),
                        legendgroup='posterior',
                    ),
                    row=row, col=col
                )
                
                # Add vertical line for posterior mean
                post_mean = param_data.get("posterior_mean", np.mean(post_samples))
                fig.add_vline(
                    x=post_mean,
                    line_dash="dash",
                    line_color="rgba(239, 85, 59, 0.8)",
                    line_width=2,
                    row=row, col=col
                )
        
        fig.update_layout(
            height=280 * n_rows,
            barmode='overlay',
            title=dict(
                text="Prior (Blue) vs Posterior (Red) Distributions",
                font=dict(size=16),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(t=80, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"pp_plot_{selected_category}")
        
        # Summary statistics table with shrinkage
        st.markdown("---")
        st.markdown("### Prior vs Posterior Summary Statistics")
        
        summary_data = []
        for param in params_with_prior:
            param_data = parameters.get(param, {})
            
            row = {
                "Parameter": param.replace('_', ' ').title(),
                "Posterior Mean": param_data.get("posterior_mean"),
                "Posterior Std": param_data.get("posterior_std"),
                "94% HDI Low": param_data.get("posterior_hdi_3"),
                "94% HDI High": param_data.get("posterior_hdi_97"),
            }
            
            if param_data.get("prior_mean") is not None:
                row["Prior Mean"] = param_data.get("prior_mean")
                row["Prior Std"] = param_data.get("prior_std")
                row["Shrinkage (%)"] = param_data.get("shrinkage_pct")
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format the dataframe for display
        format_cols = ["Posterior Mean", "Posterior Std", "Prior Mean", "Prior Std", 
                       "94% HDI Low", "94% HDI High"]
        for col in format_cols:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        if "Shrinkage (%)" in summary_df.columns:
            summary_df["Shrinkage (%)"] = summary_df["Shrinkage (%)"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
            )
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Shrinkage interpretation
        st.markdown("---")
        st.markdown("### 📊 Shrinkage Analysis")
        
        # Create shrinkage bar chart for parameters with prior data
        shrinkage_data = []
        for param in params_with_prior:
            param_data = parameters.get(param, {})
            shrinkage = param_data.get("shrinkage_pct")
            if shrinkage is not None:
                shrinkage_data.append({
                    "Parameter": param.replace('_', ' ').title(),
                    "Shrinkage (%)": shrinkage,
                })
        
        if shrinkage_data:
            shrinkage_df = pd.DataFrame(shrinkage_data)
            shrinkage_df = shrinkage_df.sort_values("Shrinkage (%)", ascending=True)
            
            # Color based on shrinkage level
            colors = []
            for s in shrinkage_df["Shrinkage (%)"]:
                if s >= 70:
                    colors.append("#2ecc71")  # Green - high information
                elif s >= 40:
                    colors.append("#f39c12")  # Orange - moderate information
                else:
                    colors.append("#e74c3c")  # Red - low information
            
            fig_shrinkage = go.Figure()
            fig_shrinkage.add_trace(go.Bar(
                x=shrinkage_df["Shrinkage (%)"],
                y=shrinkage_df["Parameter"],
                orientation='h',
                marker_color=colors,
                text=shrinkage_df["Shrinkage (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
            ))
            
            fig_shrinkage.update_layout(
                title="Posterior Shrinkage by Parameter",
                xaxis_title="Shrinkage (%)",
                yaxis_title="",
                height=max(300, len(shrinkage_data) * 35),
                xaxis=dict(range=[0, 105]),
                margin=dict(l=150, r=50),
            )
            
            # Add reference lines
            fig_shrinkage.add_vline(x=40, line_dash="dash", line_color="gray", opacity=0.5)
            fig_shrinkage.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_shrinkage, use_container_width=True, key=f"shrinkage_{selected_category}")
            
            # Interpretation guide
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **High (>70%)**")
                st.caption("Data strongly informs this parameter")
            with col2:
                st.markdown("🟠 **Moderate (40-70%)**")
                st.caption("Data provides useful information")
            with col3:
                st.markdown("🔴 **Low (<40%)**")
                st.caption("Prior dominates; consider more data")
        else:
            st.info("Shrinkage data not available for these parameters.")
        
        # Key insights
        st.markdown("---")
        st.info("""
        💡 **Interpretation Guide:**
        - **Shrinkage** = (1 - Posterior Std / Prior Std) × 100%
        - Higher shrinkage means the data is more informative about that parameter
        - If prior and posterior are similar (low shrinkage), the data provides little information
        - Large shifts from prior to posterior mean indicate strong evidence from the data
        - Parameters with low shrinkage may benefit from more informative priors or additional data
        """)
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading prior vs posterior data: {e}")
        import traceback
        st.code(traceback.format_exc())


# =============================================================================
# Response Curves Tab
# =============================================================================

@st.fragment
def render_response_curves_tab(model_id: str):
    """Render response curves with uncertainty bands."""
    import plotly.graph_objects as go
    
    st.markdown("### Response Curves")
    
    st.markdown("""
    Response curves show how media spend translates to sales effect after accounting for 
    saturation (diminishing returns). The **shaded bands** represent the 94% HDI (Highest Density Interval),
    capturing the uncertainty in the relationship.
    """)
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading response curves..."):
            curves_data = fetch_response_curves(client, model_id)
        
        if not curves_data or not curves_data.get("channels"):
            st.info("Response curve data not available.")
            return
        
        channels = curves_data.get("channels", {})
        
        # Settings
        col1, col2 = st.columns([1, 3])
        with col1:
            show_observed = st.checkbox("Show Observed Spend", value=True, key="rc_show_observed",
                                       help="Mark the range of observed spend values")
        
        # Create response curves plot with HDI bands
        fig = go.Figure()
        
        colors = [
            '#4285f4', '#ea4335', '#fbbc04', '#34a853', 
            '#ff6d01', '#46bdc6', '#9334e6', '#e91e63'
        ]
        
        for i, (channel, data) in enumerate(channels.items()):
            color = colors[i % len(colors)]
            color_rgba = rgb_to_rgba(color, 0.2)
            
            spend = data.get("spend", [])
            response = data.get("response", [])
            hdi_low = data.get("response_hdi_low", [])
            hdi_high = data.get("response_hdi_high", [])
            current_spend = data.get("current_spend", 0)
            spend_max = data.get("spend_max", max(spend) if spend else 0)
            
            # Add HDI band if available
            if hdi_low and hdi_high and len(hdi_low) == len(spend):
                fig.add_trace(go.Scatter(
                    x=spend + spend[::-1],
                    y=hdi_high + hdi_low[::-1],
                    fill='toself',
                    fillcolor=color_rgba,
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f"{channel} 94% HDI",
                    showlegend=False,
                    hoverinfo='skip',
                ))
            
            # Mean response curve
            fig.add_trace(go.Scatter(
                x=spend,
                y=response,
                mode='lines',
                name=channel,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{channel}</b><br>Spend: $%{{x:,.0f}}<br>Response: %{{y:,.0f}}<extra></extra>",
            ))
            
            # Mark current/observed spend
            if show_observed and spend_max > 0:
                fig.add_vline(
                    x=spend_max,
                    line_dash="dot",
                    line_color=color,
                    opacity=0.6,
                    annotation_text=f"{channel} max",
                    annotation_position="top",
                    annotation_font_size=10,
                )
        
        fig.update_layout(
            title="Response Curves by Channel (with 94% HDI)",
            xaxis_title="Media Spend ($)",
            yaxis_title="Contribution to Sales",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode='x unified',
        )
        
        st.plotly_chart(fig, use_container_width=True, key="response_curves_main")
        
        # Saturation analysis
        st.markdown("---")
        st.markdown("### 📊 Diminishing Returns Analysis")
        
        saturation_data = []
        for channel, data in channels.items():
            spend = data.get("spend", [])
            response = data.get("response", [])
            current_spend = data.get("current_spend", 0)
            spend_max = data.get("spend_max", max(spend) if spend else 0)
            
            if not spend or not response:
                continue
            
            # Find saturation level at current spend
            max_response = response[-1] if response else 0
            
            # Find response at current spend
            current_idx = min(range(len(spend)), key=lambda i: abs(spend[i] - current_spend))
            response_at_current = response[current_idx] if current_idx < len(response) else 0
            
            # Find response at max observed spend  
            max_idx = min(range(len(spend)), key=lambda i: abs(spend[i] - spend_max))
            response_at_max = response[max_idx] if max_idx < len(response) else 0
            
            saturation_at_current = (response_at_current / max_response * 100) if max_response > 0 else 0
            saturation_at_max = (response_at_max / max_response * 100) if max_response > 0 else 0
            
            saturation_data.append({
                "Channel": channel,
                "Current Spend": f"${current_spend:,.0f}",
                "Max Spend": f"${spend_max:,.0f}",
                "Saturation @ Current": f"{saturation_at_current:.0f}%",
                "Saturation @ Max": f"{saturation_at_max:.0f}%",
                "Room to Grow": f"{100 - saturation_at_max:.0f}%",
            })
        
        if saturation_data:
            st.dataframe(pd.DataFrame(saturation_data), use_container_width=True, hide_index=True)
            
            st.info("""
            💡 **Interpretation:**
            - **Saturation @ Current**: How much of the channel's potential is being utilized at average spend
            - **Saturation @ Max**: Saturation level at maximum observed spend
            - **Room to Grow**: Remaining potential before full saturation
            - Channels with low saturation and high ROAS are good candidates for budget increases
            """)
        
        # ROAS analysis
        st.markdown("---")
        st.markdown("### 💰 Marginal ROAS with Uncertainty")
        
        with st.spinner("Loading ROAS data..."):
            roas_data = fetch_marginal_roas(client, model_id)
        
        if roas_data and roas_data.get("channels"):
            roas_channels = roas_data.get("channels", {})
            
            # Create ROAS bar chart with error bars
            fig_roas = go.Figure()
            
            channel_names = list(roas_channels.keys())
            means = []
            errors_minus = []
            errors_plus = []
            bar_colors = []
            
            for i, (ch, data) in enumerate(roas_channels.items()):
                mean = data.get("mean", 0)
                hdi_low = data.get("hdi_low", mean)
                hdi_high = data.get("hdi_high", mean)
                
                means.append(mean)
                errors_minus.append(mean - hdi_low)
                errors_plus.append(hdi_high - mean)
                
                # Color based on whether ROAS > 1 (break-even)
                if hdi_low > 1:
                    bar_colors.append("#2ecc71")  # Green - confident above break-even
                elif hdi_high < 1:
                    bar_colors.append("#e74c3c")  # Red - confident below break-even
                else:
                    bar_colors.append("#f39c12")  # Orange - uncertain
            
            fig_roas.add_trace(go.Bar(
                x=channel_names,
                y=means,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=errors_plus,
                    arrayminus=errors_minus,
                    color='rgba(0,0,0,0.5)',
                    thickness=2,
                    width=6,
                ),
                marker_color=bar_colors,
                text=[f"{m:.2f}" for m in means],
                textposition='outside',
            ))
            
            fig_roas.add_hline(
                y=1.0, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Break-even (ROAS = 1.0)",
                annotation_position="right",
            )
            
            fig_roas.update_layout(
                title="Marginal ROAS by Channel (with 94% HDI)",
                xaxis_title="Channel",
                yaxis_title="ROAS",
                height=400,
                showlegend=False,
            )
            
            st.plotly_chart(fig_roas, use_container_width=True, key="roas_bars")
            
            # ROAS table
            roas_table = []
            for ch, data in roas_channels.items():
                roas_table.append({
                    "Channel": ch,
                    "Mean ROAS": f"{data.get('mean', 0):.3f}",
                    "Std": f"{data.get('std', 0):.3f}",
                    "94% HDI Low": f"{data.get('hdi_low', 0):.3f}",
                    "94% HDI High": f"{data.get('hdi_high', 0):.3f}",
                    "Total Spend": f"${data.get('total_spend', 0):,.0f}",
                })
            
            st.dataframe(pd.DataFrame(roas_table), use_container_width=True, hide_index=True)
            
            # Color legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Confident Above Break-even**")
                st.caption("94% HDI entirely above 1.0")
            with col2:
                st.markdown("🟠 **Uncertain**")
                st.caption("94% HDI spans break-even")
            with col3:
                st.markdown("🔴 **Confident Below Break-even**")
                st.caption("94% HDI entirely below 1.0")
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
        "🔄 Prior vs Posterior",
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
        render_prior_posterior_tab(model_id)
    
    with tabs[3]:
        render_posteriors_tab(model_id)
    
    with tabs[4]:
        render_response_curves_tab(model_id)
    
    with tabs[5]:
        render_contributions_tab(model_id)
    
    with tabs[6]:
        render_decomposition_tab(model_id)
    
    with tabs[7]:
        render_summary_tab(model_id)


if __name__ == "__main__":
    main()
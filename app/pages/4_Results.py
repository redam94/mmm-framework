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
# Trend & Seasonality Analysis Tab
# =============================================================================

@st.fragment
def render_trend_seasonality_tab(model_id: str):
    """Render trend and seasonality analysis with detailed visualizations."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### Trend & Seasonality Analysis")
    
    st.markdown("""
    Analyze the **time-varying components** of your model: underlying **trend** (long-term growth/decline) 
    and **seasonality** (cyclical patterns). These components capture systematic time effects independent 
    of marketing activities.
    """)
    
    try:
        client = get_api_client()
        
        with st.spinner("Loading trend and seasonality data..."):
            decomposition = fetch_decomposition(client, model_id)
        
        if not decomposition:
            st.info("Decomposition data not available.")
            return
        
        periods = decomposition.get("periods", [])
        components = decomposition.get("components", {})
        observed = decomposition.get("observed", [])
        metadata = decomposition.get("metadata", {})
        by_geography = decomposition.get("by_geography", {})
        
        n_periods = len(periods)
        
        # Convert periods to datetime if possible
        try:
            period_dates = pd.to_datetime(periods)
            periods_display = period_dates
        except:
            period_dates = None
            periods_display = list(range(n_periods))
        
        # Get trend and seasonality data
        trend_data = components.get("trend", [0] * n_periods)
        seasonality_data = components.get("seasonality", [0] * n_periods)
        baseline_data = components.get("baseline", [0] * n_periods)
        
        has_trend = metadata.get("has_trend", False) or np.any(np.array(trend_data) != 0)
        has_seasonality = metadata.get("has_seasonality", False) or np.any(np.array(seasonality_data) != 0)
        trend_type = metadata.get("trend_type", "unknown")
        
        # Info cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Type", trend_type.replace("_", " ").title())
        with col2:
            if has_trend:
                trend_change = trend_data[-1] - trend_data[0] if len(trend_data) > 1 else 0
                st.metric("Total Trend Change", f"{trend_change:,.0f}", 
                         delta=f"{(trend_change/abs(trend_data[0])*100) if trend_data[0] != 0 else 0:.1f}%")
            else:
                st.metric("Trend", "Not included")
        with col3:
            if has_seasonality:
                seas_range = max(seasonality_data) - min(seasonality_data)
                st.metric("Seasonality Range", f"±{seas_range/2:,.0f}")
            else:
                st.metric("Seasonality", "Not included")
        
        st.markdown("---")
        
        # Sub-tabs for different views
        ts_tabs = st.tabs([
            "📈 Trend Analysis",
            "🔄 Seasonality Analysis", 
            "🌍 Geographic View",
            "📊 Combined View"
        ])
        
        # =================================================================
        # Tab 1: Trend Analysis
        # =================================================================
        with ts_tabs[0]:
            st.markdown("#### Trend Component")
            
            if not has_trend:
                st.info("No trend component was included in this model (trend_type = 'none').")
            else:
                # Trend over time
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Scatter(
                    x=periods_display,
                    y=trend_data,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#e74c3c', width=3),
                    hovertemplate="<b>Trend</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
                ))
                
                # Add baseline for reference
                if baseline_data and np.any(np.array(baseline_data) != 0):
                    mean_baseline = np.mean(baseline_data)
                    fig_trend.add_hline(y=0, line_dash="dash", line_color="gray", 
                                       annotation_text="Zero line")
                
                fig_trend.update_layout(
                    title=f"Trend Component Over Time ({trend_type.replace('_', ' ').title()})",
                    xaxis_title="Period",
                    yaxis_title="Trend Effect (Sales Units)",
                    height=400,
                    hovermode='x unified',
                )
                
                st.plotly_chart(fig_trend, use_container_width=True, key="trend_main")
                
                # Trend statistics
                st.markdown("##### Trend Statistics")
                
                trend_arr = np.array(trend_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Start Value", f"{trend_arr[0]:,.0f}")
                with col2:
                    st.metric("End Value", f"{trend_arr[-1]:,.0f}")
                with col3:
                    avg_change = (trend_arr[-1] - trend_arr[0]) / max(n_periods - 1, 1)
                    st.metric("Avg Period Change", f"{avg_change:+,.1f}")
                with col4:
                    total_contrib = np.sum(trend_arr)
                    st.metric("Total Contribution", f"{total_contrib:,.0f}")
                
                # Interpretation
                with st.expander("📖 Trend Interpretation", expanded=False):
                    if trend_type == "linear":
                        st.markdown("""
                        **Linear Trend**: A simple straight-line trend capturing overall growth or decline.
                        - Positive slope = steady growth over time
                        - Negative slope = steady decline over time
                        - The slope coefficient indicates the average change per period
                        """)
                    elif trend_type == "piecewise":
                        st.markdown("""
                        **Piecewise Linear Trend** (Prophet-style): Allows the trend to change at 
                        multiple "changepoints" throughout the time series.
                        - Changepoints indicate where the growth rate shifted
                        - Useful for capturing market changes, product launches, etc.
                        """)
                    elif trend_type == "spline":
                        st.markdown("""
                        **B-Spline Trend**: A smooth, flexible trend using spline basis functions.
                        - Can capture complex non-linear patterns
                        - Smoother than piecewise trends
                        - Number of knots controls flexibility
                        """)
                    elif trend_type == "gaussian_process":
                        st.markdown("""
                        **Gaussian Process Trend**: A fully flexible, probabilistic trend component.
                        - Captures any smooth underlying pattern
                        - Includes uncertainty in the trend estimate
                        - Lengthscale controls how quickly the trend can change
                        """)
        
        # =================================================================
        # Tab 2: Seasonality Analysis
        # =================================================================
        with ts_tabs[1]:
            st.markdown("#### Seasonality Component")
            
            if not has_seasonality:
                st.info("No seasonality component was included in this model (yearly_seasonality = 0).")
            else:
                # Seasonality over time
                fig_seas = go.Figure()
                
                fig_seas.add_trace(go.Scatter(
                    x=periods_display,
                    y=seasonality_data,
                    mode='lines',
                    name='Seasonality',
                    line=dict(color='#2ecc71', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.2)',
                    hovertemplate="<b>Seasonality</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
                ))
                
                fig_seas.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_seas.update_layout(
                    title="Seasonality Component Over Time",
                    xaxis_title="Period",
                    yaxis_title="Seasonal Effect (Sales Units)",
                    height=400,
                    hovermode='x unified',
                )
                
                st.plotly_chart(fig_seas, use_container_width=True, key="seasonality_main")
                
                # Seasonal pattern (one cycle)
                if n_periods >= 52:
                    st.markdown("##### Annual Seasonal Pattern")
                    
                    # Aggregate to get average seasonal pattern
                    seasonal_arr = np.array(seasonality_data)
                    n_complete_years = n_periods // 52
                    
                    if n_complete_years >= 1:
                        # Average across years
                        seasonal_pattern = np.zeros(52)
                        for i in range(n_complete_years):
                            seasonal_pattern += seasonal_arr[i*52:(i+1)*52]
                        seasonal_pattern /= n_complete_years
                        
                        # Create week labels
                        week_labels = [f"Week {i+1}" for i in range(52)]
                        
                        fig_pattern = go.Figure()
                        
                        fig_pattern.add_trace(go.Bar(
                            x=week_labels,
                            y=seasonal_pattern,
                            marker_color=['#2ecc71' if v >= 0 else '#e74c3c' for v in seasonal_pattern],
                            hovertemplate="<b>%{x}</b><br>Effect: %{y:,.0f}<extra></extra>",
                        ))
                        
                        fig_pattern.update_layout(
                            title="Average Weekly Seasonal Pattern (52 Weeks)",
                            xaxis_title="Week of Year",
                            yaxis_title="Seasonal Effect",
                            height=350,
                            xaxis=dict(tickangle=45, dtick=4),
                        )
                        
                        st.plotly_chart(fig_pattern, use_container_width=True, key="seasonal_pattern")
                        
                        # Peak and trough weeks
                        peak_week = np.argmax(seasonal_pattern) + 1
                        trough_week = np.argmin(seasonal_pattern) + 1
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"📈 **Peak Season**: Week {peak_week} (+{seasonal_pattern[peak_week-1]:,.0f})")
                        with col2:
                            st.error(f"📉 **Low Season**: Week {trough_week} ({seasonal_pattern[trough_week-1]:,.0f})")
                
                # Interpretation
                with st.expander("📖 Seasonality Interpretation", expanded=False):
                    st.markdown("""
                    **Fourier Seasonality**: Cyclical patterns captured using sine and cosine basis functions.
                    
                    - **Positive values**: Time periods where sales are above the baseline trend
                    - **Negative values**: Time periods where sales are below the baseline trend
                    - **Amplitude**: The strength of seasonal effects (larger = more seasonality)
                    - **Order**: Higher orders capture more complex seasonal patterns
                    
                    Common seasonal patterns:
                    - Holiday spikes (Q4 for retail)
                    - Summer/winter patterns
                    - Back-to-school periods
                    - Industry-specific cycles
                    """)
        
        # =================================================================
        # Tab 3: Geographic View
        # =================================================================
        with ts_tabs[2]:
            st.markdown("#### Geographic Breakdown")
            
            geo_names = metadata.get("geo_names", ["National"])
            
            if len(geo_names) <= 1:
                st.info("This model has only one geography. Geographic breakdown is not applicable.")
            else:
                # Select component to view
                available_geo_components = list(by_geography.keys())
                
                if not available_geo_components:
                    st.warning("Geographic breakdown not available for this model.")
                else:
                    component_select = st.selectbox(
                        "Select Component",
                        options=["trend", "seasonality", "baseline"],
                        format_func=lambda x: x.title(),
                        key="geo_component_select"
                    )
                    
                    if component_select in by_geography:
                        geo_data = by_geography[component_select]
                        
                        # Line chart by geography
                        fig_geo = go.Figure()
                        
                        colors = [
                            '#4285f4', '#ea4335', '#fbbc04', '#34a853', 
                            '#ff6d01', '#46bdc6', '#9334e6', '#e91e63'
                        ]
                        
                        for i, (geo, values) in enumerate(geo_data.items()):
                            fig_geo.add_trace(go.Scatter(
                                x=periods_display,
                                y=values,
                                mode='lines',
                                name=geo,
                                line=dict(color=colors[i % len(colors)], width=2),
                            ))
                        
                        fig_geo.update_layout(
                            title=f"{component_select.title()} by Geography",
                            xaxis_title="Period",
                            yaxis_title=f"{component_select.title()} Effect",
                            height=450,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            hovermode='x unified',
                        )
                        
                        st.plotly_chart(fig_geo, use_container_width=True, key="geo_trend_seas")
                        
                        # Summary table
                        st.markdown("##### Geographic Summary")
                        
                        geo_summary = []
                        for geo, values in geo_data.items():
                            arr = np.array(values)
                            geo_summary.append({
                                "Geography": geo,
                                "Total": f"{np.sum(arr):,.0f}",
                                "Mean": f"{np.mean(arr):,.0f}",
                                "Std Dev": f"{np.std(arr):,.0f}",
                                "Min": f"{np.min(arr):,.0f}",
                                "Max": f"{np.max(arr):,.0f}",
                            })
                        
                        st.dataframe(pd.DataFrame(geo_summary), use_container_width=True, hide_index=True)
                    else:
                        st.warning(f"{component_select.title()} not available by geography.")
        
        # =================================================================
        # Tab 4: Combined View
        # =================================================================
        with ts_tabs[3]:
            st.markdown("#### Trend + Seasonality Combined")
            
            # Combined effect
            combined = np.array(trend_data) + np.array(seasonality_data)
            
            fig_combined = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Trend", "Seasonality", "Combined (Trend + Seasonality)"),
                row_heights=[0.3, 0.3, 0.4]
            )
            
            # Trend
            fig_combined.add_trace(
                go.Scatter(x=periods_display, y=trend_data, mode='lines',
                          name='Trend', line=dict(color='#e74c3c', width=2)),
                row=1, col=1
            )
            
            # Seasonality
            fig_combined.add_trace(
                go.Scatter(x=periods_display, y=seasonality_data, mode='lines',
                          name='Seasonality', line=dict(color='#2ecc71', width=2),
                          fill='tozeroy', fillcolor='rgba(46, 204, 113, 0.2)'),
                row=2, col=1
            )
            
            # Combined
            fig_combined.add_trace(
                go.Scatter(x=periods_display, y=combined, mode='lines',
                          name='Combined', line=dict(color='#3498db', width=2)),
                row=3, col=1
            )
            
            # Add baseline reference
            fig_combined.add_trace(
                go.Scatter(x=periods_display, y=baseline_data, mode='lines',
                          name='Baseline', line=dict(color='gray', width=1, dash='dot')),
                row=3, col=1
            )
            
            fig_combined.update_layout(
                height=650,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified',
            )
            
            # Add zero lines
            for i in range(1, 4):
                fig_combined.add_hline(y=0, line_dash="dash", line_color="lightgray", row=i, col=1)
            
            st.plotly_chart(fig_combined, use_container_width=True, key="combined_view")
            
            # Contribution summary
            st.markdown("##### Component Contribution Summary")
            
            total_observed = np.sum(observed) if observed else 0
            total_baseline = np.sum(baseline_data)
            total_trend = np.sum(trend_data)
            total_seasonality = np.sum(seasonality_data)
            
            summary_data = [
                {"Component": "Baseline", "Total": f"{total_baseline:,.0f}", 
                 "% of Observed": f"{total_baseline/total_observed*100:.1f}%" if total_observed else "N/A"},
                {"Component": "Trend", "Total": f"{total_trend:,.0f}", 
                 "% of Observed": f"{total_trend/total_observed*100:.1f}%" if total_observed else "N/A"},
                {"Component": "Seasonality", "Total": f"{total_seasonality:,.0f}", 
                 "% of Observed": f"{total_seasonality/total_observed*100:.1f}%" if total_observed else "N/A"},
            ]
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading trend/seasonality data: {e}")
        import traceback
        st.code(traceback.format_exc())


# =============================================================================
# Component Decomposition Tab
# =============================================================================

@st.fragment
def render_decomposition_tab(model_id: str):
    """Render comprehensive component decomposition with stacked area and YoY waterfall."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("### Component Decomposition")
    
    st.markdown("""
    Break down the model's predictions into individual components: **baseline** (intercept), 
    **trend**, **seasonality**, **media effects**, and **control variables**. This shows how 
    each factor contributes to the overall outcome.
    """)
    
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
        metadata = decomposition.get("metadata", {})
        by_geography = decomposition.get("by_geography", {})
        by_product = decomposition.get("by_product", {})
        observed_by_geo = decomposition.get("observed_by_geography", {})
        observed_by_prod = decomposition.get("observed_by_product", {})
        
        if not periods or not components:
            st.warning("Insufficient data for decomposition visualization.")
            return
        
        n_periods = len(periods)
        
        # Get available geo and product names
        geo_names = metadata.get("geo_names", ["National"])
        product_names = metadata.get("product_names", ["All"])
        has_geo = metadata.get("has_geo", False) or len(geo_names) > 1
        has_product = metadata.get("has_product", False) or len(product_names) > 1
        
        # =================================================================
        # FILTER CONTROLS
        # =================================================================
        st.markdown("---")
        
        filter_cols = st.columns([2, 2, 2])
        
        with filter_cols[0]:
            view_level = st.radio(
                "View Level",
                options=["Aggregate", "By Geography", "By Product"] if (has_geo or has_product) else ["Aggregate"],
                horizontal=True,
                key="decomp_view_level"
            )
        
        selected_geo = None
        selected_product = None
        
        with filter_cols[1]:
            if view_level == "By Geography" and has_geo:
                selected_geo = st.selectbox(
                    "Select Geography",
                    options=geo_names,
                    key="decomp_geo_filter"
                )
            elif has_geo:
                st.caption(f"📍 Aggregated over {len(geo_names)} geographies")
        
        with filter_cols[2]:
            if view_level == "By Product" and has_product:
                selected_product = st.selectbox(
                    "Select Product",
                    options=product_names,
                    key="decomp_product_filter"
                )
            elif has_product:
                st.caption(f"📦 Aggregated over {len(product_names)} products")
        
        st.markdown("---")
        
        # =================================================================
        # GET COMPONENTS BASED ON FILTER SELECTION
        # =================================================================
        def get_component_data(comp_name):
            """Get component data based on current filter selection."""
            if view_level == "By Geography" and selected_geo and comp_name in by_geography:
                geo_data = by_geography[comp_name]
                if selected_geo in geo_data:
                    return np.array(geo_data[selected_geo])
            elif view_level == "By Product" and selected_product and comp_name in by_product:
                prod_data = by_product[comp_name]
                if selected_product in prod_data:
                    return np.array(prod_data[selected_product])
            
            # Default to aggregate
            if comp_name in components:
                return np.array(components[comp_name])
            return np.zeros(n_periods)
        
        def get_observed_data():
            """Get observed data based on current filter selection."""
            if view_level == "By Geography" and selected_geo and selected_geo in observed_by_geo:
                return np.array(observed_by_geo[selected_geo])
            elif view_level == "By Product" and selected_product and selected_product in observed_by_prod:
                return np.array(observed_by_prod[selected_product])
            return np.array(observed)
        
        # Get filtered data
        baseline_comp = get_component_data("baseline").tolist()
        trend_comp = get_component_data("trend").tolist()
        seasonality_comp = get_component_data("seasonality").tolist()
        observed = get_observed_data().tolist()
        
        # Debug expander to show available components
        with st.expander("🔍 Debug: Available Components", expanded=False):
            st.write(f"**View Level:** {view_level}")
            if selected_geo:
                st.write(f"**Selected Geo:** {selected_geo}")
            if selected_product:
                st.write(f"**Selected Product:** {selected_product}")
            st.write(f"**Periods:** {n_periods}")
            st.write(f"**Aggregate Components:** {list(components.keys())}")
            st.write(f"**By Geography Components:** {list(by_geography.keys())}")
            st.write(f"**By Product Components:** {list(by_product.keys())}")
            for k, v in components.items():
                if isinstance(v, list):
                    st.write(f"  - {k}: {len(v)} values, range [{min(v):.2f}, {max(v):.2f}]")
        
        # Convert periods to datetime if possible
        try:
            period_dates = pd.to_datetime(periods)
            periods_display = period_dates
        except:
            period_dates = None
            periods_display = periods
        
        # Aggregate media components (individual channels prefixed with media_)
        media_components = {}
        for k in components.keys():
            if k.startswith("media_") and k != "media_total":
                media_components[k] = get_component_data(k).tolist()
        
        # If we have a pre-computed media_total, use that; otherwise sum individual channels
        if "media_total" in components and not media_components:
            media_total = get_component_data("media_total")
        else:
            media_total = np.zeros(n_periods)
            for comp_values in media_components.values():
                if len(comp_values) == n_periods:
                    media_total += np.array(comp_values)
        
        # Aggregate control components
        control_components = {}
        for k in components.keys():
            if k.startswith("control_") and k != "controls_total":
                control_components[k] = get_component_data(k).tolist()
        
        # If we have a pre-computed controls_total, use that; otherwise sum individual controls
        if "controls_total" in components and not control_components:
            controls_total = get_component_data("controls_total")
        else:
            controls_total = np.zeros(n_periods)
            for comp_values in control_components.values():
                if len(comp_values) == n_periods:
                    controls_total += np.array(comp_values)
        
        # Get geo and product effects if available (these are already filtered by get_component_data)
        geo_effects = get_component_data("geo_effects").tolist()
        product_effects = get_component_data("product_effects").tolist()
        
        # Calculate predicted total from components
        predicted = (np.array(baseline_comp) + np.array(trend_comp) + 
                    np.array(seasonality_comp) + media_total + controls_total +
                    np.array(geo_effects) + np.array(product_effects))
        
        # Check which components actually have non-zero values
        has_trend_data = np.any(np.array(trend_comp) != 0)
        has_seasonality_data = np.any(np.array(seasonality_comp) != 0)
        has_media_data = np.any(media_total != 0)
        has_controls_data = np.any(controls_total != 0)
        has_geo_effects = np.any(np.array(geo_effects) != 0)
        has_product_effects = np.any(np.array(product_effects) != 0)
        
        # Warn if missing expected components
        if not has_media_data and not media_components:
            st.warning("⚠️ No media channel contributions found in the decomposition. This may indicate the model's media contribution variables aren't being captured correctly.")
        
        # Sub-tabs for different views
        decomp_tabs = st.tabs([
            "📊 Summary", 
            "📈 Stacked Area Chart", 
            "📉 Component Time Series",
            "🌍 Geographic View",
            "📅 Year-over-Year Waterfall",
            "🥧 Relative Breakdown"
        ])
        
        # =================================================================
        # Tab 1: Summary
        # =================================================================
        with decomp_tabs[0]:
            st.markdown("#### Component Contribution Summary")
            
            # Calculate total contributions
            summary_data = []
            total_predicted = np.sum(predicted)
            
            components_summary = {
                "Baseline": np.sum(baseline_comp),
                "Trend": np.sum(trend_comp),
                "Seasonality": np.sum(seasonality_comp),
                "Media (Total)": np.sum(media_total),
                "Controls (Total)": np.sum(controls_total),
            }
            
            for name, value in components_summary.items():
                pct = (value / total_predicted * 100) if total_predicted != 0 else 0
                summary_data.append({
                    "Component": name,
                    "Total Contribution": value,
                    "% of Predicted": pct,
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=summary_df["Component"],
                    values=summary_df["Total Contribution"].abs(),
                    hole=0.4,
                    marker_colors=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'],
                    textinfo='label+percent',
                    textposition='outside',
                )])
                fig_pie.update_layout(
                    title="Share of Predicted Outcome",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="decomp_pie")
            
            with col2:
                # Summary table
                display_df = summary_df.copy()
                display_df["Total Contribution"] = display_df["Total Contribution"].apply(lambda x: f"{x:,.0f}")
                display_df["% of Predicted"] = display_df["% of Predicted"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Media channel breakdown
                if media_components:
                    st.markdown("##### Media Channel Breakdown")
                    media_summary = []
                    for name, values in media_components.items():
                        channel_name = name.replace("media_", "")
                        total = np.sum(values)
                        pct_of_media = (total / np.sum(media_total) * 100) if np.sum(media_total) != 0 else 0
                        media_summary.append({
                            "Channel": channel_name,
                            "Contribution": f"{total:,.0f}",
                            "% of Media": f"{pct_of_media:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(media_summary), use_container_width=True, hide_index=True)
        
        # =================================================================
        # Tab 2: Stacked Area Chart
        # =================================================================
        with decomp_tabs[1]:
            st.markdown("#### Stacked Component Contributions Over Time")
            
            # Color scheme for components
            colors = {
                'baseline': '#3498db',
                'trend': '#e74c3c', 
                'seasonality': '#2ecc71',
                'media': '#9b59b6',
                'controls': '#f39c12',
                'geo': '#1abc9c',
                'product': '#e67e22',
            }
            
            fig_stack = go.Figure()
            
            # Add components in stacking order (only those with non-zero values)
            stack_order = [
                ('Baseline', baseline_comp, colors['baseline']),
                ('Trend', trend_comp, colors['trend']),
                ('Seasonality', seasonality_comp, colors['seasonality']),
                ('Media', media_total.tolist(), colors['media']),
                ('Controls', controls_total.tolist(), colors['controls']),
                ('Geography', geo_effects, colors['geo']),
                ('Product', product_effects, colors['product']),
            ]
            
            components_added = 0
            for name, values, color in stack_order:
                if np.any(np.array(values) != 0):
                    fig_stack.add_trace(go.Scatter(
                        x=periods_display,
                        y=values,
                        name=name,
                        mode='lines',
                        stackgroup='components',
                        line=dict(width=0.5, color=color),
                        fillcolor=color,
                        hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
                    ))
                    components_added += 1
            
            if components_added == 0:
                st.warning("No component data available for stacked area chart.")
            else:
                # Add observed as line
                if observed:
                    fig_stack.add_trace(go.Scatter(
                        x=periods_display,
                        y=observed,
                        name='Observed',
                        mode='markers',
                        marker=dict(color='black', size=5, symbol='circle'),
                        hovertemplate="<b>Observed</b><br>%{x}<br>%{y:,.0f}<extra></extra>",
                    ))
                
                fig_stack.update_layout(
                    title="Stacked Component Contributions vs Observed",
                    xaxis_title="Period",
                    yaxis_title="Sales",
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
                
                st.plotly_chart(fig_stack, use_container_width=True, key="decomp_stacked")
                
                # Show individual media channels option
                if media_components:
                    with st.expander("📺 Show Individual Media Channels", expanded=False):
                        fig_media_stack = go.Figure()
                        
                        media_colors = [
                            '#4285f4', '#ea4335', '#fbbc04', '#34a853', 
                            '#ff6d01', '#46bdc6', '#9334e6', '#e91e63'
                        ]
                        
                        for i, (name, values) in enumerate(media_components.items()):
                            channel_name = name.replace("media_", "")
                            color = media_colors[i % len(media_colors)]
                            
                            fig_media_stack.add_trace(go.Scatter(
                                x=periods_display,
                                y=values,
                                name=channel_name,
                                mode='lines',
                                stackgroup='media',
                                line=dict(width=0.5),
                                hovertemplate=f"<b>{channel_name}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
                            ))
                        
                        fig_media_stack.update_layout(
                            title="Media Channel Contributions (Stacked)",
                            xaxis_title="Period",
                            yaxis_title="Media Contribution",
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            hovermode='x unified',
                        )
                        
                        st.plotly_chart(fig_media_stack, use_container_width=True, key="decomp_media_stack")
        
        # =================================================================
        # Tab 3: Component Time Series (Individual Lines)
        # =================================================================
        with decomp_tabs[2]:
            st.markdown("#### Individual Component Time Series")
            
            # Component selection
            all_components = ["Baseline", "Trend", "Seasonality", "Media (Total)", "Controls (Total)"]
            all_components.extend([f"Media: {k.replace('media_', '')}" for k in media_components.keys()])
            all_components.extend([f"Control: {k.replace('control_', '')}" for k in control_components.keys()])
            
            selected = st.multiselect(
                "Select Components to Display",
                options=all_components,
                default=["Trend", "Seasonality", "Media (Total)"],
                key="decomp_ts_select",
            )
            
            if selected:
                fig_ts = go.Figure()
                
                component_map = {
                    "Baseline": (baseline_comp, '#3498db'),
                    "Trend": (trend_comp, '#e74c3c'),
                    "Seasonality": (seasonality_comp, '#2ecc71'),
                    "Media (Total)": (media_total.tolist(), '#9b59b6'),
                    "Controls (Total)": (controls_total.tolist(), '#f39c12'),
                }
                
                # Add individual media/control components
                for k, v in media_components.items():
                    component_map[f"Media: {k.replace('media_', '')}"] = (v, '#9b59b6')
                for k, v in control_components.items():
                    component_map[f"Control: {k.replace('control_', '')}"] = (v, '#f39c12')
                
                for name in selected:
                    if name in component_map:
                        values, color = component_map[name]
                        fig_ts.add_trace(go.Scatter(
                            x=periods_display,
                            y=values,
                            name=name,
                            mode='lines',
                            line=dict(width=2),
                        ))
                
                fig_ts.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig_ts.update_layout(
                    title="Component Contributions Over Time",
                    xaxis_title="Period",
                    yaxis_title="Contribution",
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode='x unified',
                )
                
                st.plotly_chart(fig_ts, use_container_width=True, key="decomp_ts")
        
        # =================================================================
        # Tab 4: Geographic View
        # =================================================================
        with decomp_tabs[3]:
            st.markdown("#### Geographic Breakdown of Components")
            
            if len(geo_names) <= 1 or not by_geography:
                st.info("Geographic breakdown requires multiple geographies and geo-level data from the model.")
            else:
                # Component selector for geo view
                geo_component_options = []
                if "baseline" in by_geography:
                    geo_component_options.append("Baseline")
                if "trend" in by_geography:
                    geo_component_options.append("Trend")
                if "seasonality" in by_geography:
                    geo_component_options.append("Seasonality")
                
                # Add media channels
                media_geo_keys = [k for k in by_geography.keys() if k.startswith("media_")]
                for k in media_geo_keys:
                    geo_component_options.append(f"Media: {k.replace('media_', '')}")
                
                # Add controls
                control_geo_keys = [k for k in by_geography.keys() if k.startswith("control_")]
                for k in control_geo_keys:
                    geo_component_options.append(f"Control: {k.replace('control_', '')}")
                
                if not geo_component_options:
                    st.warning("No geographic breakdown data available.")
                else:
                    selected_geo_comp = st.selectbox(
                        "Select Component",
                        options=geo_component_options,
                        key="decomp_geo_component"
                    )
                    
                    # Map selection to data key
                    if selected_geo_comp == "Baseline":
                        geo_key = "baseline"
                    elif selected_geo_comp == "Trend":
                        geo_key = "trend"
                    elif selected_geo_comp == "Seasonality":
                        geo_key = "seasonality"
                    elif selected_geo_comp.startswith("Media:"):
                        geo_key = f"media_{selected_geo_comp.replace('Media: ', '')}"
                    elif selected_geo_comp.startswith("Control:"):
                        geo_key = f"control_{selected_geo_comp.replace('Control: ', '')}"
                    else:
                        geo_key = None
                    
                    if geo_key and geo_key in by_geography:
                        geo_data = by_geography[geo_key]
                        
                        # Plot by geography
                        fig_geo_decomp = go.Figure()
                        
                        geo_colors = [
                            '#4285f4', '#ea4335', '#fbbc04', '#34a853', 
                            '#ff6d01', '#46bdc6', '#9334e6', '#e91e63'
                        ]
                        
                        for i, (geo, values) in enumerate(geo_data.items()):
                            fig_geo_decomp.add_trace(go.Scatter(
                                x=periods_display,
                                y=values,
                                mode='lines',
                                name=geo,
                                line=dict(color=geo_colors[i % len(geo_colors)], width=2),
                                hovertemplate=f"<b>{geo}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
                            ))
                        
                        fig_geo_decomp.update_layout(
                            title=f"{selected_geo_comp} Contribution by Geography",
                            xaxis_title="Period",
                            yaxis_title="Contribution",
                            height=450,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            hovermode='x unified',
                        )
                        
                        st.plotly_chart(fig_geo_decomp, use_container_width=True, key="decomp_geo_lines")
                        
                        # Summary table
                        st.markdown("##### Geographic Contribution Summary")
                        
                        geo_summary = []
                        total_all_geos = sum(np.sum(v) for v in geo_data.values())
                        
                        for geo, values in geo_data.items():
                            arr = np.array(values)
                            total = np.sum(arr)
                            pct = (total / total_all_geos * 100) if total_all_geos != 0 else 0
                            geo_summary.append({
                                "Geography": geo,
                                "Total": f"{total:,.0f}",
                                "% Share": f"{pct:.1f}%",
                                "Mean": f"{np.mean(arr):,.0f}",
                                "Std Dev": f"{np.std(arr):,.0f}",
                            })
                        
                        st.dataframe(pd.DataFrame(geo_summary), use_container_width=True, hide_index=True)
                        
                        # Stacked area by geo
                        with st.expander("📊 Stacked Area by Geography", expanded=False):
                            fig_geo_stack = go.Figure()
                            
                            for i, (geo, values) in enumerate(geo_data.items()):
                                fig_geo_stack.add_trace(go.Scatter(
                                    x=periods_display,
                                    y=values,
                                    name=geo,
                                    mode='lines',
                                    stackgroup='geos',
                                    line=dict(width=0.5),
                                    hovertemplate=f"<b>{geo}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
                                ))
                            
                            fig_geo_stack.update_layout(
                                title=f"Stacked {selected_geo_comp} by Geography",
                                xaxis_title="Period",
                                yaxis_title="Contribution",
                                height=400,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            )
                            
                            st.plotly_chart(fig_geo_stack, use_container_width=True, key="decomp_geo_stack")
                        
                        # Observed by geography if available
                        if observed_by_geo:
                            with st.expander("📈 Observed Values by Geography", expanded=False):
                                fig_obs_geo = go.Figure()
                                
                                for i, (geo, values) in enumerate(observed_by_geo.items()):
                                    fig_obs_geo.add_trace(go.Scatter(
                                        x=periods_display,
                                        y=values,
                                        mode='lines',
                                        name=geo,
                                        line=dict(width=2),
                                    ))
                                
                                fig_obs_geo.update_layout(
                                    title="Observed Sales by Geography",
                                    xaxis_title="Period",
                                    yaxis_title="Sales",
                                    height=400,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                    hovermode='x unified',
                                )
                                
                                st.plotly_chart(fig_obs_geo, use_container_width=True, key="obs_by_geo")
        
        # =================================================================
        # Tab 5: Year-over-Year Waterfall
        # =================================================================
        with decomp_tabs[4]:
            st.markdown("#### Year-over-Year Change Analysis")
            
            if period_dates is None:
                st.info("""
                📅 **Date-formatted periods required**
                
                Year-over-year analysis requires periods that can be parsed as dates 
                (e.g., '2023-01-01', '2023-W01', etc.).
                
                Your data has numeric or non-date period identifiers.
                """)
            else:
                # Create DataFrame with all data
                df = pd.DataFrame({
                    'period': period_dates,
                    'observed': observed,
                    'baseline': baseline_comp,
                    'trend': trend_comp,
                    'seasonality': seasonality_comp,
                    'media': media_total,
                    'controls': controls_total,
                })
                df['year'] = df['period'].dt.year
                
                # Aggregate by year
                yearly = df.groupby('year').agg({
                    'observed': 'sum',
                    'baseline': 'sum',
                    'trend': 'sum',
                    'seasonality': 'sum',
                    'media': 'sum',
                    'controls': 'sum',
                }).reset_index()
                
                if len(yearly) < 2:
                    st.info("Need at least 2 years of data for YoY analysis.")
                else:
                    # Year selector
                    years = sorted(yearly['year'].unique())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        base_year = st.selectbox("Base Year", options=years[:-1], index=0, key="yoy_base")
                    with col2:
                        compare_years = [y for y in years if y > base_year]
                        if compare_years:
                            compare_year = st.selectbox("Compare Year", options=compare_years, index=0, key="yoy_compare")
                        else:
                            st.warning("No years available for comparison.")
                            compare_year = None
                    
                    if compare_year:
                        base_data = yearly[yearly['year'] == base_year].iloc[0]
                        compare_data = yearly[yearly['year'] == compare_year].iloc[0]
                        
                        # Calculate changes
                        changes = {
                            'Baseline': compare_data['baseline'] - base_data['baseline'],
                            'Trend': compare_data['trend'] - base_data['trend'],
                            'Seasonality': compare_data['seasonality'] - base_data['seasonality'],
                            'Media': compare_data['media'] - base_data['media'],
                            'Controls': compare_data['controls'] - base_data['controls'],
                        }
                        
                        # Add individual media channel changes
                        for k, v in media_components.items():
                            channel_name = k.replace("media_", "")
                            df[f'media_{channel_name}'] = v
                        
                        # Re-aggregate if we added media columns
                        if media_components:
                            df_media = df.copy()
                            df_media['year'] = df_media['period'].dt.year
                            media_yearly = df_media.groupby('year')[[f'media_{k.replace("media_", "")}' for k in media_components.keys()]].sum().reset_index()
                        
                        # Build waterfall
                        base_total = base_data['observed']
                        compare_total = compare_data['observed']
                        
                        waterfall_data = [
                            {'x': f'{base_year} Total', 'y': base_total, 'measure': 'absolute'},
                        ]
                        
                        for name, change in changes.items():
                            waterfall_data.append({
                                'x': f'Δ {name}',
                                'y': change,
                                'measure': 'relative',
                            })
                        
                        waterfall_data.append({
                            'x': f'{compare_year} Total',
                            'y': compare_total,
                            'measure': 'total',
                        })
                        
                        wf_df = pd.DataFrame(waterfall_data)
                        
                        # Create waterfall chart
                        fig_waterfall = go.Figure(go.Waterfall(
                            name="YoY Change",
                            orientation="v",
                            measure=wf_df['measure'],
                            x=wf_df['x'],
                            y=wf_df['y'],
                            textposition="outside",
                            text=[f"{v:+,.0f}" if m == 'relative' else f"{v:,.0f}" 
                                  for v, m in zip(wf_df['y'], wf_df['measure'])],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "#2ecc71"}},
                            decreasing={"marker": {"color": "#e74c3c"}},
                            totals={"marker": {"color": "#3498db"}},
                        ))
                        
                        fig_waterfall.update_layout(
                            title=f"Year-over-Year Change: {base_year} → {compare_year}",
                            xaxis_title="",
                            yaxis_title="Sales",
                            height=500,
                            showlegend=False,
                        )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True, key="yoy_waterfall")
                        
                        # Summary table
                        st.markdown("##### Change Summary")
                        
                        change_summary = []
                        for name, change in changes.items():
                            pct_change = (change / base_data['observed'] * 100) if base_data['observed'] != 0 else 0
                            change_summary.append({
                                'Component': name,
                                f'{base_year}': f"{base_data[name.lower().replace(' ', '_').replace('(total)', '').strip()]:,.0f}" if name.lower().replace(' ', '_').replace('(total)', '').strip() in base_data else "N/A",
                                f'{compare_year}': f"{compare_data[name.lower().replace(' ', '_').replace('(total)', '').strip()]:,.0f}" if name.lower().replace(' ', '_').replace('(total)', '').strip() in compare_data else "N/A",
                                'Change': f"{change:+,.0f}",
                                '% Impact': f"{pct_change:+.1f}%",
                            })
                        
                        # Simpler approach - just show the changes we calculated
                        change_df = pd.DataFrame([
                            {'Component': 'Baseline', f'{base_year}': f"{base_data['baseline']:,.0f}", f'{compare_year}': f"{compare_data['baseline']:,.0f}", 'Change': f"{changes['Baseline']:+,.0f}", '% of Total Change': f"{changes['Baseline']/(compare_total-base_total)*100:.1f}%" if (compare_total-base_total) != 0 else "0%"},
                            {'Component': 'Trend', f'{base_year}': f"{base_data['trend']:,.0f}", f'{compare_year}': f"{compare_data['trend']:,.0f}", 'Change': f"{changes['Trend']:+,.0f}", '% of Total Change': f"{changes['Trend']/(compare_total-base_total)*100:.1f}%" if (compare_total-base_total) != 0 else "0%"},
                            {'Component': 'Seasonality', f'{base_year}': f"{base_data['seasonality']:,.0f}", f'{compare_year}': f"{compare_data['seasonality']:,.0f}", 'Change': f"{changes['Seasonality']:+,.0f}", '% of Total Change': f"{changes['Seasonality']/(compare_total-base_total)*100:.1f}%" if (compare_total-base_total) != 0 else "0%"},
                            {'Component': 'Media', f'{base_year}': f"{base_data['media']:,.0f}", f'{compare_year}': f"{compare_data['media']:,.0f}", 'Change': f"{changes['Media']:+,.0f}", '% of Total Change': f"{changes['Media']/(compare_total-base_total)*100:.1f}%" if (compare_total-base_total) != 0 else "0%"},
                            {'Component': 'Controls', f'{base_year}': f"{base_data['controls']:,.0f}", f'{compare_year}': f"{compare_data['controls']:,.0f}", 'Change': f"{changes['Controls']:+,.0f}", '% of Total Change': f"{changes['Controls']/(compare_total-base_total)*100:.1f}%" if (compare_total-base_total) != 0 else "0%"},
                        ])
                        
                        st.dataframe(change_df, use_container_width=True, hide_index=True)
                        
                        # Total change
                        total_change = compare_total - base_total
                        pct_total = (total_change / base_total * 100) if base_total != 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{base_year} Total", f"{base_total:,.0f}")
                        with col2:
                            st.metric(f"{compare_year} Total", f"{compare_total:,.0f}")
                        with col3:
                            st.metric("Change", f"{total_change:+,.0f}", delta=f"{pct_total:+.1f}%")
        
        # =================================================================
        # Tab 6: Contribution Breakdown (Relative)
        # =================================================================
        with decomp_tabs[5]:
            st.markdown("#### Relative Contribution Over Time")
            
            # Calculate relative contributions (percentage of total)
            total_by_period = np.array(baseline_comp) + np.array(trend_comp) + np.array(seasonality_comp) + media_total + controls_total
            
            # Avoid division by zero
            total_by_period = np.where(total_by_period == 0, 1, total_by_period)
            
            fig_relative = go.Figure()
            
            stack_order_rel = [
                ('Baseline', np.array(baseline_comp) / total_by_period * 100, '#3498db'),
                ('Trend', np.array(trend_comp) / total_by_period * 100, '#e74c3c'),
                ('Seasonality', np.array(seasonality_comp) / total_by_period * 100, '#2ecc71'),
                ('Media', media_total / total_by_period * 100, '#9b59b6'),
                ('Controls', controls_total / total_by_period * 100, '#f39c12'),
            ]
            
            for name, values, color in stack_order_rel:
                if np.any(values != 0):
                    fig_relative.add_trace(go.Scatter(
                        x=periods_display,
                        y=values,
                        name=name,
                        mode='lines',
                        stackgroup='components',
                        line=dict(width=0.5, color=color),
                        fillcolor=color,
                        hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
                    ))
            
            fig_relative.update_layout(
                title="Relative Component Contributions (%)",
                xaxis_title="Period",
                yaxis_title="% of Predicted",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified',
                yaxis=dict(range=[0, 100]),
            )
            
            st.plotly_chart(fig_relative, use_container_width=True, key="decomp_relative")
            
            st.info("""
            💡 **Interpretation:**
            - This chart shows how the relative importance of each component changes over time
            - A growing media share indicates increasing marketing effectiveness
            - Seasonal patterns will show periodic fluctuations in the seasonality component
            """)
    
    except APIError as e:
        display_api_error(e)
    except Exception as e:
        st.error(f"Error loading decomposition: {e}")
        import traceback
        st.code(traceback.format_exc())


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
        "📅 Trend & Seasonality",
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
        render_trend_seasonality_tab(model_id)
    
    with tabs[6]:
        render_contributions_tab(model_id)
    
    with tabs[7]:
        render_decomposition_tab(model_id)
    
    with tabs[8]:
        render_summary_tab(model_id)


if __name__ == "__main__":
    main()
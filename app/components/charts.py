"""
Chart Components for MMM Visualization.

Provides reusable Plotly chart components for model results visualization.
All chart functions are designed to work with st.fragment for isolated reruns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any

from components.common import CHART_COLORS, COMPONENT_COLORS, rgb_to_rgba


# =============================================================================
# Model Fit Charts
# =============================================================================

@st.fragment
def plot_model_fit(
    periods: list,
    observed: list[float],
    predicted_mean: list[float],
    predicted_std: list[float] | None = None,
    title: str = "Model Fit: Observed vs Predicted",
    y_label: str = "Value",
):
    """Plot observed vs predicted values with uncertainty bands."""
    fig = go.Figure()
    
    # Add uncertainty band if available
    if predicted_std is not None:
        upper = np.array(predicted_mean) + 1.96 * np.array(predicted_std)
        lower = np.array(predicted_mean) - 1.96 * np.array(predicted_std)
        
        fig.add_trace(go.Scatter(
            x=list(periods) + list(periods)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill='toself',
            fillcolor=rgb_to_rgba(COMPONENT_COLORS["Predicted"], 0.2),
            line=dict(color='rgba(0,0,0,0)'),
            name='95% CI',
            showlegend=True,
        ))
    
    # Observed values
    fig.add_trace(go.Scatter(
        x=periods,
        y=observed,
        mode='lines+markers',
        name='Observed',
        line=dict(color=COMPONENT_COLORS["Observed"], width=2),
        marker=dict(size=4),
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=periods,
        y=predicted_mean,
        mode='lines',
        name='Predicted',
        line=dict(color=COMPONENT_COLORS["Predicted"], width=2, dash='dash'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_residuals(
    periods: list,
    residuals: list[float],
    title: str = "Model Residuals",
):
    """Plot model residuals over time."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Residuals Over Time", "Residual Distribution", "Residual ACF", "Q-Q Plot"),
        specs=[[{}, {}], [{}, {}]],
    )
    
    # Residuals over time
    fig.add_trace(go.Scatter(
        x=periods,
        y=residuals,
        mode='markers',
        marker=dict(color=COMPONENT_COLORS["Residual"], size=5),
        name='Residuals',
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residual histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color=COMPONENT_COLORS["Residual"],
        name='Distribution',
    ), row=1, col=2)
    
    # Simple ACF (first 20 lags)
    residuals_array = np.array(residuals)
    n = len(residuals_array)
    acf_values = [1.0]
    for lag in range(1, min(21, n)):
        acf = np.corrcoef(residuals_array[lag:], residuals_array[:-lag])[0, 1]
        acf_values.append(acf)
    
    fig.add_trace(go.Bar(
        x=list(range(len(acf_values))),
        y=acf_values,
        marker_color=COMPONENT_COLORS["Residual"],
        name='ACF',
    ), row=2, col=1)
    
    # Confidence bounds for ACF
    conf = 1.96 / np.sqrt(n)
    fig.add_hline(y=conf, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-conf, line_dash="dash", line_color="red", row=2, col=1)
    
    # Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_residuals))
    from scipy import stats
    theoretical_values = stats.norm.ppf(theoretical_quantiles)
    
    fig.add_trace(go.Scatter(
        x=theoretical_values,
        y=sorted_residuals,
        mode='markers',
        marker=dict(color=COMPONENT_COLORS["Residual"], size=4),
        name='Q-Q',
    ), row=2, col=2)
    
    # Add reference line
    min_val = min(theoretical_values.min(), sorted_residuals.min())
    max_val = max(theoretical_values.max(), sorted_residuals.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Reference',
        showlegend=False,
    ), row=2, col=2)
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Contribution Charts
# =============================================================================

@st.fragment
def plot_channel_contributions(
    contributions: dict[str, float],
    title: str = "Channel Contributions",
    show_percentage: bool = True,
):
    """Plot channel contributions as a bar chart."""
    channels = list(contributions.keys())
    values = list(contributions.values())
    total = sum(values)
    
    if show_percentage:
        percentages = [v / total * 100 for v in values]
        text = [f"{p:.1f}%" for p in percentages]
    else:
        text = [f"{v:,.0f}" for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=channels,
        orientation='h',
        text=text,
        textposition='auto',
        marker_color=CHART_COLORS[:len(channels)],
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Contribution",
        yaxis_title="Channel",
        height=max(300, len(channels) * 40),
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_contribution_waterfall(
    contributions: dict[str, float],
    baseline: float = 0,
    title: str = "Contribution Waterfall",
):
    """Plot contributions as a waterfall chart."""
    channels = list(contributions.keys())
    values = list(contributions.values())
    
    # Build waterfall data
    measure = ["absolute"] + ["relative"] * len(channels) + ["total"]
    x = ["Baseline"] + channels + ["Total"]
    y = [baseline] + values + [None]
    
    fig = go.Figure(go.Waterfall(
        measure=measure,
        x=x,
        y=y,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": CHART_COLORS[0]}},
        decreasing={"marker": {"color": CHART_COLORS[1]}},
        totals={"marker": {"color": CHART_COLORS[2]}},
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_contribution_pie(
    contributions: dict[str, float],
    title: str = "Contribution Share",
):
    """Plot contributions as a pie chart."""
    channels = list(contributions.keys())
    values = list(contributions.values())
    
    fig = go.Figure(go.Pie(
        labels=channels,
        values=values,
        hole=0.4,
        marker_colors=CHART_COLORS[:len(channels)],
    ))
    
    fig.update_layout(
        title=title,
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_contribution_timeseries(
    periods: list,
    contributions_by_channel: dict[str, list[float]],
    title: str = "Channel Contributions Over Time",
    stacked: bool = True,
):
    """Plot channel contributions over time."""
    fig = go.Figure()
    
    for i, (channel, values) in enumerate(contributions_by_channel.items()):
        fig.add_trace(go.Scatter(
            x=periods,
            y=values,
            name=channel,
            mode='lines',
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            stackgroup='one' if stacked else None,
            fill='tonexty' if stacked else None,
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Contribution",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Response Curves
# =============================================================================

@st.fragment
def plot_response_curves(
    curves: dict[str, dict[str, Any]],
    title: str = "Channel Response Curves",
):
    """
    Plot response curves for multiple channels.
    
    curves: dict mapping channel name to {spend, response, current_spend}
    """
    n_channels = len(curves)
    n_cols = min(3, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(curves.keys()),
    )
    
    for i, (channel, data) in enumerate(curves.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(go.Scatter(
            x=data["spend"],
            y=data["response"],
            mode='lines',
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            name=channel,
            showlegend=False,
        ), row=row, col=col)
        
        # Mark current spend
        if "current_spend" in data:
            current_idx = np.argmin(np.abs(np.array(data["spend"]) - data["current_spend"]))
            fig.add_trace(go.Scatter(
                x=[data["spend"][current_idx]],
                y=[data["response"][current_idx]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name='Current',
                showlegend=False,
            ), row=row, col=col)
    
    fig.update_layout(
        title=title,
        height=300 * n_rows,
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_marginal_roas(
    roas_data: dict[str, dict[str, float]],
    title: str = "Marginal ROAS by Channel",
):
    """
    Plot marginal ROAS for each channel with uncertainty.
    
    roas_data: dict mapping channel to {mean, lower, upper}
    """
    channels = list(roas_data.keys())
    means = [roas_data[c]["mean"] for c in channels]
    lowers = [roas_data[c].get("lower", roas_data[c]["mean"]) for c in channels]
    uppers = [roas_data[c].get("upper", roas_data[c]["mean"]) for c in channels]
    
    errors_lower = [m - l for m, l in zip(means, lowers)]
    errors_upper = [u - m for m, u in zip(means, uppers)]
    
    fig = go.Figure(go.Bar(
        x=channels,
        y=means,
        error_y=dict(
            type='data',
            symmetric=False,
            array=errors_upper,
            arrayminus=errors_lower,
        ),
        marker_color=CHART_COLORS[:len(channels)],
    ))
    
    fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Break-even")
    
    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title="Marginal ROAS",
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Component Decomposition
# =============================================================================

@st.fragment
def plot_component_decomposition(
    periods: list,
    components: dict[str, list[float]],
    observed: list[float] | None = None,
    title: str = "Component Decomposition",
):
    """Plot stacked component decomposition."""
    fig = go.Figure()
    
    # Add component traces
    for component, values in components.items():
        color = COMPONENT_COLORS.get(component, CHART_COLORS[len(fig.data) % len(CHART_COLORS)])
        fig.add_trace(go.Scatter(
            x=periods,
            y=values,
            name=component,
            mode='lines',
            line=dict(width=0),
            stackgroup='one',
            fillcolor=rgb_to_rgba(color, 0.6),
        ))
    
    # Add observed line
    if observed is not None:
        fig.add_trace(go.Scatter(
            x=periods,
            y=observed,
            name='Observed',
            mode='lines+markers',
            line=dict(color='black', width=2),
            marker=dict(size=4),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Value",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified',
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Posterior Distribution Charts
# =============================================================================

@st.fragment
def plot_posterior_distributions(
    posteriors: dict[str, dict[str, Any]],
    title: str = "Posterior Distributions",
):
    """
    Plot posterior distributions for parameters.
    
    posteriors: dict mapping param name to {samples, mean, hdi_low, hdi_high}
    """
    n_params = len(posteriors)
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(posteriors.keys()),
    )
    
    for i, (param, data) in enumerate(posteriors.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(go.Histogram(
            x=data["samples"],
            nbinsx=50,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            opacity=0.7,
            showlegend=False,
        ), row=row, col=col)
        
        # Add mean line
        if "mean" in data:
            fig.add_vline(
                x=data["mean"],
                line_dash="solid",
                line_color="red",
                row=row,
                col=col,
            )
        
        # Add HDI lines
        if "hdi_low" in data and "hdi_high" in data:
            fig.add_vline(
                x=data["hdi_low"],
                line_dash="dash",
                line_color="gray",
                row=row,
                col=col,
            )
            fig.add_vline(
                x=data["hdi_high"],
                line_dash="dash",
                line_color="gray",
                row=row,
                col=col,
            )
    
    fig.update_layout(
        title=title,
        height=250 * n_rows,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def plot_trace(
    samples: np.ndarray,
    param_name: str,
    n_chains: int = 4,
):
    """Plot MCMC trace for a parameter."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Trace", "Density"],
    )
    
    # Reshape samples to (chains, draws)
    n_total = len(samples)
    n_draws = n_total // n_chains
    samples_by_chain = samples.reshape(n_chains, n_draws)
    
    # Trace plot
    for i in range(n_chains):
        fig.add_trace(go.Scatter(
            y=samples_by_chain[i],
            mode='lines',
            line=dict(width=0.5),
            opacity=0.7,
            name=f'Chain {i+1}',
        ), row=1, col=1)
    
    # Density plot
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=50,
        opacity=0.7,
        showlegend=False,
    ), row=1, col=2)
    
    fig.update_layout(
        title=f"Trace Plot: {param_name}",
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Scenario Planning Charts
# =============================================================================

@st.fragment
def plot_scenario_comparison(
    baseline: float,
    scenario: float,
    channel_effects: dict[str, float],
    title: str = "Scenario Comparison",
):
    """Plot scenario comparison showing baseline vs scenario outcome."""
    change = scenario - baseline
    change_pct = change / baseline * 100 if baseline != 0 else 0
    
    # Create comparison bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Baseline", "Scenario"],
        y=[baseline, scenario],
        marker_color=[CHART_COLORS[0], CHART_COLORS[1]],
        text=[f"{baseline:,.0f}", f"{scenario:,.0f}"],
        textposition='outside',
    ))
    
    # Add change annotation
    fig.add_annotation(
        x=1,
        y=max(baseline, scenario) * 1.1,
        text=f"Change: {change:+,.0f} ({change_pct:+.1f}%)",
        showarrow=False,
        font=dict(size=14, color='green' if change >= 0 else 'red'),
    )
    
    fig.update_layout(
        title=title,
        yaxis_title="Outcome",
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Channel effects breakdown
    if channel_effects:
        st.markdown("### Channel Effects")
        effect_df = pd.DataFrame([
            {"Channel": k, "Effect": v}
            for k, v in channel_effects.items()
        ])
        st.dataframe(effect_df, use_container_width=True, hide_index=True)


@st.fragment
def plot_budget_optimization(
    current_allocation: dict[str, float],
    optimal_allocation: dict[str, float],
    title: str = "Budget Allocation Comparison",
):
    """Plot current vs optimal budget allocation."""
    channels = list(current_allocation.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current',
        x=channels,
        y=[current_allocation[c] for c in channels],
        marker_color=CHART_COLORS[0],
    ))
    
    fig.add_trace(go.Bar(
        name='Optimal',
        x=channels,
        y=[optimal_allocation[c] for c in channels],
        marker_color=CHART_COLORS[1],
    ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title="Channel",
        yaxis_title="Budget",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)
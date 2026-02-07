"""
Chart Components for MMM Framework.

Provides reusable Plotly chart components with consistent styling.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Any

# =============================================================================
# Color Utilities
# =============================================================================

CHART_COLORS = px.colors.qualitative.Set2
COMPONENT_COLORS = {
    "baseline": "#636EFA",
    "trend": "#EF553B",
    "seasonality": "#00CC96",
    "media": "#AB63FA",
    "control": "#FFA15A",
    "residual": "#19D3F3",
}


def rgb_to_rgba(color: str, alpha: float = 0.3) -> str:
    """Convert hex or rgb color to rgba with specified alpha."""
    if color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    elif color.startswith("rgb"):
        return color.replace("rgb", "rgba").replace(")", f",{alpha})")
    return color


# =============================================================================
# Model Fit Charts
# =============================================================================


@st.fragment
def plot_model_fit(
    periods: list,
    observed: list[float],
    predicted_mean: list[float],
    predicted_std: list[float] | None = None,
    y_label: str = "Value",
    title: str = "Model Fit: Observed vs Predicted",
):
    """Plot observed vs predicted values with uncertainty band."""
    fig = go.Figure()

    # Uncertainty band
    if predicted_std:
        upper = [m + 2 * s for m, s in zip(predicted_mean, predicted_std)]
        lower = [m - 2 * s for m, s in zip(predicted_mean, predicted_std)]

        fig.add_trace(
            go.Scatter(
                x=periods + periods[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=rgb_to_rgba(CHART_COLORS[0], 0.2),
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            )
        )

    # Observed
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=observed,
            mode="lines+markers",
            name="Observed",
            line=dict(color=CHART_COLORS[1], width=2),
            marker=dict(size=4),
        )
    )

    # Predicted
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=predicted_mean,
            mode="lines",
            name="Predicted",
            line=dict(color=CHART_COLORS[0], width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"model_fit_{hash(str(periods[:5]))}"
    )


@st.fragment
def plot_residuals(
    periods: list, residuals: list[float], title: str = "Residual Analysis"
):
    """Plot residuals with diagnostic subplots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals Over Time",
            "Residual Distribution",
            "Q-Q Plot",
            "ACF",
        ),
    )

    # Time series
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=residuals,
            mode="lines+markers",
            name="Residuals",
            line=dict(color=CHART_COLORS[0]),
            marker=dict(size=3),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name="Distribution",
            marker_color=CHART_COLORS[1],
            nbinsx=30,
        ),
        row=1,
        col=2,
    )

    # Q-Q plot (approximate)
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical = np.percentile(
        np.random.normal(0, np.std(residuals), 10000), np.linspace(0, 100, n)
    )

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sorted_res,
            mode="markers",
            name="Q-Q",
            marker=dict(color=CHART_COLORS[2], size=4),
        ),
        row=2,
        col=1,
    )

    # 45-degree line for Q-Q
    line_range = [min(theoretical), max(theoretical)]
    fig.add_trace(
        go.Scatter(
            x=line_range,
            y=line_range,
            mode="lines",
            line=dict(color="red", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ACF (simple implementation)
    n_lags = min(20, len(residuals) // 4)
    acf_values = []
    for lag in range(n_lags):
        if lag == 0:
            acf_values.append(1.0)
        else:
            acf = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
            acf_values.append(acf)

    fig.add_trace(
        go.Bar(
            x=list(range(n_lags)),
            y=acf_values,
            name="ACF",
            marker_color=CHART_COLORS[3],
        ),
        row=2,
        col=2,
    )

    # Confidence bands for ACF
    conf = 1.96 / np.sqrt(len(residuals))
    fig.add_hline(y=conf, line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_hline(y=-conf, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        title=title,
        height=600,
        showlegend=False,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"residuals_{hash(str(residuals[:5]))}"
    )


# =============================================================================
# Contribution Charts
# =============================================================================


@st.fragment
def plot_channel_contributions(
    contributions: dict[str, float],
    show_percentage: bool = True,
    title: str = "Channel Contributions",
):
    """Plot bar chart of channel contributions."""
    channels = list(contributions.keys())
    values = list(contributions.values())

    total = sum(values) if sum(values) > 0 else 1

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=channels,
            y=values,
            text=[
                f"{v:,.0f}<br>({v/total*100:.1f}%)" if show_percentage else f"{v:,.0f}"
                for v in values
            ],
            textposition="outside",
            marker_color=[
                CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(channels))
            ],
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title="Contribution",
        height=400,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"contrib_bar_{hash(str(channels))}"
    )


@st.fragment
def plot_contribution_waterfall(
    contributions: dict[str, float],
    baseline: float = 0,
    title: str = "Contribution Waterfall",
):
    """Plot waterfall chart of contributions."""
    channels = list(contributions.keys())
    values = list(contributions.values())

    # Add baseline if provided
    if baseline > 0:
        channels = ["Baseline"] + channels + ["Total"]
        values = [baseline] + values + [baseline + sum(values)]
        measures = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]
    else:
        channels = channels + ["Total"]
        values = values + [sum(values)]
        measures = ["relative"] * (len(values) - 1) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            name="Contributions",
            orientation="v",
            measure=measures,
            x=channels,
            y=values,
            textposition="outside",
            text=[f"{v:,.0f}" for v in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title=title,
        height=400,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"contrib_waterfall_{hash(str(channels))}"
    )


@st.fragment
def plot_contribution_pie(
    contributions: dict[str, float],
    title: str = "Contribution Share",
):
    """Plot pie chart of contributions."""
    channels = list(contributions.keys())
    values = list(contributions.values())

    fig = go.Figure(
        go.Pie(
            labels=channels,
            values=values,
            textinfo="label+percent",
            marker_colors=[
                CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(channels))
            ],
        )
    )

    fig.update_layout(
        title=title,
        height=400,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"contrib_pie_{hash(str(channels))}"
    )


@st.fragment
def plot_contribution_timeseries(
    periods: list,
    contributions_by_channel: dict[str, list[float]],
    stacked: bool = True,
    title: str = "Contributions Over Time",
):
    """Plot time series of contributions by channel."""
    fig = go.Figure()

    for i, (channel, values) in enumerate(contributions_by_channel.items()):
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                name=channel,
                mode="lines",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                stackgroup="one" if stacked else None,
                fill="tonexty" if stacked else None,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Contribution",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"contrib_ts_{hash(str(periods[:5]))}"
    )


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
    if n_channels == 0:
        st.info("No response curve data available.")
        return

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

        spend = data.get("spend", [])
        response = data.get("response", [])

        if not spend or not response:
            continue

        fig.add_trace(
            go.Scatter(
                x=spend,
                y=response,
                mode="lines",
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                name=channel,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Mark current spend if available
        current_spend = data.get("current_spend")
        if current_spend is not None and spend:
            spend_arr = np.array(spend)
            current_idx = int(np.argmin(np.abs(spend_arr - current_spend)))
            fig.add_trace(
                go.Scatter(
                    x=[spend[current_idx]],
                    y=[response[current_idx]],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="star"),
                    name="Current",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=title,
        height=300 * n_rows,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"response_curves_{hash(str(list(curves.keys())))}",
    )


@st.fragment
def plot_marginal_roas(
    roas_data: dict[str, dict[str, float]],
    title: str = "Marginal ROAS by Channel",
):
    """
    Plot marginal ROAS for each channel with uncertainty.

    roas_data: dict mapping channel name to {mean, std, hdi_low, hdi_high}
    """
    channels = list(roas_data.keys())

    if not channels:
        st.info("No ROAS data available.")
        return

    means = [roas_data[ch].get("mean", 0) for ch in channels]
    errors_low = [
        roas_data[ch].get("mean", 0) - roas_data[ch].get("hdi_low", 0)
        for ch in channels
    ]
    errors_high = [
        roas_data[ch].get("hdi_high", 0) - roas_data[ch].get("mean", 0)
        for ch in channels
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=channels,
            y=means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=errors_high,
                arrayminus=errors_low,
            ),
            marker_color=[
                CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(channels))
            ],
            text=[f"{m:.2f}" for m in means],
            textposition="outside",
        )
    )

    fig.add_hline(
        y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title="Marginal ROAS",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"roas_{hash(str(channels))}")


# =============================================================================
# Decomposition Charts
# =============================================================================


@st.fragment
def plot_component_decomposition(
    periods: list,
    components: dict[str, list[float]],
    observed: list[float] | None = None,
    title: str = "Component Decomposition",
):
    """Plot stacked area chart of component decomposition."""
    fig = go.Figure()

    # Add component traces
    for i, (component, values) in enumerate(components.items()):
        # Determine color
        if component.lower() in COMPONENT_COLORS:
            color = COMPONENT_COLORS[component.lower()]
        elif "media" in component.lower():
            color = COMPONENT_COLORS["media"]
        elif "control" in component.lower():
            color = COMPONENT_COLORS["control"]
        else:
            color = CHART_COLORS[i % len(CHART_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                name=component,
                mode="lines",
                line=dict(width=0),
                stackgroup="one",
                fillcolor=rgb_to_rgba(color, 0.7),
            )
        )

    # Add observed line if provided
    if observed:
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=observed,
                name="Observed",
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Value",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"decomposition_{hash(str(periods[:5]))}"
    )


# =============================================================================
# Posterior Charts
# =============================================================================


@st.fragment
def plot_posterior_distributions(
    posteriors: dict[str, dict[str, Any]],
    title: str = "Posterior Distributions",
):
    """
    Plot posterior distributions for parameters.

    posteriors: dict mapping parameter name to {samples, mean, std} or just list of samples
    """
    n_params = len(posteriors)
    if n_params == 0:
        st.info("No posterior data to display.")
        return

    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(posteriors.keys()),
    )

    for i, (param, data) in enumerate(posteriors.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Handle different data formats
        if isinstance(data, dict):
            samples = data.get("samples", [])
            mean_val = data.get("mean")
        elif isinstance(data, list):
            samples = data
            mean_val = np.mean(samples) if samples else None
        else:
            continue

        if not samples:
            continue

        samples = np.array(samples).flatten()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=samples,
                name=param,
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                nbinsx=50,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add mean line
        if mean_val is not None:
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                row=row,
                col=col,
            )

    fig.update_layout(
        title=title,
        height=250 * n_rows,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"posteriors_{hash(str(list(posteriors.keys())))}",
    )


@st.fragment
def plot_trace(
    samples: np.ndarray,
    param_name: str,
    n_chains: int = 4,
):
    """Plot trace plot for a parameter."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Trace", "Distribution"])

    # Assume samples are (chains * draws,) and reshape
    total_samples = len(samples)
    draws_per_chain = total_samples // n_chains

    for i in range(n_chains):
        chain_samples = samples[i * draws_per_chain : (i + 1) * draws_per_chain]
        fig.add_trace(
            go.Scatter(
                y=chain_samples,
                mode="lines",
                name=f"Chain {i+1}",
                line=dict(width=0.5),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

    # Distribution
    fig.add_trace(
        go.Histogram(
            x=samples,
            name="Posterior",
            marker_color=CHART_COLORS[0],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Trace: {param_name}",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"trace_{param_name}")


# =============================================================================
# Scenario Charts
# =============================================================================


@st.fragment
def plot_scenario_comparison(
    baseline: dict[str, float],
    scenario: dict[str, float],
    labels: tuple[str, str] = ("Baseline", "Scenario"),
    title: str = "Scenario Comparison",
):
    """Plot side-by-side comparison of baseline and scenario."""
    metrics = list(baseline.keys())

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name=labels[0],
            x=metrics,
            y=[baseline[m] for m in metrics],
            marker_color=CHART_COLORS[0],
        )
    )

    fig.add_trace(
        go.Bar(
            name=labels[1],
            x=metrics,
            y=[scenario[m] for m in metrics],
            marker_color=CHART_COLORS[1],
        )
    )

    fig.update_layout(
        title=title,
        barmode="group",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"scenario_{hash(str(metrics))}")


@st.fragment
def plot_budget_optimization(
    current_allocation: dict[str, float],
    optimal_allocation: dict[str, float],
    title: str = "Budget Optimization",
):
    """Plot current vs optimal budget allocation."""
    channels = list(current_allocation.keys())

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Current Allocation", "Optimal Allocation"],
    )

    fig.add_trace(
        go.Pie(
            labels=channels,
            values=[current_allocation[ch] for ch in channels],
            name="Current",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            labels=channels,
            values=[optimal_allocation[ch] for ch in channels],
            name="Optimal",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=400,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"optimization_{hash(str(channels))}"
    )

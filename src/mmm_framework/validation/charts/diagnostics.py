"""
Diagnostic visualization charts for model validation.

Provides Plotly-based charts for residual diagnostics and channel analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

if TYPE_CHECKING:
    from ..results import ChannelDiagnosticsResults, ResidualDiagnosticsResults


def create_residual_panel(
    results: ResidualDiagnosticsResults,
    title: str = "Residual Diagnostics",
) -> go.Figure:
    """
    Create multi-panel residual diagnostic chart.

    Includes:
    - Residuals vs Fitted
    - Q-Q Plot
    - ACF Plot
    - Residuals over time

    Parameters
    ----------
    results : ResidualDiagnosticsResults
        Results from residual diagnostics.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with 4 diagnostic panels.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Residuals vs Fitted",
            "Q-Q Plot",
            "ACF",
            "Residuals Over Time",
        ),
    )

    # Convert to lists for JSON serialization
    residuals = results.residuals.tolist() if hasattr(results.residuals, 'tolist') else list(results.residuals)
    fitted = results.fitted_values.tolist() if hasattr(results.fitted_values, 'tolist') else list(results.fitted_values)

    # 1. Residuals vs Fitted
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker=dict(color="steelblue", size=5, opacity=0.6),
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # 2. Q-Q Plot
    sorted_residuals = np.sort(results.residuals)
    n = len(results.residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    # Convert to lists for JSON serialization
    theoretical_list = theoretical_quantiles.tolist()
    sorted_list = sorted_residuals.tolist()

    fig.add_trace(
        go.Scatter(
            x=theoretical_list,
            y=sorted_list,
            mode="markers",
            marker=dict(color="steelblue", size=5),
            name="Q-Q",
        ),
        row=1,
        col=2,
    )
    # Add reference line
    min_val = float(min(theoretical_quantiles.min(), sorted_residuals.min()))
    max_val = float(max(theoretical_quantiles.max(), sorted_residuals.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Reference",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. ACF Plot
    acf = results.acf_values.tolist() if hasattr(results.acf_values, 'tolist') else list(results.acf_values)
    lags = list(range(len(acf)))
    # Confidence bounds (approximate 95% CI)
    conf_bound = float(1.96 / np.sqrt(len(results.residuals)))

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf,
            marker_color="steelblue",
            name="ACF",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=2, col=1)

    # 4. Residuals over time
    time_x = list(range(len(residuals)))
    fig.add_trace(
        go.Scatter(
            x=time_x,
            y=residuals,
            mode="lines",
            line=dict(color="steelblue"),
            name="Residuals",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(
        title=title,
        showlegend=False,
        height=600,
        width=900,
    )

    # Update axis labels
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=1)
    fig.update_xaxes(title_text="Observation", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)

    return fig


def create_acf_chart(
    acf_values: np.ndarray,
    pacf_values: np.ndarray | None = None,
    n_obs: int = 100,
    title: str = "Autocorrelation",
) -> go.Figure:
    """
    Create ACF and optionally PACF chart.

    Parameters
    ----------
    acf_values : np.ndarray
        ACF values.
    pacf_values : np.ndarray, optional
        PACF values. If provided, creates side-by-side plot.
    n_obs : int
        Number of observations (for confidence bounds).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if pacf_values is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
        cols = 2
    else:
        fig = go.Figure()
        cols = 1

    # Convert to lists for JSON serialization
    lags = list(range(len(acf_values)))
    acf_list = acf_values.tolist() if hasattr(acf_values, 'tolist') else list(acf_values)
    conf_bound = float(1.96 / np.sqrt(n_obs))

    # ACF
    if cols == 2:
        fig.add_trace(
            go.Bar(x=lags, y=acf_list, marker_color="steelblue", name="ACF"),
            row=1,
            col=1,
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=1)
    else:
        fig.add_trace(
            go.Bar(x=lags, y=acf_list, marker_color="steelblue", name="ACF")
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red")
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red")

    # PACF
    if pacf_values is not None:
        pacf_lags = list(range(len(pacf_values)))
        pacf_list = pacf_values.tolist() if hasattr(pacf_values, 'tolist') else list(pacf_values)
        fig.add_trace(
            go.Bar(x=pacf_lags, y=pacf_list, marker_color="indianred", name="PACF"),
            row=1,
            col=2,
        )
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(
        title=title,
        showlegend=False,
        height=400,
        width=800 if cols == 2 else 500,
    )

    return fig


def create_qq_plot(
    residuals: np.ndarray,
    title: str = "Q-Q Plot",
) -> go.Figure:
    """
    Create Q-Q plot for normality assessment.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    sorted_residuals = np.sort(residuals)
    n = len(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

    # Convert to lists for JSON serialization
    theoretical_list = theoretical_quantiles.tolist()
    sorted_list = sorted_residuals.tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical_list,
            y=sorted_list,
            mode="markers",
            marker=dict(color="steelblue", size=6),
            name="Residuals",
        )
    )

    # Reference line
    min_val = float(min(theoretical_quantiles.min(), sorted_residuals.min()))
    max_val = float(max(theoretical_quantiles.max(), sorted_residuals.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Normal Reference",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400,
        width=500,
    )

    return fig


def create_residual_vs_fitted(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    title: str = "Residuals vs Fitted",
) -> go.Figure:
    """
    Create residuals vs fitted values plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    fitted_values : np.ndarray
        Fitted values.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    # Convert to lists for JSON serialization
    fitted_list = fitted_values.tolist() if hasattr(fitted_values, 'tolist') else list(fitted_values)
    residuals_list = residuals.tolist() if hasattr(residuals, 'tolist') else list(residuals)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fitted_list,
            y=residuals_list,
            mode="markers",
            marker=dict(color="steelblue", size=6, opacity=0.6),
            name="Residuals",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    # Add smoothed trend (LOWESS-style using moving average)
    sorted_indices = np.argsort(fitted_values)
    sorted_fitted = fitted_values[sorted_indices]
    sorted_residuals = residuals[sorted_indices]

    # Simple moving average for trend
    window = max(len(residuals) // 10, 5)
    if len(residuals) > window:
        smoothed = np.convolve(sorted_residuals, np.ones(window) / window, mode="valid")
        smoothed_x = sorted_fitted[window // 2 : -(window - window // 2) + 1]
        if len(smoothed_x) == len(smoothed):
            # Convert to lists for JSON serialization
            smoothed_x_list = smoothed_x.tolist() if hasattr(smoothed_x, 'tolist') else list(smoothed_x)
            smoothed_list = smoothed.tolist() if hasattr(smoothed, 'tolist') else list(smoothed)
            fig.add_trace(
                go.Scatter(
                    x=smoothed_x_list,
                    y=smoothed_list,
                    mode="lines",
                    line=dict(color="orange", width=2),
                    name="Trend",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        height=400,
        width=600,
    )

    return fig


def create_vif_chart(
    results: ChannelDiagnosticsResults,
    title: str = "Variance Inflation Factors",
) -> go.Figure:
    """
    Create VIF bar chart for multicollinearity assessment.

    Parameters
    ----------
    results : ChannelDiagnosticsResults
        Channel diagnostics results.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    channels = list(results.vif_scores.keys())
    # Convert to Python floats for JSON serialization
    vif_values = [float(v) for v in results.vif_scores.values()]

    # Color based on VIF threshold
    colors = [
        "red" if v > 10 else "orange" if v > 5 else "steelblue" for v in vif_values
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=channels,
            y=vif_values,
            marker_color=colors,
            name="VIF",
        )
    )

    # Add threshold lines
    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="High (10)")
    fig.add_hline(
        y=5, line_dash="dash", line_color="orange", annotation_text="Moderate (5)"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Channel",
        yaxis_title="VIF",
        height=400,
        width=max(400, len(channels) * 60),
    )

    return fig


def create_ppc_density_plot(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    n_samples: int = 50,
    title: str = "Posterior Predictive Check: Density Overlay",
) -> go.Figure:
    """
    Create PPC density overlay plot.

    Shows observed data density overlaid with replicated data densities.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed data.
    y_rep : np.ndarray
        Replicated data, shape (n_samples, n_obs).
    n_samples : int
        Number of replicated samples to plot.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    fig = go.Figure()

    # Subsample replicated data if too many
    if y_rep.shape[0] > n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(y_rep.shape[0], size=n_samples, replace=False)
        y_rep_subset = y_rep[indices]
    else:
        y_rep_subset = y_rep

    # Convert observed data to list for JSON serialization
    y_obs_list = y_obs.tolist() if hasattr(y_obs, 'tolist') else list(y_obs)

    # Plot replicated densities (light gray)
    for i in range(y_rep_subset.shape[0]):
        # Convert each row to list for JSON serialization
        y_rep_row = y_rep_subset[i].tolist() if hasattr(y_rep_subset[i], 'tolist') else list(y_rep_subset[i])
        fig.add_trace(
            go.Histogram(
                x=y_rep_row,
                histnorm="probability density",
                opacity=0.1,
                marker_color="gray",
                showlegend=i == 0,
                name="Replicated" if i == 0 else None,
                nbinsx=30,
            )
        )

    # Plot observed density (bold)
    fig.add_trace(
        go.Histogram(
            x=y_obs_list,
            histnorm="probability density",
            opacity=0.8,
            marker_color="steelblue",
            name="Observed",
            nbinsx=30,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Density",
        barmode="overlay",
        height=400,
        width=700,
        showlegend=True,
    )

    return fig


def create_ppc_statistics_plot(
    checks: list,
    title: str = "Posterior Predictive Check: Test Statistics",
) -> go.Figure:
    """
    Create PPC test statistics comparison plot.

    Shows observed vs replicated statistics with error bars.

    Parameters
    ----------
    checks : list
        List of PPCCheckResult objects.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    check_names = [c.check_name for c in checks]
    # Convert to Python floats for JSON serialization
    observed = [float(c.observed_statistic) for c in checks]
    replicated_mean = [float(c.replicated_mean) for c in checks]
    replicated_std = [float(c.replicated_std) for c in checks]
    passed = [c.passed for c in checks]

    # Colors based on pass/fail
    colors = ["green" if p else "red" for p in passed]

    fig = go.Figure()

    # Replicated means with error bars
    fig.add_trace(
        go.Bar(
            x=check_names,
            y=replicated_mean,
            error_y=dict(type="data", array=[2 * s for s in replicated_std]),
            name="Replicated (±2σ)",
            marker_color="lightgray",
        )
    )

    # Observed values as points
    fig.add_trace(
        go.Scatter(
            x=check_names,
            y=observed,
            mode="markers",
            marker=dict(size=12, color=colors, symbol="diamond"),
            name="Observed",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Check",
        yaxis_title="Statistic Value",
        height=400,
        width=600,
        showlegend=True,
    )

    return fig


def create_ppc_time_series_plot(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    title: str = "Posterior Predictive Check: Time Series",
) -> go.Figure:
    """
    Create PPC time series plot with uncertainty bands.

    Parameters
    ----------
    y_obs : np.ndarray
        Observed data.
    y_rep : np.ndarray
        Replicated data, shape (n_samples, n_obs).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    n_obs = len(y_obs)
    x = list(range(n_obs))

    # Compute percentiles and convert to lists for JSON serialization
    rep_median = np.median(y_rep, axis=0).tolist()
    rep_lower = np.percentile(y_rep, 2.5, axis=0).tolist()
    rep_upper = np.percentile(y_rep, 97.5, axis=0).tolist()
    rep_lower_50 = np.percentile(y_rep, 25, axis=0).tolist()
    rep_upper_50 = np.percentile(y_rep, 75, axis=0).tolist()
    y_obs_list = y_obs.tolist() if hasattr(y_obs, 'tolist') else list(y_obs)

    fig = go.Figure()

    # 95% CI band
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=rep_upper + rep_lower[::-1],
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

    # 50% CI band
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=rep_upper_50 + rep_lower_50[::-1],
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.4)",
            line=dict(color="rgba(255,255,255,0)"),
            name="50% CI",
            showlegend=True,
        )
    )

    # Replicated median
    fig.add_trace(
        go.Scatter(
            x=x,
            y=rep_median,
            mode="lines",
            line=dict(color="steelblue", width=2),
            name="Replicated Median",
        )
    )

    # Observed data
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_obs_list,
            mode="lines",
            line=dict(color="black", width=2),
            name="Observed",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title="Value",
        height=400,
        width=900,
        showlegend=True,
    )

    return fig


def create_residual_time_series_plot(
    residuals: np.ndarray,
    title: str = "Residuals Over Time",
) -> go.Figure:
    """
    Create residual time series plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    n_obs = len(residuals)
    x = list(range(n_obs))

    # Convert to list for JSON serialization
    residuals_list = residuals.tolist() if hasattr(residuals, 'tolist') else list(residuals)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=residuals_list,
            mode="lines+markers",
            line=dict(color="steelblue", width=1),
            marker=dict(size=4),
            name="Residuals",
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    # Add ±2σ bands
    std = float(np.std(residuals))
    fig.add_hline(y=2 * std, line_dash="dot", line_color="orange")
    fig.add_hline(y=-2 * std, line_dash="dot", line_color="orange")

    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title="Residual",
        height=350,
        width=800,
    )

    return fig


__all__ = [
    "create_residual_panel",
    "create_acf_chart",
    "create_qq_plot",
    "create_residual_vs_fitted",
    "create_vif_chart",
    "create_ppc_density_plot",
    "create_ppc_statistics_plot",
    "create_ppc_time_series_plot",
    "create_residual_time_series_plot",
]

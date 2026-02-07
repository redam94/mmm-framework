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
    residuals = (
        results.residuals.tolist()
        if hasattr(results.residuals, "tolist")
        else list(results.residuals)
    )
    fitted = (
        results.fitted_values.tolist()
        if hasattr(results.fitted_values, "tolist")
        else list(results.fitted_values)
    )

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
    acf = (
        results.acf_values.tolist()
        if hasattr(results.acf_values, "tolist")
        else list(results.acf_values)
    )
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
        autosize=True,
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
    acf_list = (
        acf_values.tolist() if hasattr(acf_values, "tolist") else list(acf_values)
    )
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
        fig.add_trace(go.Bar(x=lags, y=acf_list, marker_color="steelblue", name="ACF"))
        fig.add_hline(y=conf_bound, line_dash="dash", line_color="red")
        fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red")

    # PACF
    if pacf_values is not None:
        pacf_lags = list(range(len(pacf_values)))
        pacf_list = (
            pacf_values.tolist()
            if hasattr(pacf_values, "tolist")
            else list(pacf_values)
        )
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
        autosize=True,
    )

    return fig


def create_qq_plot(
    residuals: np.ndarray,
    title: str = "Q-Q Plot",
) -> go.Figure:
    """
    Create Q-Q plot for normality assessment.

    Residuals are standardized to z-scores before comparison with
    the theoretical standard normal distribution.

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
    # Standardize residuals to z-scores
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals, ddof=1)  # Use sample std
    if residuals_std < 1e-10:
        residuals_std = 1.0  # Avoid division by zero
    standardized_residuals = (residuals - residuals_mean) / residuals_std

    # Sort standardized residuals
    sorted_residuals = np.sort(standardized_residuals)
    n = len(residuals)

    # Theoretical quantiles from standard normal
    # Use probability plotting positions (Filliben's formula)
    p = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)
    theoretical_quantiles = stats.norm.ppf(p)

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
            name="Standardized Residuals",
        )
    )

    # Reference line (y = x for standardized data)
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
        xaxis_title="Theoretical Quantiles (Standard Normal)",
        yaxis_title="Sample Quantiles (Standardized Residuals)",
        height=400,
        autosize=True,
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
    fitted_list = (
        fitted_values.tolist()
        if hasattr(fitted_values, "tolist")
        else list(fitted_values)
    )
    residuals_list = (
        residuals.tolist() if hasattr(residuals, "tolist") else list(residuals)
    )

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
            smoothed_x_list = (
                smoothed_x.tolist()
                if hasattr(smoothed_x, "tolist")
                else list(smoothed_x)
            )
            smoothed_list = (
                smoothed.tolist() if hasattr(smoothed, "tolist") else list(smoothed)
            )
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
        autosize=True,
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
        autosize=True,
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
    y_obs_list = y_obs.tolist() if hasattr(y_obs, "tolist") else list(y_obs)

    # Plot replicated densities (light gray)
    for i in range(y_rep_subset.shape[0]):
        # Convert each row to list for JSON serialization
        y_rep_row = (
            y_rep_subset[i].tolist()
            if hasattr(y_rep_subset[i], "tolist")
            else list(y_rep_subset[i])
        )
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
        autosize=True,
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
        autosize=True,
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
    y_obs_list = y_obs.tolist() if hasattr(y_obs, "tolist") else list(y_obs)

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
        autosize=True,
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
    residuals_list = (
        residuals.tolist() if hasattr(residuals, "tolist") else list(residuals)
    )

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
        autosize=True,
    )

    return fig


def create_pit_histogram(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    n_bins: int = 10,
    title: str = "Probability Integral Transform (PIT) Histogram",
) -> go.Figure:
    """
    Create PIT histogram for calibration assessment.

    The Probability Integral Transform tests whether the posterior predictive
    distribution is well-calibrated. For each observation, we compute the
    proportion of posterior predictive samples that fall below the observed value
    (the empirical CDF). If the model is well-calibrated, these PIT values
    should follow a Uniform(0,1) distribution.

    Interpretation:
    - Uniform histogram: Well-calibrated model
    - U-shaped: Underdispersed (overconfident) predictions
    - Inverse U-shaped: Overdispersed (underconfident) predictions
    - Left-skewed: Model systematically overpredicts
    - Right-skewed: Model systematically underpredicts

    Parameters
    ----------
    y_obs : np.ndarray
        Observed data, shape (n_obs,).
    y_rep : np.ndarray
        Replicated/predicted data from posterior predictive, shape (n_samples, n_obs).
    n_bins : int
        Number of histogram bins. Default is 10.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with PIT histogram and uniform reference.

    Examples
    --------
    >>> pit_fig = create_pit_histogram(ppc_results.y_obs, ppc_results.y_rep)
    >>> pit_fig.show()
    """
    # Compute PIT values: proportion of samples below observed value
    # For each observation i: PIT_i = (1/S) * sum(y_rep[:, i] <= y_obs[i])
    n_samples = y_rep.shape[0]
    pit_values = np.mean(y_rep <= y_obs[np.newaxis, :], axis=0)

    # Create histogram
    fig = go.Figure()

    # Add histogram of PIT values
    fig.add_trace(
        go.Histogram(
            x=pit_values.tolist(),
            nbinsx=n_bins,
            marker_color="steelblue",
            opacity=0.7,
            name="PIT values",
            histnorm="probability",
        )
    )

    # Add uniform reference line (expected height for uniform distribution)
    expected_height = 1.0 / n_bins
    fig.add_hline(
        y=expected_height,
        line_dash="dash",
        line_color="red",
        annotation_text="Uniform reference",
        annotation_position="top right",
    )

    # Add confidence band for uniform distribution (approximate 95% CI)
    # Using normal approximation to binomial
    n_obs = len(y_obs)
    se = np.sqrt(expected_height * (1 - expected_height) / n_obs)
    ci_upper = expected_height + 1.96 * se
    ci_lower = expected_height - 1.96 * se

    fig.add_hrect(
        y0=ci_lower,
        y1=ci_upper,
        fillcolor="lightgray",
        opacity=0.3,
        line_width=0,
        annotation_text="95% CI",
        annotation_position="top left",
    )

    # Compute diagnostic statistics
    # Kolmogorov-Smirnov test against uniform
    ks_stat, ks_pvalue = stats.kstest(pit_values, "uniform")

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>KS test: D={ks_stat:.3f}, p={ks_pvalue:.3f}</sub>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="PIT Value (Empirical CDF)",
        yaxis_title="Proportion",
        height=400,
        autosize=True,
        showlegend=False,
        bargap=0.05,
    )

    # Set x-axis to [0, 1]
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])

    return fig


def create_pit_ecdf(
    y_obs: np.ndarray,
    y_rep: np.ndarray,
    title: str = "PIT Empirical CDF vs Uniform",
) -> go.Figure:
    """
    Create PIT ECDF plot comparing to uniform distribution.

    An alternative visualization to the histogram that shows the empirical
    cumulative distribution function of PIT values compared to the theoretical
    uniform CDF (45-degree line).

    Parameters
    ----------
    y_obs : np.ndarray
        Observed data, shape (n_obs,).
    y_rep : np.ndarray
        Replicated/predicted data from posterior predictive, shape (n_samples, n_obs).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with ECDF and uniform reference.
    """
    # Compute PIT values
    pit_values = np.mean(y_rep <= y_obs[np.newaxis, :], axis=0)

    # Sort for ECDF
    pit_sorted = np.sort(pit_values)
    n = len(pit_sorted)
    ecdf_y = np.arange(1, n + 1) / n

    # Convert to lists for JSON serialization
    pit_list = pit_sorted.tolist()
    ecdf_list = ecdf_y.tolist()

    fig = go.Figure()

    # Add ECDF of PIT values
    fig.add_trace(
        go.Scatter(
            x=pit_list,
            y=ecdf_list,
            mode="lines",
            line=dict(color="steelblue", width=2),
            name="PIT ECDF",
        )
    )

    # Add uniform reference (45-degree line)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="Uniform CDF",
        )
    )

    # Add confidence band (Kolmogorov-Smirnov critical value)
    # Approximate 95% confidence band
    ks_critical = 1.36 / np.sqrt(n)  # 95% critical value
    uniform_x = np.linspace(0, 1, 100)

    fig.add_trace(
        go.Scatter(
            x=uniform_x.tolist() + uniform_x[::-1].tolist(),
            y=(uniform_x + ks_critical).tolist()
            + (uniform_x[::-1] - ks_critical).tolist(),
            fill="toself",
            fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=True,
        )
    )

    # Compute KS statistic
    ks_stat, ks_pvalue = stats.kstest(pit_values, "uniform")

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>KS test: D={ks_stat:.3f}, p={ks_pvalue:.3f}</sub>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="PIT Value",
        yaxis_title="Cumulative Probability",
        height=400,
        autosize=True,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    return fig


def create_cv_fold_metrics_chart(
    fold_results: list,
    mean_rmse: float,
    mean_mae: float,
    mean_r2: float,
    title: str = "Cross-Validation Metrics by Fold",
) -> go.Figure:
    """
    Create grouped bar chart showing CV metrics per fold.

    Displays RMSE, MAE, and R² for each cross-validation fold with
    horizontal reference lines showing the mean values.

    Parameters
    ----------
    fold_results : list
        List of CVFoldResult objects with fold-level metrics.
    mean_rmse : float
        Mean RMSE across all folds.
    mean_mae : float
        Mean MAE across all folds.
    mean_r2 : float
        Mean R² across all folds.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with grouped bars and subplots.
    """
    fold_indices = [f"Fold {f.fold_idx + 1}" for f in fold_results]
    rmse_values = [f.rmse for f in fold_results]
    mae_values = [f.mae for f in fold_results]
    r2_values = [f.r2 for f in fold_results]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Error Metrics (RMSE, MAE)", "R² Score by Fold"),
        horizontal_spacing=0.12,
    )

    # RMSE bars
    fig.add_trace(
        go.Bar(
            name="RMSE",
            x=fold_indices,
            y=rmse_values,
            marker_color="#c97067",
            text=[f"{v:.4f}" for v in rmse_values],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # MAE bars
    fig.add_trace(
        go.Bar(
            name="MAE",
            x=fold_indices,
            y=mae_values,
            marker_color="#d4a86a",
            text=[f"{v:.4f}" for v in mae_values],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Mean RMSE line
    fig.add_hline(
        y=mean_rmse,
        line_dash="dash",
        line_color="#c97067",
        annotation_text=f"Mean RMSE: {mean_rmse:.4f}",
        annotation_position="top right",
        row=1,
        col=1,
    )

    # Mean MAE line
    fig.add_hline(
        y=mean_mae,
        line_dash="dot",
        line_color="#d4a86a",
        annotation_text=f"Mean MAE: {mean_mae:.4f}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # R² bars
    fig.add_trace(
        go.Bar(
            name="R²",
            x=fold_indices,
            y=r2_values,
            marker_color="#6abf8a",
            text=[f"{v:.4f}" for v in r2_values],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    # Mean R² line
    fig.add_hline(
        y=mean_r2,
        line_dash="dash",
        line_color="#6abf8a",
        annotation_text=f"Mean R²: {mean_r2:.4f}",
        annotation_position="top right",
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        barmode="group",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # Set y-axis for R² to reasonable range
    fig.update_yaxes(range=[0, max(1.0, max(r2_values) * 1.1)], row=1, col=2)

    return fig


def create_cv_coverage_chart(
    fold_results: list,
    mean_coverage: float,
    target_coverage: float = 0.80,
    title: str = "Credible Interval Coverage by Fold",
) -> go.Figure:
    """
    Create bar chart showing CI coverage per fold with target line.

    Coverage indicates the proportion of observations falling within
    the credible interval. Well-calibrated models should have coverage
    close to the nominal credible interval level.

    Parameters
    ----------
    fold_results : list
        List of CVFoldResult objects with coverage data.
    mean_coverage : float
        Mean coverage across all folds.
    target_coverage : float
        Target coverage level (default 0.80 for 80% CI).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with coverage bars.
    """
    fold_indices = [f"Fold {f.fold_idx + 1}" for f in fold_results]
    coverage_values = [f.coverage for f in fold_results]

    # Color based on coverage quality
    colors = []
    for cov in coverage_values:
        if cov >= 0.85:
            colors.append("#6abf8a")  # Good (green)
        elif cov >= 0.70:
            colors.append("#d4a86a")  # Acceptable (orange)
        else:
            colors.append("#c97067")  # Poor (red)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=fold_indices,
            y=[c * 100 for c in coverage_values],
            marker_color=colors,
            text=[f"{c:.1%}" for c in coverage_values],
            textposition="outside",
            name="Coverage",
        )
    )

    # Target coverage line
    fig.add_hline(
        y=target_coverage * 100,
        line_dash="dash",
        line_color="steelblue",
        annotation_text=f"Target: {target_coverage:.0%}",
        annotation_position="top right",
    )

    # Mean coverage line
    fig.add_hline(
        y=mean_coverage * 100,
        line_dash="dot",
        line_color="#555555",
        annotation_text=f"Mean: {mean_coverage:.1%}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Fold",
        yaxis_title="Coverage (%)",
        height=400,
        yaxis=dict(range=[0, 105]),
        showlegend=False,
    )

    return fig


def create_cv_actual_vs_predicted_chart(
    fold_results: list,
    full_y_actual: np.ndarray,
    title: str = "Cross-Validation: Actual vs Predicted",
) -> go.Figure:
    """
    Create time-series plot showing actual vs predicted for each CV fold.

    Shows the full actual series as a baseline with per-fold out-of-sample
    predictions overlaid, including credible intervals.

    Parameters
    ----------
    fold_results : list
        List of CVFoldResult objects with prediction data.
    full_y_actual : np.ndarray
        Full observed time series (for context).
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly figure with actual vs predicted overlay.
    """
    fig = go.Figure()

    # Plot full actual series (gray baseline)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(full_y_actual))),
            y=full_y_actual.tolist(),
            mode="lines",
            name="Actual (Full Series)",
            line=dict(color="gray", width=1.5),
        )
    )

    # Color palette for folds
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot each fold's test predictions
    for i, fold in enumerate(fold_results):
        if fold.test_indices is None or fold.y_pred_mean is None:
            continue

        color = colors[i % len(colors)]
        x = fold.test_indices.tolist()

        # Parse color to RGB for transparency
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # CI band (94% credible interval)
        if fold.y_pred_ci_low is not None and fold.y_pred_ci_high is not None:
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=fold.y_pred_ci_high.tolist() + fold.y_pred_ci_low.tolist()[::-1],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"Fold {fold.fold_idx + 1} 94% CI",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Predicted mean line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fold.y_pred_mean.tolist(),
                mode="lines",
                name=f"Fold {fold.fold_idx + 1} Predicted",
                line=dict(color=color, width=2),
            )
        )

        # Actual test values (markers)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fold.y_true.tolist(),
                mode="markers",
                name=f"Fold {fold.fold_idx + 1} Actual",
                marker=dict(color=color, size=8, symbol="circle"),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Time Index",
        yaxis_title="Value (Original Scale)",
        height=500,
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
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
    "create_pit_histogram",
    "create_pit_ecdf",
    "create_cv_fold_metrics_chart",
    "create_cv_coverage_chart",
    "create_cv_actual_vs_predicted_chart",
]

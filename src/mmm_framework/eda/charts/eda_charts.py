"""
Plotly figure builders for pre-fit data quality (EDA, validation, outliers).

Every function returns a ``plotly.graph_objects.Figure`` (NOT an HTML div —
the agent pipeline serializes figures to JSON and the frontend renders them).
Panel (geo/product) data is aggregated to the period level for display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..loading import EDAPanel
from ..results import DecompositionResult, OutlierFlag


def _period_frame(panel: EDAPanel, variables: list[str]) -> pd.DataFrame:
    """Variables as a Period-indexed frame (panel slices summed for display)."""
    cols = [v for v in variables if v in panel.df_wide.columns]
    frame = panel.df_wide[cols].astype(float)
    if panel.dims:
        frame = frame.groupby(level=panel.date_col).sum(min_count=1)
    return frame


def _grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(n / cols))
    return rows, cols


# ---------------------------------------------------------------------------
# profiling / EDA
# ---------------------------------------------------------------------------


def fig_sparkline_grid(
    panel: EDAPanel, variables: list[str] | None = None
) -> go.Figure:
    """Small-multiple timeseries, one panel per variable."""
    variables = variables or panel.variables
    frame = _period_frame(panel, variables)
    rows, cols = _grid(len(frame.columns))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(frame.columns),
        vertical_spacing=0.12,
    )
    for i, var in enumerate(frame.columns):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[var],
                mode="lines",
                name=var,
                showlegend=False,
                line=dict(width=1.5),
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        title="Variables over time",
        height=max(300, 220 * rows),
    )
    return fig


def fig_distributions(panel: EDAPanel, variables: list[str] | None = None) -> go.Figure:
    """Histogram small-multiples with skew/kurtosis annotations."""
    from scipy import stats as sps

    variables = variables or panel.variables
    frame = _period_frame(panel, variables)
    rows, cols = _grid(len(frame.columns))
    titles = []
    for var in frame.columns:
        v = frame[var].dropna().to_numpy()
        if v.size > 3:
            titles.append(f"{var} (skew {sps.skew(v):.1f}, kurt {sps.kurtosis(v):.1f})")
        else:
            titles.append(var)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.12
    )
    for i, var in enumerate(frame.columns):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Histogram(x=frame[var].dropna(), name=var, showlegend=False, nbinsx=30),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(title="Value distributions", height=max(300, 220 * rows))
    return fig


def fig_missingness(matrix: pd.DataFrame) -> go.Figure:
    """Availability heatmap: variables x periods (1 observed, 0 missing)."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix.T.to_numpy(),
            x=[str(pd.Timestamp(i).date()) for i in matrix.index],
            y=list(matrix.columns),
            colorscale=[[0, "#dc2626"], [1, "#16a34a"]],
            zmin=0,
            zmax=1,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Data availability (green = observed, red = missing)",
        height=max(260, 36 * len(matrix.columns) + 140),
    )
    return fig


def fig_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=np.round(corr.to_numpy(), 2),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.to_numpy(), 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Correlation matrix", height=max(360, 40 * len(corr) + 160))
    return fig


def fig_vif(vif: dict[str, float], threshold: float = 10.0) -> go.Figure:
    names = list(vif.keys())
    values = [min(v, 50.0) if np.isfinite(v) else 50.0 for v in vif.values()]
    colors = [
        (
            "#dc2626"
            if (np.isfinite(v) and v > threshold) or not np.isfinite(v)
            else "#f59e0b" if v > threshold / 2 else "#16a34a"
        )
        for v in vif.values()
    ]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"VIF {threshold:g}",
    )
    fig.update_layout(
        title="Variance Inflation Factors (capped at 50)",
        yaxis_title="VIF",
        height=380,
    )
    return fig


def fig_spend_share(
    share_over_time: pd.DataFrame,
    shares: dict[str, float],
    hhi: float | None,
) -> go.Figure:
    """Stacked spend-share area over time, with totals + HHI in the title."""
    fig = go.Figure()
    for ch in share_over_time.columns:
        fig.add_trace(
            go.Scatter(
                x=share_over_time.index,
                y=share_over_time[ch],
                stackgroup="share",
                name=f"{ch} ({shares.get(ch, 0):.0%})",
                mode="lines",
                line=dict(width=0.5),
            )
        )
    hhi_txt = f" — HHI {hhi:.2f}" if hhi is not None else ""
    fig.update_layout(
        title=f"Media spend share over time{hhi_txt}",
        yaxis=dict(title="Share", tickformat=".0%", range=[0, 1]),
        height=420,
    )
    return fig


def fig_decomposition(result: DecompositionResult) -> go.Figure:
    """Observed / trend / seasonal / residual panel for one series."""
    parts: list[tuple[str, pd.Series]] = [
        ("Trend", result.trend),
    ]
    if result.seasonal is not None:
        parts.append(("Seasonal", result.seasonal))
    parts.append(("Residual", result.resid))
    observed = (
        result.trend
        + result.resid
        + (result.seasonal if result.seasonal is not None else 0)
    )
    rows = 1 + len(parts)
    titles = ["Observed"] + [p[0] for p in parts]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed,
            mode="lines",
            name="Observed",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for i, (label, series) in enumerate(parts):
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series, mode="lines", name=label, showlegend=False
            ),
            row=i + 2,
            col=1,
        )
    strength = f"trend strength {result.trend_strength:.2f}"
    if result.seasonal_strength is not None:
        strength += f", seasonal strength {result.seasonal_strength:.2f}"
    fig.update_layout(
        title=f"{result.variable} — {result.method.upper()} decomposition ({strength})",
        height=160 * rows + 120,
    )
    return fig


def fig_kpi_vs_media(panel: EDAPanel) -> go.Figure:
    """Per-channel scatter of KPI vs spend with a binned-mean overlay."""
    if not panel.kpi or not panel.media:
        return go.Figure().update_layout(title="KPI vs media (roles unknown)")
    frame = _period_frame(panel, [panel.kpi, *panel.media]).dropna()
    rows, cols = _grid(len(panel.media))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=list(panel.media), vertical_spacing=0.14
    )
    for i, ch in enumerate(panel.media):
        r, c = divmod(i, cols)
        x, y = frame[ch], frame[panel.kpi]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=ch,
                showlegend=False,
                marker=dict(size=5, opacity=0.55),
            ),
            row=r + 1,
            col=c + 1,
        )
        # Binned means stand in for a smoother (no sklearn/LOWESS dependency).
        if x.nunique() > 8:
            bins = pd.qcut(x, q=min(8, x.nunique() // 2), duplicates="drop")
            means = frame.groupby(bins, observed=True).agg(
                {ch: "mean", panel.kpi: "mean"}
            )
            fig.add_trace(
                go.Scatter(
                    x=means[ch],
                    y=means[panel.kpi],
                    mode="lines+markers",
                    name=f"{ch} binned mean",
                    showlegend=False,
                    line=dict(width=2.5, color="#dc2626"),
                ),
                row=r + 1,
                col=c + 1,
            )
    fig.update_layout(
        title=f"{panel.kpi} vs media spend (red: binned means)",
        height=max(320, 260 * rows),
    )
    return fig


def fig_stationarity(results: dict[str, dict], significance: float = 0.05) -> go.Figure:
    """ADF p-values per variable (low = stationary evidence) with verdicts."""
    names, adf_p, verdicts = [], [], []
    for var, res in results.items():
        names.append(var)
        adf_p.append(
            res.get("adf_pvalue") if res.get("adf_pvalue") is not None else 1.0
        )
        verdicts.append(res.get("verdict", "?"))
    colors = [
        (
            "#16a34a"
            if v == "stationary"
            else "#f59e0b" if v == "trend_stationary" else "#dc2626"
        )
        for v in verdicts
    ]
    fig = go.Figure(
        go.Bar(
            x=names, y=adf_p, marker_color=colors, text=verdicts, textposition="outside"
        )
    )
    fig.add_hline(
        y=significance,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text=f"p = {significance:g}",
    )
    fig.update_layout(
        title="Stationarity (ADF p-value; bar label = ADF+KPSS verdict)",
        yaxis=dict(title="ADF p-value", range=[0, 1.1]),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# outliers
# ---------------------------------------------------------------------------


def fig_outlier_series(
    series: pd.Series,
    flags: list[OutlierFlag],
    expected: pd.Series | None = None,
    variable: str = "",
) -> go.Figure:
    """One series with flagged points marked and the expected baseline."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=variable or "observed",
            line=dict(width=1.5),
        )
    )
    if expected is not None:
        fig.add_trace(
            go.Scatter(
                x=expected.index,
                y=expected,
                mode="lines",
                name="expected",
                line=dict(width=1, dash="dot", color="#6b7280"),
            )
        )
    if flags:
        xs = [pd.Timestamp(f.period) for f in flags]
        ys = [f.value for f in flags]
        labels = [f"{f.kind} (score {f.score:.2f})" for f in flags]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="flagged",
                text=labels,
                marker=dict(size=10, symbol="x", color="#dc2626"),
            )
        )
    fig.update_layout(
        title=f"Outliers — {variable}" if variable else "Outliers",
        height=380,
    )
    return fig


def fig_outlier_severity(flags: list[OutlierFlag]) -> go.Figure:
    """Severity heatmap: variables x flagged periods."""
    if not flags:
        return go.Figure().update_layout(title="No outliers detected")
    periods = sorted({f.period for f in flags})
    variables = sorted({f.variable for f in flags})
    z = np.zeros((len(variables), len(periods)))
    for f in flags:
        z[variables.index(f.variable), periods.index(f.period)] = max(
            z[variables.index(f.variable), periods.index(f.period)], f.score
        )
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=periods,
            y=variables,
            colorscale="Reds",
            zmin=0,
            zmax=1,
            colorbar=dict(title="severity"),
        )
    )
    fig.update_layout(
        title="Outlier severity by variable and period",
        height=max(280, 40 * len(variables) + 160),
    )
    return fig


def fig_before_after(
    before: pd.Series,
    after: pd.Series,
    variable: str = "",
) -> go.Figure:
    """Original vs treated series (winsorize/exclude comparison)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=before.index,
            y=before,
            mode="lines",
            name="before",
            line=dict(width=1.5, color="#9ca3af"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=after.index,
            y=after,
            mode="lines",
            name="after",
            line=dict(width=1.5, color="#2563eb"),
        )
    )
    fig.update_layout(
        title=f"Treatment effect — {variable}" if variable else "Treatment effect",
        height=380,
    )
    return fig


__all__ = [
    "fig_sparkline_grid",
    "fig_distributions",
    "fig_missingness",
    "fig_correlation_heatmap",
    "fig_vif",
    "fig_spend_share",
    "fig_decomposition",
    "fig_kpi_vs_media",
    "fig_stationarity",
    "fig_outlier_series",
    "fig_outlier_severity",
    "fig_before_after",
]

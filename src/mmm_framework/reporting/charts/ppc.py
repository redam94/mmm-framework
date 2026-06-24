"""
Posterior-predictive check (PPC) charts for MMM goodness-of-fit.

These render the *predictive* distribution of the outcome against what was
actually observed -- the honest test of whether the fitted model can reproduce
the data it was trained on. Four complementary views:

- ``create_ppc_observed_vs_predicted``: observed vs posterior-predictive mean with
  per-observation predictive intervals and a 45° identity line.
- ``create_ppc_density_overlay``: the density of the observed KPI overlaid on an
  ensemble of replicated-dataset densities drawn from the posterior predictive
  (the classic ``plot_ppc`` view).
- ``create_ppc_interval_calibration``: empirical vs nominal coverage of the
  predictive intervals -- a well-calibrated model tracks the 45° line.
- ``create_ppc_residual_plot``: standardized-scale residuals vs fitted values with
  a zero reference and a ±2·σ guide band, to surface structure / heteroscedasticity.

All functions take already-computed arrays (no model access) and return an HTML
string via :func:`create_plotly_div`, matching the rest of ``reporting.charts``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ChartConfig, ReportConfig
from .base import _hex_to_rgb, create_plotly_div

# Cap on the number of replicate density traces in the overlay (the extractor
# down-samples upstream; this is a defensive belt-and-suspenders cap).
_MAX_OVERLAY_CURVES = 40


def _arr(x: Any) -> list:
    """Coerce array-like to a JSON-serializable Python list."""
    return np.asarray(x, dtype=float).tolist()


def create_ppc_observed_vs_predicted(
    observed: np.ndarray,
    pred_mean: np.ndarray,
    pred_lower: np.ndarray | None,
    pred_upper: np.ndarray | None,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "ppcObservedVsPredicted",
) -> str:
    """Observed vs posterior-predictive mean, with predictive-interval error bars.

    A perfectly fitting model would place every point on the dashed 45° line; the
    vertical bars show each observation's predictive interval so the reader can see
    whether the observed value plausibly falls within what the model predicts.
    """
    chart_config = chart_config or ChartConfig(height=420)
    colors = config.color_scheme

    obs = np.asarray(observed, dtype=float)
    mean = np.asarray(pred_mean, dtype=float)

    error_y = None
    if pred_lower is not None and pred_upper is not None:
        lo = np.asarray(pred_lower, dtype=float)
        hi = np.asarray(pred_upper, dtype=float)
        error_y = {
            "type": "data",
            "symmetric": False,
            "array": _arr(np.clip(hi - mean, 0, None)),
            "arrayminus": _arr(np.clip(mean - lo, 0, None)),
            "color": f"rgba({_hex_to_rgb(colors.accent)}, 0.35)",
            "thickness": 1,
            "width": 0,
        }

    lo_lim = float(min(obs.min(), mean.min()))
    hi_lim = float(max(obs.max(), mean.max()))
    pad = (hi_lim - lo_lim) * 0.05 or 1.0

    scatter = {
        "type": "scatter",
        "x": _arr(obs),
        "y": _arr(mean),
        "mode": "markers",
        "name": "Observations",
        "marker": {"color": colors.primary_dark, "size": 6, "opacity": 0.75},
        "hovertemplate": "Observed: %{x:,.0f}<br>Predicted: %{y:,.0f}<extra></extra>",
    }
    if error_y is not None:
        scatter["error_y"] = error_y

    identity = {
        "type": "scatter",
        "x": [lo_lim - pad, hi_lim + pad],
        "y": [lo_lim - pad, hi_lim + pad],
        "mode": "lines",
        "name": "Perfect fit (45°)",
        "line": {"color": colors.text_muted, "width": 1.5, "dash": "dash"},
        "hoverinfo": "skip",
    }

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Observed vs Posterior-Predicted",
        "font": {"size": 16},
    }
    layout["xaxis"] = {**layout.get("xaxis", {}), "title": "Observed"}
    layout["yaxis"] = {**layout.get("yaxis", {}), "title": "Predicted (posterior mean)"}
    layout["hovermode"] = "closest"

    return create_plotly_div([identity, scatter], layout, div_id)


def create_ppc_density_overlay(
    observed: np.ndarray,
    samples: np.ndarray | None,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "ppcDensityOverlay",
) -> str:
    """Density of observed KPI overlaid on replicated posterior-predictive datasets.

    Each faint curve is one dataset simulated from the posterior; if the bold
    observed density sits inside the cloud of replicate densities, the model
    reproduces the marginal distribution of the data.
    """
    chart_config = chart_config or ChartConfig(height=420)
    colors = config.color_scheme

    obs = np.asarray(observed, dtype=float)
    traces: list[dict] = []

    if samples is not None:
        reps = np.asarray(samples, dtype=float)
        if reps.ndim == 1:
            reps = reps[None, :]
        n_curves = min(reps.shape[0], _MAX_OVERLAY_CURVES)
        for i in range(n_curves):
            traces.append(
                {
                    "type": "histogram",
                    "x": _arr(reps[i]),
                    "histnorm": "probability density",
                    "opacity": 0.08,
                    "marker": {"color": colors.text_muted},
                    "name": "Replicated" if i == 0 else None,
                    "showlegend": i == 0,
                    "nbinsx": 30,
                    "hoverinfo": "skip",
                }
            )

    traces.append(
        {
            "type": "histogram",
            "x": _arr(obs),
            "histnorm": "probability density",
            "opacity": 0.85,
            "marker": {"color": colors.accent},
            "name": "Observed",
            "nbinsx": 30,
            "hovertemplate": "%{x:,.0f}<extra>Observed</extra>",
        }
    )

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Posterior-Predictive Density: Observed vs Replicated",
        "font": {"size": 16},
    }
    layout["barmode"] = "overlay"
    layout["xaxis"] = {**layout.get("xaxis", {}), "title": "KPI value"}
    layout["yaxis"] = {**layout.get("yaxis", {}), "title": "Density"}

    return create_plotly_div(traces, layout, div_id)


def create_ppc_interval_calibration(
    coverage: list[dict[str, float]],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "ppcCalibration",
) -> str:
    """Empirical vs nominal coverage of posterior-predictive intervals.

    ``coverage`` is a list of ``{"nominal": p, "empirical": q}`` points. A
    well-calibrated model tracks the dashed 45° line; points below it mean the
    intervals are too narrow (over-confident), above means too wide.
    """
    chart_config = chart_config or ChartConfig(height=380)
    colors = config.color_scheme

    pts = sorted(coverage, key=lambda d: d.get("nominal", 0.0))
    nominal = [float(d.get("nominal", 0.0)) for d in pts]
    empirical = [float(d.get("empirical", 0.0)) for d in pts]

    ideal = {
        "type": "scatter",
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "mode": "lines",
        "name": "Ideal",
        "line": {"color": colors.text_muted, "width": 1.5, "dash": "dash"},
        "hoverinfo": "skip",
    }
    curve = {
        "type": "scatter",
        "x": nominal,
        "y": empirical,
        "mode": "lines+markers",
        "name": "Observed coverage",
        "line": {"color": colors.primary, "width": 2},
        "marker": {"color": colors.primary, "size": 7},
        "hovertemplate": ("Nominal: %{x:.0%}<br>Empirical: %{y:.0%}<extra></extra>"),
    }

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Predictive-Interval Calibration",
        "font": {"size": 16},
    }
    layout["xaxis"] = {
        **layout.get("xaxis", {}),
        "title": "Nominal interval",
        "tickformat": ".0%",
        "range": [0, 1],
    }
    layout["yaxis"] = {
        **layout.get("yaxis", {}),
        "title": "Empirical coverage",
        "tickformat": ".0%",
        "range": [0, 1],
    }
    layout["hovermode"] = "closest"

    return create_plotly_div([ideal, curve], layout, div_id)


def create_ppc_residual_plot(
    observed: np.ndarray,
    pred_mean: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "ppcResiduals",
) -> str:
    """Residuals (observed − predicted mean) vs fitted, with a ±2σ guide band.

    Residuals should scatter structurelessly around zero. A funnel shape flags
    heteroscedasticity; a trend or clustering flags unmodelled structure.
    """
    chart_config = chart_config or ChartConfig(height=380)
    colors = config.color_scheme

    obs = np.asarray(observed, dtype=float)
    mean = np.asarray(pred_mean, dtype=float)
    resid = obs - mean
    sigma = float(np.std(resid)) if resid.size else 0.0

    lo_lim = float(mean.min())
    hi_lim = float(mean.max())
    pad = (hi_lim - lo_lim) * 0.05 or 1.0
    band_x = [lo_lim - pad, hi_lim + pad]

    band = {
        "type": "scatter",
        "x": band_x + band_x[::-1],
        "y": [2 * sigma, 2 * sigma, -2 * sigma, -2 * sigma],
        "fill": "toself",
        "fillcolor": f"rgba({_hex_to_rgb(colors.accent)}, 0.12)",
        "line": {"color": "transparent"},
        "name": "±2σ",
        "hoverinfo": "skip",
    }
    zero_line = {
        "type": "scatter",
        "x": band_x,
        "y": [0.0, 0.0],
        "mode": "lines",
        "name": "Zero",
        "line": {"color": colors.text_muted, "width": 1.5, "dash": "dash"},
        "hoverinfo": "skip",
    }
    points = {
        "type": "scatter",
        "x": _arr(mean),
        "y": _arr(resid),
        "mode": "markers",
        "name": "Residual",
        "marker": {"color": colors.primary_dark, "size": 6, "opacity": 0.7},
        "hovertemplate": "Fitted: %{x:,.0f}<br>Residual: %{y:,.0f}<extra></extra>",
    }

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {"text": "Residuals vs Fitted", "font": {"size": 16}}
    layout["xaxis"] = {**layout.get("xaxis", {}), "title": "Fitted (posterior mean)"}
    layout["yaxis"] = {**layout.get("yaxis", {}), "title": "Residual"}
    layout["hovermode"] = "closest"

    return create_plotly_div([band, zero_line, points], layout, div_id)


__all__ = [
    "create_ppc_observed_vs_predicted",
    "create_ppc_density_overlay",
    "create_ppc_interval_calibration",
    "create_ppc_residual_plot",
]

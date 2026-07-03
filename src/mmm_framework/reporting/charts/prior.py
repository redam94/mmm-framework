"""Pre-fit chart kit: prior densities, prior-predictive checks, and SBC.

The visual vocabulary of the Model Design Readout — everything a reader needs
to *see* what the priors imply before any data has spoken:

- ``create_prior_predictive_fan``: observed KPI over the prior-predictive
  quantile fan (the "could the model even generate data like ours?" view).
- ``create_prior_stat_distribution``: the distribution of a replicate statistic
  (mean / sd) against the observed value.
- ``create_prior_density_chart``: one parameter's prior density (small multiple).
- ``create_prior_saturation_band`` / ``create_prior_adstock_band``: the response
  shapes the priors imply per channel, median + credible band.
- ``create_sbc_rank_histogram`` / ``create_sbc_ecdf_diff``: Simulation-Based
  Calibration verdict charts, rendered natively in the reporting kit (the
  band math is reused from :mod:`mmm_framework.diagnostics.sbc`).

All functions take already-computed arrays / JSON-safe dicts (no model access)
and return an HTML string via :func:`create_plotly_div`, matching the rest of
``reporting.charts``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ChartConfig, ReportConfig
from .base import _hex_to_rgb, create_plotly_div

_FONT = {"family": "IBM Plex Sans, system-ui, sans-serif", "size": 12.5}
_GRID = "#efece0"
_LINE = "#e3dfd0"
_TICK = {"family": "JetBrains Mono, monospace", "size": 11, "color": "#7a8a78"}


def _arr(x: Any) -> list:
    return np.asarray(x, dtype=float).tolist()


def _base_layout(config: ReportConfig, height: int, extra: dict | None = None) -> dict:
    colors = config.color_scheme
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {**_FONT, "color": colors.text},
        "height": height,
        "margin": {"t": 30, "r": 18, "b": 44, "l": 58},
        "xaxis": {
            "gridcolor": _GRID,
            "zerolinecolor": _LINE,
            "linecolor": _LINE,
            "tickfont": _TICK,
        },
        "yaxis": {
            "gridcolor": _GRID,
            "zerolinecolor": _LINE,
            "linecolor": _LINE,
            "tickfont": _TICK,
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 12},
        },
    }
    for key, val in (extra or {}).items():
        if isinstance(val, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout


# ─────────────────────────────────────────────────────────────────────────────
# Prior predictive
# ─────────────────────────────────────────────────────────────────────────────
def create_prior_predictive_fan(
    dates: list[str],
    observed: np.ndarray,
    bands: dict[str, np.ndarray],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorPredictiveFan",
    kpi_label: str = "KPI",
    sample_traces: np.ndarray | None = None,
) -> str:
    """Observed KPI over the prior-predictive quantile fan (period axis).

    ``bands`` maps ``p05/p25/p50/p75/p95`` to per-period arrays. A healthy prior
    wraps the observed series loosely — neither excluding it nor allowing
    physically absurd magnitudes. ``sample_traces`` (K × periods) overlays a few
    individual simulated series so the reader sees single realizations, not
    just the envelope.
    """
    chart_config = chart_config or ChartConfig(height=380)
    colors = config.color_scheme
    accent_rgb = _hex_to_rgb(colors.accent)
    primary_rgb = _hex_to_rgb(colors.primary)

    traces: list[dict] = []
    if sample_traces is not None:
        st = np.asarray(sample_traces, dtype=float)
        for i, row in enumerate(st):
            traces.append(
                {
                    "type": "scatter",
                    "x": dates,
                    "y": _arr(row),
                    "mode": "lines",
                    "line": {"color": f"rgba({accent_rgb}, 0.28)", "width": 0.9},
                    "name": "Prior draws",
                    "legendgroup": "priordraws",
                    "showlegend": i == 0,
                    "hoverinfo": "skip",
                }
            )
    if "p05" in bands and "p95" in bands:
        traces.append(
            {
                "type": "scatter",
                "x": dates + dates[::-1],
                "y": _arr(bands["p95"]) + _arr(bands["p05"])[::-1],
                "fill": "toself",
                "fillcolor": f"rgba({accent_rgb}, 0.14)",
                "line": {"color": "rgba(255,255,255,0)"},
                "name": "Prior predictive 90%",
                "hoverinfo": "skip",
            }
        )
    if "p25" in bands and "p75" in bands:
        traces.append(
            {
                "type": "scatter",
                "x": dates + dates[::-1],
                "y": _arr(bands["p75"]) + _arr(bands["p25"])[::-1],
                "fill": "toself",
                "fillcolor": f"rgba({accent_rgb}, 0.22)",
                "line": {"color": "rgba(255,255,255,0)"},
                "name": "Prior predictive 50%",
                "hoverinfo": "skip",
            }
        )
    if "p50" in bands:
        traces.append(
            {
                "type": "scatter",
                "x": dates,
                "y": _arr(bands["p50"]),
                "mode": "lines",
                "line": {
                    "color": f"rgba({primary_rgb}, 0.9)",
                    "width": 1.4,
                    "dash": "dot",
                },
                "name": "Prior median",
                "hovertemplate": "Prior median %{y:,.0f}<extra></extra>",
            }
        )
    traces.append(
        {
            "type": "scatter",
            "x": dates,
            "y": _arr(observed),
            "mode": "lines",
            "line": {"color": "#2a3528", "width": 1.8},
            "name": f"Observed {kpi_label}",
            "hovertemplate": "Observed %{y:,.0f}<extra></extra>",
        }
    )

    layout = _base_layout(
        config,
        chart_config.height,
        {
            "hovermode": "x unified",
            "yaxis": {"title": {"text": kpi_label, "font": {"size": 11.5}}},
        },
    )
    return create_plotly_div(traces, layout, div_id)


def create_prior_stat_distribution(
    values: np.ndarray,
    observed: float,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorStatDist",
    stat_label: str = "replicate mean",
) -> str:
    """Distribution of one replicate statistic under the prior vs the observed value."""
    chart_config = chart_config or ChartConfig(height=280)
    colors = config.color_scheme
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    traces = [
        {
            "type": "histogram",
            "x": _arr(vals),
            "nbinsx": 40,
            "marker": {"color": f"rgba({_hex_to_rgb(colors.accent)}, 0.55)"},
            "name": f"Prior {stat_label}",
            "hovertemplate": "%{x:,.0f}: %{y}<extra></extra>",
        }
    ]
    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "bargap": 0.04,
            "xaxis": {"title": {"text": stat_label, "font": {"size": 11}}},
            "yaxis": {"title": {"text": "draws", "font": {"size": 11}}},
            "shapes": [
                {
                    "type": "line",
                    "x0": observed,
                    "x1": observed,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#2a3528", "width": 2},
                }
            ],
            "annotations": [
                {
                    "x": observed,
                    "y": 1,
                    "yref": "paper",
                    "text": "observed",
                    "showarrow": False,
                    "yshift": 10,
                    "font": {"size": 11, "color": "#2a3528"},
                }
            ],
        },
    )
    return create_plotly_div(traces, layout, div_id)


# ─────────────────────────────────────────────────────────────────────────────
# Prior densities
# ─────────────────────────────────────────────────────────────────────────────
def create_prior_density_chart(
    name: str,
    samples: np.ndarray,
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorDensity",
    color: str | None = None,
    reference: float | None = None,
    reference_label: str = "break-even",
) -> str:
    """One parameter's prior density (KDE; histogram fallback), as a small multiple.

    ``reference`` draws a dashed vertical line (e.g. break-even 1.0 for a prior
    ROI density) so the reader sees how much prior mass sits on each side."""
    chart_config = chart_config or ChartConfig(height=210)
    colors = config.color_scheme
    col = color or colors.primary_dark
    rgb = _hex_to_rgb(col)

    vals = np.asarray(samples, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]

    traces: list[dict]
    if vals.size >= 10 and float(np.std(vals)) > 0:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(vals)
            lo, hi = np.percentile(vals, [0.5, 99.5])
            pad = (hi - lo) * 0.08 or abs(hi) * 0.1 or 1.0
            grid = np.linspace(lo - pad, hi + pad, 160)
            dens = kde(grid)
            traces = [
                {
                    "type": "scatter",
                    "x": _arr(grid),
                    "y": _arr(dens),
                    "mode": "lines",
                    "line": {"color": col, "width": 2},
                    "fill": "tozeroy",
                    "fillcolor": f"rgba({rgb}, 0.14)",
                    "hovertemplate": "%{x:.3g}<extra></extra>",
                }
            ]
        except Exception:  # noqa: BLE001
            traces = []
    else:
        traces = []
    if not traces:
        traces = [
            {
                "type": "histogram",
                "x": _arr(vals),
                "nbinsx": 30,
                "marker": {"color": f"rgba({rgb}, 0.55)"},
                "hovertemplate": "%{x:.3g}: %{y}<extra></extra>",
            }
        ]

    extra: dict[str, Any] = {
        "showlegend": False,
        "margin": {"t": 28, "r": 12, "b": 34, "l": 40},
        "title": {
            "text": name,
            "font": {"family": "JetBrains Mono, monospace", "size": 12},
            "x": 0.02,
            "xanchor": "left",
        },
        "yaxis": {"showticklabels": False},
    }
    if reference is not None:
        extra["shapes"] = [
            {
                "type": "line",
                "x0": reference,
                "x1": reference,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#9aa498", "width": 1.3, "dash": "dash"},
            }
        ]
        extra["annotations"] = [
            {
                "x": reference,
                "y": 1,
                "yref": "paper",
                "text": reference_label,
                "showarrow": False,
                "yshift": 8,
                "font": {"size": 10, "color": "#4a5a48"},
            }
        ]
    layout = _base_layout(config, chart_config.height, extra)
    return create_plotly_div(traces, layout, div_id)


# ─────────────────────────────────────────────────────────────────────────────
# Prior structural components over time
# ─────────────────────────────────────────────────────────────────────────────
def create_prior_component_chart(
    label: str,
    component: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorComponent",
    color: str | None = None,
) -> str:
    """One structural component's prior in time (band + median + prior traces).

    ``component`` is one entry of
    :func:`~mmm_framework.reporting.helpers.prefit.prior_component_facts`:
    ``{dates, bands{lower,median,upper}, traces}`` in original KPI units. A
    dashed zero line anchors the reader — everything above/below is KPI the
    priors already grant this component.
    """
    chart_config = chart_config or ChartConfig(height=260)
    col = color or config.color_scheme.accent_dark
    rgb = _hex_to_rgb(col)
    dates = [str(d) for d in component["dates"]]
    bands = component["bands"]

    traces: list[dict] = [
        {
            "type": "scatter",
            "x": dates + dates[::-1],
            "y": _arr(bands["upper"]) + _arr(bands["lower"])[::-1],
            "fill": "toself",
            "fillcolor": f"rgba({rgb}, 0.14)",
            "line": {"color": "rgba(255,255,255,0)"},
            "name": "Prior 90%",
            "hoverinfo": "skip",
            "showlegend": False,
        }
    ]
    st = component.get("traces")
    if st is not None:
        for i, row in enumerate(np.asarray(st, dtype=float)):
            traces.append(
                {
                    "type": "scatter",
                    "x": dates,
                    "y": _arr(row),
                    "mode": "lines",
                    "line": {"color": f"rgba({rgb}, 0.30)", "width": 0.9},
                    "showlegend": False,
                    "hoverinfo": "skip",
                }
            )
    traces.append(
        {
            "type": "scatter",
            "x": dates,
            "y": _arr(bands["median"]),
            "mode": "lines",
            "line": {"color": col, "width": 2},
            "name": "Prior median",
            "hovertemplate": "%{y:,.0f}<extra></extra>",
            "showlegend": False,
        }
    )

    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "margin": {"t": 34, "r": 12, "b": 38, "l": 52},
            "title": {
                "text": label,
                "font": {"family": "Fraunces, serif", "size": 14, "color": "#2a3528"},
                "x": 0.02,
                "xanchor": "left",
            },
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "#9aa498", "width": 1.1, "dash": "dash"},
                }
            ],
            "yaxis": {"title": {"text": "KPI units / period", "font": {"size": 10}}},
        },
    )
    return create_plotly_div(traces, layout, div_id)


# ─────────────────────────────────────────────────────────────────────────────
# Prior-implied response curves
# ─────────────────────────────────────────────────────────────────────────────
def _band_traces(
    x: list, lower: list, upper: list, median: list, color: str, name: str
) -> list[dict]:
    rgb = _hex_to_rgb(color)
    return [
        {
            "type": "scatter",
            "x": x + x[::-1],
            "y": upper + lower[::-1],
            "fill": "toself",
            "fillcolor": f"rgba({rgb}, 0.16)",
            "line": {"color": "rgba(255,255,255,0)"},
            "name": f"{name} 90%",
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "x": x,
            "y": median,
            "mode": "lines",
            "line": {"color": color, "width": 2.2},
            "name": name,
            "hovertemplate": "%{x:.2f}: %{y:.2f}<extra></extra>",
            "showlegend": False,
        },
    ]


def create_prior_saturation_band(
    channel: str,
    curve: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorSat",
    color: str | None = None,
) -> str:
    """Prior-implied saturation shape for one channel (median + 90% band)."""
    chart_config = chart_config or ChartConfig(height=225)
    col = color or config.color_scheme.primary_dark
    x = _arr(curve["x"])
    traces = _band_traces(
        x,
        _arr(curve["lower"]),
        _arr(curve["upper"]),
        _arr(curve["median"]),
        col,
        channel,
    )
    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "margin": {"t": 30, "r": 12, "b": 38, "l": 44},
            "title": {
                "text": channel,
                "font": {"family": "Fraunces, serif", "size": 15, "color": "#2a3528"},
                "x": 0.02,
                "xanchor": "left",
            },
            "xaxis": {
                "title": {"text": "spend (share of observed max)", "font": {"size": 10}}
            },
            "yaxis": {"rangemode": "tozero"},
        },
    )
    return create_plotly_div(traces, layout, div_id)


def create_prior_adstock_band(
    channel: str,
    curve: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "priorAdstock",
    color: str | None = None,
) -> str:
    """Prior-implied carryover weights for one channel (median + 90% band)."""
    chart_config = chart_config or ChartConfig(height=225)
    col = color or config.color_scheme.accent_dark
    lags = _arr(curve["lags"])
    traces = _band_traces(
        lags,
        _arr(curve["lower"]),
        _arr(curve["upper"]),
        _arr(curve["median"]),
        col,
        channel,
    )
    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "margin": {"t": 30, "r": 12, "b": 38, "l": 44},
            "title": {
                "text": channel,
                "font": {"family": "Fraunces, serif", "size": 15, "color": "#2a3528"},
                "x": 0.02,
                "xanchor": "left",
            },
            "xaxis": {
                "title": {"text": "weeks after spend", "font": {"size": 10}},
                "dtick": 1,
            },
            "yaxis": {"rangemode": "tozero"},
        },
    )
    return create_plotly_div(traces, layout, div_id)


# ─────────────────────────────────────────────────────────────────────────────
# SBC (Simulation-Based Calibration)
# ─────────────────────────────────────────────────────────────────────────────
def create_sbc_rank_histogram(
    param: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "sbcRankHist",
) -> str:
    """SBC rank histogram with a simultaneous band, from a dashboard param dict.

    ``param`` is one entry of ``SBCResult.to_dashboard()["params"]`` — the bars
    come from ``bin_counts`` (always present) so a stored SBC result renders
    without the raw ranks. Bars outside the band are flagged in rust.
    """
    from ...diagnostics.sbc import rank_hist_band

    chart_config = chart_config or ChartConfig(height=260)
    colors = config.color_scheme

    counts = np.asarray(param.get("bin_counts", []), dtype=float)
    if counts.size == 0:
        return ""
    L = int(param.get("L", 100))
    n_sims = int(param.get("n_sims", int(counts.sum())))
    nb = int(param.get("n_bins", counts.size))

    lower, upper = rank_hist_band(n_sims, L, nb, prob=0.95)
    edges = np.linspace(-0.5, L + 0.5, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    expected = n_sims / nb

    danger = colors.danger
    accent = colors.accent
    bar_colors = [
        danger if (c < lo or c > hi) else accent
        for c, lo, hi in zip(counts, lower, upper)
    ]

    traces = [
        {
            "type": "scatter",
            "x": _arr(centers) + _arr(centers)[::-1],
            "y": _arr(upper) + _arr(lower)[::-1],
            "fill": "toself",
            "fillcolor": "rgba(122,138,120,0.16)",
            "line": {"color": "rgba(255,255,255,0)"},
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "bar",
            "x": _arr(centers),
            "y": _arr(counts),
            "width": width * 0.92,
            "marker": {"color": bar_colors},
            "hovertemplate": "rank≈%{x:.0f}<br>count=%{y}<extra></extra>",
            "showlegend": False,
        },
    ]

    p = param.get("chi2_pvalue")
    shape = param.get("shape", "")
    sub_bits = []
    if p is not None:
        sub_bits.append(f"χ² p={float(p):.3f}")
    if shape:
        sub_bits.append(str(shape))
    title = str(param.get("name", "parameter"))
    if sub_bits:
        title += f'<br><sub style="color:#7a8a78">{" · ".join(sub_bits)}</sub>'

    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "bargap": 0.04,
            "margin": {"t": 44, "r": 12, "b": 36, "l": 40},
            "title": {
                "text": title,
                "font": {"family": "JetBrains Mono, monospace", "size": 12},
                "x": 0.02,
                "xanchor": "left",
            },
            "shapes": [
                {
                    "type": "line",
                    "x0": float(edges[0]),
                    "x1": float(edges[-1]),
                    "y0": expected,
                    "y1": expected,
                    "line": {"color": "#9aa498", "width": 1.2, "dash": "dash"},
                }
            ],
            "xaxis": {
                "title": {"text": f"rank of θ* in {L} draws", "font": {"size": 10}}
            },
            "yaxis": {"title": {"text": "count", "font": {"size": 10}}},
        },
    )
    return create_plotly_div(traces, layout, div_id)


def create_sbc_ecdf_diff(
    param: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "sbcEcdfDiff",
) -> str:
    """SBC ECDF-difference plot with a Säilynoja simultaneous band.

    Needs the raw integer ranks (``param["int_ranks"]``, present when the
    result was serialized with ``to_dashboard(max_ranks=...)``); returns ``""``
    when they are unavailable so callers can degrade to the histogram alone.
    """
    ranks = param.get("int_ranks")
    if not ranks:
        return ""
    from ...diagnostics.sbc import ecdf_diff_band, normalized_ranks

    chart_config = chart_config or ChartConfig(height=260)
    colors = config.color_scheme

    r = np.asarray(ranks, dtype=int)
    L = int(param.get("L", int(r.max()) or 100))
    u = np.sort(normalized_ranks(r, L))
    z, lo, hi = ecdf_diff_band(r.size, prob=0.95, n_points=100)
    ecdf = np.searchsorted(u, z, side="right") / u.size
    diff = ecdf - z
    in_band = bool(np.all((diff >= lo - 1e-9) & (diff <= hi + 1e-9)))

    traces = [
        {
            "type": "scatter",
            "x": _arr(z) + _arr(z)[::-1],
            "y": _arr(hi) + _arr(lo)[::-1],
            "fill": "toself",
            "fillcolor": "rgba(122,138,120,0.16)",
            "line": {"color": "rgba(255,255,255,0)"},
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "x": _arr(z),
            "y": _arr(diff),
            "mode": "lines",
            "line": {
                "color": colors.accent if in_band else colors.danger,
                "width": 2,
            },
            "hovertemplate": "u=%{x:.2f}<br>ECDF−u=%{y:.3f}<extra></extra>",
            "showlegend": False,
        },
    ]
    layout = _base_layout(
        config,
        chart_config.height,
        {
            "showlegend": False,
            "margin": {"t": 44, "r": 12, "b": 36, "l": 46},
            "title": {
                "text": str(param.get("name", "parameter")),
                "font": {"family": "JetBrains Mono, monospace", "size": 12},
                "x": 0.02,
                "xanchor": "left",
            },
            "shapes": [
                {
                    "type": "line",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "#9aa498", "width": 1.2, "dash": "dash"},
                }
            ],
            "xaxis": {
                "range": [0, 1],
                "title": {"text": "normalized rank u", "font": {"size": 10}},
            },
            "yaxis": {"title": {"text": "ECDF(u) − u", "font": {"size": 10}}},
        },
    )
    return create_plotly_div(traces, layout, div_id)


__all__ = [
    "create_prior_predictive_fan",
    "create_prior_stat_distribution",
    "create_prior_density_chart",
    "create_prior_component_chart",
    "create_prior_saturation_band",
    "create_prior_adstock_band",
    "create_sbc_rank_histogram",
    "create_sbc_ecdf_diff",
]

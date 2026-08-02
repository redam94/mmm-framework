"""Charts for confounding sensitivity — tipping points, curves and surfaces.

Three views of the same question, in rising order of detail:

* :func:`create_tipping_point_chart` — one bar per channel showing how large a
  bias would have to be to overturn it, with the strongest observed-covariate
  benchmark drawn on top. This is the chart that answers "which of my numbers is
  fragile, and is that fragility plausible?" at a glance.
* :func:`create_bias_curve_chart` — the decision probability as the assumed
  overstatement grows, one line per channel. Shows the *shape* of the decline,
  which a single tipping point cannot.
* :func:`create_bias_contour_chart` — the two-dimensional audit over
  ``(mu, sigma)`` commitments for one channel: which analyst positions support
  the conclusion and which do not.

All three read a :meth:`ConfoundingSensitivityReport.to_dict` payload, so they
render identically from a live model, a stored artifact or a REST response.
"""

from __future__ import annotations

import html
from typing import Any

from ..config import ChartConfig, ReportConfig
from .base import create_plotly_div

__all__ = [
    "create_bias_contour_chart",
    "create_bias_curve_chart",
    "create_tipping_point_chart",
]

# Verdict -> semantic color role on the ColorScheme. "resilient" deliberately
# does not get the success color: the claim is only that the conclusion survived
# the range scanned, and a green bar would read as a clean bill of health.
_VERDICT_ROLE = {
    "overturned": "danger",
    "fragile": "warning",
    "resilient": "primary",
    "not_assessable": "text_muted",
}


def _verdict_color(colors: Any, verdict: str) -> str:
    return getattr(colors, _VERDICT_ROLE.get(verdict, "text_muted"), colors.text_muted)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def create_tipping_point_chart(
    report: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "confoundingTippingPlot",
) -> str:
    """Per-channel tipping points, with observed-covariate benchmarks overlaid.

    The bar is how much the estimate would have to be overstated before the
    channel stops clearing break-even. The diamond is what a confounder as strong
    as the strongest measured covariate would actually imply — so a diamond to
    the *right* of its bar is the case worth acting on, and the one the caption
    has to name.
    """
    channels = report.get("channels") or []
    if not channels:
        return ""

    colors = config.color_scheme
    chart_config = chart_config or ChartConfig(
        height=max(260, 58 * len(channels)),
        x_title="Bias needed to overturn (share of the estimate)",
    )

    names: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []
    texts: list[str] = []
    hovers: list[str] = []
    max_scanned = 0.0

    bench_x: list[float] = []
    bench_y: list[str] = []
    bench_hover: list[str] = []

    for ch in channels:
        sens = ch.get("sensitivity") or {}
        tip = sens.get("tipping_mu") or {}
        name = html.escape(str(ch.get("channel", "")))
        verdict = str(sens.get("verdict", "not_assessable"))
        scanned = float(tip.get("max_scanned") or 0.0)
        max_scanned = max(max_scanned, scanned)

        if tip.get("already_below"):
            value, label = 0.0, "already below"
        elif tip.get("crossed") and tip.get("value") is not None:
            value, label = float(tip["value"]), _pct(float(tip["value"]))
        else:
            # Survived the whole scan. Drawing the full bar would imply the
            # tipping point sits exactly at the edge; the ">" in the label is
            # what keeps it honest.
            value, label = scanned, f">{_pct(scanned)}"

        names.append(name)
        values.append(value)
        bar_colors.append(_verdict_color(colors, verdict))
        texts.append(label)
        hovers.append(
            f"<b>{name}</b><br>{html.escape(str(ch.get('metric_label', 'ROI')))} "
            f"{sens.get('estimate', float('nan')):.2f}"
            f"<br>P(above break-even) {_pct(sens.get('prob_at_zero_bias'))}"
            f"<br>Tipping point {label}"
            f"<br>Verdict: {html.escape(verdict)}<extra></extra>"
        )

        best = None
        for b in ch.get("benchmarks") or []:
            if b.get("status") != "ok":
                continue
            frac = b.get("fractional_bias")
            if frac is None or frac != frac:  # NaN
                continue
            if best is None or frac > best["fractional_bias"]:
                best = b
        if best is not None:
            bench_x.append(float(best["fractional_bias"]))
            bench_y.append(name)
            bench_hover.append(
                f"<b>{name}</b><br>a confounder "
                f"{best['kd']:g}x as strong as "
                f"{html.escape(str(best['covariate']))}"
                f"<br>implies {_pct(best['fractional_bias'])} bias<extra></extra>"
            )

    traces: list[dict] = [
        {
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": names,
            "marker": {"color": bar_colors},
            "text": texts,
            "textposition": "auto",
            "hovertemplate": hovers,
            "name": "Bias needed to overturn",
        }
    ]
    if bench_x:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": bench_x,
                "y": bench_y,
                "marker": {
                    "symbol": "diamond",
                    "size": 13,
                    "color": colors.accent_dark,
                    "line": {"width": 1.5, "color": colors.background},
                },
                "hovertemplate": bench_hover,
                "name": "Strongest measured benchmark",
            }
        )

    layout = chart_config.to_plotly_layout(colors)
    layout.update(
        {
            "barmode": "overlay",
            "xaxis": {
                **layout.get("xaxis", {}),
                "tickformat": ".0%",
                "range": [0, max(max_scanned, max(bench_x, default=0.0)) * 1.08],
            },
            "yaxis": {**layout.get("yaxis", {}), "automargin": True},
            "showlegend": bool(bench_x),
            "legend": {"orientation": "h", "y": -0.18},
        }
    )
    return create_plotly_div(traces, layout, div_id)


def create_bias_curve_chart(
    report: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "confoundingCurvePlot",
) -> str:
    """Decision probability against assumed overstatement, one line per channel.

    Draws the curve the scan already produced rather than recomputing it, so the
    line and the reported tipping point can never disagree.
    """
    channels = report.get("channels") or []
    colors = config.color_scheme
    chart_config = chart_config or ChartConfig(
        height=380,
        x_title="Assumed overstatement (share of the estimate)",
        y_title="P(above break-even)",
    )

    traces: list[dict] = []
    threshold = float(report.get("threshold") or 0.95)
    for ch in channels:
        sens = ch.get("sensitivity") or {}
        curve = ((sens.get("tipping_mu") or {}).get("curve")) or []
        if not curve:
            continue
        name = html.escape(str(ch.get("channel", "")))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [float(p[0]) for p in curve],
                "y": [float(p[1]) for p in curve],
                "name": name,
                "line": {"width": 2.5},
                "hovertemplate": (
                    f"<b>{name}</b><br>overstatement %{{x:.0%}}"
                    "<br>P = %{y:.1%}<extra></extra>"
                ),
            }
        )
    if not traces:
        return ""

    x_max = max(
        max(
            float(p[0])
            for p in ((c.get("sensitivity") or {}).get("tipping_mu") or {}).get("curve")
            or [[0.0, 0.0]]
        )
        for c in channels
    )
    traces.append(
        {
            "type": "scatter",
            "mode": "lines",
            "x": [0.0, x_max],
            "y": [threshold, threshold],
            "line": {"color": colors.text_muted, "width": 1.5, "dash": "dot"},
            "name": f"decision threshold ({threshold:.0%})",
            "hoverinfo": "skip",
        }
    )

    layout = chart_config.to_plotly_layout(colors)
    layout.update(
        {
            "xaxis": {**layout.get("xaxis", {}), "tickformat": ".0%"},
            "yaxis": {
                **layout.get("yaxis", {}),
                "tickformat": ".0%",
                "range": [0, 1.02],
            },
            "legend": {"orientation": "h", "y": -0.22},
        }
    )
    return create_plotly_div(traces, layout, div_id)


def create_bias_contour_chart(
    surface: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "confoundingSurfacePlot",
    title: str = "",
) -> str:
    """The ``(mu, sigma)`` audit for one channel.

    Every point is an analyst position — a belief about how large the bias is and
    how sure they are of it — and the contour at the decision threshold separates
    the positions that support the conclusion from the ones that do not. The
    partition is the output; a single verdict is a point on it.
    """
    if not surface:
        return ""
    mu_grid = surface.get("mu_grid") or []
    sigma_grid = surface.get("sigma_grid") or []
    prob = surface.get("prob") or []
    if not mu_grid or not sigma_grid or not prob:
        return ""

    colors = config.color_scheme
    threshold = float(surface.get("threshold") or 0.95)
    fraction = surface.get("scale") == "fraction_of_mean"
    tickformat = ".0%" if fraction else None
    chart_config = chart_config or ChartConfig(
        height=400,
        x_title="Assumed bias (mu)",
        y_title="Uncertainty about the bias (sigma)",
    )

    traces = [
        {
            "type": "contour",
            "x": [float(v) for v in mu_grid],
            "y": [float(v) for v in sigma_grid],
            "z": [[float(v) for v in row] for row in prob],
            "colorscale": [
                [0.0, colors.danger],
                [threshold * 0.9, colors.warning],
                [1.0, colors.primary],
            ],
            "contours": {
                "start": 0.0,
                "end": 1.0,
                "size": 0.05,
                "showlabels": True,
                "labelfont": {"size": 10, "color": colors.background},
            },
            "colorbar": {"title": "P", "tickformat": ".0%"},
            "hovertemplate": (
                "bias %{x:.2f}<br>uncertainty %{y:.2f}<br>P = %{z:.1%}<extra></extra>"
            ),
        }
    ]

    layout = chart_config.to_plotly_layout(colors)
    xaxis = {**layout.get("xaxis", {})}
    yaxis = {**layout.get("yaxis", {})}
    if tickformat:
        xaxis["tickformat"] = tickformat
        yaxis["tickformat"] = tickformat
    layout.update({"xaxis": xaxis, "yaxis": yaxis, "showlegend": False})
    if title:
        layout["title"] = {"text": html.escape(title)}
    return create_plotly_div(traces, layout, div_id)

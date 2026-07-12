"""Spec-curve robustness chart (issue #103).

Shows how each channel's ROI moves across a pre-registered set of defensible
specifications — the honest alternative to a single hand-picked number. One
marker per (channel × spec), the primary spec ringed, and the LOO-stacking
model-averaged (BMA) estimate as a bold diamond, all against the break-even
reference. Tight clustering = a robust finding; a wide spread (especially one
that crosses break-even) = a number the single-spec report would have hidden.
"""

from __future__ import annotations

from typing import Any

from ..config import ChartConfig, ReportConfig
from .base import create_plotly_div

# Qualitative palette for the specs (distinct from the channel palette, since
# here color encodes the SPEC, not the channel).
_SPEC_PALETTE = [
    "#4a6d8a",
    "#b8860b",
    "#6d8a4a",
    "#a04535",
    "#7b5ea5",
    "#3a8a8a",
    "#8a6408",
    "#5a7a3a",
]


def create_spec_curve_plot(
    spec_curve: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "specCurvePlot",
    reference_line: float = 1.0,
) -> str:
    """Render the spec-curve from a :meth:`SpecCurveResult.to_dict` payload.

    ``spec_curve`` carries ``channels``, ``specs``, ``primary``, ``bma``
    (per-channel model-averaged ROI + CI), and ``per_spec`` (each spec's
    per-channel ROI). Channels are laid out on the y-axis; each spec is a small
    marker jittered within the channel's row, the primary spec ringed, and the
    BMA a bold diamond with its credible interval.
    """
    channels: list[str] = list(spec_curve.get("channels") or [])
    specs: list[str] = list(spec_curve.get("specs") or [])
    primary: str | None = spec_curve.get("primary")
    per_spec: dict[str, Any] = spec_curve.get("per_spec") or {}
    bma: dict[str, Any] = spec_curve.get("bma") or {}
    if not channels or not specs:
        return ""

    colors = config.color_scheme
    chart_config = chart_config or ChartConfig(
        height=max(260, 64 * len(channels)),
        x_title="ROI across specifications",
    )
    y_of = {ch: i for i, ch in enumerate(channels)}
    n_spec = max(len(specs), 1)

    traces: list[dict[str, Any]] = []

    # One scatter per spec (so the legend reads as the spec set).
    for si, spec in enumerate(specs):
        rows = (per_spec.get(spec) or {}).get("roi") or {}
        xs, ys, cds = [], [], []
        for ch in channels:
            r = rows.get(ch)
            if not r or r.get("mean") is None:
                continue
            # Vertical jitter so co-located spec points don't overprint.
            jitter = (si - (n_spec - 1) / 2.0) / (n_spec * 1.6)
            xs.append(float(r["mean"]))
            ys.append(y_of[ch] + jitter)
            cds.append([ch, r.get("lower"), r.get("upper")])
        if not xs:
            continue
        is_primary = spec == primary
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": ys,
                "mode": "markers",
                "name": spec + (" (primary)" if is_primary else ""),
                "marker": {
                    "color": _SPEC_PALETTE[si % len(_SPEC_PALETTE)],
                    "size": 12 if is_primary else 9,
                    "symbol": "circle",
                    "line": {
                        "color": colors.text if is_primary else "#ffffff",
                        "width": 2.2 if is_primary else 0.8,
                    },
                },
                "hovertemplate": (
                    "%{customdata[0]} · " + spec + "<br>ROI: %{x:.2f}"
                    "<br>CI: [%{customdata[1]:.2f}, %{customdata[2]:.2f}]<extra></extra>"
                ),
                "customdata": cds,
            }
        )

    # BMA (LOO-stacked) diamond per channel, with its credible interval.
    bx, by, blo, bhi = [], [], [], []
    for ch in channels:
        b = bma.get(ch)
        if not b or b.get("mean") is None:
            continue
        bx.append(float(b["mean"]))
        by.append(y_of[ch])
        blo.append(float(b["mean"]) - float(b.get("lower", b["mean"])))
        bhi.append(float(b.get("upper", b["mean"])) - float(b["mean"]))
    if bx:
        traces.append(
            {
                "type": "scatter",
                "x": bx,
                "y": by,
                "mode": "markers",
                "name": "Model-averaged (BMA)",
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": bhi,
                    "arrayminus": blo,
                    "color": colors.text_muted,
                    "thickness": 2,
                    "width": 7,
                },
                "marker": {
                    "color": colors.text,
                    "size": 15,
                    "symbol": "diamond",
                    "line": {"color": "#ffffff", "width": 1},
                },
                "hovertemplate": "%{y}<br>BMA ROI: %{x:.2f}<extra></extra>",
            }
        )

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Channel ROI across the pre-registered spec set",
        "font": {"size": 16},
    }
    layout["yaxis"].update(
        {
            "tickmode": "array",
            "tickvals": list(range(len(channels))),
            "ticktext": channels,
            "autorange": "reversed",
            "range": [-0.6, len(channels) - 0.4],
        }
    )
    layout["shapes"] = [
        {
            "type": "line",
            "x0": reference_line,
            "x1": reference_line,
            "y0": -0.6,
            "y1": len(channels) - 0.4,
            "line": {"color": colors.text_muted, "width": 1, "dash": "dash"},
        }
    ]
    layout["annotations"] = [
        {
            "x": reference_line,
            "y": -0.5,
            "xref": "x",
            "yref": "y",
            "text": "Break-even" if reference_line == 1.0 else "Zero",
            "showarrow": False,
            "font": {"size": 10, "color": colors.text_muted},
        }
    ]
    layout["legend"] = {"orientation": "h", "y": -0.18, "font": {"size": 10}}

    return create_plotly_div(traces, layout, div_id)


__all__ = ["create_spec_curve_plot"]

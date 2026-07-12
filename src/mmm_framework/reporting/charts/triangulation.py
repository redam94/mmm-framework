"""Triangulation chart — MMM × experiment × platform per channel (issue #104).

For each channel, the three sources' estimates of the return sit side by side
(experiment / MMM / platform, each colored + with its interval) with the
reconciled recommendation as a bold outlined diamond, against the break-even
reference. Convergence reads as three markers stacked together; a platform
figure floating far to the right of the incremental estimates is the
last-touch-inflation story at a glance.
"""

from __future__ import annotations

import html as _html
from typing import Any

from ..config import ChartConfig, ReportConfig
from .base import create_plotly_div

_SOURCE_COLORS = {
    "experiment": "#5a7a3a",  # sage — the causal anchor
    "mmm": "#4a6d8a",  # steel — model-identified
    "platform": "#b8860b",  # gold — correlational context
}


def create_triangulation_chart(
    triangulation: dict[str, Any],
    config: ReportConfig,
    chart_config: ChartConfig | None = None,
    div_id: str = "triangulationPlot",
    reference_line: float = 1.0,
) -> str:
    """Render the triangulation panel from a :meth:`TriangulationResult.to_dict`."""
    channels = list(triangulation.get("channels") or [])
    if not channels:
        return ""

    colors = config.color_scheme
    chart_config = chart_config or ChartConfig(
        height=max(280, 70 * len(channels)),
        x_title="Return per $ (incremental unless noted)",
    )
    # Escape tick labels: a no-op for ordinary channel names, but neutralizes a
    # "</script>" breakout from an adversarial column name embedded in the div's
    # JSON (Plotly renders these as SVG text, so entities are harmless).
    names = [_html.escape(str(c["channel"])) for c in channels]
    y_of = {c["channel"]: i for i, c in enumerate(channels)}

    # One scatter per source type so the legend reads experiment/MMM/platform.
    traces: list[dict[str, Any]] = []
    order = ["experiment", "mmm", "platform"]
    for si, stype in enumerate(order):
        xs, ys, errp, errm, cds = [], [], [], [], []
        for c in channels:
            src = next((s for s in c["sources"] if s["source"] == stype), None)
            if not src or src.get("value") is None:
                continue
            jitter = (si - 1) * 0.16
            xs.append(src["value"])
            ys.append(y_of[c["channel"]] + jitter)
            lo, hi = src.get("lower"), src.get("upper")
            errp.append((hi - src["value"]) if hi is not None else 0)
            errm.append((src["value"] - lo) if lo is not None else 0)
            cds.append([src["label"], src.get("attribution_window") or "—"])
        if not xs:
            continue
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": ys,
                "mode": "markers",
                "name": {
                    "experiment": "Experiment",
                    "mmm": "MMM",
                    "platform": "Platform",
                }[stype],
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": errp,
                    "arrayminus": errm,
                    "color": colors.text_muted,
                    "thickness": 1.6,
                    "width": 6,
                },
                "marker": {
                    "color": _SOURCE_COLORS[stype],
                    "size": 12,
                    "symbol": "diamond" if stype == "platform" else "circle",
                    "line": {"color": "#ffffff", "width": 1},
                },
                "hovertemplate": (
                    "%{customdata[0]}<br>Return: %{x:.2f}×"
                    "<br>Window: %{customdata[1]}<extra></extra>"
                ),
                "customdata": cds,
            }
        )

    # Reconciled recommendation per channel (bold ringed marker).
    rx, ry, rc = [], [], []
    for c in channels:
        rec = c.get("reconciled") or {}
        if rec.get("value") is None:
            continue
        rx.append(rec["value"])
        ry.append(y_of[c["channel"]])
        rc.append(rec.get("basis") or "—")
    if rx:
        traces.append(
            {
                "type": "scatter",
                "x": rx,
                "y": ry,
                "mode": "markers",
                "name": "Reconciled",
                "marker": {
                    "color": "rgba(0,0,0,0)",
                    "size": 20,
                    "symbol": "circle-open",
                    "line": {"color": colors.text, "width": 2.2},
                },
                "hovertemplate": "Reconciled: %{x:.2f}× (from %{customdata})<extra></extra>",
                "customdata": rc,
            }
        )

    layout = chart_config.to_plotly_layout(colors)
    layout["title"] = {
        "text": "Triangulation — MMM × experiment × platform",
        "font": {"size": 16},
    }
    layout["yaxis"].update(
        {
            "tickmode": "array",
            "tickvals": list(range(len(names))),
            "ticktext": names,
            "autorange": "reversed",
            "range": [-0.6, len(names) - 0.4],
        }
    )
    layout["shapes"] = [
        {
            "type": "line",
            "x0": reference_line,
            "x1": reference_line,
            "y0": -0.6,
            "y1": len(names) - 0.4,
            "line": {"color": colors.text_muted, "width": 1, "dash": "dash"},
        }
    ]
    layout["legend"] = {"orientation": "h", "y": -0.16, "font": {"size": 10}}
    return create_plotly_div(traces, layout, div_id)


__all__ = ["create_triangulation_chart"]

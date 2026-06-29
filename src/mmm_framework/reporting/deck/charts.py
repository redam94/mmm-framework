"""Matplotlib chart renderers for the MMM slide deck — each returns PNG bytes.

These draw the deck's figures directly from model-derived numbers (no Plotly, no
interactive HTML, no AI). Rendering is pyplot-free (a bare ``matplotlib.figure.
Figure`` with the Agg canvas), so it never touches the global backend and is safe
to call from server threads or inside a notebook. Styling is intentionally
restrained and re-themeable via the ``palette`` argument so it can be matched to a
client template's brand colors later.

The centerpiece is :func:`saturation_zones_png`, which visualizes a channel's
response curve together with the **breakthrough / optimal / saturation** spend
zones defined on marginal-ROI break-even bands (see
:func:`mmm_framework.reporting.helpers.compute_response_zones`).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from matplotlib.figure import Figure  # pyplot-free: no global backend, thread-safe

# Default palette (re-themeable). Zone colors carry the "traffic-light" meaning:
# breakthrough = under-invested (go/green), optimal = on-target (blue),
# saturation = over-invested (amber).
PALETTE = {
    "primary": "#1c3d5a",
    "accent": "#1c7ed6",
    "response": "#1c3d5a",
    "roi": "#2f9e44",
    "mroi": "#e8590c",
    "breakthrough": "#2f9e44",
    "optimal": "#1c7ed6",
    "saturation": "#f08c00",
    "current": "#212529",
    "optimal_mark": "#1c7ed6",
    "grid": "#dee2e6",
    "muted": "#868e96",
}

_BAR_COLORS = [
    "#1c7ed6",
    "#2f9e44",
    "#e8590c",
    "#7048e8",
    "#0ca678",
    "#f783ac",
    "#495057",
    "#fab005",
    "#15aabf",
    "#d6336c",
]


def _palette(overrides: dict[str, str] | None) -> dict[str, str]:
    p = dict(PALETTE)
    if overrides:
        p.update({k: v for k, v in overrides.items() if v})
    return p


def _finish(fig) -> bytes:
    """Serialize a figure to PNG bytes (pyplot-free, so no global figure registry
    to close — the Figure is GC'd)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def _style_ax(ax, palette):
    ax.grid(True, color=palette["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def saturation_zones_png(
    zones: Any,
    *,
    currency: str = "$",
    palette: dict[str, str] | None = None,
    width: float = 10.0,
    height: float = 5.4,
) -> bytes:
    """Render a channel's response curve with its breakthrough / optimal /
    saturation spend zones and the ROI + marginal-ROI overlay.

    ``zones`` is a :class:`~mmm_framework.reporting.helpers.results.SpendResponseZones`.
    Left axis: response (KPI) vs per-period spend with an HDI band. Right axis:
    average ROI and marginal ROI, with a dashed break-even line. Vertical markers
    show current and optimal spend; the three zones are shaded behind the curves.
    """
    p = _palette(palette)
    x = np.asarray(zones.spend_grid, dtype=float)
    fig = Figure(figsize=(width, height))
    ax1 = fig.subplots()

    # --- zone shading (behind everything) ---
    zone_specs = [
        ("breakthrough", zones.breakthrough_range, "Breakthrough"),
        ("optimal", zones.optimal_range, "Optimal"),
        ("saturation", zones.saturation_range, "Saturation"),
    ]
    for key, (lo, hi), label in zone_specs:
        if hi - lo <= 0:
            continue
        ax1.axvspan(lo, hi, color=p[key], alpha=0.10, lw=0)
        ax1.text(
            (lo + hi) / 2.0,
            0.97,
            label,
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=p[key],
            fontweight="bold",
            alpha=0.9,
        )

    # --- response curve (left axis) ---
    ax1.fill_between(
        x,
        zones.response_lower,
        zones.response_upper,
        color=p["response"],
        alpha=0.12,
        lw=0,
    )
    ax1.plot(x, zones.response_mean, color=p["response"], lw=2.2, label="Response")
    ax1.set_xlabel(f"Spend per period ({currency})")
    ax1.set_ylabel("Response (KPI)", color=p["response"])
    ax1.tick_params(axis="y", labelcolor=p["response"])
    _style_ax(ax1, p)

    # --- ROI + marginal ROI (right axis) ---
    ax2 = ax1.twinx()
    ax2.plot(x, zones.roi_mean, color=p["roi"], lw=1.8, ls="-", label="Avg ROI")
    ax2.plot(x, zones.mroi_mean, color=p["mroi"], lw=1.8, ls="--", label="Marginal ROI")
    ax2.axhline(zones.break_even, color=p["muted"], lw=1.2, ls=":", label="Break-even")
    ax2.set_ylabel("ROI / marginal ROI", color=p["mroi"])
    ax2.tick_params(axis="y", labelcolor=p["mroi"])
    # Keep the break-even region visible (ROI/mROI both peak at low spend).
    top = float(np.nanmax(zones.mroi_mean[0:1]))
    top = max(top, zones.break_even * 1.5, float(zones.current_mroi) * 1.2)
    ax2.set_ylim(0, top * 1.05)
    for spine in ("top",):
        ax2.spines[spine].set_visible(False)

    # --- current + optimal markers ---
    ax1.axvline(zones.current_spend, color=p["current"], lw=1.6, alpha=0.85)
    ax1.text(
        zones.current_spend,
        0.04,
        "current",
        transform=ax1.get_xaxis_transform(),
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color=p["current"],
    )
    if zones.optimal_spend is not None:
        ax1.axvline(
            zones.optimal_spend, color=p["optimal_mark"], lw=1.6, ls="--", alpha=0.9
        )
        ax1.text(
            zones.optimal_spend,
            0.04,
            "optimal",
            transform=ax1.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8,
            color=p["optimal_mark"],
        )

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9)

    ax1.set_title(
        f"{zones.channel} — response & efficiency zones", fontsize=12, fontweight="bold"
    )
    ax1.set_xlim(float(x[0]), float(x[-1]))
    return _finish(fig)


def roi_forest_png(
    channel_roi: dict[str, dict[str, float]],
    *,
    break_even: float = 1.0,
    palette: dict[str, str] | None = None,
    width: float = 8.0,
    height: float = 4.8,
) -> bytes:
    """Horizontal ROI point + HDI per channel, sorted, with a break-even line."""
    p = _palette(palette)
    items = sorted(channel_roi.items(), key=lambda kv: kv[1].get("mean", 0.0))
    names = [k for k, _ in items]
    means = np.array([v.get("mean", 0.0) for _, v in items])
    lows = np.array([v.get("lower", v.get("mean", 0.0)) for _, v in items])
    highs = np.array([v.get("upper", v.get("mean", 0.0)) for _, v in items])
    y = np.arange(len(names))

    fig = Figure(figsize=(width, max(height, 0.5 * len(names) + 1.5)))
    ax = fig.subplots()
    ax.errorbar(
        means,
        y,
        xerr=[means - lows, highs - means],
        fmt="o",
        color=p["accent"],
        ecolor=p["muted"],
        elinewidth=1.6,
        capsize=3,
        markersize=6,
    )
    ax.axvline(
        break_even,
        color=p["saturation"],
        lw=1.3,
        ls="--",
        label=f"break-even ({break_even:g})",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("ROI (KPI per unit spend)")
    ax.set_title("Return on investment by channel", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _style_ax(ax, p)
    return _finish(fig)


def decomposition_png(
    component_totals: dict[str, float],
    *,
    palette: dict[str, str] | None = None,
    width: float = 8.0,
    height: float = 4.8,
) -> bytes:
    """Horizontal bar of total contribution by component (base + channels),
    sorted, with share-of-total labels."""
    p = _palette(palette)
    items = sorted(component_totals.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items], dtype=float)
    total = float(np.sum(np.abs(vals))) or 1.0
    y = np.arange(len(names))

    fig = Figure(figsize=(width, max(height, 0.5 * len(names) + 1.5)))
    ax = fig.subplots()
    colors = [_BAR_COLORS[i % len(_BAR_COLORS)] for i in range(len(names))]
    ax.barh(y, vals, color=colors, alpha=0.9)
    for yi, v in zip(y, vals):
        ax.text(
            v + (0.01 * total if v >= 0 else -0.01 * total),
            yi,
            f"{v/total:+.0%}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=p["muted"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Total contribution (KPI units)")
    ax.set_title("KPI decomposition by component", fontsize=12, fontweight="bold")
    _style_ax(ax, p)
    return _finish(fig)


def fit_png(
    dates: Any,
    actual: Any,
    predicted: dict[str, Any] | None,
    *,
    r2: float | None = None,
    palette: dict[str, str] | None = None,
    width: float = 10.0,
    height: float = 4.4,
) -> bytes:
    """Actual vs predicted over time, with a predictive-interval band."""
    p = _palette(palette)
    actual = np.asarray(actual, dtype=float)
    x = np.asarray(dates) if dates is not None else np.arange(len(actual))
    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    if predicted:
        mean = np.asarray(predicted.get("mean"), dtype=float)
        lo = predicted.get("lower")
        hi = predicted.get("upper")
        if lo is not None and hi is not None:
            ax.fill_between(
                x,
                np.asarray(lo, dtype=float),
                np.asarray(hi, dtype=float),
                color=p["accent"],
                alpha=0.15,
                lw=0,
                label="predictive interval",
            )
        ax.plot(x, mean, color=p["accent"], lw=2.0, label="predicted")
    ax.plot(x, actual, color=p["primary"], lw=1.4, alpha=0.85, label="actual")
    title = "Model fit: actual vs. predicted"
    if r2 is not None and np.isfinite(r2):
        title += f"  (R² = {r2:.2f})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("KPI")
    ax.legend(loc="best", fontsize=8)
    _style_ax(ax, p)
    fig.autofmt_xdate()
    return _finish(fig)


def reallocation_png(
    rows: list[dict[str, Any]],
    *,
    currency: str = "$",
    palette: dict[str, str] | None = None,
    width: float = 9.0,
    height: float = 4.8,
) -> bytes:
    """Grouped bars of current vs. optimal (profit-maximizing) spend per channel.

    ``rows`` = ``[{"channel", "current", "optimal"}, ...]`` (channels without an
    in-range optimum pass ``optimal=None`` and show only current spend).
    """
    p = _palette(palette)
    names = [r["channel"] for r in rows]
    cur = np.array([float(r.get("current", 0.0)) for r in rows])
    opt = np.array(
        [float(r["optimal"]) if r.get("optimal") is not None else np.nan for r in rows]
    )
    y = np.arange(len(names))
    h = 0.38

    fig = Figure(figsize=(width, max(height, 0.6 * len(names) + 1.5)))
    ax = fig.subplots()
    ax.barh(y + h / 2, cur, height=h, color=p["muted"], alpha=0.9, label="current")
    ax.barh(
        y - h / 2,
        np.nan_to_num(opt),
        height=h,
        color=p["optimal"],
        alpha=0.9,
        label="optimal",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"Spend per period ({currency})")
    ax.set_title("Current vs. profit-maximizing spend", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _style_ax(ax, p)
    return _finish(fig)


__all__ = [
    "PALETTE",
    "saturation_zones_png",
    "roi_forest_png",
    "decomposition_png",
    "fit_png",
    "reallocation_png",
]

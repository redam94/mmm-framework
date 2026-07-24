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


def _money(v: float | None, currency: str = "$") -> str:
    """Compact money label ("$1.2M"); shared with the deck builder."""
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(v)
    if av >= 1e9:
        return f"{currency}{v/1e9:.1f}B"
    if av >= 1e6:
        return f"{currency}{v/1e6:.1f}M"
    if av >= 1e3:
        return f"{currency}{v/1e3:.1f}K"
    return f"{currency}{v:,.0f}"


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


def _roi_axis_top(zones: Any) -> float:
    """Upper limit for the ROI / marginal-ROI (right) axis of the zone chart.

    Avg ROI and marginal ROI DIVERGE as spend→0 for concave curves (the first
    dollar is the most efficient; for Hill slope < 1 they go to ``+inf``), so
    anchoring to the whole-curve max would pin the axis to that degenerate
    near-zero spike and squash everything else. Scale to the decision-relevant
    region instead: drop the near-zero toe for a CONCAVE curve (marginal-ROI peak
    at the toe), but keep an INTERIOR peak for an S-shaped curve (its breakthrough
    region). Break-even and the current/optimal levels are always kept in frame.
    """
    x = np.asarray(zones.spend_grid, dtype=float)
    mroi_full = np.asarray(zones.mroi_mean, dtype=float)
    roi_full = np.asarray(zones.roi_mean, dtype=float)
    finite = np.isfinite(mroi_full)
    if finite.any():
        i_peak = int(np.nanargmax(np.where(finite, mroi_full, -np.inf)))
        # concave ⇒ marginal-ROI peaks at the toe ⇒ scale from current spend on;
        # S-shaped ⇒ interior peak is meaningful, keep the whole curve in frame.
        sel = (x >= zones.current_spend) & finite if i_peak <= 1 else finite
        if not sel.any():
            sel = finite
    else:
        sel = np.zeros_like(x, dtype=bool)

    def _region_max(arr):
        mask = sel & np.isfinite(arr)
        return float(np.nanmax(arr[mask])) if mask.any() else float("-inf")

    cands = [_region_max(mroi_full), _region_max(roi_full), zones.break_even * 1.5]
    for v in (zones.current_mroi, zones.current_roi):
        if np.isfinite(v):
            cands.append(float(v) * 1.2)
    top = max((c for c in cands if np.isfinite(c)), default=float(zones.break_even))
    return max(top, 1e-6)


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
    axis_w = float(x[-1] - x[0]) or 1.0
    for key, (lo, hi), label in zone_specs:
        if hi - lo <= 0:  # empty zone (e.g. no breakthrough on a concave curve)
            continue
        ax1.axvspan(lo, hi, color=p[key], alpha=0.10, lw=0)
        # only label zones wide enough to hold the text (avoid cramped overlaps)
        if (hi - lo) / axis_w >= 0.08:
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
    # scale the right axis to the decision-relevant region (see _roi_axis_top):
    # avg/marginal ROI diverge as spend→0 for concave curves and would otherwise
    # blow out the scale.
    ax2.set_ylim(0, _roi_axis_top(zones) * 1.05)
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
    title: str = "Return on investment by channel",
    xlabel: str = "ROI (KPI per unit spend)",
) -> bytes:
    """Horizontal point + HDI per channel, sorted, with a break-even line. Used
    for both average ROI and (via ``title``/``xlabel``) marginal ROI."""
    p = _palette(palette)
    items = sorted(channel_roi.items(), key=lambda kv: kv[1].get("mean", 0.0))
    names = [k for k, _ in items]
    means = np.array([v.get("mean", 0.0) for _, v in items])
    lows = np.array([v.get("lower", v.get("mean", 0.0)) for _, v in items])
    highs = np.array([v.get("upper", v.get("mean", 0.0)) for _, v in items])
    y = np.arange(len(names))

    fig = Figure(figsize=(width, height))
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
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    _style_ax(ax, p)
    return _finish(fig)


# components that are background structure (not media, not the intercept) —
# drawn hatched, matching the template's "not causally identified" legend
_BACKGROUND_NAMES = {
    "Trend",
    "Seasonality",
    "Events",
    "Price & Promotion",
    "Geo",
    "Product",
    "Controls",
}

# waterfall palette, themed to the template legend (slide 5 swatches)
_WF_BASELINE = "#C9C2AC"  # tan — "Baseline"
_WF_HATCH_EDGE = "#6C8FAD"  # steel blue — "Control variables"
_WF_TOTAL = "#2A3528"  # near-black green — "Total revenue"
_WF_MEDIA = [
    "#5A7A3A",
    "#3A5A75",
    "#8A6408",
    "#7A3525",
    "#4A6D2A",
    "#6E5A8A",
    "#2F6B5E",
    "#B0713A",
    "#5E5E5E",
    "#9C8A3A",
]


def _is_background(name: str, media_keys: set[str] | None) -> bool:
    """Background (hatched, "not causally identified") vs causal/media.

    The extractors emit background components from a CLOSED vocabulary
    (:data:`_BACKGROUND_NAMES` + the ``Control:`` prefix), while media/pathway
    names are open-ended — raw channel names, but also the extended models'
    decorated keys ("Via awareness", "tv (direct)", "Cross-outcome effects").
    So classification defaults to *causal*: only the known background
    vocabulary is hatched. ``media_keys`` can only ADD certainty (a name listed
    there is never background), never demote an unknown name to background —
    demoting would hatch every mediated pathway of a NestedMMM as "not causal"
    and re-anchor its floor into the baseline bar."""
    if name == "Baseline":
        return False
    if media_keys is not None and name in media_keys:
        return False
    return name in _BACKGROUND_NAMES or name.startswith("Control:")


def waterfall_entries(
    component_totals: dict[str, float],
    series: dict[str, Any] | None = None,
    media_keys: list[str] | None = None,
    total_label: str = "Total revenue",
    max_background: int = 5,
) -> tuple[list[tuple[str, float, str]], bool]:
    """The waterfall's ``(name, value, kind)`` entries + whether any background
    block was re-anchored. Kinds: ``baseline`` / ``background`` / ``media`` /
    ``total``. Pure data prep, split out of :func:`decomposition_png` so the
    re-anchoring arithmetic is testable.

    Mean-centred regressors (z-scored controls, Fourier seasonality) sum to
    ≈ 0 over the window *by construction* — displayed raw they all read "+0".
    Given a component's per-period ``series``, its block is **re-anchored to
    the component's weakest period**: the block shows the lift above that
    floor, and the floor itself (× n periods) moves into the baseline bar — so
    the waterfall still sums exactly to the modelled total.
    """
    totals = {str(k): float(v) for k, v in component_totals.items()}
    media_set = set(map(str, media_keys)) if media_keys is not None else None

    baseline = totals.get("Baseline", 0.0)
    background: list[tuple[str, float]] = []
    media: list[tuple[str, float]] = []
    re_anchored = False
    for name, total in totals.items():
        if name == "Baseline":
            continue
        if _is_background(name, media_set):
            block = total
            s = (
                np.asarray(series.get(name), dtype=float)
                if series and name in series
                else None
            )
            if s is not None and s.size > 1 and np.all(np.isfinite(s)):
                floor = float(s.min())
                block = float((s - floor).sum())
                baseline += floor * s.size
                if abs(block - total) > 1e-9:
                    re_anchored = True
            if name.startswith("Control: "):
                pretty = name.removeprefix("Control: ").replace("_", " ").strip()
                pretty = pretty[:1].upper() + pretty[1:]
            else:
                pretty = name
            background.append((pretty, block))
        else:
            media.append((name, total))

    background.sort(key=lambda kv: -abs(kv[1]))
    if len(background) > max_background:
        head, tail = background[: max_background - 1], background[max_background - 1 :]
        background = head + [("Other controls", float(sum(v for _, v in tail)))]
    media.sort(key=lambda kv: -abs(kv[1]))

    grand_total = baseline + sum(v for _, v in background) + sum(v for _, v in media)

    entries: list[tuple[str, float, str]] = [("Baseline", baseline, "baseline")]
    entries += [(n, v, "background") for n, v in background]
    entries += [(n, v, "media") for n, v in media]
    entries += [(total_label, grand_total, "total")]
    return entries, re_anchored


def decomposition_png(
    component_totals: dict[str, float],
    *,
    series: dict[str, Any] | None = None,
    media_keys: list[str] | None = None,
    currency: str = "$",
    total_label: str = "Total revenue",
    kpi_name: str = "Revenue",
    palette: dict[str, str] | None = None,
    width: float = 8.0,
    height: float = 4.8,
    max_background: int = 5,
) -> bytes:
    """Waterfall of the KPI decomposition, matching the template's designed
    chart: tan baseline, **hatched** background blocks (seasonality, trend,
    controls — "not causally identified"), solid media blocks, dark total.
    See :func:`waterfall_entries` for the block arithmetic."""
    p = _palette(palette)
    entries, re_anchored = waterfall_entries(
        component_totals,
        series=series,
        media_keys=media_keys,
        total_label=total_label,
        max_background=max_background,
    )
    grand_total = entries[-1][1]

    fig = Figure(figsize=(width, height))
    ax = fig.subplots()
    scale = max(abs(grand_total), 1e-9)
    cum = 0.0
    media_i = 0
    y_lo, y_hi = 0.0, 0.0
    for i, (name, v, kind) in enumerate(entries):
        if kind == "baseline":
            bottom, h = 0.0, v
            cum = v
            ax.bar(
                i,
                h,
                bottom=bottom,
                width=0.62,
                color=_WF_BASELINE,
                edgecolor="#A99F87",
                linewidth=0.8,
            )
        elif kind == "total":
            bottom, h = 0.0, v
            ax.bar(
                i, h, bottom=bottom, width=0.62, color=_WF_TOTAL, edgecolor=_WF_TOTAL
            )
        elif kind == "background":
            bottom, h = (cum, v) if v >= 0 else (cum + v, -v)
            ax.bar(
                i,
                h,
                bottom=bottom,
                width=0.62,
                facecolor="white",
                edgecolor=_WF_HATCH_EDGE,
                hatch="///",
                linewidth=1.0,
            )
            cum += v
        else:
            bottom, h = (cum, v) if v >= 0 else (cum + v, -v)
            ax.bar(
                i,
                h,
                bottom=bottom,
                width=0.62,
                color=_WF_MEDIA[media_i % len(_WF_MEDIA)],
                edgecolor="white",
                linewidth=0.5,
            )
            media_i += 1
            cum += v
        top = bottom + h
        label_y = top + 0.015 * scale if v >= 0 else bottom - 0.045 * scale
        y_lo, y_hi = min(y_lo, bottom, label_y), max(y_hi, top, label_y)
        ax.text(
            i,
            label_y,
            _money(v, currency),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#3A4838",
            fontweight="bold" if kind in ("baseline", "total") else "normal",
        )

    ax.set_xticks(np.arange(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(f"Contribution ({currency})")
    ax.set_ylim(y_lo - 0.02 * scale, y_hi + 0.08 * scale)
    ax.yaxis.set_major_formatter(lambda v, _pos: _money(v, currency))
    ax.set_title(
        f"{kpi_name} decomposition — how the total builds",
        fontsize=12,
        fontweight="bold",
    )
    _style_ax(ax, p)
    if re_anchored:
        fig.subplots_adjust(bottom=0.28)
        fig.text(
            0.005,
            0.005,
            "Hatched: seasonality & controls shown as lift above each component's weakest period "
            "(their net over the window is ≈ 0 by construction) — associational, not causal.",
            fontsize=7,
            color=p["muted"],
            ha="left",
            va="bottom",
        )
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
    "_roi_axis_top",
    "saturation_zones_png",
    "roi_forest_png",
    "decomposition_png",
    "waterfall_entries",
    "fit_png",
    "reallocation_png",
]

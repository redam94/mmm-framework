"""Deterministic per-slide data engine for the MMM deck.

:func:`build_deck` turns a fitted model into an ordered list of :class:`Slide`
objects, computing every slide's numbers, tables, and chart images **directly
from the model** — no AI, no template. Each slide also carries:

* ``metrics`` — the raw numbers (for filling template placeholders), and
* ``notes`` — a deterministic context string the per-slide AI insight is
  generated from later (PR 3), plus ``is_summary`` to flag deck-level slides
  whose insight should be synthesized from the whole deck.

The breakthrough / optimal / saturation logic comes from
:func:`mmm_framework.reporting.helpers.compute_response_zones` (ROI & marginal
ROI, not percent of response).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import charts


@dataclass
class Slide:
    """One slide's fully-computed, deterministic content."""

    kind: str
    title: str
    subtitle: str = ""
    bullets: list[str] = field(default_factory=list)
    table: dict[str, Any] | None = None  # {"columns": [...], "rows": [[...], ...]}
    chart_png: bytes | None = None
    chart_caption: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""  # deterministic context the per-slide AI insight is built from
    is_summary: bool = (
        False  # deck-level slide (insight synthesized from the whole deck)
    )
    insight: str | None = None  # filled by the agentic layer (PR 3)

    def to_dict(self, *, include_chart: bool = True) -> dict[str, Any]:
        d = {
            "kind": self.kind,
            "title": self.title,
            "subtitle": self.subtitle,
            "bullets": list(self.bullets),
            "table": self.table,
            "chart_caption": self.chart_caption,
            "metrics": self.metrics,
            "notes": self.notes,
            "is_summary": self.is_summary,
            "insight": self.insight,
        }
        if include_chart:
            d["chart_png_b64"] = (
                base64.b64encode(self.chart_png).decode("ascii")
                if self.chart_png
                else None
            )
        return d


@dataclass
class Deck:
    """An ordered, fully-computed deck (pre-AI, pre-template)."""

    title: str
    slides: list[Slide]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_charts: bool = True) -> dict[str, Any]:
        return {
            "title": self.title,
            "meta": self.meta,
            "slides": [s.to_dict(include_chart=include_charts) for s in self.slides],
        }


def _fmt_money(v: float | None, currency: str = "$") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(v)
    if av >= 1e9:
        return f"{currency}{v/1e9:.1f}B"
    if av >= 1e6:
        return f"{currency}{v/1e6:.1f}M"
    if av >= 1e3:
        return f"{currency}{v/1e3:.0f}K"
    return f"{currency}{v:,.0f}"


def _fmt_x(v: float | None) -> str:
    return "—" if (v is None or not np.isfinite(v)) else f"{v:.2f}x"


_ZONE_LABEL = {
    "breakthrough": "Breakthrough (under-invested)",
    "optimal": "Optimal",
    "saturation": "Saturation (over-invested)",
}
_REC_VERB = {"increase": "scale up", "hold": "hold", "reduce": "reduce"}


def build_deck(
    model: Any,
    results: Any = None,
    *,
    title: str | None = None,
    client: str | None = None,
    kpi_name: str = "KPI",
    currency: str = "$",
    break_even: float = 1.0,
    margin: float | None = None,
    channels: list[str] | None = None,
    hdi_prob: float = 0.94,
    per_channel_saturation: bool = True,
    palette: dict[str, str] | None = None,
) -> Deck:
    """Build the deterministic :class:`Deck` from a fitted model.

    Parameters
    ----------
    model, results
        The fitted MMM (and optional results object).
    title, client, kpi_name, currency
        Deck framing.
    break_even, margin
        Marginal-ROI break-even target. ``margin`` (gross margin, 0–1) overrides
        ``break_even`` with ``1/margin`` so the optimum is the *profit*-maximizing
        spend when the KPI is revenue.
    channels
        Channels to cover (default: all).
    per_channel_saturation
        One saturation/zone slide per channel (default) vs. a single combined slide.
    """
    from ..helpers import compute_response_zones, compute_roi_with_uncertainty

    eff_be = (1.0 / float(margin)) if margin else float(break_even)
    chan = list(channels or list(getattr(model, "channel_names", []) or []))

    # --- pull model-derived numbers (each best-effort; a missing piece just
    #     drops its slide rather than failing the whole deck) ---
    roi_df = None
    try:
        roi_df = compute_roi_with_uncertainty(model, hdi_prob=hdi_prob)
    except Exception:
        roi_df = None

    zones = {}
    try:
        zones = compute_response_zones(
            model, chan or None, break_even=eff_be, hdi_prob=hdi_prob
        )
    except Exception:
        zones = {}

    bundle = None
    try:
        from ..extractors import create_extractor

        bundle = create_extractor(model, results=results).extract()
    except Exception:
        bundle = None

    meta = {
        "client": client,
        "kpi": kpi_name,
        "currency": currency,
        "break_even": eff_be,
        "margin": margin,
        "channels": chan,
        "n_channels": len(chan),
    }
    deck_title = (
        title or (f"{client} — " if client else "") + "Marketing Mix Modeling Results"
    )
    slides: list[Slide] = []

    # ---- 1. Title (summary) ----
    slides.append(
        Slide(
            kind="title",
            title=deck_title,
            subtitle=client or "Bayesian Marketing Mix Modeling",
            is_summary=True,
            metrics={"client": client, "kpi": kpi_name, "n_channels": len(chan)},
            notes=(
                "Title slide. The headline should frame the single most important, "
                "decision-relevant takeaway across the whole deck (ROI leaders, "
                "where to reallocate)."
            ),
        )
    )

    # ---- 2. Executive summary (summary) ----
    roi_records = (
        roi_df.to_dict("records") if roi_df is not None and len(roi_df) else []
    )
    total_contrib = float(sum(r.get("contribution_mean", 0.0) for r in roi_records))
    total_spend = float(sum(r.get("spend", 0.0) for r in roi_records))
    top = (
        max(roi_records, key=lambda r: r.get("roi_mean", 0.0)) if roi_records else None
    )
    r2 = None
    if bundle is not None and getattr(bundle, "fit_statistics", None):
        r2 = bundle.fit_statistics.get("r2")
    zone_counts = {"breakthrough": 0, "optimal": 0, "saturation": 0}
    for z in zones.values():
        zone_counts[z.current_zone] = zone_counts.get(z.current_zone, 0) + 1

    exec_bullets = [
        f"{len(chan)} media channels modeled on {kpi_name}.",
        f"Total modeled media contribution: {_fmt_money(total_contrib, currency)} "
        f"on {_fmt_money(total_spend, currency)} spend.",
    ]
    if top:
        exec_bullets.append(
            f"Most efficient channel: {top['channel']} at {_fmt_x(top.get('roi_mean'))} ROI."
        )
    if any(zone_counts.values()):
        exec_bullets.append(
            f"{zone_counts['breakthrough']} channel(s) under-invested, "
            f"{zone_counts['optimal']} near optimal, {zone_counts['saturation']} saturated "
            f"(at a {eff_be:g} marginal-ROI break-even)."
        )
    if r2 is not None and np.isfinite(r2):
        exec_bullets.append(f"Model explains {r2:.0%} of KPI variance (R²).")

    slides.append(
        Slide(
            kind="executive_summary",
            title="Executive summary",
            bullets=exec_bullets,
            metrics={
                "total_contribution": total_contrib,
                "total_spend": total_spend,
                "n_channels": len(chan),
                "top_channel": top["channel"] if top else None,
                "top_roi": top.get("roi_mean") if top else None,
                "r2": r2,
                "zone_counts": zone_counts,
                "break_even": eff_be,
            },
            is_summary=True,
            notes=(
                "Executive summary. Synthesize the deck's headline finding and the "
                "1–2 highest-value actions (which channels to scale up vs. pull back) "
                "from the per-channel ROI and the breakthrough/optimal/saturation zones."
            ),
        )
    )

    # ---- 3. Model fit ----
    if (
        bundle is not None
        and getattr(bundle, "actual", None) is not None
        and getattr(bundle, "predicted", None)
    ):
        try:
            png = charts.fit_png(
                getattr(bundle, "dates", None),
                bundle.actual,
                bundle.predicted,
                r2=r2,
                palette=palette,
            )
            fs = bundle.fit_statistics or {}
            fit_bullets = []
            if fs.get("r2") is not None:
                fit_bullets.append(f"R² = {fs['r2']:.2f}")
            if fs.get("mape") is not None:
                fit_bullets.append(f"MAPE = {fs['mape']:.1%}")
            slides.append(
                Slide(
                    kind="model_fit",
                    title="Model fit",
                    bullets=fit_bullets,
                    chart_png=png,
                    chart_caption="Actual vs. predicted KPI with predictive interval.",
                    metrics={"fit_statistics": fs},
                    notes=(
                        "Model fit / validation. Comment on how well the model tracks "
                        "the KPI and whether the fit supports trusting the decomposition."
                    ),
                )
            )
        except Exception:
            pass

    # ---- 4. Decomposition ----
    comp = getattr(bundle, "component_totals", None) if bundle is not None else None
    if comp:
        try:
            png = charts.decomposition_png(comp, palette=palette)
            total = float(sum(abs(v) for v in comp.values())) or 1.0
            rows = [
                [k, _fmt_money(v, currency), f"{v/total:+.0%}"]
                for k, v in sorted(comp.items(), key=lambda kv: -kv[1])
            ]
            slides.append(
                Slide(
                    kind="decomposition",
                    title="KPI decomposition",
                    table={
                        "columns": ["Component", "Contribution", "Share"],
                        "rows": rows,
                    },
                    chart_png=png,
                    chart_caption="Total contribution by component.",
                    metrics={"component_totals": comp},
                    notes=(
                        "KPI decomposition. Call out the base vs. media split and which "
                        "components drive the outcome."
                    ),
                )
            )
        except Exception:
            pass

    # ---- 5. Channel ROI ----
    if roi_records:
        channel_roi = {
            r["channel"]: {
                "mean": r.get("roi_mean", 0.0),
                "lower": r.get("roi_hdi_low", r.get("roi_mean", 0.0)),
                "upper": r.get("roi_hdi_high", r.get("roi_mean", 0.0)),
            }
            for r in roi_records
        }
        try:
            png = charts.roi_forest_png(channel_roi, break_even=eff_be, palette=palette)
        except Exception:
            png = None
        rows = [
            [
                r["channel"],
                _fmt_money(r.get("spend"), currency),
                _fmt_x(r.get("roi_mean")),
                f"[{r.get('roi_hdi_low', float('nan')):.2f}, {r.get('roi_hdi_high', float('nan')):.2f}]",
                f"{r.get('prob_profitable', float('nan')):.0%}",
            ]
            for r in sorted(roi_records, key=lambda r: -r.get("roi_mean", 0.0))
        ]
        slides.append(
            Slide(
                kind="channel_roi",
                title="Return on investment by channel",
                table={
                    "columns": ["Channel", "Spend", "ROI", "94% HDI", "P(ROI>1)"],
                    "rows": rows,
                },
                chart_png=png,
                chart_caption="ROI point estimate and 94% credible interval per channel.",
                metrics={"channel_roi": channel_roi},
                notes=(
                    "Channel ROI. Identify the most and least efficient channels and "
                    "whether their intervals exclude the break-even line."
                ),
            )
        )

    # ---- 6. Saturation / spend zones (per channel) ----
    sat_zone_channels = list(zones.keys()) if per_channel_saturation else []
    for ch in sat_zone_channels:
        z = zones[ch]
        try:
            png = charts.saturation_zones_png(z, currency=currency, palette=palette)
        except Exception:
            png = None

        def _rng(r):
            return f"{_fmt_money(r[0], currency)} – {_fmt_money(r[1], currency)}"

        table = {
            "columns": ["Metric", "Value"],
            "rows": [
                ["Current spend / period", _fmt_money(z.current_spend, currency)],
                ["Current zone", _ZONE_LABEL.get(z.current_zone, z.current_zone)],
                ["Current ROI", _fmt_x(z.current_roi)],
                ["Current marginal ROI", _fmt_x(z.current_mroi)],
                [
                    "Optimal spend (mROI = break-even)",
                    _fmt_money(z.optimal_spend, currency),
                ],
                ["Breakthrough range", _rng(z.breakthrough_range)],
                ["Optimal range", _rng(z.optimal_range)],
                ["Saturation range", _rng(z.saturation_range)],
            ],
        }
        head = (
            f"scale up — strong marginal returns"
            if z.recommendation == "increase"
            else (
                "hold near current spend"
                if z.recommendation == "hold"
                else "reduce / reallocate"
            )
        )
        bullets = [
            f"Current spend sits in the {_ZONE_LABEL.get(z.current_zone, z.current_zone)} zone.",
            f"Marginal ROI at current spend: {_fmt_x(z.current_mroi)} "
            f"(break-even {eff_be:g}).",
        ]
        if z.optimal_spend is not None and z.headroom_to_optimal is not None:
            direction = "more" if z.headroom_to_optimal > 0 else "less"
            bullets.append(
                f"Profit-maximizing spend ≈ {_fmt_money(z.optimal_spend, currency)} "
                f"({_fmt_money(abs(z.headroom_to_optimal), currency)} {direction} per period)."
            )
        slides.append(
            Slide(
                kind="saturation",
                title=f"{ch}: spend efficiency & headroom",
                subtitle=f"Recommendation: {head}",
                bullets=bullets,
                table=table,
                chart_png=png,
                chart_caption=(
                    "Response curve with breakthrough/optimal/saturation zones "
                    "(marginal-ROI break-even bands) and the ROI / marginal-ROI overlay."
                ),
                metrics={"zones": z.to_dict()},
                notes=(
                    f"Saturation/headroom for {ch}. It is in the '{z.current_zone}' zone "
                    f"with current marginal ROI {z.current_mroi:.2f} vs a {eff_be:g} "
                    f"break-even; the deterministic recommendation is to "
                    f"{_REC_VERB.get(z.recommendation, z.recommendation)}. Turn this into a "
                    "specific, quantified spend recommendation for this channel."
                ),
            )
        )

    # ---- 7. Optimization summary (reallocation; summary) ----
    if zones:
        rows_chart = [
            {"channel": ch, "current": z.current_spend, "optimal": z.optimal_spend}
            for ch, z in zones.items()
        ]
        try:
            png = charts.reallocation_png(
                rows_chart, currency=currency, palette=palette
            )
        except Exception:
            png = None
        table_rows = []
        for ch, z in zones.items():
            hr = z.headroom_to_optimal
            table_rows.append(
                [
                    ch,
                    _fmt_money(z.current_spend, currency),
                    _fmt_money(z.optimal_spend, currency),
                    (
                        "—"
                        if hr is None
                        else f"{'+' if hr > 0 else ''}{_fmt_money(hr, currency)}"
                    ),
                    _REC_VERB.get(z.recommendation, z.recommendation),
                    _ZONE_LABEL.get(z.current_zone, z.current_zone),
                ]
            )
        n_up = sum(1 for z in zones.values() if z.recommendation == "increase")
        n_down = sum(1 for z in zones.values() if z.recommendation == "reduce")
        slides.append(
            Slide(
                kind="optimization",
                title="Where to reallocate",
                bullets=[
                    f"{n_up} channel(s) under-invested (scale up), "
                    f"{n_down} saturated (pull back).",
                    f"Optimum = spend where marginal ROI reaches the {eff_be:g} break-even.",
                ],
                table={
                    "columns": [
                        "Channel",
                        "Current",
                        "Optimal",
                        "Headroom",
                        "Action",
                        "Zone",
                    ],
                    "rows": table_rows,
                },
                chart_png=png,
                chart_caption="Current vs. profit-maximizing spend per channel.",
                metrics={
                    "reallocation": table_rows,
                    "n_underinvested": n_up,
                    "n_saturated": n_down,
                },
                is_summary=True,
                notes=(
                    "Portfolio reallocation. Synthesize a budget-neutral move across the "
                    "whole deck: pull from saturated channels, push into under-invested "
                    "ones, and state the expected direction of impact."
                ),
            )
        )

    # ---- 8. Methodology ----
    spec = getattr(bundle, "model_specification", None) if bundle is not None else None
    method_bullets = [
        f"KPI: {kpi_name}; {len(chan)} media channels.",
        "Bayesian MMM with adstock (carryover) and saturation (diminishing returns); "
        "all figures carry posterior credible intervals.",
        f"Spend zones defined on marginal ROI vs a {eff_be:g} break-even"
        + (f" (gross margin {margin:.0%})" if margin else "")
        + " — not percent of response.",
    ]
    slides.append(
        Slide(
            kind="methodology",
            title="Methodology & assumptions",
            bullets=method_bullets,
            metrics={
                "model_specification": spec,
                "break_even": eff_be,
                "margin": margin,
            },
            notes=(
                "Methodology. Keep it brief and credibility-building; note the causal "
                "assumptions and that intervals reflect genuine uncertainty."
            ),
        )
    )

    return Deck(title=deck_title, slides=slides, meta=meta)


__all__ = ["Deck", "Slide", "build_deck"]

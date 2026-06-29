"""Fill the designed PowerPoint template from a fitted MMM — model numbers and
charts go straight into the template's shapes; no AI in this layer.

The template (bundled at ``reporting/deck/templates/report_template.pptx``; see
:func:`default_template_path`) is a finished 24-slide readout. :func:`build_pptx`
reads its slides by stable label text + geometry and
fills the data-bearing ones: the headline KPI cards, the channel scorecard, the
ROI / next-dollar (marginal-ROI) charts, the decomposition, and the per-channel
deep-dives (each with the breakthrough/optimal/saturation zone chart from
:func:`mmm_framework.reporting.helpers.compute_response_zones`). It uses **80%
ranges** to match the template, fills the existing channel rows/slides up to the
model's channel count, and trims the rest.

Per-slide AI insights and the whole-deck synthesis (PR 3) are injected via the
optional ``insights`` map; everything here is deterministic.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from pptx.util import Inches

from . import charts, template as T


def default_template_path() -> Path:
    """Path to the bundled deck template.

    Resolution order: the ``MMM_DECK_TEMPLATE`` env override, then the template
    packaged with the library. The agent/API pass an explicit path; this is the
    default used by tools and tests.
    """
    import os

    env = os.environ.get("MMM_DECK_TEMPLATE")
    if env and Path(env).exists():
        return Path(env)
    return Path(__file__).parent / "templates" / "report_template.pptx"


def _pct(samples: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    lo = float(np.percentile(samples, (1 - hdi_prob) / 2 * 100))
    hi = float(np.percentile(samples, (1 + hdi_prob) / 2 * 100))
    return lo, hi


def _money(v: float | None, currency: str = "$") -> str:
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


def _read_action(mean: float, lo: float, hi: float, be: float) -> tuple[str, str]:
    """The template's READ / ACTION vocabulary, from the ROI credible interval vs
    the break-even line."""
    if lo > be:
        return "Confidently profitable", "Scale"
    if hi < be:
        return "Below break-even", "Reduce"
    if (hi - lo) > 0.9 * be and mean > 0.9 * be:
        return "High upside, unproven", "Test"
    return "Near break-even", "Hold"


def _half_life_weeks(model: Any, channel: str) -> float | None:
    try:
        from ..helpers import _get_adstock_alpha
        from ..helpers.utils import _get_posterior

        alpha = _get_adstock_alpha(model, _get_posterior(model), channel)
        alpha = float(np.mean(alpha)) if alpha is not None else None
        if alpha is None or not (0 < alpha < 1):
            return None
        return math.log(0.5) / math.log(alpha)
    except Exception:
        return None


def _portfolio_metrics(
    model: Any, roi_records: list[dict], total_revenue: float | None, hdi_prob: float
) -> dict[str, Any]:
    """Total marketing-attributed revenue, share of revenue, and blended return
    per $1 — point + 80% range, from posterior portfolio-contribution draws."""
    total_spend = float(sum(r.get("spend", 0.0) for r in roi_records))
    portfolio = None
    try:
        cc = model.sample_channel_contributions(max_draws=300)  # (draws, obs, channel)
        portfolio = np.asarray(cc).sum(axis=(1, 2))  # per-draw total media contribution
    except Exception:
        portfolio = None

    out: dict[str, Any] = {"total_spend": total_spend}
    if portfolio is not None and portfolio.size:
        rev_mean = float(portfolio.mean())
        rev_lo, rev_hi = _pct(portfolio, hdi_prob)
        out["revenue"] = (rev_mean, rev_lo, rev_hi)
        if total_spend > 0:
            roi = portfolio / total_spend
            out["blended_roi"] = (float(roi.mean()), *_pct(roi, hdi_prob))
        if total_revenue and total_revenue > 0:
            sh = portfolio / total_revenue
            out["share"] = (float(sh.mean()), *_pct(sh, hdi_prob))
    else:  # point-only fallback
        rev_mean = float(sum(r.get("contribution_mean", 0.0) for r in roi_records))
        out["revenue"] = (rev_mean, float("nan"), float("nan"))
        if total_spend > 0:
            out["blended_roi"] = (rev_mean / total_spend, float("nan"), float("nan"))
        if total_revenue:
            out["share"] = (rev_mean / total_revenue, float("nan"), float("nan"))
    return out


def _cluster_rows(shapes, gap_in: float = 0.3) -> list[list]:
    """Group shapes into visual rows by their top coordinate."""
    from pptx.util import Inches

    if not shapes:
        return []
    shapes = sorted(shapes, key=lambda s: T._emu(s.top))
    gap = int(Inches(gap_in))
    rows, cur, last = [], [shapes[0]], T._emu(shapes[0].top)
    for sh in shapes[1:]:
        t = T._emu(sh.top)
        if t - last > gap:
            rows.append(cur)
            cur = []
        cur.append(sh)
        last = t
    rows.append(cur)
    return rows


def _fill_scorecard(slide, rows: list[dict], currency: str, be: float) -> None:
    """Fill the channel scorecard (positional: columns by header left, rows by
    top), filling one model channel per template row and blanking the rest."""
    from pptx.util import Inches

    cols = {}
    for key, label in (
        ("channel", "CHANNEL"),
        ("spend", "SPEND"),
        ("return", "RETURN / $1"),
        ("range", "80% RANGE"),
        ("read", "READ"),
        ("action", "ACTION"),
    ):
        sh = T.find_by_label(slide, label)
        if sh is not None:
            cols[key] = T._emu(sh.left)
    head = T.find_by_label(slide, "CHANNEL")
    if head is None or "channel" not in cols:
        return
    header_top = T._emu(head.top)
    tol = int(Inches(0.6))

    # data text shapes below the header, excluding the full-width footer note
    data = [
        sh
        for sh in T.iter_text_shapes(slide)
        if T._emu(sh.top) > header_top + int(Inches(0.3))
        and T._emu(sh.width) < int(Inches(12))
    ]
    row_groups = _cluster_rows(data)

    for i, group in enumerate(row_groups):
        # assign each shape in the row to its nearest column
        cell = {}
        for sh in group:
            left = T._emu(sh.left)
            best = min(cols, key=lambda k: abs(cols[k] - left))
            if abs(cols[best] - left) <= tol and best not in cell:
                cell[best] = sh
        if i < len(rows):
            r = rows[i]
            mean, lo, hi = r["roi_mean"], r["roi_lo"], r["roi_hi"]
            read, action = _read_action(mean, lo, hi, be)
            if "channel" in cell:
                T.set_text(cell["channel"], r["channel"])
            if "spend" in cell:
                T.set_text(cell["spend"], _money(r["spend"], currency))
            if "return" in cell:
                T.set_text(cell["return"], f"{mean:.2f}")
            if "range" in cell:
                T.set_text(cell["range"], f"{lo:.2f} – {hi:.2f}")
            if "read" in cell:
                T.set_text(cell["read"], read)
            if "action" in cell:
                T.set_text(cell["action"], action)
        else:  # blank the template's extra rows
            for sh in group:
                T.set_text(sh, "")


def _fill_channel_slide(
    slide, r: dict, z: Any, currency: str, be: float, narrative: str | None
) -> None:
    """Fill one per-channel deep-dive slide (channel name, action pill, the five
    metric cards, the optional AI narrative, and the saturation/zone chart)."""
    # channel name = the prominent text near the top-left
    for sh in T.iter_text_shapes(slide):
        if T._emu(sh.top) < int(Inches(1.2)) and T._emu(sh.left) < int(Inches(2.0)):
            T.set_text(sh, r["channel"])
            break

    _, action = _read_action(r["roi_mean"], r["roi_lo"], r["roi_hi"], be)
    for sh in T.iter_text_shapes(slide):  # action pill (Scale/Test/Hold/Reduce)
        if T._norm(sh.text_frame.text) in ("scale", "test", "hold", "reduce"):
            T.set_text(sh, action)
            break

    mroi = z.current_mroi if z is not None else float("nan")
    T.fill_card(
        slide,
        "RETURN / $1",
        f"{r['roi_mean']:.2f}",
        f"80% {r['roi_lo']:.2f}–{r['roi_hi']:.2f}",
    )
    T.fill_card(slide, "CONTRIBUTION", _money(r["contribution"], currency))
    T.fill_card(slide, "SPEND", _money(r["spend"], currency))
    T.fill_card(
        slide,
        "MARGINAL / $1",
        "—" if not np.isfinite(mroi) else f"{mroi:.2f}",
        (
            "clears break-even"
            if (np.isfinite(mroi) and mroi >= be)
            else "below break-even"
        ),
    )
    hl = r.get("half_life")
    T.fill_card(
        slide,
        "CARRYOVER HALF-LIFE",
        "—" if hl is None else f"{hl:.1f}w",
        (
            "fast decay"
            if (hl or 0) < 1.5
            else "slow decay" if (hl or 0) > 4 else "medium decay"
        ),
    )

    # optional AI narrative (PR 3): the wide standfirst near the top
    if narrative:
        for sh in T.iter_text_shapes(slide):
            if T._emu(sh.top) < int(Inches(2.2)) and T._emu(sh.width) > int(Inches(8)):
                T.set_text(sh, narrative)
                break

    # the saturation/zone chart on the left panel — rendered to fit the box's
    # exact aspect ratio (no squish).
    if z is not None:
        try:
            T.replace_image_fit(
                slide,
                lambda w, h: charts.saturation_zones_png(
                    z, currency=currency, width=w, height=h
                ),
                match=T.pictures_in_region(slide, Inches(1.05), Inches(4.43), 0, 0),
            )
        except Exception:
            pass


def build_pptx(
    model: Any,
    *,
    template_path: str | Path | None = None,
    out_path: str | Path | None = None,
    deck: Any = None,
    client: str | None = None,
    kpi_name: str = "Revenue",
    currency: str = "$",
    break_even: float = 1.0,
    margin: float | None = None,
    hdi_prob: float = 0.8,
    max_channels: int = 7,
    insights: dict[str, str] | None = None,
) -> bytes:
    """Fill the template deck from a fitted model and return the .pptx bytes
    (also written to ``out_path`` if given).

    ``hdi_prob`` defaults to 0.8 to match the template's "80% range". ``margin``
    sets a profit-maximizing break-even (1/margin). ``insights`` (PR 3) maps slide
    keys to AI narrative text; omitted here.
    """
    from pptx import Presentation
    from pptx.util import Inches

    from ..helpers import compute_response_zones, compute_roi_with_uncertainty

    template_path = template_path or default_template_path()
    eff_be = (1.0 / float(margin)) if margin else float(break_even)

    roi_df = compute_roi_with_uncertainty(model, hdi_prob=hdi_prob)
    roi_records = (
        roi_df.to_dict("records") if roi_df is not None and len(roi_df) else []
    )
    zones = {}
    try:
        zones = compute_response_zones(model, break_even=eff_be, hdi_prob=hdi_prob)
    except Exception:
        zones = {}

    bundle = None
    total_revenue = None
    try:
        from ..extractors import create_extractor

        bundle = create_extractor(model).extract()
        if getattr(bundle, "actual", None) is not None:
            total_revenue = float(np.asarray(bundle.actual).sum())
    except Exception:
        bundle = None

    # per-channel rows (sorted by action priority then ROI)
    _ORDER = {"Scale": 0, "Test": 1, "Hold": 2, "Reduce": 3}
    rows = []
    for r in roi_records:
        ch = r["channel"]
        lo = r.get("roi_hdi_low", r.get("roi_mean", 0.0))
        hi = r.get("roi_hdi_high", r.get("roi_mean", 0.0))
        _, action = _read_action(r.get("roi_mean", 0.0), lo, hi, eff_be)
        rows.append(
            {
                "channel": ch,
                "spend": r.get("spend", 0.0),
                "contribution": r.get("contribution_mean", 0.0),
                "roi_mean": r.get("roi_mean", 0.0),
                "roi_lo": lo,
                "roi_hi": hi,
                "action": action,
                "half_life": _half_life_weeks(model, ch),
            }
        )
    rows.sort(key=lambda r: (_ORDER.get(r["action"], 9), -r["roi_mean"]))
    rows = rows[:max_channels]

    pf = _portfolio_metrics(model, roi_records, total_revenue, hdi_prob)
    insights = insights or {}

    prs = Presentation(str(template_path))
    slides = list(prs.slides)

    # ---- S0: cover ----
    if len(slides) > 0:
        s = slides[0]
        sub = T.find_by_prefix(s, "Prepared for")
        if sub is not None and client:
            T.set_text(sub, f"Prepared for {client} · Planning & analytics")

    # ---- S1: headline KPI cards ----
    if len(slides) > 1:
        s = slides[1]
        if "revenue" in pf:
            m, lo, hi = pf["revenue"]
            rng = (
                ""
                if not np.isfinite(lo)
                else f"80% range {_money(lo, currency)} – {_money(hi, currency)}"
            )
            T.fill_card(s, "MARKETING-ATTRIBUTED REVENUE", _money(m, currency), rng)
        if "share" in pf:
            m, lo, hi = pf["share"]
            rng = "" if not np.isfinite(lo) else f"80% range {lo:.1%} – {hi:.1%}"
            T.fill_card(s, "SHARE OF TOTAL REVENUE", f"{m:.1%}", rng)
        if "blended_roi" in pf:
            m, lo, hi = pf[
                "blended_roi"
            ]  # a ratio ($ returned per $1) — 2 decimals, not abbreviated
            rng = (
                ""
                if not np.isfinite(lo)
                else f"80% range {currency}{lo:.2f} – {currency}{hi:.2f}"
            )
            T.fill_card(s, "BLENDED RETURN PER $1", f"{currency}{m:.2f}", rng)
        if "headline" in insights or "standfirst" in insights:
            h = T.find_by_label(s, "THE HEADLINE")
            if h is not None:
                # below the eyebrow: [0] = the big title, [1] = the standfirst para
                below = T.shapes_below(s, h, left_tol_in=2.0, max_n=2)
                if below and insights.get("headline"):
                    T.set_text(below[0], insights["headline"])
                if len(below) > 1 and insights.get("standfirst"):
                    T.set_text(below[1], insights["standfirst"])

    # ---- S5: channel scorecard ----
    sc = next((s for s in slides if T.find_by_label(s, "CHANNEL SCORECARD")), None)
    if sc is not None:
        _fill_scorecard(sc, rows, currency, eff_be)

    # ---- S6 / S7: ROI and marginal-ROI forests ----
    roi_dict = {
        r["channel"]: {
            "mean": r["roi_mean"],
            "lower": r["roi_lo"],
            "upper": r["roi_hi"],
        }
        for r in rows
    }
    s6 = next((s for s in slides if T.find_by_label(s, "RETURN & UNCERTAINTY")), None)
    if s6 is not None and roi_dict:
        try:
            T.replace_image_fit(
                s6,
                lambda w, h: charts.roi_forest_png(
                    roi_dict, break_even=eff_be, width=w, height=h
                ),
            )
        except Exception:
            pass
    s7 = next((s for s in slides if T.find_by_label(s, "THE NEXT DOLLAR")), None)
    if s7 is not None and zones:
        mroi_dict = {
            ch: {
                "mean": z.current_mroi,
                "lower": z.current_mroi_hdi[0],
                "upper": z.current_mroi_hdi[1],
            }
            for ch, z in zones.items()
        }
        try:
            T.replace_image_fit(
                s7,
                lambda w, h: charts.roi_forest_png(
                    mroi_dict,
                    break_even=eff_be,
                    width=w,
                    height=h,
                    title="Marginal return on the next dollar",
                    xlabel="Marginal ROI (next-dollar return)",
                ),
            )
        except Exception:
            pass

    # ---- S4: decomposition ----
    if bundle is not None and getattr(bundle, "component_totals", None):
        s4 = next(
            (
                s
                for s in slides
                if T.find_by_prefix(s, "The full revenue decomposition")
            ),
            None,
        )
        if s4 is not None:
            comp = bundle.component_totals
            try:
                T.replace_image_fit(
                    s4, lambda w, h: charts.decomposition_png(comp, width=w, height=h)
                )
            except Exception:
                pass

    # ---- S12-18: per-channel deep-dive slides (fill N, delete extras) ----
    deep = [
        (idx, s)
        for idx, s in enumerate(slides)
        if T.find_by_label(s, "RETURN / $1") is not None
        and T.find_by_label(s, "CARRYOVER HALF-LIFE") is not None
    ]
    used_idx = set()
    for j, (idx, s) in enumerate(deep):
        if j < len(rows):
            r = rows[j]
            _fill_channel_slide(
                s,
                r,
                zones.get(r["channel"]),
                currency,
                eff_be,
                insights.get(f"channel:{r['channel']}"),
            )
            used_idx.add(idx)
    # delete the unused deep-dive slides (highest index first to keep indices valid)
    for idx, _ in sorted(deep, key=lambda t: -t[0]):
        if idx not in used_idx:
            T.delete_slide(prs, idx)

    # Clear the template's cached shrink-to-fit scale on every text box so
    # PowerPoint recomputes the fit for the actual (possibly substituted) font —
    # fixes the one-character text wrapping on filled AND untouched slides.
    for s in prs.slides:
        T.clear_autofit_scale(s)

    buf = io.BytesIO()
    prs.save(buf)
    data = buf.getvalue()
    if out_path is not None:
        Path(out_path).write_bytes(data)
    return data


__all__ = ["build_pptx", "default_template_path"]

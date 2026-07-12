"""Section renderers for the Augur "Media Performance Readout".

These produce the editorial, evidence-coded layout of the Augur template — a
masthead headline with a KPI strip and recommendations, a channel scorecard with
Scale/Test/Hold/Reduce tiers, ROI-with-uncertainty, the marginal (next-dollar)
return, saturation, the reallocation, per-channel deep dives, carryover, a
posterior-predictive *fit-over-time* + *checks* block, an evidence legend,
recommended tests and next steps.

Each section is a normal :class:`~mmm_framework.reporting.sections.Section`
(same ``data``/``config``/``section_config`` contract and ``render() -> str``),
so they slot straight into ``MMMReportGenerator``. They reuse the existing chart
kit (auto-themed by the Augur ``ColorScheme``) and the shared tier classifier,
and they read CMO/planner narrative from ``config.cmo_insights`` (filled by
``reporting.insights.build_report_insights``). Every section no-ops to an empty
string when its bundle data is missing, so a partial model still renders.
"""

from __future__ import annotations

import html
import re

import numpy as np

from . import charts
from .config import ChartConfig
from .evidence import EvidenceTier, evidence_chip_html
from .helpers.reallocation import (
    TIER_META,
    channel_rows,
    illustrative_flighting_totals,
    reallocation_groups,
    test_candidates,
)
from .sections import Section


class AugurSection(Section):
    """Base for Augur sections: an eyebrow + serif title + content wrapper, and
    convenience accessors for tier rows, insights and number formatting."""

    eyebrow: str = ""

    # cache the classified channel rows once per section instance
    _rows_cache: list | None = None

    # ── shared accessors ────────────────────────────────────────────────────
    def rows(self) -> list[dict]:
        if self._rows_cache is None:
            try:
                self._rows_cache = channel_rows(self.data)
            except Exception:
                self._rows_cache = []
        return self._rows_cache

    def insight(self, key: str, default: str = "") -> str:
        return (self.config.cmo_insights or {}).get(key, default)

    def _evidence(self, name: str) -> dict | None:
        """The evidence annotation (tier + identifiability) for a channel, or
        ``None`` when the extractor did not attach it (issue #102)."""
        ev = getattr(self.data, "channel_evidence", None) or {}
        return ev.get(name)

    def _evidence_chip(self, name: str) -> str:
        """Augur-themed evidence chip for a channel (empty when unavailable)."""
        return evidence_chip_html(self._evidence(name), theme="augur")

    def _ratio(self, v: float | None) -> str:
        cur = self.config.currency_symbol
        if v is None or not np.isfinite(v):
            return "—"
        return f"{cur}{v:.2f}"

    def _money(self, v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "—"
        return self.config.format_currency(v)

    def _pct(self, v: float | None) -> str:
        if v is None or not np.isfinite(v):
            return "—"
        return f"{v * 100:.1f}%"

    def _ch(self, name: str) -> str:
        """Escaped channel name (post-processed to display form by the generator
        when ``format_channel_names`` is on)."""
        return html.escape(str(name))

    @staticmethod
    def _slug(name: str) -> str:
        """A selector-safe id fragment (lowercase alphanumerics + dashes).

        Sanitize BEFORE use as an HTML id — never ``html.escape`` an id, which
        would inject ``&amp;`` entities and break CSS/JS selectors."""
        s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
        return s or "ch"

    @staticmethod
    def _has_curve(sat: dict, ch: str) -> bool:
        """True when ``sat[ch]`` is a usable ``{"spend", "response"}`` record."""
        rec = sat.get(ch) if isinstance(sat, dict) else None
        return (
            isinstance(rec, dict)
            and rec.get("spend") is not None
            and rec.get("response") is not None
        )

    def _t(self, text: str) -> str:
        """Escape model/templated narrative text before injecting into HTML."""
        return html.escape(str(text)) if text else ""

    def _wrap(self, content: str, *, illustrative: bool = False) -> str:
        eyebrow = (
            f'<div class="section-eyebrow">{html.escape(self.eyebrow)}</div>'
            if self.eyebrow
            else ""
        )
        tag = (
            '<span class="illus-tag">Illustrative weekly pattern</span>'
            if illustrative
            else ""
        )
        return f"""
        <section class="section" id="{self.section_id}">
            {eyebrow}
            <h2>{html.escape(self.title)}</h2>
            {tag}
            {content}
        </section>
        """

    @property
    def title(self) -> str:  # default title overridable by section_config
        return self.section_config.title or self.default_title


# ─────────────────────────────────────────────────────────────────────────────
# 01 — The headline
# ─────────────────────────────────────────────────────────────────────────────
class AugurHeadlineSection(AugurSection):
    section_id = "summary"
    default_title = "The headline"
    eyebrow = "The headline"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        headline = self.insight("headline")
        standfirst = self.insight("standfirst")
        kpis = self._kpi_strip()
        recs = self._recommendations()
        if not (headline or standfirst or kpis or recs):
            return ""

        lede = f'<p class="lede">{self._t(standfirst)}</p>' if standfirst else ""
        # The headline insight replaces the generic section title when present.
        title = html.escape(headline) if headline else html.escape(self.title)
        eyebrow = f'<div class="section-eyebrow">{html.escape(self.eyebrow)}</div>'
        caveat = self._caveat_banner()
        return f"""
        <section class="section" id="{self.section_id}">
            {eyebrow}
            <h2>{title}</h2>
            {caveat}
            {lede}
            {kpis}
            {recs}
        </section>
        """

    def _caveat_banner(self) -> str:
        """A client-facing stop sign when the model was fit APPROXIMATELY (MAP /
        ADVI / Pathfinder — uncertainty not calibrated) or did NOT converge.

        The Augur deck is otherwise a confident client readout with no
        diagnostics section, so without this a MAP/non-converged model reads as
        trustworthy. Kept short and plain-language for a non-technical audience.
        """
        diag = getattr(self.data, "diagnostics", None) or {}
        try:
            from ..diagnostics.convergence import is_converged
        except Exception:  # noqa: BLE001
            is_converged = None  # type: ignore
        approx = bool(diag.get("approximate"))
        not_conv = is_converged is not None and is_converged(diag) is False
        if not (approx or not_conv):
            return ""
        if approx:
            method = html.escape(str(diag.get("fit_method") or "approximate")).upper()
            msg = (
                f"These figures come from a fast <strong>approximate</strong> fit "
                f"({method}), not full modelling — the ranges shown are indicative and "
                f"<strong>not calibrated</strong>. Re-run the full fit before committing "
                f"budget."
            )
        else:
            msg = (
                "This model did not pass its statistical convergence checks, so the "
                "ranges shown are <strong>not reliable</strong>. Treat the numbers as "
                "provisional and re-fit before acting."
            )
        return (
            '<div class="callout" style="border-left:4px solid #b45309;'
            "background:#fbf3e4;color:#5c3d00;padding:12px 16px;border-radius:8px;"
            'margin:8px 0 4px;font-size:0.92em;">'
            f"⚠️ {msg}</div>"
        )

    def _kpi_strip(self) -> str:
        ci = int((self.section_config.credible_interval or 0.8) * 100)
        cards: list[str] = []

        rev = getattr(self.data, "marketing_attributed_revenue", None)
        if isinstance(rev, dict) and rev.get("mean") is not None:
            cards.append(
                self._kpi(
                    "Marketing-attributed revenue",
                    self._money(rev["mean"]),
                    f"{ci}% range&nbsp; {self._money(rev.get('lower'))} – "
                    f"{self._money(rev.get('upper'))}",
                )
            )
        share = getattr(self.data, "marketing_contribution_pct", None)
        if isinstance(share, dict) and share.get("mean") is not None:
            cards.append(
                self._kpi(
                    "Share of total revenue",
                    self._pct(share["mean"]),
                    f"{ci}% range&nbsp; {self._pct(share.get('lower'))} – "
                    f"{self._pct(share.get('upper'))}",
                )
            )
        roi = getattr(self.data, "blended_roi", None)
        if isinstance(roi, dict) and roi.get("mean") is not None:
            cards.append(
                self._kpi(
                    "Blended return per $1",
                    self._ratio(roi["mean"]),
                    f"{ci}% range&nbsp; {self._ratio(roi.get('lower'))} – "
                    f"{self._ratio(roi.get('upper'))}",
                )
            )
        if not cards:
            return ""
        return f'<div class="kpi-grid">{"".join(cards)}</div>'

    @staticmethod
    def _kpi(label: str, value: str, ci: str) -> str:
        return (
            f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="ci">{ci}</div></div>'
        )

    def _recommendations(self) -> str:
        rows = self.rows()
        if not rows:
            return ""
        groups = reallocation_groups(rows)

        def _names(rs: list[dict]) -> str:
            return ", ".join(self._ch(r["name"]) for r in rs)

        def _spend_sum(rs: list[dict]) -> float:
            return sum((r.get("spend") or 0.0) for r in rs)

        bullets: list[str] = []
        if groups["scale"]:
            bullets.append(
                f"<b>Scale {_names(groups['scale'])}</b> — the channel"
                f"{'s' if len(groups['scale']) > 1 else ''} whose entire plausible "
                "range clears break-even, and whose next dollar still pays back."
            )
        if groups["reduce"]:
            spend = _spend_sum(groups["reduce"])
            spend_str = f" together {self._money(spend)} of spend" if spend > 0 else ""
            bullets.append(
                f"<b>Reduce {_names(groups['reduce'])}</b> —{spend_str} returning "
                "well under a dollar across every plausible outcome."
            )
        if groups["test"]:
            bullets.append(
                f"<b>Test {_names(groups['test'])} before scaling</b> — strong "
                "averages, but the uncertainty is too wide to fund at risk."
            )
        if groups["hold"] and not (groups["scale"] or groups["reduce"]):
            bullets.append(
                f"<b>Hold {_names(groups['hold'])}</b> — near break-even with no "
                "clear case to scale or cut yet."
            )
        if not bullets:
            return ""
        items = "".join(
            f'<li><span class="marker"></span><span>{b}</span></li>' for b in bullets
        )
        return '<div class="rec"><h4>What we recommend</h4>' f"<ul>{items}</ul></div>"


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Where revenue comes from (decomposition)
# ─────────────────────────────────────────────────────────────────────────────
class AugurDecompositionSection(AugurSection):
    section_id = "decomp"
    default_title = "Where revenue comes from"
    eyebrow = "Where revenue comes from"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        cts = getattr(self.data, "component_time_series", None)
        dates = getattr(self.data, "dates", None)
        if not cts or dates is None:
            # Fall back to a contribution-by-channel snapshot if no time series.
            return self._render_snapshot()

        chart = charts.create_decomposition_chart(
            dates=dates,
            components=cts,
            config=self.config,
            chart_config=ChartConfig(height=380, y_title="Revenue"),
            div_id="augurDecomp",
        )
        intro = (
            "<p>Total revenue separates into a <strong>baseline</strong> the "
            "business would earn with no media, and the <strong>marketing-driven"
            "</strong> increment on top. The mix shifts as investment grows and "
            "burst channels pulse in and out.</p>"
        )
        caption = (
            '<p class="chart-caption">Revenue components over time; contribution '
            "tracks adstocked spend, so it builds during flights and decays after "
            "them.</p>"
        )
        content = f'{intro}<div class="chart-card">{chart}</div>{caption}'
        return self._wrap(content)

    def _render_snapshot(self) -> str:
        totals = getattr(self.data, "component_totals", None)
        if not isinstance(totals, dict) or not totals:
            return ""
        names = list(totals.keys())
        values = np.array([float(totals[n]) for n in names])
        colors = [self.config.channel_colors.get(n) for n in names]
        trace = [
            {
                "type": "bar",
                "x": names,
                "y": [float(v) for v in values],
                "marker": {"color": colors},
                "hovertemplate": "%{x}: %{y:,.0f}<extra></extra>",
            }
        ]
        layout = ChartConfig(height=340, y_title="Contribution").to_plotly_layout(
            self.config.color_scheme
        )
        layout["showlegend"] = False
        chart = charts.create_plotly_div(trace, layout, "augurDecompSnap")
        intro = (
            "<p>Where the modelled revenue comes from — baseline plus each "
            "channel's marketing-driven contribution over the period.</p>"
        )
        return self._wrap(f'{intro}<div class="chart-card">{chart}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Channel scorecard
# ─────────────────────────────────────────────────────────────────────────────
class AugurScorecardSection(AugurSection):
    section_id = "scorecard"
    default_title = "What each channel returns today"
    eyebrow = "Channel scorecard"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        if not rows:
            return ""
        ci = int((self.section_config.credible_interval or 0.8) * 100)

        has_evidence = any(self._evidence(r["name"]) for r in rows)
        body = []
        for r in rows:
            swatch = self.config.channel_colors.get(r["name"])
            spend = self._money(r.get("spend")) if r.get("spend") else "—"
            ev_cell = (
                f"<td>{self._evidence_chip(r['name'])}</td>" if has_evidence else ""
            )
            body.append(f"""
                <tr>
                    <td class="chname"><span class="swatch" style="background:{swatch}"></span>{self._ch(r['name'])}</td>
                    <td class="mono">{spend}</td>
                    <td class="mono">{r['roi']:.2f}</td>
                    <td class="mono">{r['roi_lower']:.2f} – {r['roi_upper']:.2f}</td>
                    <td><span class="tier-chip {r['css']}">{html.escape(r['read'])}</span></td>
                    {ev_cell}
                    <td class="action-cell {r['css']}">{html.escape(r['action'])}</td>
                </tr>
                """)
        intro = (
            "<p>Return is revenue generated per dollar of media, with an "
            f"{ci}% credible interval — the range the data considers plausible. "
            "The <em>read</em> reflects where that whole range sits relative to "
            "break-even, and the <em>action</em> follows from it.</p>"
        )
        ev_head = "<th>Evidence</th>" if has_evidence else ""
        table = f"""
            <table class="data-table">
              <thead><tr><th>Channel</th><th>Spend</th><th>Return / $1</th>
                <th>{ci}% range</th><th>Read</th>{ev_head}<th>Action</th></tr></thead>
              <tbody>{''.join(body)}</tbody>
            </table>
        """
        note = (
            '<p class="note" style="margin-top:.85rem">Ordered by recommended '
            "priority. A return below $1 means the channel is, on current "
            "evidence, returning less than it costs.</p>"
        )
        return self._wrap(f"{intro}{table}{note}")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Return & uncertainty (ROI forest, tier-coloured)
# ─────────────────────────────────────────────────────────────────────────────
class AugurROISection(AugurSection):
    section_id = "roi"
    default_title = "Return on investment, with the uncertainty shown"
    eyebrow = "Return & uncertainty"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        if not rows:
            return ""
        # Highest return at the top.
        ordered = sorted(rows, key=lambda r: r["roi"], reverse=True)
        ys = [r["name"] for r in ordered]
        xs = [r["roi"] for r in ordered]
        plus = [r["roi_upper"] - r["roi"] for r in ordered]
        minus = [r["roi"] - r["roi_lower"] for r in ordered]
        cols = [r["color"] for r in ordered]
        cd = [[r["roi_lower"], r["roi_upper"]] for r in ordered]
        cs = self.config.color_scheme
        xmax = max(3.0, max(r["roi_upper"] for r in ordered) * 1.08)

        trace = [
            {
                "type": "scatter",
                "x": xs,
                "y": [self._fmt_y(n) for n in ys],
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": plus,
                    "arrayminus": minus,
                    "color": cs.text_muted,
                    "thickness": 2,
                    "width": 7,
                },
                "mode": "markers",
                "marker": {
                    "color": cols,
                    "size": 13,
                    "line": {"color": "#ffffff", "width": 1},
                },
                "customdata": cd,
                "hovertemplate": (
                    "%{y}<br>Return $%{x:.2f} per $1<br>"
                    "80% CI [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>"
                ),
            }
        ]
        layout = ChartConfig(
            height=max(300, 48 * len(ordered)),
            x_title="Return per $1 of spend",
        ).to_plotly_layout(cs)
        layout["showlegend"] = False
        layout["xaxis"]["range"] = [0, xmax]
        layout["xaxis"]["tickprefix"] = "$"
        layout["yaxis"]["autorange"] = "reversed"
        layout["shapes"] = [
            {
                "type": "line",
                "x0": 1,
                "x1": 1,
                "y0": -0.6,
                "y1": len(ordered) - 0.4,
                "line": {"color": cs.text_muted, "width": 1.3, "dash": "dash"},
            }
        ]
        layout["annotations"] = [
            {
                "x": 1,
                "y": -0.5,
                "xref": "x",
                "yref": "y",
                "text": "Break-even · $1.00",
                "showarrow": False,
                "font": {"size": 11, "color": cs.text},
                "yshift": 6,
            }
        ]
        chart = charts.create_plotly_div(trace, layout, "augurForest")
        intro = (
            "<p>Each marker is the channel's central return; the bar is its 80% "
            "credible interval. Where the whole bar sits to the right of the "
            "dashed break-even line, the channel pays back with confidence. Where "
            "it straddles the line, the honest answer is that we do not yet know — "
            "a cue to test, not to bet.</p>"
        )
        return self._wrap(f'{intro}<div class="chart-card">{chart}</div>')

    def _fmt_y(self, name: str) -> str:
        # y-axis tick labels are not post-processed, so format display names here.
        if self.config.format_channel_names and "_" in name:
            return name.replace("_", " ").title()
        return name


# ─────────────────────────────────────────────────────────────────────────────
# 05 — The next dollar (marginal vs average)
# ─────────────────────────────────────────────────────────────────────────────
class AugurMarginalSection(AugurSection):
    section_id = "marginal"
    default_title = "What the next dollar returns — not just the average"
    eyebrow = "The next dollar"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        # Only chart channels that HAVE a marginal ROAS — keeping the x/avg/mar
        # arrays aligned and never passing None into the Plotly bar/hover.
        have_mroas = [r for r in rows if r.get("mroas") is not None]
        if len(have_mroas) < 1:
            return ""  # no marginal ROAS available — section degrades to empty
        cs = self.config.color_scheme
        names = [self._fmt_name(r["name"]) for r in have_mroas]
        avg = [r["roi"] for r in have_mroas]
        mar = [r["mroas"] for r in have_mroas]
        traces = [
            {
                "type": "bar",
                "x": names,
                "y": avg,
                "name": "Average return (all spend)",
                "marker": {"color": "#a8c485"},
                "hovertemplate": "%{x}<br>Avg $%{y:.2f}<extra></extra>",
            },
            {
                "type": "bar",
                "x": names,
                "y": mar,
                "name": "Marginal return (next $1)",
                "marker": {"color": cs.accent},
                "hovertemplate": "%{x}<br>Marginal $%{y:.2f}<extra></extra>",
            },
        ]
        layout = ChartConfig(height=340, y_title="Return per $1").to_plotly_layout(cs)
        layout["barmode"] = "group"
        layout["bargap"] = 0.32
        layout["bargroupgap"] = 0.12
        layout["yaxis"]["tickprefix"] = "$"
        layout["shapes"] = [
            {
                "type": "line",
                "x0": -0.5,
                "x1": len(names) - 0.5,
                "y0": 1,
                "y1": 1,
                "line": {"color": cs.text_muted, "width": 1.2, "dash": "dash"},
            }
        ]
        chart = charts.create_plotly_div(traces, layout, "augurMarginal")
        intro = (
            "<p>Average return spreads a channel's revenue across <em>all</em> its "
            "spend. The decision that matters is the <strong>marginal</strong> "
            "return: what the <em>next</em> dollar would earn at today's spend "
            "level. When marginal sits well below average, the channel is into "
            "diminishing returns.</p>"
        )
        return self._wrap(f'{intro}<div class="chart-card">{chart}</div>')

    def _fmt_name(self, name: str) -> str:
        if self.config.format_channel_names and "_" in name:
            return name.replace("_", " ").title()
        return name


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Saturation by channel
# ─────────────────────────────────────────────────────────────────────────────
class AugurSaturationSection(AugurSection):
    section_id = "saturation"
    default_title = "Diminishing returns, channel by channel"
    eyebrow = "Saturation by channel"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        sat = getattr(self.data, "saturation_curves", None)
        if not sat:
            return ""
        # Only chart channels with a well-formed curve (one malformed channel
        # must not blank the whole section).
        channels = [ch for ch in sat if self._has_curve(sat, ch)]
        if not channels:
            return ""
        try:
            chart = charts.create_saturation_curves(
                channels=channels,
                spend_ranges={ch: sat[ch]["spend"] for ch in channels},
                response_curves={ch: sat[ch]["response"] for ch in channels},
                current_spend=getattr(self.data, "current_spend", None) or {},
                config=self.config,
                chart_config=ChartConfig(height=280),
                div_id="augurSaturation",
            )
        except (KeyError, TypeError):
            return ""
        intro = (
            "<p>Each curve shows modelled response as spend rises. The "
            '<span style="color:var(--gold-700);font-weight:600">gold diamond</span> '
            "marks current spend: below the efficient ceiling means headroom; past "
            "it, added spend mostly buys the flat part of the curve.</p>"
        )
        caption = (
            '<p class="chart-caption">Fitted saturation (Hill) response per '
            "channel; the marker shows where current spend sits on the curve.</p>"
        )
        return self._wrap(f'{intro}<div class="chart-card">{chart}</div>{caption}')


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Where to move budget (reallocation cards)
# ─────────────────────────────────────────────────────────────────────────────
class AugurReallocationSection(AugurSection):
    section_id = "budget"
    default_title = "The reallocation, in four moves"
    eyebrow = "Where to move budget"

    _BLURB = {
        "scale": "Add budget while watching the saturation curve — the range "
        "clears break-even and the marginal dollar still pays back.",
        "test": "Attractive central returns, but the ranges are too wide to fund "
        "on faith. A geo holdout would convert these from guesses into evidence.",
        "hold": "Sitting near break-even with no clear case to scale or cut. Keep "
        "steady until a test or more data moves them off the line.",
        "reduce": "Returning under a dollar across the full range. Redirect freed "
        "spend into the proven winner and the tests above.",
    }

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        if not rows:
            return ""
        groups = reallocation_groups(rows)
        cards = []
        for tier in ("scale", "test", "hold", "reduce"):
            members = groups[tier]
            if not members:
                continue
            meta = TIER_META[tier]
            chs = " · ".join(self._ch(r["name"]) for r in members)
            cards.append(f"""
                <div class="realloc {meta['css']}">
                  <div class="rl-head"><span class="rl-action">{html.escape(meta['action'])}</span>
                    <span class="tier-chip {meta['css']}">{meta['increase']}</span></div>
                  <div class="rl-chs">{chs}</div>
                  <p>{html.escape(self._BLURB[tier])}</p>
                </div>
                """)
        if not cards:
            return ""
        intro = (
            "<p>No change to total investment is required to improve the blended "
            "return. These moves shift weight from proven losers toward the proven "
            "winner and toward learning.</p>"
        )
        return self._wrap(f'{intro}<div class="realloc-grid">{"".join(cards)}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# 07b — The optimized plan, in dollars (Planner allocation)
# ─────────────────────────────────────────────────────────────────────────────
class AugurAllocationSection(AugurSection):
    """The optimized budget plan rendered in the Augur voice — current vs.
    recommended spend per channel, with the portfolio uplift and its credible
    interval.

    This is the *quantitative* companion to the tier-based reallocation guidance
    above: where that section says which way to lean, this one puts dollars on it.
    Data-driven and default-off — it renders only when the bundle carries a
    concrete ``allocation_results`` plan (from :func:`planning.default_reallocation`
    in a report, or a saved Planner plan), so a model with no plan attached
    silently omits it.
    """

    section_id = "allocation"
    default_title = "The optimized plan, in dollars"
    eyebrow = "Recommended allocation"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        alloc = getattr(self.data, "allocation_results", None)
        if not isinstance(alloc, dict) or not alloc.get("allocation"):
            return ""
        parts = [
            self._intro(alloc),
            self._kpis(alloc),
            self._table(alloc),
        ]
        if alloc.get("geo_allocation"):
            parts.append(self._geo_table(alloc))
        parts.append(self._note(alloc))
        return self._wrap("".join(p for p in parts if p))

    # ── pieces ────────────────────────────────────────────────────────────────
    def _intro(self, alloc: dict) -> str:
        dev = alloc.get("deviation_cap")
        band = (
            f" — moving each channel by at most ±{dev * 100:.0f}% of today's spend"
            if isinstance(dev, (int, float)) and np.isfinite(dev)
            else ""
        )
        return (
            "<p>Holding total investment constant, this is where the model would "
            f"move the money{band}. The split below maximizes expected return at "
            "the same budget; treat the uplift range, not the single number, as the "
            "size of the prize.</p>"
        )

    def _kpis(self, alloc: dict) -> str:
        total = alloc.get("total_budget", 0.0)
        uplift = alloc.get("expected_uplift", 0.0)
        hdi = alloc.get("uplift_hdi") or [0.0, 0.0]
        prob = alloc.get("prob_positive_uplift", 0.0)
        regret = alloc.get("expected_regret")
        # Lead with the confidence, not the allocation (issue #105).
        cards = [
            self._kpi(
                "Chance it beats today",
                self._pct(prob),
                "Posterior probability the plan out-performs the current split",
            ),
            self._kpi(
                "Expected KPI uplift",
                self._money(uplift),
                f"90% range&nbsp; {self._money(hdi[0])} – {self._money(hdi[1])}",
            ),
        ]
        if regret is not None:
            cards.append(
                self._kpi(
                    "Expected regret",
                    self._money(regret),
                    "KPI you'd forgo vs a perfectly-informed plan",
                )
            )
        cards.append(
            self._kpi(
                "Budget allocated",
                self._money(total),
                "Same as the current total — a pure reallocation",
            )
        )
        return f'<div class="kpi-grid">{"".join(cards)}</div>'

    @staticmethod
    def _kpi(label: str, value: str, ci: str) -> str:
        return (
            f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="ci">{ci}</div></div>'
        )

    @staticmethod
    def _change_chip(chg: float) -> str:
        """A tier-coloured change chip: sage for increases, rust for cuts, steel
        for ≈no change. Reuses the scorecard's tier-chip CSS."""
        if not np.isfinite(chg):
            return '<span class="tier-chip t-hold">—</span>'
        if chg > 1:
            return f'<span class="tier-chip t-scale">+{chg:.0f}%</span>'
        if chg < -1:
            return f'<span class="tier-chip t-reduce">{chg:.0f}%</span>'
        return '<span class="tier-chip t-hold">no change</span>'

    def _table(self, alloc: dict) -> str:
        has_range = any(
            "within_observed_range" in r for r in alloc.get("allocation", [])
        )
        body = []
        for r in alloc["allocation"]:
            name = r.get("channel", "")
            swatch = self.config.channel_colors.get(name)
            cur = float(r.get("current_spend", 0.0) or 0.0)
            opt = float(r.get("optimal_spend", 0.0) or 0.0)
            chg = float(r.get("change_pct", 0.0) or 0.0)
            range_cell = ""
            if has_range:
                if r.get("within_observed_range", True):
                    range_cell = (
                        '<td><span class="tier-chip t-scale">in range</span></td>'
                    )
                else:
                    range_cell = (
                        '<td><span class="tier-chip t-test">⚠ extrapolated</span></td>'
                    )
            body.append(
                f'<tr><td class="chname"><span class="swatch" '
                f'style="background:{swatch}"></span>{self._ch(name)}</td>'
                f'<td class="mono">{self._money(cur)}</td>'
                f'<td class="mono">{self._money(opt)}</td>'
                f"<td>{self._change_chip(chg)}</td>{range_cell}</tr>"
            )
        range_head = "<th>Range</th>" if has_range else ""
        return f"""
            <table class="data-table">
              <thead><tr><th>Channel</th><th>Current spend</th>
                <th>Recommended</th><th>Change</th>{range_head}</tr></thead>
              <tbody>{''.join(body)}</tbody>
            </table>
        """

    def _geo_table(self, alloc: dict) -> str:
        rows = alloc.get("geo_allocation") or []
        if not rows:
            return ""
        body = []
        for r in rows:
            geo = self._ch(r.get("geo", ""))
            name = self._ch(r.get("channel", ""))
            cur = float(r.get("current_spend", 0.0) or 0.0)
            opt = float(r.get("optimal_spend", 0.0) or 0.0)
            chg = float(r.get("change_pct", 0.0) or 0.0)
            body.append(
                f"<tr><td>{geo}</td><td>{name}</td>"
                f'<td class="mono">{self._money(cur)}</td>'
                f'<td class="mono">{self._money(opt)}</td>'
                f"<td>{self._change_chip(chg)}</td></tr>"
            )
        return f"""
            <h3 style="margin-top:1.4rem">By geography</h3>
            <table class="data-table">
              <thead><tr><th>Geography</th><th>Channel</th><th>Current</th>
                <th>Recommended</th><th>Change</th></tr></thead>
              <tbody>{''.join(body)}</tbody>
            </table>
        """

    def _note(self, alloc: dict) -> str:
        dev = alloc.get("deviation_cap")
        guard = (
            f"Each channel is held within ±{dev * 100:.0f}% of its current spend, so "
            "no channel is switched off and every move stays inside the range the "
            "model has direct evidence for. "
            if isinstance(dev, (int, float)) and np.isfinite(dev)
            else ""
        )
        return (
            f'<p class="note" style="margin-top:.85rem">{guard}A reallocation at '
            "constant budget shifts weight toward the channels whose next dollar "
            "still pays back; the uplift interval is the basis for the decision, "
            "not the point estimate.</p>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 08 — The flighting plan (illustrative)
# ─────────────────────────────────────────────────────────────────────────────
class AugurFlightingSection(AugurSection):
    section_id = "flighting"
    default_title = "Same budget, steadier weight"
    eyebrow = "The flighting plan"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        fl = illustrative_flighting_totals(rows) if rows else None
        if not fl:
            return ""
        cs = self.config.color_scheme
        traces = [
            {
                "type": "scatter",
                "x": fl["weeks"],
                "y": fl["current_total"],
                "name": "Current plan",
                "mode": "lines",
                "line": {"color": cs.text_muted, "width": 1.2},
                "fill": "tozeroy",
                "fillcolor": f"rgba({charts._hex_to_rgb(cs.text_muted)}, 0.18)",
                "hovertemplate": "Wk %{x}: $%{y:,.0f}<extra></extra>",
            },
            {
                "type": "scatter",
                "x": fl["weeks"],
                "y": fl["recommended_total"],
                "name": "Recommended plan",
                "mode": "lines",
                "line": {"color": cs.primary, "width": 2},
                "hovertemplate": "Wk %{x}: $%{y:,.0f}<extra></extra>",
            },
        ]
        layout = ChartConfig(height=320, y_title="Weekly spend").to_plotly_layout(cs)
        layout["xaxis"]["title"] = "Week"
        layout["yaxis"]["tickprefix"] = "$"
        chart = charts.create_plotly_div(traces, layout, "augurFlighting")
        intro = (
            "<p>The reallocation does not raise the total — it changes <em>when</em> "
            "the money lands. The recommended plan trims heavy bursts on the weak "
            "channels and redeploys into a more continuous presence on the winner. "
            "Continuity matters because carryover compounds.</p>"
        )
        caption = (
            '<p class="chart-caption">Total weekly media spend — current (grey) vs. '
            "recommended (sage), at the same annual budget. Weekly shapes are "
            "illustrative, derived from the modelled annual spend and each "
            "channel's evidence tier.</p>"
        )
        return self._wrap(
            f'{intro}<div class="chart-card">{chart}</div>{caption}',
            illustrative=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 09 — Channel deep dives
# ─────────────────────────────────────────────────────────────────────────────
class AugurDeepDivesSection(AugurSection):
    section_id = "deepdives"
    default_title = "Each channel, in full"
    eyebrow = "Channel deep dives"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        if not rows:
            return ""
        sat = getattr(self.data, "saturation_curves", None) or {}
        adstock = getattr(self.data, "adstock_curves", None) or {}
        dives = [self._dive(i, r, sat, adstock) for i, r in enumerate(rows)]
        intro = (
            "<p>For every channel: what it returns and how sure we are, where it "
            "sits on its saturation curve, how long its effect lingers, and the "
            "specific spend action. Channels are ordered by recommended "
            "priority.</p>"
        )
        return self._wrap(intro + "".join(dives))

    def _half_life(self, ch: str, adstock: dict) -> str:
        w = adstock.get(ch)
        if w is None:
            return "—"
        w = np.asarray(w, dtype=float)
        if w.size == 0 or w[0] <= 0:
            return "—"
        for k in range(1, w.size):
            if w[k] <= 0.5 * w[0]:
                return f"{k}w"
        return f"≥{w.size - 1}w"

    def _dive(self, idx: int, r: dict, sat: dict, adstock: dict) -> str:
        meta = TIER_META[r["tier"]]
        name = self._ch(r["name"])
        whatdo = self.insight(f"channel:{r['name']}")
        # KPI row
        contribution = (
            self._money(r.get("contribution")) if r.get("contribution") else "—"
        )
        mroas = f"{r['mroas']:.2f}" if r.get("mroas") is not None else "—"
        spend = self._money(r.get("spend")) if r.get("spend") else "—"
        share = (
            f"{r['spend_share'] * 100:.0f}% of budget"
            if r.get("spend_share") is not None
            else ""
        )
        half = self._half_life(r["name"], adstock)
        kpis = f"""
          <div class="dd-kpis">
            <div class="dd-kpi"><div class="l">Return / $1</div><div class="v">{r['roi']:.2f}</div><div class="sub">80% {r['roi_lower']:.2f}–{r['roi_upper']:.2f}</div></div>
            <div class="dd-kpi"><div class="l">Contribution</div><div class="v">{contribution}</div><div class="sub">&nbsp;</div></div>
            <div class="dd-kpi"><div class="l">Spend</div><div class="v">{spend}</div><div class="sub">{html.escape(share)}</div></div>
            <div class="dd-kpi"><div class="l">Marginal / $1</div><div class="v">{mroas}</div><div class="sub">&nbsp;</div></div>
            <div class="dd-kpi"><div class="l">Carryover half-life</div><div class="v">{half}</div><div class="sub">&nbsp;</div></div>
          </div>
        """
        # charts (saturation + adstock for this one channel)
        chart_parts = []
        if self._has_curve(sat, r["name"]):
            try:
                sc = charts.create_saturation_curves(
                    channels=[r["name"]],
                    spend_ranges={r["name"]: sat[r["name"]]["spend"]},
                    response_curves={r["name"]: sat[r["name"]]["response"]},
                    current_spend=getattr(self.data, "current_spend", None) or {},
                    config=self.config,
                    chart_config=ChartConfig(height=240),
                    div_id=f"ddSat{idx}",
                )
                chart_parts.append(
                    f'<div class="dd-chart"><h5>Saturation</h5>{sc}</div>'
                )
            except (KeyError, TypeError):
                pass
        if r["name"] in adstock:
            ac = charts.create_adstock_chart(
                channels=[r["name"]],
                lag_weights={r["name"]: adstock[r["name"]]},
                config=self.config,
                chart_config=ChartConfig(height=240),
                div_id=f"ddAd{idx}",
            )
            chart_parts.append(
                f'<div class="dd-chart"><h5>Carryover decay</h5>{ac}</div>'
            )
        charts_html = (
            f'<div class="dd-charts">{"".join(chart_parts)}</div>'
            if chart_parts
            else ""
        )
        rec = (
            f'<div class="dd-rec {meta["css"]}"><h5>What to do</h5>'
            f"<p>{self._t(whatdo)}</p></div>"
            if whatdo
            else ""
        )
        return f"""
        <div class="dd" id="dd-{self._slug(r['name'])}">
          <div class="dd-head">
            <div class="dd-title"><span class="dot" style="background:{meta['color']}"></span>{name}</div>
            <div class="dd-chips" style="display:flex;gap:.4rem;align-items:center;flex-wrap:wrap">{self._evidence_chip(r['name'])}<span class="tier-chip {meta['css']}">{html.escape(meta['action'])}</span></div>
          </div>
          {kpis}
          {charts_html}
          {rec}
        </div>
        """


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Carryover & continuity (adstock, all channels)
# ─────────────────────────────────────────────────────────────────────────────
class AugurCarryoverSection(AugurSection):
    section_id = "carryover"
    default_title = "Advertising keeps working after the flight ends"
    eyebrow = "Carryover & continuity"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        adstock = getattr(self.data, "adstock_curves", None)
        if not adstock:
            return ""
        chart = charts.create_adstock_chart(
            channels=list(adstock.keys()),
            lag_weights=adstock,
            config=self.config,
            chart_config=ChartConfig(height=320),
            div_id="augurAdstockAll",
        )
        intro = (
            "<p>Every channel's effect persists beyond the week the money is spent, "
            "then fades. The decay rate differs by channel, but the planning lesson "
            "is shared: weight builds over a flight and continuity compounds.</p>"
        )
        implications = (
            "<h3>Two planning implications</h3>"
            '<p style="margin-bottom:.4rem"><b>Don\'t judge a flight in week one.</b> '
            "Early reads understate true performance because the effect is still "
            "accumulating.</p>"
            "<p><b>Avoid hard on/off gaps.</b> Continuity compounds; abrupt dark "
            "periods waste the carryover you already paid for.</p>"
        )
        return self._wrap(f'{intro}<div class="chart-card">{chart}</div>{implications}')


# ─────────────────────────────────────────────────────────────────────────────
# 11 — Does the model hold up? (posterior-predictive fit over time)  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
class AugurModelFitSection(AugurSection):
    section_id = "ppc-fit"
    default_title = "Does the model track reality?"
    eyebrow = "Posterior-predictive fit"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        dates = getattr(self.data, "dates", None)
        actual = getattr(self.data, "actual", None)
        predicted = getattr(self.data, "predicted", None)
        if dates is None or actual is None or not isinstance(predicted, dict):
            return ""
        mean = predicted.get("mean")
        if mean is None:
            return ""
        lower = predicted.get("lower", mean)
        upper = predicted.get("upper", mean)
        cc = ChartConfig(
            height=self.section_config.chart_height or 380,
            ci_level=self.section_config.credible_interval or 0.8,
            y_title="KPI",
        )
        chart = charts.create_model_fit_chart(
            dates=dates,
            actual=np.asarray(actual, dtype=float),
            predicted_mean=np.asarray(mean, dtype=float),
            predicted_lower=np.asarray(lower, dtype=float),
            predicted_upper=np.asarray(upper, dtype=float),
            config=self.config,
            chart_config=cc,
            div_id="augurPPCFit",
        )
        gloss = self.insight("fit_gloss")
        intro = (
            f"<p>{self._t(gloss)}</p>"
            if gloss
            else (
                "<p>Before trusting the splits above, the model has to reproduce "
                "the revenue it was trained on. The line is the posterior mean; the "
                "band is the credible interval — the outcomes the model considers "
                "plausible, including observation noise.</p>"
            )
        )
        cards = self._fit_cards(cc.ci_level)
        caption = (
            '<p class="chart-caption">Observed KPI (points) against the model\'s '
            f"posterior-predictive mean (line) and {int(cc.ci_level * 100)}% "
            "credible band over the analysis period.</p>"
        )
        return self._wrap(
            f'{intro}{cards}<div class="chart-card">{chart}</div>{caption}'
        )

    def _fit_cards(self, ci_level: float) -> str:
        cards: list[str] = []
        fit = getattr(self.data, "fit_statistics", None) or {}
        pp = getattr(self.data, "posterior_predictive", None) or {}
        r2 = fit.get("r2")
        if r2 is None:
            r2 = pp.get("r2")
        if r2 is not None and np.isfinite(float(r2)):
            cards.append(self._card(f"{float(r2):.2f}", "R² (observed vs predicted)"))
        mape = fit.get("mape")
        if mape is not None and np.isfinite(float(mape)):
            cards.append(self._card(f"{float(mape):.1%}", "Mean abs. % error"))
        cov = pp.get("coverage")
        if isinstance(cov, list) and cov:
            target = float(pp.get("ci_level", ci_level))
            best = min(cov, key=lambda d: abs(float(d.get("nominal", 0)) - target))
            cards.append(
                self._card(
                    f"{float(best.get('empirical', 0.0)):.0%}",
                    f"Coverage of the {float(best.get('nominal', target)):.0%} band",
                )
            )
        if not cards:
            return ""
        return f'<div class="kpi-grid">{"".join(cards)}</div>'

    @staticmethod
    def _card(value: str, label: str) -> str:
        return (
            f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value}</div></div>'
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Posterior predictive checks  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
class AugurPPCSection(AugurSection):
    section_id = "ppc-checks"
    default_title = "Can the model reproduce the data?"
    eyebrow = "Posterior predictive checks"

    _BAYES_P_LABELS = {
        "mean": "Mean",
        "std": "Std. deviation",
        "min": "Minimum",
        "max": "Maximum",
    }

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        pp = getattr(self.data, "posterior_predictive", None)
        if not pp or pp.get("observed") is None:
            return ""
        observed = np.asarray(pp["observed"], dtype=float)
        pred_mean = pp.get("pred_mean")
        if pred_mean is None or len(np.asarray(pred_mean)) != len(observed):
            return ""
        pred_mean = np.asarray(pred_mean, dtype=float)
        height = self.section_config.chart_height or 360
        cc = ChartConfig(height=height)

        intro = (
            "<p>These checks ask the honest question of fit: can the model "
            "reproduce the data it was trained on? Observed points should sit near "
            "the 45° line and inside their predictive intervals; the observed "
            "density should nest within the replicated cloud; calibration should "
            "track the diagonal; residuals should scatter structurelessly around "
            "zero.</p>"
        )
        parts = [intro]

        parts.append('<div class="chart-grid-2">')
        parts.append(
            '<div class="chart-card">'
            + charts.create_ppc_observed_vs_predicted(
                observed=observed,
                pred_mean=pred_mean,
                pred_lower=pp.get("pred_lower"),
                pred_upper=pp.get("pred_upper"),
                config=self.config,
                chart_config=cc,
                div_id="augurPPCObsPred",
            )
            + "</div>"
        )
        parts.append(
            '<div class="chart-card">'
            + charts.create_ppc_residual_plot(
                observed=observed,
                pred_mean=pred_mean,
                config=self.config,
                chart_config=cc,
                div_id="augurPPCResid",
            )
            + "</div>"
        )
        parts.append("</div>")

        if pp.get("samples") is not None:
            parts.append(
                '<div class="chart-card">'
                + charts.create_ppc_density_overlay(
                    observed=observed,
                    samples=np.asarray(pp["samples"], dtype=float),
                    config=self.config,
                    chart_config=cc,
                    div_id="augurPPCDensity",
                )
                + "</div>"
            )
        if pp.get("coverage"):
            parts.append(
                '<div class="chart-card">'
                + charts.create_ppc_interval_calibration(
                    coverage=pp["coverage"],
                    config=self.config,
                    chart_config=ChartConfig(height=min(height, 360)),
                    div_id="augurPPCCalib",
                )
                + "</div>"
            )

        bayes_p = pp.get("bayes_p")
        if bayes_p:
            parts.append(self._bayes_p(bayes_p))

        return self._wrap("\n".join(parts))

    def _bayes_p(self, bayes_p: dict) -> str:
        body = []
        for stat, p in bayes_p.items():
            if p is None:
                continue
            p = float(p)
            label = self._BAYES_P_LABELS.get(stat, str(stat).title())
            ok = 0.05 <= p <= 0.95
            status = "Reproduced" if ok else "Poorly reproduced"
            css = "t-scale" if ok else "t-reduce"
            body.append(
                f'<tr><td>{html.escape(label)}</td><td class="mono">{p:.2f}</td>'
                f'<td><span class="tier-chip {css}">{status}</span></td></tr>'
            )
        if not body:
            return ""
        return f"""
            <h3>Predictive p-values (summary statistics)</h3>
            <p>The probability a replicated dataset is more extreme than the
            observed data on each statistic. Values near 0.5 mean the model
            reproduces that feature; values near 0 or 1 mean it does not.</p>
            <table class="data-table" style="max-width:520px;">
              <thead><tr><th>Statistic</th><th>p-value</th><th>Read</th></tr></thead>
              <tbody>{''.join(body)}</tbody>
            </table>
        """


# ─────────────────────────────────────────────────────────────────────────────
# 13 — Reading the evidence (legend)
# ─────────────────────────────────────────────────────────────────────────────
class AugurEvidenceSection(AugurSection):
    section_id = "evidence"
    default_title = "How to read this report"
    eyebrow = "Reading the evidence"

    _LEGEND = [
        (
            "scale",
            "Scale",
            "Profitable across the full plausible range. Safe to lean into.",
        ),
        (
            "test",
            "Test",
            "High central estimate, wide range. Worth money — once a test confirms it.",
        ),
        ("hold", "Hold", "Near break-even and model-only. No case to move it yet."),
        ("reduce", "Reduce", "Below break-even across the range. Redirect the spend."),
    ]

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = "".join(
            f'<div class="row"><span class="sw {key}"></span><div>'
            f'<div class="lg-name">{html.escape(name)}</div>'
            f'<div class="lg-desc">{html.escape(desc)}</div></div></div>'
            for key, name, desc in self._LEGEND
        )
        intro = (
            "<p>Every recommendation is colour-coded by how much we trust it — the "
            "colour is the confidence, not just the number.</p>"
        )
        outro = (
            "<p>A wide range is not a flaw in the model — it is the model being "
            "honest about thin data. The cure for uncertainty is evidence: a test, "
            "not a more confident spreadsheet.</p>"
        )

        # Evidence-provenance tiers (issue #102) — a SECOND colour language, on
        # every channel number: where its credibility comes from. Only when the
        # extractor attached evidence to at least one channel.
        provenance = ""
        ev = getattr(self.data, "channel_evidence", None) or {}
        if ev:
            tier_rows = "".join(
                f'<div class="row">{evidence_chip_html({"tier": t.value}, theme="augur", show_caveat=False)}'
                f'<div><div class="lg-desc">{html.escape(gloss)}</div></div></div>'
                for t, gloss in (
                    (
                        EvidenceTier.EXPERIMENT_VALIDATED,
                        "Calibrated against a randomized experiment folded into this "
                        "fit — the strongest causal anchor.",
                    ),
                    (
                        EvidenceTier.MODEL_IDENTIFIED,
                        "The data moved this effect off its prior and the channel is "
                        "separately identifiable — a genuine model finding, not yet "
                        "experimentally confirmed.",
                    ),
                    (
                        EvidenceTier.PRIOR_DOMINATED,
                        "The posterior barely moved off its prior — this number "
                        "reflects the assumed prior more than the data. Treat it as a "
                        "placeholder until confirmed.",
                    ),
                )
            )
            provenance = (
                '<p style="margin-top:1.4rem">Separately, every channel number '
                "carries an <em>evidence tier</em> — how much of it is data versus "
                "assumption. A <em>not separately identified</em> flag means two "
                "channels are collinear and their individual numbers cannot be "
                "trusted apart (the combined effect can).</p>"
                f'<div class="legend">{tier_rows}</div>'
            )

        return self._wrap(f'{intro}<div class="legend">{rows}</div>{outro}{provenance}')


# ─────────────────────────────────────────────────────────────────────────────
# 14 — Recommended tests
# ─────────────────────────────────────────────────────────────────────────────
class AugurTestsSection(AugurSection):
    section_id = "tests"
    default_title = "Experiments that would tighten the next plan"
    eyebrow = "Recommended tests"

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        rows = self.rows()
        cands = test_candidates(rows)[:3] if rows else []
        if not cands:
            return ""
        items = []
        for i, r in enumerate(cands, start=1):
            name = self._ch(r["name"])
            if r["tier"] == "reduce":
                title = f"Spend-down test on {name}"
                desc = (
                    f"Confirms {name} can be cut without losing revenue before "
                    f"reallocating its budget. Low risk given the tight "
                    f"{r['roi_lower']:.2f}–{r['roi_upper']:.2f} range."
                )
            else:
                title = f"Geo holdout on {name}"
                desc = (
                    f"Highest unproven upside (central {r['roi']:.2f}, range "
                    f"{r['roi_lower']:.2f}–{r['roi_upper']:.2f}). A matched-market "
                    "holdout would settle whether to scale or stand down."
                )
            items.append(
                f'<div class="test-item"><div class="tnum">{i:02d}</div>'
                f'<div class="tbody"><b>{title}</b><span>{desc}</span></div></div>'
            )
        intro_text = self.insight("tests_intro") or (
            "Each test converts a channel from model-only to evidence-backed, "
            "shrinking the ranges that currently force a hold."
        )
        intro = f"<p>{self._t(intro_text)}</p>"
        return self._wrap(f'{intro}<div class="tests">{"".join(items)}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# 15 — Next steps (measurement loop)
# ─────────────────────────────────────────────────────────────────────────────
class AugurNextStepsSection(AugurSection):
    section_id = "next"
    default_title = "Where this sits in the measurement loop"
    eyebrow = "Next steps"

    _LOOP = ["Fit", "Prioritize", "Experiment", "Calibrate", "Allocate", "Re-evaluate"]
    _CURRENT = {"Prioritize", "Experiment"}

    def render(self) -> str:
        if not self.is_enabled:
            return ""
        steps = []
        for i, s in enumerate(self._LOOP):
            cur = " current" if s in self._CURRENT else ""
            steps.append(f'<span class="step{cur}">{html.escape(s)}</span>')
            if i < len(self._LOOP) - 1:
                steps.append('<span class="arrow">→</span>')
        loop = f'<div class="loop">{"".join(steps)}</div>'

        next_steps = self.insight("next_steps")
        body = self._render_next_steps(next_steps)
        return self._wrap(f"{loop}{body}")

    def _render_next_steps(self, text: str) -> str:
        if not text:
            return ""
        # Parse "Planning: … Analytics: … Together: …" into who/what rows.
        labels = ["Planning", "Analytics", "Together"]
        positions = []
        low = text
        for lab in labels:
            idx = low.find(lab + ":")
            if idx >= 0:
                positions.append((idx, lab))
        positions.sort()
        if positions:
            items = []
            for j, (idx, lab) in enumerate(positions):
                start = idx + len(lab) + 1
                end = positions[j + 1][0] if j + 1 < len(positions) else len(text)
                chunk = text[start:end].strip().rstrip(".") + "."
                items.append(
                    f'<li><span class="who">{html.escape(lab)}</span>'
                    f"<span>{self._t(chunk)}</span></li>"
                )
            return f'<ul class="next-steps">{"".join(items)}</ul>'
        return f"<p>{self._t(text)}</p>"


# Ordered Augur section set (matches the contents nav). Each entry maps to a
# ReportConfig SectionConfig attribute that gates it.
AUGUR_SECTIONS: list[tuple[str, type[AugurSection], str]] = [
    ("summary", AugurHeadlineSection, "headline"),
    ("decomp", AugurDecompositionSection, "decomposition"),
    ("scorecard", AugurScorecardSection, "channel_roi"),
    ("roi", AugurROISection, "channel_roi"),
    ("marginal", AugurMarginalSection, "marginal_returns"),
    ("saturation", AugurSaturationSection, "saturation"),
    ("budget", AugurReallocationSection, "reallocation"),
    ("allocation", AugurAllocationSection, "allocation"),
    ("flighting", AugurFlightingSection, "flighting"),
    ("deepdives", AugurDeepDivesSection, "deep_dives"),
    ("carryover", AugurCarryoverSection, "carryover"),
    ("ppc-fit", AugurModelFitSection, "ppc_timeseries"),
    ("ppc-checks", AugurPPCSection, "posterior_predictive"),
    ("evidence", AugurEvidenceSection, "evidence_guide"),
    ("tests", AugurTestsSection, "recommended_tests"),
    ("next", AugurNextStepsSection, "next_steps"),
]


__all__ = [
    "AugurSection",
    "AugurHeadlineSection",
    "AugurDecompositionSection",
    "AugurScorecardSection",
    "AugurROISection",
    "AugurMarginalSection",
    "AugurSaturationSection",
    "AugurReallocationSection",
    "AugurAllocationSection",
    "AugurFlightingSection",
    "AugurDeepDivesSection",
    "AugurCarryoverSection",
    "AugurModelFitSection",
    "AugurPPCSection",
    "AugurEvidenceSection",
    "AugurTestsSection",
    "AugurNextStepsSection",
    "AUGUR_SECTIONS",
]

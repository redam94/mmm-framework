"""Interactive MMM Results Report generator.

The fitted-model sibling of the Model Design Readout
(:class:`~mmm_framework.reporting.prefit.PrefitReadoutGenerator`): the same
editorial shell (masthead, numbered sticky contents nav, cream/ink palette),
but every headline number is **recomputable in the browser** — date-window
selectors, geo and estimand selectors, response-curve modes and the budget
reallocator all re-aggregate the posterior draws embedded in the page
(see :mod:`.facts` / :mod:`.script`). No selector ever refits the model.

Usage::

    from mmm_framework.reporting import InteractiveReportGenerator

    gen = InteractiveReportGenerator(model, results)
    html = gen.generate_report()            # templated insights
    gen.save_report("results_report.html")
"""

from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..augur_theme import AUGUR_FONTS_LINK, MASTHEAD_LOGO_SVG, augur_css
from ..charts.base import NumpyEncoder
from ..charts.prior import create_prior_predictive_fan, create_prior_stat_distribution
from ..config import ChannelColors, ColorPalette, ColorScheme, ReportConfig
from .facts import interactive_report_facts
from .insights import build_interactive_insights
from .script import INTERACTIVE_REPORT_JS

__all__ = ["InteractiveReportGenerator"]


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    av = abs(v)
    if av >= 1e9:
        return f"{v / 1e9:.1f}B"
    if av >= 1e6:
        return f"{v / 1e6:.1f}M"
    if av >= 1e4:
        return f"{v / 1e3:.0f}K"
    return f"{v:,.{digits}f}"


@dataclass
class _NavEntry:
    section_id: str
    title: str


#: JS-facing subset of the facts dict (everything else renders server-side).
_PAYLOAD_KEYS = (
    "meta",
    "periods",
    "actual_national",
    "fit",
    "ppc_stats",
    "contrib",
    "marginal",
    "spend",
    "divisor_meta",
    "curves",
    "carryover",
    "prior_posterior",
    "sensitivity",
    "yoy",
    "mediation",
    "latent",
)

_EXTRA_CSS = r"""
/* Balanced cards: auto-fit instead of the theme's fixed 3 columns, so four
   cards fill one row instead of leaving a lonely straggler. */
.kpi-grid{grid-template-columns:repeat(auto-fit,minmax(185px,1fr));}
.ir-controls{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:0 0 1rem;padding:.7rem .9rem;background:var(--cream-100);border:1px solid var(--line-200);border-radius:10px;}
.ir-lbl{font-size:.78rem;color:var(--ink-400);}
.ir-select{font-family:var(--font-mono);font-size:.78rem;color:var(--ink-700);background:var(--cream-50);border:1px solid var(--line-300);border-radius:7px;padding:.35rem .5rem;max-width:190px;}
.ir-btn{font-family:var(--font-sans);font-size:.75rem;font-weight:600;color:var(--ink-600);background:var(--cream-50);border:1px solid var(--line-300);border-radius:999px;padding:.3rem .7rem;cursor:pointer;transition:all .15s;}
.ir-btn:hover{border-color:var(--sage-700);color:var(--sage-800);}
.ir-btn.active{background:var(--sage-700);border-color:var(--sage-700);color:#fff;}
.ir-toggle{display:inline-flex;align-items:center;gap:.4rem;font-size:.78rem;color:var(--ink-600);margin-left:auto;cursor:pointer;}
.ir-banner{display:flex;gap:.6rem;align-items:flex-start;background:var(--gold-100);border:1px solid var(--gold-300);border-left:4px solid var(--gold-600);border-radius:10px;padding:.9rem 1.1rem;margin:0 0 1.2rem;font-size:.9rem;color:var(--ink-700);}
.ir-banner b{color:var(--gold-700);}
.ir-slider-row{display:grid;grid-template-columns:180px 1fr 170px;gap:.9rem;align-items:center;padding:.45rem 0;border-bottom:1px dashed var(--line-200);}
.ir-slider-row input[type=range]{width:100%;accent-color:var(--sage-700);}
.ir-slider-name{display:flex;align-items:center;gap:.5rem;font-weight:600;font-size:.85rem;color:var(--ink-900);}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none;}
.ir-slider-val{font-size:.78rem;color:var(--ink-400);text-align:right;}
.mono{font-family:var(--font-mono);}
.chip-approx{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--gold-700);background:var(--gold-100);border:1px solid var(--gold-300);border-radius:999px;padding:.15rem .6rem;margin-left:.5rem;vertical-align:middle;}
@media(max-width:700px){.ir-slider-row{grid-template-columns:110px 1fr 110px;}}
"""


class InteractiveReportGenerator:
    """Generate the interactive **MMM Results Report** HTML document.

    Parameters
    ----------
    model:
        A fitted ``BayesianMMM``. May be ``None`` when ``facts`` is supplied
        directly (tests / cached facts).
    results:
        Optional ``MMMResults`` — used for fit provenance (approximate flag,
        R-hat/ESS) so the report never presents an approximate posterior as a
        calibrated one.
    config:
        A :class:`~mmm_framework.reporting.config.ReportConfig`; only the
        masthead fields, ``color_scheme`` and the Plotly CDN knobs are read.
    llm:
        Optional LangChain chat model — enriches the narrative
        (:func:`build_interactive_insights`); ``None`` keeps templated prose.
    facts:
        Precomputed :func:`interactive_report_facts` output.
    facts_kwargs:
        Extra keyword arguments forwarded to
        :func:`interactive_report_facts` (``max_draws``,
        ``curve_max_draws``, ``include_counterfactual_spec``, …).
    """

    def __init__(
        self,
        model: Any = None,
        results: Any = None,
        *,
        config: ReportConfig | None = None,
        llm: Any | None = None,
        facts: dict[str, Any] | None = None,
        channel_colors: ChannelColors | None = None,
        **facts_kwargs: Any,
    ) -> None:
        if model is None and facts is None:
            raise ValueError(
                "InteractiveReportGenerator needs a model or precomputed facts."
            )
        self.model = model
        self.config = config or ReportConfig(
            title="MMM Results Report",
            color_scheme=ColorScheme.from_palette(ColorPalette.AUGUR),
            confidential=True,
        )
        self.channel_colors = channel_colors or ChannelColors()
        self.facts = (
            facts
            if facts is not None
            else interactive_report_facts(model, results, **facts_kwargs)
        )
        self.insights = build_interactive_insights(self.facts, llm=llm)

    # ── public API ──────────────────────────────────────────────────────────
    def generate_report(self) -> str:
        """Render the full HTML document."""
        sections: list[tuple[_NavEntry, str]] = []
        for builder in (
            self._section_insights,
            self._section_exec,
            self._section_decomp,
            self._section_fit,
            self._section_ppc_stats,
            self._section_roi,
            self._section_yoy,
            self._section_estimands,
            self._section_curves,
            self._section_carryover,
            self._section_pathways,
            self._section_latent,
            self._section_prior_posterior,
            self._section_realloc,
            self._section_sensitivity,
            self._section_ppc_prior,
            self._section_assumptions,
        ):
            entry = builder()
            if entry is not None:
                sections.append(entry)
        return self._assemble(sections)

    def save_report(self, path: str) -> str:
        html_doc = self.generate_report()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        return path

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _wrap(sec_id: str, eyebrow: str, title: str, body: str) -> str:
        return (
            f'<section class="section" id="{sec_id}">'
            f'<div class="section-eyebrow">{_esc(eyebrow)}</div>'
            f"<h2>{_esc(title)}</h2>{body}</section>"
        )

    def _insight(self, key: str) -> str:
        return _esc(self.insights.get(key, ""))

    def _approx_banner(self) -> str:
        if not self.facts.get("meta", {}).get("approximate"):
            return ""
        method = self.facts.get("meta", {}).get("fit_method") or "approximate"
        return (
            '<div class="ir-banner"><span>⚠︎</span><div>'
            f"<b>Approximate fit ({_esc(str(method).upper())})</b> — this "
            "posterior comes from a fast approximation, not full MCMC. Every "
            "interval in this report is <b>not calibrated</b>; re-fit with "
            "NUTS before using these numbers for decisions.</div></div>"
        )

    # ── sections ─────────────────────────────────────────────────────────────
    def _section_insights(self) -> tuple[_NavEntry, str]:
        head = self.facts.get("headline", {})
        rows = sorted(
            head.get("channels", []),
            key=lambda r: r.get("roi_mean") or 0,
            reverse=True,
        )
        bullets: list[str] = []
        if rows:
            b = rows[0]
            bullets.append(
                f"<li><b>{_esc(b['channel'])}</b> is the strongest channel: "
                f"{b['roi_mean']:.2f} ({b['roi_lower']:.2f}–{b['roi_upper']:.2f}) "
                f"{_esc(b['label'])}.</li>"
            )
            crossing = [
                r
                for r in rows
                if r["roi_lower"] < r.get("reference", 1.0) < r["roi_upper"]
            ]
            if crossing:
                bullets.append(
                    "<li>Return is <b>genuinely uncertain</b> (interval spans "
                    f"break-even) for {_esc(', '.join(r['channel'] for r in crossing[:4]))} "
                    "— prime candidates for experiments.</li>"
                )
            w = rows[-1]
            if w is not rows[0] and w["roi_upper"] < w.get("reference", 1.0):
                bullets.append(
                    f"<li><b>{_esc(w['channel'])}</b> is credibly below "
                    f"break-even ({w['roi_mean']:.2f}, upper {w['roi_upper']:.2f}) "
                    "— a reallocation source.</li>"
                )
        share = head.get("media_share")
        blended = head.get("blended_roi")
        if share and blended:
            bullets.append(
                f"<li>Media drives <b>{share['mean']:.0%}</b> of the KPI "
                f"({share['lower']:.0%}–{share['upper']:.0%}) at a blended ROI of "
                f"<b>{blended['mean']:.2f}</b> ({blended['lower']:.2f}–"
                f"{blended['upper']:.2f}).</li>"
            )
        fit = head.get("fit") or {}
        if fit.get("r2") is not None:
            cov = fit.get("coverage90")
            bullets.append(
                f"<li>The model tracks the data with R² {fit['r2']:.2f}"
                + (
                    f" and {cov:.0%} coverage of the 90% predictive band"
                    if cov is not None
                    else ""
                )
                + ".</li>"
            )
        body = (
            f"{self._approx_banner()}"
            f'<p class="lede">{self._insight("standfirst")}</p>'
            f'<ul class="rec">{"".join(bullets)}</ul>'
            f'<p>{self._insight("next_steps")}</p>'
        )
        return _NavEntry("insights", "Primary insights"), self._wrap(
            "insights", "Primary insights", "What this model found", body
        )

    def _section_exec(self) -> tuple[_NavEntry, str]:
        body = (
            f'<p class="lede">{self._insight("exec_gloss")}</p>'
            '<div class="ir-controls" id="execWindow"></div>'
            '<div class="kpi-grid" id="execCards"></div>'
            '<p class="chart-caption">All cards recompute from the embedded '
            "posterior draws for the selected window; intervals are "
            "central credible intervals.</p>"
        )
        return _NavEntry("executive-summary", "Executive summary"), self._wrap(
            "executive-summary",
            "Executive summary",
            "The headline numbers, on your window",
            body,
        )

    def _section_decomp(self) -> tuple[_NavEntry, str]:
        body = (
            f'<p class="lede">{self._insight("decomp_gloss")}</p>'
            '<div class="chart-card"><div id="decompChart"></div></div>'
            '<p class="chart-caption">Posterior-mean incremental contribution '
            "per channel, stacked on the non-media baseline (trend, "
            "seasonality, controls, intercept), against the observed KPI "
            "(dotted). The baseline is the model's predictive mean minus "
            "total media, so the stack always adds up to the fit.</p>"
            "<h3>Share of spend vs share of effect</h3>"
            '<div class="chart-card"><div id="sharesChart"></div></div>'
            '<p class="chart-caption">Where the money goes vs where the '
            "effect comes from (full window). A channel whose effect share "
            "(with credible interval) sits above its spend share is "
            "over-delivering for its budget; below, under-delivering.</p>"
        )
        return _NavEntry("decomposition", "Decomposition"), self._wrap(
            "decomposition",
            "Decomposition",
            "Where the KPI comes from",
            body,
        )

    def _section_fit(self) -> tuple[_NavEntry, str] | None:
        fit = self.facts.get("fit") or {}
        if not fit.get("order"):
            return None
        geo_ctl = (
            '<div class="ir-controls"><span class="ir-lbl">Series</span>'
            '<select class="ir-select" id="fitGeoSelect"></select></div>'
            if len(fit["order"]) > 1
            else '<select id="fitGeoSelect" style="display:none"></select>'
        )
        body = (
            f'<p class="lede">{self._insight("fit_gloss")}</p>'
            f"{geo_ctl}"
            '<div class="chart-card"><div id="fitChart"></div></div>'
            '<p class="chart-caption">Observed KPI (dark line) against the '
            "posterior-predictive distribution: nested 50 / 80 / 90 / 95% "
            "intervals shade from dark to light around the predictive mean."
            "</p>"
            '<div class="kpi-grid" id="fitStats"></div>'
        )
        return _NavEntry("model-fit", "Model fit"), self._wrap(
            "model-fit", "Model fit", "Does the model track the business?", body
        )

    def _section_ppc_stats(self) -> tuple[_NavEntry, str] | None:
        stats = (self.facts.get("ppc_stats") or {}).get("stats") or []
        if not stats:
            return None
        n_extreme = sum(1 for s in stats if s.get("extreme"))
        trs = ""
        for s in stats:
            ok = not s.get("extreme")
            chip = (
                '<span class="tier-chip t-scale">Consistent</span>'
                if ok
                else '<span class="tier-chip t-reduce">Extreme</span>'
            )
            trs += (
                "<tr>"
                f'<td class="chname">{_esc(s["label"])}</td>'
                f"<td>{_esc(s['desc'])}</td>"
                f'<td class="mono">{_fmt(s["observed"], 2)}</td>'
                f'<td class="mono">{_fmt(s["rep_mean"], 2)}</td>'
                f'<td class="mono">{s["bayes_p"]:.2f}</td>'
                f"<td>{chip}</td></tr>"
            )
        n_draws = (self.facts.get("ppc_stats") or {}).get("n_draws", 0)
        verdict = (
            "<p>All test statistics are consistent with the posterior "
            "predictive — the model reproduces these properties of the KPI, "
            "not just its week-by-week level.</p>"
            if n_extreme == 0
            else (
                f"<p><b>{n_extreme} of {len(stats)}</b> test statistics fall "
                "in the extreme tails (p &lt; 0.05 or &gt; 0.95): the model "
                "systematically fails to reproduce "
                f"{_esc(', '.join(s['label'] for s in stats if s['extreme']))}. "
                "Treat forecasts and counterfactuals that lean on those "
                "properties with caution.</p>"
            )
        )
        body = (
            f'<p class="lede">{self._insight("ppc_stats_gloss")}</p>'
            '<table class="data-table"><thead><tr><th>Statistic</th>'
            "<th>What it measures</th><th>Observed</th><th>Replicate mean</th>"
            "<th>Bayes p</th><th>Verdict</th></tr></thead>"
            f"<tbody>{trs}</tbody></table>"
            f"{verdict}"
            '<div class="sat-grid" id="ppcStatsGrid"></div>'
            '<p class="chart-caption">Each panel: the distribution of a test '
            f"statistic across {n_draws} replicated datasets drawn from the "
            "posterior predictive, with the observed value as the vertical "
            "line. The Bayesian p-value is P(T(y_rep) ≥ T(y_obs)); values "
            "near 0 or 1 mean the observed data would be surprising if the "
            "model were true. Computed on the national period-summed KPI.</p>"
            "<h3>Interval calibration</h3>"
            '<div class="chart-card"><div id="calibChart"></div></div>'
            '<p class="chart-caption">Empirical coverage of the 50 / 80 / 90 '
            "/ 95% predictive intervals against their nominal level "
            "(national series). Points on the diagonal mean the model's "
            "uncertainty is neither overstated nor understated.</p>"
        )
        return _NavEntry("predictive-checks", "Predictive checks"), self._wrap(
            "predictive-checks",
            "Validation",
            "Does the model reproduce the KPI's character?",
            body,
        )

    def _section_roi(self) -> tuple[_NavEntry, str]:
        body = (
            f'<p class="lede">{self._insight("roi_gloss")}</p>'
            '<div class="ir-controls" id="roiWindow"></div>'
            '<div class="ir-controls"><label class="ir-toggle" style="margin-left:0">'
            '<input type="checkbox" id="roiYoY"> Split by calendar year '
            "(year-over-year)</label></div>"
            '<div class="chart-card"><div id="roiChart"></div>'
            '<div id="roiChartEff"></div></div>'
            '<p class="chart-caption">Forest plot of per-channel returns over '
            "the selected window: point = posterior mean, thick bar = 50% "
            "interval, thin whisker = the wide interval. The dashed line is "
            "break-even; channels measured in volume (impressions/clicks) "
            "appear in a separate efficiency panel with a zero reference.</p>"
        )
        return _NavEntry("channel-roi", "Channel ROI"), self._wrap(
            "channel-roi", "Channel ROI", "What each channel returned", body
        )

    def _section_yoy(self) -> tuple[_NavEntry, str] | None:
        yoy = self.facts.get("yoy")
        if not yoy or len(yoy.get("years", [])) < 2:
            return None
        body = (
            f'<p class="lede">{self._insight("yoy_gloss")}</p>'
            '<div class="ir-controls">'
            '<span class="ir-lbl">Compare</span>'
            '<select class="ir-select" id="yoyA"></select>'
            '<span class="ir-lbl">→</span>'
            '<select class="ir-select" id="yoyB"></select></div>'
            '<div class="chart-card"><div id="yoyChart"></div></div>'
            '<p class="chart-caption" id="yoyNote"></p>'
            '<p class="chart-caption">Waterfall from the first year\'s total '
            "to the second's: green bars are drivers that added KPI, rust "
            "bars drivers that cost KPI. Media bars are posterior "
            "contribution deltas with credible intervals; the baseline bar "
            "is the residual non-media change, so the bridge always closes "
            "to the observed totals.</p>"
        )
        return _NavEntry("yoy-drivers", "YoY drivers"), self._wrap(
            "yoy-drivers",
            "Year over year",
            "What drove the change vs last year?",
            body,
        )

    def _section_estimands(self) -> tuple[_NavEntry, str]:
        body = (
            f'<p class="lede">{self._insight("estimands_gloss")}</p>'
            '<div class="ir-controls"><span class="ir-lbl">Estimand</span>'
            '<select class="ir-select" id="estimandSelect" style="max-width:320px"></select>'
            "</div>"
            '<div class="ir-controls" id="estimandWindow"></div>'
            '<div class="chart-card"><div id="estimandChart"></div></div>'
            '<p class="chart-caption" id="estimandNote"></p>'
        )
        return _NavEntry("estimands", "Estimands"), self._wrap(
            "estimands",
            "Estimand explorer",
            "One posterior, several causal questions",
            body,
        )

    def _section_curves(self) -> tuple[_NavEntry, str]:
        body = (
            f'<p class="lede">{self._insight("curves_gloss")}</p>'
            '<div class="ir-controls">'
            '<span class="ir-lbl">Channel</span>'
            '<select class="ir-select" id="curveChannelSelect"></select>'
            '<span id="curveModeCtl" style="display:inline-flex;gap:.5rem">'
            '<button type="button" class="ir-btn active" data-curvemode="response">Response</button>'
            '<button type="button" class="ir-btn" data-curvemode="roi">ROI</button>'
            '<button type="button" class="ir-btn" data-curvemode="mroi">Marginal ROI</button>'
            "</span></div>"
            '<div class="sat-grid" id="curvesGrid"></div>'
            '<p class="chart-caption">Bands are posterior uncertainty about '
            "each curve; the dotted gold line marks the channel's current "
            "average weekly spend. Curves re-evaluate the fitted model at "
            "proportionally scaled spend histories (carryover respected), so "
            "they are total-budget response curves, not single-week ones.</p>"
        )
        return _NavEntry("response-curves", "Response curves"), self._wrap(
            "response-curves",
            "Response, ROI and marginal ROI curves",
            "Where the next dollar works hardest",
            body,
        )

    def _section_carryover(self) -> tuple[_NavEntry, str] | None:
        if not self.facts.get("carryover"):
            return None
        body = (
            f'<p class="lede">{self._insight("carryover_gloss")}</p>'
            '<div class="sat-grid" id="carryoverGrid"></div>'
            '<p class="chart-caption">Posterior adstock kernels: the share of '
            "a week's effect landing in each subsequent week, with credible "
            "bands and the implied carryover half-life.</p>"
        )
        return _NavEntry("carryover", "Carryover"), self._wrap(
            "carryover", "Carryover effects", "How long each channel echoes", body
        )

    def _section_pathways(self) -> tuple[_NavEntry, str] | None:
        med = self.facts.get("mediation")
        if not med or not med.get("links"):
            return None
        neg = ""
        if med.get("negatives"):
            neg = (
                '<p class="note">Negative flows shown as magnitude (see '
                "hover): "
                f"{_esc(', '.join(med['negatives']))}.</p>"
            )
        body = (
            f'<p class="lede">{self._insight("pathways_gloss")}</p>'
            '<div class="chart-card"><div id="pathwaysChart"></div></div>'
            '<p class="chart-caption">How each channel\'s effect reaches '
            f"{_esc(med.get('outcome', 'the KPI'))}: flows through "
            f"{_esc(', '.join(med.get('mediators', [])) or 'the mediator')} "
            "are indirect (mediated) effects, flows straight to the outcome "
            "are direct. Link widths are posterior means in "
            f"{_esc(med.get('units', 'effect units'))}; hover any link for "
            "its credible interval.</p>"
            f"{neg}"
        )
        return _NavEntry("pathways", "Effect pathways"), self._wrap(
            "pathways",
            "Structural pathways",
            "How the effect flows: direct vs mediated",
            body,
        )

    def _section_latent(self) -> tuple[_NavEntry, str] | None:
        lat = self.facts.get("latent")
        if not lat:
            return None
        parts = [f'<p class="lede">{self._insight("latent_gloss")}</p>']
        if lat.get("loadings"):
            parts.append(
                "<h3>Factor loadings</h3>"
                '<div class="chart-card"><div id="latentLoadings"></div></div>'
                '<p class="chart-caption">How strongly each indicator moves '
                "with the latent factor (posterior mean with credible "
                "interval). Signs matter: a negative loading means the "
                "indicator falls when the factor rises.</p>"
            )
        if lat.get("trajectories"):
            parts.append(
                "<h3>Latent states over time</h3>"
                '<div class="sat-grid" id="latentGrid"></div>'
                '<p class="chart-caption">Posterior median and credible band '
                "of each latent state, on the model's latent scale. These "
                "are the unobserved quantities the model inferred from the "
                "indicators and the KPI jointly.</p>"
            )
        return _NavEntry("latent-structure", "Latent structure"), self._wrap(
            "latent-structure",
            "Latent structure",
            "The unobserved drivers the model inferred",
            "".join(parts),
        )

    def _section_prior_posterior(self) -> tuple[_NavEntry, str] | None:
        rows = (self.facts.get("prior_posterior") or {}).get("rows") or []
        if not rows:
            return None
        body = (
            f'<p class="lede">{self._insight("prior_posterior_gloss")}</p>'
            '<div class="ir-controls">'
            '<span class="ir-lbl">Channel</span>'
            '<select class="ir-select" id="ppChannelSelect"></select></div>'
            '<div class="sat-grid" id="ppGrid"></div>'
            '<p class="chart-caption">Prior (dotted grey) vs posterior '
            "(filled) densities of each channel's return — the estimand "
            "scale a decision-maker acts on, not raw coefficients. The "
            "dashed line is break-even.</p>"
        )
        return _NavEntry("prior-posterior", "Prior → posterior"), self._wrap(
            "prior-posterior",
            "Prior → posterior",
            "What the data changed our mind about",
            body,
        )

    def _section_realloc(self) -> tuple[_NavEntry, str] | None:
        if not (self.facts.get("curves") or {}).get("draws_b64"):
            return None
        body = (
            f'<p class="lede">{self._insight("realloc_gloss")}</p>'
            "<h3>Reallocate the budget"
            '<span class="chip-approx">Approximate</span></h3>'
            '<div class="ir-controls">'
            '<button type="button" class="ir-btn" id="reallocOptimize">Optimize (budget-neutral)</button>'
            '<button type="button" class="ir-btn" id="reallocReset">Reset to current</button>'
            "</div>"
            '<div id="reallocRows"></div>'
            '<div class="kpi-grid" id="reallocCards" style="margin-top:1.1rem"></div>'
            '<p class="chart-caption">Expected outcomes interpolate the '
            "posterior between precomputed spend levels (0–2× per channel), "
            "so results are approximate. The optimizer moves budget toward "
            "equal marginal return, holding total spend fixed; it assumes "
            "historical flighting scales proportionally and channels do not "
            "interact.</p>"
        )
        return _NavEntry("reallocation", "Budget reallocation"), self._wrap(
            "reallocation",
            "Budget reallocation",
            "What-if: move the money",
            body,
        )

    def _section_sensitivity(self) -> tuple[_NavEntry, str] | None:
        sens = self.facts.get("sensitivity") or {}
        if not sens.get("specs"):
            return None
        notes = "".join(f"<li>{_esc(n)}</li>" for n in sens.get("notes", []))
        body = (
            f'<p class="lede">{self._insight("sensitivity_gloss")}</p>'
            '<div class="chart-card"><div id="sensChart"></div>'
            '<div id="sensChartEff"></div></div>'
            '<p class="chart-caption">Each column is an alternative analysis '
            "choice; whiskers are credible intervals. Findings that hold "
            "across every column are robust; estimates that swing are "
            "fragile and worth validating experimentally.</p>"
            f'<ul class="rec">{notes}</ul>'
        )
        return _NavEntry("sensitivity", "Sensitivity"), self._wrap(
            "sensitivity",
            "Sensitivity analysis",
            "Do the conclusions survive different choices?",
            body,
        )

    def _section_ppc_prior(self) -> tuple[_NavEntry, str] | None:
        ppc = self.facts.get("ppc_prior")
        if not ppc:
            return None
        fan = create_prior_predictive_fan(
            list(ppc["dates"]),
            np.asarray(ppc["observed"], dtype=float),
            {k: np.asarray(v, dtype=float) for k, v in ppc["bands"].items()},
            self.config,
            div_id="priorPredictiveFan",
            kpi_label=str(ppc.get("kpi_label", "KPI")),
            sample_traces=ppc.get("traces"),
        )
        mean_hist = create_prior_stat_distribution(
            np.asarray(ppc["rep_means"], dtype=float),
            float(ppc["obs_mean"]),
            self.config,
            div_id="priorRepMeans",
            stat_label="replicate mean",
        )
        sd_hist = create_prior_stat_distribution(
            np.asarray(ppc["rep_sds"], dtype=float),
            float(ppc["obs_sd"]),
            self.config,
            div_id="priorRepSds",
            stat_label="replicate std. dev.",
        )
        cov = ppc.get("coverage_90")
        kpis = (
            '<div class="kpi-grid">'
            f'<div class="kpi"><div class="label">Prior 90% band coverage</div>'
            f'<div class="value">{cov * 100:.0f}%</div>'
            f'<div class="ci">of observed periods, before fitting</div></div>'
            f'<div class="kpi"><div class="label">Simulated datasets</div>'
            f'<div class="value">{ppc.get("n_draws", 0)}</div>'
            f'<div class="ci">{(ppc.get("frac_negative") or 0) * 100:.1f}% of draws negative</div></div>'
            "</div>"
            if cov is not None
            else ""
        )
        body = (
            f'<p class="lede">{self._insight("ppc_gloss")}</p>'
            f"{kpis}"
            f'<div class="chart-card">{fan}</div>'
            '<p class="chart-caption">The prior-predictive fan from the '
            "model design: observed KPI against datasets simulated before "
            "any fitting. A posterior reached from a loose prior fan earns "
            "more trust than one the priors had already decided.</p>"
            '<div class="chart-grid-2">'
            f'<div class="chart-card">{mean_hist}</div>'
            f'<div class="chart-card">{sd_hist}</div>'
            "</div>"
        )
        return _NavEntry("prior-predictive", "Prior predictive"), self._wrap(
            "prior-predictive",
            "Prior predictive checks",
            "Could this model have produced our data?",
            body,
        )

    def _section_assumptions(self) -> tuple[_NavEntry, str]:
        rows = self.facts.get("assumptions") or []
        table = ""
        if rows:
            trs = "".join(
                "<tr>"
                f'<td class="chname">{_esc(r.get("topic", ""))}</td>'
                f"<td>{_esc(r.get('setting', ''))}</td>"
                f"<td>{_esc(r.get('detail', ''))}</td>"
                f'<td class="mono">{_esc(", ".join(r.get("channels", []) or []))}</td>'
                "</tr>"
                for r in rows
            )
            table = (
                '<h3>Model specification</h3><table class="data-table">'
                "<thead><tr><th>Topic</th><th>Setting</th><th>Detail</th>"
                f"<th>Channels</th></tr></thead><tbody>{trs}</tbody></table>"
            )
        diag = self.facts.get("diagnostics") or {}
        meta = self.facts.get("meta", {})
        approx = bool(meta.get("approximate"))

        def _dcell(label: str, value: str, ci: str) -> str:
            return (
                f'<div class="kpi"><div class="label">{_esc(label)}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="ci">{_esc(ci)}</div></div>'
            )

        rhat = diag.get("rhat_max")
        ess = diag.get("ess_bulk_min")
        div = diag.get("divergences")
        cards = (
            '<div class="kpi-grid">'
            + _dcell(
                "Inference",
                _esc(str(meta.get("fit_method") or "nuts").upper())
                + ('<span class="chip-approx">Approx.</span>' if approx else ""),
                "uncertainty not calibrated" if approx else "full MCMC posterior",
            )
            + _dcell(
                "Max R-hat",
                "—" if rhat is None else f"{float(rhat):.3f}",
                "n/a for approximate fits" if approx else "should be < 1.01",
            )
            + _dcell(
                "Min bulk ESS",
                "—" if ess is None else f"{float(ess):,.0f}",
                "n/a for approximate fits" if approx else "should be > 400",
            )
            + _dcell(
                "Divergences",
                "—" if div is None else f"{int(div)}",
                "post-warmup transitions",
            )
            + "</div>"
        )

        n_exp = len(getattr(self.model, "experiments", []) or []) if self.model else 0
        exp_bullet = (
            f"<li><b>Experiment calibration:</b> {n_exp} experiment"
            f"{'s' if n_exp != 1 else ''} anchor this model's channel effects "
            "to randomized evidence.</li>"
            if n_exp
            else "<li><b>Experiment calibration:</b> no channel is yet anchored "
            "to randomized evidence — the returns above rest on observational "
            "identification alone.</li>"
        )
        causal = (
            "<h3>Causal assumptions behind these numbers</h3>"
            "<ul class='rec'>"
            "<li><b>Declared confounders controlled:</b> the controls listed "
            "in the specification enter the model directly; anything not "
            "listed is assumed not to jointly drive spend and the KPI.</li>"
            "<li><b>No unobserved demand-chasing:</b> given those controls, "
            "spend is treated as good-as-random. If budgets follow demand the "
            "model cannot see, channel returns are biased upward.</li>"
            "<li><b>Functional form:</b> the fitted saturation and carryover "
            "families above. The sensitivity section varies analysis choices "
            "but cannot rule out misspecification outside this family.</li>"
            f"{exp_bullet}"
            "</ul>"
            "<p class='note'>Stating these assumptions does not validate them "
            "— it converts an implicit leap of faith into an explicit, "
            "reviewable claim.</p>"
        )
        body = (
            f'<p class="lede">{self._insight("assumptions_gloss")}</p>'
            f"{cards}{table}{causal}"
        )
        return _NavEntry("assumptions", "Assumptions & methodology"), self._wrap(
            "assumptions",
            "Causal assumptions & methodology",
            "What these numbers rest on",
            body,
        )

    # ── assembly ─────────────────────────────────────────────────────────────
    def _payload_json(self) -> str:
        payload = {k: self.facts.get(k) for k in _PAYLOAD_KEYS}
        return json.dumps(payload, cls=NumpyEncoder).replace("</", "<\\/")

    def _theme_json(self) -> str:
        channels = self.facts.get("meta", {}).get("channels", [])
        theme = {
            "font": getattr(
                self.config.color_scheme,
                "font_sans",
                '"IBM Plex Sans", system-ui, sans-serif',
            ),
            "ink": "#3a4838",
            "muted": "#7a8a78",
            "grid": "#e8e4d5",
            "accent": "#5a7a3a",
            "gold": "#b8860b",
            "rust": "#a04535",
            "channels": {ch: self.channel_colors.get(ch) for ch in channels},
        }
        return json.dumps(theme).replace("</", "<\\/")

    def _assemble(self, sections: list[tuple[_NavEntry, str]]) -> str:
        cfg = self.config
        meta = self.facts.get("meta", {})
        generated_date = cfg.generated_date or datetime.now().strftime("%B %Y")

        meta_bits = []
        if cfg.client:
            meta_bits.append(f"Prepared for {_esc(cfg.client)}")
        if cfg.subtitle:
            meta_bits.append(_esc(cfg.subtitle))
        period = cfg.analysis_period or (
            f"{meta.get('date_start')} – {meta.get('date_end')}"
            if meta.get("date_start")
            else ""
        )
        if period:
            meta_bits.append(_esc(period))
        meta_bits.append(_esc(generated_date))
        meta_line = '<span class="sep">·</span>'.join(meta_bits)
        conf = '<div class="conf">Confidential</div>' if cfg.confidential else ""
        header = (
            '<header class="report-header">'
            f'<div class="masthead-logo">{MASTHEAD_LOGO_SVG}</div>'
            '<div class="masthead-text">'
            '<div class="masthead-eyebrow">Marketing mix modeling · '
            "Results readout · Interactive</div>"
            f"<h1>{_esc(cfg.title)}</h1>"
            f'<div class="meta">{meta_line}</div>{conf}</div></header>'
        )

        nav_items = "".join(
            f'<a class="nav-item" href="#{entry.section_id}">'
            f'<span class="nav-num">{i:02d}</span>{_esc(entry.title)}</a>'
            for i, (entry, _) in enumerate(sections, start=1)
        )
        nav_html = (
            '<nav class="report-nav" id="reportNav" aria-label="Report contents">'
            '<div class="nav-head">Contents</div>'
            f"{nav_items}</nav>"
        )

        footer = (
            '<footer class="report-footer">'
            "<p>Every number in this report derives from one fitted posterior. "
            "The window, geo and estimand selectors re-aggregate the embedded "
            "posterior draws in your browser — they never refit the model. "
            "Reallocation results interpolate between precomputed spend levels "
            "and are labelled approximate. "
            f"Generated {_esc(generated_date)} with the MMM Framework.</p></footer>"
        )

        sections_html = "".join(body for _, body in sections)
        content = (
            f'<div class="report-body">{nav_html}'
            f'<div class="report-container">{header}{sections_html}{footer}</div></div>'
        )

        plotly_script = (
            f'<script src="https://cdn.plot.ly/plotly-{cfg.plotly_cdn_version}.min.js"></script>'
            if cfg.include_plotly_js
            else ""
        )
        scrollspy = """
<script>
(function() {
  var items = Array.prototype.slice.call(document.querySelectorAll('.report-nav .nav-item'));
  if (!items.length) return;
  var sections = items.map(function(a){ return document.getElementById(a.getAttribute('href').slice(1)); }).filter(Boolean);
  function setActive(id){ items.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href').slice(1)===id); }); }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(e){ if (e.isIntersecting) setActive(e.target.id); });
    }, { rootMargin:'-18% 0px -72% 0px', threshold:0 });
    sections.forEach(function(s){ io.observe(s); });
  }
  if (sections[0]) setActive(sections[0].id);
})();
</script>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_esc(cfg.title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    {AUGUR_FONTS_LINK}
    {plotly_script}
    <style>
{augur_css(cfg.color_scheme)}
{_EXTRA_CSS}
    </style>
</head>
<body>
    {content}
    {scrollspy}
    <script>window.__IR_DATA__ = {self._payload_json()};</script>
    <script>window.__IR_THEME__ = {self._theme_json()};</script>
    <script>
{INTERACTIVE_REPORT_JS}
    </script>
</body>
</html>"""

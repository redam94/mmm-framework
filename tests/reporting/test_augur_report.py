"""Tests for the Augur "Media Performance Readout" shell, theme and sections.

Covers the new ``shell="augur"`` path end-to-end: the editorial masthead +
numbered contents nav, the cream/ink theme + chart fonts, the evidence-coded
scorecard, the two posterior-predictive sections (fit-over-time + checks), CMO
insight slots, HTML-escaping of channel/title names, and that the classic
default shell is left untouched.
"""

from __future__ import annotations

import re

import numpy as np

from mmm_framework.reporting.config import (
    ChartConfig,
    ColorPalette,
    ColorScheme,
    ReportConfig,
)
from mmm_framework.reporting.extractors.bundle import MMMDataBundle
from mmm_framework.reporting.generator import MMMReportGenerator, ReportBuilder

_CHS = ["Video", "Print", "Radio", "Social", "Display", "Search", "TV"]
_ROI = {
    "Video": (1.52, 1.08, 2.08),
    "Print": (1.66, 0.46, 3.22),
    "Radio": (1.08, 0.31, 2.07),
    "Social": (0.83, 0.50, 1.23),
    "Display": (0.71, 0.36, 1.13),
    "Search": (0.68, 0.41, 1.00),
    "TV": (0.45, 0.35, 0.57),
}
_SPEND = {
    "Video": 4300,
    "Print": 2100,
    "Radio": 3200,
    "Social": 6100,
    "Display": 5000,
    "Search": 7600,
    "TV": 11300,
}
_MROAS = {
    "Video": 1.49,
    "Print": 0.70,
    "Radio": 0.48,
    "Social": 0.53,
    "Display": 0.54,
    "Search": 0.42,
    "TV": 0.57,
}


def _full_bundle(channels=None) -> MMMDataBundle:
    channels = channels or _CHS
    rng = np.random.default_rng(0)
    n = 52
    dates = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(7 * i, "D") for i in range(n)]
    )
    actual = 20000 + 3000 * np.sin(np.arange(n) / 8) + rng.normal(0, 600, n)
    pmean = actual + rng.normal(0, 300, n)
    sat, ad = {}, {}
    for ch in channels:
        x = np.linspace(0, _SPEND.get(ch, 4000) * 2, 30)
        K = _SPEND.get(ch, 4000) * 0.7
        sat[ch] = {"spend": x, "response": 5.0 * x / (K + x)}
        a = np.array([1.0, 0.5, 0.25, 0.12, 0.06])
        ad[ch] = a / a.sum()
    return MMMDataBundle(
        dates=dates,
        actual=actual,
        predicted={"mean": pmean, "lower": pmean - 1200, "upper": pmean + 1200},
        fit_statistics={"r2": 0.91, "mape": 0.06},
        total_revenue=127000,
        marketing_attributed_revenue={"mean": 32100, "lower": 17500, "upper": 49700},
        blended_roi={"mean": 0.81, "lower": 0.44, "upper": 1.26},
        marketing_contribution_pct={"mean": 0.253, "lower": 0.138, "upper": 0.392},
        channel_roi={
            ch: {
                "mean": _ROI.get(ch, (1.0, 0.5, 1.5))[0],
                "lower": _ROI.get(ch, (1.0, 0.5, 1.5))[1],
                "upper": _ROI.get(ch, (1.0, 0.5, 1.5))[2],
            }
            for ch in channels
        },
        channel_spend={ch: _SPEND.get(ch, 4000) for ch in channels},
        channel_contribution={
            ch: {"mean": 5000.0, "lower": 3000.0, "upper": 8000.0} for ch in channels
        },
        component_totals={"Baseline": 95000, **{ch: 5000 for ch in channels}},
        component_time_series={
            "Baseline": actual * 0.75,
            **{ch: np.full(n, 100.0) for ch in channels},
        },
        saturation_curves=sat,
        adstock_curves=ad,
        current_spend={ch: _SPEND.get(ch, 4000) for ch in channels},
        estimands={
            f"marginal_roas:{ch}": {"mean": _MROAS.get(ch, 0.9), "status": "ok"}
            for ch in channels
        },
        posterior_predictive={
            "observed": actual,
            "pred_mean": pmean,
            "pred_lower": pmean - 1200,
            "pred_upper": pmean + 1200,
            "samples": pmean[None, :] + rng.normal(0, 400, (40, n)),
            "coverage": [
                {"nominal": p, "empirical": min(1.0, p + 0.02)} for p in (0.5, 0.8, 0.9)
            ],
            "bayes_p": {"mean": 0.48, "std": 0.55, "min": 0.12, "max": 0.88},
            "r2": 0.91,
            "ci_level": 0.8,
        },
        channel_names=list(channels),
    )


def _render(bundle=None, **builder_kw) -> str:
    bundle = bundle if bundle is not None else _full_bundle()
    builder = (
        ReportBuilder()
        .with_data(bundle)
        .with_title("Media Performance Readout")
        .with_client("Acme Corp")
        .augur_readout()
    )
    return builder.build().render()


# ── config / theme ───────────────────────────────────────────────────────────
class TestAugurConfig:
    def test_augur_readout_factory(self):
        cfg = ReportConfig.augur_readout(title="T", client="C")
        assert cfg.shell == "augur"
        assert cfg.color_scheme.primary == "#5a7a3a"
        assert cfg.show_nav and cfg.confidential and cfg.format_channel_names

    def test_chart_font_follows_scheme(self):
        augur = ColorScheme.from_palette(ColorPalette.AUGUR)
        layout = ChartConfig().to_plotly_layout(augur)
        assert "IBM Plex Sans" in layout["font"]["family"]

    def test_default_chart_font_unchanged(self):
        # the default (sage) scheme keeps the historical chart font byte-for-byte
        layout = ChartConfig().to_plotly_layout(ColorScheme())
        assert layout["font"]["family"] == "Source Sans 3, sans-serif"


# ── shell ────────────────────────────────────────────────────────────────────
class TestAugurShell:
    def test_masthead_and_numbered_nav(self):
        html = _render()
        assert "masthead-logo" in html
        assert 'class="report-nav"' in html
        assert 'class="nav-num"' in html
        assert ">01<" in html  # first nav number

    def test_fonts_and_theme(self):
        html = _render()
        assert "Fraunces" in html and "IBM+Plex+Sans" in html
        assert "JetBrains+Mono" in html
        assert "--sage-700" in html  # Augur :root present
        assert ".tier-chip" in html

    def test_injection_vectors_preserved(self):
        html = _render()
        assert "cdn.plot.ly/plotly" in html  # plotly script in head
        assert "<style>" in html and "</style>" in html  # css landed
        assert "Plotly.newPlot" in html  # section charts landed in body
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_title_and_client_escaped(self):
        bundle = _full_bundle()
        html = (
            ReportBuilder()
            .with_data(bundle)
            .with_title("<script>alert('x')</script>")
            .with_client("<b>Evil</b>")
            .augur_readout()
            .build()
            .render()
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert "<b>Evil</b>" not in html

    def test_confidential_badge(self):
        html = _render()
        assert "Confidential" in html

    def test_section_and_nav_counts_match(self):
        html = _render()
        n_sections = len(re.findall(r'<section class="section"', html))
        n_nav = len(re.findall(r'class="nav-item"', html))
        assert n_sections == n_nav
        assert n_sections >= 12  # full template renders the whole set


# ── sections ─────────────────────────────────────────────────────────────────
class TestAugurSections:
    def test_headline_uses_insight(self):
        html = _render()
        # templated headline names the scale winner + a reduce loser
        assert re.search(r"<h2>Scale Video[^<]*</h2>", html)

    def test_kpi_strip(self):
        html = _render()
        assert 'class="kpi-grid"' in html
        assert "$32.1K" in html  # marketing-attributed revenue
        assert "$0.81" in html  # blended return as a ratio, not $1

    def test_scorecard_tiers(self):
        html = _render()
        # Video is a scale row; TV is a reduce row
        assert re.search(r"Video.*?tier-chip t-scale", html, re.S)
        assert re.search(r"TV.*?tier-chip t-reduce", html, re.S)

    def test_reallocation_cards(self):
        html = _render()
        assert 'class="realloc t-scale"' in html
        assert 'class="realloc t-reduce"' in html

    def test_ppc_timeseries_fit_section(self):
        html = _render()
        assert 'id="ppc-fit"' in html
        assert "augurPPCFit" in html
        assert "Does the model track reality" in html

    def test_ppc_checks_section(self):
        html = _render()
        assert 'id="ppc-checks"' in html
        assert "augurPPCObsPred" in html  # observed vs predicted
        assert "augurPPCResid" in html  # residuals
        # bayes-p table
        assert "Predictive p-values" in html

    def test_deep_dives(self):
        html = _render()
        assert 'class="dd-kpis"' in html
        assert "Carryover half-life" in html

    def test_marginal_section_present_with_mroas(self):
        html = _render()
        assert 'id="marginal"' in html
        assert "augurMarginal" in html

    def test_recommended_tests_and_next_steps(self):
        html = _render()
        assert 'id="tests"' in html
        assert 'class="loop"' in html  # measurement-loop chips
        assert 'class="next-steps"' in html

    def test_flighting_illustrative_tagged(self):
        html = _render()
        assert 'id="flighting"' in html
        assert "illus-tag" in html  # marked illustrative

    def test_channel_name_xss_escaped_in_scorecard(self):
        # The scorecard is a pure HTML-text context (no embedded chart JSON),
        # so a malicious channel name must be escaped there.
        bundle = _full_bundle(channels=["X<img src=x>", "Search", "TV"])
        html = _render(bundle)
        m = re.search(
            r'<section class="section" id="scorecard">.*?</section>', html, re.S
        )
        assert m, "scorecard section present"
        scorecard = m.group(0)
        assert "<img" not in scorecard  # not injected raw
        assert "&lt;img" in scorecard  # escaped


# ── graceful degradation ─────────────────────────────────────────────────────
class TestAugurDegradation:
    def test_ppc_fit_absent_without_timeseries(self):
        bundle = _full_bundle()
        bundle.predicted = None
        bundle.actual = None
        html = _render(bundle)
        assert "augurPPCFit" not in html
        # the rest still renders
        assert 'id="scorecard"' in html

    def test_marginal_absent_without_mroas(self):
        bundle = _full_bundle()
        bundle.estimands = None
        html = _render(bundle)
        assert "augurMarginal" not in html

    def test_marginal_no_none_when_partial_mroas(self):
        # only some channels have a marginal ROAS -> the chart must chart ONLY
        # those (no null y-values reach Plotly's hovertemplate).
        bundle = _full_bundle()
        bundle.estimands = {"marginal_roas:Video": {"mean": 1.49, "status": "ok"}}
        html = _render(bundle)
        assert "augurMarginal" in html
        m = re.search(r'Plotly\.newPlot\(\s*"augurMarginal",\s*(\[.*?\]),', html, re.S)
        assert m, "marginal chart present"
        assert "null" not in m.group(1)  # no None in the trace arrays

    def test_deep_dive_id_is_selector_safe(self):
        bundle = _full_bundle(channels=["Brand & Performance", "Search", "TV"])
        html = _render(bundle)
        # id slug is sanitized (no entities / spaces / ampersands)
        assert 'id="dd-brand-performance"' in html
        assert "dd-brand-&" not in html and "&amp;amp;" not in html

    def test_minimal_bundle_renders(self):
        bundle = MMMDataBundle(
            channel_roi={"A": {"mean": 1.2, "lower": 1.05, "upper": 1.4}},
            channel_spend={"A": 1000.0},
            channel_names=["A"],
        )
        html = _render(bundle)
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "masthead-logo" in html
        assert 'id="scorecard"' in html


# ── default shell untouched ──────────────────────────────────────────────────
class TestDefaultShellUnchanged:
    def test_default_has_no_augur_markup(self):
        bundle = _full_bundle()
        html = MMMReportGenerator(
            data=bundle, config=ReportConfig(title="Classic")
        ).render()
        assert "masthead-logo" not in html
        assert 'class="nav-num"' not in html
        assert "--sage-700" not in html

    def test_insights_not_computed_for_default(self):
        cfg = ReportConfig(title="Classic")
        MMMReportGenerator(data=_full_bundle(), config=cfg)
        assert cfg.cmo_insights == {}

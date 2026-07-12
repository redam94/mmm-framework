"""Tests for the triangulation panel — MMM × experiment × platform (issue #104)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.triangulation import (
    TriangulationSource as S,
    build_triangulation,
    reconcile_channel,
    triangulation_from_model,
)


# ---------------------------------------------------------------------------
# Reconciliation logic
# ---------------------------------------------------------------------------
class TestReconcile:
    def test_convergent_when_experiment_and_mmm_agree(self):
        c = reconcile_channel(
            "Social", [S("experiment", 2.1, 1.8, 2.4), S("mmm", 2.2, 1.9, 2.6)]
        )
        assert c.agreement == "convergent"
        assert c.convergent is True
        assert c.reconciled["basis"] == "experiment"  # experiment anchors
        assert any("agree" in n for n in c.notes)

    def test_divergent_when_intervals_disjoint(self):
        c = reconcile_channel(
            "TV", [S("experiment", 1.2, 1.0, 1.4), S("mmm", 2.0, 1.7, 2.4)]
        )
        assert c.agreement == "divergent"
        assert any("disagree" in n for n in c.notes)
        assert c.reconciled["value"] == 1.2  # anchors on the experiment

    def test_platform_inflated(self):
        c = reconcile_channel(
            "Search",
            [
                S("experiment", 1.9, 1.6, 2.2),
                S("mmm", 1.8, 1.4, 2.3),
                S("platform", 3.5, incremental=False, attribution_window="7-day click"),
            ],
        )
        assert c.agreement == "platform-inflated"
        assert any("last-touch" in n for n in c.notes)
        # Still reconciles on the incremental experiment, not the platform figure.
        assert c.reconciled["value"] == 1.9

    def test_single_source_mmm_only(self):
        c = reconcile_channel("Radio", [S("mmm", 1.5, 0.9, 2.1)])
        assert c.agreement == "single-source"
        assert c.reconciled["basis"] == "mmm"
        assert any("model-identified" in n for n in c.notes)

    def test_incremental_divergence_dominates_platform_inflation(self):
        # MMM and experiment disagree AND platform is high → still "divergent"
        # (the incremental disagreement is the bigger story).
        c = reconcile_channel(
            "TV",
            [
                S("experiment", 1.0, 0.8, 1.2),
                S("mmm", 2.5, 2.1, 2.9),
                S("platform", 4.0, incremental=False),
            ],
        )
        assert c.agreement == "divergent"

    def test_points_close_without_intervals(self):
        # No CIs → fall back to relative tolerance (30%).
        c = reconcile_channel("X", [S("experiment", 2.0), S("mmm", 2.2)])
        assert c.agreement == "convergent"

    def test_platform_only_is_single_source(self):
        c = reconcile_channel("Y", [S("platform", 3.0, incremental=False)])
        assert c.agreement == "single-source"


class TestBuildTriangulation:
    def test_summary_counts(self):
        res = build_triangulation(
            {
                "A": [S("experiment", 2.0, 1.8, 2.2), S("mmm", 2.1, 1.9, 2.3)],
                "B": [S("mmm", 1.5, 1.0, 2.0)],
            }
        )
        d = res.to_dict()
        assert d["summary"]["n_channels"] == 2
        assert d["summary"]["by_agreement"]["convergent"] == 1
        assert d["summary"]["by_agreement"]["single-source"] == 1

    def test_empty_channels_skipped(self):
        res = build_triangulation({"A": [], "B": [S("mmm", 2.0)]})
        assert len(res.channels) == 1


# ---------------------------------------------------------------------------
# Builder from a (fake) model
# ---------------------------------------------------------------------------
class _FakeEstimandResult:
    def __init__(self, mean, lo, hi):
        self.kind = "roi"
        self.status = "ok"
        self.mean = mean
        self.hdi_low = lo
        self.hdi_high = hi
        self.hdi_prob = 0.94
        self.units = ""
        self.extra = {}


class _FakeExp:
    def __init__(self, channel, value, se, estimand="roas"):
        self.channel = channel
        self.value = value
        self.se = se
        self.estimand = estimand


class _FakeModel:
    def __init__(self, estimands, experiments):
        self.channel_names = ["TV", "Search"]
        self._estimands = estimands
        self.experiments = experiments

    def evaluate_estimands(self, *a, **k):
        return self._estimands


class TestBuilder:
    def _model(self):
        return _FakeModel(
            estimands={
                "contribution_roi:TV": _FakeEstimandResult(2.0, 1.6, 2.4),
                "contribution_roi:Search": _FakeEstimandResult(1.8, 1.4, 2.2),
                "marginal_roas:TV": _FakeEstimandResult(1.5, 1.0, 2.0),  # ignored
            },
            experiments=[_FakeExp("Search", 1.9, 0.15, "roas")],
        )

    def test_mmm_and_experiment_joined(self):
        tri = triangulation_from_model(self._model())
        by_ch = {c.channel: c for c in tri.channels}
        # Search has MMM + experiment → convergent.
        assert by_ch["Search"].agreement == "convergent"
        # TV has MMM only → single-source.
        assert by_ch["TV"].agreement == "single-source"

    def test_platform_dict_ingested(self):
        tri = triangulation_from_model(
            self._model(),
            platform={"Search": {"value": 3.6, "attribution_window": "7-day click"}},
        )
        search = next(c for c in tri.channels if c.channel == "Search")
        assert search.agreement == "platform-inflated"
        assert any(s.source == "platform" for s in search.sources)

    def test_contribution_estimand_experiment_excluded(self):
        # A raw-contribution experiment readout is off the return-per-$ axis.
        m = _FakeModel(
            estimands={"contribution_roi:TV": _FakeEstimandResult(2.0, 1.6, 2.4)},
            experiments=[_FakeExp("TV", 5000.0, 200.0, "contribution")],
        )
        tri = triangulation_from_model(m)
        tv = next(c for c in tri.channels if c.channel == "TV")
        assert all(s.source != "experiment" for s in tv.sources)
        assert tv.agreement == "single-source"

    def test_channel_measured_only_by_experiment_appears(self):
        m = _FakeModel(
            estimands={"contribution_roi:TV": _FakeEstimandResult(2.0, 1.6, 2.4)},
            experiments=[_FakeExp("Podcast", 2.5, 0.3, "roas")],
        )
        tri = triangulation_from_model(m)
        assert any(c.channel == "Podcast" for c in tri.channels)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _tri_payload():
    return build_triangulation(
        {
            "Search": [
                S("experiment", 1.9, 1.6, 2.2),
                S("mmm", 1.8, 1.4, 2.3),
                S("platform", 3.5, incremental=False, attribution_window="7-day click"),
            ],
            "TV": [S("experiment", 1.2, 1.0, 1.4), S("mmm", 2.0, 1.7, 2.4)],
            "Radio": [S("mmm", 1.5, 0.9, 2.1)],
        }
    ).to_dict()


class TestReportSection:
    def test_section_renders(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["Search", "TV", "Radio"]),
            config=ReportConfig(),
            triangulation=_tri_payload(),
        ).render()
        assert "Triangulation — MMM" in html
        assert "triangulationPlot" in html
        assert "Platform-inflated" in html
        assert "Divergent" in html
        assert "Why the sources differ" in html

    def test_section_absent_without_data(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV"]), config=ReportConfig()
        ).render()
        assert "Triangulation — MMM" not in html

    def test_section_html_escapes_channel_names(self):
        # The section's table + notes (the HTML-injection surface) must escape a
        # malicious channel name — matching the #102 XSS hardening. (Plotly tick
        # text is JSON-in-<script>, rendered as non-HTML SVG text.)
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        tri = build_triangulation({"<b>X</b>": [S("mmm", 2.0, 1.5, 2.5)]}).to_dict()
        html = MMMReportGenerator(
            data=MMMDataBundle(), config=ReportConfig(), triangulation=tri
        ).render()
        # Escaped form present (the table cell / notes heading escaped it)…
        assert "&lt;b&gt;X&lt;/b&gt;" in html
        # …and no live <b> tag injected into the document body markup.
        assert "<b>X</b>" not in html

    def test_chart_returns_html(self):
        from mmm_framework.reporting import ReportConfig
        from mmm_framework.reporting.charts import create_triangulation_chart

        div = create_triangulation_chart(_tri_payload(), ReportConfig())
        assert "Plotly.newPlot" in div or "plotly" in div.lower()


# ---------------------------------------------------------------------------
# Real-fit end-to-end (slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_triangulation_from_real_model():
    from mmm_framework.calibration.likelihood import (
        ExperimentEstimand,
        ExperimentMeasurement,
    )
    from mmm_framework.config import (
        DimensionType,
        InferenceMethod,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
        ModelConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    periods = pd.date_range("2021-01-04", periods=48, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    t = np.arange(n)
    tv = np.abs(rng.normal(120, 30, n))
    search = np.abs(rng.normal(60, 20, n))
    y = pd.Series(
        1000 + 8 * t + 1.4 * tv + 2.2 * search + rng.normal(0, 25, n), name="Sales"
    )
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Search"],
        controls=None,
    )
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Search", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Search": search}),
        X_controls=None,
        coords=coords,
        index=periods,
        config=cfg,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=120,
            n_tune=120,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.add_experiment_calibration(
        [
            ExperimentMeasurement(
                channel="Search",
                test_period=(periods[10], periods[20]),
                value=2.2,
                se=0.3,
                estimand=ExperimentEstimand.ROAS,
            )
        ]
    )
    model.fit(random_seed=0)

    tri = triangulation_from_model(
        model, platform={"Search": {"value": 4.0, "attribution_window": "7-day click"}}
    )
    by_ch = {c.channel: c for c in tri.channels}
    assert "Search" in by_ch and "TV" in by_ch
    # Search has MMM + experiment + platform.
    search_sources = {s.source for s in by_ch["Search"].sources}
    assert {"mmm", "experiment", "platform"} <= search_sources
    # TV has only the MMM (no experiment, no platform).
    assert by_ch["TV"].agreement == "single-source"
    # Every channel has a reconciled recommendation.
    assert all(c.reconciled.get("value") is not None for c in tri.channels)

"""Tests for the short-term vs long-term / brand section (issue #106)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mmm_framework.reporting.helpers.longterm import (
    build_long_term_facts,
    carryover_split,
    long_term_scenario,
)


class TestCarryoverSplit:
    def test_basic_split(self):
        s = carryover_split([0.5, 0.25, 0.125, 0.0625])
        assert s is not None
        assert abs(s["immediate_pct"] + s["carryover_pct"] - 1.0) < 1e-9
        assert s["immediate_pct"] > 0.5  # normalized weight[0]

    def test_all_immediate(self):
        s = carryover_split([1.0])
        assert s["immediate_pct"] == 1.0
        assert s["carryover_pct"] == 0.0
        assert s["effective_weeks"] == 1

    def test_effective_weeks_counts_material_lags(self):
        s = carryover_split([0.5, 0.3, 0.19, 0.005, 0.005])
        assert s["effective_weeks"] == 3  # last two < 1%

    def test_bad_input_returns_none(self):
        assert carryover_split(None) is None
        assert carryover_split([]) is None
        assert carryover_split([0.0, 0.0]) is None


class TestScenario:
    def test_multiplier(self):
        sc = long_term_scenario(100.0, 2.0)
        assert sc["long_term"] == 200.0
        assert sc["uplift"] == 100.0
        assert sc["multiplier"] == 2.0


class TestBuildFacts:
    def _adstock(self):
        return {"TV": [0.5, 0.25, 0.13, 0.06, 0.03], "Search": [0.9, 0.08, 0.02]}

    def test_per_channel_and_blend(self):
        facts = build_long_term_facts(
            ["TV", "Search"],
            self._adstock(),
            contribution={"TV": 100000.0, "Search": 60000.0},
        )
        assert facts["measured"] == "carryover"
        tv = next(r for r in facts["channels"] if r["channel"] == "TV")
        assert tv["carryover_pct"] > 0.4  # TV is carryover-heavy
        assert "immediate_contribution" in tv
        assert "blended" in facts

    def test_scenario_added_with_multiplier(self):
        facts = build_long_term_facts(
            ["TV"], self._adstock(), contribution={"TV": 100000.0}, multiplier=2.0
        )
        tv = facts["channels"][0]
        assert tv["scenario_long_term"] == 200000.0
        assert facts["blended"]["scenario_uplift"] == 100000.0

    def test_caveat_only_when_no_adstock(self):
        facts = build_long_term_facts(["TV"], None)
        assert facts["measured"] == "none"
        assert facts["channels"] == []

    def test_funnel_flag(self):
        facts = build_long_term_facts(["TV"], None, has_structural_funnel=True)
        assert facts["has_structural_funnel"] is True


class TestExtractor:
    def test_extractor_stamps_bundle(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle
        from mmm_framework.reporting.extractors.mixins import EstimandPPCMixin

        class _Ex(EstimandPPCMixin):
            def __init__(self, model):
                self.mmm = model
                self.model = model

        class _M:
            channel_names = ["TV", "Search"]
            mediator_names = None

        bundle = MMMDataBundle(
            channel_names=["TV", "Search"],
            adstock_curves={
                "TV": np.array([0.5, 0.3, 0.2]),
                "Search": np.array([0.9, 0.1]),
            },
            component_totals={"TV": 100000.0, "Search": 60000.0},
        )
        bundle = _Ex(_M())._extract_long_term(bundle)
        assert bundle.long_term is not None
        assert len(bundle.long_term["channels"]) == 2


def _bundle():
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle

    lt = build_long_term_facts(
        ["TV", "Search"],
        {"TV": [0.5, 0.25, 0.13, 0.06, 0.03], "Search": [0.9, 0.08, 0.02]},
        contribution={"TV": 100000.0, "Search": 60000.0},
    )
    return MMMDataBundle(channel_names=["TV", "Search"], long_term=lt)


class TestSectionRender:
    def _html(self, cfg=None):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig

        return MMMReportGenerator(data=_bundle(), config=cfg or ReportConfig()).render()

    def test_caveat_and_split_render(self):
        html = self._html()
        assert "Short-term vs long-term" in html
        assert "not measured here" in html or "does not measure" in html.lower()
        assert "under-credits brand" in html
        assert "Immediate vs carryover" in html
        assert "Measuring long-term effects" in html

    def test_scenario_renders_with_multiplier(self):
        from mmm_framework.reporting import ReportConfig

        cfg = dataclasses.replace(ReportConfig(), long_term_multiplier=2.0)
        html = self._html(cfg)
        assert "Long-term scenario" in html
        assert "assumption" in html.lower()
        assert "With 2× long-term" in html

    def test_scenario_absent_without_multiplier(self):
        assert "Long-term scenario" not in self._html()

    def test_section_absent_without_data(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["TV"]), config=ReportConfig()
        ).render()
        assert "Short-term vs long-term" not in html

    def test_channel_names_escaped(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        lt = build_long_term_facts(["<b>X</b>"], {"<b>X</b>": [0.6, 0.4]})
        html = MMMReportGenerator(
            data=MMMDataBundle(channel_names=["<b>X</b>"], long_term=lt),
            config=ReportConfig(),
        ).render()
        assert "<b>X</b>" not in html
        assert "&lt;b&gt;X&lt;/b&gt;" in html


@pytest.mark.slow
def test_real_fit_long_term_extracted():
    import pandas as pd

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
    from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

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
    model.fit(random_seed=0)
    bundle = BayesianMMMExtractor(
        model, model._results if hasattr(model, "_results") else None
    ).extract()
    assert bundle.long_term is not None
    # At least one channel produced an immediate/carryover split.
    assert bundle.long_term["channels"]
    row = bundle.long_term["channels"][0]
    assert 0.0 <= row["immediate_pct"] <= 1.0
    assert abs(row["immediate_pct"] + row["carryover_pct"] - 1.0) < 1e-6

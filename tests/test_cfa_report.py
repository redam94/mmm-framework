"""The CFA HTML report section: a non-MMM (CFA) model renders a Factor Analysis
section (loadings table + fit-index cards) while the MMM-only channel/ROI/
decomposition/saturation sections gate off. Also covers the MAP-fit diagnostics
guard (R-hat/ESS are None for an approximate fit)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from bayesian_cfa import BayesianCFA, synthetic_cfa_panel  # noqa: E402

from mmm_framework.config import ModelConfig  # noqa: E402
from mmm_framework.model import TrendConfig  # noqa: E402
from mmm_framework.model.trend_config import TrendType  # noqa: E402


@pytest.fixture(scope="module")
def cfa_html():
    from mmm_framework.reporting import MMMReportGenerator

    panel, _ = synthetic_cfa_panel(n=300)
    mmm = BayesianCFA(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params={"n_factors": 2, "factor_assignment": [0, 0, 0, 1, 1, 1]},
    )
    mmm.fit(method="map", random_seed=7)
    gen = MMMReportGenerator(model=mmm)
    return gen, gen.render()


@pytest.mark.slow
class TestCFAReport:
    def test_extractor_routes_to_factor_analysis(self, cfa_html):
        from mmm_framework.reporting.extractors import FactorAnalysisExtractor

        gen, _ = cfa_html
        # The bundle was produced by the FactorAnalysisExtractor (model_kind=cfa).
        assert gen.data.model_kind == "cfa"
        assert gen.data.factor_loadings  # non-empty list of loading rows
        assert gen.data.cfa_fit_indices  # srmr / cov_fit
        assert FactorAnalysisExtractor is not None

    def test_factor_analysis_section_rendered(self, cfa_html):
        _, html = cfa_html
        assert 'id="factor-analysis"' in html
        assert "Confirmatory Factor Analysis" in html
        assert "Factor loadings" in html and "Fit indices" in html
        assert "SRMR" in html and "COV_FIT" in html
        assert "x1" in html  # an indicator row

    def test_mmm_sections_gated_off(self, cfa_html):
        _, html = cfa_html
        for sid in ('id="channel-roi"', 'id="decomposition"', 'id="saturation"'):
            assert sid not in html, sid

    def test_diagnostics_section_handles_map_fit(self, cfa_html):
        _, html = cfa_html
        # MAP fit -> R-hat/ESS are None; the diagnostics section degrades to N/A
        # rather than crashing.
        assert "N/A" in html


def test_mmm_report_still_has_channel_sections(tmp_path):
    """Guard: an ordinary MMM report is unaffected — channel sections present,
    factor-analysis absent."""
    import numpy as np
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.model import BayesianMMM
    from mmm_framework.reporting import MMMReportGenerator

    periods = pd.date_range("2021-01-04", periods=30, freq="W-MON")
    rng = np.random.default_rng(3)
    tv = np.abs(rng.normal(100, 25, 30))
    y = pd.Series(1000 + tv + rng.normal(0, 15, 30), name="Sales")
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV"],
            controls=None,
        ),
        index=periods,
        config=cfg,
    )
    mmm = BayesianMMM(panel, ModelConfig())
    mmm.fit(method="map", random_seed=3)
    gen = MMMReportGenerator(model=mmm)
    assert gen.data.model_kind == "mmm"
    html = gen.render()
    assert 'id="factor-analysis"' not in html  # gated off for an MMM

"""Deck chart renderers + deterministic slide-data engine (PR 2, template-free).

Fast tests render each matplotlib chart to PNG bytes (no model needed) and check
they are valid images; the slow test builds the whole :class:`Deck` from a fitted
model and asserts the slide structure, the ROI/mROI zone content, and that charts
were produced — all with no AI and no PowerPoint dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.reporting.deck import build_deck, charts
from mmm_framework.reporting.helpers.results import SpendResponseZones

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _fake_zones(channel: str = "TV") -> SpendResponseZones:
    x = np.linspace(0.0, 100.0, 60)
    mroi = 3.0 * np.exp(-x / 40.0)  # decreasing marginal ROI
    resp = 200.0 * (1.0 - np.exp(-x / 40.0))
    roi = np.where(x > 0, resp / np.maximum(x, 1e-9), mroi)
    return SpendResponseZones(
        channel=channel,
        spend_grid=x,
        response_mean=resp,
        response_lower=resp * 0.9,
        response_upper=resp * 1.1,
        roi_mean=roi,
        roi_lower=roi * 0.9,
        roi_upper=roi * 1.1,
        mroi_mean=mroi,
        mroi_lower=mroi * 0.9,
        mroi_upper=mroi * 1.1,
        current_spend=50.0,
        current_response=float(np.interp(50.0, x, resp)),
        current_roi=float(np.interp(50.0, x, roi)),
        current_roi_hdi=(1.0, 2.0),
        current_mroi=float(np.interp(50.0, x, mroi)),
        current_mroi_hdi=(0.8, 1.6),
        break_even=1.0,
        band=0.15,
        breakthrough_range=(0.0, 20.0),
        optimal_range=(20.0, 55.0),
        saturation_range=(55.0, 100.0),
        optimal_spend=40.0,
        optimal_roi=1.5,
        current_zone="optimal",
        recommendation="hold",
        headroom_to_optimal=-10.0,
    )


# ---------------------------------------------------------------------------
# fast: chart renderers produce valid PNGs (no model)
# ---------------------------------------------------------------------------


def test_saturation_zones_png_is_valid():
    png = charts.saturation_zones_png(_fake_zones(), currency="$")
    assert isinstance(png, bytes) and png.startswith(_PNG_MAGIC) and len(png) > 2000


def test_roi_forest_png_is_valid():
    roi = {
        "TV": {"mean": 2.1, "lower": 1.4, "upper": 2.9},
        "Search": {"mean": 3.4, "lower": 2.6, "upper": 4.3},
        "Social": {"mean": 0.8, "lower": 0.3, "upper": 1.3},
    }
    png = charts.roi_forest_png(roi, break_even=1.0)
    assert png.startswith(_PNG_MAGIC) and len(png) > 2000


def test_decomposition_png_is_valid():
    png = charts.decomposition_png({"Base": 5000.0, "TV": 1200.0, "Search": 900.0})
    assert png.startswith(_PNG_MAGIC)


def test_fit_png_is_valid():
    n = 52
    actual = np.linspace(100, 140, n) + np.random.RandomState(0).normal(0, 5, n)
    pred = {
        "mean": np.linspace(100, 140, n),
        "lower": np.linspace(90, 130, n),
        "upper": np.linspace(110, 150, n),
    }
    png = charts.fit_png(np.arange(n), actual, pred, r2=0.82)
    assert png.startswith(_PNG_MAGIC)


def test_reallocation_png_is_valid():
    rows = [
        {"channel": "TV", "current": 50.0, "optimal": 40.0},
        {"channel": "Search", "current": 30.0, "optimal": 55.0},
        {"channel": "Social", "current": 20.0, "optimal": None},  # no in-range optimum
    ]
    png = charts.reallocation_png(rows)
    assert png.startswith(_PNG_MAGIC)


# ---------------------------------------------------------------------------
# slow: the full deck engine on a fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model():
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig
    from mmm_framework.model.trend_config import TrendType
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=0, n_weeks=104).panel()
    mmm = BayesianMMM(
        panel,
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
    )
    mmm.fit(
        draws=300,
        tune=600,
        chains=2,
        target_accept=0.9,
        random_seed=3,
        progressbar=False,
    )
    return mmm


@pytest.mark.slow
class TestDeckEngine:
    def test_build_deck_structure(self, fitted_model):
        deck = build_deck(fitted_model, client="Acme", kpi_name="Sales", currency="$")
        kinds = [s.kind for s in deck.slides]
        # the spine of the deck is always present
        for k in (
            "title",
            "executive_summary",
            "channel_roi",
            "saturation",
            "optimization",
            "methodology",
        ):
            assert k in kinds, (k, kinds)
        # summary slides are flagged for the whole-deck synthesis pass (PR 3)
        summaries = [s.kind for s in deck.slides if s.is_summary]
        assert "executive_summary" in summaries and "optimization" in summaries
        # every slide carries deterministic insight-context but NO AI insight yet
        assert all(s.notes for s in deck.slides)
        assert all(s.insight is None for s in deck.slides)

    def test_saturation_slides_carry_zone_data_and_charts(self, fitted_model):
        deck = build_deck(fitted_model, kpi_name="Sales")
        sat = [s for s in deck.slides if s.kind == "saturation"]
        assert len(sat) >= 1
        for s in sat:
            assert s.chart_png and s.chart_png.startswith(_PNG_MAGIC)
            z = s.metrics["zones"]
            # the zones are ROI/mROI-based, not % of response
            assert z["current_zone"] in ("breakthrough", "optimal", "saturation")
            assert z["recommendation"] in ("increase", "hold", "reduce")
            assert "optimal_range" in z and "break_even" in z
            # the slide's subtitle states the deterministic recommendation
            assert "Recommendation:" in s.subtitle
            # the deck uses the DEFAULT (adaptive) spend range, which must extend
            # far enough that the optimal knee and saturation onset are actually
            # on the chart — otherwise the whole axis collapses to "breakthrough"
            # with no optimal marker (the bug this guards against). So every
            # deep-dive must carry a present optimal point and non-degenerate
            # optimal + saturation zones.
            opt_lo, opt_hi = z["optimal_range"]
            sat_lo, sat_hi = z["saturation_range"]
            assert z["optimal_spend"] is not None, (z["channel"], z)
            assert z["headroom_to_optimal"] is not None
            assert opt_hi > opt_lo, ("empty optimal zone", z["channel"], z)
            assert sat_hi > sat_lo, ("empty saturation zone", z["channel"], z)

    def test_margin_sets_break_even(self, fitted_model):
        deck = build_deck(
            fitted_model, margin=0.5
        )  # break-even ROI reference = 1/0.5 = 2.0
        # margin sets the ROI/mROI reference level used on the chart + carried in meta
        # (the zones themselves are elasticity-based, not break-even-based)
        assert abs(deck.meta["break_even"] - 2.0) < 1e-9
        method = next(s for s in deck.slides if s.kind == "methodology")
        assert "50%" in "".join(method.bullets)  # the gross margin surfaces in the copy

    def test_to_dict_is_serializable(self, fitted_model):
        import json

        deck = build_deck(fitted_model, kpi_name="Sales")
        d = deck.to_dict(include_charts=True)
        # charts are base64 strings; the whole thing round-trips through JSON
        blob = json.dumps(d)
        assert isinstance(blob, str) and len(blob) > 1000
        sat = next(s for s in d["slides"] if s["kind"] == "saturation")
        assert sat["chart_png_b64"] and isinstance(sat["chart_png_b64"], str)

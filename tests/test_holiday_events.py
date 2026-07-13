"""Holiday / event effects — #143.

A first-class way to declare sharp, date-specific effects (Black Friday, a
launch) that the smooth Fourier seasonality cannot represent. Contract:

* the regressor helper builds windowed/decayed columns from named holidays +
  custom events, without hand-authored dummies;
* off by default — no event block, graph byte-identical (R0.1);
* events enter as an additive ``event_component`` distinct from seasonality and
  appear as a SEPARATE line in the decomposition/report (AC2);
* a planted spike is attributed to the event.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.config import (
    AdstockConfig,
    DimensionType,
    EventsConfig,
    EventSpec,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import TrendType
from mmm_framework.transforms import build_event_regressors

PERIODS = pd.date_range("2023-01-02", periods=104, freq="W-MON")


# --------------------------------------------------------------------------- #
# regressor helper
# --------------------------------------------------------------------------- #
def test_custom_event_window_and_decay():
    cfg = EventsConfig(
        custom_events=[
            EventSpec(name="Launch", dates=["2023-06-19"], post_weeks=3, decay=0.5)
        ]
    )
    X = build_event_regressors(PERIODS, cfg)
    assert list(X.columns) == ["Launch"]
    col = X["Launch"].to_numpy()
    peak = int(np.argmax(col))
    assert col[peak] == 1.0
    # geometric decay over the 3 post-weeks
    np.testing.assert_allclose(col[peak + 1 : peak + 4], [0.5, 0.25, 0.125], rtol=1e-9)


def test_named_us_holidays():
    holidays = pytest.importorskip("holidays")  # noqa: F841
    cfg = EventsConfig(country="US", holidays=["Christmas", "Thanksgiving"])
    X = build_event_regressors(PERIODS, cfg)
    assert any("Christmas" in c for c in X.columns)
    assert any("Thanksgiving" in c for c in X.columns)
    # two years of data -> each holiday peaks twice
    xmas = X[[c for c in X.columns if "Christmas" in c][0]]
    assert int((xmas == 1.0).sum()) == 2


def test_empty_and_out_of_window():
    assert EventsConfig().is_empty()
    # an event far outside the data window contributes no column
    cfg = EventsConfig(custom_events=[EventSpec(name="X", dates=["2099-01-01"])])
    X = build_event_regressors(PERIODS, cfg)
    assert X.shape[1] == 0


# --------------------------------------------------------------------------- #
# model wiring
# --------------------------------------------------------------------------- #
def _panel(spike_at: int | None = None):
    rng = np.random.default_rng(0)
    chans = ["TV", "Search"]
    X = pd.DataFrame({c: np.abs(rng.normal(100, 30, 104)) for c in chans})
    y = 1000.0 + 2 * X["TV"] + 1.5 * X["Search"] + rng.normal(0, 30, 104)
    if spike_at is not None:
        y.iloc[spike_at] += 500.0
    coords = PanelCoordinates(
        periods=PERIODS, geographies=None, products=None, channels=chans, controls=[]
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(
                name=c, dimensions=[DimensionType.PERIOD], adstock=AdstockConfig.none()
            )
            for c in chans
        ],
        controls=[],
    )
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=X,
        X_controls=None,
        coords=coords,
        index=PERIODS,
        config=config,
    )


def _events(week: int):
    return EventsConfig(
        custom_events=[EventSpec(name="Launch", dates=[str(PERIODS[week].date())])]
    )


def test_off_has_no_event_block():
    m = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    assert "beta_events" not in {v.name for v in m.model.free_RVs}
    assert "event_component" not in set(m.model.named_vars)


def test_on_builds_event_block():
    m = BayesianMMM(
        _panel(),
        ModelConfig(events=_events(50)),
        TrendConfig(type=TrendType.LINEAR),
    )
    assert "beta_events" in {v.name for v in m.model.free_RVs}
    assert "event_component" in set(m.model.named_vars)


def test_events_out_of_window_warns_and_no_block():
    ev = EventsConfig(custom_events=[EventSpec(name="X", dates=["2099-01-01"])])
    with pytest.warns(UserWarning, match="no holiday/event"):
        m = BayesianMMM(
            _panel(), ModelConfig(events=ev), TrendConfig(type=TrendType.LINEAR)
        )
    assert "beta_events" not in {v.name for v in m.model.free_RVs}


@pytest.mark.slow
class TestEventFit:
    def _fit(self, panel, events):
        cfg = ModelConfigBuilder().map_fit().with_events(events).build()
        m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m

    def test_spike_attributed_to_event_and_waterfall_closes(self):
        m = self._fit(_panel(spike_at=50), _events(50))
        # positive event coefficient captures the spike
        be = m._trace.posterior["beta_events"].mean(dim=["chain", "draw"]).values
        assert float(be[0]) > 0

        dec = m.compute_component_decomposition()
        assert dec.total_events is not None and dec.total_events > 0
        # the waterfall closes exactly (events included)
        total = (
            dec.intercept
            + dec.trend
            + dec.seasonality
            + dec.media_total
            + dec.controls_total
            + dec.events
        )
        post = m._trace.posterior

        def cm(v):
            return (
                post[v].mean(dim=["chain", "draw"]).values
                if v in post
                else np.zeros(m.n_obs)
            )

        expected = m.y_mean + m.y_std * (
            cm("intercept_component")
            + cm("trend_component")
            + cm("seasonality_component")
            + cm("media_total")
            + cm("controls_total")
            + cm("event_component")
        )
        np.testing.assert_allclose(total, expected, rtol=1e-8)
        assert "Events / Holidays" in dec.summary()["Component"].tolist()

    def test_reporting_separates_events_from_seasonality(self):
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        m = self._fit(_panel(spike_at=50), _events(50))
        totals = BayesianMMMExtractor(m)._get_component_totals()
        assert "Events" in totals  # distinct from "Seasonality"

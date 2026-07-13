"""Price & promotion first-class levers — #138.

Promote a control column to a price lever (log-price elasticity, ≤ 0) or a promo
lever (lift with own carryover). Contract:

* off by default — price/promo are ordinary linear controls (no lever RVs, R0.1);
* on ⇒ the named column is REMOVED from the linear control block (no double
  count), the price lever emits a sign-guarded ``price_elasticity``, the promo
  lever a non-negative ``beta_promo_<var>`` with its own adstock;
* the levers are a separate decomposition line (waterfall closes);
* recovers a planted negative elasticity + positive promo lift.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework import BayesianMMM, ModelConfig, ModelConfigBuilder, TrendConfig
from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    PriceConfig,
    PromoConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import TrendType


def _geom(x, a):
    o = np.zeros_like(x, dtype=float)
    acc = 0.0
    for t in range(len(x)):
        acc = x[t] + a * acc
        o[t] = acc
    return o


def _panel(*, with_effect: bool = True, n: int = 130, seed: int = 3) -> PanelDataset:
    rng = np.random.default_rng(seed)
    per = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    chans = ["TV", "Search"]
    X = pd.DataFrame({c: np.abs(rng.normal(100, 30, n)) for c in chans})
    price = 10.0 + rng.normal(0, 1.5, n)
    ref = float(np.median(price))
    promo = (rng.random(n) < 0.25) * rng.uniform(0.2, 0.6, n)
    y = 1000.0 + 2 * X["TV"] + 1.5 * X["Search"] + rng.normal(0, 25, n)
    if with_effect:
        y = (
            y
            - 320.0 * np.log(price / ref)  # negative price elasticity
            + 140.0 * _geom(promo / promo.max(), 0.4)  # promo lift + carryover
        )
    ctrls = pd.DataFrame({"Price": price, "Promo": promo})
    coords = PanelCoordinates(
        periods=per,
        geographies=None,
        products=None,
        channels=chans,
        controls=["Price", "Promo"],
    )
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(
                name=c, dimensions=[DimensionType.PERIOD], adstock=AdstockConfig.none()
            )
            for c in chans
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD]),
            ControlVariableConfig(name="Promo", dimensions=[DimensionType.PERIOD]),
        ],
    )
    return PanelDataset(
        y=pd.Series(y, name="Sales"),
        X_media=X,
        X_controls=ctrls,
        coords=coords,
        index=per,
        config=cfg,
    )


def _levers():
    return dict(
        price=PriceConfig(variable="Price", reference="median"),
        promotions=[PromoConfig(variable="Promo", adstock_lmax=4)],
    )


def test_off_price_is_a_linear_control():
    m = BayesianMMM(_panel(), ModelConfig(), TrendConfig(type=TrendType.LINEAR))
    named = set(m.model.named_vars)
    assert "price_elasticity" not in named and "lever_component" not in named
    assert m.n_controls == 2  # Price + Promo remain linear controls


def test_on_promotes_levers_and_excludes_from_controls():
    m = BayesianMMM(
        _panel(), ModelConfig(**_levers()), TrendConfig(type=TrendType.LINEAR)
    )
    named = set(m.model.named_vars)
    free = {v.name for v in m.model.free_RVs}
    assert "price_elasticity" in named  # sign-guarded elasticity Deterministic
    assert "price_elasticity_mag" in free
    assert "beta_promo_Promo" in free
    assert "promo_alpha_Promo" in free  # own carryover
    assert "lever_component" in named
    # levers removed from the linear control block (no double-count)
    assert m.control_names == []
    assert m.n_controls == 0


def test_unknown_lever_warns():
    with pytest.warns(UserWarning, match="not a control column"):
        BayesianMMM(
            _panel(),
            ModelConfig(price=PriceConfig(variable="Nope")),
            TrendConfig(type=TrendType.LINEAR),
        )


@pytest.mark.slow
class TestLeverFit:
    def _fit(self):
        cfg = (
            ModelConfigBuilder()
            .map_fit()
            .with_price(_levers()["price"])
            .with_promotions(*_levers()["promotions"])
            .build()
        )
        m = BayesianMMM(_panel(), cfg, TrendConfig(type=TrendType.LINEAR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(random_seed=0)
        return m

    def test_recovers_negative_elasticity_and_positive_promo(self):
        m = self._fit()
        pe = float(m._trace.posterior["price_elasticity"].mean())
        bp = float(m._trace.posterior["beta_promo_Promo"].mean())
        assert pe < 0  # planted price elasticity is negative
        assert bp > 0  # planted promo lift is positive

    def test_decomposition_has_levers_and_closes(self):
        m = self._fit()
        dec = m.compute_component_decomposition()
        assert dec.total_levers is not None
        total = (
            dec.intercept
            + dec.trend
            + dec.seasonality
            + dec.media_total
            + dec.controls_total
            + dec.levers
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
            + cm("lever_component")
        )
        np.testing.assert_allclose(total, expected, rtol=1e-8)
        assert "Price & Promotion" in dec.summary()["Component"].tolist()

        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        assert "Price & Promotion" in BayesianMMMExtractor(m)._get_component_totals()

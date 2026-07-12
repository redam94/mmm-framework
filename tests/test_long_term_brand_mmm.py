"""Long-term brand estimator — dual-decay brand-equity garden model (issue #122).

Fast tests build the graph / prior-predictive / MAP without MCMC; a @slow NUTS
test proves the model recovers a NONZERO, substantial long-term (brand) share of
the media effect on a synthetic world with a known short-vs-long split.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from long_term_brand_mmm import LongTermBrandMMM, LongTermBrandParams  # noqa: E402


def _geom_adstock(x: np.ndarray, rho: float) -> np.ndarray:
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        for k in range(t + 1):
            out[t] += rho**k * x[t - k]
    return out


def _brand_world(n: int = 156, seed: int = 7):
    """A national series with a KNOWN fast (activation) + slow (brand) split."""
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    tv = np.abs(rng.normal(100, 30, n)) * (1 + 0.5 * np.sin(np.arange(n) / 6))
    dig = np.abs(rng.normal(80, 25, n))
    tvn, dign = tv / tv.max(), dig / dig.max()
    fast = 2.0 * _geom_adstock(tvn, 0.3) + 1.5 * _geom_adstock(dign, 0.3)
    slow = 1.6 * _geom_adstock(tvn, 0.94) + 1.2 * _geom_adstock(dign, 0.94)
    true_fraction = float(slow.sum() / (fast.sum() + slow.sum()))
    y = pd.Series(50 + fast + slow + rng.normal(0, 0.8, n), name="Sales")
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": dig}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=None,
        ),
        index=periods,
        config=cfg,
    )
    return panel, true_fraction


def _model(**overrides):
    panel, _ = _brand_world()
    return LongTermBrandMMM(
        panel,
        ModelConfig(),
        TrendConfig(type=TrendType.NONE),
        model_params=overrides or None,
    )


class TestConfigAndGraph:
    def test_config_schema_defaults(self):
        mmm = _model()
        assert isinstance(mmm.model_params, LongTermBrandParams)
        assert mmm.model_params.slow_retention_alpha == 47.0

    def test_bad_param_rejected(self):
        with pytest.raises(Exception):
            _model(brand_effect_sigma=-1)

    def test_graph_registers_split_deterministics(self):
        mmm = _model()
        dets = {d.name for d in mmm.model.deterministics}
        assert {
            "activation_contributions",
            "brand_contributions",
            "channel_contributions",
            "media_total",
            "brand_total",
            "activation_total",
            "long_term_fraction",
        } <= dets
        # two distinct persistence knobs
        free = {v.name for v in mmm.model.free_RVs}
        assert {"activation_retention", "brand_retention"} <= free
        # brand magnitude per channel
        assert {"beta_brand_TV", "beta_brand_Digital"} <= free

    def test_kind_is_mmm(self):
        assert LongTermBrandMMM.__garden_model_kind__ == "mmm"

    def test_prior_predictive_runs(self):
        import pymc as pm

        mmm = _model()
        with mmm.model:
            pp = pm.sample_prior_predictive(draws=20, random_seed=0)
        assert "y_obs" in pp.prior_predictive


class TestFit:
    def test_map_fit_produces_long_term_fraction(self):
        mmm = _model()
        mmm.fit(method="map", random_seed=0)
        assert mmm._trace is not None
        frac = np.asarray(mmm._trace.posterior["long_term_fraction"].values).reshape(-1)
        assert frac.size and np.all((frac >= 0) & (frac <= 1))

    @pytest.mark.slow
    def test_recovers_nonzero_long_term_share(self):
        """The headline #122 proof: on a world with a real long-term component the
        model estimates a NONZERO, substantial brand share — not an assumption."""
        from mmm_framework.config.enums import InferenceMethod

        panel, true_fraction = _brand_world()
        assert true_fraction > 0.3  # the DGP has a genuine long-term component
        mmm = LongTermBrandMMM(
            panel,
            ModelConfig(inference_method=InferenceMethod.BAYESIAN_NUMPYRO),
            TrendConfig(type=TrendType.NONE),
        )
        mmm.fit(draws=400, tune=600, chains=2, random_seed=0)
        post = mmm._trace.posterior
        brand_total = np.asarray(post["brand_total"].values).reshape(-1)
        frac = np.asarray(post["long_term_fraction"].values).reshape(-1)
        # A genuinely-nonzero long-term contribution (an ESTIMATE, not the caveat).
        assert np.mean(brand_total > 0) > 0.95
        # A substantial estimated long-term share that RECOVERS the planted split
        # (this world's slow stock accumulates a large share; est ≈ true).
        assert frac.mean() > 0.5
        assert abs(frac.mean() - true_fraction) < 0.15
        # the planted long-term share sits inside the estimated 90% interval.
        lo, hi = np.percentile(frac, 5), np.percentile(frac, 95)
        assert lo - 0.05 <= true_fraction <= hi + 0.05

"""what_if_scenario carries decision uncertainty (Phase 1 / B3).

A point estimate is not decision-grade in a budget meeting. what_if_scenario now
returns a credible interval for the outcome change and P(scenario beats baseline),
computed from PAIRED posterior draws (the optimizer's machinery).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.agents import model_ops
from mmm_framework.config import (
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM
from mmm_framework.model.trend_config import TrendConfig, TrendType


@pytest.fixture(scope="module")
def fitted_model():
    periods = pd.date_range("2020-01-06", periods=60, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=[],
    )
    tv = np.abs(rng.standard_normal(n) * 40 + 120)
    digital = np.abs(rng.standard_normal(n) * 25 + 90)
    y = pd.Series(800 + 2.0 * tv + 1.5 * digital + rng.standard_normal(n) * 50, name="Sales")
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
        X_controls=None,
        coords=coords,
        index=periods,
        config=cfg,
    )
    mc = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=80, n_tune=50
    )
    mmm = BayesianMMM(panel, mc, TrendConfig(type=TrendType.LINEAR))
    mmm.fit(method="map", random_seed=42)
    return mmm


def test_what_if_returns_uncertainty(fitted_model):
    res = fitted_model.what_if_scenario({"TV": 1.2}, random_seed=0, max_draws=40)
    assert "outcome_change_hdi" in res
    lo, hi = res["outcome_change_hdi"]
    assert lo <= res["outcome_change"] + 1e-6  # point sits within the interval-ish
    assert hi >= res["outcome_change"] - abs(res["outcome_change"]) - 1e-6
    assert lo <= hi
    assert 0.0 <= res["prob_positive"] <= 1.0
    assert res["n_draws"] > 0


def test_increasing_spend_likely_positive(fitted_model):
    # A genuine +20% on a positive-ROI channel should mostly help.
    res = fitted_model.what_if_scenario({"TV": 1.2}, random_seed=0, max_draws=40)
    assert res["outcome_change"] > 0
    assert res["prob_positive"] > 0.5


def test_compute_uncertainty_false_skips(fitted_model):
    res = fitted_model.what_if_scenario(
        {"TV": 1.1}, compute_uncertainty=False, random_seed=0
    )
    assert "outcome_change_hdi" not in res
    assert "outcome_change" in res  # point estimate still present


def test_model_op_surfaces_uncertainty(fitted_model):
    res = model_ops.budget_scenario(fitted_model, spend_changes={"TV": 1.2})
    assert res["error"] is None
    assert "credible interval" in res["content"]
    assert "P(scenario beats baseline)" in res["content"]

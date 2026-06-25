"""Per-channel budget constraints reach the optimizer (Phase 1 / B2).

The optimizer always supported a per-channel ``bounds`` dict, but the agent
surface only exposed a single global min/max multiplier, so a planner could not
encode real constraints (a partner cap, a frozen line). These tests pin that
``bounds`` is threaded through the model-op and validated.
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
    rng = np.random.default_rng(3)
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
    X_media = pd.DataFrame({"TV": tv, "Digital": digital})
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
    )
    panel = PanelDataset(
        y=y, X_media=X_media, X_controls=None, coords=coords, index=periods, config=cfg
    )
    mc = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=50, n_tune=50
    )
    mmm = BayesianMMM(panel, mc, TrendConfig(type=TrendType.LINEAR))
    mmm.fit(method="map", random_seed=42)
    return mmm


def test_bounds_freeze_a_channel(fitted_model):
    res = model_ops.optimize_budget(
        fitted_model, bounds={"TV": [1.0, 1.0]}, max_draws=20
    )
    assert res["error"] is None, res["error"]
    assert "Per-channel constraints applied" in res["content"]
    alloc = res["dashboard"]["budget_optimization"]["allocation"]
    tv = next(r for r in alloc if r["channel"] == "TV")
    # Frozen at current spend -> no change.
    assert abs(float(tv["change_pct"])) < 1e-6


def test_unknown_channel_rejected(fitted_model):
    res = model_ops.optimize_budget(fitted_model, bounds={"Nope": [0.0, 2.0]})
    assert res["error"] is not None
    assert "Unknown channel" in res["error"]


def test_malformed_bounds_rejected(fitted_model):
    res = model_ops.optimize_budget(fitted_model, bounds={"TV": [2.0, 1.0]})  # hi < lo
    assert res["error"] is not None
    assert "low <= high" in res["error"]


def test_no_bounds_still_works(fitted_model):
    res = model_ops.optimize_budget(fitted_model, max_draws=20)
    assert res["error"] is None
    assert "Per-channel constraints applied" not in res["content"]

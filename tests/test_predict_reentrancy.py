"""Regression tests for counterfactual-prediction state restoration.

`predict()` / `sample_channel_contributions()` / `sample_latent_under()` swap
counterfactual data into the model's ``pm.Data`` containers via ``pm.set_data``.
Previously they never restored the training values, so a counterfactual call
left the fitted model *dirty* (the containers held the last scenario's inputs
until some later default call happened to reset them). These tests pin that the
model is returned to its training state after every scenario call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    ControlVariableConfig,
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


@pytest.fixture
def fitted_model():
    periods = pd.date_range("2020-01-06", periods=52, freq="W-MON")
    n = len(periods)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    rng = np.random.default_rng(7)
    y = pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales")
    X_media = pd.DataFrame(
        {
            "TV": np.abs(rng.standard_normal(n) * 50 + 100),
            "Digital": np.abs(rng.standard_normal(n) * 30 + 80),
        }
    )
    X_controls = pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5})
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    panel = PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=periods,
        config=cfg,
    )
    model_config = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=50,
        n_tune=50,
    )
    mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
    mmm.fit(method="map", random_seed=42)
    return mmm


def _media_snapshot(mmm: BayesianMMM) -> dict[str, np.ndarray]:
    """Current values of the media ``pm.Data`` containers (path-agnostic)."""
    with mmm.model:
        if mmm.use_parametric_adstock:
            keys = ["X_media_raw"]
        else:
            keys = ["X_media_low", "X_media_high"]
        return {k: np.asarray(mmm.model[k].get_value()).copy() for k in keys}


def _assert_same(before: dict[str, np.ndarray], after: dict[str, np.ndarray]):
    assert before.keys() == after.keys()
    for k in before:
        np.testing.assert_allclose(
            before[k], after[k], err_msg=f"container {k!r} not restored"
        )


def test_counterfactual_predict_restores_training_data(fitted_model):
    mmm = fitted_model
    before = _media_snapshot(mmm)

    # Counterfactual: zero out TV.
    cf = mmm.X_media_raw.copy()
    cf[:, mmm.channel_names.index("TV")] = 0.0
    mmm.predict(X_media=cf, random_seed=0)

    after = _media_snapshot(mmm)
    _assert_same(before, after)


def test_default_predict_leaves_training_data(fitted_model):
    mmm = fitted_model
    before = _media_snapshot(mmm)
    mmm.predict(random_seed=0)
    after = _media_snapshot(mmm)
    _assert_same(before, after)


def test_sample_channel_contributions_restores(fitted_model):
    mmm = fitted_model
    before = _media_snapshot(mmm)
    cf = mmm.X_media_raw.copy()
    cf[:, mmm.channel_names.index("Digital")] = 0.0
    mmm.sample_channel_contributions(X_media=cf, max_draws=10, random_seed=0)
    after = _media_snapshot(mmm)
    _assert_same(before, after)


def test_counterfactual_contributions_leaves_model_clean(fitted_model):
    """The whole counterfactual loop must end with the model in training state."""
    mmm = fitted_model
    before = _media_snapshot(mmm)
    mmm.compute_counterfactual_contributions(compute_uncertainty=False, random_seed=0)
    after = _media_snapshot(mmm)
    _assert_same(before, after)


def test_counterfactual_predict_value_matches_explicit_baseline(fitted_model):
    """A baseline predict run AFTER a counterfactual must equal a fresh baseline
    (i.e. the counterfactual did not corrupt subsequent predictions)."""
    mmm = fitted_model
    baseline_first = mmm.predict(random_seed=123).y_pred_mean

    cf = mmm.X_media_raw.copy()
    cf[:, mmm.channel_names.index("TV")] = 0.0
    mmm.predict(X_media=cf, random_seed=0)

    baseline_after = mmm.predict(random_seed=123).y_pred_mean
    np.testing.assert_allclose(baseline_first, baseline_after, rtol=1e-6, atol=1e-6)

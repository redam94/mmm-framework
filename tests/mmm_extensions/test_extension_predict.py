"""``predict()`` on the extension models (BayesianMMM parity).

NestedMMM and StructuralNestedMMM gain the standard posterior-predictive
surface — ``PredictionResults`` with per-draw samples, original-scale
rescaling, and counterfactual media via the in-graph data swap. The joint
multi-outcome models refuse loudly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions.config import (
    MediatorConfig,
    MediatorType,
    MultivariateModelConfig,
    NestedModelConfig,
    OutcomeConfig,
)
from mmm_framework.mmm_extensions.models.multivariate import MultivariateMMM
from mmm_framework.mmm_extensions.models.nested import NestedMMM
from mmm_framework.model.results import PredictionResults

N = 60


@pytest.fixture(scope="module")
def idx():
    return pd.date_range("2022-01-03", periods=N, freq="W-MON")


@pytest.fixture(scope="module")
def media():
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(100, 20, (N, 2)))


def _y(media):
    rng = np.random.default_rng(1)
    aware = 40 + 0.3 * media[:, 0] + rng.normal(0, 4, N)
    return 1000 + 4 * aware + 2 * media[:, 1] + rng.normal(0, 40, N)


@pytest.fixture(scope="module")
def fitted_nested(media, idx):
    cfg = NestedModelConfig(
        mediators=(
            MediatorConfig(name="Awareness", mediator_type=MediatorType.FULLY_LATENT),
        )
    )
    m = NestedMMM(media, _y(media), ["TV", "Digital"], cfg, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(method="map", random_seed=0)
    return m


class TestNestedPredict:
    def test_prediction_results_contract(self, fitted_nested, media):
        pred = fitted_nested.predict(random_seed=3)
        assert isinstance(pred, PredictionResults)
        assert pred.y_pred_samples.ndim == 2
        assert pred.y_pred_samples.shape[1] == N
        for arr in (
            pred.y_pred_mean,
            pred.y_pred_std,
            pred.y_pred_hdi_low,
            pred.y_pred_hdi_high,
        ):
            assert arr.shape == (N,)
            assert np.all(np.isfinite(arr))
        assert np.all(pred.y_pred_hdi_low <= pred.y_pred_hdi_high)

    def test_original_scale_matches_kpi_units(self, fitted_nested, media):
        y = _y(media)
        pred = fitted_nested.predict(random_seed=3)
        # Original-scale predictions live at the KPI's magnitude…
        assert abs(pred.y_pred_mean.mean() - y.mean()) < 2 * y.std()
        # …and the standardized ones near zero.
        pred_std = fitted_nested.predict(return_original_scale=False, random_seed=3)
        assert abs(pred_std.y_pred_mean.mean()) < 2.0

    def test_counterfactual_zero_media_lowers_prediction(self, fitted_nested, media):
        base = fitted_nested.predict(random_seed=11)
        cf = fitted_nested.predict(X_media=np.zeros_like(media), random_seed=11)
        # Positive media effects (mediated + direct) → switching media off
        # must lower the paired-seed predicted total.
        assert cf.y_pred_mean.sum() < base.y_pred_mean.sum()
        # and the training data is restored afterwards
        post = fitted_nested.predict(random_seed=11)
        np.testing.assert_allclose(post.y_pred_mean, base.y_pred_mean, rtol=1e-6)

    def test_controls_unsupported_on_nested(self, fitted_nested):
        with pytest.raises(ValueError, match="control"):
            fitted_nested.predict(X_controls=np.zeros((N, 1)))

    def test_unfitted_raises(self, media, idx):
        cfg = NestedModelConfig(
            mediators=(
                MediatorConfig(
                    name="Awareness", mediator_type=MediatorType.FULLY_LATENT
                ),
            )
        )
        m = NestedMMM(media, _y(media), ["TV", "Digital"], cfg, index=idx)
        with pytest.raises(ValueError, match="not fitted"):
            m.predict()


class TestStructuralPredict:
    @pytest.fixture(scope="class")
    def fitted_structural(self):
        import sys
        from pathlib import Path

        sys.path.insert(
            0, str(Path(__file__).resolve().parent)
        )  # reuse the funnel fixture
        from test_structural_nested import _funnel_model

        model, sc = _funnel_model(n_weeks=60, seed=21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(method="map", random_seed=0)
        return model, sc

    def test_prediction_results_contract(self, fitted_structural):
        model, sc = fitted_structural
        pred = model.predict(random_seed=5)
        assert isinstance(pred, PredictionResults)
        assert pred.y_pred_samples.shape[1] == len(sc.y)
        assert np.all(np.isfinite(pred.y_pred_mean))
        y = sc.y.to_numpy(float)
        assert abs(pred.y_pred_mean.mean() - y.mean()) < 2 * y.std()

    def test_counterfactual_zero_media_lowers_prediction(self, fitted_structural):
        model, sc = fitted_structural
        base = model.predict(random_seed=13)
        cf = model.predict(
            X_media=np.zeros_like(sc.spend.to_numpy(float)), random_seed=13
        )
        assert cf.y_pred_mean.sum() < base.y_pred_mean.sum()
        post = model.predict(random_seed=13)
        np.testing.assert_allclose(post.y_pred_mean, base.y_pred_mean, rtol=1e-6)

    def test_controls_swap_supported(self, fitted_structural):
        model, sc = fitted_structural
        xc = sc.controls.to_numpy(float)
        base = model.predict(random_seed=17)
        alt = model.predict(X_controls=xc * 1.5, random_seed=17)
        # Price is a real control in the funnel world: perturbing it must
        # move the prediction, and training data must be restored after.
        assert not np.allclose(alt.y_pred_mean, base.y_pred_mean)
        post = model.predict(random_seed=17)
        np.testing.assert_allclose(post.y_pred_mean, base.y_pred_mean, rtol=1e-6)


class TestMultiOutcomeGuard:
    def test_multivariate_refuses(self, media, idx):
        rng = np.random.default_rng(2)
        outcomes = {
            "A": 1000 + 2 * media[:, 0] + rng.normal(0, 30, N),
            "B": 800 + 1.5 * media[:, 1] + rng.normal(0, 25, N),
        }
        cfg = MultivariateModelConfig(
            outcomes=(
                OutcomeConfig(name="A", column="A"),
                OutcomeConfig(name="B", column="B"),
            )
        )
        m = MultivariateMMM(media, outcomes, ["TV", "Digital"], cfg, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(method="map", random_seed=0)
        with pytest.raises(NotImplementedError, match="multi-outcome|y_obs"):
            m.predict()

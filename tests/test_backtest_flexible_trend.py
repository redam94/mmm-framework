"""Out-of-time backtest for flexible (spline/GP/piecewise) trends (V1).

Flexible trends have no closed-form out-of-time extrapolation, so the forecaster
replays the fitted ``trend_component`` and HOLDS its last value beyond the
training window. The fast test pins that hold-last indexing; the slow tests pin
that the NumPy forward pass reproduces the in-graph posterior predictive for a
spline-trend model (so the trend component is replayed correctly).
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.validation.backtest import PosteriorForecaster


def test_trend_at_holds_last_value_for_flexible_trend():
    # Build a forecaster shell without a fit to unit-test the trend indexing.
    f = PosteriorForecaster.__new__(PosteriorForecaster)
    f._trend_type = "spline"
    f._n_samples = 1
    f._trend_component = np.array([[10.0, 20.0, 30.0, 40.0]])  # (1 sample, 4 train)

    # Within the training window: exact fitted component.
    np.testing.assert_allclose(
        f._trend_at(np.array([0, 1, 2, 3])).ravel(), [10, 20, 30, 40]
    )
    # Beyond training: hold the LAST fitted level.
    np.testing.assert_allclose(f._trend_at(np.array([4, 10])).ravel(), [40, 40])


def test_trend_at_none_is_zero():
    f = PosteriorForecaster.__new__(PosteriorForecaster)
    f._trend_type = "none"
    f._n_samples = 3
    out = f._trend_at(np.array([0, 1, 5]))
    assert out.shape == (3, 3)
    assert not out.any()


# --- slow: real fit reproduces the in-graph forward pass for a spline trend ---
def _spline_model(panel, seed: int = 7):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=200,
        n_tune=200,
        use_parametric_adstock=True,
        optim_seed=seed,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.SPLINE))


@pytest.mark.slow
def test_spline_forward_pass_reproduces_graph():
    from mmm_framework.synth import dgp

    mmm = _spline_model(dgp.build("clean", seed=7, n_weeks=120).panel())
    mmm.fit(random_seed=7, progressbar=False)

    # Previously raised NotImplementedError for a non-linear trend.
    f = PosteriorForecaster(mmm)
    positions = np.arange(mmm.n_periods)
    samples = f.forecast(
        mmm.X_media_raw,
        mmm.X_controls_raw,
        positions,
        include_noise=False,
        random_seed=7,
    )
    mu_numpy = samples.mean(axis=0)
    mu_graph = mmm.predict(random_seed=7).y_pred_mean
    corr = np.corrcoef(mu_numpy, mu_graph)[0, 1]
    assert corr > 0.99, f"spline forward-pass mismatch: corr={corr:.4f}"
    rel = np.abs(mu_numpy - mu_graph).mean() / np.abs(mu_graph).mean()
    assert rel < 0.06, f"spline forward-pass mismatch: rel_err={rel:.3%}"


@pytest.mark.slow
def test_spline_backtest_runs_end_to_end():
    from mmm_framework.synth import dgp
    from mmm_framework.validation.backtest import BacktestConfig, run_backtest

    mmm = _spline_model(dgp.build("clean", seed=7, n_weeks=120).panel())
    cfg = BacktestConfig(min_train_size=80, horizon=8, step=8, draws=150, tune=150)
    result = run_backtest(mmm, cfg, progressbar=False)
    assert result.n_origins >= 1
    assert np.isfinite(result.mape)  # backtest now produces a number for spline

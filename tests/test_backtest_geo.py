"""Geo-panel out-of-time backtest (deferred V1 geo half).

The forecaster reuses the single-cell forward pass PER CELL (effective intercept
= national + per-geo offset; each cell's own media series so adstock carryover
stays within a cell). Obs are period-major / cell-minor. Validated with real
(slow) geo fits — exactly the cost that had this deferred.
"""

from __future__ import annotations

import numpy as np
import pytest


def _geo_model(n_weeks: int = 70):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.synth import dgp_geo

    panel = dgp_geo.make_geo_clean(n_weeks=n_weeks).panel()
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=200,
        n_tune=200,
        use_parametric_adstock=True,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


@pytest.mark.slow
def test_geo_forward_pass_reproduces_graph():
    """The NumPy geo forward pass must reproduce the in-graph posterior predictive
    across all (period, geo) obs."""
    from mmm_framework.validation.backtest import PosteriorForecaster

    mmm = _geo_model()
    mmm.fit(random_seed=7, progressbar=False)
    assert mmm.n_cells > 1

    f = PosteriorForecaster(mmm)  # previously raised for n_cells>1
    samples = f.forecast(
        mmm.X_media_raw,
        mmm.X_controls_raw,
        np.arange(mmm.n_obs),
        include_noise=False,
        random_seed=7,
    )
    mu_numpy = samples.mean(axis=0)
    mu_graph = mmm.predict(random_seed=7).y_pred_mean
    corr = np.corrcoef(mu_numpy, mu_graph)[0, 1]
    assert corr > 0.99, f"geo forward-pass mismatch: corr={corr:.4f}"
    rel = np.abs(mu_numpy - mu_graph).mean() / np.abs(mu_graph).mean()
    assert rel < 0.06, f"geo forward-pass mismatch: rel_err={rel:.3%}"


@pytest.mark.slow
def test_geo_backtest_beats_baselines_across_cells():
    from mmm_framework.validation.backtest import BacktestConfig, run_backtest

    mmm = _geo_model()
    cfg = BacktestConfig(
        min_train_size=50, horizon=8, step=8, draws=150, tune=150, max_origins=2
    )
    res = run_backtest(mmm, cfg, progressbar=False)

    # Every geo is represented in the forecast records.
    assert sorted(res.records["cell"].unique().tolist()) == [0, 1, 2, 3]
    assert res.n_origins >= 1

    s = res.summary()
    assert np.isfinite(s.loc["mmm", "mape"])
    # On the clean panel positive control, the model out-of-time-forecasts better
    # than the seasonal-naive baseline.
    assert s.loc["mmm", "mape"] < s.loc["seasonal_naive", "mape"]

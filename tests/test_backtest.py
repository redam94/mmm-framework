"""Tests for mmm_framework.validation.backtest (rolling-origin backtesting).

Fast tests cover the pure logic (origin generation, metrics, baselines,
result summaries) with hand-computed values. Slow tests fit real (tiny)
models: an end-to-end backtest on a synthetic world, and the load-bearing
consistency check that :class:`PosteriorForecaster`'s NumPy forward pass
reproduces the model's own in-graph posterior predictive on the training
window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.validation.backtest import (
    BacktestConfig,
    BacktestResult,
    _point_metrics,
    _seasonal_naive_pred,
    _seasonal_naive_scale,
    rolling_origins,
)


# ---------------------------------------------------------------------------
# rolling_origins
# ---------------------------------------------------------------------------


class TestRollingOrigins:
    def test_basic_non_overlapping(self):
        # 156 weeks, first cutoff at 104, 13-week windows
        assert rolling_origins(156, min_train_size=104, horizon=13, step=13) == [
            104,
            117,
            130,
            143,
        ]

    def test_only_full_windows_emitted(self):
        # Origin 144 would leave only 12 test periods -> excluded
        origins = rolling_origins(156, min_train_size=144, horizon=13)
        assert origins == []

    def test_step_defaults_to_horizon(self):
        assert rolling_origins(130, min_train_size=104, horizon=13) == rolling_origins(
            130, min_train_size=104, horizon=13, step=13
        )

    def test_max_origins_caps(self):
        origins = rolling_origins(
            260, min_train_size=104, horizon=13, step=13, max_origins=3
        )
        assert len(origins) == 3
        assert origins[0] == 104

    def test_overlapping_windows(self):
        origins = rolling_origins(120, min_train_size=104, horizon=8, step=4)
        assert origins == [104, 108, 112]

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            rolling_origins(100, min_train_size=1, horizon=13)
        with pytest.raises(ValueError):
            rolling_origins(100, min_train_size=52, horizon=0)
        with pytest.raises(ValueError):
            rolling_origins(100, min_train_size=52, horizon=13, step=0)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestPointMetrics:
    def test_hand_computed(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        m = _point_metrics(y_true, y_pred)
        # APE: 10/100 = 0.10, 20/200 = 0.10
        assert m["mape"] == pytest.approx(0.10)
        # sMAPE: 10/105, 20/190
        assert m["smape"] == pytest.approx((10 / 105 + 20 / 190) / 2)
        assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
        assert m["mae"] == pytest.approx(15.0)
        assert m["bias"] == pytest.approx((10 - 20) / 2)

    def test_perfect_forecast(self):
        y = np.array([5.0, 7.0, 9.0])
        m = _point_metrics(y, y.copy())
        assert m["mape"] == 0.0
        assert m["rmse"] == 0.0
        assert m["bias"] == 0.0

    def test_zero_actuals_excluded_from_mape(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        m = _point_metrics(y_true, y_pred)
        assert m["mape"] == pytest.approx(0.10)  # only the finite APE survives


class TestSeasonalNaive:
    def test_scale_hand_computed(self):
        # season=2: diffs |y[2]-y[0]|, |y[3]-y[1]| = 2, 2 -> 2.0
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _seasonal_naive_scale(y, season=2) == pytest.approx(2.0)

    def test_scale_falls_back_to_naive_when_short(self):
        y = np.array([1.0, 3.0, 6.0])
        # season longer than the series -> one-step diffs 2, 3 -> 2.5
        assert _seasonal_naive_scale(y, season=10) == pytest.approx(2.5)

    def test_pred_uses_same_phase_last_season(self):
        y = np.arange(20.0)
        # origin=12, season=4: position 12 -> y[8], 13 -> y[9], ...
        pred = _seasonal_naive_pred(y, origin=12, positions=np.arange(12, 16), season=4)
        np.testing.assert_allclose(pred, y[8:12])

    def test_pred_never_peeks_past_origin(self):
        y = np.arange(20.0)
        # origin=6, season=4: position 12 - 4 = 8 >= origin -> step back to 4
        pred = _seasonal_naive_pred(y, origin=6, positions=np.array([12]), season=4)
        assert pred[0] == y[4]


# ---------------------------------------------------------------------------
# BacktestResult summaries (hand-built records)
# ---------------------------------------------------------------------------


def _toy_result() -> BacktestResult:
    config = BacktestConfig(coverage_levels=(0.8,), min_train_size=10, horizon=2)
    records = pd.DataFrame(
        [
            # origin 10: perfect h=1, 10% high at h=2; truth inside both PIs
            dict(
                origin=10,
                position=10,
                date="d0",
                horizon=1,
                y_true=100.0,
                y_pred=100.0,
                pred_naive=90.0,
                pred_snaive=95.0,
                lo_80=90.0,
                hi_80=110.0,
                cov_80=True,
            ),
            dict(
                origin=10,
                position=11,
                date="d1",
                horizon=2,
                y_true=100.0,
                y_pred=110.0,
                pred_naive=90.0,
                pred_snaive=95.0,
                lo_80=95.0,
                hi_80=125.0,
                cov_80=True,
            ),
            # origin 12: truth outside the PI at h=2
            dict(
                origin=12,
                position=12,
                date="d2",
                horizon=1,
                y_true=200.0,
                y_pred=190.0,
                pred_naive=180.0,
                pred_snaive=210.0,
                lo_80=170.0,
                hi_80=210.0,
                cov_80=True,
            ),
            dict(
                origin=12,
                position=13,
                date="d3",
                horizon=2,
                y_true=200.0,
                y_pred=240.0,
                pred_naive=180.0,
                pred_snaive=210.0,
                lo_80=220.0,
                hi_80=260.0,
                cov_80=False,
            ),
        ]
    )
    fits = pd.DataFrame(
        [
            dict(
                origin=10, train_size=10, fit_seconds=1.0, rhat_max=1.0, divergences=0
            ),
            dict(
                origin=12, train_size=12, fit_seconds=1.0, rhat_max=1.0, divergences=0
            ),
        ]
    ).set_index("origin")
    return BacktestResult(
        config=config,
        records=records,
        fits=fits,
        season_period=4,
        mase_scales={10: 10.0, 12: 20.0},
    )


class TestBacktestResult:
    def test_summary_metrics_and_coverage(self):
        res = _toy_result()
        s = res.summary()
        assert set(s.index) == {"mmm", "seasonal_naive", "naive_last_value"}
        # MAPE: |0|/100, |10|/100, |10|/200, |40|/200 -> (0 + .1 + .05 + .2)/4
        assert s.loc["mmm", "mape"] == pytest.approx(0.0875)
        assert s.loc["mmm", "coverage_80"] == pytest.approx(3 / 4)
        # Baselines carry no coverage columns
        assert np.isnan(s.loc["naive_last_value", "coverage_80"])

    def test_mase_per_origin_scaling(self):
        res = _toy_result()
        s = res.summary()
        # origin 10: mean |err| = (0 + 10)/2 = 5, scale 10 -> 0.5
        # origin 12: mean |err| = (10 + 40)/2 = 25, scale 20 -> 1.25
        assert s.loc["mmm", "mase"] == pytest.approx((0.5 + 1.25) / 2)

    def test_by_horizon_shape_and_decay(self):
        res = _toy_result()
        bh = res.by_horizon()
        assert list(bh.index) == [1, 2]
        # h=1 errors (0%, 5%) < h=2 errors (10%, 20%)
        assert bh.loc[1, "mape"] < bh.loc[2, "mape"]
        assert bh.loc[2, "coverage_80"] == pytest.approx(0.5)

    def test_by_origin(self):
        res = _toy_result()
        bo = res.by_origin()
        assert list(bo.index) == [10, 12]
        assert bo.loc[10, "mape"] == pytest.approx(0.05)

    def test_coverage_table(self):
        res = _toy_result()
        cov = res.coverage()
        assert cov.loc[0.8, "empirical"] == pytest.approx(0.75)
        # widths: 20, 30, 40, 40 -> mean 32.5; mean |y| = 150
        assert cov.loc[0.8, "mean_width"] == pytest.approx(32.5)
        assert cov.loc[0.8, "mean_width_pct_of_kpi"] == pytest.approx(32.5 / 150)

    def test_n_origins(self):
        assert _toy_result().n_origins == 2


# ---------------------------------------------------------------------------
# slow: real fits
# ---------------------------------------------------------------------------


def _small_world(n_weeks: int = 120):
    from mmm_framework.synth import dgp

    return dgp.build("clean", seed=7, n_weeks=n_weeks)


def _small_model(panel, *, parametric: bool, seed: int = 7):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=200,
        n_tune=200,
        use_parametric_adstock=parametric,
        optim_seed=seed,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


@pytest.mark.slow
class TestForecasterConsistency:
    """The NumPy forward pass must match the in-graph posterior predictive."""

    @pytest.mark.parametrize("parametric", [True, False])
    def test_training_window_reproduction(self, parametric):
        from mmm_framework.validation.backtest import PosteriorForecaster

        sc = _small_world()
        mmm = _small_model(sc.panel(), parametric=parametric)
        mmm.fit(random_seed=7, progressbar=False)

        forecaster = PosteriorForecaster(mmm)
        positions = np.arange(mmm.n_periods)
        samples = forecaster.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            positions,
            include_noise=False,
            random_seed=7,
        )
        mu_numpy = samples.mean(axis=0)

        pred = mmm.predict(random_seed=7)  # in-graph, includes mean-zero noise
        mu_graph = pred.y_pred_mean

        # Same posterior, same structure: near-perfect agreement up to MC noise
        corr = np.corrcoef(mu_numpy, mu_graph)[0, 1]
        assert corr > 0.99, f"forward-pass mismatch: corr={corr:.4f}"
        scale = np.abs(mu_graph).mean()
        rel_err = np.abs(mu_numpy - mu_graph).mean() / scale
        assert rel_err < 0.05, f"forward-pass mismatch: rel_err={rel_err:.3%}"


@pytest.mark.slow
class TestEndToEndBacktest:
    def test_clean_world_backtest(self):
        from mmm_framework.validation.backtest import run_backtest

        sc = _small_world(n_weeks=120)
        mmm = _small_model(sc.panel(), parametric=True)
        config = BacktestConfig(
            min_train_size=96,
            horizon=8,
            step=8,
            draws=200,
            tune=200,
            chains=2,
            coverage_levels=(0.8, 0.95),
            random_seed=7,
        )
        result = run_backtest(mmm, config, progressbar=False)

        # 120 weeks, cutoffs at 96, 104, 112 -> 3 origins x 8 horizons
        assert result.n_origins == 3
        assert len(result.records) == 24
        assert set(result.fits.columns) >= {"fit_seconds", "rhat_max", "divergences"}

        s = result.summary()
        assert np.isfinite(s.loc["mmm", "mape"])
        assert 0.0 <= s.loc["mmm", "coverage_80"] <= 1.0
        # On the clean world the model should not be wildly off
        assert s.loc["mmm", "mape"] < 0.5

        bh = result.by_horizon()
        assert list(bh.index) == list(range(1, 9))
        cov = result.coverage()
        assert list(cov.index) == [0.8, 0.95]

    # NOTE: geo/product panels are no longer rejected — run_backtest supports them
    # (period-major obs, per-cell forward pass). See tests/test_backtest_geo.py.


@pytest.mark.slow
class TestValidatorCVUsesForecaster:
    """ModelValidator cross-validation must work for the (parametric) default.

    The previous hand-rolled CV forward pass silently mispredicted
    parametric-adstock models; it now delegates to PosteriorForecaster.
    """

    def test_cv_runs_on_default_parametric_model(self):
        from mmm_framework.validation import ModelValidator, ValidationConfigBuilder

        sc = _small_world(n_weeks=120)
        mmm = _small_model(sc.panel(), parametric=True)
        results = mmm.fit(random_seed=7, progressbar=False)

        config = (
            ValidationConfigBuilder()
            .silent()
            .without_ppc()
            .without_residuals()
            .without_channel_diagnostics()
            .with_cross_validation(
                n_folds=2, strategy="expanding", min_train_size=96
            )
            .build()
        )
        config.run_model_comparison = False
        summary = ModelValidator(mmm, results).validate(config)

        cv = summary.cross_validation
        assert cv is not None and cv.n_folds >= 1
        fold = cv.fold_results[0]
        # Predictions must be on the KPI scale and sane -- the old blend
        # fallback on a parametric model produced systematically wrong levels.
        y_scale = float(np.abs(mmm.y_raw).mean())
        assert np.isfinite(fold.rmse) and fold.rmse < y_scale
        assert fold.r2 > -1.0

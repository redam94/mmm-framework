"""The training-window offset is a PERIOD offset, not an observation index (#273).

``PosteriorForecaster.forecast`` takes ``positions`` on the *observation* axis
and ``train_offset`` on the *period* axis. On a national panel those coincide,
so the difference is invisible; on a geo/product panel they differ by
``n_cells``, and the validator's rolling-window cross-validation passed
``min(train_indices)`` — an observation index — straight into the period slot.
The trend was then evaluated ``(n_cells - 1) * offset`` periods early and the
Fourier basis was that far out of phase, which moves the forecast *level* while
leaving the interval width untouched: wrong and confident.

Nothing in the suite had ever exercised a nonzero offset (``grep -rn
train_offset tests/`` returned zero hits before this file), which is why the
defect survived the #219 fix that introduced half of it.

These tests are fast by construction: the forecaster reads only the posterior
and the prepared data, so a hand-built posterior with zero media coefficients
isolates the trend and seasonality arithmetic exactly, with no MCMC.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.config import InferenceMethod, ModelConfig
from mmm_framework.frequentist._transforms import SATURATION_PARAMS
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.utils.arviz_compat import posterior_from_dict
from mmm_framework.validation.backtest import (
    ForecastUnsupportedError,
    PosteriorForecaster,
)
from mmm_framework.validation.config import CrossValidationConfig
from mmm_framework.validation.validator import ModelValidator, _periods_to_obs

N_DRAWS = 4
TREND_SLOPE = 2.0


def _model(panel):
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=N_DRAWS,
        use_parametric_adstock=True,
    )
    return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))


def _geo_model(n_weeks: int = 60):
    from mmm_framework.synth import dgp_geo

    return _model(dgp_geo.make_geo_clean(n_weeks=n_weeks).panel())


def _national_model(n_weeks: int = 60):
    from mmm_framework.synth import dgp

    return _model(dgp.build("clean", seed=3, n_weeks=n_weeks).panel())


def _attach_posterior(mmm):
    """A hand-built posterior isolating intercept + trend + seasonality.

    Media coefficients are zero, so the media forward pass still runs (its
    parameters must be present or ``_saturate`` raises) but contributes nothing
    — leaving a mean whose value at every position is known in closed form.
    """
    post = {
        "intercept": np.zeros((1, N_DRAWS)),
        "sigma": np.full((1, N_DRAWS), 0.1),
        "trend_slope": np.full((1, N_DRAWS), TREND_SLOPE),
    }
    for ch in mmm.channel_names:
        post[f"beta_{ch}"] = np.zeros((1, N_DRAWS))
        post[f"adstock_alpha_{ch}"] = np.full((1, N_DRAWS), 0.5)
        for name in SATURATION_PARAMS[mmm._get_saturation_config(ch).type]:
            post[f"{name}_{ch}"] = np.full((1, N_DRAWS), 1.0)
    for name, features in mmm.seasonality_features.items():
        coefs = np.linspace(0.5, 1.0, features.shape[1])
        post[f"season_{name}"] = np.tile(coefs, (1, N_DRAWS, 1))
    mmm._trace = posterior_from_dict(post)
    return mmm


def _expected(mmm, period_positions, period_offset):
    """Closed-form KPI-scale mean at ``period_positions`` for one draw."""
    from mmm_framework.transforms.seasonality import create_fourier_features

    shifted = np.asarray(period_positions, dtype=float) - period_offset
    mu = shifted / max(mmm.n_periods - 1, 1) * TREND_SLOPE
    periods_by_freq = {"W": {"yearly": 52.0, "monthly": 52.0 / 12.0}}
    freq = getattr(mmm.mff_config, "frequency", "W") or "W"
    for name, features in mmm.seasonality_features.items():
        order = features.shape[1] // 2
        period = periods_by_freq.get(freq, periods_by_freq["W"])[name]
        coefs = np.linspace(0.5, 1.0, features.shape[1])
        mu = mu + create_fourier_features(shifted, period, order) @ coefs
    return mu * mmm.y_std + mmm.y_mean


class TestForecastOffsetUnits:
    def test_geo_train_positions_are_in_phase_with_the_training_window(self):
        """The regression: fails on the pre-fix code, which shifted by n_cells x.

        Asserted against the closed form rather than against another forecast,
        so it pins the arithmetic and not merely self-consistency.
        """
        mmm = _attach_posterior(_geo_model())
        assert mmm.n_cells > 1
        f = PosteriorForecaster(mmm)

        period_start = 20
        obs_start = period_start * mmm.n_cells
        train_obs = np.arange(obs_start, obs_start + 30 * mmm.n_cells)
        test_periods = np.arange(50, 56)
        test_obs = _periods_to_obs(test_periods, mmm.n_cells)

        got = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            test_obs,
            include_noise=False,
            train_positions=train_obs,
        )

        want = _expected(mmm, np.repeat(test_periods, mmm.n_cells), period_start)
        np.testing.assert_allclose(got[0], want, rtol=0, atol=1e-9)

        # Non-vacuity: the buggy offset produces a materially different answer,
        # so this test cannot pass by the two branches coinciding.
        wrong = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            test_obs,
            include_noise=False,
            train_offset=obs_start,
        )
        assert np.abs(wrong - got).max() > 1.0

    def test_train_positions_and_period_offset_agree(self):
        mmm = _attach_posterior(_geo_model())
        f = PosteriorForecaster(mmm)
        period_start = 12
        train_obs = np.arange(
            period_start * mmm.n_cells, (period_start + 20) * mmm.n_cells
        )
        test_obs = _periods_to_obs(np.arange(40, 45), mmm.n_cells)

        by_positions = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            test_obs,
            include_noise=False,
            train_positions=train_obs,
        )
        by_offset = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            test_obs,
            include_noise=False,
            train_offset=period_start,
        )
        np.testing.assert_array_equal(by_positions, by_offset)

    def test_national_offset_is_unchanged(self):
        """On a national panel the two axes coincide; nothing may move."""
        mmm = _attach_posterior(_national_model())
        assert mmm.n_cells == 1
        f = PosteriorForecaster(mmm)
        train = np.arange(10, 40)
        positions = np.arange(45, 55)

        by_positions = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            positions,
            include_noise=True,
            random_seed=11,
            train_positions=train,
        )
        by_offset = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            positions,
            include_noise=True,
            random_seed=11,
            train_offset=10,
        )
        np.testing.assert_array_equal(by_positions, by_offset)

        noiseless = f.forecast(
            mmm.X_media_raw,
            mmm.X_controls_raw,
            positions,
            include_noise=False,
            train_positions=train,
        )
        np.testing.assert_allclose(
            noiseless[0], _expected(mmm, positions, 10), rtol=0, atol=1e-9
        )

    def test_window_off_a_period_boundary_is_refused(self):
        mmm = _attach_posterior(_geo_model())
        f = PosteriorForecaster(mmm)
        ragged = np.arange(10, 10 + 20 * mmm.n_cells)  # 10 % 4 != 0
        with pytest.raises(ForecastUnsupportedError, match="period boundary"):
            f.forecast(
                mmm.X_media_raw,
                mmm.X_controls_raw,
                np.arange(mmm.n_cells),
                train_positions=ragged,
            )

    def test_both_offset_forms_together_is_refused(self):
        mmm = _attach_posterior(_national_model())
        f = PosteriorForecaster(mmm)
        with pytest.raises(ValueError, match="not both"):
            f.forecast(
                mmm.X_media_raw,
                mmm.X_controls_raw,
                np.arange(5),
                train_offset=3,
                train_positions=np.arange(3, 10),
            )


class _StubValidator:
    """Enough of ``ModelValidator`` to exercise the split/slice arithmetic."""

    def __init__(self, model):
        self.model = model

    _create_cv_splits = ModelValidator._create_cv_splits
    _slice_panel_data = ModelValidator._slice_panel_data
    _predict_at_indices = ModelValidator._predict_at_indices


class TestCVSplitsCoverWholePeriods:
    @pytest.mark.parametrize("strategy", ["expanding", "rolling", "blocked"])
    def test_geo_splits_are_cell_aligned(self, strategy):
        """Every window must start on, and span, whole periods.

        Before the fix a rolling window started at observation 10 of a 4-cell
        panel (mid-period), and expanding/blocked windows had lengths that were
        not multiples of ``n_cells``.
        """
        mmm = _geo_model()
        v = _StubValidator(mmm)
        cfg = CrossValidationConfig(
            strategy=strategy,
            n_folds=3,
            min_train_size=25,
            test_size=5 if strategy == "rolling" else None,
        )
        splits = v._create_cv_splits(mmm.n_obs, cfg)

        assert splits
        for train_idx, test_idx in splits:
            for idx in (train_idx, test_idx):
                assert int(idx.min()) % mmm.n_cells == 0
                assert len(idx) % mmm.n_cells == 0
                # Contiguous whole periods, cell-minor within each.
                assert np.array_equal(idx, np.arange(idx.min(), idx.max() + 1))
            assert train_idx.max() < test_idx.min()
            assert not (set(train_idx.tolist()) & set(test_idx.tolist()))

    def test_national_splits_are_unchanged(self):
        """n_cells == 1 makes the period axis the observation axis."""
        mmm = _national_model()
        v = _StubValidator(mmm)
        cfg = CrossValidationConfig(strategy="expanding", n_folds=5, min_train_size=30)
        splits = v._create_cv_splits(mmm.n_obs, cfg)
        assert splits
        for train_idx, test_idx in splits:
            assert train_idx.min() == 0
            assert train_idx.max() < test_idx.min()

    def test_ragged_slice_is_refused_not_reshaped(self):
        """101 observations of a 4-cell panel used to rebuild as 26 periods."""
        mmm = _geo_model()
        v = _StubValidator(mmm)
        with pytest.raises(ValueError, match="whole periods"):
            v._slice_panel_data(mmm.panel, np.arange(101))
        with pytest.raises(ValueError, match="whole periods"):
            v._slice_panel_data(mmm.panel, np.arange(2, 2 + 40))

    def test_cv_prediction_path_is_in_phase(self):
        """The call site itself, not just the forecaster it calls.

        ``_predict_at_indices`` is where the observation index was handed to a
        period slot, so it is graded end to end against the closed form.
        """
        mmm = _attach_posterior(_geo_model())
        v = _StubValidator(mmm)

        period_start = 20
        train_obs = np.arange(
            period_start * mmm.n_cells, (period_start + 30) * mmm.n_cells
        )
        test_periods = np.arange(52, 58)
        test_obs = _periods_to_obs(test_periods, mmm.n_cells)

        _, samples = v._predict_at_indices(mmm, train_obs, test_obs)

        want = _expected(mmm, np.repeat(test_periods, mmm.n_cells), period_start)
        # include_noise=True is forced on this path; sigma is 0.1 standardized.
        tol = 6.0 * 0.1 * mmm.y_std
        assert np.abs(samples[0] - want).max() < tol

        wrong = _expected(mmm, np.repeat(test_periods, mmm.n_cells), 0)
        assert np.abs(wrong - want).max() > tol

    def test_aligned_slice_keeps_the_panel_rectangular(self):
        mmm = _geo_model()
        v = _StubValidator(mmm)
        n_train = 30
        sliced = v._slice_panel_data(mmm.panel, np.arange(n_train * mmm.n_cells))
        assert len(sliced.y) == n_train * mmm.n_cells
        assert sliced.coords.n_periods == n_train

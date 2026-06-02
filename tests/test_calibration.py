"""Tests for experiment-calibrated priors (mmm_framework.calibration).

Fast tests cover the pure derivation math and the config/model wiring (the
latter via ``pm.draw`` on the prior, with no MCMC). The end-to-end two-stage
flow (fit -> derive -> refit) is marked ``slow``.
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mmm_framework.calibration import (
    CalibrationOutcome,
    ExperimentCalibrator,
    LiftObservation,
    LiftTestResult,
    calibrate_with_experiments,
    combine_inverse_variance,
    derive_channel_prior,
    design_factor,
    mean_sd_to_gamma,
)
from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
    PriorConfig,
    PriorType,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType


# =============================================================================
# Pure math
# =============================================================================


class TestPureMath:
    def test_mean_sd_to_gamma_moment_match(self):
        for mean, sd in [(1.5, 1.0), (2.0, 0.5), (0.3, 0.2), (10.0, 3.0)]:
            alpha, rate = mean_sd_to_gamma(mean, sd)
            assert math.isclose(alpha / rate, mean, rel_tol=1e-9)
            assert math.isclose(math.sqrt(alpha) / rate, sd, rel_tol=1e-9)

    def test_mean_sd_to_gamma_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            mean_sd_to_gamma(0.0, 1.0)
        with pytest.raises(ValueError):
            mean_sd_to_gamma(-1.0, 1.0)

    def test_design_factor_recovers_K_when_contrib_is_beta_times_K(self):
        rng = np.random.default_rng(0)
        beta = rng.gamma(3.0, 1.0, size=8000)
        K = 4.2
        contrib = beta * K
        assert math.isclose(design_factor(contrib, beta), K, rel_tol=1e-9)

    def test_design_factor_is_per_draw_not_ratio_of_means(self):
        # Construct anti-correlated beta and K so mean(contrib)/mean(beta) differs
        # materially from mean(contrib/beta). design_factor must give the latter.
        beta = np.array([1.0, 2.0, 3.0, 4.0])
        K = np.array([8.0, 4.0, 2.0, 1.0])  # K falls as beta rises
        contrib = beta * K
        per_draw = float(np.mean(contrib / beta))  # = mean(K) = 3.75
        ratio_of_means = float(np.mean(contrib) / np.mean(beta))
        assert not math.isclose(per_draw, ratio_of_means, rel_tol=1e-3)
        assert math.isclose(design_factor(contrib, beta), per_draw, rel_tol=1e-12)

    def test_design_factor_filters_zero_beta(self):
        beta = np.array([0.0, 0.0, 2.0])
        contrib = np.array([0.0, 0.0, 6.0])  # K = 3 on the valid draw
        assert math.isclose(
            design_factor(beta_samples=beta, contribution_samples=contrib), 3.0
        )

    def test_design_factor_all_zero_beta_raises(self):
        with pytest.raises(ValueError):
            design_factor(np.zeros(5), np.zeros(5))

    def test_combine_inverse_variance(self):
        # Two identical estimates -> same mean, sd reduced by sqrt(2)
        mean, sd = combine_inverse_variance([2.0, 2.0], [0.5, 0.5])
        assert math.isclose(mean, 2.0, rel_tol=1e-9)
        assert math.isclose(sd, 0.5 / math.sqrt(2), rel_tol=1e-9)

    def test_combine_inverse_variance_precision_weighted(self):
        # The tighter estimate dominates the combined mean.
        mean, _ = combine_inverse_variance([1.0, 3.0], [0.1, 1.0])
        assert mean < 1.5  # pulled toward the precise estimate (1.0)


# =============================================================================
# Channel-level derivation
# =============================================================================


class TestDeriveChannelPrior:
    def test_single_test_inverts_design_factor(self):
        K = 4.2
        obs = LiftObservation(
            ("a", "b"), measured_lift=8.4, lift_se=0.84, design_factor=K, usable=True
        )
        cal = derive_channel_prior("TV", [obs])
        assert cal.calibrated
        assert math.isclose(cal.beta_target * K, 8.4, rel_tol=1e-9)
        # relative uncertainty preserved
        assert math.isclose(cal.beta_sigma / cal.beta_target, 0.84 / 8.4, rel_tol=1e-9)
        # roi_prior is a moment-matched Gamma
        assert cal.roi_prior.distribution == PriorType.GAMMA
        a, r = cal.roi_prior.params["alpha"], cal.roi_prior.params["beta"]
        assert math.isclose(a / r, cal.beta_target, rel_tol=1e-9)

    def test_multiple_tests_combined(self):
        # Two tests on the same channel, same period/design -> tighter combined sd.
        K = 2.0
        obs = [
            LiftObservation(("a", "b"), 4.0, 0.4, K, True),
            LiftObservation(("a", "b"), 4.0, 0.4, K, True),
        ]
        cal = derive_channel_prior("TV", obs)
        single = derive_channel_prior("TV", [obs[0]])
        assert math.isclose(cal.beta_target, single.beta_target, rel_tol=1e-9)
        assert cal.beta_sigma < single.beta_sigma  # combining shrinks uncertainty

    def test_nonpositive_lift_excluded(self):
        obs = LiftObservation(("a", "b"), -1.0, 0.5, 2.0, usable=False, note="neg")
        cal = derive_channel_prior("TV", [obs])
        assert not cal.calibrated
        assert cal.roi_prior is None
        assert cal.skipped_reason

    def test_mixed_usable_and_unusable(self):
        obs = [
            LiftObservation(("a", "b"), -1.0, 0.5, 2.0, usable=False, note="neg"),
            LiftObservation(("c", "d"), 6.0, 0.6, 2.0, usable=True),
        ]
        cal = derive_channel_prior("TV", obs)
        assert cal.calibrated
        assert math.isclose(cal.beta_target, 3.0, rel_tol=1e-9)  # 6.0 / 2.0


# =============================================================================
# Fixtures for model wiring
# =============================================================================


@pytest.fixture
def periods():
    return pd.date_range("2021-01-04", periods=40, freq="W-MON")


def _make_panel(periods, tv_roi_prior=None):
    n = len(periods)
    rng = np.random.default_rng(7)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    tv_spend = np.abs(rng.normal(100, 30, n))
    digital_spend = np.abs(rng.normal(80, 20, n))
    # Build a y with a genuine TV effect so the model can identify it.
    y = pd.Series(
        1000 + 2.5 * tv_spend + 1.0 * digital_spend + rng.normal(0, 40, n), name="Sales"
    )
    X_media = pd.DataFrame({"TV": tv_spend, "Digital": digital_spend})
    X_controls = pd.DataFrame({"Price": 10 + rng.normal(0, 0.5, n)})
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(
                name="TV", dimensions=[DimensionType.PERIOD], roi_prior=tv_roi_prior
            ),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=periods,
        config=config,
    )


@pytest.fixture
def model_config():
    return ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=120,
        n_tune=120,
        target_accept=0.85,
    )


# =============================================================================
# Wiring (no MCMC): roi_prior must change the beta prior actually used
# =============================================================================


class TestRoiPriorWiring:
    def test_roi_prior_overrides_beta_prior(self, periods, model_config):
        import pymc as pm

        # Target a clearly non-default coefficient mean (default is Gamma mu=1.5).
        alpha, rate = mean_sd_to_gamma(3.0, 0.3)
        tv_prior = PriorConfig.gamma(alpha=alpha, beta=rate)
        panel = _make_panel(periods, tv_roi_prior=tv_prior)
        mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        built = mmm._build_model()
        with built:
            tv = pm.draw(built["beta_TV"], draws=6000, random_seed=0)
            dig = pm.draw(built["beta_Digital"], draws=6000, random_seed=0)
        # TV honors the experiment-calibrated prior (mean ~3.0)...
        assert abs(float(tv.mean()) - 3.0) < 0.2
        # ...Digital keeps the model default (Gamma mu=1.5).
        assert abs(float(dig.mean()) - 1.5) < 0.25

    def test_no_roi_prior_keeps_default(self, periods, model_config):
        import pymc as pm

        panel = _make_panel(periods, tv_roi_prior=None)
        mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        built = mmm._build_model()
        with built:
            tv = pm.draw(built["beta_TV"], draws=6000, random_seed=0)
        assert abs(float(tv.mean()) - 1.5) < 0.25


# =============================================================================
# Calibrator guards (no MCMC)
# =============================================================================


class TestCalibratorGuards:
    def test_requires_fitted_model(self, periods, model_config):
        panel = _make_panel(periods)
        mmm = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        with pytest.raises(ValueError, match="fitted"):
            ExperimentCalibrator(mmm)

    def test_geo_holdout_warns_when_pooled(self):
        # Lightweight stand-in: only the attributes _geo_warning reads, plus a
        # non-None _trace so __init__ accepts it.
        fake = SimpleNamespace(
            _trace=object(),
            has_geo=True,
            hierarchical_config=SimpleNamespace(pool_across_geo=True),
        )
        cal = ExperimentCalibrator.__new__(ExperimentCalibrator)
        cal.model = fake
        cal.results = None
        cal._validator = None
        tests = [
            LiftTestResult(
                "TV",
                ("2021-01-01", "2021-03-01"),
                100.0,
                10.0,
                holdout_regions=["West"],
            )
        ]
        with pytest.warns(UserWarning, match="holdout_regions"):
            msg = cal._geo_warning(tests, strict=False)
        assert msg

    def test_geo_holdout_strict_raises(self):
        fake = SimpleNamespace(
            _trace=object(),
            has_geo=True,
            hierarchical_config=SimpleNamespace(pool_across_geo=True),
        )
        cal = ExperimentCalibrator.__new__(ExperimentCalibrator)
        cal.model = fake
        cal.results = None
        cal._validator = None
        tests = [
            LiftTestResult(
                "TV",
                ("2021-01-01", "2021-03-01"),
                100.0,
                10.0,
                holdout_regions=["West"],
            )
        ]
        with pytest.raises(ValueError, match="holdout_regions"):
            cal._geo_warning(tests, strict=True)


# =============================================================================
# End-to-end two-stage flow (slow)
# =============================================================================


@pytest.mark.slow
class TestTwoStageCalibration:
    def test_calibration_moves_estimate_toward_experiment(self, periods, model_config):
        panel = _make_panel(periods)
        base = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        base.fit(random_seed=0)

        period = (str(periods[0].date()), str(periods[-1].date()))
        baseline_contrib = float(
            base.compute_counterfactual_contributions(
                channels=["TV"]
            ).total_contributions["TV"]
        )
        assert baseline_contrib > 0

        # Experiment says TV is ~2x as effective as the baseline model thinks,
        # measured very precisely (tight prior should dominate the refit).
        measured = 2.0 * baseline_contrib
        lift_se = 0.01 * measured
        lt = LiftTestResult("TV", period, measured, lift_se)

        calibrator = ExperimentCalibrator(base)
        report = calibrator.derive_priors([lt])
        assert report.calibrated_channels == ["TV"]
        tv_cal = report.channel_calibrations[0]
        # Inversion is exact: beta_target * K_c reproduces the measured lift.
        K_c = tv_cal.observations[0].design_factor
        assert math.isclose(tv_cal.beta_target * K_c, measured, rel_tol=1e-6)
        # The experiment (2x the baseline contribution) implies a larger coefficient
        # than the baseline fit -- this is the signal the refit must absorb.
        assert tv_cal.beta_target > tv_cal.beta_fit_mean

        outcome = calibrator.calibrate(
            [lt], refit=True, draws=120, tune=120, chains=2, random_seed=0
        )
        assert isinstance(outcome, CalibrationOutcome)
        assert outcome.model is not None and outcome.results is not None
        # The refit channel's roi_prior is set on the cloned config.
        tv_cfg = next(m for m in outcome.config.media_channels if m.name == "TV")
        assert tv_cfg.roi_prior is not None

        new_contrib = float(
            outcome.model.compute_counterfactual_contributions(
                channels=["TV"]
            ).total_contributions["TV"]
        )
        # Moved toward the experimental value and got closer to it than baseline.
        assert new_contrib > baseline_contrib
        assert abs(new_contrib - measured) < abs(baseline_contrib - measured)

    def test_calibrate_no_usable_tests_skips_refit(self, periods, model_config):
        panel = _make_panel(periods)
        base = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        base.fit(random_seed=0)
        # Negative measured lift -> positive-coefficient model can't be anchored.
        lt = LiftTestResult(
            "TV", (str(periods[0].date()), str(periods[-1].date())), -50.0, 5.0
        )
        outcome = calibrate_with_experiments(base, [lt])
        assert outcome.model is None  # no refit when nothing is calibrated
        assert not outcome.report.calibrated_channels


# =============================================================================
# Validator period parser (delegates to the corrected model parser)
# =============================================================================


class TestValidatorPeriodParser:
    def _validator(self, periods, model_config):
        from mmm_framework.validation.validator import ModelValidator

        panel = _make_panel(periods)
        model = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        return ModelValidator(model, None)

    def test_full_range_starts_at_zero(self, periods, model_config):
        # Off-by-one fix: a window starting at period 0 must include period 0.
        val = self._validator(periods, model_config)
        start, end = val._parse_period_to_indices(
            (str(periods[0].date()), str(periods[-1].date()))
        )
        assert (start, end) == (0, len(periods) - 1)

    def test_out_of_range_raises(self, periods, model_config):
        val = self._validator(periods, model_config)
        with pytest.raises(ValueError, match="outside"):
            val._parse_period_to_indices(("2018-01-01", "2018-12-31"))

    def test_out_of_range_lift_test_skipped_not_scored(self, periods, model_config):
        # A lift test outside the panel must be skipped by the calibration loop,
        # not silently scored against the whole panel.
        from mmm_framework.validation.validator import ModelValidator

        panel = _make_panel(periods)
        model = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
        model.fit(random_seed=0)
        val = ModelValidator(model, None)
        lt = LiftTestResult("TV", ("2018-01-01", "2018-12-31"), 1e4, 1e3)
        with pytest.raises(ValueError):
            val._get_model_estimate_for_lift_test(lt, 0.95)

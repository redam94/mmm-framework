"""Tests for the causal checks: unobserved-confounding sensitivity (P0-2) and
the causal refutation suite (P0-3)."""

import math
from types import SimpleNamespace

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
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.validation import (
    CausalRefutationResults,
    ModelValidator,
    UnobservedConfoundingAnalysis,
    UnobservedConfoundingSensitivity,
    ValidationConfigBuilder,
)
from mmm_framework.validation.sensitivity_unobserved import (
    FRAGILE_RV_THRESHOLD,
    partial_r2_from_t,
    robustness_value,
)


# =============================================================================
# Robustness-value math (fast, pure)
# =============================================================================


class TestRobustnessValueMath:
    def test_monotone_and_bounded(self):
        rvs = [robustness_value(t, 100) for t in (0.1, 1.0, 3.0, 10.0, 50.0)]
        assert all(0.0 <= rv <= 1.0 for rv in rvs)
        assert rvs == sorted(rvs)  # increasing in |t|
        assert rvs[-1] > 0.9  # very strong effect is robust

    def test_symmetric_in_sign(self):
        assert math.isclose(robustness_value(3.0, 80), robustness_value(-3.0, 80))

    def test_partial_r2(self):
        assert partial_r2_from_t(0.0, 100) == 0.0
        assert 0.0 < partial_r2_from_t(2.0, 100) < 1.0
        # matches t^2/(t^2+dof)
        assert math.isclose(partial_r2_from_t(2.0, 100), 4.0 / 104.0)

    def test_invalid_dof_is_nan(self):
        assert math.isnan(robustness_value(3.0, 0))
        assert math.isnan(partial_r2_from_t(3.0, 0))


# =============================================================================
# UnobservedConfoundingAnalysis (fast, fake posterior)
# =============================================================================


def _fake_model_with_betas(beta_specs, n_obs=120, n_controls=1):
    """Build a stand-in model exposing a posterior with the given beta draws.

    ``beta_specs`` maps channel -> (mean, sd) for a normal draw cloud.
    """
    import xarray as xr

    rng = np.random.default_rng(0)
    data_vars = {}
    for ch, (mean, sd) in beta_specs.items():
        # RV depends only on |t| = |mean/sd|, so the draw cloud's sign is
        # irrelevant; use a plain normal cloud with the requested moments.
        draws = rng.normal(mean, sd, size=(2, 2000))
        data_vars[f"beta_{ch}"] = (("chain", "draw"), draws)
    posterior = xr.Dataset(data_vars)
    return SimpleNamespace(
        _trace=SimpleNamespace(posterior=posterior),
        channel_names=list(beta_specs.keys()),
        n_obs=n_obs,
        n_controls=n_controls,
    )


class TestUnobservedConfoundingAnalysis:
    def test_strong_channel_robust_weak_channel_fragile(self):
        # TV: tight, far from 0 (robust). Weak: wide, near 0 (fragile).
        model = _fake_model_with_betas({"TV": (2.0, 0.1), "Weak": (0.05, 0.5)})
        res = UnobservedConfoundingAnalysis(model).run()
        assert isinstance(res, UnobservedConfoundingSensitivity)
        by_ch = {c.channel: c for c in res.channels}
        assert by_ch["TV"].robustness_value > by_ch["Weak"].robustness_value
        assert by_ch["TV"].robustness_value > FRAGILE_RV_THRESHOLD
        assert "Weak" in res.fragile_channels
        # caveat names the no-unobserved-confounding assumption
        assert "UNOBSERVED" in res.caveat.upper()

    def test_dof_is_generous(self):
        model = _fake_model_with_betas({"TV": (2.0, 0.1)}, n_obs=100, n_controls=2)
        res = UnobservedConfoundingAnalysis(model).run()
        # nominal dof = n_obs - (n_media + n_controls + 1) = 100 - (1+2+1) = 96
        assert res.dof == 96

    def test_requires_fitted_model(self):
        with pytest.raises(ValueError, match="fitted"):
            UnobservedConfoundingAnalysis(SimpleNamespace(_trace=None))


# =============================================================================
# Fixtures for slow integration
# =============================================================================


@pytest.fixture
def fitted_model():
    periods = pd.date_range("2021-01-04", periods=44, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    tv = np.abs(rng.normal(100, 30, n))
    digital = np.abs(rng.normal(80, 20, n))
    # A strong, *fittable* signal: a linear trend + yearly seasonality the model
    # captures well (high R^2 on real y), plus modest media. (A purely linear
    # media effect would exceed the model's saturating capacity and fit poorly,
    # which would make the negative-control comparison meaningless.) With a high
    # real-y R^2, scrambling the KPI must drop the fit sharply.
    t = np.arange(n)
    y = pd.Series(
        1000
        + 12.0 * t
        + 60.0 * np.sin(2 * np.pi * t / 52)
        + 1.2 * tv
        + 0.6 * digital
        + rng.normal(0, 20, n),
        name="Sales",
    )
    config = MFFConfig(
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
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
        X_controls=pd.DataFrame({"Price": 10 + rng.normal(0, 0.5, n)}),
        coords=coords,
        index=periods,
        config=config,
    )
    model_config = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=150,
        n_tune=150,
        target_accept=0.85,
    )
    model = BayesianMMM(panel, model_config, TrendConfig(type=TrendType.LINEAR))
    model.fit(random_seed=0)
    return model


# =============================================================================
# Validator integration (slow)
# =============================================================================


@pytest.mark.slow
class TestCausalChecksIntegration:
    def test_unobserved_confounding_runs_in_validator(self, fitted_model):
        config = (
            ValidationConfigBuilder().silent().with_unobserved_confounding().build()
        )
        # isolate the cheap check
        config.run_ppc = config.run_residuals = config.run_channel_diagnostics = False
        config.run_model_comparison = False
        summary = ModelValidator(fitted_model).validate(config)
        uc = summary.unobserved_confounding
        assert uc is not None
        assert {c.channel for c in uc.channels} == {"TV", "Digital"}
        for c in uc.channels:
            assert 0.0 <= c.robustness_value <= 1.0
            assert np.isfinite(c.partial_r2)

    def test_refutation_suite_runs_and_is_directional(self, fitted_model):
        config = (
            ValidationConfigBuilder()
            .silent()
            .with_causal_refutation(draws=120, tune=120, chains=2)
            .build()
        )
        config.run_ppc = config.run_residuals = config.run_channel_diagnostics = False
        config.run_model_comparison = False
        summary = ModelValidator(fitted_model).validate(config)

        cr = summary.causal_refutation
        assert isinstance(cr, CausalRefutationResults)
        names = {t.name for t in cr.tests}
        assert names == {
            "placebo_treatment",
            "negative_control_outcome",
            "random_common_cause",
            "data_subset",
        }
        assert cr.n_passed + cr.n_failed == len(cr.tests)

        tests = {t.name: t for t in cr.tests}
        # Negative control: a model that fits the real KPI well (high R^2) cannot
        # fit a scrambled KPI -- the refit R^2 must collapse below threshold AND
        # be far below the original fit, so the test passes.
        nc = tests["negative_control_outcome"]
        assert nc.original_effect is not None and nc.refuted_effect is not None
        assert nc.original_effect > 0.5  # fixture is genuinely fittable
        assert nc.refuted_effect < nc.original_effect
        assert nc.passed is True
        # Placebo: scrambled media adds no more incremental fit than real media.
        pl = tests["placebo_treatment"]
        assert pl.refuted_effect is not None
        assert pl.refuted_effect <= pl.original_effect + 0.1
        # Stability tests report the worst-moving channel and a refit precision.
        for nm in ("random_common_cause", "data_subset"):
            assert tests[nm].channel in {"TV", "Digital"}
            assert tests[nm].precision is not None

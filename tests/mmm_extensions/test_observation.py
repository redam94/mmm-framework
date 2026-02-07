"""
Test suite for observation module in mmm_framework.mmm_extensions.components.

Tests cover:
- build_gaussian_likelihood
- build_partial_observation_model
- build_multivariate_likelihood
- build_aggregated_survey_observation (binomial, normal, beta-binomial)
- compute_survey_observation_indices
- build_mediator_observation_dispatch

Note: Bypasses PyTensor compilation issues with special config.
"""

import pytensor

pytensor.config.exception_verbosity = "high"
pytensor.config.cxx = ""

import numpy as np
import pytest

import pymc as pm
import pytensor.tensor as pt

from mmm_framework.mmm_extensions.components.observation import (
    build_gaussian_likelihood,
    build_partial_observation_model,
    build_multivariate_likelihood,
    build_aggregated_survey_observation,
    compute_survey_observation_indices,
    build_mediator_observation_dispatch,
)
from mmm_framework.mmm_extensions.config import (
    AggregatedSurveyConfig,
    AggregatedSurveyLikelihood,
    MediatorObservationType,
    MediatorConfigExtended,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_observed_data():
    """Sample observed outcome data."""
    np.random.seed(42)
    return 1000 + np.random.randn(52) * 100


@pytest.fixture
def sample_multivariate_data():
    """Sample multivariate outcome data."""
    np.random.seed(42)
    return np.random.randn(52, 2) * 100 + 1000


@pytest.fixture
def sample_partial_data():
    """Sample partially observed data with missing values."""
    np.random.seed(42)
    data = np.random.randn(52) * 10 + 50
    # Set some values to NaN (missing)
    data[1::4] = np.nan  # Every 4th observation starting at index 1
    data[2::4] = np.nan  # Every 4th observation starting at index 2
    data[3::4] = np.nan  # Every 4th observation starting at index 3
    return data


@pytest.fixture
def sample_mask():
    """Sample observation mask for partial data."""
    mask = np.zeros(52, dtype=bool)
    mask[::4] = True  # Only every 4th observation
    return mask


@pytest.fixture
def sample_survey_proportions():
    """Sample survey proportion data (0-1)."""
    np.random.seed(42)
    return np.clip(np.random.randn(12) * 0.1 + 0.5, 0.1, 0.9)


@pytest.fixture
def sample_survey_config():
    """Sample aggregated survey configuration."""
    # 48 weeks -> 12 monthly surveys
    aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
    return AggregatedSurveyConfig(
        aggregation_map=aggregation_map,
        sample_sizes=tuple([500] * 12),
        likelihood=AggregatedSurveyLikelihood.BINOMIAL,
        design_effect=1.0,
        aggregation_function="mean",
    )


# =============================================================================
# build_gaussian_likelihood Tests
# =============================================================================


class TestBuildGaussianLikelihood:
    """Tests for build_gaussian_likelihood function."""

    def test_basic_build(self, sample_observed_data):
        """Test basic Gaussian likelihood building."""
        with pm.Model() as model:
            mu = pt.ones(52) * 1000
            y_obs, sigma = build_gaussian_likelihood(
                name="y",
                mu=mu,
                observed=sample_observed_data,
            )

            # Check variables created
            var_names = [v.name for v in model.free_RVs]
            assert "y_sigma" in var_names
            assert "y" in [v.name for v in model.observed_RVs]

    def test_with_custom_sigma_prior(self, sample_observed_data):
        """Test with custom sigma prior."""
        with pm.Model() as model:
            mu = pt.ones(52) * 1000
            y_obs, sigma = build_gaussian_likelihood(
                name="y",
                mu=mu,
                observed=sample_observed_data,
                sigma_prior_sigma=1.0,
            )

            assert sigma is not None

    def test_with_dims(self, sample_observed_data):
        """Test with dimension specification."""
        with pm.Model(coords={"obs": range(52)}) as model:
            mu = pt.ones(52) * 1000
            y_obs, sigma = build_gaussian_likelihood(
                name="y",
                mu=mu,
                observed=sample_observed_data,
                dims="obs",
            )

            assert y_obs is not None


# =============================================================================
# build_partial_observation_model Tests
# =============================================================================


class TestBuildPartialObservationModel:
    """Tests for build_partial_observation_model function."""

    def test_with_observed_periods(self, sample_partial_data, sample_mask):
        """Test with some observed periods."""
        with pm.Model() as model:
            latent = pt.ones(52) * 50
            obs_rv, sigma = build_partial_observation_model(
                name="mediator",
                latent=latent,
                observed=sample_partial_data,
                mask=sample_mask,
                sigma_prior_sigma=0.1,
            )

            # Should create observation model
            assert obs_rv is not None
            assert "mediator_obs_sigma" in [v.name for v in model.free_RVs]

    def test_with_no_observations(self):
        """Test when mask has no True values."""
        data = np.full(52, np.nan)
        mask = np.zeros(52, dtype=bool)

        with pm.Model() as model:
            latent = pt.ones(52) * 50
            obs_rv, sigma = build_partial_observation_model(
                name="mediator",
                latent=latent,
                observed=data,
                mask=mask,
            )

            # Should return None for obs_rv but still create sigma
            assert obs_rv is None
            assert sigma is not None

    def test_custom_sigma_prior(self, sample_partial_data, sample_mask):
        """Test with custom sigma prior."""
        with pm.Model() as model:
            latent = pt.ones(52) * 50
            obs_rv, sigma = build_partial_observation_model(
                name="test",
                latent=latent,
                observed=sample_partial_data,
                mask=sample_mask,
                sigma_prior_sigma=0.2,
            )

            assert sigma is not None


# =============================================================================
# build_multivariate_likelihood Tests
# =============================================================================


class TestBuildMultivariateLikelihood:
    """Tests for build_multivariate_likelihood function."""

    def test_basic_build(self, sample_multivariate_data):
        """Test basic multivariate likelihood building."""
        with pm.Model() as model:
            mu = pt.ones((52, 2)) * 1000
            y_obs, chol, corr = build_multivariate_likelihood(
                name="y",
                mu=mu,
                observed=sample_multivariate_data,
                n_outcomes=2,
            )

            assert y_obs is not None
            assert chol is not None
            assert corr is not None
            assert "y_correlation" in model.named_vars

    def test_lkj_eta_parameter(self, sample_multivariate_data):
        """Test with different LKJ eta values."""
        for eta in [0.5, 1.0, 2.0, 4.0]:
            with pm.Model() as model:
                mu = pt.ones((52, 2)) * 1000
                y_obs, chol, corr = build_multivariate_likelihood(
                    name="y",
                    mu=mu,
                    observed=sample_multivariate_data,
                    n_outcomes=2,
                    lkj_eta=eta,
                )

                assert y_obs is not None

    def test_custom_sigma_prior(self, sample_multivariate_data):
        """Test with custom sigma prior."""
        with pm.Model() as model:
            mu = pt.ones((52, 2)) * 1000
            y_obs, chol, corr = build_multivariate_likelihood(
                name="y",
                mu=mu,
                observed=sample_multivariate_data,
                n_outcomes=2,
                sigma_prior_sigma=1.0,
            )

            assert y_obs is not None


# =============================================================================
# build_aggregated_survey_observation Tests
# =============================================================================


class TestBuildAggregatedSurveyObservation:
    """Tests for build_aggregated_survey_observation function."""

    def test_binomial_likelihood(self, sample_survey_proportions, sample_survey_config):
        """Test binomial likelihood for survey data."""
        with pm.Model() as model:
            # Latent values at model frequency (48 weeks)
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=sample_survey_config,
                is_proportion=True,
            )

            # Should create binomial observation
            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_normal_likelihood(self, sample_survey_proportions):
        """Test normal approximation likelihood."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
            likelihood=AggregatedSurveyLikelihood.NORMAL,
        )

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=config,
                is_proportion=True,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_beta_binomial_likelihood(self, sample_survey_proportions):
        """Test beta-binomial likelihood for overdispersed data."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
            likelihood=AggregatedSurveyLikelihood.BETA_BINOMIAL,
            overdispersion_prior_sigma=0.1,
        )

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=config,
                is_proportion=True,
            )

            # Should create kappa parameter for overdispersion
            assert "awareness_kappa" in [v.name for v in model.free_RVs]
            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_mean_aggregation(self, sample_survey_proportions, sample_survey_config):
        """Test mean aggregation function."""
        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=sample_survey_config,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_last_aggregation(self, sample_survey_proportions):
        """Test last aggregation function."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
            aggregation_function="last",
        )

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=config,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_sum_aggregation(self, sample_survey_proportions):
        """Test sum aggregation function."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
            aggregation_function="sum",
        )

        with pm.Model() as model:
            latent = pt.ones(48) * 0.1  # Lower values since sum

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=config,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_with_design_effect(self, sample_survey_proportions):
        """Test with design effect applied."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
            likelihood=AggregatedSurveyLikelihood.NORMAL,
            design_effect=2.0,  # Effective n = 250
        )

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=sample_survey_proportions,
                config=config,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_with_counts_not_proportions(self):
        """Test with count data instead of proportions."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(3)}
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500, 500, 500]),
        )

        # Count data
        observed_counts = np.array([250, 245, 260])

        with pm.Model() as model:
            latent = pt.ones(12) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=observed_counts,
                config=config,
                is_proportion=False,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]


# =============================================================================
# compute_survey_observation_indices Tests
# =============================================================================


class TestComputeSurveyObservationIndices:
    """Tests for compute_survey_observation_indices function."""

    def test_weekly_to_monthly_simple(self):
        """Test weekly to monthly conversion (simple)."""
        indices = compute_survey_observation_indices(
            model_frequency="weekly",
            survey_frequency="monthly",
            n_periods=52,
        )

        # 52 weeks / 4 weeks per month = 13 surveys
        assert len(indices) == 13
        assert indices[0] == (0, 1, 2, 3)
        assert indices[1] == (4, 5, 6, 7)

    def test_weekly_to_quarterly_simple(self):
        """Test weekly to quarterly conversion (simple)."""
        indices = compute_survey_observation_indices(
            model_frequency="weekly",
            survey_frequency="quarterly",
            n_periods=52,
        )

        # 52 weeks / 13 weeks per quarter = 4 surveys
        assert len(indices) == 4
        assert len(indices[0]) == 13

    def test_daily_to_weekly_simple(self):
        """Test daily to weekly conversion (simple)."""
        indices = compute_survey_observation_indices(
            model_frequency="daily",
            survey_frequency="weekly",
            n_periods=365,
        )

        # 365 days / 7 days per week = 52 surveys
        assert len(indices) == 52
        assert len(indices[0]) == 7

    def test_daily_to_monthly_simple(self):
        """Test daily to monthly conversion (simple)."""
        indices = compute_survey_observation_indices(
            model_frequency="daily",
            survey_frequency="monthly",
            n_periods=365,
        )

        # 365 days / 30 days per month = 12 surveys
        assert len(indices) == 12
        assert len(indices[0]) == 30

    def test_with_start_date_monthly(self):
        """Test calendar-based monthly aggregation."""
        indices = compute_survey_observation_indices(
            model_frequency="weekly",
            survey_frequency="monthly",
            n_periods=52,
            start_date="2023-01-02",  # First Monday of 2023
        )

        # Should create monthly groups based on calendar
        assert len(indices) > 0
        # All model period indices should be covered
        all_indices = set()
        for idx_tuple in indices.values():
            all_indices.update(idx_tuple)
        assert len(all_indices) == 52

    def test_with_start_date_quarterly(self):
        """Test calendar-based quarterly aggregation."""
        indices = compute_survey_observation_indices(
            model_frequency="weekly",
            survey_frequency="quarterly",
            n_periods=52,
            start_date="2023-01-02",
        )

        # Should create quarterly groups
        assert len(indices) <= 5  # At most 5 quarters in 52 weeks

    def test_unsupported_frequency_raises(self):
        """Test that unsupported frequency combination raises error."""
        with pytest.raises(ValueError, match="Unsupported frequency combination"):
            compute_survey_observation_indices(
                model_frequency="hourly",
                survey_frequency="monthly",
                n_periods=100,
            )

    def test_indices_are_tuples(self):
        """Test that returned indices are tuples."""
        indices = compute_survey_observation_indices(
            model_frequency="weekly",
            survey_frequency="monthly",
            n_periods=52,
        )

        for idx_tuple in indices.values():
            assert isinstance(idx_tuple, tuple)


# =============================================================================
# build_mediator_observation_dispatch Tests
# =============================================================================


class TestBuildMediatorObservationDispatch:
    """Tests for build_mediator_observation_dispatch function."""

    def test_fully_latent_no_observation(self):
        """Test fully latent mediator creates no observation model."""
        config = MediatorConfigExtended(
            name="awareness",
            observation_type=MediatorObservationType.FULLY_LATENT,
        )

        with pm.Model() as model:
            latent = pt.ones(52) * 0.5

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={},
                mediator_masks={},
            )

            # Should have no observed RVs for the mediator
            mediator_obs = [v.name for v in model.observed_RVs if "awareness" in v.name]
            assert len(mediator_obs) == 0

    def test_fully_observed_mediator(self):
        """Test fully observed mediator."""
        config = MediatorConfigExtended(
            name="traffic",
            observation_type=MediatorObservationType.FULLY_OBSERVED,
            observation_noise_sigma=0.05,
        )

        np.random.seed(42)
        obs_data = np.random.randn(52) * 10 + 100

        with pm.Model() as model:
            latent = pt.ones(52) * 100

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={"traffic": obs_data},
                mediator_masks={},
            )

            assert "traffic_obs" in [v.name for v in model.observed_RVs]

    def test_partially_observed_mediator(self):
        """Test partially observed mediator."""
        config = MediatorConfigExtended(
            name="awareness",
            observation_type=MediatorObservationType.PARTIALLY_OBSERVED,
            observation_noise_sigma=0.1,
        )

        np.random.seed(42)
        obs_data = np.full(52, np.nan)
        obs_data[::4] = np.random.randn(13) * 5 + 50  # Monthly observations

        with pm.Model() as model:
            latent = pt.ones(52) * 50

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={"awareness": obs_data},
                mediator_masks={},
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_partially_observed_with_explicit_mask(self):
        """Test partially observed mediator with explicit mask."""
        config = MediatorConfigExtended(
            name="awareness",
            observation_type=MediatorObservationType.PARTIALLY_OBSERVED,
            observation_noise_sigma=0.1,
        )

        np.random.seed(42)
        obs_data = np.random.randn(52) * 5 + 50
        mask = np.zeros(52, dtype=bool)
        mask[::4] = True

        with pm.Model() as model:
            latent = pt.ones(52) * 50

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={"awareness": obs_data},
                mediator_masks={"awareness": mask},
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_aggregated_survey_mediator(self):
        """Test aggregated survey mediator."""
        aggregation_map = {i: tuple(range(i * 4, (i + 1) * 4)) for i in range(12)}
        survey_config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=tuple([500] * 12),
        )

        config = MediatorConfigExtended(
            name="awareness",
            observation_type=MediatorObservationType.AGGREGATED_SURVEY,
            aggregated_survey_config=survey_config,
        )

        np.random.seed(42)
        obs_data = np.clip(np.random.randn(12) * 0.1 + 0.5, 0.1, 0.9)

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={"awareness": obs_data},
                mediator_masks={},
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]

    def test_missing_data_no_observation(self):
        """Test that missing data results in no observation model."""
        config = MediatorConfigExtended(
            name="awareness",
            observation_type=MediatorObservationType.FULLY_OBSERVED,
        )

        with pm.Model() as model:
            latent = pt.ones(52) * 50

            build_mediator_observation_dispatch(
                med_config=config,
                mediator_latent=latent,
                mediator_data={},  # No data provided
                mediator_masks={},
            )

            # Should have no observed RVs for the mediator
            mediator_obs = [v.name for v in model.observed_RVs if "awareness" in v.name]
            assert len(mediator_obs) == 0

    def test_aggregated_survey_without_config_raises(self):
        """Test that aggregated survey without config raises error."""
        # Create config bypassing validation
        config = MediatorConfigExtended.__new__(MediatorConfigExtended)
        object.__setattr__(config, "name", "awareness")
        object.__setattr__(
            config, "observation_type", MediatorObservationType.AGGREGATED_SURVEY
        )
        object.__setattr__(config, "aggregated_survey_config", None)
        object.__setattr__(config, "observation_noise_sigma", 0.1)

        np.random.seed(42)
        obs_data = np.random.randn(12) * 0.1 + 0.5

        with pm.Model() as model:
            latent = pt.ones(48) * 0.5

            with pytest.raises(ValueError, match="aggregated_survey_config required"):
                build_mediator_observation_dispatch(
                    med_config=config,
                    mediator_latent=latent,
                    mediator_data={"awareness": obs_data},
                    mediator_masks={},
                )


# =============================================================================
# Integration Tests
# =============================================================================


class TestObservationIntegration:
    """Integration tests for observation models."""

    def test_gaussian_with_prior_sampling(self, sample_observed_data):
        """Test Gaussian likelihood can sample from prior."""
        with pm.Model() as model:
            intercept = pm.Normal("intercept", mu=1000, sigma=100)
            mu = pt.ones(52) * intercept

            y_obs, sigma = build_gaussian_likelihood(
                name="y",
                mu=mu,
                observed=sample_observed_data,
            )

            # Should be able to sample from prior
            prior = pm.sample_prior_predictive(samples=10, random_seed=42)
            assert "y_sigma" in prior.prior

    def test_multivariate_with_prior_sampling(self, sample_multivariate_data):
        """Test multivariate likelihood can sample from prior."""
        with pm.Model() as model:
            intercept = pm.Normal("intercept", mu=1000, sigma=100, shape=2)
            mu = pt.ones((52, 2)) * intercept

            y_obs, chol, corr = build_multivariate_likelihood(
                name="y",
                mu=mu,
                observed=sample_multivariate_data,
                n_outcomes=2,
            )

            # Should be able to sample from prior
            prior = pm.sample_prior_predictive(samples=10, random_seed=42)
            assert "y_correlation" in prior.prior

    def test_survey_observation_with_varying_sample_sizes(self):
        """Test survey observation handles varying sample sizes."""
        aggregation_map = {
            0: (0, 1, 2, 3),
            1: (4, 5, 6, 7),
            2: (8, 9, 10, 11),
        }
        config = AggregatedSurveyConfig(
            aggregation_map=aggregation_map,
            sample_sizes=(500, 300, 600),  # Varying sample sizes
            likelihood=AggregatedSurveyLikelihood.BINOMIAL,
        )

        np.random.seed(42)
        obs_data = np.array([0.5, 0.45, 0.55])

        with pm.Model() as model:
            latent = pt.ones(12) * 0.5

            build_aggregated_survey_observation(
                name="awareness",
                latent=latent,
                observed_data=obs_data,
                config=config,
            )

            assert "awareness_obs" in [v.name for v in model.observed_RVs]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

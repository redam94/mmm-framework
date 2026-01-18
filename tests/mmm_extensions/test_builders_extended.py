"""
Test suite for extended builder classes in mmm_framework.mmm_extensions.builders.

Tests cover:
- AdstockConfigBuilder
- SaturationConfigBuilder
- EffectPriorConfigBuilder
- AggregatedSurveyConfigBuilder
- MediatorConfigBuilderExtended
- CrossEffectConfigBuilder
- CombinedModelConfigBuilder
- Factory functions (awareness_mediator, foot_traffic_mediator, etc.)
- survey_awareness_mediator factory

Note: Bypasses PyTensor compilation issues with special config.
"""

import pytensor
pytensor.config.exception_verbosity = 'high'
pytensor.config.cxx = ""

import numpy as np
import pytest

from mmm_framework.mmm_extensions.config import (
    SaturationType,
    MediatorType,
    MediatorObservationType,
    CrossEffectType,
    EffectConstraint,
    AggregatedSurveyLikelihood,
    AggregatedSurveyConfig,
)
from mmm_framework.mmm_extensions.builders import (
    # Base builders
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    EffectPriorConfigBuilder,
    # Mediator/Outcome builders
    MediatorConfigBuilder,
    MediatorConfigBuilderExtended,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    AggregatedSurveyConfigBuilder,
    # Model builders
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    # Factory functions
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
    survey_awareness_mediator,
    sparse_controls,
    selection_with_inclusion_probs,
    dense_controls,
)


# =============================================================================
# AdstockConfigBuilder Tests
# =============================================================================


class TestAdstockConfigBuilder:
    """Tests for AdstockConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        config = AdstockConfigBuilder().build()
        assert config.l_max == 8
        assert config.prior_type == "beta"
        assert config.prior_alpha == 2.0
        assert config.prior_beta == 2.0
        assert config.normalize is True

    def test_with_max_lag(self):
        """Test setting maximum lag."""
        config = AdstockConfigBuilder().with_max_lag(16).build()
        assert config.l_max == 16

    def test_with_beta_prior(self):
        """Test setting beta prior parameters."""
        config = AdstockConfigBuilder().with_beta_prior(alpha=3.0, beta=1.0).build()
        assert config.prior_type == "beta"
        assert config.prior_alpha == 3.0
        assert config.prior_beta == 1.0

    def test_with_uniform_prior(self):
        """Test setting uniform prior."""
        config = AdstockConfigBuilder().with_uniform_prior().build()
        assert config.prior_type == "uniform"

    def test_with_slow_decay(self):
        """Test slow decay preset."""
        config = AdstockConfigBuilder().with_slow_decay().build()
        assert config.prior_alpha == 3.0
        assert config.prior_beta == 1.0

    def test_with_fast_decay(self):
        """Test fast decay preset."""
        config = AdstockConfigBuilder().with_fast_decay().build()
        assert config.prior_alpha == 1.0
        assert config.prior_beta == 3.0

    def test_without_normalization(self):
        """Test disabling normalization."""
        config = AdstockConfigBuilder().without_normalization().build()
        assert config.normalize is False

    def test_fluent_chaining(self):
        """Test fluent method chaining."""
        config = (
            AdstockConfigBuilder()
            .with_max_lag(12)
            .with_slow_decay()
            .without_normalization()
            .build()
        )
        assert config.l_max == 12
        assert config.prior_alpha == 3.0
        assert config.normalize is False


# =============================================================================
# SaturationConfigBuilder Tests
# =============================================================================


class TestSaturationConfigBuilder:
    """Tests for SaturationConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults (logistic)."""
        config = SaturationConfigBuilder().build()
        assert config.type == SaturationType.LOGISTIC
        assert config.lam_prior_alpha == 3.0
        assert config.lam_prior_beta == 1.0

    def test_logistic_type(self):
        """Test logistic saturation configuration."""
        config = SaturationConfigBuilder().logistic(lam_alpha=4.0, lam_beta=2.0).build()
        assert config.type == SaturationType.LOGISTIC
        assert config.lam_prior_alpha == 4.0
        assert config.lam_prior_beta == 2.0

    def test_hill_type(self):
        """Test Hill saturation configuration."""
        config = (
            SaturationConfigBuilder()
            .hill(
                kappa_alpha=3.0,
                kappa_beta=3.0,
                slope_alpha=2.0,
                slope_beta=2.0,
            )
            .build()
        )
        assert config.type == SaturationType.HILL
        assert config.kappa_prior_alpha == 3.0
        assert config.kappa_prior_beta == 3.0
        assert config.slope_prior_alpha == 2.0
        assert config.slope_prior_beta == 2.0

    def test_with_strong_saturation(self):
        """Test strong saturation preset."""
        config = SaturationConfigBuilder().logistic().with_strong_saturation().build()
        assert config.lam_prior_alpha == 5.0
        assert config.lam_prior_beta == 1.0

    def test_with_weak_saturation(self):
        """Test weak saturation preset."""
        config = SaturationConfigBuilder().logistic().with_weak_saturation().build()
        assert config.lam_prior_alpha == 1.0
        assert config.lam_prior_beta == 2.0


# =============================================================================
# EffectPriorConfigBuilder Tests
# =============================================================================


class TestEffectPriorConfigBuilder:
    """Tests for EffectPriorConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        config = EffectPriorConfigBuilder().build()
        assert config.constraint == EffectConstraint.NONE
        assert config.mu == 0.0
        assert config.sigma == 1.0

    def test_unconstrained(self):
        """Test unconstrained effect."""
        config = EffectPriorConfigBuilder().unconstrained(mu=0.5, sigma=2.0).build()
        assert config.constraint == EffectConstraint.NONE
        assert config.mu == 0.5
        assert config.sigma == 2.0

    def test_positive(self):
        """Test positive constraint."""
        config = EffectPriorConfigBuilder().positive(sigma=0.5).build()
        assert config.constraint == EffectConstraint.POSITIVE
        assert config.sigma == 0.5

    def test_negative(self):
        """Test negative constraint."""
        config = EffectPriorConfigBuilder().negative(sigma=0.3).build()
        assert config.constraint == EffectConstraint.NEGATIVE
        assert config.sigma == 0.3

    def test_with_tight_prior(self):
        """Test tight prior modifier."""
        config = (
            EffectPriorConfigBuilder()
            .unconstrained(sigma=1.0)
            .with_tight_prior()
            .build()
        )
        assert config.sigma == 0.5

    def test_with_wide_prior(self):
        """Test wide prior modifier."""
        config = (
            EffectPriorConfigBuilder()
            .unconstrained(sigma=1.0)
            .with_wide_prior()
            .build()
        )
        assert config.sigma == 2.0

    def test_chained_modifiers(self):
        """Test chaining multiple modifiers."""
        config = (
            EffectPriorConfigBuilder()
            .positive(sigma=1.0)
            .with_tight_prior()
            .with_tight_prior()
            .build()
        )
        assert config.sigma == 0.25


# =============================================================================
# AggregatedSurveyConfigBuilder Tests
# =============================================================================


class TestAggregatedSurveyConfigBuilder:
    """Tests for AggregatedSurveyConfigBuilder."""

    def test_with_aggregation_map(self):
        """Test setting aggregation map directly."""
        agg_map = {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map(agg_map)
            .with_sample_sizes([500, 500])
            .build()
        )
        assert len(config.aggregation_map) == 2
        assert config.aggregation_map[0] == (0, 1, 2, 3)

    def test_from_frequencies_weekly_monthly(self):
        """Test computing aggregation map from frequencies."""
        config = (
            AggregatedSurveyConfigBuilder()
            .from_frequencies("weekly", "monthly", n_periods=52)
            .with_sample_sizes([500] * 13)
            .build()
        )
        assert len(config.aggregation_map) == 13

    def test_monthly_in_weekly_model_convenience(self):
        """Test monthly_in_weekly_model convenience method."""
        config = (
            AggregatedSurveyConfigBuilder()
            .monthly_in_weekly_model(n_weeks=52)
            .with_sample_sizes([500] * 13)
            .build()
        )
        assert len(config.aggregation_map) == 13

    def test_quarterly_in_weekly_model_convenience(self):
        """Test quarterly_in_weekly_model convenience method."""
        config = (
            AggregatedSurveyConfigBuilder()
            .quarterly_in_weekly_model(n_weeks=52)
            .with_sample_sizes([1000] * 4)
            .build()
        )
        assert len(config.aggregation_map) == 4

    def test_with_constant_sample_size(self):
        """Test setting constant sample size."""
        config = (
            AggregatedSurveyConfigBuilder()
            .monthly_in_weekly_model(n_weeks=52)
            .with_constant_sample_size(500)
            .build()
        )
        assert all(n == 500 for n in config.sample_sizes)

    def test_constant_sample_size_before_map_raises(self):
        """Test error when setting constant size before aggregation map."""
        with pytest.raises(ValueError, match="Set aggregation_map before sample sizes"):
            AggregatedSurveyConfigBuilder().with_constant_sample_size(500)

    def test_binomial_likelihood(self):
        """Test binomial likelihood setting."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .binomial()
            .build()
        )
        assert config.likelihood == AggregatedSurveyLikelihood.BINOMIAL

    def test_normal_approximation_likelihood(self):
        """Test normal approximation likelihood setting."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .normal_approximation()
            .build()
        )
        assert config.likelihood == AggregatedSurveyLikelihood.NORMAL

    def test_beta_binomial_likelihood(self):
        """Test beta-binomial likelihood setting."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .beta_binomial(overdispersion_prior_sigma=0.2)
            .build()
        )
        assert config.likelihood == AggregatedSurveyLikelihood.BETA_BINOMIAL
        assert config.overdispersion_prior_sigma == 0.2

    def test_with_design_effect(self):
        """Test design effect setting."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .with_design_effect(1.5)
            .build()
        )
        assert config.design_effect == 1.5

    def test_with_effective_sample_size(self):
        """Test computing design effect from effective sample size."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .with_effective_sample_size(nominal_n=500, effective_n=250)
            .build()
        )
        assert config.design_effect == 2.0

    def test_aggregate_by_mean(self):
        """Test mean aggregation function."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .aggregate_by_mean()
            .build()
        )
        assert config.aggregation_function == "mean"

    def test_aggregate_by_last(self):
        """Test last aggregation function."""
        config = (
            AggregatedSurveyConfigBuilder()
            .with_aggregation_map({0: (0, 1)})
            .with_sample_sizes([500])
            .aggregate_by_last()
            .build()
        )
        assert config.aggregation_function == "last"

    def test_build_without_aggregation_map_raises(self):
        """Test error when building without aggregation map."""
        with pytest.raises(ValueError, match="aggregation_map is required"):
            AggregatedSurveyConfigBuilder().with_sample_sizes([500]).build()

    def test_build_without_sample_sizes_raises(self):
        """Test error when building without sample sizes."""
        with pytest.raises(ValueError, match="sample_sizes is required"):
            AggregatedSurveyConfigBuilder().with_aggregation_map({0: (0, 1)}).build()


# =============================================================================
# MediatorConfigBuilderExtended Tests
# =============================================================================


class TestMediatorConfigBuilderExtended:
    """Tests for MediatorConfigBuilderExtended."""

    def test_basic_build(self):
        """Test basic configuration."""
        config = MediatorConfigBuilderExtended("awareness").build()
        assert config.name == "awareness"
        assert config.observation_type == MediatorObservationType.PARTIALLY_OBSERVED

    def test_fully_latent(self):
        """Test fully latent observation type."""
        config = MediatorConfigBuilderExtended("latent").fully_latent().build()
        assert config.observation_type == MediatorObservationType.FULLY_LATENT

    def test_partially_observed(self):
        """Test partially observed observation type."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .partially_observed(observation_noise=0.15)
            .build()
        )
        assert config.observation_type == MediatorObservationType.PARTIALLY_OBSERVED
        assert config.observation_noise_sigma == 0.15

    def test_fully_observed(self):
        """Test fully observed observation type."""
        config = (
            MediatorConfigBuilderExtended("traffic")
            .fully_observed(observation_noise=0.05)
            .build()
        )
        assert config.observation_type == MediatorObservationType.FULLY_OBSERVED
        assert config.observation_noise_sigma == 0.05

    def test_aggregated_survey(self):
        """Test aggregated survey observation type."""
        survey_config = AggregatedSurveyConfig(
            aggregation_map={0: (0, 1, 2, 3)},
            sample_sizes=(500,),
        )
        config = (
            MediatorConfigBuilderExtended("awareness")
            .aggregated_survey()
            .with_survey_config(survey_config)
            .build()
        )
        assert config.observation_type == MediatorObservationType.AGGREGATED_SURVEY
        assert config.aggregated_survey_config is not None

    def test_with_positive_media_effect(self):
        """Test positive media effect."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .with_positive_media_effect(sigma=1.5)
            .build()
        )
        assert config.media_effect_constraint == "positive"
        assert config.media_effect_sigma == 1.5

    def test_with_unconstrained_media_effect(self):
        """Test unconstrained media effect."""
        config = (
            MediatorConfigBuilderExtended("test")
            .with_unconstrained_media_effect(sigma=2.0)
            .build()
        )
        assert config.media_effect_constraint == "none"
        assert config.media_effect_sigma == 2.0

    def test_with_direct_effect(self):
        """Test enabling direct effect."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .with_direct_effect(sigma=0.3)
            .build()
        )
        assert config.allow_direct_effect is True
        assert config.direct_effect_sigma == 0.3

    def test_without_direct_effect(self):
        """Test disabling direct effect."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .without_direct_effect()
            .build()
        )
        assert config.allow_direct_effect is False

    def test_with_adstock(self):
        """Test enabling adstock."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .with_adstock()
            .build()
        )
        assert config.apply_adstock is True

    def test_without_adstock(self):
        """Test disabling adstock."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .without_adstock()
            .build()
        )
        assert config.apply_adstock is False

    def test_with_saturation(self):
        """Test enabling saturation."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .with_saturation()
            .build()
        )
        assert config.apply_saturation is True

    def test_without_saturation(self):
        """Test disabling saturation."""
        config = (
            MediatorConfigBuilderExtended("awareness")
            .without_saturation()
            .build()
        )
        assert config.apply_saturation is False


# =============================================================================
# CrossEffectConfigBuilder Tests
# =============================================================================


class TestCrossEffectConfigBuilder:
    """Tests for CrossEffectConfigBuilder."""

    def test_default_build(self):
        """Test default build (cannibalization)."""
        config = CrossEffectConfigBuilder("source", "target").build()
        assert config.source_outcome == "source"
        assert config.target_outcome == "target"
        assert config.effect_type == CrossEffectType.CANNIBALIZATION
        assert config.prior_sigma == 0.3
        assert config.lag == 0

    def test_cannibalization_type(self):
        """Test cannibalization effect type."""
        config = CrossEffectConfigBuilder("a", "b").cannibalization().build()
        assert config.effect_type == CrossEffectType.CANNIBALIZATION

    def test_halo_type(self):
        """Test halo effect type."""
        config = CrossEffectConfigBuilder("a", "b").halo().build()
        assert config.effect_type == CrossEffectType.HALO

    def test_symmetric_type(self):
        """Test symmetric effect type."""
        config = CrossEffectConfigBuilder("a", "b").symmetric().build()
        assert config.effect_type == CrossEffectType.SYMMETRIC

    def test_asymmetric_type(self):
        """Test asymmetric effect type."""
        config = CrossEffectConfigBuilder("a", "b").asymmetric().build()
        assert config.effect_type == CrossEffectType.ASYMMETRIC

    def test_with_prior_sigma(self):
        """Test setting prior sigma."""
        config = CrossEffectConfigBuilder("a", "b").with_prior_sigma(0.5).build()
        assert config.prior_sigma == 0.5

    def test_modulated_by_promotion(self):
        """Test promotion modulation."""
        config = (
            CrossEffectConfigBuilder("a", "b")
            .modulated_by_promotion("promo_col")
            .build()
        )
        assert config.promotion_modulated is True
        assert config.promotion_column == "promo_col"

    def test_always_active(self):
        """Test always active (no promotion modulation)."""
        config = CrossEffectConfigBuilder("a", "b").always_active().build()
        assert config.promotion_modulated is False
        assert config.promotion_column is None

    def test_with_lag(self):
        """Test setting temporal lag."""
        config = CrossEffectConfigBuilder("a", "b").with_lag(2).build()
        assert config.lag == 2

    def test_lagged_convenience(self):
        """Test lagged convenience method."""
        config = CrossEffectConfigBuilder("a", "b").lagged().build()
        assert config.lag == 1


# =============================================================================
# CombinedModelConfigBuilder Tests
# =============================================================================


class TestCombinedModelConfigBuilder:
    """Tests for CombinedModelConfigBuilder."""

    def test_basic_build(self):
        """Test basic combined model building."""
        config = CombinedModelConfigBuilder().build()
        assert config.nested is not None
        assert config.multivariate is not None

    def test_add_mediator(self):
        """Test adding mediator."""
        mediator = MediatorConfigBuilder("awareness").build()
        config = CombinedModelConfigBuilder().add_mediator(mediator).build()
        assert len(config.nested.mediators) == 1

    def test_with_awareness_mediator(self):
        """Test with_awareness_mediator convenience."""
        config = (
            CombinedModelConfigBuilder()
            .with_awareness_mediator("brand_awareness")
            .build()
        )
        assert len(config.nested.mediators) == 1
        assert config.nested.mediators[0].name == "brand_awareness"

    def test_with_traffic_mediator(self):
        """Test with_traffic_mediator convenience."""
        config = (
            CombinedModelConfigBuilder()
            .with_traffic_mediator("store_visits")
            .build()
        )
        assert len(config.nested.mediators) == 1
        assert config.nested.mediators[0].name == "store_visits"

    def test_add_outcome(self):
        """Test adding outcome."""
        outcome = OutcomeConfigBuilder("sales", column="sales_col").build()
        config = CombinedModelConfigBuilder().add_outcome(outcome).build()
        assert len(config.multivariate.outcomes) == 1

    def test_with_outcomes(self):
        """Test with_outcomes convenience."""
        config = (
            CombinedModelConfigBuilder()
            .with_outcomes("product_a", "product_b")
            .build()
        )
        assert len(config.multivariate.outcomes) == 2

    def test_map_channels_to_mediator(self):
        """Test mapping channels to mediator."""
        config = (
            CombinedModelConfigBuilder()
            .with_awareness_mediator("awareness")
            .map_channels_to_mediator("awareness", ["tv", "digital"])
            .build()
        )
        assert "awareness" in config.nested.media_to_mediator_map
        assert "tv" in config.nested.media_to_mediator_map["awareness"]

    def test_map_mediator_to_outcomes(self):
        """Test mapping mediator to outcomes."""
        config = (
            CombinedModelConfigBuilder()
            .with_awareness_mediator("awareness")
            .with_outcomes("product_a", "product_b")
            .map_mediator_to_outcomes("awareness", ["product_a"])
            .build()
        )
        assert "awareness" in config.mediator_to_outcome_map
        assert "product_a" in config.mediator_to_outcome_map["awareness"]

    def test_with_cannibalization(self):
        """Test adding cannibalization effect."""
        config = (
            CombinedModelConfigBuilder()
            .with_outcomes("a", "b")
            .with_cannibalization("b", "a")
            .build()
        )
        assert len(config.multivariate.cross_effects) == 1
        assert config.multivariate.cross_effects[0].effect_type == CrossEffectType.CANNIBALIZATION

    def test_with_halo_effect(self):
        """Test adding halo effect."""
        config = (
            CombinedModelConfigBuilder()
            .with_outcomes("a", "b")
            .with_halo_effect("a", "b")
            .build()
        )
        assert len(config.multivariate.cross_effects) == 1
        assert config.multivariate.cross_effects[0].effect_type == CrossEffectType.HALO

    def test_with_lkj_eta(self):
        """Test setting LKJ eta parameter."""
        config = CombinedModelConfigBuilder().with_lkj_eta(4.0).build()
        assert config.multivariate.lkj_eta == 4.0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_awareness_mediator(self):
        """Test awareness_mediator factory."""
        config = awareness_mediator(name="brand_awareness", observation_noise=0.12)
        assert config.name == "brand_awareness"
        assert config.mediator_type == MediatorType.PARTIALLY_OBSERVED
        assert config.observation_noise_sigma == 0.12
        assert config.adstock.l_max == 12

    def test_awareness_mediator_defaults(self):
        """Test awareness_mediator factory with defaults."""
        config = awareness_mediator()
        assert config.name == "awareness"
        assert config.observation_noise_sigma == 0.15

    def test_foot_traffic_mediator(self):
        """Test foot_traffic_mediator factory."""
        config = foot_traffic_mediator(name="store_visits", observation_noise=0.08)
        assert config.name == "store_visits"
        assert config.mediator_type == MediatorType.FULLY_OBSERVED
        assert config.observation_noise_sigma == 0.08

    def test_foot_traffic_mediator_defaults(self):
        """Test foot_traffic_mediator factory with defaults."""
        config = foot_traffic_mediator()
        assert config.name == "foot_traffic"
        assert config.observation_noise_sigma == 0.05

    def test_cannibalization_effect(self):
        """Test cannibalization_effect factory."""
        config = cannibalization_effect(source="multipack", target="single")
        assert config.source_outcome == "multipack"
        assert config.target_outcome == "single"
        assert config.effect_type == CrossEffectType.CANNIBALIZATION
        assert config.promotion_modulated is True

    def test_cannibalization_effect_with_promotion(self):
        """Test cannibalization_effect with promotion column."""
        config = cannibalization_effect(
            source="a",
            target="b",
            promotion_column="promo_a",
        )
        assert config.promotion_column == "promo_a"

    def test_cannibalization_effect_lagged(self):
        """Test cannibalization_effect with lag."""
        config = cannibalization_effect(
            source="a",
            target="b",
            lagged=True,
        )
        assert config.lag == 1

    def test_halo_effect(self):
        """Test halo_effect factory."""
        config = halo_effect(source="premium", target="budget")
        assert config.source_outcome == "premium"
        assert config.target_outcome == "budget"
        assert config.effect_type == CrossEffectType.HALO
        assert config.promotion_modulated is False

    def test_sparse_controls(self):
        """Test sparse_controls factory."""
        config = sparse_controls(expected_nonzero=5)
        assert config.horseshoe.expected_nonzero == 5

    def test_sparse_controls_with_confounders(self):
        """Test sparse_controls with confounders."""
        config = sparse_controls(3, "price", "distribution")
        assert "price" in config.exclude_variables
        assert "distribution" in config.exclude_variables

    def test_selection_with_inclusion_probs(self):
        """Test selection_with_inclusion_probs factory."""
        config = selection_with_inclusion_probs(prior_inclusion=0.3)
        assert config.spike_slab.prior_inclusion_prob == 0.3

    def test_selection_with_inclusion_probs_with_confounders(self):
        """Test selection_with_inclusion_probs with confounders."""
        config = selection_with_inclusion_probs(0.4, "confounder1")
        assert "confounder1" in config.exclude_variables

    def test_dense_controls(self):
        """Test dense_controls factory."""
        config = dense_controls(regularization=2.0)
        assert config.lasso.regularization == 2.0

    def test_dense_controls_with_confounders(self):
        """Test dense_controls with confounders."""
        config = dense_controls(1.5, "conf1", "conf2")
        assert "conf1" in config.exclude_variables
        assert "conf2" in config.exclude_variables


# =============================================================================
# survey_awareness_mediator Factory Tests
# =============================================================================


class TestSurveyAwarenessMediatorFactory:
    """Tests for survey_awareness_mediator factory function."""

    def test_basic_creation(self):
        """Test basic survey_awareness_mediator creation."""
        config = survey_awareness_mediator(
            name="brand_awareness",
            n_model_periods=52,
            sample_sizes=500,
        )
        assert config.name == "brand_awareness"
        assert config.observation_type == MediatorObservationType.AGGREGATED_SURVEY
        assert config.aggregated_survey_config is not None

    def test_with_list_sample_sizes(self):
        """Test with list of sample sizes."""
        # 52 weeks -> 13 monthly surveys
        sample_sizes = [500 + i * 10 for i in range(13)]
        config = survey_awareness_mediator(
            name="awareness",
            n_model_periods=52,
            sample_sizes=sample_sizes,
        )
        assert config.aggregated_survey_config is not None
        assert len(config.aggregated_survey_config.sample_sizes) == 13

    def test_with_constant_sample_size(self):
        """Test with constant sample size."""
        config = survey_awareness_mediator(
            name="awareness",
            n_model_periods=52,
            sample_sizes=500,
        )
        assert all(
            n == 500 for n in config.aggregated_survey_config.sample_sizes
        )

    def test_with_design_effect(self):
        """Test with custom design effect."""
        config = survey_awareness_mediator(
            name="awareness",
            n_model_periods=52,
            sample_sizes=500,
            design_effect=1.5,
        )
        assert config.aggregated_survey_config.design_effect == 1.5

    def test_with_beta_binomial(self):
        """Test with beta-binomial likelihood."""
        config = survey_awareness_mediator(
            name="awareness",
            n_model_periods=52,
            sample_sizes=500,
            use_beta_binomial=True,
        )
        assert (
            config.aggregated_survey_config.likelihood
            == AggregatedSurveyLikelihood.BETA_BINOMIAL
        )

    def test_quarterly_frequency(self):
        """Test with quarterly survey frequency."""
        config = survey_awareness_mediator(
            name="awareness",
            n_model_periods=52,
            sample_sizes=[1000, 1000, 1000, 1000],
            survey_frequency="quarterly",
        )
        assert len(config.aggregated_survey_config.aggregation_map) == 4

    def test_default_parameters(self):
        """Test default parameter values."""
        config = survey_awareness_mediator(
            n_model_periods=104,
            sample_sizes=500,
        )
        assert config.name == "awareness"
        assert config.allow_direct_effect is True
        assert config.aggregated_survey_config.design_effect == 1.0
        assert (
            config.aggregated_survey_config.likelihood
            == AggregatedSurveyLikelihood.BINOMIAL
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

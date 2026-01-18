"""
Test suite for extended configuration classes in mmm_framework.mmm_extensions.config.

Tests cover:
- AggregatedSurveyConfig and its validation
- MediatorConfigExtended with survey support
- MediatorObservationType and AggregatedSurveyLikelihood enums
- AdstockConfig and SaturationConfig
- EffectPriorConfig and its constraint handling
- OutcomeConfig
- CrossEffectConfig
- NestedModelConfig
- MultivariateModelConfig
- CombinedModelConfig

Note: Bypasses PyTensor compilation issues with special config.
"""

import pytensor
pytensor.config.exception_verbosity = 'high'
pytensor.config.cxx = ""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from mmm_framework.mmm_extensions.config import (
    # Enums
    MediatorType,
    MediatorObservationType,
    CrossEffectType,
    EffectConstraint,
    AggregatedSurveyLikelihood,
    VariableSelectionMethod,
    # Base configs
    AdstockConfig,
    SaturationConfig,
    EffectPriorConfig,
    # Mediator/Outcome configs
    MediatorConfig,
    MediatorConfigExtended,
    AggregatedSurveyConfig,
    OutcomeConfig,
    CrossEffectConfig,
    # Top-level configs
    NestedModelConfig,
    MultivariateModelConfig,
    CombinedModelConfig,
    # Factory functions
    sparse_selection_config,
    dense_selection_config,
    inclusion_prob_selection_config,
)
from mmm_framework.config import SaturationType


# =============================================================================
# Enum Tests
# =============================================================================


class TestMediatorType:
    """Tests for MediatorType enum."""

    def test_values(self):
        """Test all MediatorType values exist."""
        assert MediatorType.FULLY_OBSERVED.value == "fully_observed"
        assert MediatorType.PARTIALLY_OBSERVED.value == "partially_observed"
        assert MediatorType.AGGREGATED_SURVEY.value == "aggregated_survey"
        assert MediatorType.FULLY_LATENT.value == "fully_latent"

    def test_is_string_enum(self):
        """Test MediatorType inherits from str."""
        assert isinstance(MediatorType.FULLY_OBSERVED, str)


class TestMediatorObservationType:
    """Tests for MediatorObservationType enum."""

    def test_values(self):
        """Test all MediatorObservationType values exist."""
        assert MediatorObservationType.FULLY_OBSERVED.value == "fully_observed"
        assert MediatorObservationType.PARTIALLY_OBSERVED.value == "partially_observed"
        assert MediatorObservationType.AGGREGATED_SURVEY.value == "aggregated_survey"
        assert MediatorObservationType.FULLY_LATENT.value == "fully_latent"


class TestCrossEffectType:
    """Tests for CrossEffectType enum."""

    def test_values(self):
        """Test all CrossEffectType values exist."""
        assert CrossEffectType.CANNIBALIZATION.value == "cannibalization"
        assert CrossEffectType.HALO.value == "halo"
        assert CrossEffectType.SYMMETRIC.value == "symmetric"
        assert CrossEffectType.ASYMMETRIC.value == "asymmetric"


class TestEffectConstraint:
    """Tests for EffectConstraint enum."""

    def test_values(self):
        """Test all EffectConstraint values exist."""
        assert EffectConstraint.NONE.value == "none"
        assert EffectConstraint.POSITIVE.value == "positive"
        assert EffectConstraint.NEGATIVE.value == "negative"


class TestAggregatedSurveyLikelihood:
    """Tests for AggregatedSurveyLikelihood enum."""

    def test_values(self):
        """Test all AggregatedSurveyLikelihood values exist."""
        assert AggregatedSurveyLikelihood.BINOMIAL.value == "binomial"
        assert AggregatedSurveyLikelihood.NORMAL.value == "normal"
        assert AggregatedSurveyLikelihood.BETA_BINOMIAL.value == "beta_binomial"


# =============================================================================
# AdstockConfig Tests
# =============================================================================


class TestAdstockConfig:
    """Tests for AdstockConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = AdstockConfig()
        assert config.l_max == 8
        assert config.prior_type == "beta"
        assert config.prior_alpha == 2.0
        assert config.prior_beta == 2.0
        assert config.normalize is True

    def test_custom_values(self):
        """Test custom values."""
        config = AdstockConfig(
            l_max=12,
            prior_type="uniform",
            prior_alpha=3.0,
            prior_beta=1.0,
            normalize=False,
        )
        assert config.l_max == 12
        assert config.prior_type == "uniform"
        assert config.prior_alpha == 3.0
        assert config.prior_beta == 1.0
        assert config.normalize is False

    def test_immutability(self):
        """Test that config is frozen."""
        config = AdstockConfig()
        with pytest.raises(FrozenInstanceError):
            config.l_max = 16


# =============================================================================
# SaturationConfig Tests
# =============================================================================


class TestSaturationConfig:
    """Tests for SaturationConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = SaturationConfig()
        assert config.type == SaturationType.LOGISTIC
        assert config.lam_prior_alpha == 3.0
        assert config.lam_prior_beta == 1.0
        assert config.kappa_prior_alpha == 2.0
        assert config.kappa_prior_beta == 2.0
        assert config.slope_prior_alpha == 3.0
        assert config.slope_prior_beta == 1.0

    def test_hill_configuration(self):
        """Test Hill saturation configuration."""
        config = SaturationConfig(
            type=SaturationType.HILL,
            kappa_prior_alpha=3.0,
            kappa_prior_beta=3.0,
            slope_prior_alpha=2.0,
            slope_prior_beta=2.0,
        )
        assert config.type == SaturationType.HILL
        assert config.kappa_prior_alpha == 3.0

    def test_immutability(self):
        """Test that config is frozen."""
        config = SaturationConfig()
        with pytest.raises(FrozenInstanceError):
            config.type = SaturationType.HILL


# =============================================================================
# EffectPriorConfig Tests
# =============================================================================


class TestEffectPriorConfig:
    """Tests for EffectPriorConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = EffectPriorConfig()
        assert config.constraint == EffectConstraint.NONE
        assert config.mu == 0.0
        assert config.sigma == 1.0

    def test_positive_constraint(self):
        """Test positive constraint configuration."""
        config = EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE,
            sigma=0.5,
        )
        assert config.constraint == EffectConstraint.POSITIVE
        assert config.sigma == 0.5

    def test_negative_constraint(self):
        """Test negative constraint configuration."""
        config = EffectPriorConfig(
            constraint=EffectConstraint.NEGATIVE,
            sigma=0.3,
        )
        assert config.constraint == EffectConstraint.NEGATIVE

    def test_immutability(self):
        """Test that config is frozen."""
        config = EffectPriorConfig()
        with pytest.raises(FrozenInstanceError):
            config.sigma = 2.0


# =============================================================================
# AggregatedSurveyConfig Tests
# =============================================================================


class TestAggregatedSurveyConfig:
    """Tests for AggregatedSurveyConfig dataclass."""

    @pytest.fixture
    def valid_aggregation_map(self):
        """Valid aggregation map for monthly surveys in weekly model."""
        return {
            0: (0, 1, 2, 3),
            1: (4, 5, 6, 7),
            2: (8, 9, 10, 11),
        }

    @pytest.fixture
    def valid_sample_sizes(self):
        """Valid sample sizes matching aggregation map."""
        return (500, 450, 520)

    def test_basic_creation(self, valid_aggregation_map, valid_sample_sizes):
        """Test basic initialization."""
        config = AggregatedSurveyConfig(
            aggregation_map=valid_aggregation_map,
            sample_sizes=valid_sample_sizes,
        )
        assert len(config.aggregation_map) == 3
        assert len(config.sample_sizes) == 3
        assert config.likelihood == AggregatedSurveyLikelihood.BINOMIAL
        assert config.design_effect == 1.0
        assert config.aggregation_function == "mean"

    def test_with_design_effect(self, valid_aggregation_map, valid_sample_sizes):
        """Test with custom design effect."""
        config = AggregatedSurveyConfig(
            aggregation_map=valid_aggregation_map,
            sample_sizes=valid_sample_sizes,
            design_effect=1.5,
        )
        assert config.design_effect == 1.5

    def test_with_normal_likelihood(self, valid_aggregation_map, valid_sample_sizes):
        """Test with normal likelihood."""
        config = AggregatedSurveyConfig(
            aggregation_map=valid_aggregation_map,
            sample_sizes=valid_sample_sizes,
            likelihood=AggregatedSurveyLikelihood.NORMAL,
        )
        assert config.likelihood == AggregatedSurveyLikelihood.NORMAL

    def test_with_beta_binomial_likelihood(self, valid_aggregation_map, valid_sample_sizes):
        """Test with beta-binomial likelihood."""
        config = AggregatedSurveyConfig(
            aggregation_map=valid_aggregation_map,
            sample_sizes=valid_sample_sizes,
            likelihood=AggregatedSurveyLikelihood.BETA_BINOMIAL,
            overdispersion_prior_sigma=0.2,
        )
        assert config.likelihood == AggregatedSurveyLikelihood.BETA_BINOMIAL
        assert config.overdispersion_prior_sigma == 0.2

    def test_aggregation_function_options(self, valid_aggregation_map, valid_sample_sizes):
        """Test different aggregation functions."""
        for func in ["mean", "sum", "last"]:
            config = AggregatedSurveyConfig(
                aggregation_map=valid_aggregation_map,
                sample_sizes=valid_sample_sizes,
                aggregation_function=func,
            )
            assert config.aggregation_function == func

    def test_mismatched_sample_sizes_raises(self, valid_aggregation_map):
        """Test that mismatched sample sizes raise error."""
        with pytest.raises(ValueError, match="sample_sizes length"):
            AggregatedSurveyConfig(
                aggregation_map=valid_aggregation_map,
                sample_sizes=(500, 450),  # Only 2, but aggregation_map has 3
            )

    def test_negative_design_effect_raises(self, valid_aggregation_map, valid_sample_sizes):
        """Test that negative design effect raises error."""
        with pytest.raises(ValueError, match="design_effect must be positive"):
            AggregatedSurveyConfig(
                aggregation_map=valid_aggregation_map,
                sample_sizes=valid_sample_sizes,
                design_effect=-1.0,
            )

    def test_zero_design_effect_raises(self, valid_aggregation_map, valid_sample_sizes):
        """Test that zero design effect raises error."""
        with pytest.raises(ValueError, match="design_effect must be positive"):
            AggregatedSurveyConfig(
                aggregation_map=valid_aggregation_map,
                sample_sizes=valid_sample_sizes,
                design_effect=0.0,
            )

    def test_immutability(self, valid_aggregation_map, valid_sample_sizes):
        """Test that config is frozen."""
        config = AggregatedSurveyConfig(
            aggregation_map=valid_aggregation_map,
            sample_sizes=valid_sample_sizes,
        )
        with pytest.raises(FrozenInstanceError):
            config.design_effect = 2.0


# =============================================================================
# MediatorConfig Tests
# =============================================================================


class TestMediatorConfig:
    """Tests for MediatorConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = MediatorConfig(name="awareness")
        assert config.name == "awareness"
        assert config.mediator_type == MediatorType.PARTIALLY_OBSERVED
        assert config.observation_noise_sigma == 0.1
        assert config.allow_direct_effect is True
        assert config.apply_adstock is True
        assert config.apply_saturation is True

    def test_fully_latent_type(self):
        """Test fully latent mediator."""
        config = MediatorConfig(
            name="latent_awareness",
            mediator_type=MediatorType.FULLY_LATENT,
        )
        assert config.mediator_type == MediatorType.FULLY_LATENT

    def test_fully_observed_type(self):
        """Test fully observed mediator."""
        config = MediatorConfig(
            name="traffic",
            mediator_type=MediatorType.FULLY_OBSERVED,
            observation_noise_sigma=0.05,
        )
        assert config.mediator_type == MediatorType.FULLY_OBSERVED
        assert config.observation_noise_sigma == 0.05

    def test_effect_priors(self):
        """Test effect prior configuration."""
        media_effect = EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE,
            sigma=1.5,
        )
        outcome_effect = EffectPriorConfig(
            constraint=EffectConstraint.NONE,
            sigma=0.5,
        )
        config = MediatorConfig(
            name="consideration",
            media_effect=media_effect,
            outcome_effect=outcome_effect,
        )
        assert config.media_effect.constraint == EffectConstraint.POSITIVE
        assert config.outcome_effect.sigma == 0.5

    def test_without_direct_effect(self):
        """Test disabling direct effect."""
        config = MediatorConfig(
            name="awareness",
            allow_direct_effect=False,
        )
        assert config.allow_direct_effect is False

    def test_transformation_configs(self):
        """Test transformation configuration."""
        adstock = AdstockConfig(l_max=12, prior_type="beta")
        saturation = SaturationConfig(type=SaturationType.HILL)

        config = MediatorConfig(
            name="awareness",
            adstock=adstock,
            saturation=saturation,
        )
        assert config.adstock.l_max == 12
        assert config.saturation.type == SaturationType.HILL

    def test_without_transformations(self):
        """Test disabling transformations."""
        config = MediatorConfig(
            name="direct_measure",
            apply_adstock=False,
            apply_saturation=False,
        )
        assert config.apply_adstock is False
        assert config.apply_saturation is False


# =============================================================================
# MediatorConfigExtended Tests
# =============================================================================


class TestMediatorConfigExtended:
    """Tests for MediatorConfigExtended with aggregated survey support."""

    @pytest.fixture
    def valid_survey_config(self):
        """Valid survey configuration."""
        return AggregatedSurveyConfig(
            aggregation_map={0: (0, 1, 2, 3), 1: (4, 5, 6, 7)},
            sample_sizes=(500, 500),
        )

    def test_basic_creation(self):
        """Test basic initialization."""
        config = MediatorConfigExtended(name="awareness")
        assert config.name == "awareness"
        assert config.observation_type == MediatorObservationType.PARTIALLY_OBSERVED
        assert config.media_effect_constraint == "positive"
        assert config.allow_direct_effect is True

    def test_aggregated_survey_type(self, valid_survey_config):
        """Test aggregated survey observation type."""
        config = MediatorConfigExtended(
            name="brand_awareness",
            observation_type=MediatorObservationType.AGGREGATED_SURVEY,
            aggregated_survey_config=valid_survey_config,
        )
        assert config.observation_type == MediatorObservationType.AGGREGATED_SURVEY
        assert config.aggregated_survey_config is not None

    def test_aggregated_survey_without_config_raises(self):
        """Test that aggregated survey without config raises error."""
        with pytest.raises(ValueError, match="aggregated_survey_config is required"):
            MediatorConfigExtended(
                name="awareness",
                observation_type=MediatorObservationType.AGGREGATED_SURVEY,
                aggregated_survey_config=None,
            )

    def test_effect_constraints(self):
        """Test different effect constraint options."""
        for constraint in ["none", "positive", "negative"]:
            config = MediatorConfigExtended(
                name="test",
                media_effect_constraint=constraint,
            )
            assert config.media_effect_constraint == constraint

    def test_transformation_settings(self):
        """Test adstock and saturation settings."""
        config = MediatorConfigExtended(
            name="awareness",
            apply_adstock=False,
            apply_saturation=True,
        )
        assert config.apply_adstock is False
        assert config.apply_saturation is True


# =============================================================================
# OutcomeConfig Tests
# =============================================================================


class TestOutcomeConfig:
    """Tests for OutcomeConfig dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        config = OutcomeConfig(name="sales", column="sales_col")
        assert config.name == "sales"
        assert config.column == "sales_col"
        assert config.intercept_prior_sigma == 2.0
        assert config.include_trend is True
        assert config.include_seasonality is True

    def test_custom_intercept_prior(self):
        """Test custom intercept prior."""
        config = OutcomeConfig(
            name="sales",
            column="sales_col",
            intercept_prior_sigma=3.0,
        )
        assert config.intercept_prior_sigma == 3.0

    def test_without_trend(self):
        """Test without trend component."""
        config = OutcomeConfig(
            name="sales",
            column="col",
            include_trend=False,
        )
        assert config.include_trend is False

    def test_without_seasonality(self):
        """Test without seasonality component."""
        config = OutcomeConfig(
            name="sales",
            column="col",
            include_seasonality=False,
        )
        assert config.include_seasonality is False

    def test_media_effect_prior(self):
        """Test media effect prior configuration."""
        effect = EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE,
            sigma=0.8,
        )
        config = OutcomeConfig(
            name="sales",
            column="col",
            media_effect=effect,
        )
        assert config.media_effect.constraint == EffectConstraint.POSITIVE
        assert config.media_effect.sigma == 0.8


# =============================================================================
# CrossEffectConfig Tests
# =============================================================================


class TestCrossEffectConfig:
    """Tests for CrossEffectConfig dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        config = CrossEffectConfig(
            source_outcome="product_a",
            target_outcome="product_b",
        )
        assert config.source_outcome == "product_a"
        assert config.target_outcome == "product_b"
        assert config.effect_type == CrossEffectType.CANNIBALIZATION
        assert config.prior_sigma == 0.3
        assert config.promotion_modulated is True
        assert config.lag == 0

    def test_halo_effect_type(self):
        """Test halo effect configuration."""
        config = CrossEffectConfig(
            source_outcome="premium",
            target_outcome="budget",
            effect_type=CrossEffectType.HALO,
        )
        assert config.effect_type == CrossEffectType.HALO

    def test_with_promotion_column(self):
        """Test with promotion column."""
        config = CrossEffectConfig(
            source_outcome="a",
            target_outcome="b",
            promotion_modulated=True,
            promotion_column="promo_a",
        )
        assert config.promotion_column == "promo_a"

    def test_not_promotion_modulated(self):
        """Test without promotion modulation."""
        config = CrossEffectConfig(
            source_outcome="a",
            target_outcome="b",
            promotion_modulated=False,
        )
        assert config.promotion_modulated is False

    def test_lagged_effect(self):
        """Test lagged cross-effect."""
        config = CrossEffectConfig(
            source_outcome="a",
            target_outcome="b",
            lag=1,
        )
        assert config.lag == 1

    def test_custom_prior_sigma(self):
        """Test custom prior sigma."""
        config = CrossEffectConfig(
            source_outcome="a",
            target_outcome="b",
            prior_sigma=0.5,
        )
        assert config.prior_sigma == 0.5


# =============================================================================
# NestedModelConfig Tests
# =============================================================================


class TestNestedModelConfig:
    """Tests for NestedModelConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = NestedModelConfig()
        assert config.mediators == ()
        assert config.media_to_mediator_map == {}
        assert config.share_adstock_across_mediators is True
        assert config.share_saturation_across_mediators is False

    def test_with_single_mediator(self):
        """Test with single mediator."""
        mediator = MediatorConfig(name="awareness")
        config = NestedModelConfig(mediators=(mediator,))
        assert len(config.mediators) == 1
        assert config.mediators[0].name == "awareness"

    def test_with_multiple_mediators(self):
        """Test with multiple mediators."""
        awareness = MediatorConfig(name="awareness")
        consideration = MediatorConfig(name="consideration")
        config = NestedModelConfig(
            mediators=(awareness, consideration),
        )
        assert len(config.mediators) == 2

    def test_media_to_mediator_mapping(self):
        """Test channel-to-mediator mapping."""
        mediator = MediatorConfig(name="awareness")
        config = NestedModelConfig(
            mediators=(mediator,),
            media_to_mediator_map={
                "awareness": ("tv", "digital"),
            },
        )
        assert "awareness" in config.media_to_mediator_map
        assert "tv" in config.media_to_mediator_map["awareness"]

    def test_sharing_options(self):
        """Test parameter sharing options."""
        config = NestedModelConfig(
            share_adstock_across_mediators=False,
            share_saturation_across_mediators=True,
        )
        assert config.share_adstock_across_mediators is False
        assert config.share_saturation_across_mediators is True


# =============================================================================
# MultivariateModelConfig Tests
# =============================================================================


class TestMultivariateModelConfig:
    """Tests for MultivariateModelConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = MultivariateModelConfig()
        assert config.outcomes == ()
        assert config.cross_effects == ()
        assert config.lkj_eta == 2.0
        assert config.share_media_adstock is True
        assert config.share_media_saturation is False
        assert config.share_trend is False
        assert config.share_seasonality is True

    def test_with_single_outcome(self):
        """Test with single outcome."""
        outcome = OutcomeConfig(name="sales", column="sales_col")
        config = MultivariateModelConfig(outcomes=(outcome,))
        assert len(config.outcomes) == 1

    def test_with_multiple_outcomes(self):
        """Test with multiple outcomes."""
        outcome_a = OutcomeConfig(name="product_a", column="col_a")
        outcome_b = OutcomeConfig(name="product_b", column="col_b")
        config = MultivariateModelConfig(
            outcomes=(outcome_a, outcome_b),
        )
        assert len(config.outcomes) == 2

    def test_with_cross_effects(self):
        """Test with cross-effects."""
        outcome_a = OutcomeConfig(name="product_a", column="col_a")
        outcome_b = OutcomeConfig(name="product_b", column="col_b")
        cross = CrossEffectConfig(
            source_outcome="product_b",
            target_outcome="product_a",
        )
        config = MultivariateModelConfig(
            outcomes=(outcome_a, outcome_b),
            cross_effects=(cross,),
        )
        assert len(config.cross_effects) == 1

    def test_lkj_eta_parameter(self):
        """Test LKJ correlation prior parameter."""
        config = MultivariateModelConfig(lkj_eta=4.0)
        assert config.lkj_eta == 4.0

    def test_sharing_options(self):
        """Test all sharing options."""
        config = MultivariateModelConfig(
            share_media_adstock=False,
            share_media_saturation=True,
            share_trend=True,
            share_seasonality=False,
        )
        assert config.share_media_adstock is False
        assert config.share_media_saturation is True
        assert config.share_trend is True
        assert config.share_seasonality is False


# =============================================================================
# CombinedModelConfig Tests
# =============================================================================


class TestCombinedModelConfig:
    """Tests for CombinedModelConfig dataclass."""

    @pytest.fixture
    def nested_config(self):
        """Create sample nested config."""
        mediator = MediatorConfig(name="awareness")
        return NestedModelConfig(mediators=(mediator,))

    @pytest.fixture
    def multivariate_config(self):
        """Create sample multivariate config."""
        outcome_a = OutcomeConfig(name="product_a", column="col_a")
        outcome_b = OutcomeConfig(name="product_b", column="col_b")
        return MultivariateModelConfig(
            outcomes=(outcome_a, outcome_b),
        )

    def test_basic_creation(self, nested_config, multivariate_config):
        """Test basic initialization."""
        config = CombinedModelConfig(
            nested=nested_config,
            multivariate=multivariate_config,
        )
        assert config.nested is nested_config
        assert config.multivariate is multivariate_config
        assert config.mediator_to_outcome_map == {}

    def test_with_mediator_to_outcome_map(self, nested_config, multivariate_config):
        """Test with mediator-to-outcome mapping."""
        config = CombinedModelConfig(
            nested=nested_config,
            multivariate=multivariate_config,
            mediator_to_outcome_map={
                "awareness": ("product_a", "product_b"),
            },
        )
        assert "awareness" in config.mediator_to_outcome_map
        assert "product_a" in config.mediator_to_outcome_map["awareness"]

    def test_complex_mapping(self, nested_config, multivariate_config):
        """Test complex mediator mapping."""
        awareness = MediatorConfig(name="awareness")
        consideration = MediatorConfig(name="consideration")
        nested = NestedModelConfig(mediators=(awareness, consideration))

        config = CombinedModelConfig(
            nested=nested,
            multivariate=multivariate_config,
            mediator_to_outcome_map={
                "awareness": ("product_a",),
                "consideration": ("product_a", "product_b"),
            },
        )
        assert len(config.mediator_to_outcome_map) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions in config module."""

    def test_sparse_selection_config_basic(self):
        """Test sparse_selection_config with defaults."""
        config = sparse_selection_config()
        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 3
        assert config.horseshoe.slab_scale == 2.0

    def test_sparse_selection_config_custom_expected(self):
        """Test sparse_selection_config with custom expected relevant."""
        config = sparse_selection_config(expected_relevant=7)
        assert config.horseshoe.expected_nonzero == 7

    def test_sparse_selection_config_with_confounders(self):
        """Test sparse_selection_config with confounders."""
        config = sparse_selection_config(
            expected_relevant=3,
            confounders=("price", "distribution"),
        )
        assert "price" in config.exclude_variables
        assert "distribution" in config.exclude_variables

    def test_dense_selection_config_basic(self):
        """Test dense_selection_config with defaults."""
        config = dense_selection_config()
        assert config.method == VariableSelectionMethod.BAYESIAN_LASSO
        assert config.lasso.regularization == 1.0

    def test_dense_selection_config_custom_regularization(self):
        """Test dense_selection_config with custom regularization."""
        config = dense_selection_config(regularization=2.5)
        assert config.lasso.regularization == 2.5

    def test_dense_selection_config_with_confounders(self):
        """Test dense_selection_config with confounders."""
        config = dense_selection_config(
            regularization=1.5,
            confounders=("confounder1", "confounder2"),
        )
        assert "confounder1" in config.exclude_variables

    def test_inclusion_prob_selection_config_basic(self):
        """Test inclusion_prob_selection_config with defaults."""
        config = inclusion_prob_selection_config()
        assert config.method == VariableSelectionMethod.SPIKE_SLAB
        assert config.spike_slab.prior_inclusion_prob == 0.5
        assert config.spike_slab.temperature == 0.1

    def test_inclusion_prob_selection_config_custom(self):
        """Test inclusion_prob_selection_config with custom probability."""
        config = inclusion_prob_selection_config(prior_inclusion=0.3)
        assert config.spike_slab.prior_inclusion_prob == 0.3

    def test_inclusion_prob_selection_config_with_confounders(self):
        """Test inclusion_prob_selection_config with confounders."""
        config = inclusion_prob_selection_config(
            prior_inclusion=0.4,
            confounders=("conf1",),
        )
        assert "conf1" in config.exclude_variables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

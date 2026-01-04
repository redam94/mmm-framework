"""
Test suite for mmm_framework.mmm_extensions.models module.

Tests cover:
- Result containers (MediationEffects, CrossEffectSummary, ModelResults)
- BaseExtendedMMM class
- NestedMMM class
- MultivariateMMM class  
- CombinedMMM class
- Integration tests with model fitting (slow tests)

Note: Tests requiring MCMC sampling are marked with @pytest.mark.slow
and can be skipped with: pytest -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Import models and result containers
from mmm_framework.mmm_extensions.models import (
    # Result containers
    MediationEffects,
    CrossEffectSummary,
    ModelResults,
    # Model classes
    BaseExtendedMMM,
    NestedMMM,
    MultivariateMMM,
    CombinedMMM,
)

# Import config classes
from mmm_framework.mmm_extensions.config import (
    MediatorType,
    CrossEffectType,
    EffectConstraint,
    SaturationType,
    AdstockConfig,
    SaturationConfig,
    EffectPriorConfig,
    MediatorConfig,
    OutcomeConfig,
    CrossEffectConfig,
    NestedModelConfig,
    MultivariateModelConfig,
    CombinedModelConfig,
)

# Import builders
from mmm_framework.mmm_extensions.builders import (
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_media_data():
    """Generate sample media data."""
    np.random.seed(42)
    n_obs = 52
    n_channels = 3
    X_media = np.abs(np.random.randn(n_obs, n_channels) * 50 + 100)
    return X_media


@pytest.fixture
def sample_outcome():
    """Generate sample outcome variable."""
    np.random.seed(42)
    n_obs = 52
    y = 1000 + np.random.randn(n_obs) * 100
    return y


@pytest.fixture
def sample_outcome_data():
    """Generate sample multi-outcome data."""
    np.random.seed(42)
    n_obs = 52
    return {
        "sales_product_a": 1000 + np.random.randn(n_obs) * 100,
        "sales_product_b": 800 + np.random.randn(n_obs) * 80,
    }


@pytest.fixture
def sample_mediator_data():
    """Generate sample mediator data with partial observations."""
    np.random.seed(42)
    n_obs = 52
    # Only 12 monthly observations
    awareness = np.full(n_obs, np.nan)
    monthly_idx = np.arange(0, n_obs, 4)  # Every 4th week
    awareness[monthly_idx] = 50 + np.random.randn(len(monthly_idx)) * 10
    return {"brand_awareness": awareness}


@pytest.fixture
def channel_names():
    """Standard channel names."""
    return ["tv", "digital", "social"]


@pytest.fixture
def nested_config():
    """Create a NestedModelConfig for testing."""
    mediator = (
        MediatorConfigBuilder("brand_awareness")
        .partially_observed(observation_noise=0.15)
        .with_positive_media_effect(sigma=1.0)
        .build()
    )
    
    config = (
        NestedModelConfigBuilder()
        .add_mediator(mediator)
        .map_channels_to_mediator("brand_awareness", ["tv", "digital"])
        .build()
    )
    return config


@pytest.fixture
def multivariate_config():
    """Create a MultivariateModelConfig for testing."""
    outcome_a = (
        OutcomeConfigBuilder("sales_product_a", column="sales_a")
        .with_positive_media_effects(sigma=0.5)
        .build()
    )
    
    outcome_b = (
        OutcomeConfigBuilder("sales_product_b", column="sales_b")
        .with_positive_media_effects(sigma=0.5)
        .build()
    )
    
    cross_effect = cannibalization_effect(
        source="sales_product_b",
        target="sales_product_a",
    )
    
    config = (
        MultivariateModelConfigBuilder()
        .add_outcome(outcome_a)
        .add_outcome(outcome_b)
        .add_cross_effect(cross_effect)
        .build()
    )
    return config


@pytest.fixture
def combined_config(nested_config, multivariate_config):
    """Create a CombinedModelConfig for testing."""
    return CombinedModelConfig(
        nested=nested_config,
        multivariate=multivariate_config,
        mediator_to_outcome_map={"brand_awareness": ("sales_product_a",)},
    )


# =============================================================================
# MediationEffects Tests
# =============================================================================

class TestMediationEffects:
    """Tests for MediationEffects dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        effects = MediationEffects(
            channel="tv",
            direct_effect=0.5,
            direct_effect_sd=0.1,
            indirect_effects={"awareness": 0.3},
            total_indirect=0.3,
            total_effect=0.8,
            proportion_mediated=0.375,
        )
        
        assert effects.channel == "tv"
        assert effects.direct_effect == 0.5
        assert effects.total_effect == 0.8
        assert effects.proportion_mediated == pytest.approx(0.375)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        effects = MediationEffects(
            channel="digital",
            direct_effect=0.4,
            direct_effect_sd=0.08,
            indirect_effects={"awareness": 0.2, "consideration": 0.1},
            total_indirect=0.3,
            total_effect=0.7,
            proportion_mediated=0.43,
        )
        
        d = effects.to_dict()
        
        assert d["channel"] == "digital"
        assert d["direct_effect"] == 0.4
        assert d["indirect_via_awareness"] == 0.2
        assert d["indirect_via_consideration"] == 0.1
        assert "total_indirect" in d
        assert "proportion_mediated" in d

    def test_multiple_mediators(self):
        """Test with multiple indirect effects."""
        effects = MediationEffects(
            channel="social",
            direct_effect=0.2,
            direct_effect_sd=0.05,
            indirect_effects={
                "awareness": 0.15,
                "engagement": 0.10,
                "consideration": 0.05,
            },
            total_indirect=0.30,
            total_effect=0.50,
            proportion_mediated=0.60,
        )
        
        assert len(effects.indirect_effects) == 3
        assert effects.total_indirect == 0.30
        
        d = effects.to_dict()
        assert "indirect_via_awareness" in d
        assert "indirect_via_engagement" in d
        assert "indirect_via_consideration" in d

    def test_no_indirect_effects(self):
        """Test when there are no indirect effects."""
        effects = MediationEffects(
            channel="radio",
            direct_effect=0.5,
            direct_effect_sd=0.1,
            indirect_effects={},
            total_indirect=0.0,
            total_effect=0.5,
            proportion_mediated=0.0,
        )
        
        d = effects.to_dict()
        assert d["total_indirect"] == 0.0
        assert d["proportion_mediated"] == 0.0

    def test_nan_proportion_mediated(self):
        """Test when total effect is zero (NaN proportion)."""
        effects = MediationEffects(
            channel="print",
            direct_effect=0.0,
            direct_effect_sd=0.01,
            indirect_effects={},
            total_indirect=0.0,
            total_effect=0.0,
            proportion_mediated=float('nan'),
        )
        
        assert np.isnan(effects.proportion_mediated)


# =============================================================================
# CrossEffectSummary Tests
# =============================================================================

class TestCrossEffectSummary:
    """Tests for CrossEffectSummary dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        summary = CrossEffectSummary(
            source="product_b",
            target="product_a",
            effect_type="cannibalization",
            mean=-0.15,
            sd=0.05,
            hdi_low=-0.24,
            hdi_high=-0.06,
        )
        
        assert summary.source == "product_b"
        assert summary.target == "product_a"
        assert summary.effect_type == "cannibalization"
        assert summary.mean == -0.15
        assert summary.hdi_low < summary.mean < summary.hdi_high

    def test_halo_effect(self):
        """Test positive cross-effect (halo)."""
        summary = CrossEffectSummary(
            source="premium",
            target="budget",
            effect_type="halo",
            mean=0.10,
            sd=0.03,
            hdi_low=0.04,
            hdi_high=0.16,
        )
        
        assert summary.mean > 0  # Positive halo effect


# =============================================================================
# ModelResults Tests
# =============================================================================

class TestModelResults:
    """Tests for ModelResults dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        mock_trace = MagicMock()
        mock_model = MagicMock()
        mock_config = MagicMock()
        
        results = ModelResults(
            trace=mock_trace,
            model=mock_model,
            config=mock_config,
        )
        
        assert results.trace == mock_trace
        assert results.model == mock_model
        assert results.config == mock_config

    def test_summary_calls_arviz(self):
        """Test that summary delegates to ArviZ."""
        mock_trace = MagicMock()
        
        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )
        
        with patch('arviz.summary') as mock_summary:
            mock_summary.return_value = pd.DataFrame()
            _ = results.summary(var_names=["alpha", "beta"])
            mock_summary.assert_called_once_with(mock_trace, var_names=["alpha", "beta"])


# =============================================================================
# BaseExtendedMMM Tests
# =============================================================================

class TestBaseExtendedMMM:
    """Tests for BaseExtendedMMM base class."""

    def test_init(self, sample_media_data, sample_outcome, channel_names):
        """Test basic initialization."""
        # Note: BaseExtendedMMM is abstract-ish, but we can test init
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
        )
        
        assert model.n_obs == 52
        assert model.n_channels == 3
        assert model.channel_names == channel_names

    def test_init_with_index(self, sample_media_data, sample_outcome, channel_names):
        """Test initialization with custom index."""
        index = pd.date_range("2020-01-06", periods=52, freq="W-MON")
        
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            index=index,
        )
        
        assert len(model.index) == 52
        # Check type without ambiguous truth value comparison
        assert type(model.index).__name__ == "DatetimeIndex"

    def test_default_index(self, sample_media_data, sample_outcome, channel_names):
        """Test that default index is RangeIndex."""
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
        )
        
        assert isinstance(model.index, pd.RangeIndex)
        assert len(model.index) == 52

    def test_build_coords(self, sample_media_data, sample_outcome, channel_names):
        """Test coordinate building."""
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
        )
        
        coords = model._build_coords()
        
        assert "obs" in coords
        assert "channel" in coords
        assert coords["channel"] == channel_names
        assert len(coords["obs"]) == 52

    def test_build_model_not_implemented(self, sample_media_data, sample_outcome, channel_names):
        """Test that _build_model raises NotImplementedError."""
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
        )
        
        with pytest.raises(NotImplementedError):
            model._build_model()

    def test_check_fitted_before_fit(self, sample_media_data, sample_outcome, channel_names):
        """Test _check_fitted raises error before fitting."""
        model = BaseExtendedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
        )
        
        with pytest.raises(ValueError, match="not fitted"):
            model._check_fitted()


# =============================================================================
# NestedMMM Tests - Initialization
# =============================================================================

class TestNestedMMMInit:
    """Tests for NestedMMM initialization."""

    def test_basic_init(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test basic initialization."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        assert model.n_obs == 52
        assert model.n_channels == 3
        assert model.n_mediators == 1
        assert model.mediator_names == ["brand_awareness"]

    def test_init_with_mediator_data(
        self, sample_media_data, sample_outcome, channel_names, 
        nested_config, sample_mediator_data
    ):
        """Test initialization with mediator observations."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
            mediator_data=sample_mediator_data,
        )
        
        assert "brand_awareness" in model.mediator_data
        assert len(model.mediator_data["brand_awareness"]) == 52

    def test_init_with_mediator_masks(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test initialization with custom observation masks."""
        n_obs = 52
        mask = np.zeros(n_obs, dtype=bool)
        mask[::4] = True  # Every 4th observation
        
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
            mediator_masks={"brand_awareness": mask},
        )
        
        assert "brand_awareness" in model.mediator_masks
        assert model.mediator_masks["brand_awareness"].sum() == 13

    def test_build_coords_includes_mediator(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test that coords include mediator dimension."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        coords = model._build_coords()
        
        assert "mediator" in coords
        assert coords["mediator"] == ["brand_awareness"]

    def test_get_affecting_channels(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test getting channels that affect a mediator."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        # Config maps tv and digital to brand_awareness
        affecting = model._get_affecting_channels("brand_awareness")
        assert "tv" in affecting
        assert "digital" in affecting

    def test_get_affecting_channels_default(
        self, sample_media_data, sample_outcome, channel_names
    ):
        """Test default: all channels affect unmapped mediator."""
        # Config with no explicit mapping
        mediator = MediatorConfigBuilder("awareness").build()
        config = NestedModelConfigBuilder().add_mediator(mediator).build()
        
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=config,
        )
        
        # Should return all channels
        affecting = model._get_affecting_channels("awareness")
        assert affecting == channel_names


# =============================================================================
# NestedMMM Tests - Model Building
# =============================================================================

class TestNestedMMMModelBuilding:
    """Tests for NestedMMM model construction."""

    def test_model_property_builds_model(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test that accessing model property builds the model."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        pymc_model = model.model
        
        assert pymc_model is not None
        assert len(pymc_model.free_RVs) > 0

    def test_model_has_mediator_parameters(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test that model includes mediator-related parameters."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        var_names = [v.name for v in model.model.free_RVs]
        
        # Should have mediator-related variables
        # Exact names depend on implementation
        assert any("alpha" in name.lower() for name in var_names)


# =============================================================================
# MultivariateMMM Tests - Initialization
# =============================================================================

class TestMultivariateMMMInit:
    """Tests for MultivariateMMM initialization."""

    def test_basic_init(
        self, sample_media_data, sample_outcome_data, channel_names, multivariate_config
    ):
        """Test basic initialization."""
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=multivariate_config,
        )
        
        assert model.n_obs == 52
        assert model.n_channels == 3
        assert model.n_outcomes == 2
        assert model.outcome_names == ["sales_product_a", "sales_product_b"]

    def test_init_with_promotion_data(
        self, sample_media_data, sample_outcome_data, channel_names, multivariate_config
    ):
        """Test initialization with promotion data."""
        promo_data = {
            "product_b_promo": np.random.binomial(1, 0.3, 52).astype(float)
        }
        
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=multivariate_config,
            promotion_data=promo_data,
        )
        
        assert "product_b_promo" in model.promotion_data

    def test_build_coords_includes_outcome(
        self, sample_media_data, sample_outcome_data, channel_names, multivariate_config
    ):
        """Test that coords include outcome dimension."""
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=multivariate_config,
        )
        
        coords = model._build_coords()
        
        assert "outcome" in coords
        assert coords["outcome"] == ["sales_product_a", "sales_product_b"]


# =============================================================================
# MultivariateMMM Tests - Model Building
# =============================================================================

class TestMultivariateMMMModelBuilding:
    """Tests for MultivariateMMM model construction."""

    def test_model_property_builds_model(
        self, sample_media_data, sample_outcome_data, channel_names, multivariate_config
    ):
        """Test that accessing model property builds the model."""
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=multivariate_config,
        )
        
        pymc_model = model.model
        
        assert pymc_model is not None
        assert len(pymc_model.free_RVs) > 0


# =============================================================================
# CombinedMMM Tests - Initialization
# =============================================================================

class TestCombinedMMMInit:
    """Tests for CombinedMMM initialization."""

    def test_basic_init(
        self, sample_media_data, sample_outcome_data, channel_names, combined_config
    ):
        """Test basic initialization."""
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
        )
        
        assert model.n_obs == 52
        assert model.n_channels == 3
        assert model.n_mediators == 1
        assert model.n_outcomes == 2

    def test_init_with_all_data(
        self, sample_media_data, sample_outcome_data, channel_names,
        combined_config, sample_mediator_data
    ):
        """Test initialization with mediator and promotion data."""
        promo_data = {"promo": np.random.binomial(1, 0.2, 52).astype(float)}
        
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
            mediator_data=sample_mediator_data,
            promotion_data=promo_data,
        )
        
        assert "brand_awareness" in model.mediator_data
        assert "promo" in model.promotion_data

    def test_build_coords_includes_both(
        self, sample_media_data, sample_outcome_data, channel_names, combined_config
    ):
        """Test that coords include both mediator and outcome dimensions."""
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
        )
        
        coords = model._build_coords()
        
        assert "mediator" in coords
        assert "outcome" in coords

    def test_get_affecting_channels(
        self, sample_media_data, sample_outcome_data, channel_names, combined_config
    ):
        """Test _get_affecting_channels method."""
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
        )
        
        affecting = model._get_affecting_channels("brand_awareness")
        # Should come from nested config mapping
        assert isinstance(affecting, list)

    def test_get_affected_outcomes(
        self, sample_media_data, sample_outcome_data, channel_names, combined_config
    ):
        """Test _get_affected_outcomes method."""
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
        )
        
        # brand_awareness mapped to sales_product_a
        affected = model._get_affected_outcomes("brand_awareness")
        assert "sales_product_a" in affected


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions that create configs."""

    def test_awareness_mediator(self):
        """Test awareness_mediator factory."""
        config = awareness_mediator(
            name="brand_awareness",
            observation_noise=0.15,
        )
        
        assert config.name == "brand_awareness"
        assert config.mediator_type == MediatorType.PARTIALLY_OBSERVED

    def test_foot_traffic_mediator(self):
        """Test foot_traffic_mediator factory."""
        config = foot_traffic_mediator(
            name="store_visits",
            observation_noise=0.10,
        )
        
        assert config.name == "store_visits"
        # foot_traffic_mediator uses fully_observed(), not partially_observed
        assert config.mediator_type == MediatorType.FULLY_OBSERVED

    def test_cannibalization_effect(self):
        """Test cannibalization_effect factory."""
        config = cannibalization_effect(
            source="multipack",
            target="single_pack",
        )
        
        assert config.source_outcome == "multipack"
        assert config.target_outcome == "single_pack"
        assert config.effect_type == CrossEffectType.CANNIBALIZATION
        # Cannibalization implies negative effect (enforced in model, not config)

    def test_halo_effect(self):
        """Test halo_effect factory."""
        config = halo_effect(
            source="premium",
            target="budget",
        )
        
        assert config.source_outcome == "premium"
        assert config.target_outcome == "budget"
        assert config.effect_type == CrossEffectType.HALO
        # Halo implies positive effect (enforced in model, not config)


# =============================================================================
# Builder Tests
# =============================================================================

class TestNestedModelConfigBuilder:
    """Tests for NestedModelConfigBuilder."""

    def test_basic_build(self):
        """Test basic config building."""
        mediator = MediatorConfigBuilder("awareness").build()
        
        config = (
            NestedModelConfigBuilder()
            .add_mediator(mediator)
            .build()
        )
        
        assert len(config.mediators) == 1
        assert config.mediators[0].name == "awareness"

    def test_multiple_mediators(self):
        """Test adding multiple mediators."""
        awareness = MediatorConfigBuilder("awareness").build()
        consideration = MediatorConfigBuilder("consideration").build()
        
        config = (
            NestedModelConfigBuilder()
            .add_mediator(awareness)
            .add_mediator(consideration)
            .build()
        )
        
        assert len(config.mediators) == 2

    def test_channel_mapping(self):
        """Test mapping channels to mediators."""
        mediator = MediatorConfigBuilder("awareness").build()
        
        config = (
            NestedModelConfigBuilder()
            .add_mediator(mediator)
            .map_channels_to_mediator("awareness", ["tv", "digital"])
            .build()
        )
        
        assert "awareness" in config.media_to_mediator_map
        assert "tv" in config.media_to_mediator_map["awareness"]

    def test_share_adstock(self):
        """Test adstock sharing option."""
        mediator = MediatorConfigBuilder("awareness").build()
        
        config = (
            NestedModelConfigBuilder()
            .add_mediator(mediator)
            .share_adstock(True)
            .build()
        )
        
        assert config.share_adstock_across_mediators is True


class TestMultivariateModelConfigBuilder:
    """Tests for MultivariateModelConfigBuilder."""

    def test_basic_build(self):
        """Test basic config building."""
        outcome = OutcomeConfigBuilder("sales", column="sales_col").build()
        
        config = (
            MultivariateModelConfigBuilder()
            .add_outcome(outcome)
            .build()
        )
        
        assert len(config.outcomes) == 1
        assert config.outcomes[0].name == "sales"

    def test_multiple_outcomes(self):
        """Test adding multiple outcomes."""
        outcome_a = OutcomeConfigBuilder("product_a", column="col_a").build()
        outcome_b = OutcomeConfigBuilder("product_b", column="col_b").build()
        
        config = (
            MultivariateModelConfigBuilder()
            .add_outcome(outcome_a)
            .add_outcome(outcome_b)
            .build()
        )
        
        assert len(config.outcomes) == 2

    def test_cross_effects(self):
        """Test adding cross-effects."""
        outcome_a = OutcomeConfigBuilder("product_a", column="col_a").build()
        outcome_b = OutcomeConfigBuilder("product_b", column="col_b").build()
        cross = cannibalization_effect("product_b", "product_a")
        
        config = (
            MultivariateModelConfigBuilder()
            .add_outcome(outcome_a)
            .add_outcome(outcome_b)
            .add_cross_effect(cross)
            .build()
        )
        
        assert len(config.cross_effects) == 1

    def test_lkj_eta(self):
        """Test LKJ correlation prior setting."""
        outcome = OutcomeConfigBuilder("sales", column="col").build()
        
        config = (
            MultivariateModelConfigBuilder()
            .add_outcome(outcome)
            .with_lkj_eta(3.0)
            .build()
        )
        
        assert config.lkj_eta == 3.0


class TestMediatorConfigBuilder:
    """Tests for MediatorConfigBuilder."""

    def test_fully_observed(self):
        """Test creating fully observed mediator."""
        config = (
            MediatorConfigBuilder("store_sales")
            .fully_observed()
            .build()
        )
        
        assert config.mediator_type == MediatorType.FULLY_OBSERVED

    def test_partially_observed(self):
        """Test creating partially observed mediator."""
        config = (
            MediatorConfigBuilder("awareness")
            .partially_observed(observation_noise=0.12)
            .build()
        )
        
        assert config.mediator_type == MediatorType.PARTIALLY_OBSERVED
        assert config.observation_noise_sigma == 0.12

    def test_positive_media_effect(self):
        """Test positive media effect constraint."""
        config = (
            MediatorConfigBuilder("awareness")
            .with_positive_media_effect(sigma=1.5)
            .build()
        )
        
        assert config.media_effect.constraint == EffectConstraint.POSITIVE
        assert config.media_effect.sigma == 1.5

    def test_adstock_config(self):
        """Test adstock configuration."""
        config = (
            MediatorConfigBuilder("awareness")
            .with_slow_adstock(l_max=12)
            .build()
        )
        
        assert config.adstock.l_max == 12

    def test_allow_direct_effect(self):
        """Test allowing direct media effects."""
        config = (
            MediatorConfigBuilder("awareness")
            .with_direct_effect(sigma=0.3)
            .build()
        )
        
        assert config.allow_direct_effect is True


class TestOutcomeConfigBuilder:
    """Tests for OutcomeConfigBuilder."""

    def test_basic_build(self):
        """Test basic outcome config."""
        config = (
            OutcomeConfigBuilder("sales", column="sales_col")
            .build()
        )
        
        assert config.name == "sales"
        assert config.column == "sales_col"

    def test_positive_media_effects(self):
        """Test positive media effect constraint."""
        config = (
            OutcomeConfigBuilder("sales", column="col")
            .with_positive_media_effects(sigma=0.8)
            .build()
        )
        
        assert config.media_effect.constraint == EffectConstraint.POSITIVE
        assert config.media_effect.sigma == 0.8

    def test_with_trend(self):
        """Test trend inclusion."""
        config = (
            OutcomeConfigBuilder("sales", column="col")
            .with_trend()
            .build()
        )
        
        assert config.include_trend is True

    def test_with_seasonality(self):
        """Test seasonality inclusion."""
        config = (
            OutcomeConfigBuilder("sales", column="col")
            .with_seasonality()
            .build()
        )
        
        assert config.include_seasonality is True


# =============================================================================
# Slow Tests - Model Fitting
# =============================================================================

@pytest.mark.slow
class TestNestedMMMFitting:
    """Slow tests for NestedMMM fitting."""

    def test_fit_returns_results(
        self, sample_media_data, sample_outcome, channel_names, nested_config
    ):
        """Test that fit returns ModelResults."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
        )
        
        results = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )
        
        assert isinstance(results, ModelResults)
        assert results.trace is not None

    def test_fit_with_mediator_data(
        self, sample_media_data, sample_outcome, channel_names,
        nested_config, sample_mediator_data
    ):
        """Test fitting with partial mediator observations."""
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=nested_config,
            mediator_data=sample_mediator_data,
        )
        
        results = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )
        
        assert results.trace is not None


@pytest.mark.slow
class TestMultivariateMMMFitting:
    """Slow tests for MultivariateMMM fitting."""

    def test_fit_returns_results(
        self, sample_media_data, sample_outcome_data, channel_names, multivariate_config
    ):
        """Test that fit returns ModelResults."""
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=multivariate_config,
        )
        
        results = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )
        
        assert isinstance(results, ModelResults)
        assert results.trace is not None


@pytest.mark.slow
class TestCombinedMMMFitting:
    """Slow tests for CombinedMMM fitting."""

    def test_fit_returns_results(
        self, sample_media_data, sample_outcome_data, channel_names, combined_config
    ):
        """Test that fit returns ModelResults."""
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=combined_config,
        )
        
        results = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )
        
        assert isinstance(results, ModelResults)


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfigIntegration:
    """Integration tests for config + model workflow."""

    def test_nested_config_to_model(self, sample_media_data, sample_outcome, channel_names):
        """Test full workflow from config to model."""
        # Build config
        mediator = awareness_mediator("brand_awareness", observation_noise=0.15)
        
        config = (
            NestedModelConfigBuilder()
            .add_mediator(mediator)
            .map_channels_to_mediator("brand_awareness", ["tv", "digital"])
            .share_adstock(True)
            .build()
        )
        
        # Create model
        model = NestedMMM(
            X_media=sample_media_data,
            y=sample_outcome,
            channel_names=channel_names,
            config=config,
        )
        
        # Verify structure
        assert model.n_mediators == 1
        assert model.mediator_names == ["brand_awareness"]
        
        # Build PyMC model
        pymc_model = model.model
        assert pymc_model is not None

    def test_multivariate_config_to_model(
        self, sample_media_data, sample_outcome_data, channel_names
    ):
        """Test full workflow for multivariate model."""
        # Build config with factory functions
        cross = cannibalization_effect(
            source="sales_product_b",
            target="sales_product_a",
        )
        
        config = (
            MultivariateModelConfigBuilder()
            .add_outcome(
                OutcomeConfigBuilder("sales_product_a", column="col_a")
                .with_positive_media_effects()
                .build()
            )
            .add_outcome(
                OutcomeConfigBuilder("sales_product_b", column="col_b")
                .with_positive_media_effects()
                .build()
            )
            .add_cross_effect(cross)
            .with_lkj_eta(2.0)
            .build()
        )
        
        # Create model
        model = MultivariateMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=config,
        )
        
        # Verify structure
        assert model.n_outcomes == 2
        assert len(config.cross_effects) == 1

    def test_combined_config_to_model(
        self, sample_media_data, sample_outcome_data, channel_names
    ):
        """Test full workflow for combined model."""
        # Nested part
        mediator = awareness_mediator("awareness")
        nested_config = (
            NestedModelConfigBuilder()
            .add_mediator(mediator)
            .build()
        )
        
        # Multivariate part
        multi_config = (
            MultivariateModelConfigBuilder()
            .add_outcome(
                OutcomeConfigBuilder("sales_product_a", column="col_a").build()
            )
            .add_outcome(
                OutcomeConfigBuilder("sales_product_b", column="col_b").build()
            )
            .build()
        )
        
        # Combined config
        config = CombinedModelConfig(
            nested=nested_config,
            multivariate=multi_config,
            mediator_to_outcome_map={"awareness": ("sales_product_a", "sales_product_b")},
        )
        
        # Create model
        model = CombinedMMM(
            X_media=sample_media_data,
            outcome_data=sample_outcome_data,
            channel_names=channel_names,
            config=config,
        )
        
        # Verify structure
        assert model.n_mediators == 1
        assert model.n_outcomes == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
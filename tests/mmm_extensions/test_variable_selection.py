"""
Test suite for variable selection components in mmm_framework.mmm_extensions.

Tests cover:
- Configuration classes (HorseshoeConfig, SpikeSlabConfig, LassoConfig, VariableSelectionConfig)
- Builder classes for fluent API (HorseshoeConfigBuilder, etc.)
- Factory functions (sparse_controls, dense_controls, etc.)
- Prior creation functions (horseshoe, spike-slab, LASSO)
- Control effect builder with selection
- Diagnostic utilities

Note: Tests requiring MCMC sampling are marked with @pytest.mark.slow
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import FrozenInstanceError

# Import config classes
from mmm_framework.mmm_extensions.config import (
    VariableSelectionMethod,
    HorseshoeConfig,
    SpikeSlabConfig,
    LassoConfig,
    VariableSelectionConfig,
    sparse_selection_config,
    dense_selection_config,
    inclusion_prob_selection_config,
)

# Import builders
from mmm_framework.mmm_extensions.builders import (
    HorseshoeConfigBuilder,
    SpikeSlabConfigBuilder,
    LassoConfigBuilder,
    VariableSelectionConfigBuilder,
    sparse_controls,
    selection_with_inclusion_probs,
    dense_controls,
)

# PyMC imports (may fail if not installed)
try:
    import pymc as pm
    import pytensor.tensor as pt
    from mmm_framework.mmm_extensions.components import (
        VariableSelectionResult,
        ControlEffectResult,
        create_regularized_horseshoe_prior,
        create_finnish_horseshoe_prior,
        create_spike_slab_prior,
        create_bayesian_lasso_prior,
        create_variable_selection_prior,
        build_control_effects_with_selection,
    )

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_control_names():
    """Sample control variable names."""
    return ["weather", "gas_price", "holiday", "distribution", "price"]


@pytest.fixture
def sample_confounders():
    """Confounder variables that should be excluded from selection."""
    return ("distribution", "price")


@pytest.fixture
def sample_precision_controls():
    """Precision controls that can be subject to selection."""
    return ["weather", "gas_price", "holiday"]


# =============================================================================
# HorseshoeConfig Tests
# =============================================================================


class TestHorseshoeConfig:
    """Tests for HorseshoeConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        config = HorseshoeConfig()

        assert config.expected_nonzero == 3
        assert config.slab_scale == 2.0
        assert config.slab_df == 4.0
        assert config.local_df == 5.0
        assert config.global_df == 1.0

    def test_custom_values(self):
        """Test creation with custom values."""
        config = HorseshoeConfig(
            expected_nonzero=5,
            slab_scale=3.0,
            slab_df=6.0,
            local_df=1.0,
            global_df=2.0,
        )

        assert config.expected_nonzero == 5
        assert config.slab_scale == 3.0
        assert config.slab_df == 6.0
        assert config.local_df == 1.0
        assert config.global_df == 2.0

    def test_immutability(self):
        """Test that config is frozen (immutable)."""
        config = HorseshoeConfig()

        with pytest.raises(FrozenInstanceError):
            config.expected_nonzero = 10


# =============================================================================
# SpikeSlabConfig Tests
# =============================================================================


class TestSpikeSlabConfig:
    """Tests for SpikeSlabConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        config = SpikeSlabConfig()

        assert config.prior_inclusion_prob == 0.5
        assert config.spike_scale == 0.01
        assert config.slab_scale == 1.0
        assert config.use_continuous_relaxation is True
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test creation with custom values."""
        config = SpikeSlabConfig(
            prior_inclusion_prob=0.3,
            spike_scale=0.005,
            slab_scale=2.0,
            use_continuous_relaxation=False,
            temperature=0.05,
        )

        assert config.prior_inclusion_prob == 0.3
        assert config.spike_scale == 0.005
        assert config.slab_scale == 2.0
        assert config.use_continuous_relaxation is False
        assert config.temperature == 0.05

    def test_immutability(self):
        """Test that config is frozen (immutable)."""
        config = SpikeSlabConfig()

        with pytest.raises(FrozenInstanceError):
            config.prior_inclusion_prob = 0.8


# =============================================================================
# LassoConfig Tests
# =============================================================================


class TestLassoConfig:
    """Tests for LassoConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        config = LassoConfig()

        assert config.regularization == 1.0
        assert config.adaptive is False

    def test_custom_values(self):
        """Test creation with custom values."""
        config = LassoConfig(
            regularization=2.5,
            adaptive=True,
        )

        assert config.regularization == 2.5
        assert config.adaptive is True

    def test_immutability(self):
        """Test that config is frozen (immutable)."""
        config = LassoConfig()

        with pytest.raises(FrozenInstanceError):
            config.regularization = 5.0


# =============================================================================
# VariableSelectionConfig Tests
# =============================================================================


class TestVariableSelectionConfig:
    """Tests for VariableSelectionConfig dataclass."""

    def test_default_values(self):
        """Test that default values are correct."""
        config = VariableSelectionConfig()

        assert config.method == VariableSelectionMethod.NONE
        assert isinstance(config.horseshoe, HorseshoeConfig)
        assert isinstance(config.spike_slab, SpikeSlabConfig)
        assert isinstance(config.lasso, LassoConfig)
        assert config.exclude_variables == ()
        assert config.include_only_variables is None

    def test_custom_method(self):
        """Test creation with custom method."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            horseshoe=HorseshoeConfig(expected_nonzero=5),
        )

        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 5

    def test_exclude_variables(self):
        """Test with excluded variables."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            exclude_variables=("distribution", "price"),
        )

        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables

    def test_include_only_variables(self):
        """Test with include_only_variables."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.SPIKE_SLAB,
            include_only_variables=("weather", "gas_price"),
        )

        assert config.include_only_variables == ("weather", "gas_price")

    def test_get_selectable_variables_no_exclusions(self, sample_control_names):
        """Test partitioning with no exclusions."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        assert selectable == sample_control_names
        assert non_selectable == []

    def test_get_selectable_variables_with_exclusions(
        self, sample_control_names, sample_confounders
    ):
        """Test partitioning with excluded confounders."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            exclude_variables=sample_confounders,
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        assert "distribution" not in selectable
        assert "price" not in selectable
        assert "distribution" in non_selectable
        assert "price" in non_selectable
        assert "weather" in selectable
        assert "gas_price" in selectable
        assert "holiday" in selectable

    def test_get_selectable_variables_include_only(self, sample_control_names):
        """Test partitioning with include_only_variables."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.SPIKE_SLAB,
            include_only_variables=("weather", "gas_price"),
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        assert set(selectable) == {"weather", "gas_price"}
        assert set(non_selectable) == {"holiday", "distribution", "price"}

    def test_get_selectable_variables_both_constraints(self, sample_control_names):
        """Test partitioning with both include_only and exclude."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            include_only_variables=("weather", "gas_price", "distribution"),
            exclude_variables=("distribution",),  # Exclude from selectable
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        # distribution is in include_only but also excluded
        assert "distribution" not in selectable
        assert "weather" in selectable
        assert "gas_price" in selectable


# =============================================================================
# HorseshoeConfigBuilder Tests
# =============================================================================


class TestHorseshoeConfigBuilder:
    """Tests for HorseshoeConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        config = HorseshoeConfigBuilder().build()

        assert config.expected_nonzero == 3
        assert config.slab_scale == 2.0

    def test_with_expected_nonzero(self):
        """Test setting expected nonzero."""
        config = HorseshoeConfigBuilder().with_expected_nonzero(7).build()

        assert config.expected_nonzero == 7

    def test_with_slab_scale(self):
        """Test setting slab scale."""
        config = HorseshoeConfigBuilder().with_slab_scale(3.5).build()

        assert config.slab_scale == 3.5

    def test_with_heavy_tails(self):
        """Test heavy tails preset."""
        config = HorseshoeConfigBuilder().with_heavy_tails().build()

        assert config.slab_df == 2.0

    def test_with_aggressive_shrinkage(self):
        """Test aggressive shrinkage preset."""
        config = HorseshoeConfigBuilder().with_aggressive_shrinkage().build()

        assert (
            config.global_df == 1.0
        ), f"Expected global_df to be 1.0 for aggressive shrinkage found {config.global_df}"
        assert (
            config.local_df == 10.0
        ), f"Expected local_df to be 10.0 for aggressive shrinkage found {config.local_df}"

    def test_fluent_chaining(self):
        """Test fluent API chaining."""
        config = (
            HorseshoeConfigBuilder()
            .with_expected_nonzero(5)
            .with_slab_scale(2.5)
            .with_heavy_tails()
            .build()
        )

        assert config.expected_nonzero == 5
        assert config.slab_scale == 2.5
        assert config.slab_df == 2.0


# =============================================================================
# SpikeSlabConfigBuilder Tests
# =============================================================================


class TestSpikeSlabConfigBuilder:
    """Tests for SpikeSlabConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        config = SpikeSlabConfigBuilder().build()

        assert config.prior_inclusion_prob == 0.5
        assert config.use_continuous_relaxation is True

    def test_with_prior_inclusion(self):
        """Test setting prior inclusion probability."""
        config = SpikeSlabConfigBuilder().with_prior_inclusion(0.3).build()

        assert config.prior_inclusion_prob == 0.3

    def test_with_temperature(self):
        """Test setting temperature."""
        config = SpikeSlabConfigBuilder().with_temperature(0.05).build()

        assert config.temperature == 0.05

    def test_temperature_validation(self):
        """Test that non-positive temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            SpikeSlabConfigBuilder().with_temperature(0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            SpikeSlabConfigBuilder().with_temperature(-0.1)

    def test_continuous_mode(self):
        """Test continuous relaxation mode."""
        config = SpikeSlabConfigBuilder().continuous().build()

        assert config.use_continuous_relaxation is True

    def test_discrete_mode(self):
        """Test discrete mode."""
        config = SpikeSlabConfigBuilder().discrete().build()

        assert config.use_continuous_relaxation is False

    def test_with_sharp_selection(self):
        """Test sharp selection preset."""
        config = SpikeSlabConfigBuilder().with_sharp_selection().build()

        assert config.temperature == 0.05
        assert config.spike_scale == 0.005

    def test_with_soft_selection(self):
        """Test soft selection preset."""
        config = SpikeSlabConfigBuilder().with_soft_selection().build()

        assert config.temperature == 0.2
        assert config.spike_scale == 0.05


# =============================================================================
# LassoConfigBuilder Tests
# =============================================================================


class TestLassoConfigBuilder:
    """Tests for LassoConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        config = LassoConfigBuilder().build()

        assert config.regularization == 1.0
        assert config.adaptive is False

    def test_with_regularization(self):
        """Test setting regularization strength."""
        config = LassoConfigBuilder().with_regularization(2.5).build()

        assert config.regularization == 2.5

    def test_regularization_validation(self):
        """Test that non-positive regularization raises error."""
        with pytest.raises(ValueError, match="regularization must be positive"):
            LassoConfigBuilder().with_regularization(0)

        with pytest.raises(ValueError, match="regularization must be positive"):
            LassoConfigBuilder().with_regularization(-1.0)

    def test_adaptive_mode(self):
        """Test adaptive LASSO mode."""
        config = LassoConfigBuilder().adaptive().build()

        assert config.adaptive is True

    def test_non_adaptive_mode(self):
        """Test non-adaptive LASSO mode."""
        config = LassoConfigBuilder().adaptive().non_adaptive().build()

        assert config.adaptive is False

    def test_with_strong_regularization(self):
        """Test strong regularization preset."""
        config = LassoConfigBuilder().with_strong_regularization().build()

        assert config.regularization == 5.0

    def test_with_weak_regularization(self):
        """Test weak regularization preset."""
        config = LassoConfigBuilder().with_weak_regularization().build()

        assert config.regularization == 0.5


# =============================================================================
# VariableSelectionConfigBuilder Tests
# =============================================================================


class TestVariableSelectionConfigBuilder:
    """Tests for VariableSelectionConfigBuilder."""

    def test_default_build(self):
        """Test building with defaults (no selection)."""
        config = VariableSelectionConfigBuilder().build()

        assert config.method == VariableSelectionMethod.NONE

    def test_none_method(self):
        """Test explicitly setting no selection."""
        config = VariableSelectionConfigBuilder().none().build()

        assert config.method == VariableSelectionMethod.NONE

    def test_regularized_horseshoe(self):
        """Test regularized horseshoe method."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe(expected_nonzero=5)
            .build()
        )

        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 5

    def test_finnish_horseshoe(self):
        """Test Finnish horseshoe method."""
        config = (
            VariableSelectionConfigBuilder()
            .finnish_horseshoe(expected_nonzero=4)
            .build()
        )

        assert config.method == VariableSelectionMethod.FINNISH_HORSESHOE
        assert config.horseshoe.expected_nonzero == 4

    def test_spike_slab(self):
        """Test spike-and-slab method."""
        config = (
            VariableSelectionConfigBuilder().spike_slab(prior_inclusion=0.3).build()
        )

        assert config.method == VariableSelectionMethod.SPIKE_SLAB
        assert config.spike_slab.prior_inclusion_prob == 0.3
        assert config.spike_slab.use_continuous_relaxation is True

    def test_spike_slab_discrete(self):
        """Test spike-and-slab with discrete mode."""
        config = (
            VariableSelectionConfigBuilder()
            .spike_slab(prior_inclusion=0.5, continuous=False)
            .build()
        )

        assert config.spike_slab.use_continuous_relaxation is False

    def test_bayesian_lasso(self):
        """Test Bayesian LASSO method."""
        config = (
            VariableSelectionConfigBuilder().bayesian_lasso(regularization=2.0).build()
        )

        assert config.method == VariableSelectionMethod.BAYESIAN_LASSO
        assert config.lasso.regularization == 2.0

    def test_exclude_confounders(self):
        """Test excluding confounders."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .exclude_confounders("distribution", "price")
            .build()
        )

        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables

    def test_exclude_alias(self):
        """Test exclude() alias for exclude_confounders()."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .exclude("distribution", "price")
            .build()
        )

        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables

    def test_apply_only_to(self):
        """Test applying selection only to specific variables."""
        config = (
            VariableSelectionConfigBuilder()
            .spike_slab()
            .apply_only_to("weather", "gas_price")
            .build()
        )

        assert set(config.include_only_variables) == {"weather", "gas_price"}

    def test_clear_exclusions(self):
        """Test clearing exclusions."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .exclude_confounders("distribution")
            .apply_only_to("weather")
            .clear_exclusions()
            .build()
        )

        assert config.exclude_variables == ()
        assert config.include_only_variables is None

    def test_with_horseshoe_config(self):
        """Test setting horseshoe config from pre-built config."""
        horseshoe = HorseshoeConfig(expected_nonzero=7, slab_scale=3.0)

        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .with_horseshoe_config(horseshoe)
            .build()
        )

        assert config.horseshoe.expected_nonzero == 7
        assert config.horseshoe.slab_scale == 3.0

    def test_with_slab_scale(self):
        """Test setting slab scale via builder."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .with_slab_scale(3.5)
            .build()
        )

        assert config.horseshoe.slab_scale == 3.5

    def test_with_expected_nonzero(self):
        """Test setting expected nonzero via builder."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .with_expected_nonzero(8)
            .build()
        )

        assert config.horseshoe.expected_nonzero == 8

    def test_with_slab_df(self):
        """Test setting slab df via builder."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe()
            .with_slab_df(6.0)
            .build()
        )

        assert config.horseshoe.slab_df == 6.0

    def test_with_prior_inclusion(self):
        """Test setting prior inclusion via builder."""
        config = (
            VariableSelectionConfigBuilder()
            .spike_slab()
            .with_prior_inclusion(0.2)
            .build()
        )

        assert config.spike_slab.prior_inclusion_prob == 0.2

    def test_with_temperature(self):
        """Test setting temperature via builder."""
        config = (
            VariableSelectionConfigBuilder().spike_slab().with_temperature(0.08).build()
        )

        assert config.spike_slab.temperature == 0.08

    def test_with_sharp_selection(self):
        """Test sharp selection preset."""
        config = (
            VariableSelectionConfigBuilder().spike_slab().with_sharp_selection().build()
        )

        assert config.spike_slab.temperature == 0.05
        assert config.spike_slab.spike_scale == 0.005

    def test_with_regularization(self):
        """Test setting LASSO regularization via builder."""
        config = (
            VariableSelectionConfigBuilder()
            .bayesian_lasso()
            .with_regularization(3.0)
            .build()
        )

        assert config.lasso.regularization == 3.0

    def test_comprehensive_example(self, sample_confounders):
        """Test comprehensive configuration building."""
        config = (
            VariableSelectionConfigBuilder()
            .regularized_horseshoe(expected_nonzero=5)
            .with_slab_scale(2.5)
            .with_slab_df(4.0)
            .exclude_confounders(*sample_confounders)
            .build()
        )

        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 5
        assert config.horseshoe.slab_scale == 2.5
        assert config.horseshoe.slab_df == 4.0
        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables


# =============================================================================
# Factory Functions Tests (Config Module)
# =============================================================================


class TestConfigFactoryFunctions:
    """Tests for factory functions in config module."""

    def test_sparse_selection_config(self):
        """Test sparse_selection_config factory."""
        config = sparse_selection_config(expected_relevant=5)

        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 5
        assert config.horseshoe.slab_scale == 2.0

    def test_sparse_selection_config_with_confounders(self, sample_confounders):
        """Test sparse_selection_config with confounders."""
        config = sparse_selection_config(
            expected_relevant=3,
            confounders=sample_confounders,
        )

        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables

    def test_dense_selection_config(self):
        """Test dense_selection_config factory."""
        config = dense_selection_config(regularization=2.0)

        assert config.method == VariableSelectionMethod.BAYESIAN_LASSO
        assert config.lasso.regularization == 2.0

    def test_dense_selection_config_with_confounders(self, sample_confounders):
        """Test dense_selection_config with confounders."""
        config = dense_selection_config(
            regularization=1.5,
            confounders=sample_confounders,
        )

        assert "distribution" in config.exclude_variables

    def test_inclusion_prob_selection_config(self):
        """Test inclusion_prob_selection_config factory."""
        config = inclusion_prob_selection_config(prior_inclusion=0.3)

        assert config.method == VariableSelectionMethod.SPIKE_SLAB
        assert config.spike_slab.prior_inclusion_prob == 0.3
        assert config.spike_slab.temperature == 0.1


# =============================================================================
# Factory Functions Tests (Builders Module)
# =============================================================================


class TestBuilderFactoryFunctions:
    """Tests for factory functions in builders module."""

    def test_sparse_controls(self):
        """Test sparse_controls factory."""
        config = sparse_controls(expected_nonzero=4)

        assert config.method == VariableSelectionMethod.REGULARIZED_HORSESHOE
        assert config.horseshoe.expected_nonzero == 4

    def test_sparse_controls_with_confounders(self):
        """Test sparse_controls with confounder args."""
        config = sparse_controls(3, "distribution", "price")

        assert config.horseshoe.expected_nonzero == 3
        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables

    def test_selection_with_inclusion_probs(self):
        """Test selection_with_inclusion_probs factory."""
        config = selection_with_inclusion_probs(prior_inclusion=0.4)

        assert config.method == VariableSelectionMethod.SPIKE_SLAB
        assert config.spike_slab.prior_inclusion_prob == 0.4

    def test_selection_with_inclusion_probs_with_confounders(self):
        """Test selection_with_inclusion_probs with confounder args."""
        config = selection_with_inclusion_probs(0.3, "distribution")

        assert "distribution" in config.exclude_variables

    def test_dense_controls(self):
        """Test dense_controls factory."""
        config = dense_controls(regularization=2.5)

        assert config.method == VariableSelectionMethod.BAYESIAN_LASSO
        assert config.lasso.regularization == 2.5

    def test_dense_controls_with_confounders(self):
        """Test dense_controls with confounder args."""
        config = dense_controls(1.0, "distribution", "price")

        assert "distribution" in config.exclude_variables
        assert "price" in config.exclude_variables


# =============================================================================
# VariableSelectionResult Tests (requires PyMC)
# =============================================================================


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestVariableSelectionResult:
    """Tests for VariableSelectionResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation with just beta."""
        with pm.Model():
            beta = pm.Normal("beta", mu=0, sigma=1, shape=5)
            result = VariableSelectionResult(beta=beta)

            assert result.beta is beta
            assert result.inclusion_indicators is None
            assert result.local_shrinkage is None
            assert result.global_shrinkage is None
            assert result.effective_nonzero is None
            assert result.kappa is None

    def test_full_creation(self):
        """Test creation with all fields."""
        with pm.Model():
            beta = pm.Normal("beta", mu=0, sigma=1, shape=5)
            gamma = pm.Uniform("gamma", 0, 1, shape=5)
            tau = pm.HalfNormal("tau", sigma=1)

            result = VariableSelectionResult(
                beta=beta,
                inclusion_indicators=gamma,
                global_shrinkage=tau,
            )

            assert result.beta is beta
            assert result.inclusion_indicators is gamma
            assert result.global_shrinkage is tau


# =============================================================================
# Prior Creation Functions Tests (requires PyMC)
# =============================================================================


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestCreateRegularizedHorseshoePrior:
    """Tests for create_regularized_horseshoe_prior function."""

    def test_basic_creation(self):
        """Test basic prior creation."""
        config = HorseshoeConfig(expected_nonzero=3)

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_regularized_horseshoe_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            assert result.beta is not None
            assert "beta" in model.named_vars
            assert result.global_shrinkage is not None
            assert result.local_shrinkage is not None
            assert result.kappa is not None
            assert result.effective_nonzero is not None

    def test_with_dims(self):
        """Test prior creation with dimension name."""
        config = HorseshoeConfig(expected_nonzero=3)
        control_names = ["weather", "gas_price", "holiday"]

        with pm.Model(coords={"controls": control_names}) as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_regularized_horseshoe_prior(
                name="beta",
                n_variables=3,
                n_obs=100,
                sigma=sigma,
                config=config,
                dims="controls",
            )

            assert result.beta is not None

    def test_parameter_names(self):
        """Test that all expected parameters are created."""
        config = HorseshoeConfig()

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_regularized_horseshoe_prior(
                name="ctrl",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            # Check expected parameter names exist
            assert "ctrl_tau" in model.named_vars  # Global shrinkage
            assert "ctrl_lambda" in model.named_vars  # Local shrinkage
            assert "ctrl" in model.named_vars  # Coefficients
            assert "ctrl_kappa" in model.named_vars  # Shrinkage factors


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestCreateFinnishHorseshoePrior:
    """Tests for create_finnish_horseshoe_prior function."""

    def test_equivalent_to_regularized(self):
        """Test that Finnish horseshoe returns same structure as regularized."""
        config = HorseshoeConfig(expected_nonzero=3)

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_finnish_horseshoe_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            # Should have same structure as regularized
            assert result.beta is not None
            assert result.global_shrinkage is not None
            assert result.local_shrinkage is not None


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestCreateSpikeSlabPrior:
    """Tests for create_spike_slab_prior function."""

    def test_continuous_relaxation(self):
        """Test spike-slab with continuous relaxation."""
        config = SpikeSlabConfig(
            prior_inclusion_prob=0.5,
            use_continuous_relaxation=True,
        )

        with pm.Model() as model:
            result = create_spike_slab_prior(
                name="beta",
                n_variables=5,
                config=config,
            )

            assert result.beta is not None
            assert result.inclusion_indicators is not None
            assert "beta_gamma" in model.named_vars
            assert "beta_slab" in model.named_vars
            assert "beta_spike" in model.named_vars

    def test_discrete_mode(self):
        """Test spike-slab with discrete indicators."""
        config = SpikeSlabConfig(
            prior_inclusion_prob=0.5,
            use_continuous_relaxation=False,
        )

        with pm.Model() as model:
            result = create_spike_slab_prior(
                name="beta",
                n_variables=5,
                config=config,
            )

            assert result.beta is not None
            assert result.inclusion_indicators is not None

    def test_effective_nonzero_tracked(self):
        """Test that effective nonzero is tracked."""
        config = SpikeSlabConfig(prior_inclusion_prob=0.3)

        with pm.Model() as model:
            result = create_spike_slab_prior(
                name="beta",
                n_variables=5,
                config=config,
            )

            assert result.effective_nonzero is not None
            assert "beta_effective_nonzero" in model.named_vars


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestCreateBayesianLassoPrior:
    """Tests for create_bayesian_lasso_prior function."""

    def test_basic_creation(self):
        """Test basic LASSO prior creation."""
        config = LassoConfig(regularization=1.0)

        with pm.Model() as model:
            result = create_bayesian_lasso_prior(
                name="beta",
                n_variables=5,
                config=config,
            )

            assert result.beta is not None
            assert "beta" in model.named_vars

    def test_with_dims(self):
        """Test LASSO prior with dimension name."""
        config = LassoConfig(regularization=2.0)
        control_names = ["weather", "gas_price", "holiday"]

        with pm.Model(coords={"controls": control_names}) as model:
            result = create_bayesian_lasso_prior(
                name="beta",
                n_variables=3,
                config=config,
                dims="controls",
            )

            assert result.beta is not None


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestCreateVariableSelectionPrior:
    """Tests for create_variable_selection_prior dispatcher function."""

    def test_none_method(self):
        """Test dispatch with NONE method (standard priors)."""
        config = VariableSelectionConfig(method=VariableSelectionMethod.NONE)

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_variable_selection_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            assert result.beta is not None
            # Should have standard normal prior without selection machinery
            assert result.local_shrinkage is None

    def test_horseshoe_dispatch(self):
        """Test dispatch to regularized horseshoe."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            horseshoe=HorseshoeConfig(expected_nonzero=3),
        )

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_variable_selection_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            assert result.local_shrinkage is not None
            assert result.global_shrinkage is not None

    def test_spike_slab_dispatch(self):
        """Test dispatch to spike-slab."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.SPIKE_SLAB,
            spike_slab=SpikeSlabConfig(prior_inclusion_prob=0.3),
        )

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_variable_selection_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            assert result.inclusion_indicators is not None

    def test_lasso_dispatch(self):
        """Test dispatch to Bayesian LASSO."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.BAYESIAN_LASSO,
            lasso=LassoConfig(regularization=2.0),
        )

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_variable_selection_prior(
                name="beta",
                n_variables=5,
                n_obs=100,
                sigma=sigma,
                config=config,
            )

            assert result.beta is not None


# =============================================================================
# ControlEffectResult Tests (requires PyMC)
# =============================================================================


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestControlEffectResult:
    """Tests for ControlEffectResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation."""
        with pm.Model():
            contribution = pt.zeros(100)
            result = ControlEffectResult(contribution=contribution)

            assert result.contribution is contribution
            assert result.beta_selected is None
            assert result.beta_fixed is None
            assert result.selection_result is None
            assert result.components == {}


# =============================================================================
# Build Control Effects with Selection Tests (requires PyMC)
# =============================================================================


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestBuildControlEffectsWithSelection:
    """Tests for build_control_effects_with_selection function."""

    @pytest.fixture
    def sample_control_data(self):
        """Sample control variable data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_no_selection(self, sample_control_data, sample_control_names):
        """Test with no variable selection (method=NONE)."""
        config = VariableSelectionConfig(method=VariableSelectionMethod.NONE)

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)

            result = build_control_effects_with_selection(
                X_controls=sample_control_data,
                control_names=sample_control_names,
                n_obs=100,
                sigma=sigma,
                selection_config=config,
                name_prefix="ctrl",
            )

            assert result.contribution is not None
            # All variables should be in components
            assert len(result.components) == len(sample_control_names)

    def test_with_horseshoe_selection(self, sample_control_data, sample_control_names):
        """Test with regularized horseshoe selection."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            horseshoe=HorseshoeConfig(expected_nonzero=2),
        )

        with pm.Model(coords={"ctrl_select_dim": sample_control_names}) as model:
            sigma = pm.HalfNormal("sigma", sigma=1)

            result = build_control_effects_with_selection(
                X_controls=sample_control_data,
                control_names=sample_control_names,
                n_obs=100,
                sigma=sigma,
                selection_config=config,
                name_prefix="ctrl",
            )

            assert result.contribution is not None
            assert result.selection_result is not None
            assert result.beta_selected is not None

    def test_with_excluded_confounders(
        self, sample_control_data, sample_control_names, sample_confounders
    ):
        """Test with excluded confounders."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            horseshoe=HorseshoeConfig(expected_nonzero=2),
            exclude_variables=sample_confounders,
        )

        selectable, _ = config.get_selectable_variables(sample_control_names)

        with pm.Model(coords={"ctrl_select_dim": selectable}) as model:
            sigma = pm.HalfNormal("sigma", sigma=1)

            result = build_control_effects_with_selection(
                X_controls=sample_control_data,
                control_names=sample_control_names,
                n_obs=100,
                sigma=sigma,
                selection_config=config,
                name_prefix="ctrl",
            )

            assert result.contribution is not None
            # Should have both fixed (confounders) and selected coefficients
            assert result.beta_fixed is not None
            assert result.beta_selected is not None

    def test_components_populated(self, sample_control_data, sample_control_names):
        """Test that individual components are populated."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.SPIKE_SLAB,
            spike_slab=SpikeSlabConfig(prior_inclusion_prob=0.5),
        )

        with pm.Model(coords={"ctrl_select_dim": sample_control_names}) as model:
            sigma = pm.HalfNormal("sigma", sigma=1)

            result = build_control_effects_with_selection(
                X_controls=sample_control_data,
                control_names=sample_control_names,
                n_obs=100,
                sigma=sigma,
                selection_config=config,
                name_prefix="ctrl",
            )

            # All control names should be in components
            for name in sample_control_names:
                assert name in result.components


# =============================================================================
# Integration Tests (slow, requires sampling)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
class TestVariableSelectionIntegration:
    """Integration tests that require MCMC sampling."""

    def test_horseshoe_sampling(self):
        """Test that horseshoe model can be sampled."""
        np.random.seed(42)
        n_obs = 50
        n_vars = 5

        # True coefficients: sparse with 2 nonzero
        true_beta = np.array([0.0, 0.5, 0.0, 0.0, -0.3])
        X = np.random.randn(n_obs, n_vars)
        y = X @ true_beta + np.random.randn(n_obs) * 0.5

        config = HorseshoeConfig(expected_nonzero=2)

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=1)
            result = create_regularized_horseshoe_prior(
                name="beta",
                n_variables=n_vars,
                n_obs=n_obs,
                sigma=sigma,
                config=config,
            )

            mu = pt.dot(X, result.beta)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            # Quick sampling for test
            trace = pm.sample(
                draws=100,
                tune=100,
                chains=1,
                random_seed=42,
                progressbar=False,
            )

        # Check that we got samples
        assert "beta" in trace.posterior
        assert trace.posterior["beta"].shape[-1] == n_vars

    def test_spike_slab_sampling(self):
        """Test that spike-slab model can be sampled."""
        np.random.seed(42)
        n_obs = 50
        n_vars = 4

        X = np.random.randn(n_obs, n_vars)
        true_beta = np.array([0.0, 0.4, 0.0, -0.3])
        y = X @ true_beta + np.random.randn(n_obs) * 0.5

        config = SpikeSlabConfig(
            prior_inclusion_prob=0.5,
            use_continuous_relaxation=True,
        )

        with pm.Model() as model:
            result = create_spike_slab_prior(
                name="beta",
                n_variables=n_vars,
                config=config,
            )
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = pt.dot(X, result.beta)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(
                draws=100,
                tune=100,
                chains=1,
                random_seed=42,
                progressbar=False,
            )

        assert "beta" in trace.posterior
        assert "beta_gamma" in trace.posterior


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_exclusions(self, sample_control_names):
        """Test with empty exclusion list."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            exclude_variables=(),
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        assert selectable == sample_control_names
        assert non_selectable == []

    def test_all_excluded(self, sample_control_names):
        """Test with all variables excluded."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            exclude_variables=tuple(sample_control_names),
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        assert selectable == []
        assert non_selectable == sample_control_names

    def test_include_only_empty(self, sample_control_names):
        """Test include_only with empty list."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.SPIKE_SLAB,
            include_only_variables=(),
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        # Empty include_only means nothing is selectable
        assert selectable == []
        assert non_selectable == sample_control_names

    def test_unknown_variable_in_exclusion(self, sample_control_names):
        """Test exclusion with unknown variable name."""
        config = VariableSelectionConfig(
            method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
            exclude_variables=("unknown_var",),
        )

        selectable, non_selectable = config.get_selectable_variables(
            sample_control_names
        )

        # Unknown var should be ignored, all controls selectable
        assert selectable == sample_control_names
        assert non_selectable == []

    def test_method_enum_values(self):
        """Test all VariableSelectionMethod enum values exist."""
        assert VariableSelectionMethod.NONE.value == "none"
        assert (
            VariableSelectionMethod.REGULARIZED_HORSESHOE.value
            == "regularized_horseshoe"
        )
        assert VariableSelectionMethod.FINNISH_HORSESHOE.value == "finnish_horseshoe"
        assert VariableSelectionMethod.SPIKE_SLAB.value == "spike_slab"
        assert VariableSelectionMethod.BAYESIAN_LASSO.value == "bayesian_lasso"

    @pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        # Create a config with an invalid method value by bypassing dataclass
        config = VariableSelectionConfig.__new__(VariableSelectionConfig)
        object.__setattr__(config, "method", "invalid_method")
        object.__setattr__(config, "horseshoe", HorseshoeConfig())
        object.__setattr__(config, "spike_slab", SpikeSlabConfig())
        object.__setattr__(config, "lasso", LassoConfig())
        object.__setattr__(config, "exclude_variables", ())
        object.__setattr__(config, "include_only_variables", None)

        with pm.Model():
            sigma = pm.HalfNormal("sigma", sigma=1)

            with pytest.raises(ValueError, match="Unknown variable selection method"):
                create_variable_selection_prior(
                    name="beta",
                    n_variables=5,
                    n_obs=100,
                    sigma=sigma,
                    config=config,
                )

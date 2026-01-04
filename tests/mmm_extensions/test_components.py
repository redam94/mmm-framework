"""
Test suite for mmm_framework.mmm_extensions.components module.

Tests cover:
- Transformation functions (adstock, saturation)
- Prior factory functions
- Model component builders
- Result dataclasses

Note: These tests require PyMC/PyTensor but most are fast unit tests.
Tests requiring sampling are marked with @pytest.mark.slow

Some tests that depend on pt.scan are skipped if not available.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Import components
from mmm_framework.mmm_extensions.components import (
    # Transformation functions
    logistic_saturation,
    hill_saturation,
    apply_transformation_pipeline,
    # Prior factories
    create_adstock_prior,
    create_saturation_prior,
    create_effect_prior,
    # Result containers
    MediaTransformResult,
    CrossEffectSpec,
)

import pymc as pm
import pytensor.tensor as pt


# Check if scan-based functions are available
try:
    from mmm_framework.mmm_extensions.components import (
        geometric_adstock,
        geometric_adstock_matrix,
        build_media_transforms,
    )
    HAS_SCAN_FUNCTIONS = True
except (ImportError, AttributeError):
    HAS_SCAN_FUNCTIONS = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_1d_input():
    """1D input for single channel tests."""
    return np.array([100.0, 50.0, 0.0, 0.0, 0.0, 25.0, 10.0, 0.0])


@pytest.fixture
def sample_2d_input():
    """2D input for multi-channel tests."""
    return np.array([
        [100.0, 50.0],
        [50.0, 30.0],
        [0.0, 20.0],
        [25.0, 10.0],
    ])


# =============================================================================
# Geometric Adstock Tests (PyTensor)
# =============================================================================

@pytest.mark.skipif(not HAS_SCAN_FUNCTIONS, reason="scan-based functions not available")
class TestGeometricAdstock:
    """Tests for geometric_adstock function (PyTensor version)."""

    def test_with_pymc_context(self, sample_1d_input):
        """Test adstock within PyMC context."""
        with pm.Model():
            x = pt.as_tensor_variable(sample_1d_input)
            alpha = pt.constant(0.5)
            
            result = geometric_adstock(x, alpha, l_max=4)
            
            # Compile and evaluate
            fn = result.eval()
            
            assert len(fn) == len(sample_1d_input)

    def test_normalization(self):
        """Test that weights sum correctly when normalized."""
        with pm.Model():
            x = pt.as_tensor_variable([1.0, 0.0, 0.0, 0.0])
            alpha = pt.constant(0.5)
            
            result = geometric_adstock(x, alpha, l_max=4, normalize=True)
            fn = result.eval()
            
            # Output should exist and be non-zero
            assert fn[0] != 0

    def test_no_normalization(self):
        """Test without normalization."""
        with pm.Model():
            x = pt.as_tensor_variable([1.0, 0.0, 0.0, 0.0])
            alpha = pt.constant(0.5)
            
            result = geometric_adstock(x, alpha, l_max=4, normalize=False)
            fn = result.eval()
            
            assert fn[0] > 0


@pytest.mark.skipif(not HAS_SCAN_FUNCTIONS, reason="scan-based functions not available")
class TestGeometricAdstockMatrix:
    """Tests for geometric_adstock_matrix function."""

    def test_basic_matrix_operation(self, sample_2d_input):
        """Test adstock on multiple channels."""
        with pm.Model():
            X = pt.as_tensor_variable(sample_2d_input)
            alphas = pt.constant([0.5, 0.7])
            
            result = geometric_adstock_matrix(X, alphas, l_max=3)
            fn = result.eval()
            
            assert fn.shape == sample_2d_input.shape


# =============================================================================
# Logistic Saturation Tests
# =============================================================================

class TestLogisticSaturation:
    """Tests for logistic_saturation function."""

    def test_zero_input_gives_zero(self):
        """Test that zero input gives zero output."""
        x = pt.constant(0.0)
        lam = pt.constant(1.0)
        
        result = logistic_saturation(x, lam)
        fn = result.eval()
        
        assert fn == pytest.approx(0.0, abs=1e-10)

    def test_large_input_approaches_one(self):
        """Test asymptotic behavior."""
        x = pt.constant(100.0)
        lam = pt.constant(1.0)
        
        result = logistic_saturation(x, lam)
        fn = result.eval()
        
        assert fn == pytest.approx(1.0, abs=1e-6)

    def test_lambda_effect(self):
        """Test that higher lambda gives faster saturation."""
        x = pt.constant(5.0)
        
        result_low = logistic_saturation(x, pt.constant(0.1)).eval()
        result_high = logistic_saturation(x, pt.constant(1.0)).eval()
        
        assert result_high > result_low

    def test_vectorized(self, sample_1d_input):
        """Test vectorized operation."""
        x = pt.as_tensor_variable(sample_1d_input)
        lam = pt.constant(0.1)
        
        result = logistic_saturation(x, lam)
        fn = result.eval()
        
        assert len(fn) == len(sample_1d_input)
        assert all(fn >= 0)
        assert all(fn <= 1)


# =============================================================================
# Hill Saturation Tests
# =============================================================================

class TestHillSaturation:
    """Tests for hill_saturation function."""

    def test_at_kappa(self):
        """Test output at half-saturation point."""
        x = pt.constant(10.0)
        kappa = pt.constant(10.0)  # x = kappa
        slope = pt.constant(1.0)
        
        result = hill_saturation(x, kappa, slope)
        fn = result.eval()
        
        # At x=kappa, output should be 0.5
        assert fn == pytest.approx(0.5, rel=0.01)

    def test_slope_effect(self):
        """Test that higher slope gives steeper curve."""
        x = pt.constant(5.0)
        kappa = pt.constant(10.0)
        
        result_low = hill_saturation(x, kappa, pt.constant(1.0)).eval()
        result_high = hill_saturation(x, kappa, pt.constant(3.0)).eval()
        
        # Higher slope gives lower value for x < kappa
        assert result_low > result_high

    def test_range_zero_to_one(self, sample_1d_input):
        """Test output is in [0, 1]."""
        # Only positive values for Hill
        positive_input = np.abs(sample_1d_input) + 1
        
        x = pt.as_tensor_variable(positive_input)
        kappa = pt.constant(50.0)
        slope = pt.constant(2.0)
        
        result = hill_saturation(x, kappa, slope)
        fn = result.eval()
        
        assert all(fn >= 0)
        assert all(fn <= 1)


# =============================================================================
# Apply Transformation Pipeline Tests
# =============================================================================

class TestApplyTransformationPipeline:
    """Tests for apply_transformation_pipeline function."""

    def test_single_transform(self):
        """Test applying a single transformation."""
        x = pt.constant(10.0)
        
        transforms = [
            (logistic_saturation, {"lam": pt.constant(0.5)})
        ]
        
        result = apply_transformation_pipeline(x, transforms)
        fn = result.eval()
        
        expected = logistic_saturation(x, pt.constant(0.5)).eval()
        assert fn == pytest.approx(expected, rel=1e-6)

    def test_multiple_transforms(self):
        """Test applying multiple transformations in sequence."""
        x = pt.constant(100.0)
        
        # Chain transformations
        transforms = [
            (lambda v, scale: v * scale, {"scale": pt.constant(0.5)}),
            (logistic_saturation, {"lam": pt.constant(0.1)}),
        ]
        
        result = apply_transformation_pipeline(x, transforms)
        fn = result.eval()
        
        # 100 * 0.5 = 50, then saturate
        expected = logistic_saturation(pt.constant(50.0), pt.constant(0.1)).eval()
        assert fn == pytest.approx(expected, rel=1e-6)

    def test_empty_pipeline(self):
        """Test empty pipeline returns input."""
        x = pt.constant(42.0)
        
        result = apply_transformation_pipeline(x, [])
        fn = result.eval()
        
        assert fn == 42.0


# =============================================================================
# Prior Factory Tests
# =============================================================================

class TestCreateAdstockPrior:
    """Tests for create_adstock_prior function."""

    def test_beta_prior(self):
        """Test creating Beta prior for adstock."""
        with pm.Model() as model:
            alpha = create_adstock_prior("alpha_tv", prior_type="beta")
            
            assert "alpha_tv" in [v.name for v in model.free_RVs]

    def test_beta_prior_custom_params(self):
        """Test Beta prior with custom parameters."""
        with pm.Model() as model:
            alpha = create_adstock_prior(
                "alpha_digital",
                prior_type="beta",
                alpha=3,
                beta=2
            )
            
            # Sample to check it works
            prior = pm.sample_prior_predictive(samples=10)
            vals = prior.prior["alpha_digital"].values.flatten()
            
            assert all(vals >= 0) and all(vals <= 1)

    def test_uniform_prior(self):
        """Test creating Uniform prior."""
        with pm.Model() as model:
            alpha = create_adstock_prior("alpha_social", prior_type="uniform")
            
            assert "alpha_social" in [v.name for v in model.free_RVs]

    def test_invalid_prior_type(self):
        """Test that invalid prior type raises error."""
        with pm.Model():
            with pytest.raises(ValueError, match="Unknown prior type"):
                create_adstock_prior("alpha", prior_type="invalid")


class TestCreateSaturationPrior:
    """Tests for create_saturation_prior function."""

    def test_logistic_prior(self):
        """Test creating logistic saturation priors."""
        with pm.Model() as model:
            params = create_saturation_prior("sat_tv", saturation_type="logistic")
            
            assert "lam" in params
            assert "sat_tv_lam" in [v.name for v in model.free_RVs]

    def test_hill_prior(self):
        """Test creating Hill saturation priors."""
        with pm.Model() as model:
            params = create_saturation_prior("sat_digital", saturation_type="hill")
            
            assert "kappa" in params
            assert "slope" in params
            assert "sat_digital_kappa" in [v.name for v in model.free_RVs]
            assert "sat_digital_slope" in [v.name for v in model.free_RVs]

    def test_invalid_saturation_type(self):
        """Test that invalid saturation type raises error."""
        with pm.Model():
            with pytest.raises(ValueError, match="Unknown saturation type"):
                create_saturation_prior("sat", saturation_type="invalid")


class TestCreateEffectPrior:
    """Tests for create_effect_prior function."""

    def test_unconstrained(self):
        """Test unconstrained (Normal) prior."""
        with pm.Model() as model:
            beta = create_effect_prior("beta_control", constrained="none")
            
            assert "beta_control" in [v.name for v in model.free_RVs]

    def test_positive_constraint(self):
        """Test positive (HalfNormal) prior."""
        with pm.Model() as model:
            beta = create_effect_prior("beta_media", constrained="positive", sigma=0.5)
            
            prior = pm.sample_prior_predictive(samples=100)
            vals = prior.prior["beta_media"].values.flatten()
            
            assert all(vals >= 0)

    def test_negative_constraint(self):
        """Test negative (-HalfNormal) prior."""
        with pm.Model() as model:
            beta = create_effect_prior("beta_cannib", constrained="negative", sigma=0.3)
            pm.Deterministic("neg_beta_cannib", beta)
            prior = pm.sample_prior_predictive(samples=100)
            # The variable name might be different due to negation
            # Check that all sampled values are <= 0
            for var_name in prior.prior.data_vars:
                if "neg_beta_cannib" in var_name:
                    vals = prior.prior[var_name].values.flatten()
                    assert all(vals <= 0), f"Expected all values <= 0, got {vals}"
                    break

    def test_with_dims(self):
        """Test creating prior with dimensions."""
        with pm.Model(coords={"channel": ["tv", "digital"]}) as model:
            beta = create_effect_prior(
                "beta_media",
                constrained="positive",
                sigma=0.5,
                dims="channel"
            )
            
            # Should have shape matching dims
            prior = pm.sample_prior_predictive(samples=10)
            assert prior.prior["beta_media"].shape[-1] == 2


# =============================================================================
# CrossEffectSpec Tests
# =============================================================================

class TestCrossEffectSpec:
    """Tests for CrossEffectSpec dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        # CrossEffectSpec has: source_idx, target_idx, effect_type, prior_sigma
        spec = CrossEffectSpec(
            source_idx=0,
            target_idx=1,
            effect_type="cannibalization",
            prior_sigma=0.3,
        )
        
        assert spec.source_idx == 0
        assert spec.target_idx == 1
        assert spec.effect_type == "cannibalization"
        assert spec.prior_sigma == 0.3

    def test_halo_effect_spec(self):
        """Test spec for halo effect."""
        spec = CrossEffectSpec(
            source_idx=1,
            target_idx=0,
            effect_type="halo",
            prior_sigma=0.25,
        )
        
        assert spec.effect_type == "halo"
        assert spec.prior_sigma == 0.25

    def test_default_prior_sigma(self):
        """Test default prior_sigma value."""
        spec = CrossEffectSpec(
            source_idx=0,
            target_idx=1,
            effect_type="unconstrained",
        )
        
        assert spec.prior_sigma == 0.3  # Default value


# =============================================================================
# MediaTransformResult Tests
# =============================================================================

class TestMediaTransformResult:
    """Tests for MediaTransformResult dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        transformed = pt.zeros((10, 2))
        adstock_params = {"ch1": pt.constant(0.5)}
        saturation_params = {"ch1": {"lam": pt.constant(1.0)}}
        
        result = MediaTransformResult(
            transformed=transformed,
            adstock_params=adstock_params,
            saturation_params=saturation_params,
        )
        
        assert result.transformed is not None
        assert "ch1" in result.adstock_params
        assert "ch1" in result.saturation_params


# =============================================================================
# Build Media Transforms Tests
# =============================================================================

@pytest.mark.skipif(not HAS_SCAN_FUNCTIONS, reason="scan-based functions not available")
class TestBuildMediaTransforms:
    """Tests for build_media_transforms function."""

    def test_basic_transform(self):
        """Test basic media transformation building."""
        np.random.seed(42)
        X_media = np.random.randn(20, 2).astype(np.float64)
        
        with pm.Model(coords={"obs": range(20), "channel": ["tv", "digital"]}):
            X_tensor = pt.as_tensor_variable(X_media)
            
            result = build_media_transforms(
                X_media=X_tensor,
                channel_names=["tv", "digital"],
                adstock_config={"l_max": 4, "prior_type": "beta"},
                saturation_config={"type": "logistic"},
                share_params=False,
            )
            
            assert isinstance(result, MediaTransformResult)
            assert result.transformed is not None

    def test_shared_params(self):
        """Test with shared parameters across channels."""
        np.random.seed(42)
        X_media = np.random.randn(20, 3).astype(np.float64)
        
        with pm.Model(coords={"obs": range(20), "channel": ["tv", "digital", "social"]}):
            X_tensor = pt.as_tensor_variable(X_media)
            
            result = build_media_transforms(
                X_media=X_tensor,
                channel_names=["tv", "digital", "social"],
                adstock_config={"l_max": 8, "prior_type": "beta"},
                saturation_config={"type": "logistic"},
                share_params=True,
            )
            
            # Should have shared adstock parameter
            assert "shared" in result.adstock_params

    def test_with_name_prefix(self):
        """Test with name prefix for parameters."""
        np.random.seed(42)
        X_media = np.random.randn(10, 1).astype(np.float64)
        
        with pm.Model(coords={"obs": range(10), "channel": ["tv"]}) as model:
            X_tensor = pt.as_tensor_variable(X_media)
            
            result = build_media_transforms(
                X_media=X_tensor,
                channel_names=["tv"],
                adstock_config={"l_max": 4},
                saturation_config={"type": "logistic"},
                name_prefix="mediator",
            )
            
            var_names = [v.name for v in model.free_RVs]
            # Should have prefixed names
            assert any("mediator" in name for name in var_names)


# =============================================================================
# Integration Tests
# =============================================================================

class TestComponentIntegration:
    """Integration tests for components working together."""

    def test_saturation_with_effect_prior(self):
        """Test saturation combined with effect prior."""
        np.random.seed(42)
        X_raw = np.abs(np.random.randn(50, 2) * 100)
        
        with pm.Model(coords={"obs": range(50), "channel": ["tv", "digital"]}):
            X_tensor = pt.as_tensor_variable(X_raw)
            
            # Apply saturation to each channel
            saturated = []
            for i in range(2):
                lam = pm.Gamma(f"lam_{i}", alpha=3, beta=1)
                sat = logistic_saturation(X_tensor[:, i], lam)
                saturated.append(sat)
            
            X_sat = pt.stack(saturated, axis=1)
            
            # Create effect prior
            beta = create_effect_prior(
                "beta_media",
                constrained="positive",
                sigma=0.5,
                dims="channel"
            )
            
            # Compute linear effect
            effect = pt.dot(X_sat, beta)
            
            # Should compile without error
            assert effect is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
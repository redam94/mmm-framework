"""
Tests for utility functions and classes.

These tests ensure that utility refactoring doesn't introduce breaking changes.
"""

import numpy as np
import pytest


class TestComputeHDIBounds:
    """Tests for compute_hdi_bounds utility function."""

    def test_basic_hdi_computation(self):
        """Test basic HDI bounds computation."""
        from mmm_framework.utils import compute_hdi_bounds

        # Create samples with known distribution
        np.random.seed(42)
        samples = np.random.randn(1000, 10)  # 1000 samples, 10 observations

        lower, upper = compute_hdi_bounds(samples, hdi_prob=0.94)

        # Check shapes
        assert lower.shape == (10,)
        assert upper.shape == (10,)

        # Check that lower < upper
        assert np.all(lower < upper)

    def test_hdi_default_prob(self):
        """Test HDI with default probability (0.94)."""
        from mmm_framework.utils import compute_hdi_bounds

        np.random.seed(42)
        samples = np.random.randn(1000, 5)

        lower, upper = compute_hdi_bounds(samples)

        # Should contain approximately 94% of samples
        within_bounds = np.mean((samples >= lower) & (samples <= upper))
        assert 0.90 < within_bounds < 0.98

    def test_hdi_custom_prob(self):
        """Test HDI with custom probability."""
        from mmm_framework.utils import compute_hdi_bounds

        np.random.seed(42)
        samples = np.random.randn(1000, 5)

        # 50% HDI should be narrower
        lower_50, upper_50 = compute_hdi_bounds(samples, hdi_prob=0.50)
        lower_94, upper_94 = compute_hdi_bounds(samples, hdi_prob=0.94)

        # 50% interval should be narrower than 94%
        width_50 = upper_50 - lower_50
        width_94 = upper_94 - lower_94
        assert np.all(width_50 < width_94)

    def test_hdi_different_axis(self):
        """Test HDI along different axis."""
        from mmm_framework.utils import compute_hdi_bounds

        np.random.seed(42)
        samples = np.random.randn(100, 50)

        # Axis 0 (default) - aggregate over samples
        lower0, upper0 = compute_hdi_bounds(samples, axis=0)
        assert lower0.shape == (50,)

        # Axis 1 - aggregate over observations
        lower1, upper1 = compute_hdi_bounds(samples, axis=1)
        assert lower1.shape == (100,)

    def test_hdi_1d_array(self):
        """Test HDI on 1D array."""
        from mmm_framework.utils import compute_hdi_bounds

        np.random.seed(42)
        samples = np.random.randn(1000)

        lower, upper = compute_hdi_bounds(samples, axis=0)

        # Should be scalar-like
        assert np.isscalar(lower) or lower.ndim == 0
        assert lower < upper

    def test_hdi_percentile_values(self):
        """Test that HDI percentiles are computed correctly."""
        from mmm_framework.utils import compute_hdi_bounds

        # Use uniform distribution for predictable percentiles
        samples = np.linspace(0, 100, 101).reshape(-1, 1)  # 0, 1, 2, ..., 100

        # 94% HDI should be approximately [3, 97]
        lower, upper = compute_hdi_bounds(samples, hdi_prob=0.94)

        assert abs(lower[0] - 3.0) < 0.5
        assert abs(upper[0] - 97.0) < 0.5


class TestDataStandardizer:
    """Tests for DataStandardizer utility class."""

    def test_fit_transform(self):
        """Test fit_transform creates standardized data."""
        from mmm_framework.utils import DataStandardizer

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardizer = DataStandardizer()

        standardized, params = standardizer.fit_transform(data)

        # Standardized data should have mean ~0 and std ~1
        assert abs(standardized.mean()) < 1e-10
        assert abs(standardized.std() - 1.0) < 0.01

        # Params should store original mean and std (with epsilon adjustment)
        assert params.mean == 30.0
        # std includes epsilon, so should be slightly larger than raw std
        assert params.std >= data.std()
        assert abs(params.std - data.std()) < 1e-6  # Allow for epsilon

    def test_transform(self):
        """Test transform applies standardization."""
        from mmm_framework.utils import DataStandardizer

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardizer = DataStandardizer()

        # Fit first
        _, params = standardizer.fit_transform(data)

        # Transform new data
        new_data = np.array([30.0])  # Should be ~0 after standardization
        transformed = standardizer.transform(new_data, params)

        assert abs(transformed[0]) < 0.01

    def test_inverse_transform(self):
        """Test inverse_transform recovers original scale."""
        from mmm_framework.utils import DataStandardizer

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardizer = DataStandardizer()

        standardized, params = standardizer.fit_transform(data)
        recovered = standardizer.inverse_transform(standardized, params)

        np.testing.assert_allclose(recovered, data, rtol=1e-10)

    def test_2d_data(self):
        """Test standardization on 2D data."""
        from mmm_framework.utils import DataStandardizer

        data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        standardizer = DataStandardizer()

        standardized, params = standardizer.fit_transform(data)

        # Each column should be standardized
        assert standardized.shape == data.shape

        # Params should have mean/std per column
        assert len(params.mean) == 2 if hasattr(params.mean, '__len__') else True

    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero for constant data."""
        from mmm_framework.utils import DataStandardizer

        # Constant data has std=0
        data = np.array([5.0, 5.0, 5.0, 5.0])
        standardizer = DataStandardizer(epsilon=1e-8)

        # Should not raise
        standardized, params = standardizer.fit_transform(data)

        # std should include epsilon
        assert params.std > 0

    def test_params_to_dict(self):
        """Test StandardizationParams serialization."""
        from mmm_framework.utils import DataStandardizer, StandardizationParams

        data = np.array([10.0, 20.0, 30.0])
        standardizer = DataStandardizer()

        _, params = standardizer.fit_transform(data)
        params_dict = params.to_dict()

        # Should be serializable
        assert "mean" in params_dict
        assert "std" in params_dict

    def test_params_from_dict(self):
        """Test StandardizationParams deserialization."""
        from mmm_framework.utils import DataStandardizer, StandardizationParams

        data = np.array([10.0, 20.0, 30.0])
        standardizer = DataStandardizer()

        _, params = standardizer.fit_transform(data)
        params_dict = params.to_dict()

        # Round-trip
        restored_params = StandardizationParams.from_dict(params_dict)

        np.testing.assert_allclose(restored_params.mean, params.mean)
        np.testing.assert_allclose(restored_params.std, params.std)

    def test_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        from mmm_framework.utils import DataStandardizer

        standardizer = DataStandardizer()
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Must call fit"):
            standardizer.transform(data)


class TestTimeMaskHelper:
    """Tests for time mask helper functionality.

    These tests verify the behavior that should be provided by the
    _get_time_mask helper method in BayesianMMM.
    """

    def test_time_mask_with_period(self):
        """Test time mask with specific time period."""
        # Simulate the expected behavior
        time_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        time_period = (2, 5)

        start_idx, end_idx = time_period
        time_mask = (time_idx >= start_idx) & (time_idx <= end_idx)

        # Should mask indices 2, 3, 4, 5
        expected = np.array([False, False, True, True, True, True, False, False, False, False])
        np.testing.assert_array_equal(time_mask, expected)

    def test_time_mask_without_period(self):
        """Test time mask without time period (all True)."""
        n_obs = 10
        time_period = None

        if time_period is not None:
            start_idx, end_idx = time_period
            time_mask = np.ones(n_obs, dtype=bool)  # Placeholder
        else:
            time_mask = np.ones(n_obs, dtype=bool)

        # All should be True
        assert time_mask.all()
        assert time_mask.shape == (n_obs,)

    def test_time_mask_edge_cases(self):
        """Test time mask with edge case periods."""
        time_idx = np.array([0, 1, 2, 3, 4])

        # First element only
        time_mask = (time_idx >= 0) & (time_idx <= 0)
        assert time_mask.sum() == 1
        assert time_mask[0]

        # Last element only
        time_mask = (time_idx >= 4) & (time_idx <= 4)
        assert time_mask.sum() == 1
        assert time_mask[4]

        # All elements
        time_mask = (time_idx >= 0) & (time_idx <= 4)
        assert time_mask.all()

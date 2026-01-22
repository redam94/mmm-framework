"""
Tests for transform utility functions.

These tests ensure that transform refactoring doesn't introduce breaking changes.
They test the functions that will be moved to the transforms/ module.
"""

import numpy as np
import pytest


class TestGeometricAdstock:
    """Tests for geometric adstock transformation."""

    def test_basic_adstock(self):
        """Test basic adstock transformation."""
        from mmm_framework.transforms import geometric_adstock

        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        alpha = 0.5

        result = geometric_adstock(x, alpha)

        # First value should be unchanged
        assert result[0] == 100.0
        # Subsequent values should decay
        assert result[1] == 50.0  # 0 + 0.5 * 100
        assert result[2] == 25.0  # 0 + 0.5 * 50
        assert result[3] == 12.5  # 0 + 0.5 * 25

    def test_adstock_with_zero_alpha(self):
        """Test adstock with no carryover (alpha=0)."""
        from mmm_framework.transforms import geometric_adstock

        x = np.array([10.0, 20.0, 30.0, 40.0])
        result = geometric_adstock(x, alpha=0.0)

        # Should be unchanged
        np.testing.assert_array_equal(result, x)

    def test_adstock_with_high_alpha(self):
        """Test adstock with high carryover (alpha=0.9)."""
        from mmm_framework.transforms import geometric_adstock

        x = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(x, alpha=0.9)

        # Values should decay slowly
        assert result[0] == 100.0
        assert result[1] == 90.0  # 0.9 * 100
        assert result[7] > 40.0  # Still significant after 7 periods

    def test_adstock_formula(self):
        """Test that adstock follows y[t] = x[t] + alpha * y[t-1]."""
        from mmm_framework.transforms import geometric_adstock

        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        alpha = 0.7

        result = geometric_adstock(x, alpha)

        # Manual calculation
        expected = np.zeros(5)
        expected[0] = x[0]
        for t in range(1, 5):
            expected[t] = x[t] + alpha * expected[t - 1]

        np.testing.assert_array_almost_equal(result, expected)

    def test_adstock_preserves_shape(self):
        """Test that adstock preserves array shape."""
        from mmm_framework.transforms import geometric_adstock

        x = np.random.rand(100)
        result = geometric_adstock(x, alpha=0.5)

        assert result.shape == x.shape


class TestGeometricAdstock2D:
    """Tests for 2D geometric adstock transformation."""

    def test_basic_2d_adstock(self):
        """Test 2D adstock applies to each column."""
        from mmm_framework.transforms import geometric_adstock_2d

        X = np.array([
            [100.0, 200.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        alpha = 0.5

        result = geometric_adstock_2d(X, alpha)

        # Check first column
        assert result[0, 0] == 100.0
        assert result[1, 0] == 50.0
        assert result[2, 0] == 25.0

        # Check second column
        assert result[0, 1] == 200.0
        assert result[1, 1] == 100.0
        assert result[2, 1] == 50.0

    def test_2d_adstock_shape(self):
        """Test 2D adstock preserves shape."""
        from mmm_framework.transforms import geometric_adstock_2d

        X = np.random.rand(50, 5)
        result = geometric_adstock_2d(X, alpha=0.6)

        assert result.shape == X.shape

    def test_2d_adstock_independence(self):
        """Test columns are processed independently."""
        from mmm_framework.transforms import geometric_adstock, geometric_adstock_2d

        X = np.random.rand(20, 3)
        alpha = 0.7

        result_2d = geometric_adstock_2d(X, alpha)

        # Each column should match 1D result
        for c in range(X.shape[1]):
            result_1d = geometric_adstock(X[:, c], alpha)
            np.testing.assert_array_almost_equal(result_2d[:, c], result_1d)


class TestLogisticSaturation:
    """Tests for logistic saturation transformation."""

    def test_basic_saturation(self):
        """Test basic saturation behavior."""
        from mmm_framework.transforms import logistic_saturation

        x = np.array([0.0, 1.0, 5.0, 10.0, 100.0])
        lam = 0.5

        result = logistic_saturation(x, lam)

        # Zero input should give zero output
        assert result[0] == 0.0

        # Values should be bounded [0, 1] (can equal 1 for very large x due to floating point)
        assert all(0 <= r <= 1 for r in result)

        # Should be monotonically increasing
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_saturation_formula(self):
        """Test saturation follows 1 - exp(-lam * x)."""
        from mmm_framework.transforms import logistic_saturation

        x = np.array([0.5, 1.0, 2.0, 3.0])
        lam = 0.8

        result = logistic_saturation(x, lam)
        expected = 1.0 - np.exp(-lam * x)

        np.testing.assert_array_almost_equal(result, expected)

    def test_saturation_with_negative_clipping(self):
        """Test that negative values are clipped to zero."""
        from mmm_framework.transforms import logistic_saturation

        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = logistic_saturation(x, lam=0.5)

        # Negative values should be treated as 0
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_saturation_asymptote(self):
        """Test saturation approaches 1 for large values."""
        from mmm_framework.transforms import logistic_saturation

        x = np.array([100.0])
        result = logistic_saturation(x, lam=1.0)

        # Should be very close to 1
        assert result[0] > 0.99

    def test_saturation_lambda_effect(self):
        """Test that higher lambda gives faster saturation."""
        from mmm_framework.transforms import logistic_saturation

        x = np.array([1.0])

        result_low = logistic_saturation(x, lam=0.1)
        result_high = logistic_saturation(x, lam=2.0)

        # Higher lambda should saturate faster
        assert result_high[0] > result_low[0]


class TestCreateFourierFeatures:
    """Tests for Fourier feature creation."""

    def test_basic_fourier(self):
        """Test basic Fourier feature creation."""
        from mmm_framework.transforms import create_fourier_features

        t = np.arange(52)  # 52 weeks
        period = 52.0
        order = 3

        features = create_fourier_features(t, period, order)

        # Should have 2 * order columns (sin + cos for each order)
        assert features.shape == (52, 6)

    def test_fourier_periodicity(self):
        """Test that features are periodic."""
        from mmm_framework.transforms import create_fourier_features

        t = np.arange(104)  # 2 years
        period = 52.0
        order = 2

        features = create_fourier_features(t, period, order)

        # Values at t=0 and t=52 should be equal (same phase)
        np.testing.assert_array_almost_equal(features[0], features[52])

    def test_fourier_order_1(self):
        """Test Fourier with order 1."""
        from mmm_framework.transforms import create_fourier_features

        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * 2 * np.pi
        period = 2 * np.pi
        order = 1

        features = create_fourier_features(t, period, order)

        # Should have sin and cos columns
        assert features.shape == (5, 2)

        # First column should be sin
        np.testing.assert_array_almost_equal(features[:, 0], np.sin(t))
        # Second column should be cos
        np.testing.assert_array_almost_equal(features[:, 1], np.cos(t))

    def test_fourier_order_zero(self):
        """Test Fourier with order 0 returns empty."""
        from mmm_framework.transforms import create_fourier_features

        t = np.arange(10)
        features = create_fourier_features(t, period=10.0, order=0)

        assert features.shape == (10, 0)


class TestCreateBSplineBasis:
    """Tests for B-spline basis creation."""

    def test_basic_bspline(self):
        """Test basic B-spline basis creation."""
        from mmm_framework.transforms import create_bspline_basis

        t = np.linspace(0, 1, 100)
        n_knots = 5
        degree = 3

        basis = create_bspline_basis(t, n_knots, degree)

        # Should have n_knots + degree + 1 basis functions
        expected_cols = n_knots + degree + 1
        assert basis.shape == (100, expected_cols)

    def test_bspline_partition_of_unity(self):
        """Test that B-spline basis sums to 1 (partition of unity)."""
        from mmm_framework.transforms import create_bspline_basis

        t = np.linspace(0, 1, 50)
        basis = create_bspline_basis(t, n_knots=4, degree=3)

        # Basis should sum to ~1 at each point
        row_sums = basis.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_bspline_nonnegative(self):
        """Test that B-spline basis is non-negative."""
        from mmm_framework.transforms import create_bspline_basis

        t = np.linspace(0, 1, 100)
        basis = create_bspline_basis(t, n_knots=6, degree=3)

        assert np.all(basis >= -1e-10)  # Allow small numerical errors

    def test_bspline_boundary_clamping(self):
        """Test that B-spline is clamped at boundaries."""
        from mmm_framework.transforms import create_bspline_basis

        t = np.linspace(0, 1, 100)
        basis = create_bspline_basis(t, n_knots=5, degree=3)

        # First basis function should be 1 at t=0
        assert abs(basis[0, 0] - 1.0) < 0.01

        # Last basis function should be 1 at t=1
        assert abs(basis[-1, -1] - 1.0) < 0.01


class TestCreatePiecewiseTrendMatrix:
    """Tests for piecewise trend matrix creation."""

    def test_basic_piecewise(self):
        """Test basic piecewise trend matrix creation."""
        from mmm_framework.transforms import create_piecewise_trend_matrix

        t = np.linspace(0, 1, 100)
        n_changepoints = 5

        s, A = create_piecewise_trend_matrix(t, n_changepoints)

        # Should have n_changepoints changepoint locations
        assert len(s) == n_changepoints

        # Design matrix should have shape (n_obs, n_changepoints)
        assert A.shape == (100, n_changepoints)

    def test_piecewise_changepoint_range(self):
        """Test changepoint range parameter."""
        from mmm_framework.transforms import create_piecewise_trend_matrix

        t = np.linspace(0, 1, 100)

        s_default, _ = create_piecewise_trend_matrix(t, 5, changepoint_range=0.8)
        s_full, _ = create_piecewise_trend_matrix(t, 5, changepoint_range=1.0)

        # Default range should keep changepoints in first 80%
        assert all(cp <= 0.8 for cp in s_default)

        # Full range can have changepoints anywhere
        assert max(s_full) > 0.8

    def test_piecewise_indicator_matrix(self):
        """Test that A is an indicator matrix."""
        from mmm_framework.transforms import create_piecewise_trend_matrix

        t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        s, A = create_piecewise_trend_matrix(t, n_changepoints=2, changepoint_range=0.5)

        # A should be binary (0 or 1)
        assert set(A.flatten()).issubset({0.0, 1.0})

        # For each changepoint, indicator should be 0 before and 1 after
        for j in range(A.shape[1]):
            # Find transition point
            for i in range(len(t) - 1):
                if A[i, j] == 0 and A[i + 1, j] == 1:
                    # t[i+1] should be >= s[j]
                    assert t[i + 1] >= s[j]

    def test_piecewise_returns_tuple(self):
        """Test that function returns tuple of (s, A)."""
        from mmm_framework.transforms import create_piecewise_trend_matrix

        t = np.linspace(0, 1, 50)
        result = create_piecewise_trend_matrix(t, 3)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with model.py imports."""

    def test_can_import_from_model(self):
        """Test that functions are still importable from model.py."""
        from mmm_framework.model import (
            create_fourier_features,
            geometric_adstock_np,
            geometric_adstock_2d,
            logistic_saturation_np,
            create_bspline_basis,
            create_piecewise_trend_matrix,
        )

        # Just verify they're callable
        assert callable(create_fourier_features)
        assert callable(geometric_adstock_np)
        assert callable(geometric_adstock_2d)
        assert callable(logistic_saturation_np)
        assert callable(create_bspline_basis)
        assert callable(create_piecewise_trend_matrix)

    def test_can_import_from_transforms(self):
        """Test that functions can be imported from transforms module."""
        from mmm_framework.transforms import (
            create_fourier_features,
            geometric_adstock,
            geometric_adstock_2d,
            logistic_saturation,
            create_bspline_basis,
            create_piecewise_trend_matrix,
        )

        assert callable(create_fourier_features)
        assert callable(geometric_adstock)
        assert callable(geometric_adstock_2d)
        assert callable(logistic_saturation)
        assert callable(create_bspline_basis)
        assert callable(create_piecewise_trend_matrix)

    def test_model_and_transforms_produce_same_results(self):
        """Test that model.py and transforms produce identical results."""
        from mmm_framework.model import (
            geometric_adstock_np as model_adstock,
            logistic_saturation_np as model_saturation,
            create_fourier_features as model_fourier,
        )
        from mmm_framework.transforms import (
            geometric_adstock as transform_adstock,
            logistic_saturation as transform_saturation,
            create_fourier_features as transform_fourier,
        )

        np.random.seed(42)
        x = np.random.rand(50) * 100

        # Adstock
        np.testing.assert_array_equal(
            model_adstock(x, 0.7),
            transform_adstock(x, 0.7)
        )

        # Saturation
        np.testing.assert_array_equal(
            model_saturation(x, 0.5),
            transform_saturation(x, 0.5)
        )

        # Fourier
        t = np.arange(52)
        np.testing.assert_array_equal(
            model_fourier(t, 52.0, 3),
            transform_fourier(t, 52.0, 3)
        )

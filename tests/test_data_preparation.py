"""
Tests for data preparation module.

These tests ensure that the DataPreparator class correctly prepares
data for BayesianMMM models.
"""

import numpy as np
import pandas as pd
import pytest


class TestScalingParameters:
    """Tests for ScalingParameters dataclass."""

    def test_basic_creation(self):
        """Test basic creation of ScalingParameters."""
        from mmm_framework.data_preparation import ScalingParameters

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0, "Radio": 500.0},
        )

        assert params.y_mean == 100.0
        assert params.y_std == 25.0
        assert params.media_max == {"TV": 1000.0, "Radio": 500.0}
        assert params.control_mean is None
        assert params.control_std is None

    def test_creation_with_controls(self):
        """Test creation with control parameters."""
        from mmm_framework.data_preparation import ScalingParameters

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0},
            control_mean=np.array([2.0, 3.0]),
            control_std=np.array([1.0, 1.5]),
        )

        assert params.control_mean is not None
        np.testing.assert_array_equal(params.control_mean, [2.0, 3.0])
        np.testing.assert_array_equal(params.control_std, [1.0, 1.5])

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from mmm_framework.data_preparation import ScalingParameters

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0},
        )

        d = params.to_dict()

        assert d["y_mean"] == 100.0
        assert d["y_std"] == 25.0
        assert d["media_max"] == {"TV": 1000.0}
        assert "control_mean" not in d

    def test_to_dict_with_controls(self):
        """Test to_dict with controls."""
        from mmm_framework.data_preparation import ScalingParameters

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0},
            control_mean=np.array([2.0]),
            control_std=np.array([1.0]),
        )

        d = params.to_dict()

        assert "control_mean" in d
        assert d["control_mean"] == [2.0]
        assert d["control_std"] == [1.0]

    def test_from_dict(self):
        """Test creation from dictionary."""
        from mmm_framework.data_preparation import ScalingParameters

        d = {
            "y_mean": 100.0,
            "y_std": 25.0,
            "media_max": {"TV": 1000.0},
        }

        params = ScalingParameters.from_dict(d)

        assert params.y_mean == 100.0
        assert params.y_std == 25.0
        assert params.media_max == {"TV": 1000.0}

    def test_from_dict_with_controls(self):
        """Test from_dict with controls."""
        from mmm_framework.data_preparation import ScalingParameters

        d = {
            "y_mean": 100.0,
            "y_std": 25.0,
            "media_max": {"TV": 1000.0},
            "control_mean": [2.0, 3.0],
            "control_std": [1.0, 1.5],
        }

        params = ScalingParameters.from_dict(d)

        np.testing.assert_array_equal(params.control_mean, [2.0, 3.0])
        np.testing.assert_array_equal(params.control_std, [1.0, 1.5])

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        from mmm_framework.data_preparation import ScalingParameters

        original = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0, "Radio": 500.0},
            control_mean=np.array([2.0, 3.0]),
            control_std=np.array([1.0, 1.5]),
        )

        d = original.to_dict()
        restored = ScalingParameters.from_dict(d)

        assert restored.y_mean == original.y_mean
        assert restored.y_std == original.y_std
        assert restored.media_max == original.media_max
        np.testing.assert_array_equal(restored.control_mean, original.control_mean)
        np.testing.assert_array_equal(restored.control_std, original.control_std)


class TestPreparedData:
    """Tests for PreparedData dataclass."""

    def test_basic_creation(self):
        """Test basic creation of PreparedData."""
        from mmm_framework.data_preparation import PreparedData, ScalingParameters

        n_obs = 100
        n_channels = 3

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0, "Radio": 500.0, "Digital": 300.0},
        )

        prepared = PreparedData(
            y=np.zeros(n_obs),
            y_raw=np.ones(n_obs) * 100,
            X_media_adstocked={0.5: np.zeros((n_obs, n_channels))},
            X_media_raw=np.zeros((n_obs, n_channels)),
            X_controls=None,
            X_controls_raw=None,
            scaling_params=params,
            n_obs=n_obs,
            n_channels=n_channels,
            n_controls=0,
            channel_names=["TV", "Radio", "Digital"],
            control_names=[],
            time_idx=np.arange(n_obs),
            geo_idx=np.zeros(n_obs, dtype=np.int32),
            product_idx=np.zeros(n_obs, dtype=np.int32),
            n_periods=n_obs,
            n_geos=1,
            n_products=1,
            has_geo=False,
            has_product=False,
            t_scaled=np.linspace(0, 1, n_obs),
        )

        assert prepared.n_obs == n_obs
        assert prepared.n_channels == n_channels
        assert prepared.n_controls == 0
        assert len(prepared.channel_names) == n_channels


class TestStandardizeFunctions:
    """Tests for standalone standardization functions."""

    def test_standardize_array(self):
        """Test array standardization."""
        from mmm_framework.data_preparation import standardize_array

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardized, mean, std = standardize_array(data)

        assert mean == 30.0
        assert abs(std - np.std(data)) < 1e-6
        assert abs(standardized.mean()) < 1e-10  # Should be ~0
        assert abs(standardized.std() - 1.0) < 0.01  # Should be ~1

    def test_standardize_array_with_epsilon(self):
        """Test standardization with constant array."""
        from mmm_framework.data_preparation import standardize_array

        data = np.array([5.0, 5.0, 5.0, 5.0])
        standardized, mean, std = standardize_array(data, epsilon=1e-8)

        assert mean == 5.0
        assert std >= 1e-8  # Should include epsilon
        # All values should be 0 or very close
        assert abs(standardized.max()) < 1e-6

    def test_unstandardize_array(self):
        """Test unstandardization."""
        from mmm_framework.data_preparation import standardize_array, unstandardize_array

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardized, mean, std = standardize_array(data)
        restored = unstandardize_array(standardized, mean, std)

        np.testing.assert_array_almost_equal(restored, data)

    def test_roundtrip(self):
        """Test standardize/unstandardize roundtrip."""
        from mmm_framework.data_preparation import standardize_array, unstandardize_array

        np.random.seed(42)
        data = np.random.randn(100) * 50 + 200

        standardized, mean, std = standardize_array(data)
        restored = unstandardize_array(standardized, mean, std)

        np.testing.assert_array_almost_equal(restored, data)


class TestDataPreparatorImports:
    """Tests for DataPreparator imports."""

    def test_can_import_preparator(self):
        """Test that DataPreparator can be imported."""
        from mmm_framework.data_preparation import DataPreparator

        assert DataPreparator is not None

    def test_can_import_scaling_parameters(self):
        """Test that ScalingParameters can be imported."""
        from mmm_framework.data_preparation import ScalingParameters

        assert ScalingParameters is not None

    def test_can_import_prepared_data(self):
        """Test that PreparedData can be imported."""
        from mmm_framework.data_preparation import PreparedData

        assert PreparedData is not None

    def test_can_import_helper_functions(self):
        """Test that helper functions can be imported."""
        from mmm_framework.data_preparation import (
            standardize_array,
            unstandardize_array,
        )

        assert callable(standardize_array)
        assert callable(unstandardize_array)


class TestDataPreparatorMethods:
    """Tests for DataPreparator methods."""

    def test_compute_adstocked_media(self):
        """Test adstock computation."""
        from mmm_framework.data_preparation import DataPreparator

        # Create a minimal mock panel
        class MockCoords:
            channels = ["TV", "Radio"]
            controls = []
            periods = list(range(10))
            geographies = []
            products = []
            n_periods = 10
            n_geos = 1
            n_products = 1
            has_geo = False
            has_product = False

        class MockColumns:
            period = "period"
            geography = "geography"
            product = "product"

        class MockConfig:
            columns = MockColumns()

        class MockPanel:
            config = MockConfig()
            coords = MockCoords()
            index = pd.RangeIndex(10)
            y = pd.Series(np.random.randn(10) * 100 + 500)
            X_media = pd.DataFrame(
                np.random.rand(10, 2) * 100,
                columns=["TV", "Radio"]
            )
            X_controls = None

        panel = MockPanel()
        preparator = DataPreparator(
            panel=panel,
            adstock_alphas=[0.0, 0.5],
        )

        X_media_raw = panel.X_media.values
        adstocked, media_max = preparator._compute_adstocked_media(
            X_media_raw, ["TV", "Radio"]
        )

        # Check structure
        assert 0.0 in adstocked
        assert 0.5 in adstocked
        assert "TV" in media_max
        assert "Radio" in media_max

        # Check shapes
        assert adstocked[0.0].shape == (10, 2)
        assert adstocked[0.5].shape == (10, 2)

        # Check normalized values are in [0, 1]
        assert adstocked[0.0].max() <= 1.0
        assert adstocked[0.5].max() <= 1.0

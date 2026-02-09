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
        from mmm_framework.data_preparation import (
            standardize_array,
            unstandardize_array,
        )

        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        standardized, mean, std = standardize_array(data)
        restored = unstandardize_array(standardized, mean, std)

        np.testing.assert_array_almost_equal(restored, data)

    def test_roundtrip(self):
        """Test standardize/unstandardize roundtrip."""
        from mmm_framework.data_preparation import (
            standardize_array,
            unstandardize_array,
        )

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
            X_media = pd.DataFrame(np.random.rand(10, 2) * 100, columns=["TV", "Radio"])
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


class TestDataPreparatorPrepare:
    """Tests for DataPreparator.prepare() method."""

    @pytest.fixture
    def mock_panel_simple(self):
        """Create a simple mock panel for testing."""

        class MockCoords:
            channels = ["TV", "Radio"]
            controls = ["Price"]
            periods = list(range(20))
            geographies = []
            products = []
            n_periods = 20
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
            index = pd.RangeIndex(20)
            y = pd.Series(np.random.randn(20) * 100 + 500)
            X_media = pd.DataFrame(np.random.rand(20, 2) * 100, columns=["TV", "Radio"])
            X_controls = pd.DataFrame(np.random.rand(20, 1) * 50, columns=["Price"])

        return MockPanel()

    def test_prepare_returns_prepared_data(self, mock_panel_simple):
        """Test that prepare returns PreparedData."""
        from mmm_framework.data_preparation import DataPreparator, PreparedData

        preparator = DataPreparator(
            panel=mock_panel_simple,
            adstock_alphas=[0.0, 0.5],
        )

        result = preparator.prepare()

        assert isinstance(result, PreparedData)
        assert result.n_obs == 20
        assert result.n_channels == 2
        assert result.n_controls == 1

    def test_prepare_standardizes_target(self, mock_panel_simple):
        """Test that prepare standardizes the target variable."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_simple,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        # Standardized y should have mean ~0 and std ~1
        assert abs(result.y.mean()) < 0.1
        assert abs(result.y.std() - 1.0) < 0.1
        # Raw y should be preserved
        np.testing.assert_array_equal(result.y_raw, mock_panel_simple.y.values)

    def test_prepare_standardizes_controls(self, mock_panel_simple):
        """Test that prepare standardizes control variables."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_simple,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.X_controls is not None
        assert result.X_controls_raw is not None
        assert result.n_controls == 1
        assert result.control_names == ["Price"]

    def test_prepare_without_controls(self):
        """Test prepare when no controls are present."""
        from mmm_framework.data_preparation import DataPreparator

        class MockCoords:
            channels = ["TV"]
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
            X_media = pd.DataFrame(np.random.rand(10, 1) * 100, columns=["TV"])
            X_controls = None

        preparator = DataPreparator(
            panel=MockPanel(),
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.X_controls is None
        assert result.X_controls_raw is None
        assert result.n_controls == 0
        assert result.control_names == []

    def test_prepare_creates_scaling_params(self, mock_panel_simple):
        """Test that prepare creates correct scaling parameters."""
        from mmm_framework.data_preparation import DataPreparator, ScalingParameters

        preparator = DataPreparator(
            panel=mock_panel_simple,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert isinstance(result.scaling_params, ScalingParameters)
        assert result.scaling_params.y_mean is not None
        assert result.scaling_params.y_std > 0
        assert "TV" in result.scaling_params.media_max
        assert "Radio" in result.scaling_params.media_max

    def test_prepare_creates_time_index(self, mock_panel_simple):
        """Test that prepare creates time indices."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_simple,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.time_idx is not None
        assert len(result.time_idx) == 20
        assert result.n_periods == 20


class TestDataPreparatorWithGeo:
    """Tests for DataPreparator with geographic dimension."""

    @pytest.fixture
    def mock_panel_with_geo(self):
        """Create mock panel with geography."""

        class MockCoords:
            channels = ["TV", "Radio"]
            controls = []
            periods = ["2024-01", "2024-02", "2024-03", "2024-04"]
            geographies = ["East", "West"]
            products = []
            n_periods = 4
            n_geos = 2
            n_products = 1
            has_geo = True
            has_product = False

        class MockColumns:
            period = "period"
            geography = "geography"
            product = "product"

        class MockConfig:
            columns = MockColumns()

        # Create MultiIndex
        periods = ["2024-01", "2024-02", "2024-03", "2024-04"] * 2
        geos = ["East"] * 4 + ["West"] * 4
        multi_idx = pd.MultiIndex.from_arrays(
            [periods, geos], names=["period", "geography"]
        )

        class MockPanel:
            config = MockConfig()
            coords = MockCoords()

        panel = MockPanel()
        panel.index = multi_idx
        panel.y = pd.Series(np.random.randn(8) * 100 + 500, index=multi_idx)
        panel.X_media = pd.DataFrame(
            np.random.rand(8, 2) * 100, columns=["TV", "Radio"], index=multi_idx
        )
        panel.X_controls = None

        return panel

    def test_prepare_with_geo(self, mock_panel_with_geo):
        """Test prepare with geographic dimension."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_with_geo,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.has_geo is True
        assert result.n_geos == 2
        assert result.geo_names == ["East", "West"]
        assert len(result.geo_idx) == 8

    def test_geo_indices_are_correct(self, mock_panel_with_geo):
        """Test that geo indices map correctly."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_with_geo,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        # First 4 should be East (0), next 4 should be West (1)
        assert all(result.geo_idx[:4] == 0)
        assert all(result.geo_idx[4:] == 1)


class TestDataPreparatorWithProduct:
    """Tests for DataPreparator with product dimension."""

    @pytest.fixture
    def mock_panel_with_product(self):
        """Create mock panel with product dimension."""

        class MockCoords:
            channels = ["TV"]
            controls = []
            periods = ["2024-01", "2024-02"]
            geographies = []
            products = ["ProductA", "ProductB", "ProductC"]
            n_periods = 2
            n_geos = 1
            n_products = 3
            has_geo = False
            has_product = True

        class MockColumns:
            period = "period"
            geography = "geography"
            product = "product"

        class MockConfig:
            columns = MockColumns()

        # Create MultiIndex
        periods = ["2024-01", "2024-02"] * 3
        products = ["ProductA"] * 2 + ["ProductB"] * 2 + ["ProductC"] * 2
        multi_idx = pd.MultiIndex.from_arrays(
            [periods, products], names=["period", "product"]
        )

        class MockPanel:
            config = MockConfig()
            coords = MockCoords()

        panel = MockPanel()
        panel.index = multi_idx
        panel.y = pd.Series(np.random.randn(6) * 100 + 500, index=multi_idx)
        panel.X_media = pd.DataFrame(
            np.random.rand(6, 1) * 100, columns=["TV"], index=multi_idx
        )
        panel.X_controls = None

        return panel

    def test_prepare_with_product(self, mock_panel_with_product):
        """Test prepare with product dimension."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_with_product,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.has_product is True
        assert result.n_products == 3
        assert result.product_names == ["ProductA", "ProductB", "ProductC"]


class TestDataPreparatorSeasonality:
    """Tests for seasonality feature preparation."""

    @pytest.fixture
    def mock_panel_for_seasonality(self):
        """Create mock panel for seasonality tests."""

        class MockCoords:
            channels = ["TV"]
            controls = []
            periods = list(range(52))  # One year of weekly data
            geographies = []
            products = []
            n_periods = 52
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
            index = pd.RangeIndex(52)
            y = pd.Series(np.random.randn(52) * 100 + 500)
            X_media = pd.DataFrame(np.random.rand(52, 1) * 100, columns=["TV"])
            X_controls = None

        return MockPanel()

    def test_prepare_without_seasonality(self, mock_panel_for_seasonality):
        """Test prepare without seasonality config."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_for_seasonality,
            adstock_alphas=[0.0],
            seasonality_config=None,
        )

        result = preparator.prepare()

        assert result.seasonality_features == {}

    def test_prepare_with_yearly_seasonality(self, mock_panel_for_seasonality):
        """Test prepare with yearly seasonality."""
        from mmm_framework.data_preparation import DataPreparator

        class MockSeasonalityConfig:
            yearly = 3  # 3 Fourier harmonics

        preparator = DataPreparator(
            panel=mock_panel_for_seasonality,
            adstock_alphas=[0.0],
            seasonality_config=MockSeasonalityConfig(),
        )

        result = preparator.prepare()

        assert "yearly" in result.seasonality_features
        # 3 harmonics = 6 features (sin + cos for each)
        assert result.seasonality_features["yearly"].shape == (52, 6)

    def test_prepare_with_zero_seasonality(self, mock_panel_for_seasonality):
        """Test prepare with zero seasonality order."""
        from mmm_framework.data_preparation import DataPreparator

        class MockSeasonalityConfig:
            yearly = 0

        preparator = DataPreparator(
            panel=mock_panel_for_seasonality,
            adstock_alphas=[0.0],
            seasonality_config=MockSeasonalityConfig(),
        )

        result = preparator.prepare()

        # No features when order is 0
        assert "yearly" not in result.seasonality_features


class TestDataPreparatorTrend:
    """Tests for trend feature preparation."""

    @pytest.fixture
    def mock_panel_for_trend(self):
        """Create mock panel for trend tests."""

        class MockCoords:
            channels = ["TV"]
            controls = []
            periods = list(range(100))
            geographies = []
            products = []
            n_periods = 100
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
            index = pd.RangeIndex(100)
            y = pd.Series(np.random.randn(100) * 100 + 500)
            X_media = pd.DataFrame(np.random.rand(100, 1) * 100, columns=["TV"])
            X_controls = None

        return MockPanel()

    def test_prepare_without_trend(self, mock_panel_for_trend):
        """Test prepare without trend config."""
        from mmm_framework.data_preparation import DataPreparator

        preparator = DataPreparator(
            panel=mock_panel_for_trend,
            adstock_alphas=[0.0],
            trend_config=None,
        )

        result = preparator.prepare()

        assert result.trend_features == {}

    def test_prepare_with_spline_trend(self, mock_panel_for_trend):
        """Test prepare with spline trend."""
        from mmm_framework.data_preparation import DataPreparator
        from mmm_framework.model import TrendType

        class MockTrendConfig:
            type = TrendType.SPLINE
            n_knots = 5
            spline_degree = 3

        preparator = DataPreparator(
            panel=mock_panel_for_trend,
            adstock_alphas=[0.0],
            trend_config=MockTrendConfig(),
        )

        result = preparator.prepare()

        assert "spline_basis" in result.trend_features
        assert "n_spline_coef" in result.trend_features
        assert result.trend_features["spline_basis"].shape[0] == 100

    def test_prepare_with_piecewise_trend(self, mock_panel_for_trend):
        """Test prepare with piecewise trend."""
        from mmm_framework.data_preparation import DataPreparator
        from mmm_framework.model import TrendType

        class MockTrendConfig:
            type = TrendType.PIECEWISE
            n_changepoints = 10
            changepoint_range = 0.8

        preparator = DataPreparator(
            panel=mock_panel_for_trend,
            adstock_alphas=[0.0],
            trend_config=MockTrendConfig(),
        )

        result = preparator.prepare()

        assert "changepoints" in result.trend_features
        assert "changepoint_matrix" in result.trend_features
        assert len(result.trend_features["changepoints"]) == 10

    def test_prepare_with_gp_trend(self, mock_panel_for_trend):
        """Test prepare with GP trend."""
        from mmm_framework.data_preparation import DataPreparator
        from mmm_framework.model import TrendType

        class MockTrendConfig:
            type = TrendType.GP
            gp_lengthscale_prior_mu = 0.3
            gp_lengthscale_prior_sigma = 0.1
            gp_amplitude_prior_sigma = 1.0
            gp_n_basis = 20
            gp_c = 1.5

        preparator = DataPreparator(
            panel=mock_panel_for_trend,
            adstock_alphas=[0.0],
            trend_config=MockTrendConfig(),
        )

        result = preparator.prepare()

        assert "gp_config" in result.trend_features
        assert result.trend_features["gp_config"]["n_basis"] == 20
        assert result.trend_features["gp_config"]["lengthscale_mu"] == 0.3


class TestDataPreparatorTimeIndex:
    """Tests for time index creation."""

    def test_time_index_with_range_index(self):
        """Test time index creation with RangeIndex."""
        from mmm_framework.data_preparation import DataPreparator

        class MockCoords:
            channels = ["TV"]
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
            X_media = pd.DataFrame(np.random.rand(10, 1) * 100, columns=["TV"])
            X_controls = None

        preparator = DataPreparator(
            panel=MockPanel(),
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        # With RangeIndex, time_idx should be sequential
        np.testing.assert_array_equal(result.time_idx, np.arange(10))

    def test_time_index_with_multi_index(self):
        """Test time index creation with MultiIndex."""
        from mmm_framework.data_preparation import DataPreparator

        class MockCoords:
            channels = ["TV"]
            controls = []
            periods = ["2024-01", "2024-02"]
            geographies = ["East", "West"]
            products = []
            n_periods = 2
            n_geos = 2
            n_products = 1
            has_geo = True
            has_product = False

        class MockColumns:
            period = "period"
            geography = "geography"
            product = "product"

        class MockConfig:
            columns = MockColumns()

        periods = ["2024-01", "2024-02"] * 2
        geos = ["East", "East", "West", "West"]
        multi_idx = pd.MultiIndex.from_arrays(
            [periods, geos], names=["period", "geography"]
        )

        class MockPanel:
            config = MockConfig()
            coords = MockCoords()

        panel = MockPanel()
        panel.index = multi_idx
        panel.y = pd.Series(np.random.randn(4) * 100 + 500, index=multi_idx)
        panel.X_media = pd.DataFrame(
            np.random.rand(4, 1) * 100, columns=["TV"], index=multi_idx
        )
        panel.X_controls = None

        preparator = DataPreparator(
            panel=panel,
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        # Time index should map period to integer index
        assert len(result.time_idx) == 4
        assert result.time_idx[0] == 0  # 2024-01
        assert result.time_idx[1] == 1  # 2024-02


class TestDataPreparatorEdgeCases:
    """Edge case tests for DataPreparator."""

    def test_single_observation(self):
        """Test with single observation."""
        from mmm_framework.data_preparation import DataPreparator

        class MockCoords:
            channels = ["TV"]
            controls = []
            periods = [0]
            geographies = []
            products = []
            n_periods = 1
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
            index = pd.RangeIndex(1)
            y = pd.Series([500.0])
            X_media = pd.DataFrame([[100.0]], columns=["TV"])
            X_controls = None

        preparator = DataPreparator(
            panel=MockPanel(),
            adstock_alphas=[0.0],
        )

        result = preparator.prepare()

        assert result.n_obs == 1
        assert result.n_periods == 1

    def test_many_channels(self):
        """Test with many channels."""
        from mmm_framework.data_preparation import DataPreparator

        n_channels = 10
        channel_names = [f"Channel_{i}" for i in range(n_channels)]

        class MockCoords:
            channels = channel_names
            controls = []
            periods = list(range(20))
            geographies = []
            products = []
            n_periods = 20
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
            index = pd.RangeIndex(20)
            y = pd.Series(np.random.randn(20) * 100 + 500)
            X_media = pd.DataFrame(
                np.random.rand(20, n_channels) * 100, columns=channel_names
            )
            X_controls = None

        preparator = DataPreparator(
            panel=MockPanel(),
            adstock_alphas=[0.0, 0.5, 0.9],
        )

        result = preparator.prepare()

        assert result.n_channels == n_channels
        assert len(result.channel_names) == n_channels
        assert all(ch in result.scaling_params.media_max for ch in channel_names)

    def test_multiple_adstock_alphas(self):
        """Test with multiple adstock alpha values."""
        from mmm_framework.data_preparation import DataPreparator

        class MockCoords:
            channels = ["TV"]
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
            X_media = pd.DataFrame(np.random.rand(10, 1) * 100, columns=["TV"])
            X_controls = None

        alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        preparator = DataPreparator(
            panel=MockPanel(),
            adstock_alphas=alphas,
        )

        result = preparator.prepare()

        # All alphas should be in the adstocked dict
        for alpha in alphas:
            assert alpha in result.X_media_adstocked


class TestPreparedDataOptionalFields:
    """Tests for PreparedData optional fields."""

    def test_default_seasonality_features(self):
        """Test that seasonality_features defaults to empty dict."""
        from mmm_framework.data_preparation import PreparedData, ScalingParameters

        params = ScalingParameters(y_mean=100.0, y_std=25.0, media_max={"TV": 1000.0})

        prepared = PreparedData(
            y=np.zeros(10),
            y_raw=np.ones(10),
            X_media_adstocked={0.0: np.zeros((10, 1))},
            X_media_raw=np.zeros((10, 1)),
            X_controls=None,
            X_controls_raw=None,
            scaling_params=params,
            n_obs=10,
            n_channels=1,
            n_controls=0,
            channel_names=["TV"],
            control_names=[],
            time_idx=np.arange(10),
            geo_idx=np.zeros(10, dtype=np.int32),
            product_idx=np.zeros(10, dtype=np.int32),
            n_periods=10,
            n_geos=1,
            n_products=1,
            has_geo=False,
            has_product=False,
            t_scaled=np.linspace(0, 1, 10),
        )

        assert prepared.seasonality_features == {}
        assert prepared.trend_features == {}

    def test_geo_product_names_optional(self):
        """Test that geo_names and product_names are optional."""
        from mmm_framework.data_preparation import PreparedData, ScalingParameters

        params = ScalingParameters(y_mean=100.0, y_std=25.0, media_max={"TV": 1000.0})

        prepared = PreparedData(
            y=np.zeros(10),
            y_raw=np.ones(10),
            X_media_adstocked={0.0: np.zeros((10, 1))},
            X_media_raw=np.zeros((10, 1)),
            X_controls=None,
            X_controls_raw=None,
            scaling_params=params,
            n_obs=10,
            n_channels=1,
            n_controls=0,
            channel_names=["TV"],
            control_names=[],
            time_idx=np.arange(10),
            geo_idx=np.zeros(10, dtype=np.int32),
            product_idx=np.zeros(10, dtype=np.int32),
            n_periods=10,
            n_geos=1,
            n_products=1,
            has_geo=False,
            has_product=False,
            t_scaled=np.linspace(0, 1, 10),
        )

        assert prepared.geo_names is None
        assert prepared.product_names is None

    def test_prepared_data_with_all_fields(self):
        """Test PreparedData with all optional fields populated."""
        from mmm_framework.data_preparation import PreparedData, ScalingParameters

        params = ScalingParameters(
            y_mean=100.0,
            y_std=25.0,
            media_max={"TV": 1000.0},
            control_mean=np.array([2.0]),
            control_std=np.array([1.0]),
        )

        prepared = PreparedData(
            y=np.zeros(10),
            y_raw=np.ones(10),
            X_media_adstocked={0.0: np.zeros((10, 1))},
            X_media_raw=np.zeros((10, 1)),
            X_controls=np.zeros((10, 1)),
            X_controls_raw=np.ones((10, 1)),
            scaling_params=params,
            n_obs=10,
            n_channels=1,
            n_controls=1,
            channel_names=["TV"],
            control_names=["Price"],
            time_idx=np.arange(10),
            geo_idx=np.zeros(10, dtype=np.int32),
            product_idx=np.zeros(10, dtype=np.int32),
            n_periods=10,
            n_geos=2,
            n_products=1,
            has_geo=True,
            has_product=False,
            t_scaled=np.linspace(0, 1, 10),
            seasonality_features={"yearly": np.zeros((10, 6))},
            trend_features={"spline_basis": np.zeros((10, 5))},
            geo_names=["East", "West"],
            product_names=None,
        )

        assert prepared.n_controls == 1
        assert "yearly" in prepared.seasonality_features
        assert "spline_basis" in prepared.trend_features
        assert prepared.geo_names == ["East", "West"]

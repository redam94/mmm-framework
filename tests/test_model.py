"""
Test suite for mmm_framework.model module.

Tests cover:
- TrendType enum and TrendConfig dataclass
- Helper functions (adstock, saturation, Fourier, B-splines, piecewise)
- Result container dataclasses (MMMResults, PredictionResults, etc.)
- BayesianMMM class initialization and model building
- Model fitting, prediction, and contribution analysis (slow tests)

Note: Tests requiring MCMC sampling are marked with @pytest.mark.slow
and can be skipped with: pytest -m "not slow"
"""
import pytensor
pytensor.config.exception_verbosity = 'high'
pytensor.config.mode == 'NUMBA'
pytensor.config.cxx = "" 

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import model components
from mmm_framework.model import (
    # Enums and config
    TrendType,
    TrendConfig,
    # Helper functions
    create_fourier_features,
    geometric_adstock_np,
    geometric_adstock_2d,
    logistic_saturation_np,
    create_bspline_basis,
    create_piecewise_trend_matrix,
    # Result containers
    MMMResults,
    PredictionResults,
    ContributionResults,
    ComponentDecomposition,
    # Main model class
    BayesianMMM,
)

from mmm_framework.config import (
    DimensionType,
    ModelConfig,
    InferenceMethod,
    MFFConfig,
    KPIConfig,
    MediaChannelConfig,
    ControlVariableConfig,
    SeasonalityConfig,
    HierarchicalConfig,
)

from mmm_framework.data_loader import (
    PanelCoordinates,
    PanelDataset,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_periods():
    """Sample weekly periods."""
    return pd.date_range("2020-01-06", periods=52, freq="W-MON")


@pytest.fixture
def simple_panel(sample_periods):
    """Create a simple national-level PanelDataset for testing."""
    n_obs = len(sample_periods)

    # Create coordinates
    coords = PanelCoordinates(
        periods=sample_periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )

    # Create data
    np.random.seed(42)
    y = pd.Series(1000 + np.random.randn(n_obs) * 100, name="Sales")
    X_media = pd.DataFrame(
        {
            "TV": np.abs(np.random.randn(n_obs) * 50 + 100),
            "Digital": np.abs(np.random.randn(n_obs) * 30 + 80),
        }
    )
    X_controls = pd.DataFrame(
        {
            "Price": 10 + np.random.randn(n_obs) * 0.5,
        }
    )

    # Create simple config
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD]),
        ],
    )

    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=sample_periods,
        config=config,
    )


@pytest.fixture
def geo_panel(sample_periods):
    """Create a geo-level PanelDataset for testing."""
    geos = ["East", "West", "Central"]
    n_periods = len(sample_periods)
    n_obs = n_periods * len(geos)

    # Create multi-index
    index = pd.MultiIndex.from_product(
        [sample_periods, geos], names=["Period", "Geography"]
    )

    coords = PanelCoordinates(
        periods=sample_periods,
        geographies=geos,
        products=None,
        channels=["TV"],
        controls=["Price"],
    )

    np.random.seed(42)
    y = pd.Series(500 + np.random.randn(n_obs) * 50, index=index, name="Sales")
    X_media = pd.DataFrame(
        {"TV": np.abs(np.random.randn(n_obs) * 30 + 50)}, index=index
    )
    X_controls = pd.DataFrame({"Price": 10 + np.random.randn(n_obs) * 0.3}, index=index)

    config = MFFConfig(
        kpi=KPIConfig(
            name="Sales", dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        ),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(
                name="Price", dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY]
            ),
        ],
    )

    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=index,
        config=config,
    )


@pytest.fixture
def model_config():
    """Basic model configuration for testing."""
    return ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=100,
        n_tune=100,
        target_accept=0.8,
    )


@pytest.fixture
def trend_config():
    """Default trend configuration."""
    return TrendConfig(type=TrendType.LINEAR)


# =============================================================================
# TrendType Enum Tests
# =============================================================================


class TestTrendType:
    """Tests for TrendType enum."""

    def test_all_values(self):
        """Test all TrendType values exist."""
        assert TrendType.NONE.value == "none"
        assert TrendType.LINEAR.value == "linear"
        assert TrendType.PIECEWISE.value == "piecewise"
        assert TrendType.SPLINE.value == "spline"
        assert TrendType.GP.value == "gaussian_process"

    def test_from_string(self):
        """Test creating TrendType from string."""
        assert TrendType("linear") == TrendType.LINEAR
        assert TrendType("gaussian_process") == TrendType.GP

    def test_invalid_value_raises(self):
        """Test that invalid value raises error."""
        with pytest.raises(ValueError):
            TrendType("invalid")


# =============================================================================
# TrendConfig Tests
# =============================================================================


class TestTrendConfig:
    """Tests for TrendConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrendConfig()

        assert config.type == TrendType.LINEAR
        assert config.n_changepoints == 10
        assert config.changepoint_range == 0.8
        assert config.n_knots == 10
        assert config.spline_degree == 3
        assert config.gp_n_basis == 20

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = TrendConfig(
            type=TrendType.PIECEWISE,
            n_changepoints=15,
            changepoint_prior_scale=0.1,
        )

        assert config.type == TrendType.PIECEWISE
        assert config.n_changepoints == 15
        assert config.changepoint_prior_scale == 0.1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = TrendConfig(type=TrendType.SPLINE, n_knots=8)
        d = config.to_dict()

        assert d["type"] == "spline"
        assert d["n_knots"] == 8
        assert "gp_lengthscale_prior_mu" in d

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "type": "gaussian_process",
            "gp_n_basis": 30,
            "gp_lengthscale_prior_mu": 0.4,
            "n_changepoints": 10,
            "changepoint_range": 0.8,
            "changepoint_prior_scale": 0.05,
            "n_knots": 10,
            "spline_degree": 3,
            "spline_prior_sigma": 1.0,
            "gp_lengthscale_prior_sigma": 0.2,
            "gp_amplitude_prior_sigma": 0.5,
            "gp_c": 1.5,
            "growth_prior_mu": 0.0,
            "growth_prior_sigma": 0.1,
        }
        config = TrendConfig.from_dict(d)

        assert config.type == TrendType.GP
        assert config.gp_n_basis == 30
        assert config.gp_lengthscale_prior_mu == 0.4

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = TrendConfig(
            type=TrendType.PIECEWISE,
            n_changepoints=20,
            changepoint_prior_scale=0.02,
        )

        restored = TrendConfig.from_dict(original.to_dict())

        assert restored.type == original.type
        assert restored.n_changepoints == original.n_changepoints
        assert restored.changepoint_prior_scale == original.changepoint_prior_scale


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateFourierFeatures:
    """Tests for create_fourier_features function."""

    def test_basic_creation(self):
        """Test basic Fourier feature creation."""
        t = np.arange(52)
        period = 52
        order = 2

        features = create_fourier_features(t, period, order)

        # 2 features per order (sin + cos)
        assert features.shape == (52, 4)

    def test_order_zero(self):
        """Test with order=0 returns empty features."""
        t = np.arange(52)
        features = create_fourier_features(t, period=52, order=0)

        assert features.shape == (52, 0)

    def test_periodicity(self):
        """Test that features have correct periodicity."""
        t = np.arange(104)  # 2 years
        period = 52
        order = 1

        features = create_fourier_features(t, period, order)

        # sin feature should repeat after 52 observations
        np.testing.assert_array_almost_equal(
            features[:52, 0], features[52:, 0], decimal=10
        )

    def test_sin_cos_orthogonality(self):
        """Test that sin and cos features are approximately orthogonal."""
        t = np.arange(52)
        features = create_fourier_features(t, period=52, order=1)

        # Dot product of sin and cos should be near zero
        dot_product = np.dot(features[:, 0], features[:, 1])
        assert abs(dot_product) < 1e-10


class TestGeometricAdstockNp:
    """Tests for geometric_adstock_np function."""

    def test_no_decay(self):
        """Test with alpha=0 (no carryover)."""
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = geometric_adstock_np(x, alpha=0.0)

        np.testing.assert_array_equal(result, x)

    def test_full_decay(self):
        """Test with high alpha (strong carryover)."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        result = geometric_adstock_np(x, alpha=0.9)

        # First value unchanged
        assert result[0] == 1.0
        # Subsequent values should decay
        assert result[1] == pytest.approx(0.9, rel=1e-6)
        assert result[2] == pytest.approx(0.81, rel=1e-6)
        assert result[3] == pytest.approx(0.729, rel=1e-6)

    def test_accumulation(self):
        """Test that adstock accumulates over time."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = geometric_adstock_np(x, alpha=0.5)

        # Each value should be >= corresponding input
        assert all(result >= x)
        # Values should increase (accumulation)
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_impulse_response(self):
        """Test impulse response shape."""
        x = np.zeros(10)
        x[0] = 100.0

        result = geometric_adstock_np(x, alpha=0.7)

        # Should decay exponentially
        for i in range(1, 10):
            expected = 100 * (0.7**i)
            assert result[i] == pytest.approx(expected, rel=1e-6)


class TestGeometricAdstock2d:
    """Tests for geometric_adstock_2d function."""

    def test_shape_preserved(self):
        """Test that output shape matches input."""
        X = np.random.randn(52, 3)
        result = geometric_adstock_2d(X, alpha=0.5)

        assert result.shape == X.shape

    def test_per_channel_application(self):
        """Test that adstock is applied to each channel."""
        X = np.zeros((5, 2))
        X[0, 0] = 1.0
        X[0, 1] = 2.0

        result = geometric_adstock_2d(X, alpha=0.5)

        # Channel 0
        assert result[0, 0] == 1.0
        assert result[1, 0] == pytest.approx(0.5, rel=1e-6)

        # Channel 1
        assert result[0, 1] == 2.0
        assert result[1, 1] == pytest.approx(1.0, rel=1e-6)


class TestLogisticSaturationNp:
    """Tests for logistic_saturation_np function."""

    def test_zero_input(self):
        """Test that zero input gives zero output."""
        x = np.array([0.0])
        result = logistic_saturation_np(x, lam=1.0)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_asymptotic_behavior(self):
        """Test that output approaches 1 for large inputs."""
        x = np.array([100.0])
        result = logistic_saturation_np(x, lam=1.0)

        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_monotonic_increasing(self):
        """Test that function is monotonically increasing."""
        x = np.linspace(0, 100, 100)
        result = logistic_saturation_np(x, lam=0.1)

        # Check monotonicity
        assert all(np.diff(result) >= 0)

    def test_lambda_effect(self):
        """Test that higher lambda gives faster saturation."""
        x = np.array([10.0])

        result_low = logistic_saturation_np(x, lam=0.1)
        result_high = logistic_saturation_np(x, lam=1.0)

        # Higher lambda should give higher saturation at same x
        assert result_high[0] > result_low[0]

    def test_negative_clipped(self):
        """Test that negative values are clipped to zero."""
        x = np.array([-5.0, -1.0, 0.0, 1.0])
        result = logistic_saturation_np(x, lam=1.0)

        # Negative inputs should give 0
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)

    def test_range_zero_to_one(self):
        """Test that output is always in [0, 1)."""
        x = np.random.uniform(0, 1000, 100)
        result = logistic_saturation_np(x, lam=0.5)

        assert all(result >= 0)
        assert all(result <= 1)


class TestCreateBsplineBasis:
    """Tests for create_bspline_basis function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        t = np.linspace(0, 1, 52)
        n_knots = 5

        basis = create_bspline_basis(t, n_knots=n_knots, degree=3)

        # Number of basis functions = n_interior_knots + degree + 1
        assert basis.shape[0] == 52

    def test_partition_of_unity(self):
        """Test that basis functions sum to 1 (approximately)."""
        t = np.linspace(0, 1, 100)
        basis = create_bspline_basis(t, n_knots=10, degree=3)

        # Sum across basis functions should be ~1 for interior points
        row_sums = basis[10:-10].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)

    def test_non_negative(self):
        """Test that basis functions are non-negative."""
        t = np.linspace(0, 1, 50)
        basis = create_bspline_basis(t, n_knots=5, degree=3)

        assert np.all(basis >= -1e-10)  # Allow small numerical errors

    def test_different_degrees(self):
        """Test with different spline degrees."""
        t = np.linspace(0, 1, 50)

        basis_linear = create_bspline_basis(t, n_knots=5, degree=1)
        basis_cubic = create_bspline_basis(t, n_knots=5, degree=3)

        # Different degrees should give different shapes
        assert basis_linear.shape[1] != basis_cubic.shape[1]


class TestCreatePiecewiseTrendMatrix:
    """Tests for create_piecewise_trend_matrix function."""

    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        t = np.linspace(0, 1, 52)
        n_changepoints = 10

        s, A = create_piecewise_trend_matrix(t, n_changepoints)

        assert len(s) == n_changepoints
        assert A.shape == (52, n_changepoints)

    def test_changepoint_locations(self):
        """Test that changepoints are in correct range."""
        t = np.linspace(0, 1, 100)
        s, A = create_piecewise_trend_matrix(
            t, n_changepoints=10, changepoint_range=0.8
        )

        # All changepoints should be <= 0.8
        assert all(s <= 0.8)
        assert all(s >= 0)

    def test_design_matrix_binary(self):
        """Test that design matrix contains 0s and 1s."""
        t = np.linspace(0, 1, 50)
        s, A = create_piecewise_trend_matrix(t, n_changepoints=5)

        # A should contain only 0 and 1
        assert set(np.unique(A)) == {0.0, 1.0}

    def test_design_matrix_structure(self):
        """Test the step-function structure of design matrix."""
        t = np.linspace(0, 1, 100)
        s, A = create_piecewise_trend_matrix(t, n_changepoints=5)

        # Each column should be 0 then 1 (step function)
        for j in range(A.shape[1]):
            col = A[:, j]
            # Find first 1
            first_one = np.argmax(col)
            # All before should be 0, all after should be 1
            if first_one > 0:
                assert all(col[:first_one] == 0)
            assert all(col[first_one:] == 1)


# =============================================================================
# Result Container Tests
# =============================================================================


class TestMMMResults:
    """Tests for MMMResults dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        mock_trace = MagicMock()
        mock_model = MagicMock()
        mock_panel = MagicMock()

        results = MMMResults(
            trace=mock_trace,
            model=mock_model,
            panel=mock_panel,
            diagnostics={"divergences": 0, "rhat_max": 1.01},
            y_mean=1000.0,
            y_std=100.0,
        )

        assert results.y_mean == 1000.0
        assert results.y_std == 100.0
        assert results.diagnostics["divergences"] == 0

    def test_default_values(self):
        """Test default values."""
        mock_trace = MagicMock()
        mock_model = MagicMock()
        mock_panel = MagicMock()

        results = MMMResults(
            trace=mock_trace,
            model=mock_model,
            panel=mock_panel,
        )

        assert results.channel_contributions is None
        assert results.diagnostics == {}
        assert results.y_mean == 0.0
        assert results.y_std == 1.0


class TestPredictionResults:
    """Tests for PredictionResults dataclass."""

    def test_properties(self):
        """Test n_samples and n_obs properties."""
        mock_ppc = MagicMock()
        samples = np.random.randn(100, 52)  # 100 samples, 52 obs

        results = PredictionResults(
            posterior_predictive=mock_ppc,
            y_pred_mean=samples.mean(axis=0),
            y_pred_std=samples.std(axis=0),
            y_pred_hdi_low=np.percentile(samples, 3, axis=0),
            y_pred_hdi_high=np.percentile(samples, 97, axis=0),
            y_pred_samples=samples,
        )

        assert results.n_samples == 100
        assert results.n_obs == 52


class TestContributionResults:
    """Tests for ContributionResults dataclass."""

    def test_summary(self):
        """Test summary method."""
        channel_contrib = pd.DataFrame(
            {
                "TV": np.random.randn(52),
                "Digital": np.random.randn(52),
            }
        )
        total_contrib = pd.Series({"TV": 1000.0, "Digital": 500.0})
        contrib_pct = pd.Series({"TV": 66.7, "Digital": 33.3})

        results = ContributionResults(
            channel_contributions=channel_contrib,
            total_contributions=total_contrib,
            contribution_pct=contrib_pct,
            baseline_prediction=np.random.randn(52),
            counterfactual_predictions={},
        )

        summary = results.summary()

        assert "Channel" in summary.columns
        assert "Total Contribution" in summary.columns
        assert "Contribution %" in summary.columns
        assert len(summary) == 2


class TestComponentDecomposition:
    """Tests for ComponentDecomposition dataclass."""

    def test_summary(self):
        """Test summary method."""
        decomp = ComponentDecomposition(
            intercept=np.ones(52) * 500,
            trend=np.linspace(0, 100, 52),
            seasonality=np.sin(np.linspace(0, 4 * np.pi, 52)) * 50,
            media_total=np.ones(52) * 200,
            media_by_channel=pd.DataFrame(
                {
                    "TV": np.ones(52) * 120,
                    "Digital": np.ones(52) * 80,
                }
            ),
            controls_total=np.ones(52) * 50,
            controls_by_var=pd.DataFrame(
                {
                    "Price": np.ones(52) * 50,
                }
            ),
            geo_effects=None,
            product_effects=None,
            total_intercept=500 * 52,
            total_trend=100 * 52 / 2,
            total_seasonality=0.0,
            total_media=200 * 52,
            total_controls=50 * 52,
            total_geo=None,
            total_product=None,
            y_mean=1000.0,
            y_std=100.0,
        )

        summary = decomp.summary()

        assert "Component" in summary.columns
        assert "Total Contribution" in summary.columns
        assert "Contribution %" in summary.columns
        assert "Base (Intercept)" in summary["Component"].values

    def test_media_summary(self):
        """Test media_summary method."""
        decomp = ComponentDecomposition(
            intercept=np.ones(10) * 100,
            trend=np.zeros(10),
            seasonality=np.zeros(10),
            media_total=np.ones(10) * 50,
            media_by_channel=pd.DataFrame(
                {
                    "TV": np.ones(10) * 30,
                    "Digital": np.ones(10) * 20,
                }
            ),
            controls_total=np.zeros(10),
            controls_by_var=None,
            geo_effects=None,
            product_effects=None,
            total_intercept=1000.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=500.0,
            total_controls=0.0,
            total_geo=None,
            total_product=None,
            y_mean=100.0,
            y_std=10.0,
        )

        media_summary = decomp.media_summary()

        assert len(media_summary) == 2
        assert "Share of Media %" in media_summary.columns


# =============================================================================
# BayesianMMM Tests - Initialization
# =============================================================================


class TestBayesianMMMInit:
    """Tests for BayesianMMM initialization."""

    def test_basic_init(self, simple_panel, model_config, trend_config):
        """Test basic model initialization."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.panel == simple_panel
        assert mmm.model_config == model_config
        assert mmm.trend_config == trend_config

    def test_data_standardization(self, simple_panel, model_config, trend_config):
        """Test that data is standardized."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # y should be standardized
        assert hasattr(mmm, "y_mean")
        assert hasattr(mmm, "y_std")
        assert mmm.y_mean != 0 or mmm.y_std != 1  # At least one should differ

    def test_channel_names(self, simple_panel, model_config, trend_config):
        """Test that channel names are extracted."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.channel_names == ["TV", "Digital"]

    def test_control_names(self, simple_panel, model_config, trend_config):
        """Test that control names are extracted."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.control_names == ["Price"]

    def test_n_obs(self, simple_panel, model_config, trend_config):
        """Test n_obs property."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.n_obs == 52

    def test_n_channels(self, simple_panel, model_config, trend_config):
        """Test n_channels property."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.n_channels == 2

    def test_n_controls(self, simple_panel, model_config, trend_config):
        """Test n_controls property."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.n_controls == 1

    def test_has_geo_national(self, simple_panel, model_config, trend_config):
        """Test has_geo for national data."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.has_geo is False

    def test_has_geo_panel(self, geo_panel, model_config, trend_config):
        """Test has_geo for geo-level data."""
        mmm = BayesianMMM(geo_panel, model_config, trend_config)

        assert mmm.has_geo is True

    def test_different_trend_types(self, simple_panel, model_config):
        """Test initialization with different trend types."""
        for trend_type in [
            TrendType.NONE,
            TrendType.LINEAR,
            TrendType.PIECEWISE,
            TrendType.SPLINE,
        ]:
            trend_config = TrendConfig(type=trend_type)
            mmm = BayesianMMM(simple_panel, model_config, trend_config)

            assert mmm.trend_config.type == trend_type


# =============================================================================
# BayesianMMM Tests - Model Building
# =============================================================================


class TestBayesianMMMModelBuilding:
    """Tests for BayesianMMM model building."""

    def test_model_property_builds_model(
        self, simple_panel, model_config, trend_config
    ):
        """Test that accessing model property builds the model."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        model = mmm.model

        assert model is not None
        # PyMC model should have free random variables
        assert len(model.free_RVs) > 0

    def test_model_has_expected_variables(
        self, simple_panel, model_config, trend_config
    ):
        """Test that model has expected random variables."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        model = mmm.model

        var_names = [v.name for v in model.free_RVs]

        # Should have intercept
        assert "intercept" in var_names
        # Should have sigma
        assert "sigma" in var_names
        # Should have beta for each channel
        assert "beta_TV" in var_names or any(
            "beta" in name and "TV" in name for name in var_names
        )

    def test_model_coords(self, simple_panel, model_config, trend_config):
        """Test that model has correct coordinates."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        model = mmm.model

        assert "obs" in model.coords
        assert "channel" in model.coords
        assert set(model.coords["channel"]) == set(["TV", "Digital"])

    def test_model_with_no_controls(self, sample_periods, model_config, trend_config):
        """Test model building with no controls."""
        coords = PanelCoordinates(
            periods=sample_periods,
            channels=["TV"],
        )

        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            ],
        )

        panel = PanelDataset(
            y=pd.Series(np.random.randn(52)),
            X_media=pd.DataFrame({"TV": np.random.randn(52)}),
            X_controls=None,
            coords=coords,
            index=sample_periods,
            config=config,
        )

        mmm = BayesianMMM(panel, model_config, trend_config)
        model = mmm.model

        # Should still build successfully
        assert model is not None


# =============================================================================
# BayesianMMM Tests - Fitting (Slow)
# =============================================================================


@pytest.mark.slow
class TestBayesianMMMFitting:
    """Tests for BayesianMMM fitting (slow tests requiring MCMC)."""

    def test_fit_returns_results(self, simple_panel, model_config, trend_config):
        """Test that fit returns MMMResults."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        results = mmm.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )

        assert isinstance(results, MMMResults)
        assert results.trace is not None

    def test_fit_diagnostics(self, simple_panel, model_config, trend_config):
        """Test that fit computes diagnostics."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        results = mmm.fit(
            draws=50,
            tune=50,
            chains=2,
            random_seed=42,
        )

        assert "divergences" in results.diagnostics
        assert "rhat_max" in results.diagnostics
        assert "ess_bulk_min" in results.diagnostics

    def test_fit_stores_trace(self, simple_panel, model_config, trend_config):
        """Test that fit stores trace in the model."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        results = mmm.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
        )

        assert mmm._trace is not None
        assert mmm._trace == results.trace


# =============================================================================
# BayesianMMM Tests - Prediction (Slow)
# =============================================================================


@pytest.mark.slow
class TestBayesianMMMPrediction:
    """Tests for BayesianMMM prediction (slow tests)."""

    @pytest.fixture
    def fitted_mmm(self, simple_panel, model_config, trend_config):
        """Create a fitted MMM for prediction tests."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        mmm.fit(draws=50, tune=50, chains=1, random_seed=42)
        return mmm

    def test_predict_returns_results(self, fitted_mmm):
        """Test that predict returns PredictionResults."""
        results = fitted_mmm.predict()

        assert isinstance(results, PredictionResults)
        assert results.y_pred_mean is not None
        assert len(results.y_pred_mean) == fitted_mmm.n_obs

    def test_predict_original_scale(self, fitted_mmm):
        """Test prediction in original scale."""
        results = fitted_mmm.predict(return_original_scale=True)

        # Predictions should be in original scale (around y_mean)
        assert (
            np.abs(results.y_pred_mean.mean() - fitted_mmm.y_mean)
            < 3 * fitted_mmm.y_std
        )

    def test_predict_standardized_scale(self, fitted_mmm):
        """Test prediction in standardized scale."""
        results = fitted_mmm.predict(return_original_scale=False)

        # Predictions should be standardized (around 0)
        assert np.abs(results.y_pred_mean.mean()) < 3

    def test_predict_hdi(self, fitted_mmm):
        """Test that HDI is computed correctly."""
        results = fitted_mmm.predict(hdi_prob=0.94)

        # HDI should bracket the mean
        assert np.all(results.y_pred_hdi_low <= results.y_pred_mean)
        assert np.all(results.y_pred_mean <= results.y_pred_hdi_high)


# =============================================================================
# BayesianMMM Tests - Prior Predictive
# =============================================================================


class TestBayesianMMMPriorPredictive:
    """Tests for prior predictive sampling."""

    def test_sample_prior_predictive(self, simple_panel, model_config, trend_config):
        """Test prior predictive sampling."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        prior = mmm.sample_prior_predictive(samples=50)

        assert prior is not None
        assert "prior_predictive" in prior.groups()

    def test_prior_predictive_shape(self, simple_panel, model_config, trend_config):
        """Test prior predictive output shape."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        prior = mmm.sample_prior_predictive(samples=100)

        y_prior = prior.prior_predictive["y_obs"]
        # Should have shape (chains, samples, n_obs)
        assert y_prior.shape[-1] == mmm.n_obs


# =============================================================================
# BayesianMMM Tests - Summary
# =============================================================================


@pytest.mark.slow
class TestBayesianMMMSummary:
    """Tests for BayesianMMM summary method."""

    def test_summary_requires_fit(self, simple_panel, model_config, trend_config):
        """Test that summary raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.summary()

    def test_summary_returns_dataframe(self, simple_panel, model_config, trend_config):
        """Test that summary returns DataFrame."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        mmm.fit(draws=50, tune=50, chains=1, random_seed=42)

        summary = mmm.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "sd" in summary.columns

    def test_summary_var_names(self, simple_panel, model_config, trend_config):
        """Test summary with specific var_names."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        mmm.fit(draws=50, tune=50, chains=1, random_seed=42)

        summary = mmm.summary(var_names=["intercept", "sigma"])

        assert "intercept" in summary.index
        assert "sigma" in summary.index


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.slow
class TestModelIntegration:
    """Integration tests for the full modeling workflow."""

    def test_full_workflow_national(self, simple_panel, model_config):
        """Test complete workflow for national model."""
        # Configure
        trend_config = TrendConfig(type=TrendType.LINEAR)

        # Build
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Check model structure
        assert mmm.n_obs == 52
        assert mmm.n_channels == 2

        # Fit
        results = mmm.fit(draws=50, tune=50, chains=1, random_seed=42)

        # Verify results
        assert results.diagnostics["divergences"] >= 0
        # assert results.diagnostics["rhat_max"] > 0

        # Predict
        pred = mmm.predict()
        assert len(pred.y_pred_mean) == 52

    def test_full_workflow_geo(self, geo_panel, model_config):
        """Test complete workflow for geo-level model."""
        trend_config = TrendConfig(type=TrendType.LINEAR)

        mmm = BayesianMMM(geo_panel, model_config, trend_config)

        # Check panel structure
        assert mmm.has_geo is True
        assert mmm.n_obs == 52 * 3

        # Fit
        results = mmm.fit(draws=50, tune=50, chains=1, random_seed=42)

        assert results.trace is not None


# =============================================================================
# BayesianMMM Tests - _get_time_mask
# =============================================================================


class TestBayesianMMMTimeMask:
    """Tests for BayesianMMM _get_time_mask method."""

    def test_time_mask_none_returns_all_true(self, simple_panel, model_config, trend_config):
        """Test that time_period=None returns all True mask."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        mask = mmm._get_time_mask(None)

        assert mask.dtype == bool
        assert len(mask) == mmm.n_obs
        assert mask.all()

    def test_time_mask_with_range(self, simple_panel, model_config, trend_config):
        """Test time mask with specific range."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Test first 10 time indices
        mask = mmm._get_time_mask((0, 9))

        assert mask.dtype == bool
        assert len(mask) == mmm.n_obs
        # Should have 10 True values for time indices 0-9
        assert mask.sum() == 10

    def test_time_mask_middle_range(self, simple_panel, model_config, trend_config):
        """Test time mask with middle range."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Test middle range (10-20)
        mask = mmm._get_time_mask((10, 20))

        assert mask.sum() == 11  # Inclusive on both ends

    def test_time_mask_end_range(self, simple_panel, model_config, trend_config):
        """Test time mask with range at end."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Test last 5 periods
        mask = mmm._get_time_mask((47, 51))

        assert mask.sum() == 5

    def test_time_mask_geo_panel(self, geo_panel, model_config, trend_config):
        """Test time mask with geo panel (multi-index)."""
        mmm = BayesianMMM(geo_panel, model_config, trend_config)

        # First 5 time indices - should select observations across all geos
        mask = mmm._get_time_mask((0, 4))

        # 5 time periods * 3 geographies = 15 observations
        assert mask.sum() == 15


# =============================================================================
# BayesianMMM Tests - Error Handling
# =============================================================================


class TestBayesianMMMErrorHandling:
    """Tests for BayesianMMM error handling."""

    def test_predict_requires_fit(self, simple_panel, model_config, trend_config):
        """Test that predict raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.predict()

    def test_compute_counterfactual_requires_fit(
        self, simple_panel, model_config, trend_config
    ):
        """Test that compute_counterfactual_contributions raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.compute_counterfactual_contributions()

    def test_compute_marginal_requires_fit(
        self, simple_panel, model_config, trend_config
    ):
        """Test that compute_marginal_contributions raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.compute_marginal_contributions()

    def test_what_if_scenario_requires_fit(
        self, simple_panel, model_config, trend_config
    ):
        """Test that what_if_scenario raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.what_if_scenario(spend_changes={"TV": 1.2})

    def test_compute_component_decomposition_requires_fit(
        self, simple_panel, model_config, trend_config
    ):
        """Test that compute_component_decomposition raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="not fitted"):
            mmm.compute_component_decomposition()

    def test_save_trace_only_requires_fit(
        self, simple_panel, model_config, trend_config, tmp_path
    ):
        """Test that save_trace_only raises error before fit."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        with pytest.raises(ValueError, match="No trace"):
            mmm.save_trace_only(tmp_path / "trace.nc")


# =============================================================================
# BayesianMMM Tests - Model Properties
# =============================================================================


class TestBayesianMMMProperties:
    """Tests for BayesianMMM properties."""

    def test_n_time_periods_property(self, simple_panel, model_config, trend_config):
        """Test n_periods property."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.n_periods == 52

    def test_has_product_false(self, simple_panel, model_config, trend_config):
        """Test has_product is False for simple panel."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.has_product is False

    def test_adstock_alphas_default(self, simple_panel, model_config, trend_config):
        """Test default adstock alphas."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.adstock_alphas == [0.0, 0.3, 0.5, 0.7, 0.9]

    def test_adstock_alphas_custom(self, simple_panel, model_config, trend_config):
        """Test custom adstock alphas."""
        custom_alphas = [0.2, 0.5, 0.8]
        mmm = BayesianMMM(
            simple_panel, model_config, trend_config, adstock_alphas=custom_alphas
        )

        assert mmm.adstock_alphas == custom_alphas

    def test_scaling_params_stored(self, simple_panel, model_config, trend_config):
        """Test that scaling parameters are stored."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert "y_mean" in mmm._scaling_params
        assert "y_std" in mmm._scaling_params
        assert "media_max" in mmm._scaling_params

    def test_version_attribute(self, simple_panel, model_config, trend_config):
        """Test _VERSION class attribute."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert hasattr(mmm, "_VERSION")
        assert isinstance(BayesianMMM._VERSION, str)


# =============================================================================
# BayesianMMM Tests - Data Preparation
# =============================================================================


class TestBayesianMMMDataPreparation:
    """Tests for BayesianMMM data preparation."""

    def test_media_data_is_normalized(self, simple_panel, model_config, trend_config):
        """Test that media data is normalized to [0, 1]."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Check adstocked media is normalized
        for alpha in mmm.adstock_alphas:
            media = mmm.X_media_adstocked[alpha]
            assert media.max() <= 1.0 + 1e-8
            assert media.min() >= -1e-8

    def test_y_is_standardized(self, simple_panel, model_config, trend_config):
        """Test that y is standardized."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Standardized y should have mean ~0 and std ~1
        assert np.abs(mmm.y.mean()) < 0.01
        assert np.abs(mmm.y.std() - 1.0) < 0.01

    def test_controls_are_standardized(self, simple_panel, model_config, trend_config):
        """Test that controls are standardized."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        if mmm.X_controls is not None:
            # Standardized controls should have mean ~0 and std ~1
            for col_idx in range(mmm.X_controls.shape[1]):
                col = mmm.X_controls[:, col_idx]
                assert np.abs(col.mean()) < 0.01
                assert np.abs(col.std() - 1.0) < 0.01

    def test_time_index_is_computed(self, simple_panel, model_config, trend_config):
        """Test that time_idx is computed correctly."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert len(mmm.time_idx) == mmm.n_obs
        assert mmm.time_idx.min() == 0
        assert mmm.time_idx.max() == mmm.n_periods - 1

    def test_geo_index_for_national(self, simple_panel, model_config, trend_config):
        """Test geo_idx is zeros for national data."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert len(mmm.geo_idx) == mmm.n_obs
        assert (mmm.geo_idx == 0).all()

    def test_geo_index_for_geo_panel(self, geo_panel, model_config, trend_config):
        """Test geo_idx for geo-level data."""
        mmm = BayesianMMM(geo_panel, model_config, trend_config)

        assert len(mmm.geo_idx) == mmm.n_obs
        # Should have 3 unique values for 3 geos
        assert len(np.unique(mmm.geo_idx)) == 3


# =============================================================================
# BayesianMMM Tests - Seasonality
# =============================================================================


class TestBayesianMMMSeasonality:
    """Tests for BayesianMMM seasonality preparation."""

    def test_seasonality_features_yearly(self, simple_panel, model_config, trend_config):
        """Test yearly seasonality features."""
        # Enable yearly seasonality
        model_config.seasonality = SeasonalityConfig(yearly=4)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        if "yearly" in mmm.seasonality_features:
            features = mmm.seasonality_features["yearly"]
            # 4 order = 8 features (sin + cos for each)
            assert features.shape == (mmm.n_periods, 8)

    def test_no_seasonality_when_zero(self, simple_panel, model_config, trend_config):
        """Test no seasonality features when order is 0."""
        model_config.seasonality = SeasonalityConfig(yearly=0)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Should have empty or no yearly features
        assert "yearly" not in mmm.seasonality_features or mmm.seasonality_features[
            "yearly"
        ].shape[1] == 0


# =============================================================================
# BayesianMMM Tests - Trend Types
# =============================================================================


class TestBayesianMMMTrendTypes:
    """Tests for different trend type configurations."""

    def test_trend_none(self, simple_panel, model_config):
        """Test model with no trend."""
        trend_config = TrendConfig(type=TrendType.NONE)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert mmm.trend_config.type == TrendType.NONE
        # Model should still build
        assert mmm.model is not None

    def test_trend_piecewise_features(self, simple_panel, model_config):
        """Test piecewise trend features are prepared."""
        trend_config = TrendConfig(type=TrendType.PIECEWISE, n_changepoints=5)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert "changepoints" in mmm.trend_features
        assert "changepoint_matrix" in mmm.trend_features
        assert len(mmm.trend_features["changepoints"]) == 5

    def test_trend_spline_features(self, simple_panel, model_config):
        """Test spline trend features are prepared."""
        trend_config = TrendConfig(type=TrendType.SPLINE, n_knots=8)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert "spline_basis" in mmm.trend_features
        assert "n_spline_coef" in mmm.trend_features

    def test_trend_gp_features(self, simple_panel, model_config):
        """Test GP trend features are prepared."""
        trend_config = TrendConfig(type=TrendType.GP, gp_n_basis=15)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        assert "gp_config" in mmm.trend_features
        assert mmm.trend_features["gp_config"]["n_basis"] == 15


# =============================================================================
# BayesianMMM Tests - Model Coordinates
# =============================================================================


class TestBayesianMMMCoords:
    """Tests for BayesianMMM model coordinate building."""

    def test_build_coords_basic(self, simple_panel, model_config, trend_config):
        """Test basic coordinate building."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        coords = mmm._build_coords()

        assert "obs" in coords
        assert "channel" in coords
        assert len(coords["obs"]) == mmm.n_obs
        assert list(coords["channel"]) == ["TV", "Digital"]

    def test_build_coords_with_controls(self, simple_panel, model_config, trend_config):
        """Test coords include control dimension."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        coords = mmm._build_coords()

        assert "control" in coords
        assert coords["control"] == ["Price"]

    def test_build_coords_with_geo(self, geo_panel, model_config, trend_config):
        """Test coords include geo dimension."""
        mmm = BayesianMMM(geo_panel, model_config, trend_config)
        coords = mmm._build_coords()

        assert "geo" in coords
        assert len(coords["geo"]) == 3

    def test_build_coords_spline(self, simple_panel, model_config):
        """Test coords include spline index for spline trend."""
        trend_config = TrendConfig(type=TrendType.SPLINE, n_knots=5)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        coords = mmm._build_coords()

        assert "spline_idx" in coords

    def test_build_coords_piecewise(self, simple_panel, model_config):
        """Test coords include changepoint dimension for piecewise trend."""
        trend_config = TrendConfig(type=TrendType.PIECEWISE, n_changepoints=5)
        mmm = BayesianMMM(simple_panel, model_config, trend_config)
        coords = mmm._build_coords()

        assert "changepoint" in coords
        assert len(coords["changepoint"]) == 5


# =============================================================================
# ComponentDecomposition Tests - Controls Summary
# =============================================================================


class TestComponentDecompositionControlsSummary:
    """Tests for ComponentDecomposition.controls_summary method."""

    def test_controls_summary_with_controls(self):
        """Test controls_summary when controls exist."""
        decomp = ComponentDecomposition(
            intercept=np.ones(10) * 100,
            trend=np.zeros(10),
            seasonality=np.zeros(10),
            media_total=np.ones(10) * 50,
            media_by_channel=pd.DataFrame({"TV": np.ones(10) * 50}),
            controls_total=np.ones(10) * 30,
            controls_by_var=pd.DataFrame(
                {
                    "Price": np.ones(10) * 20,
                    "Promo": np.ones(10) * 10,
                }
            ),
            geo_effects=None,
            product_effects=None,
            total_intercept=1000.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=500.0,
            total_controls=300.0,
            total_geo=None,
            total_product=None,
            y_mean=100.0,
            y_std=10.0,
        )

        controls_summary = decomp.controls_summary()

        assert controls_summary is not None
        assert "Variable" in controls_summary.columns
        assert "Total Contribution" in controls_summary.columns
        assert "Share of Controls %" in controls_summary.columns
        assert len(controls_summary) == 2

    def test_controls_summary_no_controls(self):
        """Test controls_summary returns None when no controls."""
        decomp = ComponentDecomposition(
            intercept=np.ones(10) * 100,
            trend=np.zeros(10),
            seasonality=np.zeros(10),
            media_total=np.ones(10) * 50,
            media_by_channel=pd.DataFrame({"TV": np.ones(10) * 50}),
            controls_total=np.zeros(10),
            controls_by_var=None,
            geo_effects=None,
            product_effects=None,
            total_intercept=1000.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=500.0,
            total_controls=0.0,
            total_geo=None,
            total_product=None,
            y_mean=100.0,
            y_std=10.0,
        )

        controls_summary = decomp.controls_summary()

        assert controls_summary is None


# =============================================================================
# MMMResults Tests - Methods
# =============================================================================


class TestMMMResultsMethods:
    """Tests for MMMResults methods."""

    def test_summary_method(self):
        """Test MMMResults.summary method."""
        import arviz as az
        from unittest.mock import MagicMock, patch

        mock_trace = MagicMock(spec=az.InferenceData)
        mock_model = MagicMock()
        mock_panel = MagicMock()

        results = MMMResults(
            trace=mock_trace,
            model=mock_model,
            panel=mock_panel,
        )

        with patch("arviz.summary") as mock_summary:
            mock_summary.return_value = pd.DataFrame({"mean": [1.0], "sd": [0.1]})
            summary = results.summary(var_names=["intercept"])

            mock_summary.assert_called_once_with(mock_trace, var_names=["intercept"])


# =============================================================================
# BayesianMMM Tests - Prepare Media Data
# =============================================================================


class TestBayesianMMMPrepareMediaData:
    """Tests for _prepare_media_data_for_model method."""

    def test_prepare_media_data_default(self, simple_panel, model_config, trend_config):
        """Test preparing media data with defaults."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        X_low, X_high = mmm._prepare_media_data_for_model()

        assert X_low.shape == (mmm.n_obs, mmm.n_channels)
        assert X_high.shape == (mmm.n_obs, mmm.n_channels)
        # Should be normalized
        assert X_low.max() <= 1.0 + 1e-8
        assert X_high.max() <= 1.0 + 1e-8

    def test_prepare_media_data_custom(self, simple_panel, model_config, trend_config):
        """Test preparing custom media data."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        # Create custom media data
        X_media_custom = np.ones((mmm.n_obs, mmm.n_channels)) * 100

        X_low, X_high = mmm._prepare_media_data_for_model(X_media_custom)

        assert X_low.shape == (mmm.n_obs, mmm.n_channels)
        assert X_high.shape == (mmm.n_obs, mmm.n_channels)


# =============================================================================
# BayesianMMM Tests - Get Prior
# =============================================================================


class TestBayesianMMMGetPrior:
    """Tests for get_prior method."""

    def test_get_prior_returns_inference_data(
        self, simple_panel, model_config, trend_config
    ):
        """Test that get_prior returns InferenceData."""
        import arviz as az

        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        prior = mmm.get_prior(samples=10, random_seed=42)

        assert isinstance(prior, az.InferenceData)
        assert "prior" in prior.groups()

    def test_get_prior_sample_count(self, simple_panel, model_config, trend_config):
        """Test that get_prior returns correct number of samples."""
        mmm = BayesianMMM(simple_panel, model_config, trend_config)

        prior = mmm.get_prior(samples=50, random_seed=42)

        # Check prior predictive shape
        if "prior_predictive" in prior.groups():
            y_prior = prior.prior_predictive["y_obs"]
            # Last dimension should be n_obs
            assert y_prior.shape[-1] == mmm.n_obs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

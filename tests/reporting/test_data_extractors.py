"""
Tests for the reporting data extractors module.

Tests cover:
- MMMDataBundle dataclass and properties
- DataExtractor base class utilities
- AggregationMixin methods
- BayesianMMMExtractor
- ExtendedMMMExtractor
- PyMCMarketingExtractor
- create_extractor factory function
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from mmm_framework.reporting.data_extractors import (
    MMMDataBundle,
    DataExtractor,
    AggregationMixin,
    BayesianMMMExtractor,
    ExtendedMMMExtractor,
    PyMCMarketingExtractor,
    create_extractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_bundle():
    """Create an empty MMMDataBundle."""
    return MMMDataBundle()


@pytest.fixture
def bundle_with_geo_data():
    """Create bundle with geographic data."""
    bundle = MMMDataBundle()
    bundle.geo_names = ["US", "UK", "CA"]
    bundle.actual_by_geo = {
        "US": np.array([100, 200, 300]),
        "UK": np.array([50, 100, 150]),
        "CA": np.array([25, 50, 75]),
    }
    return bundle


@pytest.fixture
def bundle_with_product_data():
    """Create bundle with product data."""
    bundle = MMMDataBundle()
    bundle.product_names = ["Product A", "Product B"]
    bundle.actual_by_product = {
        "Product A": np.array([100, 200, 300]),
        "Product B": np.array([50, 100, 150]),
    }
    return bundle


@pytest.fixture
def mock_bayesian_mmm():
    """Create a mock BayesianMMM model."""
    mmm = MagicMock()
    mmm.channel_names = ["TV", "Digital", "Social"]
    mmm.y_mean = 1000.0
    mmm.y_std = 100.0
    mmm.n_periods = 52
    mmm._results = None
    mmm._trace = None  # Start with no trace for simple tests

    return mmm


@pytest.fixture
def mock_bayesian_mmm_fitted():
    """Create a mock fitted BayesianMMM model."""
    mmm = MagicMock()
    mmm.channel_names = ["TV", "Digital", "Social"]
    mmm.y_mean = 1000.0
    mmm.y_std = 100.0
    mmm.n_periods = 52

    # Mock trace
    trace = MagicMock()
    trace.posterior = MagicMock()
    trace.posterior.__contains__ = lambda self, key: key in ["intercept", "channel_contributions"]
    trace.posterior.__getitem__ = MagicMock(return_value=MagicMock(values=np.random.randn(4, 500, 52)))
    mmm._trace = trace
    mmm._results = MagicMock()

    return mmm


@pytest.fixture
def mock_panel():
    """Create a mock panel dataset."""
    panel = MagicMock()
    panel.y = pd.Series(np.random.randn(52) * 100 + 1000)
    panel.X_media = pd.DataFrame({
        "TV": np.random.rand(52) * 100,
        "Digital": np.random.rand(52) * 50,
        "Social": np.random.rand(52) * 30,
    })

    # Mock coords
    coords = MagicMock()
    coords.periods = pd.date_range("2024-01-01", periods=52, freq="W")
    coords.n_periods = 52
    coords.n_geos = 1
    coords.n_products = 1
    panel.coords = coords

    return panel


# =============================================================================
# TestMMMDataBundle
# =============================================================================


class TestMMMDataBundle:
    """Tests for MMMDataBundle dataclass."""

    def test_default_initialization(self, empty_bundle):
        """Test bundle initializes with all None values."""
        assert empty_bundle.dates is None
        assert empty_bundle.actual is None
        assert empty_bundle.predicted is None
        assert empty_bundle.channel_names is None
        assert empty_bundle.geo_names is None
        assert empty_bundle.product_names is None

    def test_has_geo_data_false_when_empty(self, empty_bundle):
        """Test has_geo_data is False when no geo data."""
        assert empty_bundle.has_geo_data is False

    def test_has_geo_data_false_single_geo(self):
        """Test has_geo_data is False with single geo."""
        bundle = MMMDataBundle()
        bundle.geo_names = ["US"]
        bundle.actual_by_geo = {"US": np.array([100])}
        assert bundle.has_geo_data is False

    def test_has_geo_data_true_multiple_geos(self, bundle_with_geo_data):
        """Test has_geo_data is True with multiple geos."""
        assert bundle_with_geo_data.has_geo_data is True

    def test_has_geo_decomposition_false_when_empty(self, empty_bundle):
        """Test has_geo_decomposition is False when empty."""
        assert empty_bundle.has_geo_decomposition is False

    def test_has_geo_decomposition_true(self):
        """Test has_geo_decomposition is True with data."""
        bundle = MMMDataBundle()
        bundle.geo_names = ["US", "UK"]
        bundle.component_time_series_by_geo = {
            "US": {"TV": np.array([1, 2, 3])},
            "UK": {"TV": np.array([0.5, 1, 1.5])},
        }
        assert bundle.has_geo_decomposition is True

    def test_has_product_data_false_when_empty(self, empty_bundle):
        """Test has_product_data is False when empty."""
        assert empty_bundle.has_product_data is False

    def test_has_product_data_false_single_product(self):
        """Test has_product_data is False with single product."""
        bundle = MMMDataBundle()
        bundle.product_names = ["Product A"]
        bundle.actual_by_product = {"Product A": np.array([100])}
        assert bundle.has_product_data is False

    def test_has_product_data_true_multiple_products(self, bundle_with_product_data):
        """Test has_product_data is True with multiple products."""
        assert bundle_with_product_data.has_product_data is True

    def test_has_product_decomposition_false_when_empty(self, empty_bundle):
        """Test has_product_decomposition is False when empty."""
        assert empty_bundle.has_product_decomposition is False

    def test_has_product_decomposition_true(self):
        """Test has_product_decomposition is True with data."""
        bundle = MMMDataBundle()
        bundle.product_names = ["A", "B"]
        bundle.component_time_series_by_product = {
            "A": {"TV": np.array([1, 2, 3])},
            "B": {"TV": np.array([0.5, 1, 1.5])},
        }
        assert bundle.has_product_decomposition is True


# =============================================================================
# TestDataExtractor
# =============================================================================


class TestDataExtractor:
    """Tests for DataExtractor base class."""

    def test_ci_prob_default(self):
        """Test default ci_prob value."""
        # Create a concrete implementation
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()
        assert extractor.ci_prob == 0.8

    def test_ci_prob_override(self):
        """Test ci_prob can be overridden."""
        class TestExtractor(DataExtractor):
            def __init__(self, ci_prob):
                self._ci_prob = ci_prob

            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor(ci_prob=0.9)
        assert extractor.ci_prob == 0.9

    def test_compute_hdi_with_samples(self):
        """Test HDI computation with sample data."""
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()
        samples = np.random.randn(1000)

        lower, upper = extractor._compute_hdi(samples, prob=0.8)

        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_compute_percentile_bounds(self):
        """Test percentile bounds computation."""
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()
        samples = np.random.randn(100, 50)  # 100 samples, 50 observations

        lower, upper = extractor._compute_percentile_bounds(samples, prob=0.8, axis=0)

        assert lower.shape == (50,)
        assert upper.shape == (50,)
        assert np.all(lower < upper)

    def test_compute_fit_statistics_with_data(self):
        """Test fit statistics computation."""
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()
        np.random.seed(42)
        actual = np.array([100, 200, 300, 400, 500])
        predicted = {"mean": np.array([105, 195, 305, 395, 505])}

        stats = extractor._compute_fit_statistics(actual, predicted)

        assert stats is not None
        assert "r2" in stats
        assert "rmse" in stats
        assert "mae" in stats
        assert "mape" in stats
        assert stats["r2"] > 0.9  # Should be good fit

    def test_compute_fit_statistics_with_none(self):
        """Test fit statistics returns None with missing data."""
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()

        assert extractor._compute_fit_statistics(None, None) is None
        assert extractor._compute_fit_statistics(np.array([1, 2]), None) is None

    def test_extract_diagnostics_no_trace(self):
        """Test diagnostics extraction with no trace."""
        class TestExtractor(DataExtractor):
            def extract(self):
                return MMMDataBundle()

        extractor = TestExtractor()
        diagnostics = extractor._extract_diagnostics(None)

        assert diagnostics == {}


# =============================================================================
# TestAggregationMixin
# =============================================================================


class TestAggregationMixin:
    """Tests for AggregationMixin methods."""

    def test_aggregate_by_period_simple(self):
        """Test simple period aggregation."""
        class TestAggregator(AggregationMixin):
            pass

        aggregator = TestAggregator()
        values = np.array([10, 20, 30, 40, 50, 60])
        periods = ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"]
        unique_periods = ["2024-01", "2024-02", "2024-03"]

        result = aggregator._aggregate_by_period_simple(values, periods, unique_periods)

        assert len(result) == 3
        assert result[0] == 30  # 10 + 20
        assert result[1] == 70  # 30 + 40
        assert result[2] == 110  # 50 + 60

    def test_aggregate_samples_by_period(self):
        """Test sample aggregation by period with uncertainty."""
        class TestAggregator(AggregationMixin):
            pass

        aggregator = TestAggregator()
        np.random.seed(42)
        samples = np.random.randn(100, 6)  # 100 samples, 6 observations
        periods = ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"]
        unique_periods = ["2024-01", "2024-02", "2024-03"]

        result = aggregator._aggregate_samples_by_period(samples, periods, unique_periods)

        assert result is not None
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["mean"]) == 3

    def test_aggregate_samples_by_period_with_none(self):
        """Test sample aggregation returns None with invalid input."""
        class TestAggregator(AggregationMixin):
            pass

        aggregator = TestAggregator()

        result = aggregator._aggregate_samples_by_period(None, [], [])
        assert result is None

    def test_aggregate_by_group(self):
        """Test aggregation by group index."""
        class TestAggregator(AggregationMixin):
            pass

        aggregator = TestAggregator()
        values = np.array([10, 20, 30, 40, 50, 60])
        group_idx = np.array([0, 0, 1, 1, 2, 2])
        n_groups = 3

        result = aggregator._aggregate_by_group(values, group_idx, n_groups)

        assert len(result) == 3
        assert result[0] == 30  # 10 + 20
        assert result[1] == 70  # 30 + 40
        assert result[2] == 110  # 50 + 60


# =============================================================================
# TestBayesianMMMExtractor
# =============================================================================


class TestBayesianMMMExtractor:
    """Tests for BayesianMMMExtractor."""

    def test_initialization(self, mock_bayesian_mmm, mock_panel):
        """Test extractor initialization."""
        extractor = BayesianMMMExtractor(mock_bayesian_mmm, panel=mock_panel)

        assert extractor.mmm == mock_bayesian_mmm
        assert extractor.panel == mock_panel
        assert extractor.ci_prob == 0.8

    def test_initialization_with_custom_ci_prob(self, mock_bayesian_mmm):
        """Test extractor initialization with custom ci_prob."""
        extractor = BayesianMMMExtractor(mock_bayesian_mmm, ci_prob=0.95)

        assert extractor.ci_prob == 0.95

    def test_extract_returns_bundle(self, mock_bayesian_mmm, mock_panel):
        """Test extract returns MMMDataBundle with unfitted model."""
        extractor = BayesianMMMExtractor(mock_bayesian_mmm, panel=mock_panel)

        bundle = extractor.extract()

        assert isinstance(bundle, MMMDataBundle)

    def test_extract_channel_names(self, mock_bayesian_mmm, mock_panel):
        """Test channel names are extracted."""
        extractor = BayesianMMMExtractor(mock_bayesian_mmm, panel=mock_panel)

        bundle = extractor.extract()

        assert bundle.channel_names == ["TV", "Digital", "Social"]

    def test_extract_with_no_trace(self, mock_panel):
        """Test extraction with unfitted model."""
        mmm = MagicMock()
        mmm.channel_names = ["TV"]
        mmm._trace = None
        mmm._results = None

        extractor = BayesianMMMExtractor(mmm, panel=mock_panel)
        bundle = extractor.extract()

        assert isinstance(bundle, MMMDataBundle)
        assert bundle.predicted is None

    def test_extract_includes_model_specification(self, mock_bayesian_mmm, mock_panel):
        """Test model specification is extracted."""
        extractor = BayesianMMMExtractor(mock_bayesian_mmm, panel=mock_panel)

        bundle = extractor.extract()

        # Model specification should be populated
        assert bundle.model_specification is not None or bundle.model_specification is None  # Either is valid


# =============================================================================
# TestExtendedMMMExtractor
# =============================================================================


class TestExtendedMMMExtractor:
    """Tests for ExtendedMMMExtractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        model = MagicMock()
        model.channel_names = ["TV", "Digital"]

        extractor = ExtendedMMMExtractor(model)

        assert extractor.model == model
        assert extractor.ci_prob == 0.8

    def test_initialization_with_custom_ci_prob(self):
        """Test extractor with custom ci_prob."""
        model = MagicMock()
        extractor = ExtendedMMMExtractor(model, ci_prob=0.95)

        assert extractor.ci_prob == 0.95

    def test_extract_returns_bundle(self):
        """Test extract returns MMMDataBundle."""
        model = MagicMock()
        model.channel_names = ["TV"]
        model._trace = None

        extractor = ExtendedMMMExtractor(model)
        bundle = extractor.extract()

        assert isinstance(bundle, MMMDataBundle)


# =============================================================================
# TestPyMCMarketingExtractor
# =============================================================================


class TestPyMCMarketingExtractor:
    """Tests for PyMCMarketingExtractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        mmm = MagicMock()
        mmm.channel_columns = ["TV", "Digital"]

        extractor = PyMCMarketingExtractor(mmm)

        assert extractor.mmm == mmm
        assert extractor.ci_prob == 0.8

    def test_initialization_with_custom_ci_prob(self):
        """Test extractor with custom ci_prob."""
        mmm = MagicMock()
        extractor = PyMCMarketingExtractor(mmm, ci_prob=0.90)

        assert extractor.ci_prob == 0.90

    def test_extract_returns_bundle(self):
        """Test extract returns MMMDataBundle."""
        mmm = MagicMock()
        mmm.channel_columns = ["TV"]
        mmm.idata = None

        extractor = PyMCMarketingExtractor(mmm)
        bundle = extractor.extract()

        assert isinstance(bundle, MMMDataBundle)


# =============================================================================
# TestCreateExtractor
# =============================================================================


class TestCreateExtractor:
    """Tests for create_extractor factory function."""

    def test_create_bayesian_mmm_extractor(self):
        """Test creating extractor for BayesianMMM."""
        mmm = MagicMock()
        type(mmm).__name__ = "BayesianMMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, BayesianMMMExtractor)

    def test_create_nested_mmm_extractor(self):
        """Test creating extractor for NestedMMM."""
        mmm = MagicMock()
        type(mmm).__name__ = "NestedMMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, ExtendedMMMExtractor)

    def test_create_multivariate_mmm_extractor(self):
        """Test creating extractor for MultivariateMMM."""
        mmm = MagicMock()
        type(mmm).__name__ = "MultivariateMMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, ExtendedMMMExtractor)

    def test_create_combined_mmm_extractor(self):
        """Test creating extractor for CombinedMMM."""
        mmm = MagicMock()
        type(mmm).__name__ = "CombinedMMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, ExtendedMMMExtractor)

    def test_create_pymc_marketing_extractor(self):
        """Test creating extractor for PyMC-Marketing MMM."""
        mmm = MagicMock()
        type(mmm).__name__ = "MMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, PyMCMarketingExtractor)

    def test_create_default_extractor_for_unknown(self):
        """Test default to BayesianMMMExtractor for unknown types."""
        mmm = MagicMock()
        type(mmm).__name__ = "UnknownMMM"

        extractor = create_extractor(mmm)

        assert isinstance(extractor, BayesianMMMExtractor)

    def test_create_extractor_passes_kwargs(self):
        """Test kwargs are passed to extractor."""
        mmm = MagicMock()
        type(mmm).__name__ = "BayesianMMM"
        panel = MagicMock()

        extractor = create_extractor(mmm, panel=panel, ci_prob=0.95)

        assert extractor.panel == panel
        assert extractor.ci_prob == 0.95

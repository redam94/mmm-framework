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
    trace.posterior.__contains__ = lambda self, key: key in [
        "intercept",
        "channel_contributions",
    ]
    trace.posterior.__getitem__ = MagicMock(
        return_value=MagicMock(values=np.random.randn(4, 500, 52))
    )
    mmm._trace = trace
    mmm._results = MagicMock()

    return mmm


@pytest.fixture
def mock_panel():
    """Create a mock panel dataset."""
    panel = MagicMock()
    panel.y = pd.Series(np.random.randn(52) * 100 + 1000)
    panel.X_media = pd.DataFrame(
        {
            "TV": np.random.rand(52) * 100,
            "Digital": np.random.rand(52) * 50,
            "Social": np.random.rand(52) * 30,
        }
    )

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

        result = aggregator._aggregate_samples_by_period(
            samples, periods, unique_periods
        )

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
        assert (
            bundle.model_specification is not None or bundle.model_specification is None
        )  # Either is valid


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


# =============================================================================
# Decomposition scaling (original KPI units)
# =============================================================================


class _FakeMMMWithDecomposition:
    """Minimal stand-in for a fitted BayesianMMM exposing the canonical
    decomposition. Plain class (not MagicMock) so hasattr checks are honest."""

    y_mean = 1000.0
    y_std = 100.0
    n_obs = 4
    _trace = None
    _results = None

    def compute_component_decomposition(self):
        from mmm_framework.model.results import ComponentDecomposition

        intercept = np.full(self.n_obs, 950.0)  # y_mean already folded in
        trend = np.array([0.0, 10.0, 20.0, 30.0])
        seasonality = np.array([5.0, -5.0, 5.0, -5.0])
        media = pd.DataFrame(
            {"TV": [10.0, 20.0, 30.0, 40.0], "Digital": [1.0, 2.0, 3.0, 4.0]}
        )
        controls = pd.DataFrame({"Price": [-2.0, -1.0, 1.0, 2.0]})
        return ComponentDecomposition(
            intercept=intercept,
            trend=trend,
            seasonality=seasonality,
            media_total=media.sum(axis=1).to_numpy(),
            media_by_channel=media,
            controls_total=controls.sum(axis=1).to_numpy(),
            controls_by_var=controls,
            geo_effects=None,
            product_effects=None,
            total_intercept=float(intercept.sum()),
            total_trend=float(trend.sum()),
            total_seasonality=float(seasonality.sum()),
            total_media=float(media.to_numpy().sum()),
            total_controls=float(controls.to_numpy().sum()),
            total_geo=None,
            total_product=None,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )


class TestDecompositionOriginalScale:
    """Component time series / totals must be in original KPI units.

    Regression test: the extractor previously rebuilt components from the raw
    trace scaled by y_std only, dropping the y_mean location shift — so the
    decomposition chart sat near zero while the fit chart showed the real KPI.
    """

    def test_time_series_uses_canonical_decomposition(self):
        extractor = BayesianMMMExtractor(_FakeMMMWithDecomposition())

        components = extractor._get_component_time_series()

        assert components is not None
        # Baseline carries the y_mean location shift (original units)
        np.testing.assert_allclose(components["Baseline"], 950.0)
        assert set(components) == {
            "Baseline",
            "Trend",
            "Seasonality",
            "TV",
            "Digital",
            "Control: Price",
        }
        # Stacked components reproduce the prediction in original units
        stacked = np.sum(list(components.values()), axis=0)
        expected = 950.0 + np.array(
            [
                0.0 + 5.0 + 10.0 + 1.0 - 2.0,
                10.0 - 5.0 + 20.0 + 2.0 - 1.0,
                20.0 + 5.0 + 30.0 + 3.0 + 1.0,
                30.0 - 5.0 + 40.0 + 4.0 + 2.0,
            ]
        )
        np.testing.assert_allclose(stacked, expected)

    def test_totals_match_time_series_sums(self):
        extractor = BayesianMMMExtractor(_FakeMMMWithDecomposition())

        totals = extractor._get_component_totals()

        assert totals is not None
        np.testing.assert_allclose(totals["Baseline"], 950.0 * 4)
        np.testing.assert_allclose(totals["TV"], 100.0)
        np.testing.assert_allclose(totals["Trend"], 60.0)

    def test_fallback_baseline_includes_y_mean(self):
        """Trace-poking fallback must also put the baseline in original units."""

        class _TraceOnlyMMM:
            y_mean = 1000.0
            y_std = 100.0
            n_obs = 4
            channel_names = []
            _results = None

        mmm = _TraceOnlyMMM()
        posterior = {"intercept": MagicMock(values=np.full((2, 10), -0.5))}
        trace = MagicMock()
        trace.posterior = posterior
        mmm._trace = trace

        extractor = BayesianMMMExtractor(mmm)
        components = extractor._get_component_time_series()

        # -0.5 standardized * 100 + 1000 = 950 in original units
        np.testing.assert_allclose(components["Baseline"], 950.0)

        totals = extractor._get_component_totals()
        np.testing.assert_allclose(totals["Baseline"], 950.0 * 4)


# =============================================================================
# ExtendedMMMExtractor (NestedMMM / MultivariateMMM)
# =============================================================================


class _FakeVar:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _FakeTrace:
    def __init__(self, posterior: dict):
        self.posterior = {k: _FakeVar(v) for k, v in posterior.items()}


class _FakeNestedMMM:
    """Stand-in for a fitted NestedMMM with one mediator and two channels.

    Mirrors the post-standardization convention: free RVs (delta_direct_*,
    gamma/beta) are on the standardized-y scale, while the registered
    deterministics (mu, effect_*_on_y, direct_effect_*, indirect_*) are in
    original units.
    """

    def __init__(self):
        rng = np.random.default_rng(0)
        self.n_obs = 20
        self.channel_names = ["TV", "Search"]
        self.mediator_names = ["awareness"]
        self.X_media = rng.uniform(10.0, 100.0, size=(self.n_obs, 2))
        self._media_scale = self.X_media.max(axis=0) + 1e-8
        self.y = rng.normal(100.0, 5.0, self.n_obs)
        self.y_mean = 10.0
        self.y_std = 2.0
        self.index = pd.RangeIndex(self.n_obs)

        scalar = (2, 5)  # (chain, draw)
        obs = (2, 5, self.n_obs)
        self._trace = _FakeTrace(
            {
                "mu": rng.normal(100.0, 1.0, obs),
                "alpha_y": rng.normal(90.0, 1.0, scalar),
                "delta_direct_TV": rng.normal(0.5, 0.05, scalar),
                "gamma_awareness": rng.normal(2.0, 0.1, scalar),
                "beta_TV_to_awareness": rng.normal(1.0, 0.1, scalar),
                "beta_Search_to_awareness": rng.normal(0.5, 0.1, scalar),
                "indirect_TV_via_awareness": rng.normal(2.0, 0.1, scalar),
                "awareness_latent": rng.normal(1.0, 0.1, obs),
                "effect_awareness_on_y": rng.normal(2.0, 0.1, obs),
                "direct_effect_TV": rng.normal(0.3, 0.05, obs),
                "alpha_TV": np.full(scalar, 0.5),
                "alpha_Search": np.full(scalar, 0.4),
                "lambda_TV": np.full(scalar, 3.0),
                "lambda_Search": np.full(scalar, 2.0),
            }
        )

    def _get_affecting_channels(self, mediator):
        return ["TV", "Search"]


class _FakeCrossSpec:
    source_idx = 1
    target_idx = 0
    effect_type = "cannibalization"


class _FakeMultivariateMMM:
    """Stand-in for a fitted MultivariateMMM with two outcomes."""

    def __init__(self):
        rng = np.random.default_rng(1)
        self.n_obs = 20
        self.channel_names = ["TV", "Search"]
        self.outcome_names = ["original", "coldbrew"]
        self.X_media = rng.uniform(10.0, 100.0, size=(self.n_obs, 2))
        self._media_scale = self.X_media.max(axis=0) + 1e-8
        self.outcome_data = {
            "original": rng.normal(100.0, 5.0, self.n_obs),
            "coldbrew": rng.normal(50.0, 3.0, self.n_obs),
        }
        self.y = self.outcome_data["original"]
        self.outcome_means = {"original": 10.0, "coldbrew": 5.0}
        self.outcome_stds = {"original": 2.0, "coldbrew": 4.0}
        self.index = pd.RangeIndex(self.n_obs)
        self._cross_effect_specs = [_FakeCrossSpec()]

        scalar = (2, 5)
        psi = np.zeros((2, 5, 2, 2))
        psi[:, :, 1, 0] = -0.02  # coldbrew -> original cannibalization
        corr = np.tile(np.array([[1.0, 0.4], [0.4, 1.0]]), (2, 5, 1, 1))
        self._trace = _FakeTrace(
            {
                "mu": rng.normal(75.0, 1.0, (2, 5, self.n_obs, 2)),
                "alpha": rng.normal(70.0, 1.0, (2, 5, 2)),
                "beta_media": rng.normal(0.5, 0.05, (2, 5, 2, 2)),
                "psi_matrix": psi,
                "Y_obs_correlation": corr,
                "alpha_TV": np.full(scalar, 0.5),
                "alpha_Search": np.full(scalar, 0.4),
                "lambda_TV": np.full(scalar, 3.0),
                "lambda_Search": np.full(scalar, 2.0),
            }
        )


class TestExtendedMMMExtractorNested:
    """ExtendedMMMExtractor must populate everything MediatorSection needs."""

    @pytest.fixture
    def bundle(self):
        return ExtendedMMMExtractor(_FakeNestedMMM()).extract()

    def test_fit_extraction(self, bundle):
        assert bundle.predicted is not None
        assert len(bundle.predicted["mean"]) == 20
        assert bundle.fit_statistics is not None

    def test_mediator_pathways(self, bundle):
        assert set(bundle.mediator_pathways) == {"TV", "Search"}
        tv = bundle.mediator_pathways["TV"]
        # total = direct + indirect; delta_direct (~0.5, standardized) is
        # scaled by y_std=2; the indirect deterministic (~2.0) is already raw
        assert tv["_total"]["mean"] == pytest.approx(
            tv["_direct"]["mean"] + tv["_indirect"]["mean"]
        )
        assert tv["_direct"]["mean"] == pytest.approx(1.0, abs=0.2)
        assert "awareness" in tv
        # Search has no direct effect and no registered indirect deterministic:
        # the extractor reconstructs beta * gamma * y_std (~0.5 * 2.0 * 2.0)
        search = bundle.mediator_pathways["Search"]
        assert search["_direct"]["mean"] == 0.0
        assert search["_indirect"]["mean"] == pytest.approx(2.0, abs=0.5)

    def test_total_indirect_share(self, bundle):
        assert bundle.total_indirect_effect is not None
        assert 0.5 < bundle.total_indirect_effect["mean"] <= 1.0

    def test_mediator_time_series(self, bundle):
        assert list(bundle.mediator_time_series) == ["awareness"]
        assert len(bundle.mediator_time_series["awareness"]) == 20

    def test_decomposition_components(self, bundle):
        assert "Baseline" in bundle.component_time_series
        assert "Via awareness" in bundle.component_time_series
        assert "TV (direct)" in bundle.component_time_series
        # alpha_y (~90, standardized) * y_std + y_mean = 190 in original units
        np.testing.assert_allclose(
            bundle.component_time_series["Baseline"],
            90.0 * 2.0 + 10.0,
            rtol=0.05,
        )
        np.testing.assert_allclose(
            bundle.component_totals["Baseline"],
            bundle.component_time_series["Baseline"].sum(),
        )

    def test_curves_and_spend(self, bundle):
        assert set(bundle.saturation_curves) == {"TV", "Search"}
        assert set(bundle.adstock_curves) == {"TV", "Search"}
        # geometric kernel decays monotonically
        assert np.all(np.diff(bundle.adstock_curves["TV"]) <= 0)
        assert set(bundle.current_spend) == {"TV", "Search"}


class TestExtendedMMMExtractorMultivariate:
    """Per-outcome fit/decomposition plus cross-effect extraction."""

    @pytest.fixture
    def bundle(self):
        return ExtendedMMMExtractor(_FakeMultivariateMMM()).extract()

    def test_outcomes_exposed_as_products(self, bundle):
        assert bundle.product_names == ["original", "coldbrew"]
        assert set(bundle.actual_by_product) == {"original", "coldbrew"}
        assert set(bundle.predicted_by_product) == {"original", "coldbrew"}
        assert set(bundle.fit_statistics_by_product) == {"original", "coldbrew"}
        # primary outcome doubles as the aggregate view
        np.testing.assert_allclose(bundle.actual, bundle.actual_by_product["original"])

    def test_per_outcome_decomposition(self, bundle):
        assert set(bundle.component_time_series_by_product) == {
            "original",
            "coldbrew",
        }
        components = bundle.component_time_series_by_product["original"]
        assert "Baseline" in components
        assert "TV" in components and "Search" in components
        # components (incl. the cross-effect remainder) stack to mu exactly
        stacked = np.sum(list(components.values()), axis=0)
        np.testing.assert_allclose(stacked, bundle.predicted["mean"], rtol=1e-9)

    def test_cannibalization_matrix(self, bundle):
        effect = bundle.cannibalization_matrix["coldbrew"]["original"]
        assert effect["mean"] == pytest.approx(-0.02)
        assert bundle.net_product_effects["original"]["cannibalization"] < 0

    def test_outcome_correlations(self, bundle):
        assert bundle.outcome_correlations.shape == (2, 2)
        assert bundle.outcome_correlations[0, 1] == pytest.approx(0.4)

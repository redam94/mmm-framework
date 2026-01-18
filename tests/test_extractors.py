"""
Tests for data extractor classes and utilities.

These tests ensure that the extractor refactoring doesn't introduce breaking changes
and that the shared utilities work correctly.
"""

import numpy as np
import pandas as pd
import pytest


class TestDataExtractorBase:
    """Tests for DataExtractor base class methods."""

    def test_can_import_data_extractor(self):
        """Test that DataExtractor can be imported."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        assert DataExtractor is not None

    def test_data_extractor_is_abstract(self):
        """Test that DataExtractor cannot be instantiated directly."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        with pytest.raises(TypeError, match="abstract"):
            DataExtractor()

    def test_compute_fit_statistics_basic(self):
        """Test _compute_fit_statistics with basic data."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        # Create a concrete implementation for testing
        class TestExtractor(DataExtractor):
            def extract(self):
                return None

        extractor = TestExtractor()

        actual = np.array([100, 200, 300, 400, 500])
        predicted = {"mean": np.array([110, 190, 310, 390, 510])}

        stats = extractor._compute_fit_statistics(actual, predicted)

        assert stats is not None
        assert "r2" in stats
        assert "rmse" in stats
        assert "mae" in stats
        assert "mape" in stats
        assert stats["r2"] > 0.9  # Should be high for close predictions

    def test_compute_fit_statistics_perfect_fit(self):
        """Test _compute_fit_statistics with perfect predictions."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def extract(self):
                return None

        extractor = TestExtractor()

        actual = np.array([100, 200, 300, 400, 500])
        predicted = {"mean": actual.copy()}

        stats = extractor._compute_fit_statistics(actual, predicted)

        assert stats is not None
        assert stats["r2"] == 1.0
        assert stats["rmse"] == 0.0
        assert stats["mae"] == 0.0
        assert stats["mape"] == 0.0

    def test_compute_fit_statistics_none_inputs(self):
        """Test _compute_fit_statistics handles None inputs."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def extract(self):
                return None

        extractor = TestExtractor()

        assert extractor._compute_fit_statistics(None, {"mean": np.array([1])}) is None
        assert extractor._compute_fit_statistics(np.array([1]), None) is None
        assert extractor._compute_fit_statistics(None, None) is None

    def test_compute_fit_statistics_missing_mean(self):
        """Test _compute_fit_statistics handles missing 'mean' key."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def extract(self):
                return None

        extractor = TestExtractor()

        actual = np.array([100, 200, 300])
        predicted = {"lower": np.array([90, 190, 290])}  # No "mean" key

        assert extractor._compute_fit_statistics(actual, predicted) is None

    def test_compute_hdi_basic(self):
        """Test _compute_hdi with basic samples."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def __init__(self):
                self._ci_prob = 0.8

            def extract(self):
                return None

        extractor = TestExtractor()

        np.random.seed(42)
        samples = np.random.normal(100, 10, 1000)

        lower, upper = extractor._compute_hdi(samples)

        assert lower < 100 < upper
        assert lower > 70  # Roughly 10th percentile
        assert upper < 130  # Roughly 90th percentile

    def test_compute_percentile_bounds(self):
        """Test _compute_percentile_bounds method."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def __init__(self):
                self._ci_prob = 0.8

            def extract(self):
                return None

        extractor = TestExtractor()

        np.random.seed(42)
        samples = np.random.normal(100, 10, (1000, 5))

        lower, upper = extractor._compute_percentile_bounds(samples, axis=0)

        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert all(l < u for l, u in zip(lower, upper))

    def test_ci_prob_property(self):
        """Test ci_prob property default and override."""
        from mmm_framework.reporting.data_extractors import DataExtractor

        class TestExtractor(DataExtractor):
            def extract(self):
                return None

        extractor = TestExtractor()
        assert extractor.ci_prob == 0.8  # Default

        class TestExtractorCustom(DataExtractor):
            def __init__(self):
                self._ci_prob = 0.95

            def extract(self):
                return None

        extractor_custom = TestExtractorCustom()
        assert extractor_custom.ci_prob == 0.95


class TestAggregationMixin:
    """Tests for AggregationMixin methods."""

    def test_can_import_aggregation_mixin(self):
        """Test that AggregationMixin can be imported."""
        from mmm_framework.reporting.data_extractors import AggregationMixin

        assert AggregationMixin is not None

    def test_aggregate_by_period_simple(self):
        """Test _aggregate_by_period_simple method."""
        from mmm_framework.reporting.data_extractors import AggregationMixin

        class TestMixin(AggregationMixin):
            pass

        mixin = TestMixin()

        values = np.array([10, 20, 30, 40, 50, 60])
        periods = ["2021-01", "2021-01", "2021-02", "2021-02", "2021-03", "2021-03"]
        unique_periods = ["2021-01", "2021-02", "2021-03"]

        result = mixin._aggregate_by_period_simple(values, periods, unique_periods)

        assert len(result) == 3
        assert result[0] == 30  # 10 + 20
        assert result[1] == 70  # 30 + 40
        assert result[2] == 110  # 50 + 60

    def test_aggregate_samples_by_period(self):
        """Test _aggregate_samples_by_period method."""
        from mmm_framework.reporting.data_extractors import AggregationMixin

        class TestMixin(AggregationMixin):
            pass

        mixin = TestMixin()

        # 100 samples, 6 observations
        np.random.seed(42)
        samples = np.random.normal(100, 10, (100, 6))
        periods = ["2021-01", "2021-01", "2021-02", "2021-02", "2021-03", "2021-03"]
        unique_periods = ["2021-01", "2021-02", "2021-03"]

        result = mixin._aggregate_samples_by_period(samples, periods, unique_periods)

        assert result is not None
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["mean"]) == 3
        assert all(result["lower"] < result["mean"])
        assert all(result["mean"] < result["upper"])

    def test_aggregate_samples_by_period_empty(self):
        """Test _aggregate_samples_by_period with empty data."""
        from mmm_framework.reporting.data_extractors import AggregationMixin

        class TestMixin(AggregationMixin):
            pass

        mixin = TestMixin()

        result = mixin._aggregate_samples_by_period(None, [], [])
        assert result is None

        result = mixin._aggregate_samples_by_period(np.array([[1, 2]]), [], ["2021-01"])
        assert result is None

    def test_aggregate_by_group(self):
        """Test _aggregate_by_group method."""
        from mmm_framework.reporting.data_extractors import AggregationMixin

        class TestMixin(AggregationMixin):
            pass

        mixin = TestMixin()

        values = np.array([10, 20, 30, 40, 50])
        group_idx = np.array([0, 0, 1, 1, 2])
        n_groups = 3

        result = mixin._aggregate_by_group(values, group_idx, n_groups)

        assert len(result) == 3
        assert result[0] == 30  # 10 + 20
        assert result[1] == 70  # 30 + 40
        assert result[2] == 50  # 50


class TestMMMDataBundle:
    """Tests for MMMDataBundle dataclass."""

    def test_basic_creation(self):
        """Test basic MMMDataBundle creation."""
        from mmm_framework.reporting.data_extractors import MMMDataBundle

        bundle = MMMDataBundle()

        assert bundle.actual is None
        assert bundle.predicted is None
        assert bundle.channel_names is None

    def test_creation_with_data(self):
        """Test MMMDataBundle creation with data."""
        from mmm_framework.reporting.data_extractors import MMMDataBundle

        bundle = MMMDataBundle(
            actual=np.array([100, 200, 300]),
            predicted={"mean": np.array([110, 190, 310])},
            channel_names=["TV", "Radio", "Digital"],
        )

        assert bundle.actual is not None
        assert bundle.predicted is not None
        assert len(bundle.channel_names) == 3

    def test_has_geo_data_property(self):
        """Test has_geo_data property."""
        from mmm_framework.reporting.data_extractors import MMMDataBundle

        bundle = MMMDataBundle()
        assert not bundle.has_geo_data

        bundle = MMMDataBundle(
            geo_names=["CA", "TX", "NY"],
            actual_by_geo={"CA": np.array([100]), "TX": np.array([200]), "NY": np.array([300])},
        )
        assert bundle.has_geo_data

    def test_has_product_data_property(self):
        """Test has_product_data property."""
        from mmm_framework.reporting.data_extractors import MMMDataBundle

        bundle = MMMDataBundle()
        assert not bundle.has_product_data

        bundle = MMMDataBundle(
            product_names=["Product A", "Product B"],
            actual_by_product={"Product A": np.array([100]), "Product B": np.array([200])},
        )
        assert bundle.has_product_data


class TestBayesianMMMExtractor:
    """Tests for BayesianMMMExtractor class."""

    def test_can_import_extractor(self):
        """Test that BayesianMMMExtractor can be imported."""
        from mmm_framework.reporting.data_extractors import BayesianMMMExtractor

        assert BayesianMMMExtractor is not None

    def test_inherits_from_data_extractor(self):
        """Test that BayesianMMMExtractor inherits from DataExtractor."""
        from mmm_framework.reporting.data_extractors import (
            BayesianMMMExtractor,
            DataExtractor,
        )

        assert issubclass(BayesianMMMExtractor, DataExtractor)

    def test_inherits_from_aggregation_mixin(self):
        """Test that BayesianMMMExtractor inherits from AggregationMixin."""
        from mmm_framework.reporting.data_extractors import (
            BayesianMMMExtractor,
            AggregationMixin,
        )

        assert issubclass(BayesianMMMExtractor, AggregationMixin)

    def test_init_with_mock_model(self):
        """Test extractor initialization with mock model."""
        from mmm_framework.reporting.data_extractors import BayesianMMMExtractor

        class MockModel:
            _trace = None
            channel_names = ["TV", "Radio"]

        extractor = BayesianMMMExtractor(MockModel(), ci_prob=0.9)

        assert extractor.ci_prob == 0.9
        assert extractor.mmm is not None

    def test_has_compute_fit_statistics(self):
        """Test that extractor has _compute_fit_statistics from base class."""
        from mmm_framework.reporting.data_extractors import BayesianMMMExtractor

        class MockModel:
            _trace = None

        extractor = BayesianMMMExtractor(MockModel())

        # Should be able to call the inherited method
        actual = np.array([100, 200, 300])
        predicted = {"mean": np.array([100, 200, 300])}
        stats = extractor._compute_fit_statistics(actual, predicted)

        assert stats is not None
        assert stats["r2"] == 1.0


class TestExtendedMMMExtractor:
    """Tests for ExtendedMMMExtractor class."""

    def test_can_import_extractor(self):
        """Test that ExtendedMMMExtractor can be imported."""
        from mmm_framework.reporting.data_extractors import ExtendedMMMExtractor

        assert ExtendedMMMExtractor is not None

    def test_inherits_from_data_extractor(self):
        """Test that ExtendedMMMExtractor inherits from DataExtractor."""
        from mmm_framework.reporting.data_extractors import (
            ExtendedMMMExtractor,
            DataExtractor,
        )

        assert issubclass(ExtendedMMMExtractor, DataExtractor)

    def test_ci_prob_property(self):
        """Test ci_prob property."""
        from mmm_framework.reporting.data_extractors import ExtendedMMMExtractor

        class MockModel:
            _trace = None
            channel_names = []

        extractor = ExtendedMMMExtractor(MockModel(), ci_prob=0.95)
        assert extractor.ci_prob == 0.95


class TestPyMCMarketingExtractor:
    """Tests for PyMCMarketingExtractor class."""

    def test_can_import_extractor(self):
        """Test that PyMCMarketingExtractor can be imported."""
        from mmm_framework.reporting.data_extractors import PyMCMarketingExtractor

        assert PyMCMarketingExtractor is not None

    def test_inherits_from_data_extractor(self):
        """Test that PyMCMarketingExtractor inherits from DataExtractor."""
        from mmm_framework.reporting.data_extractors import (
            PyMCMarketingExtractor,
            DataExtractor,
        )

        assert issubclass(PyMCMarketingExtractor, DataExtractor)

    def test_ci_prob_property(self):
        """Test ci_prob property."""
        from mmm_framework.reporting.data_extractors import PyMCMarketingExtractor

        class MockModel:
            idata = None

        extractor = PyMCMarketingExtractor(MockModel(), ci_prob=0.85)
        assert extractor.ci_prob == 0.85


class TestCreateExtractor:
    """Tests for create_extractor factory function."""

    def test_can_import_factory(self):
        """Test that create_extractor can be imported."""
        from mmm_framework.reporting.data_extractors import create_extractor

        assert callable(create_extractor)

    def test_creates_bayesian_extractor(self):
        """Test factory creates BayesianMMMExtractor for BayesianMMM."""
        from mmm_framework.reporting.data_extractors import (
            create_extractor,
            BayesianMMMExtractor,
        )

        class BayesianMMM:
            _trace = None

        model = BayesianMMM()
        extractor = create_extractor(model)

        assert isinstance(extractor, BayesianMMMExtractor)

    def test_creates_extended_extractor_for_nested(self):
        """Test factory creates ExtendedMMMExtractor for NestedMMM."""
        from mmm_framework.reporting.data_extractors import (
            create_extractor,
            ExtendedMMMExtractor,
        )

        class NestedMMM:
            _trace = None

        model = NestedMMM()
        extractor = create_extractor(model)

        assert isinstance(extractor, ExtendedMMMExtractor)

    def test_creates_pymc_extractor_for_mmm(self):
        """Test factory creates PyMCMarketingExtractor for MMM class."""
        from mmm_framework.reporting.data_extractors import (
            create_extractor,
            PyMCMarketingExtractor,
        )

        class MMM:
            idata = None

        model = MMM()
        extractor = create_extractor(model)

        assert isinstance(extractor, PyMCMarketingExtractor)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_available_from_reporting(self):
        """Test that extractor classes are importable from reporting module."""
        from mmm_framework.reporting import (
            MMMDataBundle,
            DataExtractor,
            AggregationMixin,
            BayesianMMMExtractor,
            ExtendedMMMExtractor,
            PyMCMarketingExtractor,
            create_extractor,
        )

        assert MMMDataBundle is not None
        assert DataExtractor is not None
        assert AggregationMixin is not None
        assert BayesianMMMExtractor is not None
        assert ExtendedMMMExtractor is not None
        assert PyMCMarketingExtractor is not None
        assert callable(create_extractor)

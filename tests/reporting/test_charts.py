"""
Tests for the reporting charts module.

Tests cover:
- NumpyEncoder JSON encoding
- Chart utility functions
- Various chart generation functions
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.charts import (
    NumpyEncoder,
    _to_json,
    _hex_to_rgb,
    create_model_fit_chart,
    create_roi_forest_plot,
    create_stacked_area_chart,
    create_waterfall_chart,
    create_saturation_curves,
    create_adstock_chart,
    create_trace_plot,
    create_prior_posterior_chart,
    create_decomposition_chart,
)
from mmm_framework.reporting.config import (
    ReportConfig,
    ChartConfig,
    ColorScheme,
    ChannelColors,
)
from mmm_framework.reporting.data_extractors import MMMDataBundle


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Create a sample ReportConfig."""
    return ReportConfig()


@pytest.fixture
def sample_chart_config():
    """Create a sample ChartConfig."""
    return ChartConfig()


@pytest.fixture
def sample_channel_colors():
    """Create sample channel colors."""
    return ChannelColors(
        channel_colors={
            "TV": "#1f77b4",
            "Digital": "#ff7f0e",
            "Social": "#2ca02c",
        }
    )


@pytest.fixture
def sample_data_bundle():
    """Create a sample MMMDataBundle for testing."""
    n_obs = 52
    n_channels = 3
    channels = ["TV", "Digital", "Social"]

    bundle = MMMDataBundle()
    bundle.periods = pd.date_range("2024-01-01", periods=n_obs, freq="W")
    bundle.channel_names = channels
    bundle.y_actual = pd.Series(
        1000 + np.random.randn(n_obs) * 100, name="Sales"
    )
    bundle.y_predicted_mean = bundle.y_actual + np.random.randn(n_obs) * 20
    bundle.y_predicted_lower = bundle.y_predicted_mean - 50
    bundle.y_predicted_upper = bundle.y_predicted_mean + 50

    # ROI data
    bundle.roi_summary = pd.DataFrame(
        {
            "channel": channels,
            "roi_mean": [1.5, 2.0, 0.8],
            "roi_hdi_low": [1.2, 1.5, 0.5],
            "roi_hdi_high": [1.8, 2.5, 1.1],
            "spend": [100000, 50000, 30000],
            "contribution_mean": [150000, 100000, 24000],
            "prob_profitable": [0.95, 0.99, 0.60],
        }
    )

    # Channel contributions
    bundle.channel_contributions_mean = pd.DataFrame(
        {
            "TV": np.random.rand(n_obs) * 100,
            "Digital": np.random.rand(n_obs) * 50,
            "Social": np.random.rand(n_obs) * 30,
        },
        index=bundle.periods,
    )

    # Decomposition
    bundle.decomposition_summary = pd.DataFrame(
        {
            "component": ["Baseline", "TV", "Digital", "Social"],
            "total_contribution": [500000, 150000, 100000, 24000],
            "pct_of_total": [0.65, 0.19, 0.13, 0.03],
        }
    )

    # Saturation curves
    bundle.saturation_curves = {
        "TV": {
            "spend_range": np.linspace(0, 200000, 50),
            "response_mean": np.log1p(np.linspace(0, 200000, 50) / 10000),
            "response_lower": np.log1p(np.linspace(0, 200000, 50) / 10000) * 0.9,
            "response_upper": np.log1p(np.linspace(0, 200000, 50) / 10000) * 1.1,
            "current_spend": 100000,
        },
        "Digital": {
            "spend_range": np.linspace(0, 100000, 50),
            "response_mean": np.log1p(np.linspace(0, 100000, 50) / 5000),
            "response_lower": np.log1p(np.linspace(0, 100000, 50) / 5000) * 0.9,
            "response_upper": np.log1p(np.linspace(0, 100000, 50) / 5000) * 1.1,
            "current_spend": 50000,
        },
    }

    # Adstock weights
    bundle.adstock_weights = {
        "TV": np.array([0.5, 0.25, 0.125, 0.0625]),
        "Digital": np.array([0.3, 0.2, 0.1]),
    }

    # Diagnostics
    bundle.diagnostics = {
        "divergences": 0,
        "rhat_max": 1.01,
        "ess_bulk_min": 500,
        "ess_tail_min": 400,
    }

    return bundle


# =============================================================================
# TestNumpyEncoder
# =============================================================================


class TestNumpyEncoder:
    """Tests for NumpyEncoder JSON encoding."""

    def test_encode_numpy_array(self):
        """Test encoding numpy array."""
        arr = np.array([1, 2, 3])
        result = json.dumps(arr, cls=NumpyEncoder)
        assert result == "[1, 2, 3]"

    def test_encode_numpy_float_array(self):
        """Test encoding numpy float array."""
        arr = np.array([1.5, 2.5, 3.5])
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert result == [1.5, 2.5, 3.5]

    def test_encode_numpy_integer(self):
        """Test encoding numpy integer."""
        val = np.int64(42)
        result = json.dumps({"value": val}, cls=NumpyEncoder)
        assert '"value": 42' in result

    def test_encode_numpy_float(self):
        """Test encoding numpy float."""
        val = np.float64(3.14)
        result = json.loads(json.dumps({"value": val}, cls=NumpyEncoder))
        assert result["value"] == 3.14

    def test_encode_numpy_datetime64(self):
        """Test encoding numpy datetime64."""
        dt = np.datetime64("2024-01-15")
        result = json.loads(json.dumps({"date": dt}, cls=NumpyEncoder))
        assert "2024-01-15" in result["date"]

    def test_encode_datetime64_array(self):
        """Test encoding array of datetime64."""
        arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
        result = json.loads(json.dumps(arr, cls=NumpyEncoder))
        assert len(result) == 2
        assert "2024-01-01" in result[0]

    def test_encode_pandas_timestamp(self):
        """Test encoding pandas Timestamp."""
        ts = pd.Timestamp("2024-01-15 12:30:00")
        result = json.loads(json.dumps({"ts": ts}, cls=NumpyEncoder))
        assert "2024-01-15" in result["ts"]

    def test_encode_nan_to_none(self):
        """Test encoding NaN as null."""
        result = json.loads(json.dumps({"value": np.nan}, cls=NumpyEncoder))
        assert result["value"] is None

    def test_encode_regular_types_passthrough(self):
        """Test regular types pass through."""
        data = {"string": "test", "int": 42, "float": 3.14, "list": [1, 2]}
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result == data

    def test_encode_nested_structures(self):
        """Test encoding nested structures."""
        data = {
            "array": np.array([1, 2, 3]),
            "nested": {"inner_array": np.array([4.0, 5.0])},
        }
        result = json.loads(json.dumps(data, cls=NumpyEncoder))
        assert result["array"] == [1, 2, 3]
        assert result["nested"]["inner_array"] == [4.0, 5.0]


# =============================================================================
# TestChartUtilities
# =============================================================================


class TestChartUtilities:
    """Tests for chart utility functions."""

    def test_to_json_basic(self):
        """Test _to_json with basic data."""
        result = _to_json({"key": "value"})
        assert '"key": "value"' in result

    def test_to_json_with_numpy(self):
        """Test _to_json with numpy data."""
        result = _to_json({"arr": np.array([1, 2, 3])})
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2, 3]

    def test_hex_to_rgb_conversion(self):
        """Test hex to RGB conversion."""
        result = _hex_to_rgb("#FF0000")
        assert result == "rgb(255, 0, 0)"

    def test_hex_to_rgb_lowercase(self):
        """Test hex to RGB with lowercase."""
        result = _hex_to_rgb("#00ff00")
        assert result == "rgb(0, 255, 0)"

    def test_hex_to_rgb_blue(self):
        """Test hex to RGB for blue."""
        result = _hex_to_rgb("#0000FF")
        assert result == "rgb(0, 0, 255)"


# =============================================================================
# TestActualVsPredictedChart
# =============================================================================


class TestActualVsPredictedChart:
    """Tests for actual vs predicted chart."""

    def test_create_basic_chart(self, sample_data_bundle, sample_config):
        """Test creating basic chart."""
        html = create_model_fit_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_chart_includes_container(self, sample_data_bundle, sample_config):
        """Test chart includes container div."""
        html = create_model_fit_chart(sample_data_bundle, sample_config)

        assert "chart-container" in html or "<div" in html

    def test_chart_structure_valid_html(self, sample_data_bundle, sample_config):
        """Test chart produces valid HTML structure."""
        html = create_model_fit_chart(sample_data_bundle, sample_config)

        assert "<div" in html
        assert "</div>" in html

    def test_chart_with_missing_data(self, sample_config):
        """Test chart handles missing data."""
        bundle = MMMDataBundle()

        result = create_model_fit_chart(bundle, sample_config)

        # Should return empty or minimal HTML
        assert isinstance(result, str)


# =============================================================================
# TestROIForestPlot
# =============================================================================


class TestROIForestPlot:
    """Tests for ROI forest plot."""

    def test_create_basic_plot(self, sample_data_bundle, sample_config):
        """Test creating basic plot."""
        html = create_roi_forest_plot(sample_data_bundle, sample_config)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_plot_with_channel_colors(
        self, sample_data_bundle, sample_config, sample_channel_colors
    ):
        """Test plot with custom channel colors."""
        sample_config.channel_colors = sample_channel_colors

        html = create_roi_forest_plot(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_plot_with_missing_roi(self, sample_config):
        """Test plot handles missing ROI data."""
        bundle = MMMDataBundle()

        result = create_roi_forest_plot(bundle, sample_config)

        assert isinstance(result, str)


# =============================================================================
# TestStackedAreaChart
# =============================================================================


class TestStackedAreaChart:
    """Tests for stacked contributions chart."""

    def test_create_basic_chart(self, sample_data_bundle, sample_config):
        """Test creating basic chart."""
        html = create_stacked_area_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_chart_with_channel_colors(
        self, sample_data_bundle, sample_config, sample_channel_colors
    ):
        """Test chart with custom channel colors."""
        sample_config.channel_colors = sample_channel_colors

        html = create_stacked_area_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)


# =============================================================================
# TestWaterfallChart
# =============================================================================


class TestWaterfallChart:
    """Tests for waterfall chart."""

    def test_create_basic_chart(self, sample_data_bundle, sample_config):
        """Test creating basic chart."""
        html = create_waterfall_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_chart_structure(self, sample_data_bundle, sample_config):
        """Test chart structure."""
        html = create_waterfall_chart(sample_data_bundle, sample_config)

        # Should contain Plotly div
        assert "div" in html.lower() or html == ""


# =============================================================================
# TestSaturationCurves
# =============================================================================


class TestSaturationCurves:
    """Tests for saturation curves."""

    def test_create_multi_channel_grid(self, sample_data_bundle, sample_config):
        """Test creating multi-channel grid."""
        html = create_saturation_curves(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_with_missing_curves(self, sample_config):
        """Test with missing saturation curves."""
        bundle = MMMDataBundle()

        result = create_saturation_curves(bundle, sample_config)

        assert isinstance(result, str)


# =============================================================================
# TestAdstockChart
# =============================================================================


class TestAdstockChart:
    """Tests for adstock chart."""

    def test_create_basic_chart(self, sample_data_bundle, sample_config):
        """Test creating basic chart."""
        html = create_adstock_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_multi_channel_display(self, sample_data_bundle, sample_config):
        """Test multi-channel display."""
        html = create_adstock_chart(sample_data_bundle, sample_config)

        # Should produce valid output
        assert isinstance(html, str)

    def test_with_missing_weights(self, sample_config):
        """Test with missing adstock weights."""
        bundle = MMMDataBundle()

        result = create_adstock_chart(bundle, sample_config)

        assert isinstance(result, str)


# =============================================================================
# TestDiagnosticsCharts
# =============================================================================


class TestDiagnosticsCharts:
    """Tests for diagnostics charts."""

    def test_create_trace_plot(self, sample_data_bundle, sample_config):
        """Test creating trace plot."""
        # Add trace data
        sample_data_bundle.parameter_traces = {
            "alpha_TV": np.random.randn(1000),
            "alpha_Digital": np.random.randn(1000),
        }

        html = create_trace_plot(sample_data_bundle, sample_config)

        assert isinstance(html, str)

    def test_create_prior_posterior_chart(self, sample_data_bundle, sample_config):
        """Test creating prior-posterior chart."""
        # Add prior-posterior data
        sample_data_bundle.prior_posterior = {
            "alpha": {
                "prior": np.random.beta(2, 5, 1000),
                "posterior": np.random.beta(3, 5, 1000),
            }
        }

        html = create_prior_posterior_chart(sample_data_bundle, sample_config)

        assert isinstance(html, str)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestChartEdgeCases:
    """Tests for edge cases in chart generation."""

    def test_empty_data_bundle(self, sample_config):
        """Test all charts handle empty data bundle."""
        bundle = MMMDataBundle()

        # These should not raise exceptions
        create_model_fit_chart(bundle, sample_config)
        create_roi_forest_plot(bundle, sample_config)
        create_stacked_area_chart(bundle, sample_config)
        create_waterfall_chart(bundle, sample_config)
        create_saturation_curves(bundle, sample_config)
        create_adstock_chart(bundle, sample_config)

    def test_single_observation(self, sample_config):
        """Test charts with single observation."""
        bundle = MMMDataBundle()
        bundle.periods = pd.date_range("2024-01-01", periods=1, freq="W")
        bundle.y_actual = pd.Series([1000])
        bundle.y_predicted_mean = pd.Series([1010])

        result = create_model_fit_chart(bundle, sample_config)
        assert isinstance(result, str)

    def test_nan_values_in_data(self, sample_config):
        """Test charts handle NaN values."""
        bundle = MMMDataBundle()
        bundle.periods = pd.date_range("2024-01-01", periods=10, freq="W")
        bundle.y_actual = pd.Series([1000, np.nan, 1100] + [1000] * 7)
        bundle.y_predicted_mean = pd.Series([1010, 1020, np.nan] + [1000] * 7)

        result = create_model_fit_chart(bundle, sample_config)
        assert isinstance(result, str)

    def test_large_values(self, sample_config):
        """Test charts handle large values."""
        bundle = MMMDataBundle()
        bundle.periods = pd.date_range("2024-01-01", periods=10, freq="W")
        bundle.y_actual = pd.Series([1e12] * 10)
        bundle.y_predicted_mean = pd.Series([1.1e12] * 10)

        result = create_model_fit_chart(bundle, sample_config)
        assert isinstance(result, str)

    def test_negative_values(self, sample_config):
        """Test charts handle negative values."""
        bundle = MMMDataBundle()
        bundle.periods = pd.date_range("2024-01-01", periods=10, freq="W")
        bundle.y_actual = pd.Series([-100, -50, 0, 50, 100] * 2)
        bundle.y_predicted_mean = pd.Series([-90, -40, 10, 60, 110] * 2)

        result = create_model_fit_chart(bundle, sample_config)
        assert isinstance(result, str)


# =============================================================================
# Test Chart Configuration
# =============================================================================


class TestChartConfiguration:
    """Tests for chart configuration options."""

    def test_custom_color_scheme(self, sample_data_bundle):
        """Test charts with custom color scheme."""
        scheme = ColorScheme(
            primary="#FF0000",
            accent="#00FF00",
        )
        config = ReportConfig(color_scheme=scheme)

        html = create_model_fit_chart(sample_data_bundle, config)

        assert isinstance(html, str)

    def test_custom_credible_interval(self, sample_data_bundle):
        """Test charts with custom credible interval."""
        config = ReportConfig(default_credible_interval=0.8)

        html = create_roi_forest_plot(sample_data_bundle, config)

        assert isinstance(html, str)

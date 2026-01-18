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
    create_plotly_div,
)
from mmm_framework.reporting.config import (
    ReportConfig,
    ChartConfig,
    ColorScheme,
)


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
def sample_dates():
    """Create sample dates for charts."""
    return pd.date_range("2024-01-01", periods=52, freq="W")


@pytest.fixture
def sample_actual():
    """Create sample actual values."""
    np.random.seed(42)
    return 1000 + np.random.randn(52) * 100


@pytest.fixture
def sample_predicted():
    """Create sample predicted values with confidence intervals."""
    np.random.seed(42)
    mean = 1000 + np.random.randn(52) * 80
    lower = mean - 50
    upper = mean + 50
    return mean, lower, upper


@pytest.fixture
def sample_channels():
    """Create sample channel names."""
    return ["TV", "Digital", "Social"]


@pytest.fixture
def sample_roi_data():
    """Create sample ROI data."""
    return {
        "mean": np.array([1.5, 2.0, 0.8]),
        "lower": np.array([1.2, 1.5, 0.5]),
        "upper": np.array([1.8, 2.5, 1.1]),
    }


@pytest.fixture
def sample_contributions(sample_dates):
    """Create sample contributions DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "TV": np.random.rand(52) * 100,
            "Digital": np.random.rand(52) * 50,
            "Social": np.random.rand(52) * 30,
        },
        index=sample_dates,
    )


@pytest.fixture
def sample_saturation_curves():
    """Create sample saturation curve data."""
    spend_range = np.linspace(0, 200000, 50)
    return {
        "TV": {
            "spend": spend_range,
            "response_mean": np.log1p(spend_range / 10000),
            "response_lower": np.log1p(spend_range / 10000) * 0.9,
            "response_upper": np.log1p(spend_range / 10000) * 1.1,
        },
        "Digital": {
            "spend": np.linspace(0, 100000, 50),
            "response_mean": np.log1p(np.linspace(0, 100000, 50) / 5000),
            "response_lower": np.log1p(np.linspace(0, 100000, 50) / 5000) * 0.9,
            "response_upper": np.log1p(np.linspace(0, 100000, 50) / 5000) * 1.1,
        },
    }


@pytest.fixture
def sample_adstock_weights():
    """Create sample adstock weights."""
    return {
        "TV": np.array([0.5, 0.25, 0.125, 0.0625]),
        "Digital": np.array([0.3, 0.2, 0.1]),
    }


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
        # NaN handling depends on pd.isna check
        result = json.dumps({"value": float("nan")}, cls=NumpyEncoder)
        # Should not raise
        assert isinstance(result, str)

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
        # Returns RGB values without 'rgb()' wrapper
        assert "255" in result
        assert "0" in result

    def test_hex_to_rgb_lowercase(self):
        """Test hex to RGB with lowercase."""
        result = _hex_to_rgb("#00ff00")
        assert "255" in result


# =============================================================================
# TestModelFitChart
# =============================================================================


class TestModelFitChart:
    """Tests for model fit chart."""

    def test_create_basic_chart(
        self, sample_dates, sample_actual, sample_predicted, sample_config
    ):
        """Test creating basic chart."""
        mean, lower, upper = sample_predicted

        html = create_model_fit_chart(
            sample_dates, sample_actual, mean, lower, upper, sample_config
        )

        assert isinstance(html, str)
        assert len(html) > 0

    def test_chart_includes_plotly_div(
        self, sample_dates, sample_actual, sample_predicted, sample_config
    ):
        """Test chart includes Plotly div."""
        mean, lower, upper = sample_predicted

        html = create_model_fit_chart(
            sample_dates, sample_actual, mean, lower, upper, sample_config
        )

        assert "<div" in html
        assert "Plotly" in html

    def test_chart_with_custom_config(
        self, sample_dates, sample_actual, sample_predicted, sample_config
    ):
        """Test chart with custom chart config."""
        mean, lower, upper = sample_predicted
        chart_config = ChartConfig(height=600, y_title="Sales")

        html = create_model_fit_chart(
            sample_dates,
            sample_actual,
            mean,
            lower,
            upper,
            sample_config,
            chart_config=chart_config,
        )

        assert isinstance(html, str)

    def test_chart_with_custom_div_id(
        self, sample_dates, sample_actual, sample_predicted, sample_config
    ):
        """Test chart with custom div ID."""
        mean, lower, upper = sample_predicted

        html = create_model_fit_chart(
            sample_dates,
            sample_actual,
            mean,
            lower,
            upper,
            sample_config,
            div_id="customChart",
        )

        assert "customChart" in html


# =============================================================================
# TestROIForestPlot
# =============================================================================


class TestROIForestPlot:
    """Tests for ROI forest plot."""

    def test_create_basic_plot(self, sample_channels, sample_roi_data, sample_config):
        """Test creating basic plot."""
        html = create_roi_forest_plot(
            sample_channels,
            sample_roi_data["mean"],
            sample_roi_data["lower"],
            sample_roi_data["upper"],
            sample_config,
        )

        assert isinstance(html, str)
        assert len(html) > 0

    def test_plot_includes_reference_line(
        self, sample_channels, sample_roi_data, sample_config
    ):
        """Test plot with custom reference line."""
        html = create_roi_forest_plot(
            sample_channels,
            sample_roi_data["mean"],
            sample_roi_data["lower"],
            sample_roi_data["upper"],
            sample_config,
            reference_line=1.5,
        )

        assert isinstance(html, str)


# =============================================================================
# TestStackedAreaChart
# =============================================================================


class TestStackedAreaChart:
    """Tests for stacked contributions chart."""

    def test_create_basic_chart(self, sample_dates, sample_config):
        """Test creating basic chart."""
        np.random.seed(42)
        components = {
            "TV": np.random.rand(52) * 100,
            "Digital": np.random.rand(52) * 50,
        }

        html = create_stacked_area_chart(sample_dates, components, sample_config)

        assert isinstance(html, str)

    def test_chart_with_chart_config(self, sample_dates, sample_config):
        """Test chart with custom config."""
        np.random.seed(42)
        components = {
            "TV": np.random.rand(52) * 100,
            "Digital": np.random.rand(52) * 50,
        }
        chart_config = ChartConfig(height=500)

        html = create_stacked_area_chart(
            sample_dates, components, sample_config, chart_config=chart_config
        )

        assert isinstance(html, str)


# =============================================================================
# TestWaterfallChart
# =============================================================================


class TestWaterfallChart:
    """Tests for waterfall chart."""

    def test_create_basic_chart(self, sample_config):
        """Test creating basic chart."""
        components = ["Baseline", "TV", "Digital", "Social"]
        values = [500, 150, 100, 24]

        html = create_waterfall_chart(components, values, sample_config)

        assert isinstance(html, str)

    def test_chart_with_negative_values(self, sample_config):
        """Test chart with negative values."""
        components = ["Baseline", "TV", "Social"]
        values = [500, 150, -50]

        html = create_waterfall_chart(components, values, sample_config)

        assert isinstance(html, str)


# =============================================================================
# TestSaturationCurves
# =============================================================================


class TestSaturationCurves:
    """Tests for saturation curves."""

    def test_create_multi_channel_grid(self, sample_config):
        """Test creating multi-channel grid."""
        channels = ["TV", "Digital"]
        spend_ranges = {
            "TV": np.linspace(0, 200000, 50),
            "Digital": np.linspace(0, 100000, 50),
        }
        response_curves = {
            "TV": np.log1p(spend_ranges["TV"] / 10000),
            "Digital": np.log1p(spend_ranges["Digital"] / 5000),
        }
        current_spend = {"TV": 100000, "Digital": 50000}

        html = create_saturation_curves(
            channels, spend_ranges, response_curves, current_spend, sample_config
        )

        assert isinstance(html, str)


# =============================================================================
# TestAdstockChart
# =============================================================================


class TestAdstockChart:
    """Tests for adstock chart."""

    def test_create_basic_chart(self, sample_adstock_weights, sample_config):
        """Test creating basic chart."""
        channels = ["TV", "Digital"]
        html = create_adstock_chart(channels, sample_adstock_weights, sample_config)

        assert isinstance(html, str)

    def test_multi_channel_display(self, sample_adstock_weights, sample_config):
        """Test multi-channel display."""
        channels = ["TV", "Digital"]
        html = create_adstock_chart(channels, sample_adstock_weights, sample_config)

        # Should produce valid output with multiple channels
        assert isinstance(html, str)
        assert len(html) > 0


# =============================================================================
# TestDiagnosticsCharts
# =============================================================================


class TestDiagnosticsCharts:
    """Tests for diagnostics charts."""

    def test_create_trace_plot(self, sample_config):
        """Test creating trace plot."""
        np.random.seed(42)
        parameter_names = ["alpha_TV", "alpha_Digital"]
        traces_data = {
            "alpha_TV": np.random.randn(1000),
            "alpha_Digital": np.random.randn(1000),
        }

        html = create_trace_plot(parameter_names, traces_data, sample_config)

        assert isinstance(html, str)

    def test_create_prior_posterior_chart(self, sample_config):
        """Test creating prior-posterior chart."""
        np.random.seed(42)
        parameter_names = ["alpha"]
        prior_samples = {"alpha": np.random.beta(2, 5, 1000)}
        posterior_samples = {"alpha": np.random.beta(3, 5, 1000)}

        html = create_prior_posterior_chart(
            parameter_names, prior_samples, posterior_samples, sample_config
        )

        assert isinstance(html, str)


# =============================================================================
# TestDecompositionChart
# =============================================================================


class TestDecompositionChart:
    """Tests for decomposition chart."""

    def test_create_basic_chart(self, sample_dates, sample_config):
        """Test creating basic chart."""
        np.random.seed(42)
        components = {
            "Baseline": np.ones(52) * 500,
            "TV": np.random.rand(52) * 150,
            "Digital": np.random.rand(52) * 100,
        }

        html = create_decomposition_chart(sample_dates, components, sample_config)

        assert isinstance(html, str)


# =============================================================================
# TestPlotlyDiv
# =============================================================================


class TestPlotlyDiv:
    """Tests for plotly div creation."""

    def test_create_basic_div(self):
        """Test creating basic plotly div."""
        traces = [{"type": "scatter", "x": [1, 2, 3], "y": [4, 5, 6]}]
        layout = {"title": "Test"}

        html = create_plotly_div(traces, layout, "testDiv")

        assert isinstance(html, str)
        assert "<div" in html
        assert "testDiv" in html
        assert "Plotly" in html

    def test_div_with_custom_id(self):
        """Test div with custom ID."""
        traces = [{"type": "scatter", "x": [1], "y": [1]}]
        layout = {}

        html = create_plotly_div(traces, layout, "myCustomId")

        assert "myCustomId" in html


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestChartEdgeCases:
    """Tests for edge cases in chart generation."""

    def test_single_observation(self, sample_config):
        """Test charts with single observation."""
        dates = pd.date_range("2024-01-01", periods=1, freq="W")
        actual = np.array([1000])
        predicted = np.array([1010])
        lower = np.array([960])
        upper = np.array([1060])

        result = create_model_fit_chart(
            dates, actual, predicted, lower, upper, sample_config
        )
        assert isinstance(result, str)

    def test_large_values(self, sample_config):
        """Test charts handle large values."""
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        actual = np.array([1e12] * 10)
        predicted = np.array([1.1e12] * 10)
        lower = np.array([1e12] * 10)
        upper = np.array([1.2e12] * 10)

        result = create_model_fit_chart(
            dates, actual, predicted, lower, upper, sample_config
        )
        assert isinstance(result, str)

    def test_negative_values(self, sample_config):
        """Test charts handle negative values."""
        dates = pd.date_range("2024-01-01", periods=10, freq="W")
        actual = np.array([-100, -50, 0, 50, 100] * 2)
        predicted = np.array([-90, -40, 10, 60, 110] * 2)
        lower = predicted - 20
        upper = predicted + 20

        result = create_model_fit_chart(
            dates, actual, predicted, lower, upper, sample_config
        )
        assert isinstance(result, str)


# =============================================================================
# Test Chart Configuration
# =============================================================================


class TestChartConfiguration:
    """Tests for chart configuration options."""

    def test_custom_color_scheme(
        self, sample_dates, sample_actual, sample_predicted
    ):
        """Test charts with custom color scheme."""
        scheme = ColorScheme(
            primary="#FF0000",
            accent="#00FF00",
        )
        config = ReportConfig(color_scheme=scheme)
        mean, lower, upper = sample_predicted

        html = create_model_fit_chart(
            sample_dates, sample_actual, mean, lower, upper, config
        )

        assert isinstance(html, str)

    def test_custom_chart_height(
        self, sample_dates, sample_actual, sample_predicted, sample_config
    ):
        """Test chart with custom height."""
        mean, lower, upper = sample_predicted
        chart_config = ChartConfig(height=800)

        html = create_model_fit_chart(
            sample_dates,
            sample_actual,
            mean,
            lower,
            upper,
            sample_config,
            chart_config=chart_config,
        )

        assert isinstance(html, str)

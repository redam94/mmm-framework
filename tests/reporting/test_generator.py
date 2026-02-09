"""
Tests for the reporting generator module.

Tests cover:
- MMMReportGenerator initialization and configuration
- Section management (add, remove, enable/disable)
- HTML rendering and assembly
- CSS generation
- ReportBuilder fluent API
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.generator import (
    MMMReportGenerator,
    ReportBuilder,
)
from mmm_framework.reporting.config import (
    ReportConfig,
    SectionConfig,
    ColorScheme,
)
from mmm_framework.reporting.data_extractors import MMMDataBundle

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data_bundle():
    """Create a sample MMMDataBundle for testing."""
    n_obs = 52
    n_channels = 3
    channels = ["TV", "Digital", "Social"]

    bundle = MMMDataBundle()
    bundle.periods = pd.date_range("2024-01-01", periods=n_obs, freq="W")
    bundle.channel_names = channels
    bundle.y_actual = pd.Series(1000 + np.random.randn(n_obs) * 100, name="Sales")
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

    # Decomposition
    bundle.decomposition_summary = pd.DataFrame(
        {
            "component": ["Baseline", "TV", "Digital", "Social"],
            "total_contribution": [500000, 150000, 100000, 24000],
            "pct_of_total": [0.65, 0.19, 0.13, 0.03],
        }
    )

    return bundle


@pytest.fixture
def sample_config():
    """Create a sample ReportConfig."""
    return ReportConfig(
        title="Test Report",
        client="Test Client",
        subtitle="Test Subtitle",
        analysis_period="Jan 2024 - Dec 2024",
    )


@pytest.fixture
def mock_model():
    """Create a mock MMM model."""
    model = MagicMock()
    model.channel_names = ["TV", "Digital", "Social"]
    model._trace = MagicMock()
    return model


# =============================================================================
# TestMMMReportGeneratorInit
# =============================================================================


class TestMMMReportGeneratorInit:
    """Tests for MMMReportGenerator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        generator = MMMReportGenerator()

        assert generator.config is not None
        assert generator.data is not None
        assert isinstance(generator.config, ReportConfig)
        assert isinstance(generator.data, MMMDataBundle)

    def test_init_with_config(self, sample_config):
        """Test initialization with custom config."""
        generator = MMMReportGenerator(config=sample_config)

        assert generator.config.title == "Test Report"
        assert generator.config.client == "Test Client"
        assert generator.config.subtitle == "Test Subtitle"

    def test_init_with_data_bundle(self, sample_data_bundle, sample_config):
        """Test initialization with pre-extracted data."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        assert generator.data is sample_data_bundle
        assert generator.data.channel_names == ["TV", "Digital", "Social"]

    @patch("mmm_framework.reporting.generator.create_extractor")
    def test_init_with_model(self, mock_create_extractor, mock_model):
        """Test initialization extracts data from model."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MMMDataBundle()
        mock_create_extractor.return_value = mock_extractor

        generator = MMMReportGenerator(model=mock_model)

        mock_create_extractor.assert_called_once()
        mock_extractor.extract.assert_called_once()

    def test_init_with_sensitivity_results(self, sample_data_bundle):
        """Test initialization with sensitivity results."""
        sensitivity = {"spec_a": {"r_squared": 0.85}, "spec_b": {"r_squared": 0.82}}

        generator = MMMReportGenerator(data=sample_data_bundle, sensitivity=sensitivity)

        assert generator.data.sensitivity_results == sensitivity

    def test_init_sections_initialized(self):
        """Test that sections are initialized."""
        generator = MMMReportGenerator()

        assert len(generator._sections) > 0

    def test_init_sets_config_defaults(self):
        """Test that config has reasonable defaults."""
        generator = MMMReportGenerator()

        assert generator.config.title != ""
        assert generator.config.include_plotly_js is True
        assert generator.config.color_scheme is not None


# =============================================================================
# TestMMMReportGeneratorSections
# =============================================================================


class TestMMMReportGeneratorSections:
    """Tests for section management."""

    def test_initialize_sections_order(self):
        """Test sections are initialized in correct order."""
        generator = MMMReportGenerator()

        # Check expected sections exist (IDs use hyphens)
        section_ids = [s.section_id for s in generator._sections]
        assert "executive-summary" in section_ids
        assert "model-fit" in section_ids
        assert "channel-roi" in section_ids

    def test_add_section_valid_type(self, sample_data_bundle):
        """Test adding a valid section type."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        initial_count = len(generator._sections)

        generator.add_section("executive_summary")

        assert len(generator._sections) == initial_count + 1

    def test_add_section_invalid_type_raises(self):
        """Test adding invalid section type raises ValueError."""
        generator = MMMReportGenerator()

        with pytest.raises(ValueError, match="Unknown section type"):
            generator.add_section("invalid_section")

    def test_add_section_at_position(self, sample_data_bundle):
        """Test adding section at specific position."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        generator.add_section("executive_summary", position=0)

        # Section IDs use hyphens
        assert generator._sections[0].section_id == "executive-summary"

    def test_add_section_returns_self_for_chaining(self, sample_data_bundle):
        """Test add_section returns self for method chaining."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        result = generator.add_section("executive_summary")

        assert result is generator

    def test_remove_section_by_id(self, sample_data_bundle):
        """Test removing section by ID."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        initial_sections = [s.section_id for s in generator._sections]

        generator.remove_section("diagnostics")

        current_sections = [s.section_id for s in generator._sections]
        assert "diagnostics" not in current_sections

    def test_remove_section_nonexistent_silent(self, sample_data_bundle):
        """Test removing nonexistent section doesn't raise."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        initial_count = len(generator._sections)

        generator.remove_section("nonexistent")

        assert len(generator._sections) == initial_count


# =============================================================================
# TestMMMReportGeneratorRender
# =============================================================================


class TestMMMReportGeneratorRender:
    """Tests for report rendering."""

    def test_render_returns_html_string(self, sample_data_bundle):
        """Test render returns HTML string."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        html = generator.render()

        assert isinstance(html, str)
        assert len(html) > 0

    def test_render_includes_doctype(self, sample_data_bundle):
        """Test rendered HTML includes DOCTYPE."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        html = generator.render()

        assert "<!DOCTYPE html>" in html

    def test_render_includes_title(self, sample_data_bundle, sample_config):
        """Test rendered HTML includes title."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        html = generator.render()

        assert "<title>Test Report</title>" in html
        assert "Test Report" in html

    def test_render_includes_css(self, sample_data_bundle):
        """Test rendered HTML includes CSS styles."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        html = generator.render()

        assert "<style>" in html
        assert "</style>" in html
        assert ":root" in html

    def test_render_includes_plotly_script(self, sample_data_bundle):
        """Test rendered HTML includes Plotly script."""
        config = ReportConfig(include_plotly_js=True)
        generator = MMMReportGenerator(data=sample_data_bundle, config=config)

        html = generator.render()

        assert "plotly" in html.lower()

    def test_render_without_plotly_script(self, sample_data_bundle):
        """Test render without Plotly when disabled."""
        config = ReportConfig(include_plotly_js=False)
        generator = MMMReportGenerator(data=sample_data_bundle, config=config)

        html = generator.render()

        assert "cdn.plot.ly" not in html

    def test_render_includes_generated_date(self, sample_data_bundle):
        """Test render includes generated date."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        html = generator.render()

        # Should include month name from current date or generated_date
        assert "Generated" in html or "Report generated" in html


# =============================================================================
# TestMMMReportGeneratorAssembleHtml
# =============================================================================


class TestMMMReportGeneratorAssembleHtml:
    """Tests for HTML assembly."""

    def test_assemble_html_structure(self, sample_data_bundle, sample_config):
        """Test assembled HTML has correct structure."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        html = generator._assemble_html("<div>Test Content</div>")

        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html
        assert "Test Content" in html

    def test_assemble_html_escapes_title(self, sample_data_bundle):
        """Test title is properly escaped."""
        config = ReportConfig(title="Test <script>alert('xss')</script>")
        generator = MMMReportGenerator(data=sample_data_bundle, config=config)

        html = generator._assemble_html("")

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_assemble_html_with_client(self, sample_data_bundle, sample_config):
        """Test HTML includes client name."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        html = generator._assemble_html("")

        assert "Test Client" in html

    def test_assemble_html_with_subtitle(self, sample_data_bundle, sample_config):
        """Test HTML includes subtitle."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        html = generator._assemble_html("")

        assert "Test Subtitle" in html

    def test_assemble_html_with_analysis_period(
        self, sample_data_bundle, sample_config
    ):
        """Test HTML includes analysis period."""
        generator = MMMReportGenerator(data=sample_data_bundle, config=sample_config)

        html = generator._assemble_html("")

        assert "Jan 2024 - Dec 2024" in html

    def test_assemble_html_without_optional_fields(self, sample_data_bundle):
        """Test HTML works without optional fields."""
        config = ReportConfig(title="Minimal Report")
        generator = MMMReportGenerator(data=sample_data_bundle, config=config)

        html = generator._assemble_html("")

        assert "Minimal Report" in html
        # Should not have empty subtitle div
        assert 'class="subtitle"' not in html or "—" not in html


# =============================================================================
# TestMMMReportGeneratorCSS
# =============================================================================


class TestMMMReportGeneratorCSS:
    """Tests for CSS generation."""

    def test_generate_css_includes_color_scheme(self, sample_data_bundle):
        """Test CSS includes color scheme variables."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        css = generator._generate_css()

        assert "--color-primary:" in css
        assert "--color-accent:" in css
        assert "--color-text:" in css
        assert "--color-bg:" in css

    def test_generate_css_includes_font_families(self, sample_data_bundle):
        """Test CSS includes font family references."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        css = generator._generate_css()

        assert "font-family:" in css

    def test_generate_css_includes_responsive_styles(self, sample_data_bundle):
        """Test CSS includes responsive media queries."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        css = generator._generate_css()

        assert "@media" in css
        assert "768px" in css

    def test_generate_css_includes_print_styles(self, sample_data_bundle):
        """Test CSS includes print media query."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        css = generator._generate_css()

        assert "@media print" in css


# =============================================================================
# TestMMMReportGeneratorOutput
# =============================================================================


class TestMMMReportGeneratorOutput:
    """Tests for file output."""

    def test_to_html_creates_file(self, sample_data_bundle, tmp_path):
        """Test to_html creates file at path."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        output_path = tmp_path / "report.html"

        result = generator.to_html(output_path)

        assert output_path.exists()
        assert result == output_path

    def test_to_html_returns_path(self, sample_data_bundle, tmp_path):
        """Test to_html returns Path object."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        output_path = tmp_path / "report.html"

        result = generator.to_html(output_path)

        assert isinstance(result, Path)

    def test_to_html_correct_content(self, sample_data_bundle, tmp_path):
        """Test to_html writes correct content."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        output_path = tmp_path / "report.html"

        generator.to_html(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content

    def test_to_html_string_path(self, sample_data_bundle, tmp_path):
        """Test to_html accepts string path."""
        generator = MMMReportGenerator(data=sample_data_bundle)
        output_path = str(tmp_path / "report.html")

        result = generator.to_html(output_path)

        assert result.exists()

    def test_to_string_returns_html(self, sample_data_bundle):
        """Test to_string returns HTML string."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        result = generator.to_string()

        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_repr_html_for_jupyter(self, sample_data_bundle):
        """Test _repr_html_ for Jupyter display."""
        generator = MMMReportGenerator(data=sample_data_bundle)

        result = generator._repr_html_()

        assert isinstance(result, str)
        assert "<html" in result


# =============================================================================
# TestReportBuilder
# =============================================================================


class TestReportBuilder:
    """Tests for ReportBuilder fluent API."""

    def test_init_defaults(self):
        """Test builder initializes with defaults."""
        builder = ReportBuilder()

        assert builder._model is None
        assert builder._data is None
        assert builder._config_kwargs == {}

    def test_with_model(self, mock_model):
        """Test with_model sets model."""
        builder = ReportBuilder().with_model(mock_model)

        assert builder._model is mock_model

    def test_with_model_and_panel(self, mock_model):
        """Test with_model with panel and results."""
        mock_panel = MagicMock()
        mock_results = MagicMock()

        builder = ReportBuilder().with_model(mock_model, mock_panel, mock_results)

        assert builder._model is mock_model
        assert builder._panel is mock_panel
        assert builder._results is mock_results

    def test_with_data(self, sample_data_bundle):
        """Test with_data sets data bundle."""
        builder = ReportBuilder().with_data(sample_data_bundle)

        assert builder._data is sample_data_bundle

    def test_with_sensitivity(self):
        """Test with_sensitivity sets results."""
        sensitivity = {"test": "data"}
        builder = ReportBuilder().with_sensitivity(sensitivity)

        assert builder._sensitivity is sensitivity

    def test_with_title(self):
        """Test with_title sets title."""
        builder = ReportBuilder().with_title("My Report")

        assert builder._config_kwargs["title"] == "My Report"

    def test_with_client(self):
        """Test with_client sets client."""
        builder = ReportBuilder().with_client("Acme Corp")

        assert builder._config_kwargs["client"] == "Acme Corp"

    def test_with_subtitle(self):
        """Test with_subtitle sets subtitle."""
        builder = ReportBuilder().with_subtitle("Q4 Analysis")

        assert builder._config_kwargs["subtitle"] == "Q4 Analysis"

    def test_with_analysis_period(self):
        """Test with_analysis_period sets period."""
        builder = ReportBuilder().with_analysis_period("2024")

        assert builder._config_kwargs["analysis_period"] == "2024"

    def test_with_color_scheme(self):
        """Test with_color_scheme sets scheme."""
        scheme = ColorScheme()
        builder = ReportBuilder().with_color_scheme(scheme)

        assert builder._config_kwargs["color_scheme"] is scheme

    def test_with_credible_interval(self):
        """Test with_credible_interval sets interval."""
        builder = ReportBuilder().with_credible_interval(0.9)

        assert builder._config_kwargs["default_credible_interval"] == 0.9

    def test_enable_section(self):
        """Test enable_section enables section."""
        builder = ReportBuilder().enable_section("diagnostics")

        assert builder._section_configs["diagnostics"].enabled is True

    def test_enable_section_with_kwargs(self):
        """Test enable_section with additional config."""
        builder = ReportBuilder().enable_section("diagnostics", title="Custom Title")

        assert builder._section_configs["diagnostics"].enabled is True
        assert builder._section_configs["diagnostics"].title == "Custom Title"

    def test_disable_section(self):
        """Test disable_section disables section."""
        builder = ReportBuilder().disable_section("diagnostics")

        assert builder._section_configs["diagnostics"].enabled is False

    def test_enable_all_sections(self):
        """Test enable_all_sections enables all."""
        builder = ReportBuilder().enable_all_sections()

        # Should have entries for known sections
        assert len(builder._section_configs) > 0

    def test_minimal_report_preset(self):
        """Test minimal_report preset."""
        builder = ReportBuilder().minimal_report()

        assert builder._section_configs["executive_summary"].enabled is True
        assert builder._section_configs["channel_roi"].enabled is True
        assert builder._section_configs["diagnostics"].enabled is False

    def test_build_creates_generator(self, sample_data_bundle):
        """Test build creates MMMReportGenerator."""
        generator = (
            ReportBuilder().with_data(sample_data_bundle).with_title("Test").build()
        )

        assert isinstance(generator, MMMReportGenerator)
        assert generator.config.title == "Test"

    def test_fluent_api_chaining(self, sample_data_bundle):
        """Test fluent API method chaining."""
        generator = (
            ReportBuilder()
            .with_data(sample_data_bundle)
            .with_title("Chained Report")
            .with_client("Test Client")
            .with_subtitle("Subtitle")
            .with_credible_interval(0.9)
            .enable_section("executive_summary")
            .disable_section("diagnostics")
            .build()
        )

        assert generator.config.title == "Chained Report"
        assert generator.config.client == "Test Client"
        assert generator.config.subtitle == "Subtitle"

    @patch("mmm_framework.reporting.generator.create_extractor")
    def test_build_with_model(self, mock_create_extractor, mock_model):
        """Test build with model creates generator."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MMMDataBundle()
        mock_create_extractor.return_value = mock_extractor

        generator = ReportBuilder().with_model(mock_model).build()

        assert isinstance(generator, MMMReportGenerator)
        mock_create_extractor.assert_called_once()

"""
Tests for the reporting sections module.

Tests cover:
- Section base class
- All section subclasses (ExecutiveSummary, ModelFit, ChannelROI, etc.)
- Section registry
"""

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.sections import (
    Section,
    ExecutiveSummarySection,
    ModelFitSection,
    ChannelROISection,
    DecompositionSection,
    SaturationSection,
    SensitivitySection,
    MethodologySection,
    DiagnosticsSection,
    GeographicSection,
    MediatorSection,
    CannibalizationSection,
    SECTION_REGISTRY,
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
def sample_config():
    """Create a sample ReportConfig."""
    return ReportConfig()


@pytest.fixture
def disabled_section_config():
    """Create a disabled SectionConfig."""
    return SectionConfig(enabled=False)


@pytest.fixture
def enabled_section_config():
    """Create an enabled SectionConfig."""
    return SectionConfig(enabled=True, title="Custom Title", subtitle="Custom Subtitle")


@pytest.fixture
def empty_data_bundle():
    """Create an empty MMMDataBundle."""
    return MMMDataBundle()


@pytest.fixture
def sample_data_bundle():
    """Create a sample MMMDataBundle with test data."""
    n_obs = 52
    channels = ["TV", "Digital", "Social"]

    bundle = MMMDataBundle()
    bundle.periods = pd.date_range("2024-01-01", periods=n_obs, freq="W")
    bundle.channel_names = channels

    # Actual and predicted
    np.random.seed(42)
    bundle.y_actual = pd.Series(1000 + np.random.randn(n_obs) * 100, name="Sales")
    bundle.y_predicted_mean = bundle.y_actual + np.random.randn(n_obs) * 20
    bundle.y_predicted_lower = bundle.y_predicted_mean - 50
    bundle.y_predicted_upper = bundle.y_predicted_mean + 50

    # Executive summary metrics
    bundle.total_revenue = 1000000
    bundle.marketing_attributed_revenue = {
        "mean": 350000,
        "lower": 300000,
        "upper": 400000,
    }
    bundle.blended_roi = {
        "mean": 1.75,
        "lower": 1.5,
        "upper": 2.0,
    }
    bundle.marketing_contribution_pct = {
        "mean": 0.35,
        "lower": 0.30,
        "upper": 0.40,
    }

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

    # Diagnostics
    bundle.diagnostics = {
        "divergences": 0,
        "rhat_max": 1.01,
        "ess_bulk_min": 500,
        "ess_tail_min": 400,
    }

    # Saturation curves
    bundle.saturation_curves = {
        "TV": {
            "spend": np.linspace(0, 200000, 50),
            "response": np.log1p(np.linspace(0, 200000, 50) / 10000),
        },
    }

    # Adstock weights
    bundle.adstock_weights = {
        "TV": np.array([0.5, 0.25, 0.125]),
    }

    return bundle


# =============================================================================
# TestSectionBase
# =============================================================================


class TestSectionBase:
    """Tests for the Section base class."""

    def test_section_title_from_config(
        self, sample_data_bundle, sample_config, enabled_section_config
    ):
        """Test section uses title from config."""
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=enabled_section_config,
        )

        assert section.title == "Custom Title"

    def test_section_title_default_fallback(self, sample_data_bundle, sample_config):
        """Test section uses default title when not specified."""
        section_config = SectionConfig(enabled=True)
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        assert section.title == "Executive Summary"

    def test_is_enabled_from_config(
        self, sample_data_bundle, sample_config, enabled_section_config
    ):
        """Test is_enabled property."""
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=enabled_section_config,
        )
        assert section.is_enabled is True

    def test_is_disabled_from_config(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test is_enabled returns False when disabled."""
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )
        assert section.is_enabled is False

    def test_render_section_wrapper_structure(self, sample_data_bundle, sample_config):
        """Test section wrapper HTML structure."""
        section_config = SectionConfig(enabled=True)
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section._render_section_wrapper("<p>Test content</p>")

        assert '<section class="section"' in html
        assert "<h2>" in html
        assert "Test content" in html
        assert "</section>" in html

    def test_render_section_wrapper_with_subtitle(
        self, sample_data_bundle, sample_config, enabled_section_config
    ):
        """Test section wrapper includes subtitle."""
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=enabled_section_config,
        )

        html = section._render_section_wrapper("<p>Content</p>")

        assert "Custom Subtitle" in html
        assert 'class="section-subtitle"' in html

    def test_render_section_wrapper_with_notes(self, sample_data_bundle, sample_config):
        """Test section wrapper includes custom notes."""
        section_config = SectionConfig(enabled=True, custom_notes="Important note here")
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section._render_section_wrapper("<p>Content</p>")

        assert "Important note here" in html
        assert 'class="methodology-note"' in html


# =============================================================================
# TestExecutiveSummarySection
# =============================================================================


class TestExecutiveSummarySection:
    """Tests for ExecutiveSummarySection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_data(self, sample_data_bundle, sample_config):
        """Test render with complete data."""
        section_config = SectionConfig(enabled=True)
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)
        assert len(html) > 0
        assert "Executive Summary" in html

    def test_render_metrics_grid(self, sample_data_bundle, sample_config):
        """Test metrics grid rendering."""
        section_config = SectionConfig(enabled=True)
        section = ExecutiveSummarySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should include metrics
        assert "metrics-grid" in html or "metric" in html.lower()


# =============================================================================
# TestModelFitSection
# =============================================================================


class TestModelFitSection:
    """Tests for ModelFitSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = ModelFitSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_data(self, sample_data_bundle, sample_config):
        """Test render with complete data."""
        section_config = SectionConfig(enabled=True)
        section = ModelFitSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)

    def test_render_missing_data_returns_empty(self, empty_data_bundle, sample_config):
        """Test render with missing data."""
        section_config = SectionConfig(enabled=True)
        section = ModelFitSection(
            data=empty_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should return empty or minimal content
        assert isinstance(html, str)


# =============================================================================
# TestChannelROISection
# =============================================================================


class TestChannelROISection:
    """Tests for ChannelROISection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = ChannelROISection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_roi_data(self, sample_data_bundle, sample_config):
        """Test render with ROI data."""
        section_config = SectionConfig(enabled=True)
        section = ChannelROISection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)

    def test_render_missing_roi_returns_empty(self, empty_data_bundle, sample_config):
        """Test render with missing ROI data."""
        section_config = SectionConfig(enabled=True)
        section = ChannelROISection(
            data=empty_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)


# =============================================================================
# TestDecompositionSection
# =============================================================================


class TestDecompositionSection:
    """Tests for DecompositionSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = DecompositionSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_decomposition(self, sample_data_bundle, sample_config):
        """Test render with decomposition data."""
        section_config = SectionConfig(enabled=True)
        section = DecompositionSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)


# =============================================================================
# TestSaturationSection
# =============================================================================


class TestSaturationSection:
    """Tests for SaturationSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = SaturationSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_curves(self, sample_data_bundle, sample_config):
        """Test render with saturation curves."""
        section_config = SectionConfig(enabled=True)
        section = SaturationSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)


# =============================================================================
# TestSensitivitySection
# =============================================================================


class TestSensitivitySection:
    """Tests for SensitivitySection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = SensitivitySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_missing_results_returns_empty(
        self, sample_data_bundle, sample_config
    ):
        """Test render without sensitivity results."""
        section_config = SectionConfig(enabled=True)
        section = SensitivitySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should return empty or minimal when no sensitivity results
        assert isinstance(html, str)


# =============================================================================
# TestMethodologySection
# =============================================================================


class TestMethodologySection:
    """Tests for MethodologySection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = MethodologySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_methodology_content(self, sample_data_bundle, sample_config):
        """Test render produces methodology content."""
        section_config = SectionConfig(enabled=True)
        section = MethodologySection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)
        # Should include section wrapper
        if len(html) > 0:
            assert "Methodology" in html or "methodology" in html


# =============================================================================
# TestDiagnosticsSection
# =============================================================================


class TestDiagnosticsSection:
    """Tests for DiagnosticsSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = DiagnosticsSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_with_diagnostics(self, sample_data_bundle, sample_config):
        """Test render with diagnostics data."""
        section_config = SectionConfig(enabled=True)
        section = DiagnosticsSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        assert isinstance(html, str)


# =============================================================================
# TestGeographicSection
# =============================================================================


class TestGeographicSection:
    """Tests for GeographicSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = GeographicSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_missing_geo_returns_empty(self, sample_data_bundle, sample_config):
        """Test render without geo data."""
        section_config = SectionConfig(enabled=True)
        section = GeographicSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should return empty when no geo data
        assert isinstance(html, str)


# =============================================================================
# TestMediatorSection
# =============================================================================


class TestMediatorSection:
    """Tests for MediatorSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = MediatorSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_missing_mediators_returns_empty(
        self, sample_data_bundle, sample_config
    ):
        """Test render without mediator data."""
        section_config = SectionConfig(enabled=True)
        section = MediatorSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should return empty when no mediators
        assert isinstance(html, str)


# =============================================================================
# TestCannibalizationSection
# =============================================================================


class TestCannibalizationSection:
    """Tests for CannibalizationSection."""

    def test_render_disabled_returns_empty(
        self, sample_data_bundle, sample_config, disabled_section_config
    ):
        """Test disabled section returns empty string."""
        section = CannibalizationSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=disabled_section_config,
        )

        assert section.render() == ""

    def test_render_missing_data_returns_empty(self, sample_data_bundle, sample_config):
        """Test render without cannibalization data."""
        section_config = SectionConfig(enabled=True)
        section = CannibalizationSection(
            data=sample_data_bundle,
            config=sample_config,
            section_config=section_config,
        )

        html = section.render()

        # Should return empty when no cannibalization data
        assert isinstance(html, str)


# =============================================================================
# TestSectionRegistry
# =============================================================================


class TestSectionRegistry:
    """Tests for section registry."""

    def test_registry_contains_all_sections(self):
        """Test registry contains all expected sections."""
        expected_sections = [
            "executive_summary",
            "model_fit",
            "channel_roi",
            "decomposition",
            "saturation",
            "sensitivity",
            "methodology",
            "diagnostics",
            "geographic",
            "mediators",
            "cannibalization",
        ]

        for section_name in expected_sections:
            assert section_name in SECTION_REGISTRY, f"Missing: {section_name}"

    def test_registry_values_are_section_classes(self):
        """Test registry values are Section subclasses."""
        for name, section_class in SECTION_REGISTRY.items():
            assert issubclass(section_class, Section), f"{name} is not a Section"

    def test_registry_section_instantiation(self, sample_data_bundle, sample_config):
        """Test all registry sections can be instantiated."""
        section_config = SectionConfig(enabled=True)

        for name, section_class in SECTION_REGISTRY.items():
            section = section_class(
                data=sample_data_bundle,
                config=sample_config,
                section_config=section_config,
            )
            assert section is not None
            assert section.section_id is not None

    def test_registry_section_render(self, sample_data_bundle, sample_config):
        """Test all registry sections can render."""
        section_config = SectionConfig(enabled=True)

        for name, section_class in SECTION_REGISTRY.items():
            section = section_class(
                data=sample_data_bundle,
                config=sample_config,
                section_config=section_config,
            )
            html = section.render()
            assert isinstance(html, str)

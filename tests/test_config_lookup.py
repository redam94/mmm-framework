"""
Tests for config lookup consolidation.

These tests ensure that:
1. The generic _get_config_by_name helper works correctly
2. get_media_config and get_control_config maintain backward compatibility
3. The new get_variable_config method works for all variable types
"""

import pytest

from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    VariableConfig,
)


class TestMFFConfigLookup:
    """Tests for MFFConfig lookup methods."""

    @pytest.fixture
    def sample_config(self) -> MFFConfig:
        """Create a sample MFFConfig with multiple channels and controls."""
        return MFFConfig(
            kpi=KPIConfig(
                name="sales",
                dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
            ),
            media_channels=[
                MediaChannelConfig(
                    name="tv",
                    display_name="Television",
                    dimensions=[DimensionType.PERIOD],
                ),
                MediaChannelConfig(
                    name="digital",
                    display_name="Digital",
                    dimensions=[DimensionType.PERIOD],
                ),
                MediaChannelConfig(
                    name="radio",
                    display_name="Radio",
                    dimensions=[DimensionType.PERIOD],
                ),
            ],
            controls=[
                ControlVariableConfig(
                    name="price",
                    dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
                ),
                ControlVariableConfig(
                    name="distribution",
                    dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
                ),
            ],
        )

    def test_get_media_config_exists(self, sample_config: MFFConfig):
        """Test get_media_config returns correct config for existing channel."""
        result = sample_config.get_media_config("tv")
        assert result is not None
        assert result.name == "tv"
        assert result.display_name == "Television"
        assert isinstance(result, MediaChannelConfig)

    def test_get_media_config_not_exists(self, sample_config: MFFConfig):
        """Test get_media_config returns None for non-existing channel."""
        result = sample_config.get_media_config("nonexistent")
        assert result is None

    def test_get_media_config_case_sensitive(self, sample_config: MFFConfig):
        """Test get_media_config is case sensitive."""
        result = sample_config.get_media_config("TV")  # Wrong case
        assert result is None

    def test_get_control_config_exists(self, sample_config: MFFConfig):
        """Test get_control_config returns correct config for existing control."""
        result = sample_config.get_control_config("price")
        assert result is not None
        assert result.name == "price"
        assert isinstance(result, ControlVariableConfig)

    def test_get_control_config_not_exists(self, sample_config: MFFConfig):
        """Test get_control_config returns None for non-existing control."""
        result = sample_config.get_control_config("nonexistent")
        assert result is None

    def test_get_control_config_case_sensitive(self, sample_config: MFFConfig):
        """Test get_control_config is case sensitive."""
        result = sample_config.get_control_config("PRICE")  # Wrong case
        assert result is None


class TestGetVariableConfig:
    """Tests for the new get_variable_config method."""

    @pytest.fixture
    def sample_config(self) -> MFFConfig:
        """Create a sample MFFConfig."""
        return MFFConfig(
            kpi=KPIConfig(
                name="revenue",
                dimensions=[DimensionType.PERIOD],
            ),
            media_channels=[
                MediaChannelConfig(
                    name="search",
                    dimensions=[DimensionType.PERIOD],
                ),
            ],
            controls=[
                ControlVariableConfig(
                    name="seasonality",
                    dimensions=[DimensionType.PERIOD],
                ),
            ],
        )

    def test_get_variable_config_finds_kpi(self, sample_config: MFFConfig):
        """Test get_variable_config finds KPI by name."""
        result = sample_config.get_variable_config("revenue")
        assert result is not None
        assert result.name == "revenue"
        assert isinstance(result, KPIConfig)

    def test_get_variable_config_finds_media(self, sample_config: MFFConfig):
        """Test get_variable_config finds media channel by name."""
        result = sample_config.get_variable_config("search")
        assert result is not None
        assert result.name == "search"
        assert isinstance(result, MediaChannelConfig)

    def test_get_variable_config_finds_control(self, sample_config: MFFConfig):
        """Test get_variable_config finds control by name."""
        result = sample_config.get_variable_config("seasonality")
        assert result is not None
        assert result.name == "seasonality"
        assert isinstance(result, ControlVariableConfig)

    def test_get_variable_config_not_exists(self, sample_config: MFFConfig):
        """Test get_variable_config returns None for non-existing variable."""
        result = sample_config.get_variable_config("nonexistent")
        assert result is None

    def test_get_variable_config_priority(self):
        """Test get_variable_config returns KPI first if names overlap."""
        # Create config where KPI has same name as a control (unlikely but possible)
        config = MFFConfig(
            kpi=KPIConfig(
                name="target",
                dimensions=[DimensionType.PERIOD],
            ),
            media_channels=[],
            controls=[],
        )
        result = config.get_variable_config("target")
        assert result is not None
        assert isinstance(result, KPIConfig)


class TestGenericLookupHelper:
    """Tests for the _get_config_by_name helper method."""

    @pytest.fixture
    def sample_config(self) -> MFFConfig:
        """Create a sample MFFConfig."""
        return MFFConfig(
            kpi=KPIConfig(
                name="sales",
                dimensions=[DimensionType.PERIOD],
            ),
            media_channels=[
                MediaChannelConfig(name="a", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="b", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="c", dimensions=[DimensionType.PERIOD]),
            ],
            controls=[],
        )

    def test_helper_finds_first_match(self, sample_config: MFFConfig):
        """Test helper finds first matching config."""
        result = sample_config._get_config_by_name(
            sample_config.media_channels, "a"
        )
        assert result is not None
        assert result.name == "a"

    def test_helper_finds_middle_match(self, sample_config: MFFConfig):
        """Test helper finds config in middle of list."""
        result = sample_config._get_config_by_name(
            sample_config.media_channels, "b"
        )
        assert result is not None
        assert result.name == "b"

    def test_helper_finds_last_match(self, sample_config: MFFConfig):
        """Test helper finds config at end of list."""
        result = sample_config._get_config_by_name(
            sample_config.media_channels, "c"
        )
        assert result is not None
        assert result.name == "c"

    def test_helper_returns_none_for_empty_list(self, sample_config: MFFConfig):
        """Test helper returns None for empty list."""
        result = sample_config._get_config_by_name([], "any")
        assert result is None

    def test_helper_returns_none_when_not_found(self, sample_config: MFFConfig):
        """Test helper returns None when name not found."""
        result = sample_config._get_config_by_name(
            sample_config.media_channels, "nonexistent"
        )
        assert result is None


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_get_media_config_returns_same_type(self):
        """Test get_media_config returns MediaChannelConfig type."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="tv", dimensions=[DimensionType.PERIOD])
            ],
            controls=[],
        )
        result = config.get_media_config("tv")
        assert isinstance(result, MediaChannelConfig)

    def test_get_control_config_returns_same_type(self):
        """Test get_control_config returns ControlVariableConfig type."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[],
            controls=[
                ControlVariableConfig(name="price", dimensions=[DimensionType.PERIOD])
            ],
        )
        result = config.get_control_config("price")
        assert isinstance(result, ControlVariableConfig)

    def test_lookup_with_special_characters(self):
        """Test lookup works with special characters in name."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(
                    name="paid_search_google", dimensions=[DimensionType.PERIOD]
                )
            ],
            controls=[],
        )
        result = config.get_media_config("paid_search_google")
        assert result is not None
        assert result.name == "paid_search_google"

    def test_lookup_with_empty_config(self):
        """Test lookup works with empty media/control lists."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[],
            controls=[],
        )
        assert config.get_media_config("any") is None
        assert config.get_control_config("any") is None
        # KPI should still be findable
        assert config.get_variable_config("sales") is not None


class TestEdgeCases:
    """Tests for edge cases in config lookup."""

    def test_multiple_channels_same_prefix(self):
        """Test lookup with channels that share a common prefix."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="tv", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="tv_cable", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(
                    name="tv_broadcast", dimensions=[DimensionType.PERIOD]
                ),
            ],
            controls=[],
        )
        result = config.get_media_config("tv")
        assert result is not None
        assert result.name == "tv"

        result2 = config.get_media_config("tv_cable")
        assert result2 is not None
        assert result2.name == "tv_cable"

    def test_lookup_preserves_all_attributes(self):
        """Test that lookup returns config with all attributes intact."""
        from mmm_framework.config import AdstockConfig, SaturationConfig

        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(
                    name="tv",
                    display_name="Television",
                    unit="GRP",
                    dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
                    adstock=AdstockConfig.geometric(l_max=12),
                    saturation=SaturationConfig.hill(),
                )
            ],
            controls=[],
        )
        result = config.get_media_config("tv")
        assert result is not None
        assert result.display_name == "Television"
        assert result.unit == "GRP"
        assert result.adstock.l_max == 12
        assert len(result.dimensions) == 2

    def test_empty_string_name(self):
        """Test lookup with empty string name."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="tv", dimensions=[DimensionType.PERIOD])
            ],
            controls=[],
        )
        result = config.get_media_config("")
        assert result is None


class TestTypeAnnotations:
    """Tests to verify type annotations are correct."""

    def test_get_media_config_type(self):
        """Test get_media_config has correct return type annotation."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="tv", dimensions=[DimensionType.PERIOD])
            ],
            controls=[],
        )
        result = config.get_media_config("tv")
        # Type should be MediaChannelConfig | None
        if result is not None:
            # These should work without type errors
            _ = result.adstock
            _ = result.saturation
            _ = result.coefficient_prior

    def test_get_control_config_type(self):
        """Test get_control_config has correct return type annotation."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[],
            controls=[
                ControlVariableConfig(name="price", dimensions=[DimensionType.PERIOD])
            ],
        )
        result = config.get_control_config("price")
        # Type should be ControlVariableConfig | None
        if result is not None:
            # These should work without type errors
            _ = result.allow_negative
            _ = result.use_shrinkage

    def test_get_variable_config_type(self):
        """Test get_variable_config has correct return type annotation."""
        config = MFFConfig(
            kpi=KPIConfig(name="sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[],
            controls=[],
        )
        result = config.get_variable_config("sales")
        # Type should be VariableConfig | None
        if result is not None:
            # Base VariableConfig attributes should work
            _ = result.name
            _ = result.dimensions
            _ = result.role

"""
MFF (Master Flat File) configuration builders.

Provides builders for MFFColumnConfig and MFFConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import (
    ControlVariableConfig,
    DimensionAlignmentConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFColumnConfig,
    MFFConfig,
)
from .model import DimensionAlignmentConfigBuilder
from .variable import (
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
    MediaChannelConfigBuilder,
)

if TYPE_CHECKING:
    from typing import Self


class MFFColumnConfigBuilder:
    """
    Builder for MFFColumnConfig objects.

    Examples
    --------
    >>> columns = (MFFColumnConfigBuilder()
    ...     .with_period_column("Week")
    ...     .with_geography_column("Region")
    ...     .build())
    """

    def __init__(self) -> None:
        self._period: str = "Period"
        self._geography: str = "Geography"
        self._product: str = "Product"
        self._campaign: str = "Campaign"
        self._outlet: str = "Outlet"
        self._creative: str = "Creative"
        self._variable_name: str = "VariableName"
        self._variable_value: str = "VariableValue"

    def with_period_column(self, name: str) -> Self:
        """Set period/date column name."""
        self._period = name
        return self

    def with_geography_column(self, name: str) -> Self:
        """Set geography column name."""
        self._geography = name
        return self

    def with_product_column(self, name: str) -> Self:
        """Set product column name."""
        self._product = name
        return self

    def with_campaign_column(self, name: str) -> Self:
        """Set campaign column name."""
        self._campaign = name
        return self

    def with_outlet_column(self, name: str) -> Self:
        """Set outlet column name."""
        self._outlet = name
        return self

    def with_creative_column(self, name: str) -> Self:
        """Set creative column name."""
        self._creative = name
        return self

    def with_variable_name_column(self, name: str) -> Self:
        """Set variable name column name."""
        self._variable_name = name
        return self

    def with_variable_value_column(self, name: str) -> Self:
        """Set variable value column name."""
        self._variable_value = name
        return self

    def build(self) -> MFFColumnConfig:
        """Build the MFFColumnConfig object."""
        return MFFColumnConfig(
            period=self._period,
            geography=self._geography,
            product=self._product,
            campaign=self._campaign,
            outlet=self._outlet,
            creative=self._creative,
            variable_name=self._variable_name,
            variable_value=self._variable_value,
        )


class MFFConfigBuilder:
    """
    Builder for MFFConfig objects - the main configuration builder.

    Examples
    --------
    >>> config = (MFFConfigBuilder()
    ...     .with_kpi(KPIConfigBuilder("Sales").by_geo().build())
    ...     .add_media(MediaChannelConfigBuilder("TV").national().with_geometric_adstock(8).build())
    ...     .add_media(MediaChannelConfigBuilder("Digital").national().with_geometric_adstock(4).build())
    ...     .add_control(ControlVariableConfigBuilder("Price").allow_negative().build())
    ...     .with_alignment(DimensionAlignmentConfigBuilder().geo_by_sales().build())
    ...     .with_date_format("%Y-%m-%d")
    ...     .weekly()
    ...     .build())
    """

    def __init__(self) -> None:
        self._columns: MFFColumnConfig | None = None
        self._kpi: KPIConfig | None = None
        self._media_channels: list[MediaChannelConfig] = []
        self._controls: list[ControlVariableConfig] = []
        self._alignment: DimensionAlignmentConfig | None = None
        self._date_format: str = "%Y-%m-%d"
        self._frequency: str = "W"
        self._fill_missing_media: float = 0.0
        self._fill_missing_controls: float | None = None

    # Column configuration
    def with_columns(self, config: MFFColumnConfig) -> Self:
        """Set column name mappings."""
        self._columns = config
        return self

    def with_columns_builder(self, builder: MFFColumnConfigBuilder) -> Self:
        """Set columns from builder."""
        self._columns = builder.build()
        return self

    # KPI configuration
    def with_kpi(self, config: KPIConfig) -> Self:
        """Set KPI configuration."""
        self._kpi = config
        return self

    def with_kpi_builder(self, builder: KPIConfigBuilder) -> Self:
        """Set KPI from builder."""
        self._kpi = builder.build()
        return self

    def with_kpi_name(self, name: str) -> Self:
        """Convenience: set simple national KPI by name."""
        self._kpi = KPIConfigBuilder(name).national().build()
        return self

    # Media channels
    def add_media(self, config: MediaChannelConfig) -> Self:
        """Add a media channel configuration."""
        self._media_channels.append(config)
        return self

    def add_media_builder(self, builder: MediaChannelConfigBuilder) -> Self:
        """Add media channel from builder."""
        self._media_channels.append(builder.build())
        return self

    def add_media_channels(self, *configs: MediaChannelConfig) -> Self:
        """Add multiple media channel configurations."""
        self._media_channels.extend(configs)
        return self

    def add_national_media(self, name: str, adstock_lmax: int = 8) -> Self:
        """Convenience: add national media with defaults."""
        config = (
            MediaChannelConfigBuilder(name)
            .national()
            .with_geometric_adstock(adstock_lmax)
            .with_hill_saturation()
            .build()
        )
        self._media_channels.append(config)
        return self

    def add_social_platforms(
        self,
        platforms: list[str],
        parent_name: str = "Social",
        adstock_lmax: int = 4,
    ) -> Self:
        """Convenience: add social platforms with hierarchy."""
        for platform in platforms:
            config = (
                MediaChannelConfigBuilder(platform)
                .national()
                .with_geometric_adstock(adstock_lmax)
                .with_hill_saturation()
                .with_parent_channel(parent_name)
                .with_split_dimensions(DimensionType.OUTLET)
                .build()
            )
            self._media_channels.append(config)
        return self

    # Controls
    def add_control(self, config: ControlVariableConfig) -> Self:
        """Add a control variable configuration."""
        self._controls.append(config)
        return self

    def add_control_builder(self, builder: ControlVariableConfigBuilder) -> Self:
        """Add control from builder."""
        self._controls.append(builder.build())
        return self

    def add_controls(self, *configs: ControlVariableConfig) -> Self:
        """Add multiple control configurations."""
        self._controls.extend(configs)
        return self

    def add_price_control(self, name: str = "Price") -> Self:
        """Convenience: add price control (allows negative)."""
        config = ControlVariableConfigBuilder(name).national().allow_negative().build()
        self._controls.append(config)
        return self

    def add_distribution_control(self, name: str = "Distribution") -> Self:
        """Convenience: add distribution/ACV control."""
        config = ControlVariableConfigBuilder(name).national().positive_only().build()
        self._controls.append(config)
        return self

    # Alignment
    def with_alignment(self, config: DimensionAlignmentConfig) -> Self:
        """Set dimension alignment configuration."""
        self._alignment = config
        return self

    def with_alignment_builder(self, builder: DimensionAlignmentConfigBuilder) -> Self:
        """Set alignment from builder."""
        self._alignment = builder.build()
        return self

    # Date and frequency
    def with_date_format(self, fmt: str) -> Self:
        """Set date parsing format."""
        self._date_format = fmt
        return self

    def weekly(self) -> Self:
        """Set weekly frequency."""
        self._frequency = "W"
        return self

    def daily(self) -> Self:
        """Set daily frequency."""
        self._frequency = "D"
        return self

    def monthly(self) -> Self:
        """Set monthly frequency."""
        self._frequency = "M"
        return self

    # Missing value handling
    def with_fill_missing_media(self, value: float) -> Self:
        """Set fill value for missing media (default: 0)."""
        self._fill_missing_media = value
        return self

    def with_fill_missing_controls(self, value: float | None) -> Self:
        """Set fill value for missing controls (None = forward fill)."""
        self._fill_missing_controls = value
        return self

    def build(self) -> MFFConfig:
        """Build the MFFConfig object."""
        if self._kpi is None:
            raise ValueError("KPI configuration is required. Call with_kpi() first.")

        if not self._media_channels:
            raise ValueError(
                "At least one media channel is required. Call add_media() first."
            )

        return MFFConfig(
            columns=self._columns or MFFColumnConfig(),
            kpi=self._kpi,
            media_channels=self._media_channels,
            controls=self._controls,
            alignment=self._alignment or DimensionAlignmentConfig(),
            date_format=self._date_format,
            frequency=self._frequency,
            fill_missing_media=self._fill_missing_media,
            fill_missing_controls=self._fill_missing_controls,
        )


__all__ = [
    "MFFColumnConfigBuilder",
    "MFFConfigBuilder",
]

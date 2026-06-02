"""Factory functions for common configurations."""

from __future__ import annotations

from .enums import DimensionType
from .mff import MFFConfig
from .transforms import AdstockConfig, SaturationConfig
from .variables import ControlVariableConfig, KPIConfig, MediaChannelConfig


def create_national_media_config(
    name: str,
    adstock_lmax: int = 8,
    display_name: str | None = None,
) -> MediaChannelConfig:
    """Create config for national-level media channel."""
    return MediaChannelConfig(
        name=name,
        display_name=display_name or name,
        dimensions=[DimensionType.PERIOD],
        adstock=AdstockConfig.geometric(l_max=adstock_lmax),
        saturation=SaturationConfig.hill(),
    )


def create_geo_media_config(
    name: str,
    adstock_lmax: int = 8,
    display_name: str | None = None,
) -> MediaChannelConfig:
    """Create config for geo-level media channel."""
    return MediaChannelConfig(
        name=name,
        display_name=display_name or name,
        dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
        adstock=AdstockConfig.geometric(l_max=adstock_lmax),
        saturation=SaturationConfig.hill(),
    )


def create_social_platform_configs(
    platforms: list[str],
    parent_name: str = "social",
    adstock_lmax: int = 4,
) -> list[MediaChannelConfig]:
    """Create configs for social media platforms with hierarchical structure."""
    configs = []
    for platform in platforms:
        configs.append(
            MediaChannelConfig(
                name=platform,
                display_name=platform.title(),
                dimensions=[DimensionType.PERIOD],
                split_dimensions=[DimensionType.OUTLET],
                parent_channel=parent_name,
                adstock=AdstockConfig.geometric(l_max=adstock_lmax),
                saturation=SaturationConfig.hill(),
            )
        )
    return configs


def create_simple_mff_config(
    kpi_name: str,
    media_names: list[str],
    control_names: list[str] | None = None,
    kpi_dimensions: list[DimensionType] | None = None,
    multiplicative: bool = False,
) -> MFFConfig:
    """Create a simple MFF config with sensible defaults."""

    kpi_dims = kpi_dimensions or [DimensionType.PERIOD]

    return MFFConfig(
        kpi=KPIConfig(
            name=kpi_name,
            dimensions=kpi_dims,
            log_transform=multiplicative,
        ),
        media_channels=[create_national_media_config(name) for name in media_names],
        controls=[
            ControlVariableConfig(
                name=name,
                dimensions=kpi_dims,
            )
            for name in (control_names or [])
        ],
    )

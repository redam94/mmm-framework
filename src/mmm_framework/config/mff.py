"""MFF data configuration: columns, dimension alignment, and the top-level MFFConfig."""

from __future__ import annotations

import logging
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

from .enums import AllocationMethod, DimensionType
from .variables import (
    ControlVariableConfig,
    KPIConfig,
    MediaChannelConfig,
    VariableConfig,
)

logger = logging.getLogger(__name__)

# TypeVar for generic config lookup
T = TypeVar("T", bound="VariableConfig")


class DimensionAlignmentConfig(BaseModel):
    """Configuration for aligning variables across different dimensions."""

    # How to allocate national media to geo/product
    geo_allocation: AllocationMethod = AllocationMethod.POPULATION
    product_allocation: AllocationMethod = AllocationMethod.SALES

    # Custom weight variable names (from MFF)
    geo_weight_variable: str | None = None
    product_weight_variable: str | None = None

    # Whether to aggregate up or disaggregate down
    prefer_disaggregation: bool = True

    model_config = {"extra": "forbid"}


class MFFColumnConfig(BaseModel):
    """Column name mappings for MFF format."""

    period: str = "Period"
    geography: str = "Geography"
    product: str = "Product"
    campaign: str = "Campaign"
    outlet: str = "Outlet"
    creative: str = "Creative"
    variable_name: str = "VariableName"
    variable_value: str = "VariableValue"

    model_config = {"extra": "forbid"}

    @property
    def dimension_columns(self) -> list[str]:
        """All dimension column names."""
        return [
            self.period,
            self.geography,
            self.product,
            self.campaign,
            self.outlet,
            self.creative,
        ]

    @property
    def all_columns(self) -> list[str]:
        """All expected columns."""
        return self.dimension_columns + [self.variable_name, self.variable_value]


class MFFConfig(BaseModel):
    """Complete configuration for MFF data and model specification."""

    # Column mappings
    columns: MFFColumnConfig = Field(default_factory=MFFColumnConfig)

    # KPI configuration (required)
    kpi: KPIConfig

    # Media channel configurations
    media_channels: list[MediaChannelConfig] = Field(default_factory=list)

    # Control variable configurations
    controls: list[ControlVariableConfig] = Field(default_factory=list)

    # Dimension alignment settings
    alignment: DimensionAlignmentConfig = Field(
        default_factory=DimensionAlignmentConfig
    )

    # Date parsing
    date_format: str = "%Y-%m-%d"
    frequency: Literal["W", "D", "M"] = "W"  # Weekly, daily, monthly
    # Source decimal mark for numeric parsing: "." (US: 1,234.56), "," (EU:
    # 1.234,56), or "auto" (best-effort per value). Lets non-US client data load.
    decimal_separator: Literal[".", ",", "auto"] = "."

    # Missing value handling
    fill_missing_media: float = 0.0
    fill_missing_controls: float | None = None  # None = forward fill
    # RaggedMFFLoader: preserve values that were EXPLICITLY missing (NaN) in the
    # source rather than filling them, tracking them in PanelDataset.explicit_nan_mask.
    preserve_explicit_nan: bool = True

    # Duplicate-row handling. A "duplicate" is two raw rows sharing the FULL MFF
    # key (period + every dimension column) for the same variable -- i.e. the same
    # cell measured twice (join fan-out / re-delivery), which must NOT be silently
    # summed. Rows that differ on a dimension the model does not split on are
    # legitimate finer granularity and are always aggregated up.
    #   "error" (default) -> raise and list the offending keys
    #   "sum" / "mean" / "first" -> combine the duplicated cells that way
    duplicate_policy: Literal["error", "sum", "mean", "first"] = "error"

    model_config = {"extra": "forbid"}

    @property
    def all_variables(self) -> list[VariableConfig]:
        """All configured variables."""
        return [self.kpi] + self.media_channels + self.controls

    @property
    def variable_names(self) -> list[str]:
        """All variable names."""
        return [v.name for v in self.all_variables]

    @property
    def media_names(self) -> list[str]:
        """Media channel names."""
        return [m.name for m in self.media_channels]

    @property
    def control_names(self) -> list[str]:
        """Control variable names."""
        return [c.name for c in self.controls]

    @property
    def target_dimensions(self) -> list[DimensionType]:
        """Dimensions of the target KPI."""
        return self.kpi.dimensions

    def _get_config_by_name(self, configs: list[T], name: str) -> T | None:
        """Generic config lookup by name.

        Parameters
        ----------
        configs : list[T]
            List of config objects to search.
        name : str
            Name to search for.

        Returns
        -------
        T | None
            The config with matching name, or None if not found.
        """
        for config in configs:
            if config.name == name:
                return config
        return None

    def get_media_config(self, name: str) -> MediaChannelConfig | None:
        """Get media channel config by name."""
        return self._get_config_by_name(self.media_channels, name)

    def get_control_config(self, name: str) -> ControlVariableConfig | None:
        """Get control config by name."""
        return self._get_config_by_name(self.controls, name)

    def get_variable_config(self, name: str) -> VariableConfig | None:
        """Get any variable config by name (media, control, or KPI).

        Parameters
        ----------
        name : str
            Variable name to search for.

        Returns
        -------
        VariableConfig | None
            The config with matching name, or None if not found.
        """
        if self.kpi.name == name:
            return self.kpi
        return self._get_config_by_name(
            self.media_channels, name
        ) or self._get_config_by_name(self.controls, name)

    @model_validator(mode="after")
    def dedupe_channel_names(self) -> MFFConfig:
        """Drop media/control entries that repeat an already-configured name.

        The panel channel axis (``coords.channels`` == :attr:`media_names`) is
        built straight from ``media_channels``, but the media matrix is assembled
        as a dict keyed by name (``data_loader``), so a repeated name silently
        collapses to ONE data column while the name axis keeps BOTH labels. That
        desync makes ``channel_names`` longer than the real data — surfacing as
        duplicate channel rows in the ROI / decomposition tables (and a
        ``channel_names`` vs ``n_channels`` mismatch). A duplicate media/control
        name is never meaningful (you cannot model the same column twice), so
        keep the FIRST occurrence and warn. Runs before ``validate_dimensions``
        so downstream validators see the de-duplicated lists.
        """
        for attr in ("media_channels", "controls"):
            items = getattr(self, attr)
            seen: set[str] = set()
            deduped = []
            for item in items:
                if item.name in seen:
                    logger.warning(
                        "Dropping duplicate %s entry %r (already configured)",
                        attr,
                        item.name,
                    )
                    continue
                seen.add(item.name)
                deduped.append(item)
            if len(deduped) != len(items):
                setattr(self, attr, deduped)
        return self

    @model_validator(mode="after")
    def validate_dimensions(self) -> MFFConfig:
        """Ensure dimension compatibility across variables."""
        kpi_dims = set(self.kpi.dimensions)

        # Media can be at same or higher level than KPI
        for media in self.media_channels:
            media_dims = set(media.dimensions)
            # Media dims should be subset of or equal to KPI dims
            # (national media is subset of geo-level KPI)
            if not media_dims.issubset(kpi_dims) and not kpi_dims.issubset(media_dims):
                # Allow if they share Period
                if DimensionType.PERIOD not in media_dims.intersection(kpi_dims):
                    raise ValueError(
                        f"Media '{media.name}' dimensions {media.dim_names} "
                        f"incompatible with KPI dimensions {self.kpi.dim_names}"
                    )

        return self

    def get_hierarchical_media_groups(self) -> dict[str, list[str]]:
        """Get parent->children mapping for hierarchical media."""
        groups: dict[str, list[str]] = {}
        for media in self.media_channels:
            if media.parent_channel:
                if media.parent_channel not in groups:
                    groups[media.parent_channel] = []
                groups[media.parent_channel].append(media.name)
        return groups

"""
Variable configuration builders.

Provides builders for MediaChannelConfig, ControlVariableConfig, and KPIConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    PriorConfig,
    SaturationConfig,
)
from .base import VariableConfigBuilderMixin
from .prior import AdstockConfigBuilder, PriorConfigBuilder, SaturationConfigBuilder

if TYPE_CHECKING:
    from typing import Self


class MediaChannelConfigBuilder(VariableConfigBuilderMixin):
    """
    Builder for MediaChannelConfig objects.

    Examples
    --------
    >>> tv_config = (MediaChannelConfigBuilder("TV")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .with_adstock(AdstockConfigBuilder().geometric().with_max_lag(8).build())
    ...     .with_saturation(SaturationConfigBuilder().hill().build())
    ...     .with_positive_prior(sigma=2.0)
    ...     .build())

    >>> social_meta = (MediaChannelConfigBuilder("Meta")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .with_parent_channel("Social")
    ...     .with_split_dimensions(DimensionType.OUTLET)
    ...     .build())
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._init_variable_fields()  # From VariableConfigBuilderMixin
        self._adstock: AdstockConfig | None = None
        self._saturation: SaturationConfig | None = None
        self._coefficient_prior: PriorConfig | None = None
        self._parent_channel: str | None = None
        self._split_dimensions: list[DimensionType] = []

    def with_adstock(self, config: AdstockConfig) -> Self:
        """Set adstock configuration."""
        self._adstock = config
        return self

    def with_adstock_builder(self, builder: AdstockConfigBuilder) -> Self:
        """Set adstock from builder."""
        self._adstock = builder.build()
        return self

    def with_geometric_adstock(self, l_max: int = 8) -> Self:
        """Convenience: set geometric adstock with max lag."""
        self._adstock = AdstockConfigBuilder().geometric().with_max_lag(l_max).build()
        return self

    def with_saturation(self, config: SaturationConfig) -> Self:
        """Set saturation configuration."""
        self._saturation = config
        return self

    def with_saturation_builder(self, builder: SaturationConfigBuilder) -> Self:
        """Set saturation from builder."""
        self._saturation = builder.build()
        return self

    def with_hill_saturation(self) -> Self:
        """Convenience: set Hill saturation with defaults."""
        self._saturation = SaturationConfigBuilder().hill().build()
        return self

    def with_coefficient_prior(self, prior: PriorConfig) -> Self:
        """Set coefficient prior."""
        self._coefficient_prior = prior
        return self

    def with_positive_prior(self, sigma: float = 2.0) -> Self:
        """Convenience: set HalfNormal prior for positive effects."""
        self._coefficient_prior = PriorConfigBuilder().half_normal(sigma).build()
        return self

    def with_parent_channel(self, parent: str) -> Self:
        """Set parent channel for hierarchical grouping."""
        self._parent_channel = parent
        return self

    def with_split_dimensions(self, *dims: DimensionType) -> Self:
        """Set additional split dimensions (e.g., Outlet for platforms)."""
        self._split_dimensions = list(dims)
        return self

    def build(self) -> MediaChannelConfig:
        """Build the MediaChannelConfig object."""
        # Set defaults if not specified
        adstock = (
            self._adstock or AdstockConfigBuilder().geometric().with_max_lag(8).build()
        )
        saturation = self._saturation or SaturationConfigBuilder().hill().build()
        coefficient_prior = (
            self._coefficient_prior or PriorConfigBuilder().half_normal(2.0).build()
        )

        return MediaChannelConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            adstock=adstock,
            saturation=saturation,
            coefficient_prior=coefficient_prior,
            parent_channel=self._parent_channel,
            split_dimensions=self._split_dimensions,
        )


class ControlVariableConfigBuilder(VariableConfigBuilderMixin):
    """
    Builder for ControlVariableConfig objects.

    Examples
    --------
    >>> price = (ControlVariableConfigBuilder("Price")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .allow_negative()
    ...     .with_normal_prior(mu=0, sigma=1)
    ...     .build())

    >>> distribution = (ControlVariableConfigBuilder("Distribution")
    ...     .national()
    ...     .with_shrinkage()
    ...     .build())
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._init_variable_fields()  # From VariableConfigBuilderMixin
        self._allow_negative: bool = True
        self._coefficient_prior: PriorConfig | None = None
        self._use_shrinkage: bool = False

    def allow_negative(self, allow: bool = True) -> Self:
        """Allow negative coefficient (e.g., for price)."""
        self._allow_negative = allow
        return self

    def positive_only(self) -> Self:
        """Constrain coefficient to be positive."""
        self._allow_negative = False
        return self

    def with_coefficient_prior(self, prior: PriorConfig) -> Self:
        """Set coefficient prior."""
        self._coefficient_prior = prior
        return self

    def with_normal_prior(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Convenience: set Normal prior."""
        self._coefficient_prior = PriorConfigBuilder().normal(mu, sigma).build()
        return self

    def with_shrinkage(self, use: bool = True) -> Self:
        """Enable shrinkage prior for variable selection."""
        self._use_shrinkage = use
        return self

    def build(self) -> ControlVariableConfig:
        """Build the ControlVariableConfig object."""
        coefficient_prior = self._coefficient_prior
        if coefficient_prior is None:
            if self._allow_negative:
                coefficient_prior = PriorConfigBuilder().normal(0, 1).build()
            else:
                coefficient_prior = PriorConfigBuilder().half_normal(1).build()

        return ControlVariableConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            allow_negative=self._allow_negative,
            coefficient_prior=coefficient_prior,
            use_shrinkage=self._use_shrinkage,
        )


class KPIConfigBuilder(VariableConfigBuilderMixin):
    """
    Builder for KPIConfig objects.

    Examples
    --------
    >>> kpi = (KPIConfigBuilder("Sales")
    ...     .with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)
    ...     .multiplicative()
    ...     .build())
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._init_variable_fields()  # From VariableConfigBuilderMixin
        self._log_transform: bool = False
        self._floor_value: float = 1e-6

    def additive(self) -> Self:
        """Use additive model specification (no log transform)."""
        self._log_transform = False
        return self

    def multiplicative(self) -> Self:
        """Use multiplicative model specification (log transform)."""
        self._log_transform = True
        return self

    def with_floor_value(self, floor: float) -> Self:
        """Set minimum value for log transform safety."""
        self._floor_value = floor
        return self

    def build(self) -> KPIConfig:
        """Build the KPIConfig object."""
        return KPIConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            log_transform=self._log_transform,
            floor_value=self._floor_value,
        )


__all__ = [
    "MediaChannelConfigBuilder",
    "ControlVariableConfigBuilder",
    "KPIConfigBuilder",
]

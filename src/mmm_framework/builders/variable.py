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
    MeasurementUnit,
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
        self._measurement_unit: MeasurementUnit = MeasurementUnit.SPEND
        self._spend_column: str | None = None
        self._cpm: float | None = None
        self._cpc: float | None = None

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

    def with_logistic_saturation(self) -> Self:
        """Convenience: set logistic saturation with defaults."""
        self._saturation = SaturationConfigBuilder().logistic().build()
        return self

    def with_michaelis_menten_saturation(self) -> Self:
        """Convenience: set Michaelis-Menten saturation with defaults."""
        self._saturation = SaturationConfigBuilder().michaelis_menten().build()
        return self

    def with_tanh_saturation(self) -> Self:
        """Convenience: set tanh saturation with defaults."""
        self._saturation = SaturationConfigBuilder().tanh().build()
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

    def measured_in(self, unit: MeasurementUnit | str) -> Self:
        """Declare how the modeled variable is measured.

        ``"spend"`` (default) keeps normal ROI. ``"impressions"`` / ``"clicks"``
        / ``"other"`` mark the variable as a *volume*, so ROI is resolved from a
        ``spend_column`` / ``cpm`` / ``cpc`` if provided, else reported as
        efficiency per 1,000 impressions (or per click / unit). The response
        curve is always fit on the modeled variable regardless.
        """
        self._measurement_unit = MeasurementUnit(unit)
        return self

    def with_spend_column(self, column: str) -> Self:
        """Use a SEPARATE MFF variable as the dollar spend for ROI (option a).

        Implies a non-spend ``measurement_unit``; defaults it to impressions if
        still ``spend``."""
        self._spend_column = column
        if self._measurement_unit is MeasurementUnit.SPEND:
            self._measurement_unit = MeasurementUnit.IMPRESSIONS
        return self

    def with_cpm(self, cpm: float) -> Self:
        """Derive ROI by costing the modeled impressions at ``cpm`` per 1,000
        (option b). Implies impressions if ``measurement_unit`` is still spend."""
        self._cpm = cpm
        if self._measurement_unit is MeasurementUnit.SPEND:
            self._measurement_unit = MeasurementUnit.IMPRESSIONS
        return self

    def with_cpc(self, cpc: float) -> Self:
        """Derive ROI by costing the modeled clicks at ``cpc`` per click
        (option b). Sets ``measurement_unit`` to clicks."""
        self._cpc = cpc
        self._measurement_unit = MeasurementUnit.CLICKS
        return self

    def build(self) -> MediaChannelConfig:
        """Build the MediaChannelConfig object."""
        # Set defaults if not specified
        adstock = (
            self._adstock or AdstockConfigBuilder().geometric().with_max_lag(8).build()
        )
        # Default saturation is logistic: that is what the core model has
        # always fit, so unspecified channels keep historical behavior now
        # that BayesianMMM honors the configured saturation type.
        saturation = self._saturation or SaturationConfigBuilder().logistic().build()

        kwargs: dict = dict(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            adstock=adstock,
            saturation=saturation,
            parent_channel=self._parent_channel,
            split_dimensions=self._split_dimensions,
            measurement_unit=self._measurement_unit,
            spend_column=self._spend_column,
            cpm=self._cpm,
            cpc=self._cpc,
        )
        # Pass coefficient_prior ONLY when explicitly configured: the core
        # model honors an *explicitly set* prior (pydantic ``model_fields_set``)
        # and keeps its historical built-in beta prior otherwise. Fabricating a
        # default here would mark every channel "explicitly configured" and
        # silently change all default fits.
        if self._coefficient_prior is not None:
            kwargs["coefficient_prior"] = self._coefficient_prior
        return MediaChannelConfig(**kwargs)


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
        # Pass coefficient_prior ONLY when the caller chose one — either
        # directly, or implicitly via positive_only() (which materializes as an
        # explicit HalfNormal so the core model actually constrains the sign).
        # An unset prior keeps the core model's historical role-based width;
        # fabricating one here would mark every control "explicitly configured".
        coefficient_prior = self._coefficient_prior
        if coefficient_prior is None and not self._allow_negative:
            coefficient_prior = PriorConfigBuilder().half_normal(1).build()

        kwargs: dict = dict(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            allow_negative=self._allow_negative,
            use_shrinkage=self._use_shrinkage,
        )
        if coefficient_prior is not None:
            kwargs["coefficient_prior"] = coefficient_prior
        return ControlVariableConfig(**kwargs)


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

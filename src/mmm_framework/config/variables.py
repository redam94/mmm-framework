"""Per-variable configurations (KPI, media channels, controls)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .enums import (
    CausalControlRole,
    DimensionType,
    MeasurementUnit,
    PriorType,
    VariableRole,
)
from .priors import PriorConfig
from .transforms import AdstockConfig, SaturationConfig


class VariableConfig(BaseModel):
    """Configuration for a single variable in the MFF."""

    name: str = Field(
        ..., description="Variable name as it appears in VariableName column"
    )
    role: VariableRole
    dimensions: list[DimensionType] = Field(
        default_factory=lambda: [DimensionType.PERIOD],
        description="Dimensions this variable is defined over",
    )

    # Optional metadata
    display_name: str | None = None
    unit: str | None = None

    model_config = {"extra": "forbid"}

    @property
    def dim_names(self) -> list[str]:
        """Get dimension names as strings."""
        return [d.value for d in self.dimensions]

    @property
    def has_geo(self) -> bool:
        return DimensionType.GEOGRAPHY in self.dimensions

    @property
    def has_product(self) -> bool:
        return DimensionType.PRODUCT in self.dimensions


class MediaChannelConfig(VariableConfig):
    """Extended configuration for media channels."""

    role: VariableRole = VariableRole.MEDIA

    # Transformation configs. The saturation default is LOGISTIC because that
    # is what the core model has always fit -- it matches historical behavior
    # now that ``BayesianMMM`` honors the configured saturation type per
    # channel (set e.g. ``SaturationConfig.hill()`` to opt in to Hill).
    adstock: AdstockConfig = Field(default_factory=AdstockConfig.geometric)
    saturation: SaturationConfig = Field(default_factory=SaturationConfig.logistic)

    # Coefficient prior (enforces positivity by default)
    coefficient_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig.half_normal(sigma=2.0)
    )

    # Experiment-calibrated prior on the channel's effect coefficient (``beta``).
    # When set -- e.g. derived from a geo-lift / incrementality experiment by
    # :mod:`mmm_framework.calibration` -- this OVERRIDES the core model's default
    # media-coefficient prior, anchoring the channel's effect to randomized
    # evidence. ``None`` preserves the model's built-in default, so the field is
    # purely additive and changes no existing behavior. Despite the name it is a
    # prior on the (saturation-scaled) coefficient, not on raw ROI; the
    # calibration module maps a measured lift to this coefficient scale.
    roi_prior: PriorConfig | None = None

    # Hierarchical grouping (e.g., "social" groups meta, snapchat, twitter)
    parent_channel: str | None = None

    # Split dimensions beyond base dimensions (e.g., Outlet for social platforms)
    split_dimensions: list[DimensionType] = Field(default_factory=list)

    # --- Measurement / cost descriptor (impression-level ROI) -----------------
    # How the modeled variable is measured. The default ``SPEND`` preserves
    # historical behavior exactly (ROI = incremental KPI / summed variable). When
    # set to ``IMPRESSIONS`` / ``CLICKS`` / ``OTHER`` the variable is a *volume*,
    # so ROI is resolved differently (see :class:`MeasurementUnit` and
    # :mod:`mmm_framework.reporting.helpers.measurement`). The response curve is
    # ALWAYS fit on the modeled variable regardless of this field.
    measurement_unit: MeasurementUnit = MeasurementUnit.SPEND

    # Option (a): the name of a SEPARATE MFF variable holding actual dollars for
    # this channel, aligned to the same index as the modeled volume. When set,
    # ROI / mROAS divide by this spend series rather than the modeled variable.
    spend_column: str | None = None

    # Option (b): a cost constant used to DERIVE a spend series from the modeled
    # volume — ``spend = (impressions / 1000) * cpm`` or ``spend = clicks * cpc``
    # — so an impression/click channel reports normal ROI / mROAS comparable to
    # spend channels. At most one of ``spend_column`` / ``cpm`` / ``cpc`` may be
    # set. ``cpm`` is cost per 1,000 modeled units; ``cpc`` is cost per click.
    cpm: float | None = None
    cpc: float | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_measurement(self) -> MediaChannelConfig:
        """Keep the measurement descriptor internally consistent.

        Only one cost source may be declared, and a cost only makes sense for a
        non-spend (volume) channel. The default (``SPEND``, no overrides) passes
        trivially, so existing configs are unaffected.
        """
        cost_sources = [
            n
            for n, v in (
                ("spend_column", self.spend_column),
                ("cpm", self.cpm),
                ("cpc", self.cpc),
            )
            if v is not None
        ]
        if len(cost_sources) > 1:
            raise ValueError(
                f"Media channel '{self.name}': at most one cost source may be set, "
                f"got {cost_sources}. Use a separate spend_column OR a cpm/cpc "
                "constant, not several."
            )
        if cost_sources and self.measurement_unit is MeasurementUnit.SPEND:
            raise ValueError(
                f"Media channel '{self.name}': {cost_sources[0]} implies the modeled "
                "variable is a volume, but measurement_unit is 'spend'. Set "
                "measurement_unit to 'impressions'/'clicks'/'other'."
            )
        if self.cpc is not None and self.measurement_unit is not MeasurementUnit.CLICKS:
            raise ValueError(
                f"Media channel '{self.name}': cpc (cost per click) requires "
                "measurement_unit='clicks'. Use cpm for impression-based cost."
            )
        for fld, val in (("cpm", self.cpm), ("cpc", self.cpc)):
            if val is not None and val <= 0:
                raise ValueError(
                    f"Media channel '{self.name}': {fld} must be positive, got {val}."
                )
        return self

    @property
    def is_child_channel(self) -> bool:
        return self.parent_channel is not None

    @property
    def all_dimensions(self) -> list[DimensionType]:
        """All dimensions including splits."""
        return list(set(self.dimensions + self.split_dimensions))


class ControlVariableConfig(VariableConfig):
    """Configuration for control variables."""

    role: VariableRole = VariableRole.CONTROL

    # Allow negative effects for controls (e.g., price)
    allow_negative: bool = True

    # Prior configuration
    coefficient_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig(
            distribution=PriorType.NORMAL, params={"mu": 0, "sigma": 1}
        )
    )

    # For sparse selection (horseshoe-like behavior)
    use_shrinkage: bool = False

    # Causal role of this control, used to prevent "bad control" bias. ``None``
    # (the default) is treated as a precision control, preserving existing
    # behavior. When set to ``CONFOUNDER`` the core model routes the coefficient
    # to a wide, un-shrunk prior (shrinking a confounder biases the media
    # effect); ``MEDIATOR`` / ``COLLIDER`` are refused at model-construction time
    # because conditioning on them induces bias for a total-effect estimate. The
    # DAG builder populates this automatically from the identified adjustment set
    # (see :mod:`mmm_framework.dag_model_builder.config_translator`).
    causal_role: CausalControlRole | None = None

    # Human-readable provenance for ``causal_role`` (e.g. which treatment makes
    # this a post-treatment variable). Populated by the DAG classifier so the
    # model's bad-control refusal can explain *why* a control was rejected.
    causal_role_reason: str | None = None

    model_config = {"extra": "forbid"}


class KPIConfig(VariableConfig):
    """Configuration for the target KPI variable."""

    role: VariableRole = VariableRole.KPI

    # Log transform for multiplicative model
    log_transform: bool = False

    # Minimum value (for log safety)
    floor_value: float = 1e-6

    model_config = {"extra": "forbid"}

"""Declarative role mapping for the flexible :class:`~mmm_framework.dataset.Dataset`.

:class:`DatasetSchema` generalizes :class:`~mmm_framework.config.mff.MFFConfig`:
where ``MFFConfig`` forces every column into the MMM roles (kpi / media / control),
a ``DatasetSchema`` tags each column with a free :class:`~mmm_framework.config.roles.DatasetRole`
so a non-MMM family (CFA, LCA, …) can declare *indicators* instead of channels.

It mirrors the model-config extensibility pattern: a pure-Pydantic, serializable
spec (``extra="forbid"``, ``schema_version``) that a model family can subclass and
declare as a class attribute (``DATASET_SCHEMA``), exactly as ``CONFIG_SCHEMA``
declares bespoke ``model_params``. The ``from_mff`` / ``to_mff`` bridge keeps the
adapters to/from the legacy ``MFFConfig`` lossless for the four shared roles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from .enums import DimensionType
from .roles import DATASET_ROLE_TO_MFF, MFF_ROLE_TO_DATASET, DatasetRole

if TYPE_CHECKING:  # avoid importing the MFF graph at module load
    from .mff import MFFConfig

#: Bumped when the serialized shape of a :class:`DatasetSchema` changes. Mirrors
#: ``ESTIMAND_SCHEMA_VERSION`` (estimands/spec.py) and the model-params version.
DATASET_SCHEMA_VERSION = "1.0"


class RoleBinding(BaseModel):
    """One column → its :class:`DatasetRole` plus per-role metadata.

    Generalizes :class:`~mmm_framework.config.variables.VariableConfig`. The open
    ``meta`` dict carries family-specific knobs (e.g. ``{"binarize_threshold": 0.5}``
    for an indicator, ``{"n_trials_col": "..."}`` for trials) without a schema
    change — the same spirit as ``LikelihoodConfig.params``.
    """

    name: str = Field(..., description="Column name as it appears in the table")
    role: DatasetRole
    dimensions: list[DimensionType] = Field(
        default_factory=lambda: [DimensionType.PERIOD]
    )
    display_name: str | None = None
    unit: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class DatasetSchema(BaseModel):
    """A declarative role mapping for a :class:`~mmm_framework.dataset.Dataset`.

    This is the *base*, family-agnostic declaration. A bespoke model family
    extends its data contract by subclassing this and assigning it to the model's
    ``DATASET_SCHEMA`` class attribute — the runtime then coerces/validates the
    incoming data against it (mirroring ``CONFIG_SCHEMA`` + ``model_params``).
    """

    #: Class-level version constant (read by the serializer, as ``CONFIG_SCHEMA``
    #: exposes ``SCHEMA_VERSION``). Subclasses may bump it independently.
    SCHEMA_VERSION: str = DATASET_SCHEMA_VERSION

    bindings: list[RoleBinding] = Field(default_factory=list)
    time_col: str = "Period"
    group_cols: list[str] = Field(default_factory=list)
    date_format: str = "%Y-%m-%d"
    frequency: Literal["W", "D", "M"] = "W"
    schema_version: str = DATASET_SCHEMA_VERSION

    model_config = {"extra": "forbid"}

    # ---- role views -------------------------------------------------------
    def names_for(self, role: DatasetRole) -> list[str]:
        """Column names bound to ``role`` (in declaration order)."""
        return [b.name for b in self.bindings if b.role == role]

    @property
    def target_names(self) -> list[str]:
        return self.names_for(DatasetRole.TARGET)

    @property
    def predictor_names(self) -> list[str]:
        return self.names_for(DatasetRole.PREDICTOR)

    @property
    def control_names(self) -> list[str]:
        return self.names_for(DatasetRole.CONTROL)

    @property
    def indicator_names(self) -> list[str]:
        return self.names_for(DatasetRole.INDICATOR)

    def binding(self, name: str) -> RoleBinding | None:
        """The :class:`RoleBinding` for column ``name`` (or ``None``)."""
        return next((b for b in self.bindings if b.name == name), None)

    @model_validator(mode="after")
    def _validate(self) -> "DatasetSchema":
        names = [b.name for b in self.bindings]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"duplicate column names in DatasetSchema.bindings: {dupes}"
            )
        return self

    # ---- the MFF bridge (lossless for the four shared roles) --------------
    @classmethod
    def from_mff(cls, mff: "MFFConfig") -> "DatasetSchema":
        """Build a :class:`DatasetSchema` from an :class:`MFFConfig`.

        Maps kpi→target, media→predictor, control→control, auxiliary→auxiliary,
        preserving each variable's dimensions. The group columns are inferred
        from whether the KPI is defined over geography / product.
        """
        bindings = [
            RoleBinding(
                name=v.name,
                role=MFF_ROLE_TO_DATASET[v.role],
                dimensions=list(v.dimensions),
                display_name=getattr(v, "display_name", None),
                unit=getattr(v, "unit", None),
            )
            for v in mff.all_variables
        ]
        group_cols: list[str] = []
        if mff.kpi.has_geo:
            group_cols.append(mff.columns.geography)
        if mff.kpi.has_product:
            group_cols.append(mff.columns.product)
        return cls(
            bindings=bindings,
            time_col=mff.columns.period,
            group_cols=group_cols,
            date_format=mff.date_format,
            frequency=mff.frequency,
        )

    def to_mff(self) -> "MFFConfig":
        """Best-effort reconstruction of an :class:`MFFConfig` from this schema.

        Lossless for the MMM roles; non-MMM roles (indicator/offset/…) are dropped
        from the MMM view. Used only as the fall-back ``config`` when a Dataset is
        adapted ``as_panel()`` with no original ``MFFConfig`` attached. When a real
        ``MFFConfig`` is present it is preserved verbatim, so this never runs on the
        MMM path.
        """
        from .mff import MFFColumnConfig, MFFConfig
        from .variables import ControlVariableConfig, KPIConfig, MediaChannelConfig

        def dims_of(name: str) -> list[DimensionType]:
            b = self.binding(name)
            return list(b.dimensions) if b else [DimensionType.PERIOD]

        targets = self.target_names
        # MFFConfig requires a KPI; synthesize one from the first observed column
        # when the dataset has no explicit target (e.g. a pure indicator dataset).
        kpi_name = (
            targets[0]
            if targets
            else (self.indicator_names + self.predictor_names + ["value"])[0]
        )
        kpi = KPIConfig(name=kpi_name, dimensions=dims_of(kpi_name))
        media = [
            MediaChannelConfig(name=n, dimensions=dims_of(n))
            for n in self.predictor_names
        ]
        controls = [
            ControlVariableConfig(name=n, dimensions=dims_of(n))
            for n in self.control_names
        ]
        columns = MFFColumnConfig(period=self.time_col)
        return MFFConfig(
            columns=columns,
            kpi=kpi,
            media_channels=media,
            controls=controls,
            date_format=self.date_format,
            frequency=self.frequency,
        )


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "RoleBinding",
    "DatasetSchema",
    "DATASET_ROLE_TO_MFF",
]

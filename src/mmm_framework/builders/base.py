"""
Base classes and mixins for configuration builders.

Provides shared functionality used across all builder types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from ..config import DimensionType

if TYPE_CHECKING:
    from typing import Self

T = TypeVar("T")


@runtime_checkable
class BuilderProtocol(Protocol[T]):
    """Protocol for configuration builders.

    All builders should implement a build() method that returns
    the final configuration object.
    """

    def build(self) -> T:
        """Build and return the configuration object."""
        ...


class VariableConfigBuilderMixin:
    """
    Mixin providing shared methods for variable configuration builders.

    This mixin provides common functionality for builders that configure
    variables with display names, units, and dimensions. It should be used
    by MediaChannelConfigBuilder, ControlVariableConfigBuilder, and
    KPIConfigBuilder to eliminate code duplication.

    Attributes
    ----------
    _display_name : str | None
        Human-readable display name for the variable.
    _unit : str | None
        Unit of measurement (e.g., 'USD', 'GRPs').
    _dimensions : list[DimensionType]
        Dimensions this variable is defined over.
    """

    _display_name: str | None
    _unit: str | None
    _dimensions: list[DimensionType]

    def _init_variable_fields(self) -> None:
        """Initialize common variable fields. Call in subclass __init__."""
        self._display_name = None
        self._unit = None
        self._dimensions = [DimensionType.PERIOD]

    def with_display_name(self, name: str) -> Self:
        """Set human-readable display name.

        Parameters
        ----------
        name : str
            Display name for the variable.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._display_name = name
        return self

    def with_unit(self, unit: str) -> Self:
        """Set unit of measurement.

        Parameters
        ----------
        unit : str
            Unit of measurement (e.g., 'USD', 'GRPs', 'Index').

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._unit = unit
        return self

    def with_dimensions(self, *dims: DimensionType) -> Self:
        """Set dimensions this variable is defined over.

        If PERIOD is not included in the provided dimensions, it will be
        automatically inserted at the beginning.

        Parameters
        ----------
        *dims : DimensionType
            Variable number of dimension types.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._dimensions = list(dims)
        if DimensionType.PERIOD not in self._dimensions:
            self._dimensions.insert(0, DimensionType.PERIOD)
        return self

    def national(self) -> Self:
        """Set as national-level (Period only).

        This is a convenience method equivalent to:
        `with_dimensions(DimensionType.PERIOD)`

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._dimensions = [DimensionType.PERIOD]
        return self

    def by_geo(self) -> Self:
        """Set as geo-level (Period + Geography).

        This is a convenience method equivalent to:
        `with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)`

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        return self

    def by_product(self) -> Self:
        """Set as product-level (Period + Product).

        This is a convenience method equivalent to:
        `with_dimensions(DimensionType.PERIOD, DimensionType.PRODUCT)`

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._dimensions = [DimensionType.PERIOD, DimensionType.PRODUCT]
        return self

    def by_geo_and_product(self) -> Self:
        """Set as geo+product level (Period + Geography + Product).

        This is a convenience method equivalent to:
        `with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY, DimensionType.PRODUCT)`

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._dimensions = [
            DimensionType.PERIOD,
            DimensionType.GEOGRAPHY,
            DimensionType.PRODUCT,
        ]
        return self


__all__ = [
    "BuilderProtocol",
    "VariableConfigBuilderMixin",
]

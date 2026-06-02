"""
Protocol definitions for MMM helper functions.

Defines structural typing protocols for models with traces, PyMC models,
and panel data.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasTrace(Protocol):
    """Protocol for objects with an ArviZ InferenceData trace."""

    @property
    def trace(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects with a PyMC model."""

    @property
    def model(self) -> Any: ...


@runtime_checkable
class HasPanel(Protocol):
    """Protocol for objects with panel data."""

    @property
    def panel(self) -> Any: ...


__all__ = [
    "HasTrace",
    "HasModel",
    "HasPanel",
]

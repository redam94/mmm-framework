"""
Protocol definitions for MMM helper functions.

Defines structural typing protocols for models with traces, PyMC models,
and panel data.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HasTrace(Protocol):
    """Protocol for objects with ArviZ trace."""
    @property
    def _trace(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects with PyMC model."""
    @property
    def model(self) -> Any: ...
    @property
    def _model(self) -> Any: ...


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

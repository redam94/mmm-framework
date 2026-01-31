"""
Protocols for model validation.

Defines the Validatable protocol that all supported model types must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd
    import pymc as pm

    from ..model.results import PredictionResults


@runtime_checkable
class Validatable(Protocol):
    """
    Protocol for models that can be validated.

    All model types (BayesianMMM, NestedMMM, MultivariateMMM, CombinedMMM)
    should satisfy this protocol to be used with the validation package.
    """

    @property
    def _trace(self) -> az.InferenceData:
        """Access the ArviZ InferenceData trace from model fitting."""
        ...

    @property
    def model(self) -> pm.Model:
        """Access the underlying PyMC model."""
        ...

    @property
    def channel_names(self) -> list[str]:
        """Get list of media channel names."""
        ...

    @property
    def n_obs(self) -> int:
        """Number of observations in the dataset."""
        ...

    def predict(self, **kwargs) -> PredictionResults:
        """Generate predictions from the fitted model."""
        ...


@runtime_checkable
class HasMediaData(Protocol):
    """Protocol for models with media spend data."""

    @property
    def X_media(self) -> np.ndarray:
        """Media spend data matrix (n_obs, n_channels)."""
        ...

    @property
    def y(self) -> np.ndarray:
        """Target variable array."""
        ...


@runtime_checkable
class HasPanelData(Protocol):
    """Protocol for models with panel dataset."""

    @property
    def panel(self) -> Any:
        """PanelDataset containing the model data."""
        ...


@runtime_checkable
class HasScalingParams(Protocol):
    """Protocol for models with scaling parameters."""

    @property
    def y_mean(self) -> float:
        """Mean of target variable used for standardization."""
        ...

    @property
    def y_std(self) -> float:
        """Standard deviation of target variable used for standardization."""
        ...


@runtime_checkable
class HasControlData(Protocol):
    """Protocol for models with control variables."""

    @property
    def X_control(self) -> np.ndarray | None:
        """Control variable data matrix."""
        ...

    @property
    def control_names(self) -> list[str] | None:
        """Control variable names."""
        ...


__all__ = [
    "Validatable",
    "HasMediaData",
    "HasPanelData",
    "HasScalingParams",
    "HasControlData",
]

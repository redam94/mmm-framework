"""
Base class for extended MMM models.

Provides common functionality for NestedMMM, MultivariateMMM, and CombinedMMM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    import arviz as az

from ..results import ModelResults


class BaseExtendedMMM:
    """Base class for extended MMM models.

    Provides common functionality for all extended model types:
    - Data storage and validation
    - Model building lifecycle
    - MCMC fitting
    - Result extraction

    Subclasses must implement:
    - _build_coords(): Return PyMC coordinate dict
    - _build_model(): Return built PyMC model
    """

    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        index: pd.Index | None = None,
    ):
        """
        Initialize the base model.

        Parameters
        ----------
        X_media : np.ndarray
            Media variable matrix (n_obs, n_channels)
        y : np.ndarray
            Target variable (n_obs,)
        channel_names : list[str]
            Names of media channels
        index : pd.Index | None
            Optional time index for the data
        """
        self.X_media = X_media
        self.y = y
        self.channel_names = channel_names
        self.index = index if index is not None else pd.RangeIndex(len(y))

        self.n_obs = len(y)
        self.n_channels = len(channel_names)

        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None

    def _build_coords(self) -> dict:
        """Build PyMC coordinates. Override in subclasses."""
        return {
            "obs": np.arange(self.n_obs),
            "channel": self.channel_names,
        }

    def _build_model(self) -> pm.Model:
        """Build the PyMC model. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _build_model")

    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        nuts_sampler: str = "pymc",
        **kwargs,
    ) -> ModelResults:
        """
        Fit the model using MCMC.

        Parameters
        ----------
        draws : int
            Number of posterior draws per chain
        tune : int
            Number of tuning iterations
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS
        random_seed : int | None
            Random seed for reproducibility
        nuts_sampler : str
            NUTS sampler to use ("pymc", "numpyro", "nutpie")
        **kwargs
            Additional arguments passed to pm.sample

        Returns
        -------
        ModelResults
            Container with trace and model
        """
        with self.model:
            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                **kwargs,
            )
        return ModelResults(
            trace=self._trace,
            model=self.model,
            config=getattr(self, "config", None),
        )

    def _check_fitted(self):
        """Check that model has been fitted."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

    @property
    def trace(self) -> az.InferenceData:
        """Get the fitted trace."""
        self._check_fitted()
        return self._trace

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""
        import arviz as az
        self._check_fitted()
        return az.summary(self._trace, var_names=var_names)


__all__ = ["BaseExtendedMMM"]

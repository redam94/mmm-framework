"""
Base classes and protocols for data extractors.

Provides the abstract DataExtractor class and protocols for model introspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    import arviz as az
except ImportError:
    az = None

from .bundle import MMMDataBundle


@runtime_checkable
class HasTrace(Protocol):
    """Protocol for objects with ArviZ InferenceData trace."""

    @property
    def trace(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects with PyMC model."""

    @property
    def model(self) -> Any: ...


class DataExtractor(ABC):
    """
    Base class for model data extractors.

    All concrete extractors should inherit from this class and implement
    the `extract()` method. Shared utilities for HDI computation, diagnostics
    extraction, and fit statistics are provided.

    Attributes
    ----------
    ci_prob : float
        Credible interval probability (default 0.8).

    Examples
    --------
    >>> class MyExtractor(DataExtractor):
    ...     def __init__(self, model, ci_prob=0.8):
    ...         self.model = model
    ...         self._ci_prob = ci_prob
    ...
    ...     @property
    ...     def ci_prob(self):
    ...         return self._ci_prob
    ...
    ...     def extract(self):
    ...         bundle = MMMDataBundle()
    ...         # ... extract data
    ...         return bundle
    """

    @property
    def ci_prob(self) -> float:
        """Credible interval probability. Override in subclass."""
        return getattr(self, "_ci_prob", 0.8)

    @abstractmethod
    def extract(self) -> MMMDataBundle:
        """Extract data from model into unified bundle."""
        pass

    def _compute_hdi(
        self,
        samples: np.ndarray,
        prob: float | None = None,
    ) -> tuple[float, float]:
        """
        Compute highest density interval from samples.

        Parameters
        ----------
        samples : np.ndarray
            MCMC samples.
        prob : float, optional
            HDI probability. If None, uses self.ci_prob.

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        if prob is None:
            prob = self.ci_prob

        if az is not None:
            hdi = az.hdi(samples, hdi_prob=prob)
            return float(hdi[0]), float(hdi[1])
        else:
            # Fallback to percentile-based interval
            alpha = (1 - prob) / 2
            return float(np.percentile(samples, alpha * 100)), float(
                np.percentile(samples, (1 - alpha) * 100)
            )

    def _compute_percentile_bounds(
        self,
        samples: np.ndarray,
        prob: float | None = None,
        axis: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute percentile-based credible interval bounds.

        Parameters
        ----------
        samples : np.ndarray
            Sample array.
        prob : float, optional
            Credible interval probability. If None, uses self.ci_prob.
        axis : int
            Axis along which to compute percentiles.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower, upper) bound arrays.
        """
        if prob is None:
            prob = self.ci_prob

        alpha = (1 - prob) / 2
        lower = np.percentile(samples, alpha * 100, axis=axis)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=axis)
        return lower, upper

    def _compute_fit_statistics(
        self,
        actual: np.ndarray | None,
        predicted: dict[str, np.ndarray] | None,
    ) -> dict[str, float] | None:
        """
        Compute model fit statistics: R², RMSE, MAE, MAPE.

        Parameters
        ----------
        actual : np.ndarray or None
            Actual observed values.
        predicted : dict or None
            Predictions dict with "mean" key.

        Returns
        -------
        dict[str, float] or None
            Dictionary with "r2", "rmse", "mae", "mape" keys.
        """
        if actual is None or predicted is None:
            return None

        y_true = actual
        y_pred = predicted.get("mean")
        if y_pred is None:
            return None

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        else:
            mape = np.nan

        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
        }

    def _extract_diagnostics(self, trace: Any) -> dict[str, Any]:
        """Extract MCMC diagnostics from ArviZ trace."""
        if az is None or trace is None:
            return {}

        try:
            # Get summary stats
            summary = az.summary(trace)

            diagnostics = {
                "rhat_max": float(summary["r_hat"].max()),
                "ess_bulk_min": float(summary["ess_bulk"].min()),
                "ess_tail_min": float(summary["ess_tail"].min()),
            }

            # Check for divergences in sample stats
            if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
                divergences = trace.sample_stats["diverging"].values.sum()
                diagnostics["divergences"] = int(divergences)
            else:
                diagnostics["divergences"] = 0

            return diagnostics
        except Exception:
            return {}


__all__ = [
    "HasTrace",
    "HasModel",
    "DataExtractor",
]

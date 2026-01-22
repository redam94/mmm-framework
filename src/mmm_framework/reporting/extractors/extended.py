"""
ExtendedMMMExtractor - Extract data from mmm-framework's extended MMM models.

Supports NestedMMM, MultivariateMMM, and CombinedMMM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DataExtractor
from .bundle import MMMDataBundle


class ExtendedMMMExtractor(DataExtractor):
    """
    Extract data from mmm-framework's extended MMM models.

    Inherits shared utilities from DataExtractor for HDI computation,
    fit statistics, and MCMC diagnostics.

    Supports NestedMMM, MultivariateMMM, and CombinedMMM.

    Parameters
    ----------
    model : Any
        Extended MMM model instance (NestedMMM, MultivariateMMM, or CombinedMMM)
    ci_prob : float
        Credible interval probability (default 0.8)
    """

    def __init__(
        self,
        model: Any,
        ci_prob: float = 0.8,
    ):
        self.model = model
        self._ci_prob = ci_prob
        self._base_extractor = None

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob

    def extract(self) -> MMMDataBundle:
        """Extract data from extended MMM model."""
        # Start with base extraction if available
        bundle = MMMDataBundle()

        # Extract common data
        bundle.channel_names = self._get_channel_names()
        bundle.dates = self._get_dates()
        bundle.actual = self._get_actual()

        # Get trace for diagnostics
        trace = getattr(self.model, "_trace", None)
        if trace is not None:
            bundle.diagnostics = self._extract_diagnostics(trace)

        # Model-specific extraction
        model_type = type(self.model).__name__

        if "Nested" in model_type:
            bundle = self._extract_nested_data(bundle)
        elif "Multivariate" in model_type:
            bundle = self._extract_multivariate_data(bundle)
        elif "Combined" in model_type:
            bundle = self._extract_combined_data(bundle)

        # Model specification
        bundle.model_specification = self._get_model_specification()

        return bundle

    def _get_channel_names(self) -> list[str]:
        """Get channel names from model."""
        if hasattr(self.model, "channel_names"):
            return list(self.model.channel_names)
        return []

    def _get_dates(self) -> np.ndarray | None:
        """Get date index from model."""
        if hasattr(self.model, "index"):
            return np.array(self.model.index)
        return None

    def _get_actual(self) -> np.ndarray | None:
        """Get actual KPI values."""
        if hasattr(self.model, "y"):
            return np.array(self.model.y)
        return None

    def _extract_nested_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data specific to NestedMMM (mediated effects)."""
        try:
            if hasattr(self.model, "mediator_names"):
                mediator_effects = {}

                trace = getattr(self.model, "_trace", None)
                if trace is not None and hasattr(trace, "posterior"):
                    for med in self.model.mediator_names:
                        # Extract media -> mediator effects
                        for ch in bundle.channel_names or []:
                            effect_name = f"gamma_{ch}_{med}"
                            if effect_name in trace.posterior:
                                samples = trace.posterior[effect_name].values.flatten()
                                mediator_effects[f"{ch} → {med}"] = {
                                    "mean": float(samples.mean()),
                                    "lower": float(np.percentile(samples, 10)),
                                    "upper": float(np.percentile(samples, 90)),
                                }

                bundle.mediator_effects = mediator_effects
        except Exception:
            pass

        return bundle

    def _extract_multivariate_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data specific to MultivariateMMM (multiple outcomes)."""
        try:
            trace = getattr(self.model, "_trace", None)
            if trace is not None and hasattr(trace, "posterior"):
                # Extract correlation matrix if present
                if "corr_chol" in trace.posterior:
                    chol = trace.posterior["corr_chol"].values.mean(axis=(0, 1))
                    bundle.outcome_correlations = chol @ chol.T
        except Exception:
            pass

        return bundle

    def _extract_combined_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data from CombinedMMM (nested + multivariate)."""
        bundle = self._extract_nested_data(bundle)
        bundle = self._extract_multivariate_data(bundle)
        return bundle

    def _get_model_specification(self) -> dict[str, Any]:
        """Get model specification for extended models."""
        spec = {
            "model_type": type(self.model).__name__,
            "likelihood": "Normal",
            "media_effects": "Configured transformations",
        }

        if hasattr(self.model, "mediator_names"):
            spec["mediators"] = list(self.model.mediator_names)

        return spec


__all__ = ["ExtendedMMMExtractor"]

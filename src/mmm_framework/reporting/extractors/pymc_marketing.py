"""
PyMCMarketingExtractor - Extract data from pymc-marketing's MMM class.

Provides compatibility with the standard pymc-marketing MMM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DataExtractor
from .bundle import MMMDataBundle


class PyMCMarketingExtractor(DataExtractor):
    """
    Extract data from pymc-marketing's MMM class.

    Inherits shared utilities from DataExtractor for HDI computation,
    fit statistics, and MCMC diagnostics.

    Provides compatibility with the standard pymc-marketing MMM.

    Parameters
    ----------
    mmm : Any
        PyMC-Marketing MMM instance
    ci_prob : float
        Credible interval probability (default 0.8)
    """

    def __init__(
        self,
        mmm: Any,
        ci_prob: float = 0.8,
    ):
        self.mmm = mmm
        self._ci_prob = ci_prob

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob

    def extract(self) -> MMMDataBundle:
        """Extract data from pymc-marketing MMM."""
        bundle = MMMDataBundle()

        # Get channel names
        if hasattr(self.mmm, "channel_columns"):
            bundle.channel_names = list(self.mmm.channel_columns)

        # Get dates
        if hasattr(self.mmm, "X") and hasattr(self.mmm, "date_column"):
            bundle.dates = self.mmm.X[self.mmm.date_column].values

        # Get actual values
        if hasattr(self.mmm, "y"):
            bundle.actual = np.array(self.mmm.y)

        # Get predictions if fitted
        if hasattr(self.mmm, "idata") and self.mmm.idata is not None:
            try:
                posterior_pred = self.mmm.sample_posterior_predictive()
                if "y" in posterior_pred.posterior_predictive:
                    samples = posterior_pred.posterior_predictive["y"].values
                    samples = samples.reshape(-1, samples.shape[-1])
                    bundle.predicted = {
                        "mean": samples.mean(axis=0),
                        "lower": np.percentile(samples, 10, axis=0),
                        "upper": np.percentile(samples, 90, axis=0),
                    }
            except Exception:
                pass

            # Diagnostics
            bundle.diagnostics = self._extract_diagnostics(self.mmm.idata)

            # Channel contributions
            try:
                contrib = self.mmm.compute_channel_contribution_original_scale()
                if contrib is not None:
                    bundle.component_time_series = {}
                    bundle.component_totals = {}

                    for ch in bundle.channel_names or []:
                        if ch in contrib.columns:
                            bundle.component_time_series[ch] = contrib[ch].values
                            bundle.component_totals[ch] = float(contrib[ch].sum())
            except Exception:
                pass

        # Model specification
        bundle.model_specification = {
            "likelihood": "Normal",
            "adstock": (
                type(self.mmm.adstock).__name__
                if hasattr(self.mmm, "adstock")
                else "Geometric"
            ),
            "saturation": (
                type(self.mmm.saturation).__name__
                if hasattr(self.mmm, "saturation")
                else "Hill"
            ),
        }

        return bundle


__all__ = ["PyMCMarketingExtractor"]

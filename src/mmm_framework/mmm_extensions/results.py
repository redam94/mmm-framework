"""
Result containers for MMM Extensions.

This module contains dataclasses used to return results from
extended model fitting, prediction, and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm


@dataclass
class MediationEffects:
    """Container for mediation analysis results."""

    channel: str
    direct_effect: float
    direct_effect_sd: float
    indirect_effects: dict[str, float]  # mediator -> effect
    total_indirect: float
    total_effect: float
    proportion_mediated: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "channel": self.channel,
            "direct_effect": self.direct_effect,
            "direct_effect_sd": self.direct_effect_sd,
            "total_indirect": self.total_indirect,
            "total_effect": self.total_effect,
            "proportion_mediated": self.proportion_mediated,
        }
        for med, eff in self.indirect_effects.items():
            result[f"indirect_via_{med}"] = eff
        return result


@dataclass
class CrossEffectSummary:
    """Container for cross-effect analysis results."""

    source: str
    target: str
    effect_type: str
    mean: float
    sd: float
    hdi_low: float
    hdi_high: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "effect_type": self.effect_type,
            "mean": self.mean,
            "sd": self.sd,
            "hdi_3%": self.hdi_low,
            "hdi_97%": self.hdi_high,
        }


@dataclass
class ModelResults:
    """Container for fitted extended model results.

    Analogous to MMMResults in the main model module.
    """

    trace: az.InferenceData
    model: pm.Model
    config: Any
    diagnostics: dict = field(default_factory=dict)

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""
        import arviz as az

        return az.summary(self.trace, var_names=var_names)

    def plot_trace(self, var_names: list[str] | None = None, **kwargs):
        """Plot trace diagnostics."""
        import arviz as az

        return az.plot_trace(self.trace, var_names=var_names, **kwargs)

    def plot_posterior(self, var_names: list[str] | None = None, **kwargs):
        """Plot posterior distributions."""
        import arviz as az

        return az.plot_posterior(self.trace, var_names=var_names, **kwargs)


@dataclass
class EffectDecomposition:
    """Container for effect decomposition results.

    Holds direct, indirect, and total effects for each channel-outcome pair.
    """

    channel: str
    outcome: str
    direct_mean: float
    direct_sd: float
    indirect_mean: float
    indirect_sd: float
    total_mean: float
    total_sd: float

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "channel": self.channel,
            "outcome": self.outcome,
            "direct_mean": self.direct_mean,
            "direct_sd": self.direct_sd,
            "indirect_mean": self.indirect_mean,
            "indirect_sd": self.indirect_sd,
            "total_mean": self.total_mean,
            "total_sd": self.total_sd,
        }


__all__ = [
    "MediationEffects",
    "CrossEffectSummary",
    "ModelResults",
    "EffectDecomposition",
]

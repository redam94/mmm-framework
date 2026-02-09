"""
Result container dataclasses for MMM helper functions.

These dataclasses hold computation results with uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ROIResult:
    """Container for ROI computation results with uncertainty."""

    channel: str
    spend: float
    contribution_mean: float
    contribution_lower: float
    contribution_upper: float
    roi_mean: float
    roi_lower: float
    roi_upper: float
    prob_positive: float  # P(ROI > 0)
    prob_profitable: float  # P(ROI > 1)
    marginal_roi_mean: float | None = None
    marginal_roi_lower: float | None = None
    marginal_roi_upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": self.spend,
            "contribution_mean": self.contribution_mean,
            "contribution_hdi_low": self.contribution_lower,
            "contribution_hdi_high": self.contribution_upper,
            "roi_mean": self.roi_mean,
            "roi_hdi_low": self.roi_lower,
            "roi_hdi_high": self.roi_upper,
            "prob_positive": self.prob_positive,
            "prob_profitable": self.prob_profitable,
            "marginal_roi_mean": self.marginal_roi_mean,
            "marginal_roi_hdi_low": self.marginal_roi_lower,
            "marginal_roi_hdi_high": self.marginal_roi_upper,
        }


@dataclass
class PriorPosteriorComparison:
    """Container for prior vs posterior comparison."""

    parameter: str
    prior_mean: float | None
    prior_sd: float | None
    posterior_mean: float
    posterior_sd: float
    posterior_hdi_low: float
    posterior_hdi_high: float
    shrinkage: float | None  # 1 - (posterior_sd / prior_sd)
    prior_samples: np.ndarray | None
    posterior_samples: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "prior_mean": self.prior_mean,
            "prior_sd": self.prior_sd,
            "posterior_mean": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "posterior_hdi_low": self.posterior_hdi_low,
            "posterior_hdi_high": self.posterior_hdi_high,
            "shrinkage": self.shrinkage,
        }


@dataclass
class SaturationCurveResult:
    """Container for saturation curve with uncertainty."""

    channel: str
    spend_grid: np.ndarray
    response_mean: np.ndarray
    response_lower: np.ndarray
    response_upper: np.ndarray
    current_spend: float
    current_response: float
    saturation_level: float  # % of max response at current spend
    marginal_response_at_current: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": self.spend_grid.tolist(),
            "response_mean": self.response_mean.tolist(),
            "response_hdi_low": self.response_lower.tolist(),
            "response_hdi_high": self.response_upper.tolist(),
            "current_spend": self.current_spend,
            "current_response": self.current_response,
            "saturation_level": self.saturation_level,
            "marginal_response_at_current": self.marginal_response_at_current,
        }


@dataclass
class AdstockResult:
    """Container for adstock decay curve."""

    channel: str
    decay_weights: np.ndarray
    alpha_mean: float
    alpha_lower: float
    alpha_upper: float
    half_life: float  # Periods until 50% decay
    total_carryover: float  # Sum of weights beyond t=0
    l_max: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "decay_weights": self.decay_weights.tolist(),
            "alpha_mean": self.alpha_mean,
            "alpha_hdi_low": self.alpha_lower,
            "alpha_hdi_high": self.alpha_upper,
            "half_life": self.half_life,
            "total_carryover": self.total_carryover,
            "l_max": self.l_max,
        }


@dataclass
class DecompositionResult:
    """Container for model component decomposition."""

    component: str
    total_contribution: float
    contribution_lower: float
    contribution_upper: float
    pct_of_total: float
    time_series: np.ndarray | None = None
    time_series_lower: np.ndarray | None = None
    time_series_upper: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "component": self.component,
            "total_contribution": self.total_contribution,
            "contribution_hdi_low": self.contribution_lower,
            "contribution_hdi_high": self.contribution_upper,
            "pct_of_total": self.pct_of_total,
        }
        if self.time_series is not None:
            result["time_series"] = self.time_series.tolist()
        return result


@dataclass
class MediatedEffectResult:
    """Container for mediated (indirect) effect decomposition."""

    channel: str
    outcome: str
    direct_mean: float
    direct_lower: float
    direct_upper: float
    indirect_mean: float
    indirect_lower: float
    indirect_upper: float
    total_mean: float
    total_lower: float
    total_upper: float
    proportion_mediated: float
    mediator_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "outcome": self.outcome,
            "direct_mean": self.direct_mean,
            "direct_hdi_low": self.direct_lower,
            "direct_hdi_high": self.direct_upper,
            "indirect_mean": self.indirect_mean,
            "indirect_hdi_low": self.indirect_lower,
            "indirect_hdi_high": self.indirect_upper,
            "total_mean": self.total_mean,
            "total_hdi_low": self.total_lower,
            "total_hdi_high": self.total_upper,
            "proportion_mediated": self.proportion_mediated,
            "mediator_breakdown": self.mediator_breakdown,
        }


__all__ = [
    "ROIResult",
    "PriorPosteriorComparison",
    "SaturationCurveResult",
    "AdstockResult",
    "DecompositionResult",
    "MediatedEffectResult",
]

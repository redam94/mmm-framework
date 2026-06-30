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
    prob_positive: float  # P(value > 0)
    prob_profitable: float | None  # P(ROI > 1); None for efficiency metrics
    marginal_roi_mean: float | None = None
    marginal_roi_lower: float | None = None
    marginal_roi_upper: float | None = None
    # Measurement metadata (impression-level ROI). Defaults describe a normal
    # spend-based ROI so existing callers and serialized output are unchanged.
    metric_is_monetary: bool = True
    metric_label: str = "ROI"
    marginal_label: str = "Marginal ROAS"
    value_units: str = "ROI"
    divisor_units: str = "$"
    reference: float = 1.0
    measurement_unit: str = "spend"
    cost_basis: str | None = None

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
            "metric_is_monetary": self.metric_is_monetary,
            "metric_label": self.metric_label,
            "marginal_label": self.marginal_label,
            "value_units": self.value_units,
            "divisor_units": self.divisor_units,
            "reference": self.reference,
            "measurement_unit": self.measurement_unit,
            "cost_basis": self.cost_basis,
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
class SpendResponseZones:
    """Per-channel spend-response curves (response, average ROI, and **marginal
    ROI**) over a spend grid, with **shape-aware** **breakthrough / optimal /
    saturation** spend zones — NOT defined on a break-even threshold and NOT on
    percent of maximum response. The zones run (low → high spend):

    * **breakthrough** (under-invested, increasing returns): the **convex** region
      ``[0, inflection]`` where the marginal dollar earns *more* than the last —
      a genuine threshold to break through. Exists only for **S-shaped** curves
      (Hill slope > 1, logistic); for a **concave** curve (exponential,
      Michaelis–Menten, tanh, Hill slope ≤ 1) there is no breakthrough level and
      this zone is **empty** (``breakthrough_range == (0, 0)``).
    * **optimal**: the efficient regime up to where the elasticity
      ``e(s) = mROI(s)/ROI(s)`` drops below ``saturation_elasticity``;
      ``optimal_spend`` is the efficient operating point within it.
    * **saturation** (over-invested): ``e < saturation_elasticity`` — marginal
      ROI far below average ROI; diminishing, reallocate.

    ``recommendation`` follows current spend's position relative to the zones and
    ``optimal_spend`` (below it → increase, at it → hold, in saturation → reduce).
    Average ROI is carried alongside, ``break_even`` is the ROI/mROI reference
    level (the chart's dashed line), and all curves carry posterior uncertainty
    (HDI). Zone boundaries are computed from the posterior-mean curves so the
    slide shows stable numbers.
    """

    channel: str
    # per-period spend grid ($) and the three curves (posterior mean + HDI)
    spend_grid: np.ndarray
    response_mean: np.ndarray
    response_lower: np.ndarray
    response_upper: np.ndarray
    roi_mean: np.ndarray
    roi_lower: np.ndarray
    roi_upper: np.ndarray
    mroi_mean: np.ndarray
    mroi_lower: np.ndarray
    mroi_upper: np.ndarray
    # current spend position + ROI/mROI there (sampled exactly, not grid-snapped)
    current_spend: float
    current_response: float
    current_roi: float
    current_roi_hdi: tuple[float, float]
    current_mroi: float
    current_mroi_hdi: tuple[float, float]
    # zone definition
    break_even: float
    band: float
    # zone spend ranges ($ per period); an empty zone collapses to (x, x)
    breakthrough_range: tuple[float, float]
    optimal_range: tuple[float, float]
    saturation_range: tuple[float, float]
    optimal_spend: float | None
    optimal_roi: float | None
    current_zone: str  # 'breakthrough' | 'optimal' | 'saturation'
    recommendation: str  # 'increase' | 'hold' | 'reduce'
    headroom_to_optimal: float | None  # optimal_spend − current_spend (>0 ⇒ spend more)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "spend": self.spend_grid.tolist(),
            "response_mean": self.response_mean.tolist(),
            "response_hdi_low": self.response_lower.tolist(),
            "response_hdi_high": self.response_upper.tolist(),
            "roi_mean": self.roi_mean.tolist(),
            "roi_hdi_low": self.roi_lower.tolist(),
            "roi_hdi_high": self.roi_upper.tolist(),
            "mroi_mean": self.mroi_mean.tolist(),
            "mroi_hdi_low": self.mroi_lower.tolist(),
            "mroi_hdi_high": self.mroi_upper.tolist(),
            "current_spend": self.current_spend,
            "current_response": self.current_response,
            "current_roi": self.current_roi,
            "current_roi_hdi": list(self.current_roi_hdi),
            "current_mroi": self.current_mroi,
            "current_mroi_hdi": list(self.current_mroi_hdi),
            "break_even": self.break_even,
            "band": self.band,
            "breakthrough_range": list(self.breakthrough_range),
            "optimal_range": list(self.optimal_range),
            "saturation_range": list(self.saturation_range),
            "optimal_spend": self.optimal_spend,
            "optimal_roi": self.optimal_roi,
            "current_zone": self.current_zone,
            "recommendation": self.recommendation,
            "headroom_to_optimal": self.headroom_to_optimal,
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

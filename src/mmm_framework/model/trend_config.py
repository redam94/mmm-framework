"""
Trend configuration classes for BayesianMMM.

This module contains the TrendType enum and TrendConfig dataclass
used to configure trend components in the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrendType(str, Enum):
    """Available trend specifications."""

    NONE = "none"
    LINEAR = "linear"
    PIECEWISE = "piecewise"
    SPLINE = "spline"
    GP = "gaussian_process"


@dataclass
class TrendConfig:
    """Configuration for trend component.

    Attributes:
        type: Type of trend to use.
        n_changepoints: Number of potential changepoints for piecewise trend
            (Prophet-style).
        changepoint_range: Proportion of time range to place changepoints (0-1).
        changepoint_prior_scale: Prior scale for changepoint magnitudes.
        n_knots: Number of knots for spline trend.
        spline_degree: Degree of B-spline (default 3 = cubic).
        spline_prior_sigma: Prior sigma for spline coefficients.
        gp_lengthscale_prior_mu: Prior mean for GP lengthscale
            (in proportion of time range).
        gp_lengthscale_prior_sigma: Prior sigma for GP lengthscale.
        gp_amplitude_prior_sigma: Prior sigma for GP amplitude (HalfNormal).
        gp_n_basis: Number of basis functions for HSGP approximation.
        gp_c: Boundary factor for HSGP (typically 1.5-2.0).
        growth_prior_mu: Prior mean for linear growth rate.
        growth_prior_sigma: Prior sigma for linear growth rate.
    """

    type: TrendType = TrendType.LINEAR

    # Piecewise trend parameters
    n_changepoints: int = 10
    changepoint_range: float = 0.8
    changepoint_prior_scale: float = 0.05

    # Spline trend parameters
    n_knots: int = 10
    spline_degree: int = 3
    spline_prior_sigma: float = 1.0

    # Gaussian Process trend parameters
    gp_lengthscale_prior_mu: float = 0.3
    gp_lengthscale_prior_sigma: float = 0.2
    gp_amplitude_prior_sigma: float = 0.5
    gp_n_basis: int = 20
    gp_c: float = 1.5

    # Linear trend parameters
    growth_prior_mu: float = 0.0
    growth_prior_sigma: float = 0.1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_knots": self.n_knots,
            "spline_degree": self.spline_degree,
            "spline_prior_sigma": self.spline_prior_sigma,
            "gp_lengthscale_prior_mu": self.gp_lengthscale_prior_mu,
            "gp_lengthscale_prior_sigma": self.gp_lengthscale_prior_sigma,
            "gp_amplitude_prior_sigma": self.gp_amplitude_prior_sigma,
            "gp_n_basis": self.gp_n_basis,
            "gp_c": self.gp_c,
            "growth_prior_mu": self.growth_prior_mu,
            "growth_prior_sigma": self.growth_prior_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrendConfig:
        """Create from dictionary."""
        data = data.copy()
        data["type"] = TrendType(data["type"])
        return cls(**data)


__all__ = ["TrendType", "TrendConfig"]

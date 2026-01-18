"""
Model component builders for BayesianMMM.

This subpackage contains strategy classes for building different
model components (trend, seasonality, media effects, etc.).
"""

from .trend import (
    TrendStrategy,
    LinearTrendStrategy,
    PiecewiseTrendStrategy,
    SplineTrendStrategy,
    GPTrendStrategy,
    TrendBuilder,
)

__all__ = [
    "TrendStrategy",
    "LinearTrendStrategy",
    "PiecewiseTrendStrategy",
    "SplineTrendStrategy",
    "GPTrendStrategy",
    "TrendBuilder",
]

"""
Model subpackage for BayesianMMM.

This package provides the core modeling functionality including:
- BayesianMMM: The main model class
- Result containers: MMMResults, PredictionResults, ContributionResults
- Trend configuration: TrendType, TrendConfig
- Data preparation utilities
- Component builders (trend strategies)

For backwards compatibility, all public APIs are re-exported here.
"""

# Result containers
from .results import (
    MMMResults,
    PredictionResults,
    ContributionResults,
    ComponentDecomposition,
)

# Trend configuration
from .trend_config import TrendType, TrendConfig

# Import transform functions from the transforms module for backwards compatibility
from ..transforms import (
    geometric_adstock,
    geometric_adstock_2d,
    logistic_saturation,
    create_fourier_features,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)

# Backward compatibility aliases (original names from model.py)
geometric_adstock_np = geometric_adstock
logistic_saturation_np = logistic_saturation

# Component builders
from .components import (
    TrendStrategy,
    LinearTrendStrategy,
    PiecewiseTrendStrategy,
    SplineTrendStrategy,
    GPTrendStrategy,
    TrendBuilder,
)

# The main BayesianMMM class is imported from base
# This import is at the end to avoid circular imports
from .base import BayesianMMM

__all__ = [
    # Main model
    "BayesianMMM",
    # Results
    "MMMResults",
    "PredictionResults",
    "ContributionResults",
    "ComponentDecomposition",
    # Trend config
    "TrendType",
    "TrendConfig",
    # Transform functions (for backwards compatibility)
    "create_fourier_features",
    "geometric_adstock_np",
    "geometric_adstock_2d",
    "logistic_saturation_np",
    "create_bspline_basis",
    "create_piecewise_trend_matrix",
    # Component builders
    "TrendStrategy",
    "LinearTrendStrategy",
    "PiecewiseTrendStrategy",
    "SplineTrendStrategy",
    "GPTrendStrategy",
    "TrendBuilder",
]

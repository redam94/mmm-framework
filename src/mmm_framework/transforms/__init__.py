"""Transform utilities for MMM Framework.

This module provides transformation functions used in Marketing Mix Models,
including adstock (carryover effects), saturation curves, and seasonality
features.
"""

from .adstock import (
    adstock_weights,
    apply_adstock,
    geometric_adstock,
    geometric_adstock_2d,
    parametric_adstock,
)
from .saturation import logistic_saturation
from .seasonality import create_fourier_features
from .trend import create_bspline_basis, create_piecewise_trend_matrix

__all__ = [
    # Adstock
    "geometric_adstock",
    "geometric_adstock_2d",
    "adstock_weights",
    "apply_adstock",
    "parametric_adstock",
    # Saturation
    "logistic_saturation",
    # Seasonality
    "create_fourier_features",
    # Trend
    "create_bspline_basis",
    "create_piecewise_trend_matrix",
]

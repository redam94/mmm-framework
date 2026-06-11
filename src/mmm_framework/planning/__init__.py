"""Decision layer on top of a fitted MMM: budget optimization and
experiment-design recommendation.

Import-light by design (numpy/pandas only) so it can run inside the agent's
session kernels via ``model_ops``.
"""

from .budget import (
    BudgetOptimizationResult,
    ResponseCurves,
    compute_response_curves,
    optimize_budget,
)
from .experiments import recommend_experiments

__all__ = [
    "BudgetOptimizationResult",
    "ResponseCurves",
    "compute_response_curves",
    "optimize_budget",
    "recommend_experiments",
]

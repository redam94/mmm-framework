"""Decision layer on top of a fitted MMM: budget optimization, EIG/EVOI
experiment prioritization, and experiment-design recommendation.

Import-light by design (numpy/pandas only) so it can run inside the agent's
session kernels via ``model_ops``.
"""

from .budget import (
    BudgetOptimizationResult,
    ResponseCurves,
    compute_response_curves,
    optimize_budget,
)
from .eig import (
    DEFAULT_HALF_LIVES_WEEKS,
    DESIGN_PRECISION,
    channel_half_life,
    decayed_sigma,
    eig_gaussian,
    eig_monte_carlo,
    reexperiment_due,
    sigma_exp_for_design,
)
from .design import (
    design_experiment,
    design_options,
    flighting_design,
    geo_lift_design,
    matched_pairs,
)
from .evoi import EvoiResult, compute_evoi_for_channel, compute_evpi
from .experiments import recommend_experiments
from .priority import ChannelPriority, compute_experiment_priorities

__all__ = [
    "design_experiment",
    "design_options",
    "geo_lift_design",
    "flighting_design",
    "matched_pairs",
    "BudgetOptimizationResult",
    "ResponseCurves",
    "compute_response_curves",
    "optimize_budget",
    "recommend_experiments",
    "ChannelPriority",
    "compute_experiment_priorities",
    "EvoiResult",
    "compute_evoi_for_channel",
    "compute_evpi",
    "eig_gaussian",
    "eig_monte_carlo",
    "sigma_exp_for_design",
    "decayed_sigma",
    "channel_half_life",
    "reexperiment_due",
    "DESIGN_PRECISION",
    "DEFAULT_HALF_LIVES_WEEKS",
]

"""Decision layer on top of a fitted MMM: budget optimization, EIG/EVOI
experiment prioritization, and experiment-design recommendation.

Import-light by design (numpy/pandas only) so it can run inside the agent's
session kernels via ``model_ops``.
"""

from .budget import (
    GEO_ARM_SEP,
    BudgetOptimizationResult,
    ResponseCurves,
    combine_geo_curves,
    compute_response_curves,
    compute_response_curves_per_geo,
    optimize_budget,
    optimize_budget_by_geo,
)
from .flighting import FLIGHTING_PATTERNS, build_flighting_schedule
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
from .opportunity_cost import OpportunityCostResult, compute_opportunity_cost
from .design_anchor import (
    model_anchored_effect,
    powered_to_detect,
    realized_sigma_exp_for_anchor,
)
from .simulation import (
    build_sim_panel,
    methodology_leaderboard,
    run_aa_simulation,
    run_ab_simulation,
)
from .experiment_optimizer import (
    cooldown_weeks,
    evaluate_experiment_grid,
    pareto_front,
    suggest_experiment,
)

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
    "GEO_ARM_SEP",
    "compute_response_curves_per_geo",
    "combine_geo_curves",
    "optimize_budget_by_geo",
    "FLIGHTING_PATTERNS",
    "build_flighting_schedule",
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
    "OpportunityCostResult",
    "compute_opportunity_cost",
    "model_anchored_effect",
    "powered_to_detect",
    "realized_sigma_exp_for_anchor",
    "build_sim_panel",
    "methodology_leaderboard",
    "run_aa_simulation",
    "run_ab_simulation",
    "cooldown_weeks",
    "evaluate_experiment_grid",
    "pareto_front",
    "suggest_experiment",
]

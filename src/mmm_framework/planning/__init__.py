"""Decision layer on top of a fitted MMM: budget optimization, EIG/EVOI
experiment prioritization, and experiment-design recommendation.

Import-light by design (numpy/pandas only) so it can run inside the agent's
session kernels via ``model_ops``.
"""

from .budget import (
    DEFAULT_REALLOC_DEVIATION,
    GEO_ARM_SEP,
    BudgetOptimizationResult,
    ResponseCurves,
    combine_geo_curves,
    compute_response_curves,
    compute_response_curves_per_geo,
    default_reallocation,
    objective_curves,
    objective_label,
    optimize_budget,
    optimize_budget_by_geo,
)
from .frontier import (
    FrontierResult,
    GoalSeekResult,
    budget_frontier,
    goal_seek,
)
from .flighting import FLIGHTING_PATTERNS, build_flighting_schedule
from .pacing import (
    DEFAULT_PACING_THRESHOLD,
    PacingChannel,
    PacingResult,
    compute_pacing,
    expected_outcome_delta,
    pacing_report,
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
from .cpa import (
    cpa_interval,
    cpa_power,
    max_detectable_cpa,
    simulate_cpa_distribution,
)
from .evoi import (
    EvoiResult,
    EvoiSurrogate,
    compute_evoi_for_channel,
    compute_evpi,
    fit_evoi_surrogate,
    preposterior_sd_ratio,
    surrogate_evoi,
)
from .experiments import recommend_experiments
from .priority import ChannelPriority, compute_experiment_priorities
from .opportunity_cost import OpportunityCostResult, compute_opportunity_cost
from .experiment_value import ExperimentNetValue, compute_experiment_net_value
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

# Importing the method registry registers the named experiment methods and wires
# the new geo estimators (synthetic control / TBR / GBR) into the simulation
# leaderboard. Kept last so simulation/design are already loaded (methods import
# from them). Any submodule import of ``planning`` runs this, so the estimators
# are always registered regardless of import path.
from . import methods  # noqa: E402,F401

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
    "objective_curves",
    "objective_label",
    "budget_frontier",
    "goal_seek",
    "FrontierResult",
    "GoalSeekResult",
    "default_reallocation",
    "DEFAULT_REALLOC_DEVIATION",
    "GEO_ARM_SEP",
    "compute_response_curves_per_geo",
    "combine_geo_curves",
    "optimize_budget_by_geo",
    "PacingChannel",
    "PacingResult",
    "compute_pacing",
    "expected_outcome_delta",
    "pacing_report",
    "DEFAULT_PACING_THRESHOLD",
    "FLIGHTING_PATTERNS",
    "build_flighting_schedule",
    "recommend_experiments",
    "ChannelPriority",
    "compute_experiment_priorities",
    "EvoiResult",
    "compute_evoi_for_channel",
    "cpa_interval",
    "cpa_power",
    "max_detectable_cpa",
    "simulate_cpa_distribution",
    "EvoiSurrogate",
    "fit_evoi_surrogate",
    "preposterior_sd_ratio",
    "surrogate_evoi",
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
    "ExperimentNetValue",
    "compute_experiment_net_value",
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

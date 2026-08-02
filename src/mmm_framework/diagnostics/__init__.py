"""Model diagnostics for the MMM framework.

Currently exposes *learning* diagnostics -- how much the data updated each parameter
relative to its prior (prior-to-posterior contraction, overlap, and location shift),
used to flag posteriors that are over-informed by the prior rather than the data.
"""

from __future__ import annotations

from .bias_sensitivity import (
    BiasPrior,
    BiasScenario,
    BiasSensitivity,
    BiasSurface,
    EValueResult,
    TippingPoint,
    bias_adjusted_moments,
    bias_sensitivity_report,
    evalue,
    named_prior_ladder,
    prob_above,
    sensitivity_surface,
    tipping_point,
    tipping_point_mu,
)
from .convergence import (
    ConvergenceWarning,
    annotate as annotate_convergence,
    compute_convergence,
    convergence_flags,
    is_converged,
    warn_if_not_converged,
)
from .coverage import (
    RecoveryCoverageResult,
    coverage_from_ranks,
    failure_mode_guide,
    run_recovery_coverage,
)
from .identification import (
    BoundaryHit,
    FlatDirection,
    IdentificationReport,
    PriorDetermined,
    UninformedParameter,
    bounded_find_MAP,
    guarded_find_MAP,
    unconstrained_box,
    weak_identification_report,
)
from .learning import (
    parameter_learning,
    plot_parameter_learning,
    plot_prior_posterior_overlay,
)
from .saturation import (
    saturation_learning,
    saturation_prior_report,
    warn_if_saturation_prior_is_unanchored,
)
from .sbc import SBCResult, run_mmm_sbc, run_sbc
from .snapshot import compute_fit_diagnostics

__all__ = [
    "BiasPrior",
    "BiasScenario",
    "BiasSensitivity",
    "BiasSurface",
    "BoundaryHit",
    "ConvergenceWarning",
    "EValueResult",
    "FlatDirection",
    "IdentificationReport",
    "PriorDetermined",
    "RecoveryCoverageResult",
    "SBCResult",
    "TippingPoint",
    "UninformedParameter",
    "annotate_convergence",
    "bias_adjusted_moments",
    "bias_sensitivity_report",
    "bounded_find_MAP",
    "compute_convergence",
    "compute_fit_diagnostics",
    "convergence_flags",
    "coverage_from_ranks",
    "evalue",
    "failure_mode_guide",
    "guarded_find_MAP",
    "is_converged",
    "named_prior_ladder",
    "parameter_learning",
    "plot_parameter_learning",
    "plot_prior_posterior_overlay",
    "prob_above",
    "run_mmm_sbc",
    "run_recovery_coverage",
    "run_sbc",
    "saturation_learning",
    "saturation_prior_report",
    "sensitivity_surface",
    "tipping_point",
    "tipping_point_mu",
    "unconstrained_box",
    "warn_if_not_converged",
    "warn_if_saturation_prior_is_unanchored",
    "weak_identification_report",
]

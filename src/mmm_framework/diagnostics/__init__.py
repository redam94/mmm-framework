"""Model diagnostics for the MMM framework.

Currently exposes *learning* diagnostics -- how much the data updated each parameter
relative to its prior (prior-to-posterior contraction, overlap, and location shift),
used to flag posteriors that are over-informed by the prior rather than the data.
"""

from __future__ import annotations

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
from .learning import (
    parameter_learning,
    plot_parameter_learning,
    plot_prior_posterior_overlay,
)
from .sbc import SBCResult, run_mmm_sbc, run_sbc
from .snapshot import compute_fit_diagnostics

__all__ = [
    "ConvergenceWarning",
    "RecoveryCoverageResult",
    "SBCResult",
    "annotate_convergence",
    "compute_convergence",
    "compute_fit_diagnostics",
    "convergence_flags",
    "coverage_from_ranks",
    "failure_mode_guide",
    "is_converged",
    "parameter_learning",
    "plot_parameter_learning",
    "plot_prior_posterior_overlay",
    "run_mmm_sbc",
    "run_recovery_coverage",
    "run_sbc",
    "warn_if_not_converged",
]

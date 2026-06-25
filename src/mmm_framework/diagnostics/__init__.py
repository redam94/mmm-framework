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
from .learning import (
    parameter_learning,
    plot_parameter_learning,
    plot_prior_posterior_overlay,
)
from .snapshot import compute_fit_diagnostics

__all__ = [
    "ConvergenceWarning",
    "annotate_convergence",
    "compute_convergence",
    "compute_fit_diagnostics",
    "convergence_flags",
    "is_converged",
    "parameter_learning",
    "plot_parameter_learning",
    "plot_prior_posterior_overlay",
    "warn_if_not_converged",
]

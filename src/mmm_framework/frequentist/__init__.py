"""Frequentist estimation — ridge, constrained QP, and bootstrap intervals.

Epic #180. See ``technical-docs/frequentist-estimation.md`` for the design spec,
including why this is not a synonym for ``fit(method="map")``: ridge is MAP under
*Gaussian* priors, and this framework's media prior is ``Gamma(mu=1.5, sigma=1)``
or ``LogNormal(0, 1)`` on ROI — neither of which is Gaussian.

Nothing here imports ``cvxpy``: ``constrained`` imports it lazily *inside*
:func:`~mmm_framework.frequentist.constrained.fit_constrained`, so importing this
package — and using the design matrix, the ridge solve and the transform search —
works with the optional ``[frequentist]`` extra absent. The lean-core invariant
(``tests/test_lean_imports.py``) blocks ``cvxpy`` alongside the web and LLM stacks
to keep it that way.

The constraint *builders* (``nonneg``, ``ordering``, ``sum_equals``,
``sum_at_most``) stay in :mod:`~mmm_framework.frequentist.constrained` rather than
being re-exported here — ``nonneg`` as a bare package-level name would read
ambiguously against the ``nonneg=`` keyword on
:func:`~mmm_framework.frequentist.ridge.fit_ridge`, which does something related
but not identical.
"""

from ._transforms import adstock_panel, adstock_series, saturate
from .bootstrap import (
    bc_interval,
    bca_interval,
    bootstrap_fit,
    estimate_block_length,
    moving_block_indices,
    residual_autocorrelation,
)
from .constrained import ConstrainedFit, InfeasibleConstraints, fit_constrained
from .design import DesignMatrix, UnsupportedModelError, build_design_matrix
from .ridge import RidgeFit, fit_ridge
from .search import SearchCandidate, SearchResult, search_transforms

__all__ = [
    "ConstrainedFit",
    "DesignMatrix",
    "InfeasibleConstraints",
    "RidgeFit",
    "SearchCandidate",
    "SearchResult",
    "UnsupportedModelError",
    "adstock_panel",
    "adstock_series",
    "bc_interval",
    "bca_interval",
    "bootstrap_fit",
    "build_design_matrix",
    "estimate_block_length",
    "fit_constrained",
    "fit_ridge",
    "moving_block_indices",
    "residual_autocorrelation",
    "search_transforms",
    "saturate",
]

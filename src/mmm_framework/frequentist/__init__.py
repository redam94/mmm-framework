"""Frequentist estimation — ridge, constrained QP, and bootstrap intervals.

Epic #180. See ``technical-docs/frequentist-estimation.md`` for the design spec,
including why this is not a synonym for ``fit(method="map")``: ridge is MAP under
*Gaussian* priors, and this framework's media prior is ``Gamma(mu=1.5, sigma=1)``
or ``LogNormal(0, 1)`` on ROI — neither of which is Gaussian.

Nothing here imports ``cvxpy``; the constrained estimator imports it lazily so the
lean-core invariant (``tests/test_lean_imports.py``) holds with the optional
``[frequentist]`` extra absent.
"""

from ._transforms import adstock_panel, adstock_series, saturate
from .design import DesignMatrix, UnsupportedModelError, build_design_matrix
from .ridge import RidgeFit, fit_ridge
from .search import SearchCandidate, SearchResult, search_transforms

__all__ = [
    "DesignMatrix",
    "RidgeFit",
    "SearchCandidate",
    "SearchResult",
    "UnsupportedModelError",
    "adstock_panel",
    "adstock_series",
    "build_design_matrix",
    "fit_ridge",
    "search_transforms",
    "saturate",
]

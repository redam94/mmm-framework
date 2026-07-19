"""DiD-MMT — matched-market difference-in-differences.

The matched-market design and its pooled / per-pair DiD estimators already exist
in :mod:`planning.design` (``geo_lift_design(randomize=False)``,
``design_key="matched_market_did"``) and :mod:`planning.simulation`
(``pooled_did_estimator`` / ``per_pair_did_estimator``). This module is a thin
wrapper that names the methodology as a first-class :class:`MethodSpec` and picks
the cluster-robust per-pair DiD as the default MMT analysis — no new math, so the
numbers never drift from the direct estimator path.
"""

from __future__ import annotations

from ..simulation import (
    Assignment,
    EstimatorResult,
    SimPanel,
    Window,
    per_pair_did_estimator,
)


def did_mmt_estimator(
    panel: SimPanel, assignment: Assignment, window: Window
) -> EstimatorResult:
    """Matched-market DiD (cluster-robust per-pair). Delegates verbatim to
    :func:`planning.simulation.per_pair_did_estimator`."""
    return per_pair_did_estimator(panel, assignment, window)


__all__ = ["did_mmt_estimator"]

"""In-graph estimand realization (PyTensor) — likelihood-time scalars.

This is the **in-graph** counterpart to :mod:`mmm_framework.estimands.evaluate`:
it assembles a model-implied estimand as a PyTensor expression to be compared
against an experiment measurement inside the PyMC graph. It is the canonical
home of :func:`build_estimand_expr` (subsuming
``mmm_framework.calibration.likelihood.build_estimand_expr``, which now
delegates here).

The two realizations deliberately share only :mod:`spec` and never call into
each other: this one is deterministic, integer-indexed and scale-folded
(``delta = (pert − base) * scale``), whereas the post-hoc engine is numpy,
paired/unpaired-seeded and boolean-masked. ``pytensor`` is imported lazily so
importing the package stays cheap.

The contribution-window inputs (``contrib_window`` / ``contrib_window_pert``) are
produced by the caller — the in-panel masked sum or the off-panel global-curve
evaluation (``BayesianMMM._offpanel_contribution_std``) — so the
``eval_spend`` / ``adstock_state`` branches live with the model that owns the
channel handle; this module is the estimand *algebra* over those windows.
"""

from __future__ import annotations

from typing import Any

# Estimand-kind string values (kept in sync with
# mmm_framework.calibration.likelihood.ExperimentEstimand to avoid an import
# cycle at module load; the enum is a str-enum with these values).
_CONTRIBUTION = "contribution"
_ROAS = "roas"
_MROAS = "mroas"


def _estimand_value(estimand: Any) -> str:
    """Normalize an ``ExperimentEstimand`` (or string) to its kind value."""
    return estimand.value if hasattr(estimand, "value") else str(estimand)


def build_estimand_expr(
    estimand: Any,
    *,
    contrib_window: Any,
    spend_window: float,
    scale: float = 1.0,
    contrib_window_pert: Any = None,
    lift: float | None = None,
) -> Any:
    """Assemble a model-implied estimand from a window's contribution.

    ``contrib_window`` is the summed per-obs contribution over the experiment
    window in *model* units; ``scale`` converts it to the estimand's natural
    scale (``y_std`` for a standardized model, ``1.0`` for a raw-``y`` model).
    ``spend_window`` is the observed window spend (the ROAS / marginal-spend
    denominator).

    For ``MROAS``, ``contrib_window_pert`` is the contribution under spend scaled
    by ``(1 + lift)`` within the window; the result is the incremental KPI per
    incremental dollar.

    Returns a PyTensor scalar. Bit-identical to the historical
    ``calibration.likelihood.build_estimand_expr`` (which now delegates here).
    """
    kind = _estimand_value(estimand)
    contribution = contrib_window * scale
    if kind == _CONTRIBUTION:
        return contribution
    if kind == _ROAS:
        return contribution / spend_window
    if kind == _MROAS:
        if contrib_window_pert is None or lift is None:
            raise ValueError("MROAS estimand requires contrib_window_pert and lift")
        delta = (contrib_window_pert - contrib_window) * scale
        return delta / (lift * spend_window)
    raise ValueError(f"Unknown estimand: {estimand!r}")


__all__ = ["build_estimand_expr"]

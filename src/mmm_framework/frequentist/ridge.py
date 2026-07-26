"""Penalized linear solve on a fixed-transform design matrix.

Given a :class:`~mmm_framework.frequentist.design.DesignMatrix` the estimate is
closed form,

.. math::

    \\hat\\theta = (X^\\top X + \\lambda P)^{-1} X^\\top y

solved through the **augmented system** :math:`[X; \\sqrt{\\lambda P}]\\theta
\\approx [y; 0]` rather than by forming an inverse — same answer, better
conditioning, and it makes the non-negative variant a one-line change to
``scipy.optimize.nnls``.

**numpy and scipy only.** ``scikit-learn`` is not needed for a penalized linear
solve and is deliberately not added; the lean-core invariant
(``tests/test_lean_imports.py``) stands.

Two things this module reports that a bare coefficient vector does not:

* **effective degrees of freedom**, :math:`\\mathrm{tr}(X(X^\\top X + \\lambda
  P)^{-1}X^\\top)`. It is the honest "how much did the penalty shrink this"
  number, and it is what makes the coverage caveat actionable: ridge is biased by
  construction, so bootstrap intervals under-cover for the true parameter exactly
  when the penalty is doing real work. A fit whose effective dof is far below its
  column count is one whose intervals should be read with that in mind.
* **which coefficients sit at a boundary** under ``nonneg=True``. A coefficient
  pinned at zero has no meaningful two-sided interval, and a bootstrap that
  treats it as interior will misreport it.

On the ridge ≡ MAP question: that equivalence holds for *Gaussian* priors, and
this framework's media prior is ``Gamma`` or ``LogNormal`` on ROI. See
``technical-docs/frequentist-estimation.md`` §1 — and
``tests/frequentist/test_ridge.py``, which pins both directions as executable
facts rather than docstring assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg, optimize

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from .design import DesignMatrix

__all__ = ["RidgeFit", "fit_ridge"]


@dataclass(frozen=True)
class RidgeFit:
    """The result of a penalized linear solve.

    Attributes:
        theta: ``(k,)`` solved coefficients, in the design's parameterization —
            so a media entry is the channel's ROI when the design was built in
            ROI space, and its coefficient otherwise.
        penalty: The scalar penalty strength applied.
        penalize: The per-column penalty weights applied (a boolean mask is
            stored as its 0/1 float form).
        effective_dof: ``tr(X(XᵀX + λP)⁻¹Xᵀ)``. Equals the column count at
            ``penalty=0`` and falls toward the number of unpenalized columns as
            the penalty grows.
        residual_sd: Residual standard deviation on the design's (standardized)
            outcome scale, using ``n - effective_dof`` rather than ``n - k``.
        rss: Residual sum of squares.
        at_boundary: ``(k,)`` boolean, ``True`` where a non-negativity constraint
            is active. Always all-``False`` when ``nonneg=False``.
        columns: The design's column names, carried for reporting.
        diagnostics: Free-form extras (``nonneg``, ``n_obs``, ``n_params``).
    """

    theta: "NDArray[np.floating]"
    penalty: float
    penalize: "NDArray[np.floating]"
    effective_dof: float
    residual_sd: float
    rss: float
    at_boundary: "NDArray[np.bool_]"
    columns: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        """Coefficients keyed by column name."""
        return dict(zip(self.columns, (float(v) for v in self.theta), strict=False))


def _augment(
    X: "NDArray[np.floating]",
    y: "NDArray[np.floating]",
    penalty: float,
    weights: "NDArray[np.floating]",
) -> tuple["NDArray[np.floating]", "NDArray[np.floating]"]:
    """Stack the penalty as extra rows: ``[X; sqrt(λ·P)]θ ≈ [y; 0]``.

    Only columns with a non-zero weight get a row, so an unpenalized intercept
    contributes nothing and the solve stays exactly least squares in that
    direction.
    """
    idx = np.flatnonzero(weights > 0)
    if penalty <= 0 or idx.size == 0:
        return X, y
    rows = np.zeros((idx.size, X.shape[1]))
    rows[np.arange(idx.size), idx] = np.sqrt(penalty * weights[idx])
    return np.vstack([X, rows]), np.concatenate([y, np.zeros(idx.size)])


def _effective_dof(
    X: "NDArray[np.floating]", penalty: float, weights: "NDArray[np.floating]"
) -> float:
    """``tr(X(XᵀX + λP)⁻¹Xᵀ)`` without forming the hat matrix.

    Uses ``tr(X A⁻¹ Xᵀ) = tr(A⁻¹ XᵀX)``, so the work is one ``k × k`` solve
    rather than an ``n × n`` product.
    """
    xtx = X.T @ X
    A = xtx + penalty * np.diag(weights)
    try:
        return float(np.trace(linalg.solve(A, xtx, assume_a="pos")))
    except (linalg.LinAlgError, ValueError):
        return float(np.trace(np.linalg.pinv(A) @ xtx))


def fit_ridge(
    design: "DesignMatrix",
    y: "NDArray[np.floating] | None" = None,
    *,
    penalty: float = 0.0,
    penalize: "NDArray[np.bool_] | NDArray[np.floating] | None" = None,
    nonneg: bool = False,
) -> RidgeFit:
    """Solve the penalized least-squares problem for a fixed-transform design.

    Args:
        design: The design matrix from
            :func:`~mmm_framework.frequentist.design.build_design_matrix`.
        y: Outcome to fit. Defaults to ``design.y`` (the standardized outcome the
            graph's likelihood sees), which is almost always what you want —
            passing something else changes the scale the penalty is defined
            against.
        penalty: Ridge strength ``λ_r``. ``0`` gives ordinary least squares.
            Selection belongs to the out-of-sample criterion in #184, never to an
            in-sample rule; an explicit value is honored as given.
        penalize: Override the design's penalty mask. Accepts a boolean mask
            (uniform penalty on the selected columns) **or** a float vector of
            per-column weights, so ``P`` can be a general diagonal. The weighted
            form is what the model's own priors imply: ``beta_controls`` has
            role-dependent widths — confounders wide, precision controls narrow —
            so the faithful penalty is ``λ_j ∝ 1/σ_j²``, not a scalar. Shrinking a
            confounder re-opens the back-door, and the Bayesian path already knows
            that. The default leaves the intercept, trend and seasonality
            unpenalized and shrinks media, controls and the geo/product dummies —
            and the geo dummies are **identified by** that choice, since the graph
            has no per-geo intercept.
        nonneg: Constrain every coefficient the mask penalizes to be ``>= 0``.
            Unpenalized structural columns (intercept, trend, seasonality) stay
            free, since a negative trend or seasonal coefficient is meaningful.
            Solved with ``scipy.optimize.nnls`` on the augmented system.

    Returns:
        The :class:`RidgeFit`.

    Raises:
        ValueError: If ``penalty`` is negative, or the shapes disagree.
    """
    if penalty < 0:
        raise ValueError(f"penalty must be non-negative, got {penalty}")

    X = np.asarray(design.X, dtype=float)
    y_vec = np.asarray(design.y if y is None else y, dtype=float)
    if y_vec.shape[0] != X.shape[0]:
        raise ValueError(f"y has {y_vec.shape[0]} rows but the design has {X.shape[0]}")

    raw = design.penalize if penalize is None else np.asarray(penalize)
    if raw.shape[0] != X.shape[1]:
        raise ValueError(
            f"penalize has {raw.shape[0]} entries but the design has "
            f"{X.shape[1]} columns"
        )
    # A boolean mask is the uniform case of a diagonal P.
    weights = raw.astype(float)
    if np.any(weights < 0):
        raise ValueError("penalize weights must be non-negative")
    mask = weights > 0

    Xa, ya = _augment(X, y_vec, penalty, weights)

    if nonneg:
        # NNLS constrains EVERY coefficient it solves for, so the unpenalized
        # structural columns are split off, sign-freed by fitting their positive
        # and negative parts, and recombined. Cheaper and more robust than a
        # bespoke active-set solver, and exact.
        free = ~mask
        if free.any():
            Xa = np.hstack([Xa, -Xa[:, free]])
        solution, _ = optimize.nnls(Xa, ya)
        theta = solution[: X.shape[1]].copy()
        if free.any():
            theta[free] -= solution[X.shape[1] :]
        at_boundary = mask & np.isclose(theta, 0.0)
    else:
        theta, *_ = linalg.lstsq(Xa, ya)
        at_boundary = np.zeros(X.shape[1], dtype=bool)

    resid = y_vec - X @ theta
    rss = float(resid @ resid)
    edof = _effective_dof(X, penalty, weights)
    dof_resid = max(X.shape[0] - edof, 1.0)

    return RidgeFit(
        theta=theta,
        penalty=float(penalty),
        penalize=weights,
        effective_dof=edof,
        residual_sd=float(np.sqrt(rss / dof_resid)),
        rss=rss,
        at_boundary=at_boundary,
        columns=list(design.columns),
        diagnostics={
            "nonneg": nonneg,
            "n_obs": int(X.shape[0]),
            "n_params": int(X.shape[1]),
        },
    )

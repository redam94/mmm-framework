"""Trend modeling utilities for time series.

Provides functions to create basis matrices for flexible trend modeling,
including B-splines and piecewise linear (Prophet-style) trends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_bspline_basis(
    t: NDArray[np.floating],
    n_knots: int,
    degree: int = 3,
) -> NDArray[np.floating]:
    """
    Create B-spline basis matrix for flexible trend modeling.

    B-splines are piecewise polynomial functions that provide a smooth,
    flexible way to model trends. The basis functions are localized,
    which helps with interpretability and numerical stability.

    Parameters
    ----------
    t : NDArray[np.floating]
        Time values, should be scaled to [0, 1] for best results.
    n_knots : int
        Number of interior knots. More knots allow more flexibility
        but risk overfitting.
    degree : int, default=3
        Spline degree. degree=3 gives cubic splines, which are
        smooth through the second derivative.

    Returns
    -------
    NDArray[np.floating]
        Basis matrix of shape (len(t), n_knots + degree + 1).
        Each column is a B-spline basis function evaluated at t.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import create_bspline_basis
    >>>
    >>> t = np.linspace(0, 1, 100)
    >>> basis = create_bspline_basis(t, n_knots=5, degree=3)
    >>> print(basis.shape)
    (100, 9)
    >>>
    >>> # Basis sums to 1 (partition of unity)
    >>> np.allclose(basis.sum(axis=1), 1.0)
    True

    Notes
    -----
    The basis is "clamped" at the boundaries, meaning the first and
    last basis functions are 1 at t=0 and t=1 respectively. This
    ensures the fitted curve passes through the endpoint predictions.

    In a Bayesian model, coefficients for each basis function are
    given priors, and the posterior captures uncertainty in the trend.

    Raises
    ------
    ImportError
        If scipy is not installed.

    See Also
    --------
    create_piecewise_trend_matrix : Alternative trend representation.
    """
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        raise ImportError("scipy is required for spline trends")

    # Create knot sequence with appropriate boundary knots
    n_interior = n_knots

    # Interior knots evenly spaced
    interior_knots = np.linspace(0, 1, n_interior + 2)[1:-1]

    # Add boundary knots (repeated for clamping)
    knots = np.concatenate([
        np.zeros(degree + 1),
        interior_knots,
        np.ones(degree + 1),
    ])

    # Number of basis functions
    n_basis = len(knots) - degree - 1

    # Create basis matrix
    basis = np.zeros((len(t), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1
        spline = BSpline(knots, c, degree)
        basis[:, i] = spline(np.clip(t, 0, 1))

    return basis


def create_piecewise_trend_matrix(
    t: NDArray[np.floating],
    n_changepoints: int,
    changepoint_range: float = 0.8,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Create design matrix for piecewise linear trend (Prophet-style).

    This approach models the trend as a piecewise linear function with
    potential changepoints. At each changepoint, the slope can change,
    allowing the trend to capture shifts in growth rates.

    Parameters
    ----------
    t : NDArray[np.floating]
        Time values, should be scaled to [0, 1].
    n_changepoints : int
        Number of potential changepoints. The actual number used is
        determined by the model (sparse priors can shrink some to zero).
    changepoint_range : float, default=0.8
        Proportion of the time range where changepoints can occur.
        Default 0.8 means changepoints only in the first 80% of data,
        which helps avoid overfitting to recent observations.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.floating]]
        (s, A) where:
        - s: array of changepoint locations, shape (n_changepoints,)
        - A: design matrix, shape (len(t), n_changepoints)
          A[i, j] = 1 if t[i] >= s[j], else 0

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import create_piecewise_trend_matrix
    >>>
    >>> t = np.linspace(0, 1, 100)
    >>> s, A = create_piecewise_trend_matrix(t, n_changepoints=5)
    >>> print(s.shape)
    (5,)
    >>> print(A.shape)
    (100, 5)

    Notes
    -----
    The full piecewise linear model is:
        trend(t) = k + m*t + A @ delta

    where:
    - k is the base intercept
    - m is the base growth rate
    - delta are the changepoint adjustments (often with sparse priors)

    This is the approach used by Facebook Prophet for trend modeling.

    The design matrix A implements the formula:
        A[t, j] = 1 if t >= s[j] else 0

    so the change at changepoint j affects all subsequent time points.

    See Also
    --------
    create_bspline_basis : Alternative smooth trend representation.
    """
    # Place changepoints in first changepoint_range of data
    s = np.linspace(0, changepoint_range, n_changepoints + 2)[1:-1]

    # Create design matrix A where A[t, j] = (t - s[j])+ indicator
    A = np.zeros((len(t), len(s)))
    for j, sj in enumerate(s):
        A[:, j] = (t >= sj).astype(float)

    return s, A

"""Adstock (carryover effect) transformations for marketing data.

Adstock models the lagged effect of marketing activities. The geometric
adstock model assumes that the effect of advertising decays exponentially
over time, with parameter alpha controlling the decay rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def geometric_adstock(x: NDArray[np.floating], alpha: float) -> NDArray[np.floating]:
    """
    Apply geometric adstock transformation to a 1D array.

    Implements the recurrence relation:
        y[t] = x[t] + alpha * y[t-1]

    where y[0] = x[0].

    This models the carryover effect of marketing activities, where past
    spending continues to have an effect in future periods, decaying
    exponentially with rate alpha.

    Parameters
    ----------
    x : NDArray[np.floating]
        Input time series (e.g., media spend), shape (n_periods,).
    alpha : float
        Decay rate parameter in [0, 1). Higher values mean slower decay
        (longer-lasting effects). alpha=0 means no carryover.

    Returns
    -------
    NDArray[np.floating]
        Adstocked time series, same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import geometric_adstock
    >>>
    >>> # Single pulse of spend
    >>> spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    >>> adstocked = geometric_adstock(spend, alpha=0.5)
    >>> print(adstocked)
    [100.  50.  25.  12.5  6.25]

    Notes
    -----
    The sum of the adstock weights is 1/(1-alpha), so the total effect
    of a unit spend is scaled by this factor. For alpha=0.5, total effect
    is 2x the immediate effect.
    """
    n = len(x)
    result = np.zeros(n, dtype=x.dtype)
    result[0] = x[0]
    for t in range(1, n):
        result[t] = x[t] + alpha * result[t - 1]
    return result


def geometric_adstock_2d(X: NDArray[np.floating], alpha: float) -> NDArray[np.floating]:
    """
    Apply geometric adstock to a 2D array (multiple channels).

    Applies the geometric adstock transformation independently to each
    column (channel) of the input matrix.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input matrix of shape (n_periods, n_channels).
    alpha : float
        Decay rate parameter in [0, 1).

    Returns
    -------
    NDArray[np.floating]
        Adstocked matrix, same shape as input.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import geometric_adstock_2d
    >>>
    >>> # Two channels with different spend patterns
    >>> X = np.array([
    ...     [100.0, 50.0],
    ...     [0.0, 50.0],
    ...     [0.0, 0.0],
    ... ])
    >>> adstocked = geometric_adstock_2d(X, alpha=0.5)

    See Also
    --------
    geometric_adstock : 1D version of this function.
    """
    result = np.zeros_like(X)
    for c in range(X.shape[1]):
        result[:, c] = geometric_adstock(X[:, c], alpha)
    return result

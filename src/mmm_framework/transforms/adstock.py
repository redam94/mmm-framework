"""Adstock (carryover effect) transformations for marketing data.

Adstock models the lagged effect of marketing activities. The classic
geometric adstock assumes the effect of advertising decays *monotonically*
from the period of spend, with parameter ``alpha`` controlling the decay rate.

Geometric adstock cannot represent a *delayed* peak (an effect that builds for
a few periods before decaying), which is common for brand/TV/video/OOH media.
Two richer weight shapes are provided for that case:

* **Delayed geometric** (Jin et al., 2017): ``w_k = alpha ** ((k - theta) ** 2)``,
  which places the peak weight at lag ``theta``.
* **Weibull** (PDF form): flexible decay whose peak can sit at lag 0 (shape < 1),
  at lag 0 (shape == 1, exponential), or be delayed (shape > 1).

All three are finite-impulse-response (FIR) convolutions that differ only in the
lag-indexed weight vector; :func:`adstock_weights` builds that vector and
:func:`apply_adstock` convolves a series with it. ``w_0`` always multiplies the
current period, so geometric weights peak at lag 0 while delayed/Weibull weights
can peak later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

AdstockKind = Literal["geometric", "delayed", "weibull", "none"]


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


# =============================================================================
# Parametric FIR adstock (geometric / delayed / Weibull)
# =============================================================================


def adstock_weights(
    kind: AdstockKind,
    l_max: int,
    *,
    alpha: float = 0.5,
    theta: float = 0.0,
    shape: float = 2.0,
    scale: float = 2.0,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Build the lag-indexed weight vector for an FIR adstock kernel.

    The returned array ``w`` has length ``l_max`` and is indexed by lag, so
    ``w[0]`` weights the current period and the adstocked series is
    ``y[t] = sum_k w[k] * x[t - k]``.

    Parameters
    ----------
    kind : {"geometric", "delayed", "weibull", "none"}
        Kernel shape. ``"none"`` returns a unit impulse (no carryover).
    l_max : int
        Number of lags (kernel length).
    alpha : float
        Decay rate in [0, 1) for ``"geometric"`` and ``"delayed"``.
    theta : float
        Peak/delay lag for ``"delayed"`` (0 reproduces geometric).
    shape : float
        Weibull shape ``k``. ``<1`` front-loads, ``1`` is exponential, ``>1``
        produces a delayed peak.
    scale : float
        Weibull scale ``lambda`` (controls how far out the mass spreads).
    normalize : bool
        If True, weights sum to 1 (the kernel becomes a weighted moving
        average and total spend magnitude is absorbed into the coefficient).

    Notes
    -----
    **Equifinality / identifiability (critique.md §3.6).** When
    ``normalize=True`` the kernel sums to 1, so the total carryover *magnitude*
    is folded into the channel coefficient ``beta`` rather than the kernel. The
    decay shape (``alpha``/``theta``/``shape``), the saturation strength, and
    ``beta`` then trade off against one another: a long-carryover/weak-saturation
    fit and a short-carryover/strong-saturation fit can be nearly
    indistinguishable in-sample. This is inherent to additive MMM, not a bug, but
    it means per-channel decay and saturation parameters are only weakly
    identified from observational data. Mitigations: informative priors on
    ``alpha``/saturation, anchoring the half-saturation point to data percentiles
    (:meth:`SaturationConfig.compute_kappa_bounds_from_data` for the Hill path),
    and -- most importantly -- experiment-calibrated coefficient priors
    (:mod:`mmm_framework.calibration`), which pin ``beta`` and thereby break the
    trade-off. Setting ``normalize=False`` keeps magnitude in the kernel but does
    not remove the shape/saturation entanglement.

    Returns
    -------
    NDArray[np.floating]
        Weight vector of length ``l_max``.
    """
    lags = np.arange(l_max, dtype=float)

    if kind == "none":
        w = np.zeros(l_max, dtype=float)
        w[0] = 1.0
        return w
    if kind == "geometric":
        w = np.power(alpha, lags)
    elif kind == "delayed":
        w = np.power(alpha, (lags - theta) ** 2)
    elif kind == "weibull":
        # Weibull PDF evaluated at lags (shifted by 1 so lag 0 is well-defined
        # for every shape, including shape < 1 where the PDF diverges at 0).
        t = lags + 1.0
        w = (
            (shape / scale)
            * (t / scale) ** (shape - 1.0)
            * np.exp(-((t / scale) ** shape))
        )
    else:  # pragma: no cover - guarded by typing/validation upstream
        raise ValueError(f"Unknown adstock kind: {kind!r}")

    if normalize:
        total = w.sum()
        w = w / total if total > 0 else w
    return w


def apply_adstock(
    x: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convolve a 1D series with an FIR adstock kernel (causal).

    Computes ``y[t] = sum_k weights[k] * x[t - k]`` with zero padding before
    the start of the series, so the output has the same length as ``x``.

    Parameters
    ----------
    x : NDArray[np.floating]
        Input series, shape ``(n_periods,)``.
    weights : NDArray[np.floating]
        Lag-indexed kernel from :func:`adstock_weights`, length ``l_max``.

    Returns
    -------
    NDArray[np.floating]
        Adstocked series, same shape as ``x``.
    """
    l_max = len(weights)
    n = len(x)
    x_padded = np.concatenate(
        [np.zeros(l_max - 1, dtype=float), np.asarray(x, dtype=float)]
    )
    # Row t selects the window [x[t-(l_max-1)], ..., x[t]]; reversing the
    # weights aligns weights[0] with the current period x[t].
    row_idx = np.arange(n)[:, None] + np.arange(l_max)[None, :]
    windows = x_padded[row_idx]
    return windows @ weights[::-1]


def parametric_adstock(
    x: NDArray[np.floating],
    kind: AdstockKind,
    l_max: int = 8,
    *,
    alpha: float = 0.5,
    theta: float = 0.0,
    shape: float = 2.0,
    scale: float = 2.0,
    normalize: bool = True,
) -> NDArray[np.floating]:
    """Apply a geometric, delayed, or Weibull FIR adstock to a 1D series.

    Convenience wrapper around :func:`adstock_weights` + :func:`apply_adstock`.
    See :func:`adstock_weights` for the meaning of each parameter.
    """
    weights = adstock_weights(
        kind,
        l_max,
        alpha=alpha,
        theta=theta,
        shape=shape,
        scale=scale,
        normalize=normalize,
    )
    return apply_adstock(x, weights)

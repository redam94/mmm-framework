"""Saturation curve transformations for marketing response modeling.

Saturation curves model diminishing returns from marketing activities.
As spend increases, the incremental effect decreases, eventually
reaching a saturation point where additional spend has minimal impact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def logistic_saturation(x: NDArray[np.floating], lam: float) -> NDArray[np.floating]:
    """
    Apply logistic saturation transformation.

    Implements the transformation:
        f(x) = 1 - exp(-lam * x)

    for x >= 0, with negative values clipped to 0.

    This creates an S-shaped response curve that:
    - Starts at 0 when x=0
    - Increases rapidly for small x
    - Asymptotically approaches 1 as x -> infinity

    Parameters
    ----------
    x : NDArray[np.floating]
        Input values (e.g., normalized media spend). Negative values
        are clipped to 0.
    lam : float
        Saturation rate parameter. Higher values cause faster saturation.
        lam > 0 is required for valid behavior.

    Returns
    -------
    NDArray[np.floating]
        Saturated values in the range [0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import logistic_saturation
    >>>
    >>> x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    >>> saturated = logistic_saturation(x, lam=1.0)
    >>> print(saturated.round(3))
    [0.    0.393 0.632 0.865 0.993 1.   ]

    Notes
    -----
    This is sometimes called "exponential saturation" in the literature.
    The half-saturation point (where f(x) = 0.5) occurs at x = ln(2)/lam.

    For modeling purposes, the input x is typically normalized (e.g., by
    dividing by max spend) so that lam can be interpreted consistently
    across channels.

    See Also
    --------
    Hill saturation is another common choice, implemented in the PyMC
    model via pm.math operations.
    """
    return 1.0 - np.exp(-lam * np.clip(x, 0, None))


def root_saturation(x: NDArray[np.floating], exponent: float) -> NDArray[np.floating]:
    """Apply root / power saturation transformation.

    Implements the transformation:
        f(x) = x ** exponent

    for x >= 0, with negative values clipped to 0. With ``0 < exponent < 1``
    this is the classic concave power-response curve — diminishing returns that
    fall off as a fixed power of (adstocked, normalized) spend. ``exponent = 1``
    is linear; ``exponent > 1`` is convex (increasing returns, not saturation).

    Parameters
    ----------
    x : NDArray[np.floating]
        Input values (e.g., normalized media spend). Negative values are
        clipped to 0.
    exponent : float
        The power ``k``. Use ``0 < k < 1`` for a saturating (concave) curve.

    Returns
    -------
    NDArray[np.floating]
        Saturated values ``x ** exponent``.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import root_saturation
    >>>
    >>> x = np.array([0.0, 0.25, 0.5, 1.0])
    >>> root_saturation(x, exponent=0.5).round(3)
    array([0.   , 0.5  , 0.707, 1.   ])

    Notes
    -----
    Unlike the logistic/Hill forms this curve has no finite asymptote; it is
    typically applied to spend normalized into ``[0, 1]`` so ``f`` stays in
    ``[0, 1]``. The marginal ``f'(x) = k * x**(k-1)`` is unbounded at ``x = 0``
    for ``k < 1``, which the model's in-graph form guards against by clamping
    ``x`` away from 0.

    See Also
    --------
    logistic_saturation : exponential-CDF saturation ``1 - exp(-lam * x)``.
    """
    return np.clip(x, 0, None) ** exponent

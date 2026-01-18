"""Statistical utility functions for MMM Framework.

This module provides statistical utilities commonly used in Bayesian
model analysis, such as computing highest density intervals (HDI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_hdi_bounds(
    samples: NDArray,
    hdi_prob: float = 0.94,
    axis: int = 0,
) -> tuple[NDArray, NDArray]:
    """Compute highest density interval bounds using percentiles.

    Computes the central credible interval bounds for a given probability
    mass. This uses a simple percentile-based approach which is appropriate
    for approximately symmetric distributions.

    Parameters
    ----------
    samples : NDArray
        Sample array from posterior distribution. Shape can be
        (n_samples,) for 1D or (n_samples, n_observations) for 2D.
    hdi_prob : float, default=0.94
        Probability mass for the HDI. For example, 0.94 gives the
        central 94% interval.
    axis : int, default=0
        Axis along which to compute percentiles. Typically axis=0
        when samples are in the first dimension.

    Returns
    -------
    tuple[NDArray, NDArray]
        Tuple of (lower_bound, upper_bound) arrays. Shape depends on
        input shape and axis parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.utils import compute_hdi_bounds
    >>>
    >>> # Generate samples
    >>> np.random.seed(42)
    >>> samples = np.random.randn(1000, 10)  # 1000 samples, 10 observations
    >>>
    >>> # Compute 94% HDI
    >>> lower, upper = compute_hdi_bounds(samples, hdi_prob=0.94)
    >>> print(f"Lower bounds shape: {lower.shape}")  # (10,)
    >>> print(f"Upper bounds shape: {upper.shape}")  # (10,)

    Notes
    -----
    This function uses a simple percentile-based approach rather than
    a true highest density interval algorithm. For symmetric distributions
    like the Normal distribution, this is equivalent to the HDI. For
    highly skewed distributions, a proper HDI algorithm may give different
    (narrower) intervals.

    The percentiles are computed as:
    - lower = (1 - hdi_prob) / 2 * 100
    - upper = (1 + hdi_prob) / 2 * 100

    For hdi_prob=0.94, this gives percentiles 3 and 97.
    """
    hdi_low_pct = (1 - hdi_prob) / 2 * 100
    hdi_high_pct = (1 + hdi_prob) / 2 * 100

    return (
        np.percentile(samples, hdi_low_pct, axis=axis),
        np.percentile(samples, hdi_high_pct, axis=axis),
    )

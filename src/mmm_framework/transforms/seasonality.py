"""Seasonality feature creation for time series modeling.

Provides functions to create periodic features (Fourier terms) that
capture seasonal patterns in time series data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_fourier_features(
    t: NDArray[np.floating],
    period: float,
    order: int,
) -> NDArray[np.floating]:
    """
    Create Fourier features for capturing seasonality.

    Generates sine and cosine features at multiple harmonics of the
    specified period. This is the standard approach for modeling
    periodic patterns in time series (e.g., weekly, yearly seasonality).

    Parameters
    ----------
    t : NDArray[np.floating]
        Time index values. Can be any numeric scale (e.g., week numbers,
        day of year, etc.).
    period : float
        The fundamental period length in the same units as t.
        For example, if t is in weeks, period=52 captures yearly seasonality.
    order : int
        Number of Fourier terms (harmonics) to include. Higher order
        captures more complex seasonal patterns but may overfit.
        order=0 returns an empty array.

    Returns
    -------
    NDArray[np.floating]
        Feature matrix of shape (len(t), 2 * order). Columns are
        [sin_1, cos_1, sin_2, cos_2, ..., sin_order, cos_order].
        Returns shape (len(t), 0) if order=0.

    Examples
    --------
    >>> import numpy as np
    >>> from mmm_framework.transforms import create_fourier_features
    >>>
    >>> # Weekly data with yearly seasonality
    >>> weeks = np.arange(104)  # 2 years of data
    >>> features = create_fourier_features(weeks, period=52.0, order=3)
    >>> print(features.shape)
    (104, 6)
    >>>
    >>> # Values repeat after one period
    >>> np.allclose(features[0], features[52])
    True

    Notes
    -----
    The Fourier features at order k are:
        sin(2 * pi * k * t / period)
        cos(2 * pi * k * t / period)

    Using both sine and cosine allows the model to capture phase shifts
    in the seasonal pattern.

    For most applications:
    - order=3-4 is sufficient for smooth seasonal patterns
    - order=6-10 can capture more complex patterns
    - Very high order risks overfitting and should be used with
      regularization

    See Also
    --------
    Prophet (Facebook) uses this same approach for seasonality modeling.
    """
    features = []
    for i in range(1, order + 1):
        features.append(np.sin(2 * np.pi * i * t / period))
        features.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(features) if features else np.zeros((len(t), 0))

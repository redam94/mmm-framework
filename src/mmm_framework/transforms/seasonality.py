"""Seasonality feature creation for time series modeling.

Provides functions to create periodic features (Fourier terms) that
capture seasonal patterns in time series data.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


#: Observations per natural seasonal period, by data frequency.
#:
#: The single source. This table previously existed in three places — the core
#: graph builder (``model.base.BayesianMMM._prepare_seasonality``), the
#: forecaster (``validation.backtest._PERIODS_BY_FREQ``, a copy-pasted literal
#: linked only by a comment) — while the extension graph
#: (``mmm_extensions.components.temporal``) derived its own periods from the
#: datetime index's median spacing, under a docstring claiming the two agreed.
#: They did not: 365.25 / 7 is **52.178571**, not 52.0, so a nested model's
#: "yearly" seasonality was a different basis from a plain MMM's. Measured on
#: the order-2 yearly design over 104 weekly points, max |Δ| is **0.08174**
#: (order 1: 0.04216; order 3: 0.12376) on a basis whose amplitude is O(1) —
#: small enough to look like noise in a plot, large enough to move a
#: decomposition.
PERIODS_BY_FREQ: dict[str, dict[str, float]] = {
    "W": {"yearly": 52.0, "monthly": 52.0 / 12.0},
    "D": {"yearly": 365.25, "monthly": 365.25 / 12.0, "weekly": 7.0},
    "M": {"yearly": 12.0},
}

#: Days per observation for each frequency in :data:`PERIODS_BY_FREQ`, used to
#: recognise a datetime index's spacing as one of the tabulated frequencies.
_DAYS_PER_OBS: dict[str, float] = {"W": 7.0, "D": 1.0, "M": 365.25 / 12.0}


class SeasonalityPeriodSource(str, Enum):
    """Where a graph gets its observations-per-seasonal-period from.

    The two model families answer this differently, and the answer changes the
    Fourier basis, so it is named rather than implied.

    ``FREQUENCY_TABLE`` is what the core :class:`~mmm_framework.model.base.BayesianMMM`
    does and the only source it supports: look the frequency up in
    :data:`PERIODS_BY_FREQ`, giving a yearly period of exactly 52.0 on weekly
    data. ``DATETIME_MEDIAN`` is what the extension graphs do: divide 365.25 by
    the median inter-observation spacing in days, giving 52.178571 on weekly
    data.

    Each site keeps its historical source by default, so no existing fit moves;
    setting :attr:`~mmm_framework.config.model.SeasonalityConfig.period_source`
    to ``FREQUENCY_TABLE`` makes an extension model's seasonal basis identical
    to the core model's.
    """

    FREQUENCY_TABLE = "frequency_table"
    DATETIME_MEDIAN = "datetime_median"


def frequency_from_median_days(median_days: float, tolerance: float = 0.25) -> str | None:
    """The :data:`PERIODS_BY_FREQ` key matching an observed spacing, or None.

    ``tolerance`` is a relative band, so a monthly panel whose median spacing is
    30 or 31 days still resolves to ``"M"`` (tabulated at 30.4375).
    """
    if not median_days or median_days <= 0:
        return None
    best, best_err = None, float("inf")
    for freq, days in _DAYS_PER_OBS.items():
        err = abs(float(median_days) - days) / days
        if err < best_err:
            best, best_err = freq, err
    return best if best_err <= tolerance else None


def periods_for_frequency(freq: str | None) -> dict[str, float]:
    """The component→period map for ``freq``, defaulting to weekly."""
    return PERIODS_BY_FREQ.get(freq or "W", PERIODS_BY_FREQ["W"])


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

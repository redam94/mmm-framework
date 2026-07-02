"""Dollar <-> scaled-unit conversion — make ``spend_ref`` real (review fix F8).

The data contract keeps spend **scaled**: every channel is divided by a fixed,
per-channel reference constant (``spend_ref``, dollars per scaled unit) chosen
once at program start — never a cluster mean, never re-estimated mid-program
(rescaling mid-flight silently changes what ``kappa`` means). Historically the
package only *carried* ``spend_ref`` on :class:`~mmm_framework.continuous_learning.model.Posterior`
and ``LearningState`` without ever applying it; these two helpers are the single
sanctioned conversion so the service layer can work in dollars at the boundary
and scaled units internally.

Convention: ``scaled = dollars / spend_ref`` elementwise over the channel axis
(the LAST axis), so both a single ``(K,)`` allocation and an ``(N, K)`` panel
convert with the same call.
"""

from __future__ import annotations

import numpy as np


def _validate_ref(spend_ref: np.ndarray) -> np.ndarray:
    """Coerce + validate a per-channel reference vector (positive, finite, 1-D)."""
    ref = np.asarray(spend_ref, dtype=float)
    if ref.ndim != 1:
        raise ValueError(f"spend_ref must be 1-D (K,), got shape {ref.shape}")
    if not np.all(np.isfinite(ref)):
        raise ValueError("spend_ref contains non-finite values")
    if np.any(ref <= 0):
        raise ValueError("spend_ref must be strictly positive per channel")
    return ref


def _check_last_axis(arr: np.ndarray, ref: np.ndarray, what: str) -> None:
    if arr.ndim >= 1 and arr.shape[-1] != ref.shape[0]:
        raise ValueError(
            f"{what} has last-axis length {arr.shape[-1]} but spend_ref has "
            f"{ref.shape[0]} channels"
        )


def to_scaled(dollars: np.ndarray, spend_ref: np.ndarray) -> np.ndarray:
    """Convert dollar spend to scaled units: ``dollars / spend_ref``.

    Args:
        dollars: spend in dollars — a ``(K,)`` allocation or an ``(N, K)``
            panel (any leading shape; the channel axis is the last one).
        spend_ref: dollars per scaled unit, shape ``(K,)``, strictly positive.

    Returns:
        The scaled spend, same shape as ``dollars`` (float64).
    """
    ref = _validate_ref(spend_ref)
    d = np.asarray(dollars, dtype=float)
    _check_last_axis(d, ref, "dollars")
    return d / ref


def to_dollars(scaled: np.ndarray, spend_ref: np.ndarray) -> np.ndarray:
    """Convert scaled spend back to dollars: ``scaled * spend_ref``.

    Args:
        scaled: scaled spend — a ``(K,)`` allocation or an ``(N, K)`` panel.
        spend_ref: dollars per scaled unit, shape ``(K,)``, strictly positive.

    Returns:
        The dollar spend, same shape as ``scaled`` (float64).
    """
    ref = _validate_ref(spend_ref)
    s = np.asarray(scaled, dtype=float)
    _check_last_axis(s, ref, "scaled")
    return s * ref

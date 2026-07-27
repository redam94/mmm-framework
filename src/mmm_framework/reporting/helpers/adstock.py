"""
Adstock computation functions for MMM reporting.

Functions for computing adstock decay weights and carryover effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .results import AdstockResult
from .utils import (
    _check_model_fitted,
    _compute_hdi,
    _flatten_samples,
    _get_channel_names,
    _get_posterior,
)


def compute_adstock_weights(
    model: Any,
    channels: list[str] | None = None,
    hdi_prob: float = 0.94,
) -> dict[str, AdstockResult]:
    """
    Compute adstock decay weights for each channel.

    Shows how advertising effects decay over time (carryover effects).

    Parameters
    ----------
    model : BayesianMMM
        Fitted model
    channels : list[str], optional
        Channels to compute. If None, uses all.
    hdi_prob : float
        HDI probability

    Returns
    -------
    dict[str, AdstockResult]
        Adstock curves by channel

    Examples
    --------
    >>> adstock = compute_adstock_weights(mmm)
    >>> for ch, result in adstock.items():
    ...     print(f"{ch}: half-life = {result.half_life:.1f} periods")
    """
    _check_model_fitted(model)

    posterior = _get_posterior(model)

    if channels is None:
        channels = _get_channel_names(model)

    from ...transforms.carryover import carryover_half_life, posterior_carryover_kernels

    kernels = posterior_carryover_kernels(model, list(channels))

    results = {}
    for channel in channels:
        k = kernels.get(channel)
        if k is None:  # pragma: no cover - posterior_carryover_kernels is total
            continue

        weights = k.mean_kernel
        if not np.all(np.isfinite(weights)):
            # Kernel unreadable — RETURN it with a reason rather than dropping
            # the channel, which is what used to make Weibull channels vanish.
            results[channel] = AdstockResult(
                channel=channel,
                decay_weights=np.array([np.nan]),
                alpha_mean=float("nan"),
                alpha_lower=float("nan"),
                alpha_upper=float("nan"),
                half_life=float("nan"),
                total_carryover=float("nan"),
                l_max=int(k.l_max),
                family=k.family,
                status=k.status,
            )
            continue

        # Half-life from the PER-DRAW kernels, then averaged — not from a
        # collapsed alpha. mean(alpha) ** lags is not mean(alpha ** lags): on a
        # real posterior the lag-5 weight was understated 7x and the half-life
        # by 41%.
        hl = carryover_half_life(k.kernel)
        half_life = float(np.nanmean(hl)) if np.any(np.isfinite(hl)) else float("nan")

        alpha_mean = k.alpha_mean
        if alpha_mean is None:
            alpha_lower = alpha_upper = float("nan")
            alpha_mean = float("nan")
        else:
            samples = _get_adstock_alpha(posterior, channel)
            if samples is not None and k.family in ("geometric", "delayed"):
                alpha_lower, alpha_upper = _compute_hdi(samples, hdi_prob)
            else:
                alpha_lower = alpha_upper = float("nan")

        results[channel] = AdstockResult(
            channel=channel,
            decay_weights=weights,
            alpha_mean=float(alpha_mean),
            alpha_lower=float(alpha_lower),
            alpha_upper=float(alpha_upper),
            half_life=half_life,
            total_carryover=float(weights[1:].sum() / weights.sum()),
            l_max=int(k.l_max),
            family=k.family,
            status=k.status,
            truncated_tail_mass=float(k.truncated_tail_mass),
        )

    return results


def _get_adstock_alpha(posterior: Any, channel: str) -> np.ndarray | None:
    """Extract the adstock decay ``alpha`` for a channel.

    The parametric path writes ``adstock_alpha_<ch>`` (geometric/delayed decay),
    so try that FIRST — otherwise the bare ``adstock_<ch>`` prefix would miss it
    and the cool-down would silently fall back to a default. ``adstock_<ch>`` is
    only the *legacy* non-parametric mixing weight (a Beta mix between two fixed
    alphas, NOT a decay rate); it stays last so a parametric fit never reads a
    mix weight as decay. Weibull (``adstock_shape_``/``adstock_scale_``) has no
    single alpha and correctly returns ``None`` (caller uses a conservative
    washout).
    """
    if posterior is None:
        return None

    for prefix in ["adstock_alpha_", "alpha_", "decay_", "adstock_"]:
        name = f"{prefix}{channel}"
        if name in posterior:
            return _flatten_samples(posterior[name].values)

    return None


def _get_adstock_lmax(model: Any, channel: str) -> int:
    """Get l_max for a channel's adstock."""
    # Try from panel config
    if hasattr(model, "panel") and model.panel is not None:
        if hasattr(model.panel, "mff_config"):
            for mc in model.panel.mff_config.media_channels:
                if mc.name == channel:
                    return mc.adstock_lmax or 8

    # Try from model attribute
    if hasattr(model, "adstock_lmax"):
        return model.adstock_lmax

    # Default
    return 8


__all__ = [
    "compute_adstock_weights",
    "_get_adstock_alpha",
    "_get_adstock_lmax",
]

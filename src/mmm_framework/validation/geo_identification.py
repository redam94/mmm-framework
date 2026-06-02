"""Geo-based identification diagnostic (P2-1, honestly scoped).

Cross-geo spend variation is quasi-experimental identifying signal -- but only
when it actually exists and is exogenous. This module REPORTS whether a model's
geos carry enough differential spend to support geo-level inference; it does NOT
turn geo into a model term (geo-heterogeneous media coefficients are a separate,
deferred change). The verdict is a *necessary* condition for credible geo
identification, never a sufficient one (see ``caveat``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Below this coefficient of variation, per-geo spend is effectively uniform and
# carries no geo-level identifying variation; above ~0.3 is comfortably strong.
DEFAULT_CV_THRESHOLD = 0.15

_GEO_EXOGENEITY_CAVEAT = (
    "Cross-geo spend variation identifies effects only if geo-level spend is "
    "exogenous -- not the result of targeting higher-demand geographies. Where "
    "spend follows local demand, geo variation does NOT remove unobserved-demand "
    "confounding; anchor geo-level claims with a randomized geo-lift experiment."
)


@dataclass
class GeoSpendVariation:
    """Cross-geo spend dispersion for a single channel."""

    channel: str
    cv_across_geos: float  # coefficient of variation of per-geo total spend
    sufficient: bool  # cv >= threshold -> enough variation to inform geo inference

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "cv_across_geos": self.cv_across_geos,
            "sufficient": self.sufficient,
        }


@dataclass
class GeoIdentificationDiagnostic:
    """Whether cross-geo spend variation can support geo-level identification."""

    n_geos: int
    channels: list[GeoSpendVariation]
    cv_threshold: float
    caveat: str

    @property
    def weak_channels(self) -> list[str]:
        """Channels whose spend is too uniform across geos to inform inference."""
        return [c.channel for c in self.channels if not c.sufficient]

    @property
    def has_identifying_variation(self) -> bool:
        return any(c.sufficient for c in self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_geos": self.n_geos,
            "channels": [c.to_dict() for c in self.channels],
            "cv_threshold": self.cv_threshold,
            "weak_channels": self.weak_channels,
            "caveat": self.caveat,
        }


def geo_spend_variation_diagnostic(
    model: Any, cv_threshold: float = DEFAULT_CV_THRESHOLD
) -> GeoIdentificationDiagnostic:
    """Report per-channel cross-geo spend variation for a fitted/built model.

    Parameters
    ----------
    model
        A model exposing ``has_geo``, ``n_geos``, ``geo_idx`` (obs -> geo code),
        ``X_media_raw`` (n_obs x n_channels) and ``channel_names``.
    cv_threshold
        Minimum coefficient of variation of per-geo spend for a channel to be
        considered to carry geo-level identifying variation.

    Raises
    ------
    ValueError
        If the model is national (no geo dimension); the diagnostic only applies
        to multi-geo panels.
    """
    if not getattr(model, "has_geo", False) or int(getattr(model, "n_geos", 1)) < 2:
        raise ValueError(
            "Geo identification diagnostic requires a multi-geo model "
            "(has_geo=True and n_geos >= 2)."
        )

    X = np.asarray(model.X_media_raw, dtype=float)
    geo_idx = np.asarray(model.geo_idx).astype(int)
    names = list(model.channel_names)
    n_geos = int(model.n_geos)

    channels: list[GeoSpendVariation] = []
    for c, name in enumerate(names):
        per_geo = np.array(
            [X[geo_idx == g, c].sum() for g in range(n_geos)], dtype=float
        )
        mean = float(per_geo.mean())
        cv = float(per_geo.std() / mean) if mean > 0 else 0.0
        channels.append(
            GeoSpendVariation(
                channel=name, cv_across_geos=cv, sufficient=cv >= cv_threshold
            )
        )

    return GeoIdentificationDiagnostic(
        n_geos=n_geos,
        channels=channels,
        cv_threshold=cv_threshold,
        caveat=_GEO_EXOGENEITY_CAVEAT,
    )


__all__ = [
    "GeoSpendVariation",
    "GeoIdentificationDiagnostic",
    "geo_spend_variation_diagnostic",
    "DEFAULT_CV_THRESHOLD",
]

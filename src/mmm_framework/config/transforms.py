"""Adstock and saturation transformation configurations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import AdstockType, SaturationType
from .priors import PriorConfig


class AdstockConfig(BaseModel):
    """Configuration for adstock transformation."""

    type: AdstockType = AdstockType.GEOMETRIC
    l_max: int = Field(default=8, ge=1, le=52, description="Maximum lag weeks")
    normalize: bool = True

    # Prior configs for parameters
    alpha_prior: PriorConfig | None = None  # Decay rate (geometric / delayed)
    theta_prior: PriorConfig | None = None  # Peak/delay lag (delayed)
    shape_prior: PriorConfig | None = None  # Weibull shape k
    scale_prior: PriorConfig | None = None  # Weibull scale lambda

    model_config = {"extra": "forbid"}

    @classmethod
    def geometric(
        cls, l_max: int = 8, alpha_prior: PriorConfig | None = None
    ) -> AdstockConfig:
        """Geometric (monotonic, peak at lag 0) carryover."""
        return cls(
            type=AdstockType.GEOMETRIC,
            l_max=l_max,
            alpha_prior=alpha_prior or PriorConfig.beta(alpha=1, beta=3),
        )

    @classmethod
    def delayed(
        cls,
        l_max: int = 8,
        alpha_prior: PriorConfig | None = None,
        theta_prior: PriorConfig | None = None,
    ) -> AdstockConfig:
        """Delayed geometric carryover with the peak effect at lag ``theta``.

        Uses the Jin et al. (2017) kernel ``w_k = alpha ** ((k - theta) ** 2)``,
        which can represent media whose effect builds before it decays.
        """
        return cls(
            type=AdstockType.DELAYED,
            l_max=l_max,
            alpha_prior=alpha_prior or PriorConfig.beta(alpha=2, beta=2),
            theta_prior=theta_prior or PriorConfig.half_normal(sigma=2.0),
        )

    @classmethod
    def weibull(
        cls,
        l_max: int = 12,
        shape_prior: PriorConfig | None = None,
        scale_prior: PriorConfig | None = None,
    ) -> AdstockConfig:
        """Weibull (PDF) carryover: a flexible, possibly delayed decay shape.

        Shape ``k > 1`` yields a delayed peak; ``k == 1`` is exponential;
        ``k < 1`` is front-loaded. Scale ``lambda`` spreads the mass over lags.
        """
        return cls(
            type=AdstockType.WEIBULL,
            l_max=l_max,
            shape_prior=shape_prior or PriorConfig.gamma(alpha=2.0, beta=1.0),
            scale_prior=scale_prior or PriorConfig.gamma(alpha=2.0, beta=1.0),
        )

    @classmethod
    def none(cls) -> AdstockConfig:
        """No carryover (unit impulse)."""
        return cls(type=AdstockType.NONE, l_max=1)


class SaturationConfig(BaseModel):
    """Configuration for saturation transformation.

    Notes
    -----
    ``kappa`` (the Hill half-saturation point) is only weakly identified from
    observational data because it trades off against adstock decay and the
    coefficient (see :func:`mmm_framework.transforms.adstock.adstock_weights` and
    critique.md §3.6). To constrain it, anchor the prior to the observed spend:
    compute bounds with :meth:`compute_kappa_bounds_from_data` (using
    ``kappa_bounds_percentiles``) and pass them as ``kappa_lower`` / ``kappa_upper``
    to :func:`mmm_framework.mmm_extensions.components.priors.create_saturation_prior`,
    which then uses a bounded ``Uniform`` instead of the default weakly-informative
    prior. This keeps the curve's "elbow" inside the region the data covers. It is
    **opt-in** -- the default prior is unchanged unless you pass bounds.

    These fields apply to the **Hill** saturation path (available via
    :mod:`mmm_framework.mmm_extensions`). The core :class:`BayesianMMM` uses
    logistic saturation (a single ``sat_lam``) and does not read ``kappa``.
    """

    type: SaturationType = SaturationType.HILL

    # Hill function priors
    kappa_prior: PriorConfig | None = None  # Half-saturation point (EC50)
    slope_prior: PriorConfig | None = None  # Curve steepness
    beta_prior: PriorConfig | None = None  # Maximum effect scaling

    # Percentiles (probabilities in [0, 1]) used by compute_kappa_bounds_from_data
    # to derive data-driven kappa bounds for the Hill path.
    kappa_bounds_percentiles: tuple[float, float] = (0.1, 0.9)

    model_config = {"extra": "forbid"}

    @staticmethod
    def compute_kappa_bounds_from_data(
        x: object,
        percentiles: tuple[float, float] = (0.1, 0.9),
    ) -> tuple[float, float]:
        """Return ``(lower, upper)`` kappa bounds from observed spend percentiles.

        ``x`` is an array-like of (adstocked) spend for one channel. ``percentiles``
        are probabilities in ``[0, 1]``. Anchoring ``kappa`` to this range keeps
        the Hill curve's half-saturation point inside the support of the data,
        which is the most defensible way to tame adstock/saturation equifinality.
        Pass the result as ``kappa_lower`` / ``kappa_upper`` to
        :func:`mmm_framework.mmm_extensions.components.priors.create_saturation_prior`.

        Raises ``ValueError`` on degenerate input (empty, all-NaN, or a collapsed
        range) rather than silently returning a meaningless prior.
        """
        import numpy as np

        lo_p, hi_p = percentiles
        if not (0.0 <= lo_p < hi_p <= 1.0):
            raise ValueError(
                f"percentiles must satisfy 0 <= low < high <= 1, got {percentiles}"
            )
        arr = np.asarray(x, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("Cannot compute kappa bounds from empty/all-NaN data.")
        lower = float(np.quantile(arr, lo_p))
        upper = float(np.quantile(arr, hi_p))
        if not (upper > lower):
            raise ValueError(
                "Degenerate kappa bounds: the data has no spread between the "
                f"{lo_p:.0%} and {hi_p:.0%} percentiles (both = {lower}). "
                "Use a wider percentile range, or skip data-anchored kappa for "
                "this channel (omit the kappa bounds)."
            )
        return lower, upper

    @classmethod
    def hill(
        cls,
        kappa_prior: PriorConfig | None = None,
        slope_prior: PriorConfig | None = None,
        beta_prior: PriorConfig | None = None,
    ) -> SaturationConfig:
        return cls(
            type=SaturationType.HILL,
            kappa_prior=kappa_prior or PriorConfig.beta(alpha=2, beta=2),
            slope_prior=slope_prior or PriorConfig.half_normal(sigma=1.5),
            beta_prior=beta_prior or PriorConfig.half_normal(sigma=1.5),
        )

    @classmethod
    def none(cls) -> SaturationConfig:
        return cls(type=SaturationType.NONE)

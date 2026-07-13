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

        The **default scale prior adapts to the lag window**: it is
        ``Gamma(alpha=2.0, beta=2.0 / m)`` with prior mean
        ``m = max(2.0, (l_max - 9.0) / 2.0)`` lag units (weeks for weekly
        data), i.e. the legacy mean-2 prior for windows up to 13 lags, then
        half the window beyond ~9 lags (mean 8.5 at ``l_max=26``).
        Rationale, measured on the synth stress harness:

        * A fixed-mean prior (the old ``Gamma(2, 1)``, mean 2) puts nearly
          all its mass in the first few lags, so a long window (``l_max=26``)
          forces the sampler to fight the prior whenever the true kernel is
          slow -- measured as a divergence storm (71 divergences, r-hat 1.07)
          on a scale-6-9 truth. With this rule (mean 8.5 at ``l_max=26``) the
          same fit sampled cleanly (0-1 divergences, r-hat < 1.03 across
          seeds) and recovered contributions on par with a hand-informed
          prior under the same protocol.
        * The growth is *offset*, not proportional (``l_max / 3`` was also
          tested): media flighting is often periodic at 7-13 weeks, and a
          mid-size window (``l_max=12``) whose scale prior puts real mass at
          those delays lets one chain settle into an *aliased* kernel
          (delayed by one flighting period; r-hat ~1.8). Keeping the prior
          mean at the legacy value for short/mid windows avoids that mode
          while long windows still get the slow-carryover mass they need.

        The shape prior stays ``Gamma(2, 1)`` -- shape is dimensionless and
        window-independent (a tighter ``Gamma(3, 1.5)`` was tested and both
        failed to suppress the aliased mode and degraded long-window
        recovery). Explicitly passed priors are used unchanged.
        """
        scale_prior_mean = max(2.0, (l_max - 9.0) / 2.0)
        return cls(
            type=AdstockType.WEIBULL,
            l_max=l_max,
            shape_prior=shape_prior or PriorConfig.gamma(alpha=2.0, beta=1.0),
            scale_prior=scale_prior
            or PriorConfig.gamma(alpha=2.0, beta=2.0 / scale_prior_mean),
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

    The core :class:`BayesianMMM` honors ``type`` per channel: ``logistic``
    (the default, a single ``sat_lam_<ch>``), ``hill`` (``sat_half_<ch>`` and
    ``sat_slope_<ch>``), ``michaelis_menten`` / ``tanh`` (``sat_half_<ch>``),
    ``root`` (``sat_exponent_<ch>``), or ``none`` (identity). The Hill
    ``kappa``/``slope``/``beta`` prior fields are also read by
    :mod:`mmm_framework.mmm_extensions`.
    """

    type: SaturationType = SaturationType.HILL

    # Hill function priors
    kappa_prior: PriorConfig | None = None  # Half-saturation point (EC50)
    slope_prior: PriorConfig | None = None  # Curve steepness
    beta_prior: PriorConfig | None = None  # Maximum effect scaling

    # Logistic (1 - exp(-lam * x)) rate prior. ``None`` keeps the core model's
    # built-in ``Exponential(lam=0.5)`` -- the historical default -- so default
    # configs build a graph bit-identical to models from before saturation
    # types were honored.
    lam_prior: PriorConfig | None = None

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
    def logistic(cls, lam_prior: PriorConfig | None = None) -> SaturationConfig:
        """Logistic saturation ``1 - exp(-lam * x)`` (the core model default).

        With ``lam_prior=None`` the core model uses its historical
        ``Exponential(lam=0.5)`` prior on ``sat_lam_<ch>``, keeping default
        models bit-identical to those built before per-channel saturation
        types were honored.
        """
        return cls(type=SaturationType.LOGISTIC, lam_prior=lam_prior)

    @classmethod
    def michaelis_menten(
        cls, kappa_prior: PriorConfig | None = None
    ) -> SaturationConfig:
        """Michaelis-Menten saturation ``x / (x + k)`` (half-saturation ``k``)."""
        return cls(
            type=SaturationType.MICHAELIS_MENTEN,
            kappa_prior=kappa_prior or PriorConfig.beta(alpha=2, beta=2),
        )

    @classmethod
    def tanh(cls, kappa_prior: PriorConfig | None = None) -> SaturationConfig:
        """Tanh saturation ``tanh(x / k)`` (scale ``k``)."""
        return cls(
            type=SaturationType.TANH,
            kappa_prior=kappa_prior or PriorConfig.beta(alpha=2, beta=2),
        )

    @classmethod
    def root(cls, exponent_prior: PriorConfig | None = None) -> SaturationConfig:
        """Root / power saturation ``x ** k`` with ``0 < k < 1`` (concave).

        The classic econometric power-response curve: constant returns fall off
        as a power of the (adstocked, normalized) spend. The exponent ``k`` is
        the ``sat_exponent_<ch>`` RV; ``exponent_prior=None`` defaults to
        ``Beta(2, 2)``, which keeps ``k`` in ``(0, 1)`` so the curve is strictly
        concave (diminishing returns). ``k`` reuses the ``slope_prior`` field.

        For an S-shaped curve use :meth:`hill`; for a hyperbolic elbow use
        :meth:`michaelis_menten`.
        """
        return cls(
            type=SaturationType.ROOT,
            slope_prior=exponent_prior or PriorConfig.beta(alpha=2, beta=2),
        )

    @classmethod
    def none(cls) -> SaturationConfig:
        return cls(type=SaturationType.NONE)

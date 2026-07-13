"""Model-level configuration: hierarchy, seasonality, control selection, and ModelConfig."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import FitMethod, InferenceMethod, ModelSpecification, PriorType
from .events import EventsConfig
from .likelihood import LikelihoodConfig
from .priors import PriorConfig


class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical/panel model structure."""

    enabled: bool = True

    # Pooling dimensions
    pool_across_geo: bool = True
    pool_across_product: bool = True

    # Per-geo CHANNEL COEFFICIENTS (effectiveness), not just intercept offsets.
    # Off by default (the model keeps one shared beta per channel + per-geo
    # intercepts). When on (with geo data), each channel's beta is partial-pooled
    # across geos so geo-heterogeneous effectiveness is estimable — the pooled
    # model otherwise lands near a spend-weighted average and scrambles regional
    # ROI. See model/base._build_channel_betas_geo (V3).
    vary_media_by_geo: bool = False
    # Prior SD of per-geo log-effectiveness deviations (between-geo spread).
    media_geo_sigma: float = 0.3

    # Parameterization
    use_non_centered: bool = True  # Better for sparse groups
    non_centered_threshold: int = 20  # Min obs for centered

    # Hyperprior settings
    mu_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig(
            distribution=PriorType.NORMAL, params={"mu": 0, "sigma": 1}
        )
    )
    sigma_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig.half_normal(sigma=0.5)
    )

    model_config = {"extra": "forbid"}


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality components.

    Amplitude scale note: the Fourier coefficients get ``Normal(0, prior_sigma)``
    priors on **standardized** ``y``, so ``prior_sigma`` bounds how much of a
    standard deviation each harmonic may swing the KPI — it is the seasonal
    amplitude prior. The historic default (0.3) suits mild seasonality; raise it
    (e.g. 0.5–1.0) for strongly seasonal categories, or the seasonal signal gets
    squeezed into trend/media. Per-component overrides win over ``prior_sigma``.
    """

    yearly: int | None = 2  # Fourier order for yearly seasonality
    monthly: int | None = None
    weekly: int | None = None

    # Amplitude prior (sigma of the Normal prior on Fourier coefficients)
    prior_sigma: float = 0.3
    yearly_prior_sigma: float | None = None
    monthly_prior_sigma: float | None = None
    weekly_prior_sigma: float | None = None

    model_config = {"extra": "forbid"}

    def prior_sigma_for(self, component: str) -> float:
        """Amplitude prior sigma for ``component`` ('yearly'/'monthly'/'weekly'),
        falling back to the shared ``prior_sigma``. getattr defaults keep
        configs pickled before these fields existed loadable."""
        override = getattr(self, f"{component}_prior_sigma", None)
        return getattr(self, "prior_sigma", 0.3) if override is None else override


class ControlSelectionConfig(BaseModel):
    """Configuration for control variable selection."""

    method: Literal["none", "horseshoe", "spike_slab", "lasso"] = "none"

    # Horseshoe settings
    expected_nonzero: int = 3

    # Regularization strength (for lasso-like)
    regularization: float = 1.0

    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    """Complete model configuration."""

    # Functional form
    specification: ModelSpecification = ModelSpecification.ADDITIVE

    # Observation model (likelihood family + link + family params). Default
    # normal/identity reproduces the historical hard-coded pm.Normal likelihood
    # byte-for-byte. The built-in additive model fits only the Gaussian families
    # (normal/student_t) directly; non-Gaussian families (e.g. binomial for an
    # awareness model) are read by models that define their own observation block.
    likelihood: LikelihoodConfig = Field(default_factory=LikelihoodConfig)

    # Intercept prior: Normal(mu, sigma) on standardized y, so mu is measured in
    # KPI standard deviations from the mean (values beyond ±2 are extreme).
    intercept_prior_mu: float = 0.0
    intercept_prior_sigma: float = Field(default=0.5, gt=0)

    # DEFAULT media-effect prior parameterization (applies only to channels with
    # no experiment-calibrated ``roi_prior`` and no explicit
    # ``coefficient_prior``):
    #
    # * ``"coefficient"`` (default) — the historical ``beta_<ch> ~
    #   Gamma(mu=1.5, sigma=1)`` on the standardized-coefficient scale. Its
    #   *implied* prior ROI depends on each channel's spend and the KPI scale,
    #   so two channels get very different (and arbitrary) prior ROIs.
    # * ``"roi"`` — sample the channel's prior ROI directly
    #   (``roi_<ch> ~ LogNormal(media_roi_prior_mu, media_roi_prior_sigma)``,
    #   default median 1.0 = break-even) and derive ``beta_<ch>`` in-graph as
    #   ``roi * spend_total / (y_std * Σ saturation)`` — the prior lives on the
    #   decision scale and is comparable across channels regardless of
    #   spend/KPI units. Channels whose ROI divisor is non-monetary
    #   (efficiency basis) or zero-spend fall back to the coefficient default.
    #   The agent's spec-built models default to this mode (agents/fitting).
    media_prior_mode: Literal["coefficient", "roi"] = "coefficient"
    # LogNormal hyper-params of the ROI-mode prior (mu is log-scale: 0 → median
    # ROI 1.0; sigma 1.0 → 90% prior interval ≈ [0.19, 5.2]).
    media_roi_prior_mu: float = 0.0
    media_roi_prior_sigma: float = Field(default=1.0, gt=0)

    # DF-2: partial-pool the coefficients of channels sharing a `parent_channel`
    # group toward a shared (log-normal) group mean, so genuinely EXCHANGEABLE
    # collinear channels (e.g. several social platforms) borrow strength and
    # their split is regularized. OFF BY DEFAULT (R0.1): the media block is
    # byte-identical when disabled. Channels with a calibrated `roi_prior` or an
    # explicit `coefficient_prior` are excluded from the pool (their prior wins).
    # Collinearity ≠ "should be pooled" — only pool a priori exchangeable
    # channels; see technical-docs/future_implementation.md (DF-2).
    use_grouped_media_priors: bool = False

    # Inference settings
    inference_method: InferenceMethod = InferenceMethod.BAYESIAN_NUMPYRO

    # MCMC settings (for Bayesian)
    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 1000
    target_accept: float = 0.9

    # Default fit method when ``fit()`` is called without an explicit ``method``.
    # NUTS (full MCMC) is the default; the approximate methods (MAP / ADVI /
    # full-rank ADVI / Pathfinder) trade calibrated uncertainty for speed and
    # are meant for fast model checks, not final inference.
    fit_method: FitMethod = FitMethod.NUTS

    # Hierarchical structure
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)

    # Seasonality
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)

    # Holiday / event effects (#143). None ⇒ no event block (default). When set,
    # sharp date-specific effects enter as an additive `event_component`, distinct
    # from the smooth Fourier seasonality.
    events: EventsConfig | None = None

    # Control selection
    control_selection: ControlSelectionConfig = Field(
        default_factory=ControlSelectionConfig
    )

    # Adstock estimation strategy.
    # True (default since the 0.1.0 development line, 2026-06): estimate a
    #   continuous adstock kernel in-graph per channel, honoring each
    #   MediaChannelConfig.adstock (type/l_max/normalize and priors), which
    #   enables geometric, delayed, and Weibull carryover shapes. Made the
    #   default after the pressure-testing series measured the legacy blend at
    #   ~28% attribution error vs ~7% parametric on carryover-sensitive worlds.
    # False (legacy): fast two-point interpolation between fixed low/high
    #   geometric adstock. Set explicitly to reproduce pre-change fits;
    #   models pickled before this change keep their original behavior.
    use_parametric_adstock: bool = True

    # Frequentist settings
    ridge_alpha: float = 1.0
    bootstrap_samples: int = 1000

    # Optimization settings (for transformation search)
    optim_maxiter: int = 500
    optim_seed: int | None = 42

    model_config = {"extra": "forbid"}

    @property
    def is_bayesian(self) -> bool:
        return self.inference_method in [
            InferenceMethod.BAYESIAN_PYMC,
            InferenceMethod.BAYESIAN_NUMPYRO,
        ]

    @property
    def use_numpyro(self) -> bool:
        return self.inference_method == InferenceMethod.BAYESIAN_NUMPYRO

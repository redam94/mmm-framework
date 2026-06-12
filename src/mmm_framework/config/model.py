"""Model-level configuration: hierarchy, seasonality, control selection, and ModelConfig."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .enums import InferenceMethod, ModelSpecification, PriorType
from .priors import PriorConfig


class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical/panel model structure."""

    enabled: bool = True

    # Pooling dimensions
    pool_across_geo: bool = True
    pool_across_product: bool = True

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

    # Intercept prior: Normal(mu, sigma) on standardized y, so mu is measured in
    # KPI standard deviations from the mean (values beyond ±2 are extreme).
    intercept_prior_mu: float = 0.0
    intercept_prior_sigma: float = Field(default=0.5, gt=0)

    # Inference settings
    inference_method: InferenceMethod = InferenceMethod.BAYESIAN_NUMPYRO

    # MCMC settings (for Bayesian)
    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 1000
    target_accept: float = 0.9

    # Hierarchical structure
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)

    # Seasonality
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)

    # Control selection
    control_selection: ControlSelectionConfig = Field(
        default_factory=ControlSelectionConfig
    )

    # Adstock estimation strategy.
    # False (default): fast two-point interpolation between fixed low/high
    #   geometric adstock (legacy behavior).
    # True: estimate a continuous adstock kernel in-graph per channel, honoring
    #   each MediaChannelConfig.adstock (type/l_max/normalize and priors), which
    #   enables geometric, delayed, and Weibull carryover shapes.
    use_parametric_adstock: bool = False

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

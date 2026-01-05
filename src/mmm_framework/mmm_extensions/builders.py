"""
Builder Patterns for MMM Extension Configurations

Fluent API for constructing complex configurations with sensible defaults.
Each builder follows the pattern:
    config = Builder().with_x(...).with_y(...).build()
"""

from __future__ import annotations

from typing import Self
from .config import (
    # Enums
    MediatorType,
    CrossEffectType,
    EffectConstraint,
    SaturationType,
    # Configs
    AdstockConfig,
    SaturationConfig,
    EffectPriorConfig,
    MediatorConfig,
    OutcomeConfig,
    CrossEffectConfig,
    NestedModelConfig,
    MultivariateModelConfig,
    CombinedModelConfig,
    VariableSelectionMethod,
    HorseshoeConfig,
    SpikeSlabConfig,
    LassoConfig,
    VariableSelectionConfig,
)


# =============================================================================
# Base Configuration Builders
# =============================================================================


class AdstockConfigBuilder:
    """Builder for AdstockConfig."""

    def __init__(self):
        self._l_max = 8
        self._prior_type = "beta"
        self._prior_alpha = 2.0
        self._prior_beta = 2.0
        self._normalize = True

    def with_max_lag(self, l_max: int) -> Self:
        """Set maximum lag length."""
        self._l_max = l_max
        return self

    def with_beta_prior(self, alpha: float = 2.0, beta: float = 2.0) -> Self:
        """Use Beta prior for decay rate."""
        self._prior_type = "beta"
        self._prior_alpha = alpha
        self._prior_beta = beta
        return self

    def with_uniform_prior(self) -> Self:
        """Use Uniform(0,1) prior for decay rate."""
        self._prior_type = "uniform"
        return self

    def with_slow_decay(self) -> Self:
        """Configure for slow decay (long memory)."""
        self._prior_alpha = 3.0
        self._prior_beta = 1.0
        return self

    def with_fast_decay(self) -> Self:
        """Configure for fast decay (short memory)."""
        self._prior_alpha = 1.0
        self._prior_beta = 3.0
        return self

    def without_normalization(self) -> Self:
        """Disable weight normalization."""
        self._normalize = False
        return self

    def build(self) -> AdstockConfig:
        """Build the configuration."""
        return AdstockConfig(
            l_max=self._l_max,
            prior_type=self._prior_type,
            prior_alpha=self._prior_alpha,
            prior_beta=self._prior_beta,
            normalize=self._normalize,
        )


class SaturationConfigBuilder:
    """Builder for SaturationConfig."""

    def __init__(self):
        self._type = SaturationType.LOGISTIC
        self._lam_alpha = 3.0
        self._lam_beta = 1.0
        self._kappa_alpha = 2.0
        self._kappa_beta = 2.0
        self._slope_alpha = 3.0
        self._slope_beta = 1.0

    def logistic(self, lam_alpha: float = 3.0, lam_beta: float = 1.0) -> Self:
        """Use logistic saturation."""
        self._type = SaturationType.LOGISTIC
        self._lam_alpha = lam_alpha
        self._lam_beta = lam_beta
        return self

    def hill(
        self,
        kappa_alpha: float = 2.0,
        kappa_beta: float = 2.0,
        slope_alpha: float = 3.0,
        slope_beta: float = 1.0,
    ) -> Self:
        """Use Hill saturation."""
        self._type = SaturationType.HILL
        self._kappa_alpha = kappa_alpha
        self._kappa_beta = kappa_beta
        self._slope_alpha = slope_alpha
        self._slope_beta = slope_beta
        return self

    def with_strong_saturation(self) -> Self:
        """Configure for strong diminishing returns."""
        if self._type == SaturationType.LOGISTIC:
            self._lam_alpha = 5.0
            self._lam_beta = 1.0
        return self

    def with_weak_saturation(self) -> Self:
        """Configure for weak diminishing returns."""
        if self._type == SaturationType.LOGISTIC:
            self._lam_alpha = 1.0
            self._lam_beta = 2.0
        return self

    def build(self) -> SaturationConfig:
        """Build the configuration."""
        return SaturationConfig(
            type=self._type,
            lam_prior_alpha=self._lam_alpha,
            lam_prior_beta=self._lam_beta,
            kappa_prior_alpha=self._kappa_alpha,
            kappa_prior_beta=self._kappa_beta,
            slope_prior_alpha=self._slope_alpha,
            slope_prior_beta=self._slope_beta,
        )


class EffectPriorConfigBuilder:
    """Builder for EffectPriorConfig."""

    def __init__(self):
        self._constraint = EffectConstraint.NONE
        self._mu = 0.0
        self._sigma = 1.0

    def unconstrained(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Unconstrained effect (Normal prior)."""
        self._constraint = EffectConstraint.NONE
        self._mu = mu
        self._sigma = sigma
        return self

    def positive(self, sigma: float = 1.0) -> Self:
        """Positive-constrained effect (HalfNormal prior)."""
        self._constraint = EffectConstraint.POSITIVE
        self._sigma = sigma
        return self

    def negative(self, sigma: float = 1.0) -> Self:
        """Negative-constrained effect (-HalfNormal prior)."""
        self._constraint = EffectConstraint.NEGATIVE
        self._sigma = sigma
        return self

    def with_tight_prior(self) -> Self:
        """Tighter prior (more regularization)."""
        self._sigma *= 0.5
        return self

    def with_wide_prior(self) -> Self:
        """Wider prior (less regularization)."""
        self._sigma *= 2.0
        return self

    def build(self) -> EffectPriorConfig:
        """Build the configuration."""
        return EffectPriorConfig(
            constraint=self._constraint,
            mu=self._mu,
            sigma=self._sigma,
        )


# =============================================================================
# Mediator Builder
# =============================================================================


class MediatorConfigBuilder:
    """Builder for MediatorConfig."""

    def __init__(self, name: str):
        self._name = name
        self._type = MediatorType.PARTIALLY_OBSERVED
        self._media_effect = EffectPriorConfigBuilder().positive().build()
        self._outcome_effect = EffectPriorConfigBuilder().unconstrained().build()
        self._obs_noise = 0.1
        self._allow_direct = True
        self._direct_effect = (
            EffectPriorConfigBuilder().unconstrained(sigma=0.5).build()
        )
        self._apply_adstock = True
        self._apply_saturation = True
        self._adstock = AdstockConfigBuilder().build()
        self._saturation = SaturationConfigBuilder().build()

    # --- Mediator Type ---

    def fully_latent(self) -> Self:
        """Mediator is never observed (pure latent variable)."""
        self._type = MediatorType.FULLY_LATENT
        return self

    def partially_observed(self, observation_noise: float = 0.1) -> Self:
        """Mediator observed in some periods (e.g., surveys)."""
        self._type = MediatorType.PARTIALLY_OBSERVED
        self._obs_noise = observation_noise
        return self

    def fully_observed(self, observation_noise: float = 0.05) -> Self:
        """Mediator observed in all periods (e.g., traffic counters)."""
        self._type = MediatorType.FULLY_OBSERVED
        self._obs_noise = observation_noise
        return self

    # --- Effect Priors ---

    def with_media_effect(self, effect: EffectPriorConfig) -> Self:
        """Set media → mediator effect prior."""
        self._media_effect = effect
        return self

    def with_positive_media_effect(self, sigma: float = 1.0) -> Self:
        """Media should increase mediator (e.g., awareness)."""
        self._media_effect = EffectPriorConfigBuilder().positive(sigma).build()
        return self

    def with_outcome_effect(self, effect: EffectPriorConfig) -> Self:
        """Set mediator → outcome effect prior."""
        self._outcome_effect = effect
        return self

    # --- Direct Effects ---

    def with_direct_effect(self, sigma: float = 0.5) -> Self:
        """Allow direct media → outcome effect."""
        self._allow_direct = True
        self._direct_effect = (
            EffectPriorConfigBuilder().unconstrained(sigma=sigma).build()
        )
        return self

    def without_direct_effect(self) -> Self:
        """No direct effect (all media effect flows through mediator)."""
        self._allow_direct = False
        return self

    # --- Transformations ---

    def with_adstock(self, config: AdstockConfig) -> Self:
        """Set adstock configuration."""
        self._apply_adstock = True
        self._adstock = config
        return self

    def with_slow_adstock(self, l_max: int = 12) -> Self:
        """Configure slow-decaying adstock (awareness builds slowly)."""
        self._apply_adstock = True
        self._adstock = (
            AdstockConfigBuilder().with_max_lag(l_max).with_slow_decay().build()
        )
        return self

    def without_adstock(self) -> Self:
        """Disable adstock transformation."""
        self._apply_adstock = False
        return self

    def with_saturation(self, config: SaturationConfig) -> Self:
        """Set saturation configuration."""
        self._apply_saturation = True
        self._saturation = config
        return self

    def without_saturation(self) -> Self:
        """Disable saturation transformation."""
        self._apply_saturation = False
        return self

    def build(self) -> MediatorConfig:
        """Build the configuration."""
        return MediatorConfig(
            name=self._name,
            mediator_type=self._type,
            media_effect=self._media_effect,
            outcome_effect=self._outcome_effect,
            observation_noise_sigma=self._obs_noise,
            allow_direct_effect=self._allow_direct,
            direct_effect=self._direct_effect,
            apply_adstock=self._apply_adstock,
            apply_saturation=self._apply_saturation,
            adstock=self._adstock,
            saturation=self._saturation,
        )


# =============================================================================
# Outcome Builder
# =============================================================================


class OutcomeConfigBuilder:
    """Builder for OutcomeConfig."""

    def __init__(self, name: str, column: str | None = None):
        self._name = name
        self._column = column or name
        self._intercept_sigma = 2.0
        self._media_effect = EffectPriorConfigBuilder().unconstrained(sigma=0.5).build()
        self._include_trend = True
        self._include_seasonality = True

    def with_column(self, column: str) -> Self:
        """Set data column name."""
        self._column = column
        return self

    def with_intercept_prior(self, sigma: float) -> Self:
        """Set intercept prior scale."""
        self._intercept_sigma = sigma
        return self

    def with_media_effect(self, effect: EffectPriorConfig) -> Self:
        """Set media effect prior."""
        self._media_effect = effect
        return self

    def with_positive_media_effects(self, sigma: float = 0.5) -> Self:
        """Constrain media effects to be positive."""
        self._media_effect = EffectPriorConfigBuilder().positive(sigma).build()
        return self

    def with_trend(self) -> Self:
        """Include trend component."""
        self._include_trend = True
        return self

    def without_trend(self) -> Self:
        """Exclude trend component."""
        self._include_trend = False
        return self

    def with_seasonality(self) -> Self:
        """Include seasonality component."""
        self._include_seasonality = True
        return self

    def without_seasonality(self) -> Self:
        """Exclude seasonality component."""
        self._include_seasonality = False
        return self

    def build(self) -> OutcomeConfig:
        """Build the configuration."""
        return OutcomeConfig(
            name=self._name,
            column=self._column,
            intercept_prior_sigma=self._intercept_sigma,
            media_effect=self._media_effect,
            include_trend=self._include_trend,
            include_seasonality=self._include_seasonality,
        )


# =============================================================================
# Cross-Effect Builder
# =============================================================================


class CrossEffectConfigBuilder:
    """Builder for CrossEffectConfig."""

    def __init__(self, source: str, target: str):
        self._source = source
        self._target = target
        self._type = CrossEffectType.CANNIBALIZATION
        self._prior_sigma = 0.3
        self._promo_modulated = True
        self._promo_column = None
        self._lag = 0

    def cannibalization(self) -> Self:
        """Source reduces target sales (negative effect)."""
        self._type = CrossEffectType.CANNIBALIZATION
        return self

    def halo(self) -> Self:
        """Source increases target sales (positive effect)."""
        self._type = CrossEffectType.HALO
        return self

    def symmetric(self) -> Self:
        """Bidirectional effect."""
        self._type = CrossEffectType.SYMMETRIC
        return self

    def asymmetric(self) -> Self:
        """One-way effect (default)."""
        self._type = CrossEffectType.ASYMMETRIC
        return self

    def with_prior_sigma(self, sigma: float) -> Self:
        """Set effect prior scale."""
        self._prior_sigma = sigma
        return self

    def modulated_by_promotion(self, column: str | None = None) -> Self:
        """Effect only active when source is promoted."""
        self._promo_modulated = True
        self._promo_column = column
        return self

    def always_active(self) -> Self:
        """Effect always present (not promotion-modulated)."""
        self._promo_modulated = False
        self._promo_column = None
        return self

    def with_lag(self, lag: int) -> Self:
        """Set temporal lag (0 = contemporaneous)."""
        self._lag = lag
        return self

    def lagged(self) -> Self:
        """Use lagged effect (lag=1) for identification."""
        self._lag = 1
        return self

    def build(self) -> CrossEffectConfig:
        """Build the configuration."""
        return CrossEffectConfig(
            source_outcome=self._source,
            target_outcome=self._target,
            effect_type=self._type,
            prior_sigma=self._prior_sigma,
            promotion_modulated=self._promo_modulated,
            promotion_column=self._promo_column,
            lag=self._lag,
        )


# =============================================================================
# Top-Level Model Builders
# =============================================================================


class NestedModelConfigBuilder:
    """Builder for NestedModelConfig."""

    def __init__(self):
        self._mediators: list[MediatorConfig] = []
        self._media_to_mediator: dict[str, list[str]] = {}
        self._share_adstock = True
        self._share_saturation = False

    def add_mediator(self, mediator: MediatorConfig) -> Self:
        """Add a mediator to the model."""
        self._mediators.append(mediator)
        return self

    def with_awareness_mediator(
        self,
        name: str = "awareness",
        observation_noise: float = 0.15,
    ) -> Self:
        """Add awareness mediator with typical configuration."""
        mediator = (
            MediatorConfigBuilder(name)
            .partially_observed(observation_noise)
            .with_positive_media_effect()
            .with_slow_adstock()
            .build()
        )
        self._mediators.append(mediator)
        return self

    def with_traffic_mediator(
        self,
        name: str = "foot_traffic",
        observation_noise: float = 0.05,
    ) -> Self:
        """Add foot traffic mediator with typical configuration."""
        mediator = (
            MediatorConfigBuilder(name)
            .fully_observed(observation_noise)
            .with_positive_media_effect()
            .build()
        )
        self._mediators.append(mediator)
        return self

    def map_channels_to_mediator(
        self,
        mediator_name: str,
        channel_names: list[str],
    ) -> Self:
        """Specify which channels affect a mediator."""
        self._media_to_mediator[mediator_name] = channel_names
        return self

    def share_adstock(self, share: bool = True) -> Self:
        """Share adstock parameters across mediators."""
        self._share_adstock = share
        return self

    def share_saturation(self, share: bool = True) -> Self:
        """Share saturation parameters across mediators."""
        self._share_saturation = share
        return self

    def build(self) -> NestedModelConfig:
        """Build the configuration."""
        return NestedModelConfig(
            mediators=tuple(self._mediators),
            media_to_mediator_map={
                k: tuple(v) for k, v in self._media_to_mediator.items()
            },
            share_adstock_across_mediators=self._share_adstock,
            share_saturation_across_mediators=self._share_saturation,
        )


class MultivariateModelConfigBuilder:
    """Builder for MultivariateModelConfig."""

    def __init__(self):
        self._outcomes: list[OutcomeConfig] = []
        self._cross_effects: list[CrossEffectConfig] = []
        self._lkj_eta = 2.0
        self._share_adstock = True
        self._share_saturation = False
        self._share_trend = False
        self._share_seasonality = True

    def add_outcome(self, outcome: OutcomeConfig) -> Self:
        """Add an outcome to the model."""
        self._outcomes.append(outcome)
        return self

    def with_outcomes(self, *names: str) -> Self:
        """Add multiple outcomes with default configuration."""
        for name in names:
            self._outcomes.append(OutcomeConfigBuilder(name).build())
        return self

    def add_cross_effect(self, effect: CrossEffectConfig) -> Self:
        """Add a cross-effect between outcomes."""
        self._cross_effects.append(effect)
        return self

    def with_cannibalization(
        self,
        source: str,
        target: str,
        promotion_column: str | None = None,
    ) -> Self:
        """Add cannibalization effect."""
        effect = (
            CrossEffectConfigBuilder(source, target)
            .cannibalization()
            .modulated_by_promotion(promotion_column)
            .build()
        )
        self._cross_effects.append(effect)
        return self

    def with_halo_effect(self, source: str, target: str) -> Self:
        """Add halo effect."""
        effect = CrossEffectConfigBuilder(source, target).halo().always_active().build()
        self._cross_effects.append(effect)
        return self

    def with_lkj_eta(self, eta: float) -> Self:
        """Set LKJ correlation prior parameter."""
        self._lkj_eta = eta
        return self

    def with_weak_correlations(self) -> Self:
        """Prior favoring weak correlations (eta > 1)."""
        self._lkj_eta = 4.0
        return self

    def with_strong_correlations(self) -> Self:
        """Prior allowing strong correlations (eta < 1)."""
        self._lkj_eta = 0.5
        return self

    def share_media_adstock(self, share: bool = True) -> Self:
        """Share adstock parameters across outcomes."""
        self._share_adstock = share
        return self

    def share_media_saturation(self, share: bool = True) -> Self:
        """Share saturation parameters across outcomes."""
        self._share_saturation = share
        return self

    def share_trend(self, share: bool = True) -> Self:
        """Share trend parameters across outcomes."""
        self._share_trend = share
        return self

    def share_seasonality(self, share: bool = True) -> Self:
        """Share seasonality parameters across outcomes."""
        self._share_seasonality = share
        return self

    def build(self) -> MultivariateModelConfig:
        """Build the configuration."""
        return MultivariateModelConfig(
            outcomes=tuple(self._outcomes),
            cross_effects=tuple(self._cross_effects),
            lkj_eta=self._lkj_eta,
            share_media_adstock=self._share_adstock,
            share_media_saturation=self._share_saturation,
            share_trend=self._share_trend,
            share_seasonality=self._share_seasonality,
        )


class CombinedModelConfigBuilder:
    """Builder for combined nested + multivariate model."""

    def __init__(self):
        self._nested_builder = NestedModelConfigBuilder()
        self._mv_builder = MultivariateModelConfigBuilder()
        self._mediator_to_outcome: dict[str, list[str]] = {}

    # --- Delegate to nested builder ---

    def add_mediator(self, mediator: MediatorConfig) -> Self:
        self._nested_builder.add_mediator(mediator)
        return self

    def with_awareness_mediator(self, name: str = "awareness", **kwargs) -> Self:
        self._nested_builder.with_awareness_mediator(name, **kwargs)
        return self

    def with_traffic_mediator(self, name: str = "foot_traffic", **kwargs) -> Self:
        self._nested_builder.with_traffic_mediator(name, **kwargs)
        return self

    def map_channels_to_mediator(self, mediator: str, channels: list[str]) -> Self:
        self._nested_builder.map_channels_to_mediator(mediator, channels)
        return self

    # --- Delegate to multivariate builder ---

    def add_outcome(self, outcome: OutcomeConfig) -> Self:
        self._mv_builder.add_outcome(outcome)
        return self

    def with_outcomes(self, *names: str) -> Self:
        self._mv_builder.with_outcomes(*names)
        return self

    def with_cannibalization(self, source: str, target: str, **kwargs) -> Self:
        self._mv_builder.with_cannibalization(source, target, **kwargs)
        return self

    def with_halo_effect(self, source: str, target: str) -> Self:
        self._mv_builder.with_halo_effect(source, target)
        return self

    def with_lkj_eta(self, eta: float) -> Self:
        self._mv_builder.with_lkj_eta(eta)
        return self

    # --- Combined-specific ---

    def map_mediator_to_outcomes(
        self,
        mediator_name: str,
        outcome_names: list[str],
    ) -> Self:
        """Specify which outcomes are affected by a mediator."""
        self._mediator_to_outcome[mediator_name] = outcome_names
        return self

    def build(self) -> CombinedModelConfig:
        """Build the configuration."""
        return CombinedModelConfig(
            nested=self._nested_builder.build(),
            multivariate=self._mv_builder.build(),
            mediator_to_outcome_map={
                k: tuple(v) for k, v in self._mediator_to_outcome.items()
            },
        )


# =============================================================================
# Variable Selection Configuration Builders
# =============================================================================


class HorseshoeConfigBuilder:
    """
    Builder for HorseshoeConfig with sensible defaults.

    Examples
    --------
    >>> config = (HorseshoeConfigBuilder()
    ...     .with_expected_nonzero(5)
    ...     .with_slab_scale(2.5)
    ...     .with_heavy_tails()
    ...     .build())
    """

    def __init__(self):
        self._expected_nonzero = 3
        self._slab_scale = 2.0
        self._slab_df = 4.0
        self._local_df = 5.0
        self._global_df = 1.0

    def with_expected_nonzero(self, n: int) -> Self:
        """Set expected number of nonzero coefficients."""
        if n < 1:
            raise ValueError("expected_nonzero must be at least 1")
        self._expected_nonzero = n
        return self

    def with_slab_scale(self, scale: float) -> Self:
        """Set slab scale (max expected effect in std units)."""
        if scale <= 0:
            raise ValueError("slab_scale must be positive")
        self._slab_scale = scale
        return self

    def with_slab_df(self, df: float) -> Self:
        """Set slab degrees of freedom."""
        if df <= 0:
            raise ValueError("slab_df must be positive")
        self._slab_df = df
        return self

    def with_local_df(self, df: float) -> Self:
        """Set local shrinkage degrees of freedom."""
        if df <= 0:
            raise ValueError("local_df must be positive")
        self._local_df = df
        return self

    def with_global_df(self, df: float) -> Self:
        """Set global shrinkage degrees of freedom."""
        if df <= 0:
            raise ValueError("global_df must be positive")
        self._global_df = df
        return self

    def with_heavy_tails(self) -> Self:
        """Configure for heavier-tailed slab (allow larger effects)."""
        self._slab_df = 2.0
        return self

    def with_light_tails(self) -> Self:
        """Configure for lighter-tailed slab (more regularization)."""
        self._slab_df = 8.0
        return self

    def with_half_cauchy_local(self) -> Self:
        """Use half-Cauchy for local shrinkage (original horseshoe)."""
        self._local_df = 1.0
        return self

    def with_aggressive_shrinkage(self) -> Self:
        """Configure for more aggressive shrinkage of small effects."""
        self._local_df = 10.0
        self._slab_df = 6.0
        return self

    def build(self) -> HorseshoeConfig:
        """Build the HorseshoeConfig object."""
        return HorseshoeConfig(
            expected_nonzero=self._expected_nonzero,
            slab_scale=self._slab_scale,
            slab_df=self._slab_df,
            local_df=self._local_df,
            global_df=self._global_df,
        )


class SpikeSlabConfigBuilder:
    """
    Builder for SpikeSlabConfig with sensible defaults.

    Examples
    --------
    >>> config = (SpikeSlabConfigBuilder()
    ...     .with_prior_inclusion(0.3)
    ...     .with_sharp_selection()
    ...     .build())
    """

    def __init__(self):
        self._prior_inclusion_prob = 0.5
        self._spike_scale = 0.01
        self._slab_scale = 1.0
        self._use_continuous_relaxation = True
        self._temperature = 0.1

    def with_prior_inclusion(self, prob: float) -> Self:
        """Set prior inclusion probability."""
        if not 0 < prob < 1:
            raise ValueError("prior_inclusion must be in (0, 1)")
        self._prior_inclusion_prob = prob
        return self

    def with_spike_scale(self, scale: float) -> Self:
        """Set spike scale (should be small, e.g., 0.01)."""
        if scale <= 0:
            raise ValueError("spike_scale must be positive")
        self._spike_scale = scale
        return self

    def with_slab_scale(self, scale: float) -> Self:
        """Set slab scale (expected magnitude of nonzero effects)."""
        if scale <= 0:
            raise ValueError("slab_scale must be positive")
        self._slab_scale = scale
        return self

    def with_temperature(self, temp: float) -> Self:
        """Set temperature for continuous relaxation."""
        if temp <= 0:
            raise ValueError("temperature must be positive")
        self._temperature = temp
        return self

    def continuous(self) -> Self:
        """Use continuous relaxation (required for NUTS)."""
        self._use_continuous_relaxation = True
        return self

    def discrete(self) -> Self:
        """Use discrete selection (requires Gibbs sampler)."""
        self._use_continuous_relaxation = False
        return self

    def with_sharp_selection(self) -> Self:
        """Configure for sharper variable selection."""
        self._temperature = 0.05
        self._spike_scale = 0.005
        return self

    def with_soft_selection(self) -> Self:
        """Configure for softer variable selection."""
        self._temperature = 0.2
        self._spike_scale = 0.05
        return self

    def build(self) -> SpikeSlabConfig:
        """Build the SpikeSlabConfig object."""
        return SpikeSlabConfig(
            prior_inclusion_prob=self._prior_inclusion_prob,
            spike_scale=self._spike_scale,
            slab_scale=self._slab_scale,
            use_continuous_relaxation=self._use_continuous_relaxation,
            temperature=self._temperature,
        )


class LassoConfigBuilder:
    """
    Builder for LassoConfig.

    Examples
    --------
    >>> config = (LassoConfigBuilder()
    ...     .with_regularization(2.0)
    ...     .adaptive()
    ...     .build())
    """

    def __init__(self):
        self._regularization = 1.0
        self._adaptive = False

    def with_regularization(self, strength: float) -> Self:
        """Set regularization strength."""
        if strength <= 0:
            raise ValueError("regularization must be positive")
        self._regularization = strength
        return self

    def adaptive(self) -> Self:
        """Use adaptive LASSO with coefficient-specific penalties."""
        self._adaptive = True
        return self

    def non_adaptive(self) -> Self:
        """Use standard (non-adaptive) LASSO."""
        self._adaptive = False
        return self

    def with_strong_regularization(self) -> Self:
        """Configure for strong shrinkage."""
        self._regularization = 5.0
        return self

    def with_weak_regularization(self) -> Self:
        """Configure for weak shrinkage."""
        self._regularization = 0.5
        return self

    def build(self) -> LassoConfig:
        """Build the LassoConfig object."""
        return LassoConfig(
            regularization=self._regularization,
            adaptive=self._adaptive,
        )


class VariableSelectionConfigBuilder:
    """
    Builder for VariableSelectionConfig with fluent API.

    This is the main builder for configuring variable selection in MMM.

    CAUSAL WARNING: Variable selection should only be applied to precision
    controls, not confounders. Use exclude_confounders() to ensure confounders
    are always included with standard priors.

    Examples
    --------
    >>> # Regularized horseshoe with excluded confounders
    >>> config = (VariableSelectionConfigBuilder()
    ...     .regularized_horseshoe(expected_nonzero=5)
    ...     .with_slab_scale(2.0)
    ...     .exclude_confounders("distribution", "price", "competitor_media")
    ...     .build())

    >>> # Spike-and-slab for explicit inclusion probabilities
    >>> config = (VariableSelectionConfigBuilder()
    ...     .spike_slab(prior_inclusion=0.3)
    ...     .with_sharp_selection()
    ...     .apply_only_to("weather", "gas_price", "minor_holiday")
    ...     .build())

    >>> # Using sub-builders for full control
    >>> config = (VariableSelectionConfigBuilder()
    ...     .regularized_horseshoe()
    ...     .with_horseshoe_config(
    ...         HorseshoeConfigBuilder()
    ...         .with_expected_nonzero(3)
    ...         .with_heavy_tails()
    ...         .build()
    ...     )
    ...     .build())
    """

    def __init__(self):
        self._method = VariableSelectionMethod.NONE
        self._horseshoe = HorseshoeConfig()
        self._spike_slab = SpikeSlabConfig()
        self._lasso = LassoConfig()
        self._exclude_variables: list[str] = []
        self._include_only_variables: list[str] | None = None

    # --- Method selection ---

    def none(self) -> Self:
        """No variable selection (standard priors for all controls)."""
        self._method = VariableSelectionMethod.NONE
        return self

    def regularized_horseshoe(self, expected_nonzero: int = 3) -> Self:
        """
        Use regularized horseshoe prior (recommended default).

        The regularized horseshoe provides excellent shrinkage of small
        effects while preserving large effects, with a slab to prevent
        unrealistic coefficient magnitudes.

        Parameters
        ----------
        expected_nonzero : int
            Prior expectation of relevant control variables.
        """
        self._method = VariableSelectionMethod.REGULARIZED_HORSESHOE
        self._horseshoe = HorseshoeConfig(expected_nonzero=expected_nonzero)
        return self

    def finnish_horseshoe(self, expected_nonzero: int = 3) -> Self:
        """
        Use Finnish horseshoe prior.

        Mathematically equivalent to regularized horseshoe; name
        emphasizes the slab regularization from Piironen & Vehtari.

        Parameters
        ----------
        expected_nonzero : int
            Prior expectation of relevant control variables.
        """
        self._method = VariableSelectionMethod.FINNISH_HORSESHOE
        self._horseshoe = HorseshoeConfig(expected_nonzero=expected_nonzero)
        return self

    def spike_slab(
        self,
        prior_inclusion: float = 0.5,
        continuous: bool = True,
    ) -> Self:
        """
        Use spike-and-slab prior.

        Provides explicit posterior inclusion probabilities for each variable.

        Parameters
        ----------
        prior_inclusion : float
            Prior probability that each variable is included.
        continuous : bool
            Use continuous relaxation for NUTS sampling.
        """
        self._method = VariableSelectionMethod.SPIKE_SLAB
        self._spike_slab = SpikeSlabConfig(
            prior_inclusion_prob=prior_inclusion,
            use_continuous_relaxation=continuous,
        )
        return self

    def bayesian_lasso(self, regularization: float = 1.0) -> Self:
        """
        Use Bayesian LASSO prior.

        Better when expecting many small effects rather than sparse signals.

        Parameters
        ----------
        regularization : float
            Regularization strength (higher = more shrinkage).
        """
        self._method = VariableSelectionMethod.BAYESIAN_LASSO
        self._lasso = LassoConfig(regularization=regularization)
        return self

    # --- Horseshoe configuration ---

    def with_horseshoe_config(self, config: HorseshoeConfig) -> Self:
        """Set horseshoe configuration from pre-built config."""
        self._horseshoe = config
        return self

    def with_expected_nonzero(self, n: int) -> Self:
        """Set expected number of nonzero coefficients (horseshoe)."""
        self._horseshoe = HorseshoeConfig(
            expected_nonzero=n,
            slab_scale=self._horseshoe.slab_scale,
            slab_df=self._horseshoe.slab_df,
            local_df=self._horseshoe.local_df,
            global_df=self._horseshoe.global_df,
        )
        return self

    def with_slab_scale(self, scale: float) -> Self:
        """Set slab scale for horseshoe (max effect in std units)."""
        self._horseshoe = HorseshoeConfig(
            expected_nonzero=self._horseshoe.expected_nonzero,
            slab_scale=scale,
            slab_df=self._horseshoe.slab_df,
            local_df=self._horseshoe.local_df,
            global_df=self._horseshoe.global_df,
        )
        return self

    def with_slab_df(self, df: float) -> Self:
        """Set slab degrees of freedom (horseshoe)."""
        self._horseshoe = HorseshoeConfig(
            expected_nonzero=self._horseshoe.expected_nonzero,
            slab_scale=self._horseshoe.slab_scale,
            slab_df=df,
            local_df=self._horseshoe.local_df,
            global_df=self._horseshoe.global_df,
        )
        return self

    # --- Spike-slab configuration ---

    def with_spike_slab_config(self, config: SpikeSlabConfig) -> Self:
        """Set spike-slab configuration from pre-built config."""
        self._spike_slab = config
        return self

    def with_prior_inclusion(self, prob: float) -> Self:
        """Set prior inclusion probability (spike-slab)."""
        self._spike_slab = SpikeSlabConfig(
            prior_inclusion_prob=prob,
            spike_scale=self._spike_slab.spike_scale,
            slab_scale=self._spike_slab.slab_scale,
            use_continuous_relaxation=self._spike_slab.use_continuous_relaxation,
            temperature=self._spike_slab.temperature,
        )
        return self

    def with_temperature(self, temp: float) -> Self:
        """Set temperature for continuous spike-slab."""
        self._spike_slab = SpikeSlabConfig(
            prior_inclusion_prob=self._spike_slab.prior_inclusion_prob,
            spike_scale=self._spike_slab.spike_scale,
            slab_scale=self._spike_slab.slab_scale,
            use_continuous_relaxation=self._spike_slab.use_continuous_relaxation,
            temperature=temp,
        )
        return self

    def with_sharp_selection(self) -> Self:
        """Configure spike-slab for sharper selection."""
        self._spike_slab = SpikeSlabConfig(
            prior_inclusion_prob=self._spike_slab.prior_inclusion_prob,
            spike_scale=0.005,
            slab_scale=self._spike_slab.slab_scale,
            use_continuous_relaxation=self._spike_slab.use_continuous_relaxation,
            temperature=0.05,
        )
        return self

    # --- LASSO configuration ---

    def with_lasso_config(self, config: LassoConfig) -> Self:
        """Set LASSO configuration from pre-built config."""
        self._lasso = config
        return self

    def with_regularization(self, strength: float) -> Self:
        """Set LASSO regularization strength."""
        self._lasso = LassoConfig(
            regularization=strength,
            adaptive=self._lasso.adaptive,
        )
        return self

    # --- Variable scope ---

    def exclude_confounders(self, *variables: str) -> Self:
        """
        Exclude variables from selection (always include with standard priors).

        IMPORTANT: Use this for known confounders that affect both media
        spending and sales. These must be controlled for to identify
        causal effects and should NOT be subject to variable selection.

        Parameters
        ----------
        *variables : str
            Variable names to exclude from selection.
        """
        self._exclude_variables.extend(variables)
        return self

    def exclude(self, *variables: str) -> Self:
        """Alias for exclude_confounders."""
        return self.exclude_confounders(*variables)

    def apply_only_to(self, *variables: str) -> Self:
        """
        Apply selection only to specified variables.

        All other variables will use standard priors.

        Parameters
        ----------
        *variables : str
            Variable names to apply selection to.
        """
        self._include_only_variables = list(variables)
        return self

    def clear_exclusions(self) -> Self:
        """Clear all exclusions."""
        self._exclude_variables = []
        self._include_only_variables = None
        return self

    # --- Build ---

    def build(self) -> VariableSelectionConfig:
        """Build the VariableSelectionConfig object."""
        return VariableSelectionConfig(
            method=self._method,
            horseshoe=self._horseshoe,
            spike_slab=self._spike_slab,
            lasso=self._lasso,
            exclude_variables=tuple(self._exclude_variables),
            include_only_variables=(
                tuple(self._include_only_variables)
                if self._include_only_variables
                else None
            ),
        )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def awareness_mediator(
    name: str = "awareness",
    observation_noise: float = 0.15,
) -> MediatorConfig:
    """Create typical awareness mediator configuration."""
    return (
        MediatorConfigBuilder(name)
        .partially_observed(observation_noise)
        .with_positive_media_effect()
        .with_slow_adstock(l_max=12)
        .build()
    )


def foot_traffic_mediator(
    name: str = "foot_traffic",
    observation_noise: float = 0.05,
) -> MediatorConfig:
    """Create typical foot traffic mediator configuration."""
    return (
        MediatorConfigBuilder(name)
        .fully_observed(observation_noise)
        .with_positive_media_effect()
        .build()
    )


def cannibalization_effect(
    source: str,
    target: str,
    promotion_column: str | None = None,
    lagged: bool = False,
) -> CrossEffectConfig:
    """Create cannibalization cross-effect configuration."""
    builder = CrossEffectConfigBuilder(source, target).cannibalization()

    if promotion_column:
        builder.modulated_by_promotion(promotion_column)

    if lagged:
        builder.lagged()

    return builder.build()


def halo_effect(source: str, target: str) -> CrossEffectConfig:
    """Create halo cross-effect configuration."""
    return CrossEffectConfigBuilder(source, target).halo().always_active().build()


# =============================================================================
# Factory Functions
# =============================================================================


def sparse_controls(
    expected_nonzero: int = 3,
    *confounders: str,
) -> VariableSelectionConfig:
    """
    Create sparse control selection configuration.

    Convenience factory for the most common use case: expecting only
    a few control variables are truly relevant.

    Parameters
    ----------
    expected_nonzero : int
        Prior expectation of relevant controls.
    *confounders : str
        Confounder variable names to exclude from selection.

    Returns
    -------
    VariableSelectionConfig
        Configuration for regularized horseshoe selection.

    Examples
    --------
    >>> config = sparse_controls(3, "distribution", "price")
    """
    return (
        VariableSelectionConfigBuilder()
        .regularized_horseshoe(expected_nonzero=expected_nonzero)
        .exclude_confounders(*confounders)
        .build()
    )


def selection_with_inclusion_probs(
    prior_inclusion: float = 0.5,
    *confounders: str,
) -> VariableSelectionConfig:
    """
    Create selection configuration with explicit inclusion probabilities.

    Parameters
    ----------
    prior_inclusion : float
        Prior probability of inclusion for each variable.
    *confounders : str
        Confounder variable names to exclude from selection.

    Returns
    -------
    VariableSelectionConfig
        Configuration for spike-slab selection.
    """
    return (
        VariableSelectionConfigBuilder()
        .spike_slab(prior_inclusion=prior_inclusion)
        .exclude_confounders(*confounders)
        .build()
    )


def dense_controls(
    regularization: float = 1.0,
    *confounders: str,
) -> VariableSelectionConfig:
    """
    Create dense control selection configuration.

    Use when expecting many controls have small effects.

    Parameters
    ----------
    regularization : float
        Regularization strength.
    *confounders : str
        Confounder variable names to exclude from selection.

    Returns
    -------
    VariableSelectionConfig
        Configuration for Bayesian LASSO selection.
    """
    return (
        VariableSelectionConfigBuilder()
        .bayesian_lasso(regularization=regularization)
        .exclude_confounders(*confounders)
        .build()
    )

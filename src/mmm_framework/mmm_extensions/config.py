"""
Configuration Classes for MMM Extensions

Immutable configuration objects for nested and multivariate models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================

class MediatorType(str, Enum):
    """Type of mediating variable based on observability."""
    FULLY_LATENT = "fully_latent"
    PARTIALLY_OBSERVED = "partially_observed"
    FULLY_OBSERVED = "fully_observed"


class CrossEffectType(str, Enum):
    """Type of cross-product effect."""
    CANNIBALIZATION = "cannibalization"
    HALO = "halo"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class EffectConstraint(str, Enum):
    """Constraint on effect direction."""
    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class SaturationType(str, Enum):
    """Type of saturation function."""
    LOGISTIC = "logistic"
    HILL = "hill"


# =============================================================================
# Base Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class AdstockConfig:
    """Configuration for adstock transformation."""
    l_max: int = 8
    prior_type: str = "beta"
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    normalize: bool = True


@dataclass(frozen=True)
class SaturationConfig:
    """Configuration for saturation transformation."""
    type: SaturationType = SaturationType.LOGISTIC
    # Logistic params
    lam_prior_alpha: float = 3.0
    lam_prior_beta: float = 1.0
    # Hill params
    kappa_prior_alpha: float = 2.0
    kappa_prior_beta: float = 2.0
    slope_prior_alpha: float = 3.0
    slope_prior_beta: float = 1.0


@dataclass(frozen=True)
class EffectPriorConfig:
    """Configuration for effect coefficient prior."""
    constraint: EffectConstraint = EffectConstraint.NONE
    mu: float = 0.0
    sigma: float = 1.0


# =============================================================================
# Mediator Configuration
# =============================================================================

@dataclass(frozen=True)
class MediatorConfig:
    """Configuration for a mediating variable."""
    name: str
    mediator_type: MediatorType = MediatorType.PARTIALLY_OBSERVED
    
    # Media → Mediator effect prior
    media_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE,
            sigma=1.0
        )
    )
    
    # Mediator → Outcome effect prior
    outcome_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.NONE,
            sigma=1.0
        )
    )
    
    # Observation model parameters
    observation_noise_sigma: float = 0.1
    
    # Direct effect (media → outcome, bypassing mediator)
    allow_direct_effect: bool = True
    direct_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(sigma=0.5)
    )
    
    # Transformations for media → mediator pathway
    apply_adstock: bool = True
    apply_saturation: bool = True
    adstock: AdstockConfig = field(default_factory=AdstockConfig)
    saturation: SaturationConfig = field(default_factory=SaturationConfig)


# =============================================================================
# Outcome Configuration
# =============================================================================

@dataclass(frozen=True)
class OutcomeConfig:
    """Configuration for an outcome variable."""
    name: str
    column: str
    
    # Outcome-specific priors
    intercept_prior_sigma: float = 2.0
    media_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(sigma=0.5)
    )
    
    # Component inclusion
    include_trend: bool = True
    include_seasonality: bool = True


# =============================================================================
# Cross-Effect Configuration
# =============================================================================

@dataclass(frozen=True)
class CrossEffectConfig:
    """Configuration for cross-product effects."""
    source_outcome: str
    target_outcome: str
    effect_type: CrossEffectType = CrossEffectType.CANNIBALIZATION
    
    # Prior
    prior_sigma: float = 0.3
    
    # Modulation
    promotion_modulated: bool = True
    promotion_column: str | None = None
    
    # Temporal structure
    lag: int = 0  # 0 = contemporaneous, 1 = lagged


# =============================================================================
# Top-Level Model Configurations
# =============================================================================

@dataclass(frozen=True)
class NestedModelConfig:
    """Configuration for nested/mediated model."""
    mediators: tuple[MediatorConfig, ...] = field(default_factory=tuple)
    
    # Which channels affect which mediators
    # If empty, all channels affect all mediators
    media_to_mediator_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    
    # Shared vs separate transformations
    share_adstock_across_mediators: bool = True
    share_saturation_across_mediators: bool = False


@dataclass(frozen=True)
class MultivariateModelConfig:
    """Configuration for multivariate outcome model."""
    outcomes: tuple[OutcomeConfig, ...] = field(default_factory=tuple)
    cross_effects: tuple[CrossEffectConfig, ...] = field(default_factory=tuple)
    
    # Correlation structure
    lkj_eta: float = 2.0
    
    # Parameter sharing
    share_media_adstock: bool = True
    share_media_saturation: bool = False
    share_trend: bool = False
    share_seasonality: bool = True


@dataclass(frozen=True)
class CombinedModelConfig:
    """Configuration for combined nested + multivariate model."""
    nested: NestedModelConfig
    multivariate: MultivariateModelConfig
    
    # Whether mediators affect all outcomes or specific ones
    mediator_to_outcome_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
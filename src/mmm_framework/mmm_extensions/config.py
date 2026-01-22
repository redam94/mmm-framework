"""
Configuration Classes for MMM Extensions

Immutable configuration objects for nested and multivariate models.

Note: Shared enums like SaturationType are imported from the main config module
to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# Import shared enum from main config to avoid duplication
from mmm_framework.config import SaturationType


# =============================================================================
# Enums
# =============================================================================


class MediatorType(str, Enum):
    """Type of mediating variable based on observability."""

    FULLY_OBSERVED = "fully_observed"      # Every period has observation
    PARTIALLY_OBSERVED = "partially_observed"  # Sparse point-in-time observations
    AGGREGATED_SURVEY = "aggregated_survey"    # Temporally aggregated with known n
    FULLY_LATENT = "fully_latent"


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


# SaturationType is imported from mmm_framework.config to ensure consistency
# across the codebase. The main config defines: HILL, LOGISTIC, MICHAELIS_MENTEN, TANH, NONE


# =============================================================================
# Variable Selection Enums
# =============================================================================


class VariableSelectionMethod(str, Enum):
    """Available variable selection methods for control variables."""

    NONE = "none"
    REGULARIZED_HORSESHOE = "regularized_horseshoe"
    FINNISH_HORSESHOE = "finnish_horseshoe"
    SPIKE_SLAB = "spike_slab"
    BAYESIAN_LASSO = "bayesian_lasso"


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
            constraint=EffectConstraint.POSITIVE, sigma=1.0
        )
    )

    # Mediator → Outcome effect prior
    outcome_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.NONE, sigma=1.0
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


# =============================================================================
# Variable Selection Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class HorseshoeConfig:
    """
    Configuration for horseshoe-family priors.

    The regularized horseshoe (Piironen & Vehtari, 2017) provides:
    - Strong shrinkage of small effects toward zero
    - Minimal shrinkage of large effects (signal preservation)
    - Regularized slab to prevent unrealistic effect sizes

    Parameters
    ----------
    expected_nonzero : int
        Prior expectation of the number of nonzero coefficients (D0).
        Used to calibrate the global shrinkage parameter tau.
    slab_scale : float
        Scale parameter for the slab (c in the formulation).
        Controls maximum expected coefficient magnitude in std units.
    slab_df : float
        Degrees of freedom for the slab's distribution.
        Lower = heavier tails = allow larger effects.
    local_df : float
        Degrees of freedom for local shrinkage parameters (lambda).
        Default 5.0; use 1.0 for half-Cauchy (original horseshoe).
    global_df : float
        Degrees of freedom for global shrinkage parameter (tau).
        Default 1.0 gives half-Cauchy (standard horseshoe).
    """

    expected_nonzero: int = 3
    slab_scale: float = 2.0
    slab_df: float = 4.0
    local_df: float = 5.0
    global_df: float = 1.0


@dataclass(frozen=True)
class SpikeSlabConfig:
    """
    Configuration for spike-and-slab priors.

    The spike-and-slab uses a mixture of two distributions:
    - Spike: concentrated near zero (for excluded variables)
    - Slab: diffuse prior (for included variables)

    Parameters
    ----------
    prior_inclusion_prob : float
        Prior probability that each coefficient is nonzero.
        0.5 represents maximum uncertainty about inclusion.
    spike_scale : float
        Standard deviation of the spike (near-zero distribution).
        Should be small (0.01-0.05) to effectively zero coefficients.
    slab_scale : float
        Standard deviation of the slab (nonzero distribution).
        Should reflect expected magnitude of true effects.
    use_continuous_relaxation : bool
        If True, use continuous relaxation for gradient-based sampling.
        Required for NUTS; set False only for Gibbs samplers.
    temperature : float
        Temperature for continuous relaxation (lower = sharper selection).
    """

    prior_inclusion_prob: float = 0.5
    spike_scale: float = 0.01
    slab_scale: float = 1.0
    use_continuous_relaxation: bool = True
    temperature: float = 0.1


@dataclass(frozen=True)
class LassoConfig:
    """
    Configuration for Bayesian LASSO prior.

    The Bayesian LASSO (Park & Casella, 2008) places Laplace priors
    on coefficients, providing L1-like shrinkage in a Bayesian context.

    Parameters
    ----------
    regularization : float
        Regularization strength (lambda). Higher = more shrinkage.
    adaptive : bool
        If True, use adaptive LASSO with coefficient-specific penalties.
    """

    regularization: float = 1.0
    adaptive: bool = False


@dataclass(frozen=True)
class VariableSelectionConfig:
    """
    Complete configuration for control variable selection.

    CAUSAL WARNING: Variable selection should ONLY be applied to precision
    control variables---variables that affect the outcome but do NOT affect
    treatment assignment (media spending). Applying selection to confounders
    can introduce severe bias in causal effect estimates.

    Parameters
    ----------
    method : VariableSelectionMethod
        Which selection method to use.
    horseshoe : HorseshoeConfig
        Configuration for horseshoe methods.
    spike_slab : SpikeSlabConfig
        Configuration for spike-and-slab.
    lasso : LassoConfig
        Configuration for Bayesian LASSO.
    exclude_variables : tuple[str, ...]
        Variables to EXCLUDE from selection (always include with standard priors).
        Use for known confounders that must remain in the model.
    include_only_variables : tuple[str, ...] | None
        If specified, only apply selection to these variables.
        All others use standard priors.

    Examples
    --------
    >>> # Sparse selection with excluded confounders
    >>> config = VariableSelectionConfig(
    ...     method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
    ...     horseshoe=HorseshoeConfig(expected_nonzero=3),
    ...     exclude_variables=("distribution", "price", "competitor_media"),
    ... )
    """

    method: VariableSelectionMethod = VariableSelectionMethod.NONE
    horseshoe: HorseshoeConfig = field(default_factory=HorseshoeConfig)
    spike_slab: SpikeSlabConfig = field(default_factory=SpikeSlabConfig)
    lasso: LassoConfig = field(default_factory=LassoConfig)
    exclude_variables: tuple[str, ...] = ()
    include_only_variables: tuple[str, ...] | None = None

    def get_selectable_variables(
        self,
        all_control_names: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Partition control variables into selectable and non-selectable.

        Parameters
        ----------
        all_control_names : list[str]
            All control variable names in the model.

        Returns
        -------
        tuple[list[str], list[str]]
            (variables_with_selection, variables_without_selection)
        """
        # Start with all controls or specified subset
        if self.include_only_variables is not None:
            selectable = [
                c for c in all_control_names if c in self.include_only_variables
            ]
        else:
            selectable = list(all_control_names)

        # Remove excluded variables
        selectable = [c for c in selectable if c not in self.exclude_variables]

        # Non-selectable = everything else
        non_selectable = [c for c in all_control_names if c not in selectable]

        return selectable, non_selectable


# =============================================================================
# Factory Functions for Common Configurations
# =============================================================================


def sparse_selection_config(
    expected_relevant: int = 3,
    confounders: tuple[str, ...] = (),
) -> VariableSelectionConfig:
    """
    Create configuration for sparse control selection.

    Use when you expect only a few controls are truly relevant.

    Parameters
    ----------
    expected_relevant : int
        Prior expectation of relevant controls.
    confounders : tuple[str, ...]
        Confounder variables to exclude from selection.
    """
    return VariableSelectionConfig(
        method=VariableSelectionMethod.REGULARIZED_HORSESHOE,
        horseshoe=HorseshoeConfig(
            expected_nonzero=expected_relevant,
            slab_scale=2.0,
        ),
        exclude_variables=confounders,
    )


def dense_selection_config(
    regularization: float = 1.0,
    confounders: tuple[str, ...] = (),
) -> VariableSelectionConfig:
    """
    Create configuration for dense control selection.

    Use when you expect many controls have small effects.

    Parameters
    ----------
    regularization : float
        Regularization strength.
    confounders : tuple[str, ...]
        Confounder variables to exclude from selection.
    """
    return VariableSelectionConfig(
        method=VariableSelectionMethod.BAYESIAN_LASSO,
        lasso=LassoConfig(regularization=regularization),
        exclude_variables=confounders,
    )


def inclusion_prob_selection_config(
    prior_inclusion: float = 0.5,
    confounders: tuple[str, ...] = (),
) -> VariableSelectionConfig:
    """
    Create configuration with explicit inclusion probabilities.

    Use when you want interpretable posterior inclusion probabilities.

    Parameters
    ----------
    prior_inclusion : float
        Prior probability of inclusion for each variable.
    confounders : tuple[str, ...]
        Confounder variables to exclude from selection.
    """
    return VariableSelectionConfig(
        method=VariableSelectionMethod.SPIKE_SLAB,
        spike_slab=SpikeSlabConfig(
            prior_inclusion_prob=prior_inclusion,
            temperature=0.1,
        ),
        exclude_variables=confounders,
    )

class MediatorObservationType(str, Enum):
    """
    How the mediator is observed.
    
    Extends the original MediatorType to add aggregated survey support.
    """
    FULLY_OBSERVED = "fully_observed"      # Every period has observation
    PARTIALLY_OBSERVED = "partially_observed"  # Sparse point-in-time observations
    AGGREGATED_SURVEY = "aggregated_survey"    # Temporally aggregated with known n
    FULLY_LATENT = "fully_latent"          # Never observed


class AggregatedSurveyLikelihood(str, Enum):
    """Likelihood for aggregated survey observations."""
    BINOMIAL = "binomial"      # Exact binomial (preferred)
    NORMAL = "normal"          # Normal approximation with derived SE
    BETA_BINOMIAL = "beta_binomial"  # Overdispersed binomial


@dataclass(frozen=True)
class AggregatedSurveyConfig:
    """
    Configuration for temporally aggregated survey observations.
    
    Used when surveys are fielded continuously over a period (e.g., monthly)
    and results are aggregated, rather than point-in-time snapshots.
    
    Attributes
    ----------
    aggregation_map : dict[int, tuple[int, ...]]
        Maps observation index to constituent time indices.
        E.g., {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)} for monthly surveys in weekly model.
    sample_sizes : tuple[int, ...]
        Number of respondents per survey wave. Length must match aggregation_map.
    likelihood : AggregatedSurveyLikelihood
        Which likelihood to use for the observation model.
    design_effect : float
        Survey design effect multiplier on variance (default 1.0).
        Use >1 for clustered samples, complex weighting, etc.
    aggregation_function : Literal["mean", "sum", "last"]
        How to aggregate latent values within each period.
        "mean" is typical for awareness (average state during fielding).
    overdispersion_prior_sigma : float
        Prior sigma for overdispersion parameter (beta-binomial only).
    """
    aggregation_map: dict[int, tuple[int, ...]]
    sample_sizes: tuple[int, ...]
    likelihood: AggregatedSurveyLikelihood = AggregatedSurveyLikelihood.BINOMIAL
    design_effect: float = 1.0
    aggregation_function: Literal["mean", "sum", "last"] = "mean"
    overdispersion_prior_sigma: float = 0.1
    
    def __post_init__(self):
        if len(self.sample_sizes) != len(self.aggregation_map):
            raise ValueError(
                f"sample_sizes length ({len(self.sample_sizes)}) must match "
                f"aggregation_map length ({len(self.aggregation_map)})"
            )
        if self.design_effect <= 0:
            raise ValueError("design_effect must be positive")


@dataclass(frozen=True)
class MediatorConfigExtended:
    """
    Extended MediatorConfig with aggregated survey support.
    
    This replaces the original MediatorConfig when aggregated surveys are needed.
    All original fields are preserved for backward compatibility.
    """
    name: str
    observation_type: MediatorObservationType = MediatorObservationType.PARTIALLY_OBSERVED
    
    # --- Original fields (from MediatorConfig) ---
    # Media → Mediator effect prior
    media_effect_constraint: str = "positive"  # "none", "positive", "negative"
    media_effect_sigma: float = 1.0
    
    # Mediator → Outcome effect prior
    outcome_effect_sigma: float = 1.0
    
    # Simple observation noise (for FULLY_OBSERVED and PARTIALLY_OBSERVED)
    observation_noise_sigma: float = 0.1
    
    # Direct effect settings
    allow_direct_effect: bool = True
    direct_effect_sigma: float = 0.5
    
    # Transformations
    apply_adstock: bool = True
    apply_saturation: bool = True
    
    # --- New field for aggregated surveys ---
    aggregated_survey_config: AggregatedSurveyConfig | None = None
    
    def __post_init__(self):
        if (self.observation_type == MediatorObservationType.AGGREGATED_SURVEY 
            and self.aggregated_survey_config is None):
            raise ValueError(
                "aggregated_survey_config is required when "
                "observation_type is AGGREGATED_SURVEY"
            )
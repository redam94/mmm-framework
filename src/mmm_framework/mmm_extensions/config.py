"""
Configuration Classes for MMM Extensions

Immutable configuration objects for nested and multivariate models.

Note: Shared enums like SaturationType are imported from the main config module
to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# Import shared enum from main config to avoid duplication
from mmm_framework.config import SaturationType

# =============================================================================
# Enums
# =============================================================================


class MediatorType(str, Enum):
    """Type of mediating variable based on observability."""

    FULLY_OBSERVED = "fully_observed"  # Every period has observation
    PARTIALLY_OBSERVED = "partially_observed"  # Sparse point-in-time observations
    AGGREGATED_SURVEY = "aggregated_survey"  # Temporally aggregated with known n
    FULLY_LATENT = "fully_latent"


class CrossEffectType(str, Enum):
    """Type of cross-product effect.

    ``CANNIBALIZATION`` (``psi = -HalfNormal``) and ``HALO`` (``psi = +HalfNormal``)
    impose the *sign* a priori. ``UNCONSTRAINED`` (``psi ~ Normal(0, sigma)``) lets the
    data choose the sign -- preferable when you do not want to assume the direction, and
    the honest default given that a one-sided prior makes "the posterior is below zero"
    near-automatic. Note that on *observed* sibling outcomes the directional cross-effect
    ``psi`` is confounded with the residual correlation (only their sum is identified), so
    an unconstrained ``psi`` measures a *cross-outcome association*, not causal
    cannibalization -- see :func:`mmm_framework.mmm_extensions.builders.cross_effect`.
    """

    CANNIBALIZATION = "cannibalization"
    HALO = "halo"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    UNCONSTRAINED = "unconstrained"


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

    FULLY_OBSERVED = "fully_observed"  # Every period has observation
    PARTIALLY_OBSERVED = "partially_observed"  # Sparse point-in-time observations
    AGGREGATED_SURVEY = "aggregated_survey"  # Temporally aggregated with known n
    FULLY_LATENT = "fully_latent"  # Never observed


class AggregatedSurveyLikelihood(str, Enum):
    """Likelihood for aggregated survey observations."""

    BINOMIAL = "binomial"  # Exact binomial (preferred)
    NORMAL = "normal"  # Normal approximation with derived SE
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


# =============================================================================
# Structural nested model configuration (StructuralNestedMMM)
# =============================================================================


class MediatorDynamics(str, Enum):
    """Latent-state dynamics of a mediator (or latent factor).

    ``STATIC``      z_t = level + drivers_t (no state carryover)
    ``AR1``         z_t = level + sum_{s<=t} rho^(t-s) * (drivers_s + sigma*eps_s)
    ``RANDOM_WALK`` AR1 with rho fixed at 1 (accumulating stock)
    """

    STATIC = "static"
    AR1 = "ar1"
    RANDOM_WALK = "random_walk"


class MediatorLikelihood(str, Enum):
    """Measurement family for a mediator's observed data.

    ``GAUSSIAN`` continuous index, z-scored, masked Normal observation
    ``BINOMIAL`` per-period success counts + per-period trials, logit link
    ``ORDERED``  per-period Likert category counts, cumulative-logit Multinomial
    ``LATENT``   never observed (state identified by in-graph standardization)
    """

    GAUSSIAN = "gaussian"
    BINOMIAL = "binomial"
    ORDERED = "ordered"
    LATENT = "latent"


@dataclass(frozen=True)
class MediatorMeasurement:
    """How a mediator's latent state is observed.

    Each measured family pins the latent's scale through its own geometry:
    GAUSSIAN defines the state on the standardized survey scale (loading fixed
    at 1), BINOMIAL pins it absolutely on the logit/probability scale, ORDERED
    anchors location in the cutpoints and scale against the unit-logistic
    response noise. ``design_effect`` deflates survey information for clustered
    or weighted samples (``n_eff = n / design_effect``).
    """

    likelihood: MediatorLikelihood = MediatorLikelihood.GAUSSIAN
    noise_sigma: float = 0.3
    design_effect: float = 1.0
    n_categories: int | None = None
    cutpoint_prior_sigma: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "likelihood", MediatorLikelihood(self.likelihood))
        if self.noise_sigma <= 0:
            raise ValueError("noise_sigma must be positive")
        if self.design_effect < 1.0:
            raise ValueError(
                "design_effect must be >= 1 (it deflates survey information; "
                "values below 1 would claim MORE precision than the sample size)"
            )
        if self.cutpoint_prior_sigma <= 0:
            raise ValueError("cutpoint_prior_sigma must be positive")
        if self.likelihood == MediatorLikelihood.ORDERED:
            if self.n_categories is None or self.n_categories < 3:
                raise ValueError(
                    "ORDERED measurement requires n_categories >= 3 (a 2-category "
                    "ordered scale is a binary outcome -- use BINOMIAL)"
                )


@dataclass(frozen=True)
class MediatorSpec:
    """One structural mediator equation in a StructuralNestedMMM.

    The latent state is driven by media channels (saturated, optionally
    adstocked), upstream mediators (``parents`` -- must form a DAG), control
    columns (e.g. price), and shared latent factors; its dynamics are STATIC,
    AR1, or RANDOM_WALK; and it is observed through ``measurement``.

    ``apply_adstock=None`` (default) resolves by dynamics: adstock ON for
    STATIC equations, OFF for AR1/RANDOM_WALK -- the state itself carries the
    media effect forward there, and adstock + AR would be two
    nearly-interchangeable geometric carryovers (an alpha/rho ridge). Set it
    explicitly to override (overriding to True on an AR equation warns at
    build).

    ``direct_effect`` defaults tight (Normal(0, 0.3)): an over-wide direct
    path steals the mediated signal (see technical-docs/nested-recovery-search.md).
    """

    name: str
    channels: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    latent_factors: tuple[str, ...] = ()

    dynamics: MediatorDynamics = MediatorDynamics.STATIC
    rho_prior_alpha: float = 6.0
    rho_prior_beta: float = 2.0
    innovation_sigma: float = 0.3
    # AR-noise sampling geometry for dynamic states: "auto" picks centered when
    # the measurement is dense (>= 50% of weeks observed -- a strong tracker
    # pins z_t and the non-centered form funnels), non-centered when sparse.
    state_parameterization: str = "auto"

    measurement: MediatorMeasurement = field(default_factory=MediatorMeasurement)

    media_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE, sigma=1.0
        )
    )
    parent_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE, sigma=1.0
        )
    )
    control_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(sigma=1.0)
    )
    # (Factor-loading prior scales live on LatentFactorSpec.mediator_effect_sigma
    # -- the loading belongs to the factor, shared across its consumers.)

    outcome_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(
            constraint=EffectConstraint.POSITIVE, sigma=1.0
        )
    )
    affects_outcome: bool = True

    allow_direct_effect: bool = True
    direct_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(sigma=0.3)
    )

    apply_adstock: bool | None = None

    @property
    def adstock_enabled(self) -> bool:
        """The resolved adstock setting (dynamics-dependent when unset)."""
        if self.apply_adstock is not None:
            return self.apply_adstock
        return self.dynamics == MediatorDynamics.STATIC

    def __post_init__(self):
        if not self.name:
            raise ValueError("MediatorSpec.name must be non-empty")
        if self.rho_prior_alpha <= 0 or self.rho_prior_beta <= 0:
            raise ValueError("rho prior alpha/beta must be positive")
        if self.innovation_sigma <= 0:
            raise ValueError("innovation_sigma must be positive")
        if self.state_parameterization not in ("auto", "centered", "non_centered"):
            raise ValueError(
                "state_parameterization must be 'auto', 'centered' or "
                f"'non_centered'; got {self.state_parameterization!r}"
            )
        has_driver = bool(
            self.channels or self.parents or self.controls or self.latent_factors
        )
        if not has_driver and self.measurement.likelihood == MediatorLikelihood.LATENT:
            raise ValueError(
                f"Mediator '{self.name}' has no drivers and no measurement -- "
                "a fully latent state with no inputs is unidentified"
            )
        if self.name in self.parents:
            raise ValueError(f"Mediator '{self.name}' cannot be its own parent")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(
                f"Mediator '{self.name}' lists duplicate parents: {self.parents}"
            )
        # Coerce string enums so downstream `.value` access is safe regardless
        # of how the spec was constructed.
        object.__setattr__(self, "dynamics", MediatorDynamics(self.dynamics))


@dataclass(frozen=True)
class LatentFactorSpec:
    """A shared latent factor (e.g. a demand trend) entering one or more
    mediator equations and/or the outcome.

    The realized series is standardized in-graph to unit variance (the scale
    would otherwise trade off against the loadings), so loadings carry the
    factor's units. Sign is anchored at the outcome loading (HalfNormal) when
    ``affects_outcome`` is True, else at the first consuming mediator's loading;
    all other loadings are free-sign Normal.
    """

    name: str
    dynamics: MediatorDynamics = MediatorDynamics.AR1
    rho_prior_alpha: float = 8.0
    rho_prior_beta: float = 2.0
    affects_outcome: bool = True
    outcome_effect_sigma: float = 1.0
    mediator_effect_sigma: float = 1.0
    # Where the factor's SIGN is pinned (that loading becomes HalfNormal).
    # "auto" = the first MEASURED mediator consumer (topological order), else
    # the outcome. Anchor where the loading is believed materially nonzero:
    # a reflected factor mode escapes a HalfNormal anchor whose true loading
    # is small by pushing it to the (cost-free) mode at zero -- observed as
    # split R-hat ~1.75 chains in the brand-funnel recovery. May name a
    # consuming mediator explicitly, or "outcome".
    anchor: str = "auto"

    def __post_init__(self):
        if not self.name:
            raise ValueError("LatentFactorSpec.name must be non-empty")
        if self.rho_prior_alpha <= 0 or self.rho_prior_beta <= 0:
            raise ValueError("rho prior alpha/beta must be positive")
        if self.outcome_effect_sigma <= 0 or self.mediator_effect_sigma <= 0:
            raise ValueError("effect sigmas must be positive")
        object.__setattr__(self, "dynamics", MediatorDynamics(self.dynamics))


@dataclass(frozen=True)
class StructuralNestedConfig:
    """Configuration for :class:`StructuralNestedMMM` -- a DAG of mediator
    equations with per-mediator dynamics + measurement, shared latent factors,
    and an outcome equation.

    ``outcome_controls=None`` means every control column provided to the model
    also enters the outcome equation. Channels not routed to any mediator get a
    plain direct effect with the ``nonmediated_effect`` prior.
    """

    mediators: tuple[MediatorSpec, ...] = field(default_factory=tuple)
    latent_factors: tuple[LatentFactorSpec, ...] = field(default_factory=tuple)
    outcome_controls: tuple[str, ...] | None = None
    nonmediated_effect: EffectPriorConfig = field(
        default_factory=lambda: EffectPriorConfig(sigma=1.0)
    )

    def __post_init__(self):
        if not self.mediators:
            raise ValueError("StructuralNestedConfig requires at least one mediator")
        med_names = [m.name for m in self.mediators]
        if len(set(med_names)) != len(med_names):
            raise ValueError(f"Duplicate mediator names: {med_names}")
        factor_names = [f.name for f in self.latent_factors]
        if len(set(factor_names)) != len(factor_names):
            raise ValueError(f"Duplicate latent factor names: {factor_names}")
        overlap = set(med_names) & set(factor_names)
        if overlap:
            raise ValueError(f"Names shared by mediators and factors: {overlap}")

        med_set = set(med_names)
        for m in self.mediators:
            unknown = set(m.parents) - med_set
            if unknown:
                raise ValueError(
                    f"Mediator '{m.name}' references unknown parents: {sorted(unknown)}"
                )
            unknown_f = set(m.latent_factors) - set(factor_names)
            if unknown_f:
                raise ValueError(
                    f"Mediator '{m.name}' references unknown latent factors: "
                    f"{sorted(unknown_f)}"
                )

        # Every factor needs >= 2 observation channels (measured mediators,
        # plus the outcome when affects_outcome): with only one, the factor is
        # confounded with that channel's own noise -- outcome-only makes it a
        # residual absorber, single-mediator-only makes it indistinguishable
        # from that mediator's process noise. Identification comes from
        # CO-MOVEMENT across channels.
        for f in self.latent_factors:
            measured_consumers = sum(
                1
                for m in self.mediators
                if f.name in m.latent_factors
                and m.measurement.likelihood != MediatorLikelihood.LATENT
            )
            n_channels = measured_consumers + (1 if f.affects_outcome else 0)
            if n_channels < 2 or measured_consumers < 1:
                raise ValueError(
                    f"Latent factor '{f.name}' needs at least two observation "
                    "channels including one measured mediator (e.g. a measured "
                    "mediator + the outcome) -- with fewer it is confounded "
                    "with a single equation's noise"
                )
            if f.anchor not in ("auto", "outcome"):
                consumer = next((m for m in self.mediators if m.name == f.anchor), None)
                if consumer is None or f.name not in consumer.latent_factors:
                    raise ValueError(
                        f"Latent factor '{f.name}' anchor {f.anchor!r} must be "
                        "'auto', 'outcome', or a mediator that consumes the factor"
                    )
            if f.anchor == "outcome" and not f.affects_outcome:
                raise ValueError(
                    f"Latent factor '{f.name}' anchor is 'outcome' but "
                    "affects_outcome is False"
                )

        # Acyclicity (raises on a cycle); the sorted order is reused at build.
        self.topological_order()

    def topological_order(self) -> list[str]:
        """Kahn topological sort of the mediator DAG (parents before children)."""
        med_by_name = {m.name: m for m in self.mediators}
        in_deg = {m.name: len(m.parents) for m in self.mediators}
        queue = sorted(n for n, d in in_deg.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for m in self.mediators:
                if node in m.parents:
                    in_deg[m.name] -= 1
                    if in_deg[m.name] == 0:
                        queue.append(m.name)
            queue.sort()
        if len(order) != len(med_by_name):
            cyclic = sorted(set(med_by_name) - set(order))
            raise ValueError(f"Mediator parent graph has a cycle involving: {cyclic}")
        return order


@dataclass(frozen=True)
class MediatorConfigExtended:
    """
    Extended MediatorConfig with aggregated survey support.

    This replaces the original MediatorConfig when aggregated surveys are needed.
    All original fields are preserved for backward compatibility.
    """

    name: str
    observation_type: MediatorObservationType = (
        MediatorObservationType.PARTIALLY_OBSERVED
    )

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
        if (
            self.observation_type == MediatorObservationType.AGGREGATED_SURVEY
            and self.aggregated_survey_config is None
        ):
            raise ValueError(
                "aggregated_survey_config is required when "
                "observation_type is AGGREGATED_SURVEY"
            )

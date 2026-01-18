"""
MMM Extensions Package

Provides nested/mediated models, multivariate outcome support,
and variable selection for precision controls in MMM.

Quick Start
-----------
```python
from mmm_extensions import (
    # Builders for configuration
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    VariableSelectionConfigBuilder,

    # Factory functions for common configurations
    awareness_mediator,
    sparse_controls,

    # Model classes
    NestedMMM,
    CombinedMMM,
)

# Build configuration with variable selection
selection_config = (
    VariableSelectionConfigBuilder()
    .regularized_horseshoe(expected_nonzero=3)
    .exclude_confounders("distribution", "price")
    .build()
)
```
"""

from typing import TYPE_CHECKING

# Enums and configs (no heavy dependencies)
from .config import (
    # Existing
    MediatorType,
    CrossEffectType,
    EffectConstraint,
    SaturationType,
    AdstockConfig,
    SaturationConfig,
    EffectPriorConfig,
    MediatorConfig,
    OutcomeConfig,
    CrossEffectConfig,
    NestedModelConfig,
    MultivariateModelConfig,
    CombinedModelConfig,
    MediatorObservationType,
    AggregatedSurveyLikelihood,
    AggregatedSurveyConfig,
    # Variable Selection
    VariableSelectionMethod,
    HorseshoeConfig,
    SpikeSlabConfig,
    LassoConfig,
    VariableSelectionConfig,
    sparse_selection_config,
    dense_selection_config,
    inclusion_prob_selection_config,
)

# Builders (no heavy dependencies)
from .builders import (
    # Existing
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    EffectPriorConfigBuilder,
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    AggregatedSurveyConfigBuilder,
    survey_awareness_mediator,
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
    # Variable Selection
    HorseshoeConfigBuilder,
    SpikeSlabConfigBuilder,
    LassoConfigBuilder,
    VariableSelectionConfigBuilder,
    sparse_controls,
    selection_with_inclusion_probs,
    dense_controls,
)

# Result containers
from .results import (
    MediationEffects,
    CrossEffectSummary,
    ModelResults,
    EffectDecomposition,
)

# Lazy imports for heavy dependencies (PyMC, PyTensor)
if TYPE_CHECKING:
    from .components import (
        # Transforms
        geometric_adstock,
        geometric_adstock_np,
        geometric_adstock_pt,
        geometric_adstock_convolution,
        geometric_adstock_matrix,
        logistic_saturation,
        hill_saturation,
        apply_transformation_pipeline,
        # Priors
        create_adstock_prior,
        create_saturation_prior,
        create_effect_prior,
        # Builders
        build_media_transforms,
        build_linear_effect,
        MediaTransformResult,
        EffectResult,
        # Observation
        build_gaussian_likelihood,
        build_partial_observation_model,
        build_multivariate_likelihood,
        build_aggregated_survey_observation,
        compute_survey_observation_indices,
        # Cross effects
        build_cross_effect_matrix,
        compute_cross_effect_contribution,
        CrossEffectSpec,
        # Variable Selection
        VariableSelectionResult,
        ControlEffectResult,
        create_regularized_horseshoe_prior,
        create_finnish_horseshoe_prior,
        create_spike_slab_prior,
        create_bayesian_lasso_prior,
        create_variable_selection_prior,
        build_control_effects_with_selection,
        compute_inclusion_probabilities,
        summarize_variable_selection,
    )
    from .models import (
        BaseExtendedMMM,
        NestedMMM,
        MultivariateMMM,
        CombinedMMM,
    )


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    # Components module exports
    components_exports = {
        # Transforms
        "geometric_adstock",
        "geometric_adstock_np",
        "geometric_adstock_pt",
        "geometric_adstock_convolution",
        "geometric_adstock_matrix",
        "logistic_saturation",
        "hill_saturation",
        "apply_transformation_pipeline",
        # Priors
        "create_adstock_prior",
        "create_saturation_prior",
        "create_effect_prior",
        # Builders
        "build_media_transforms",
        "build_linear_effect",
        "MediaTransformResult",
        "EffectResult",
        # Observation
        "build_gaussian_likelihood",
        "build_partial_observation_model",
        "build_multivariate_likelihood",
        "build_aggregated_survey_observation",
        "compute_survey_observation_indices",
        # Cross effects
        "build_cross_effect_matrix",
        "compute_cross_effect_contribution",
        "CrossEffectSpec",
        # Variable Selection
        "VariableSelectionResult",
        "ControlEffectResult",
        "create_regularized_horseshoe_prior",
        "create_finnish_horseshoe_prior",
        "create_spike_slab_prior",
        "create_bayesian_lasso_prior",
        "create_variable_selection_prior",
        "build_control_effects_with_selection",
        "compute_inclusion_probabilities",
        "summarize_variable_selection",
    }

    # Models module exports
    models_exports = {
        "BaseExtendedMMM",
        "NestedMMM",
        "MultivariateMMM",
        "CombinedMMM",
    }

    if name in components_exports:
        from . import components
        return getattr(components, name)

    if name in models_exports:
        from . import models
        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    # Enums
    "MediatorType",
    "CrossEffectType",
    "EffectConstraint",
    "SaturationType",
    "VariableSelectionMethod",
    "MediatorObservationType",
    "AggregatedSurveyLikelihood",
    # Config classes
    "AdstockConfig",
    "SaturationConfig",
    "EffectPriorConfig",
    "MediatorConfig",
    "OutcomeConfig",
    "CrossEffectConfig",
    "NestedModelConfig",
    "MultivariateModelConfig",
    "CombinedModelConfig",
    "HorseshoeConfig",
    "SpikeSlabConfig",
    "LassoConfig",
    "VariableSelectionConfig",
    "AggregatedSurveyConfig",
    # Builders
    "AdstockConfigBuilder",
    "SaturationConfigBuilder",
    "EffectPriorConfigBuilder",
    "MediatorConfigBuilder",
    "OutcomeConfigBuilder",
    "CrossEffectConfigBuilder",
    "NestedModelConfigBuilder",
    "MultivariateModelConfigBuilder",
    "CombinedModelConfigBuilder",
    "HorseshoeConfigBuilder",
    "SpikeSlabConfigBuilder",
    "LassoConfigBuilder",
    "VariableSelectionConfigBuilder",
    "AggregatedSurveyConfigBuilder",
    # Factory functions
    "awareness_mediator",
    "survey_awareness_mediator",
    "foot_traffic_mediator",
    "cannibalization_effect",
    "halo_effect",
    "sparse_controls",
    "selection_with_inclusion_probs",
    "dense_controls",
    "sparse_selection_config",
    "dense_selection_config",
    "inclusion_prob_selection_config",
    # Result containers
    "MediationEffects",
    "CrossEffectSummary",
    "ModelResults",
    "EffectDecomposition",
    # Components (lazy loaded)
    "geometric_adstock",
    "geometric_adstock_np",
    "geometric_adstock_pt",
    "geometric_adstock_convolution",
    "geometric_adstock_matrix",
    "logistic_saturation",
    "hill_saturation",
    "apply_transformation_pipeline",
    "create_adstock_prior",
    "create_saturation_prior",
    "create_effect_prior",
    "build_media_transforms",
    "build_linear_effect",
    "MediaTransformResult",
    "EffectResult",
    "build_gaussian_likelihood",
    "build_partial_observation_model",
    "build_multivariate_likelihood",
    "build_aggregated_survey_observation",
    "compute_survey_observation_indices",
    "build_cross_effect_matrix",
    "compute_cross_effect_contribution",
    "CrossEffectSpec",
    "VariableSelectionResult",
    "ControlEffectResult",
    "create_regularized_horseshoe_prior",
    "create_finnish_horseshoe_prior",
    "create_spike_slab_prior",
    "create_bayesian_lasso_prior",
    "create_variable_selection_prior",
    "build_control_effects_with_selection",
    "compute_inclusion_probabilities",
    "summarize_variable_selection",
    # Models (lazy loaded)
    "BaseExtendedMMM",
    "NestedMMM",
    "MultivariateMMM",
    "CombinedMMM",
]

"""
Components subpackage for MMM Extensions.

This package provides modular, composable building blocks for
nested and multivariate models using strategy patterns.

Contains both NumPy (for preprocessing) and PyTensor (for model building)
versions of transforms.
"""

# Re-export transforms from the main transforms module (NumPy versions)
from mmm_framework.transforms import (
    geometric_adstock as geometric_adstock_np_base,
    geometric_adstock_2d,
    logistic_saturation as logistic_saturation_np,
    create_fourier_features,
    create_bspline_basis,
)

# Local transform implementations specific to extensions (PyTensor versions)
from .transforms import (
    geometric_adstock_pt,
    geometric_adstock_convolution,
    geometric_adstock_matrix,
    apply_transformation_pipeline,
    hill_saturation,  # PyTensor version
    logistic_saturation_pt,  # PyTensor version
)

# In mmm_extensions.components, geometric_adstock should be the PyTensor version
# for backwards compatibility with existing usage
geometric_adstock = geometric_adstock_convolution
geometric_adstock_np = geometric_adstock_np_base

# Alias for the PyTensor version
logistic_saturation = logistic_saturation_pt

# Prior factories
from .priors import (
    create_adstock_prior,
    create_saturation_prior,
    create_effect_prior,
)

# Model component builders
from .builders import (
    MediaTransformResult,
    EffectResult,
    build_media_transforms,
    build_linear_effect,
)

# Observation model builders
from .observation import (
    build_gaussian_likelihood,
    build_partial_observation_model,
    build_multivariate_likelihood,
    build_aggregated_survey_observation,
    compute_survey_observation_indices,
    build_mediator_observation_dispatch,
)

# Cross-effect builders
from .cross_effects import (
    CrossEffectSpec,
    build_cross_effect_matrix,
    compute_cross_effect_contribution,
)

# Variable selection
from .variable_selection import (
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

__all__ = [
    # Transforms (from base)
    "geometric_adstock",
    "geometric_adstock_2d",
    "logistic_saturation",
    "hill_saturation",
    "create_fourier_features",
    "create_bspline_basis",
    # Transforms (local)
    "geometric_adstock_np",
    "geometric_adstock_pt",
    "geometric_adstock_convolution",
    "geometric_adstock_matrix",
    "apply_transformation_pipeline",
    # Priors
    "create_adstock_prior",
    "create_saturation_prior",
    "create_effect_prior",
    # Builders
    "MediaTransformResult",
    "EffectResult",
    "build_media_transforms",
    "build_linear_effect",
    # Observation models
    "build_gaussian_likelihood",
    "build_partial_observation_model",
    "build_multivariate_likelihood",
    "build_aggregated_survey_observation",
    "compute_survey_observation_indices",
    "build_mediator_observation_dispatch",
    # Cross-effects
    "CrossEffectSpec",
    "build_cross_effect_matrix",
    "compute_cross_effect_contribution",
    # Variable selection
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
]

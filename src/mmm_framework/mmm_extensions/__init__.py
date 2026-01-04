"""
MMM Extensions Package

Provides nested/mediated models and multivariate outcome support for MMM.

Quick Start
-----------
```python
from mmm_extensions import (
    # Builders for configuration
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    
    # Factory functions for common configurations
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
    
    # Model classes
    NestedMMM,
    MultivariateMMM,
    CombinedMMM,
)

# Build configuration using fluent API
config = (
    CombinedModelConfigBuilder()
    .with_awareness_mediator()
    .with_outcomes("single_pack", "multipack")
    .with_cannibalization("multipack", "single_pack", promotion_column="multi_promo")
    .build()
)

# Create and fit model
model = CombinedMMM(
    X_media=X_media,
    outcome_data={"single_pack": y1, "multipack": y2},
    channel_names=["tv", "digital", "social"],
    config=config,
    mediator_data={"awareness": survey_data},
)
results = model.fit()
```
"""

from typing import TYPE_CHECKING

# Enums and configs (no heavy dependencies)
from .config import (
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
)

# Builders (no heavy dependencies)
from .builders import (
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    EffectPriorConfigBuilder,
    MediatorConfigBuilder,
    OutcomeConfigBuilder,
    CrossEffectConfigBuilder,
    NestedModelConfigBuilder,
    MultivariateModelConfigBuilder,
    CombinedModelConfigBuilder,
    # Factory functions
    awareness_mediator,
    foot_traffic_mediator,
    cannibalization_effect,
    halo_effect,
)

# Lazy imports for heavy dependencies (PyMC, PyTensor)
if TYPE_CHECKING:
    from .components import (
        geometric_adstock,
        logistic_saturation,
        hill_saturation,
        apply_transformation_pipeline,
        create_adstock_prior,
        create_saturation_prior,
        create_effect_prior,
        build_media_transforms,
        build_linear_effect,
        build_gaussian_likelihood,
        build_partial_observation_model,
        build_multivariate_likelihood,
        build_cross_effect_matrix,
        compute_cross_effect_contribution,
        MediaTransformResult,
        EffectResult,
        CrossEffectSpec,
    )
    from .models import (
        BaseExtendedMMM,
        NestedMMM,
        MultivariateMMM,
        CombinedMMM,
        MediationEffects,
        CrossEffectSummary,
        ModelResults,
    )


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    # Components module
    components_exports = {
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
        "build_gaussian_likelihood",
        "build_partial_observation_model",
        "build_multivariate_likelihood",
        "build_cross_effect_matrix",
        "compute_cross_effect_contribution",
        "MediaTransformResult",
        "EffectResult",
        "CrossEffectSpec",
    }
    
    # Models module
    models_exports = {
        "BaseExtendedMMM",
        "NestedMMM",
        "MultivariateMMM",
        "CombinedMMM",
        "MediationEffects",
        "CrossEffectSummary",
        "ModelResults",
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
    # Factory functions
    "awareness_mediator",
    "foot_traffic_mediator",
    "cannibalization_effect",
    "halo_effect",
    # Components
    "geometric_adstock",
    "logistic_saturation",
    "hill_saturation",
    "apply_transformation_pipeline",
    "create_adstock_prior",
    "create_saturation_prior",
    "create_effect_prior",
    "build_media_transforms",
    "build_linear_effect",
    "build_gaussian_likelihood",
    "build_partial_observation_model",
    "build_multivariate_likelihood",
    "build_cross_effect_matrix",
    "compute_cross_effect_contribution",
    "MediaTransformResult",
    "EffectResult",
    "CrossEffectSpec",
    # Models
    "BaseExtendedMMM",
    "NestedMMM",
    "MultivariateMMM",
    "CombinedMMM",
    "MediationEffects",
    "CrossEffectSummary",
    "ModelResults",
]
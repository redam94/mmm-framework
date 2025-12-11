"""
Flexible MMM Framework

A modular Marketing Mix Model framework extending PyMC-Marketing.
Handles variable-dimension MFF data, multiplicative specifications with Hill saturation,
hierarchical panel structures, and fast frequentist alternatives.
"""

from .config import (
    # Enums
    DimensionType,
    VariableRole,
    AdstockType,
    SaturationType,
    PriorType,
    AllocationMethod,
    InferenceMethod,
    ModelSpecification,
    # Config classes
    PriorConfig,
    AdstockConfig,
    SaturationConfig,
    VariableConfig,
    MediaChannelConfig,
    ControlVariableConfig,
    KPIConfig,
    DimensionAlignmentConfig,
    MFFColumnConfig,
    MFFConfig,
    HierarchicalConfig,
    SeasonalityConfig,
    ControlSelectionConfig,
    ModelConfig,
    # Factory functions
    create_national_media_config,
    create_geo_media_config,
    create_social_platform_configs,
    create_simple_mff_config,
)

from .builders import (
    # Config builders
    PriorConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
    DimensionAlignmentConfigBuilder,
    MFFColumnConfigBuilder,
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ControlSelectionConfigBuilder,
    ModelConfigBuilder,
    MFFConfigBuilder,
    TrendConfigBuilder,
)

from .data_loader import (
    # Validation
    MFFValidationError,
    validate_mff_structure,
    validate_variable_dimensions,
    # Data containers
    PanelCoordinates,
    PanelDataset,
    # Loader
    MFFLoader,
    # Convenience functions
    load_mff,
    mff_from_wide_format,
)

from .model import (
    # Model classes
    TrendType,
    TrendConfig,
    MMMResults,
    PredictionResults,
    ContributionResults,
    ComponentDecomposition,
    BayesianMMM,
    # Helper functions
    create_fourier_features,
    geometric_adstock_np,
    geometric_adstock_2d,
    logistic_saturation_np,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)


__version__ = "0.1.0"

__all__ = [
    # Enums
    "DimensionType",
    "VariableRole", 
    "AdstockType",
    "SaturationType",
    "PriorType",
    "AllocationMethod",
    "InferenceMethod",
    "ModelSpecification",
    "TrendType",
    # Config classes
    "PriorConfig",
    "AdstockConfig",
    "SaturationConfig",
    "VariableConfig",
    "MediaChannelConfig",
    "ControlVariableConfig",
    "KPIConfig",
    "DimensionAlignmentConfig",
    "MFFColumnConfig",
    "MFFConfig",
    "HierarchicalConfig",
    "SeasonalityConfig",
    "ControlSelectionConfig",
    "ModelConfig",
    "TrendConfig",
    # Builders
    "PriorConfigBuilder",
    "AdstockConfigBuilder",
    "SaturationConfigBuilder",
    "MediaChannelConfigBuilder",
    "ControlVariableConfigBuilder",
    "KPIConfigBuilder",
    "DimensionAlignmentConfigBuilder",
    "MFFColumnConfigBuilder",
    "HierarchicalConfigBuilder",
    "SeasonalityConfigBuilder",
    "ControlSelectionConfigBuilder",
    "ModelConfigBuilder",
    "MFFConfigBuilder",
    "TrendConfigBuilder",
    # Factory functions
    "create_national_media_config",
    "create_geo_media_config",
    "create_social_platform_configs",
    "create_simple_mff_config",
    # Data loader
    "MFFValidationError",
    "validate_mff_structure",
    "validate_variable_dimensions",
    "PanelCoordinates",
    "PanelDataset",
    "MFFLoader",
    "load_mff",
    "mff_from_wide_format",
    # Model
    "MMMResults",
    "PredictionResults",
    "ContributionResults",
    "ComponentDecomposition",
    "BayesianMMM",
    "TrendType",
    "TrendConfig",
    "create_fourier_features",
    "geometric_adstock_np",
    "geometric_adstock_2d",
    "logistic_saturation_np",
    "create_bspline_basis",
    "create_piecewise_trend_matrix",
]
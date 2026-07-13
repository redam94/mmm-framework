"""
Configuration classes for flexible MMM framework.

Handles variable-dimension MFF data with configurable KPI, media, and control
specifications. Uses Pydantic for validation and type safety.

This subpackage is organized by concern:

- ``enums``       — dimension/role/transform/prior/inference enumerations
- ``priors``      — :class:`PriorConfig`
- ``transforms``  — :class:`AdstockConfig`, :class:`SaturationConfig`
- ``variables``   — per-variable configs (KPI, media, control)
- ``mff``         — MFF column mappings, dimension alignment, :class:`MFFConfig`
- ``model``       — model-level configuration (:class:`ModelConfig` and friends)
- ``factories``   — convenience builders for common configurations

All public names remain importable directly from ``mmm_framework.config`` for
backwards compatibility.
"""

from __future__ import annotations

from .enums import (
    AdstockType,
    AllocationMethod,
    CausalControlRole,
    DimensionType,
    FitMethod,
    InferenceMethod,
    LikelihoodFamily,
    LinkFunction,
    MeasurementUnit,
    ModelSpecification,
    PriorType,
    SaturationType,
    VariableRole,
)
from .events import EventSpec, EventsConfig
from .interactions import ChannelInteraction
from .likelihood import LikelihoodConfig
from .dataset import DATASET_SCHEMA_VERSION, DatasetSchema, RoleBinding
from .roles import DatasetRole
from .factories import (
    create_geo_media_config,
    create_national_media_config,
    create_simple_mff_config,
    create_social_platform_configs,
)
from .mff import (
    DimensionAlignmentConfig,
    MFFColumnConfig,
    MFFConfig,
)
from .model import (
    ControlSelectionConfig,
    HierarchicalConfig,
    ModelConfig,
    SeasonalityConfig,
)
from .priors import PriorConfig
from .spec_diff import SpecChange, diff_spec, summarize_spec_diff
from .transforms import AdstockConfig, SaturationConfig
from .variables import (
    ControlVariableConfig,
    KPIConfig,
    MediaChannelConfig,
    VariableConfig,
)

__all__ = [
    # Enums
    "DimensionType",
    "VariableRole",
    "CausalControlRole",
    "MeasurementUnit",
    "AdstockType",
    "SaturationType",
    "PriorType",
    "AllocationMethod",
    "FitMethod",
    "InferenceMethod",
    "ModelSpecification",
    "LikelihoodFamily",
    "LinkFunction",
    # Dataset role taxonomy + schema
    "DatasetRole",
    "DatasetSchema",
    "RoleBinding",
    "DATASET_SCHEMA_VERSION",
    # Prior / transform configs
    "PriorConfig",
    "EventSpec",
    "EventsConfig",
    "ChannelInteraction",
    "LikelihoodConfig",
    "AdstockConfig",
    "SaturationConfig",
    # Pre-spec lock + diff
    "SpecChange",
    "diff_spec",
    "summarize_spec_diff",
    # Variable configs
    "VariableConfig",
    "MediaChannelConfig",
    "ControlVariableConfig",
    "KPIConfig",
    # MFF configs
    "DimensionAlignmentConfig",
    "MFFColumnConfig",
    "MFFConfig",
    # Model configs
    "HierarchicalConfig",
    "SeasonalityConfig",
    "ControlSelectionConfig",
    "ModelConfig",
    # Factory functions
    "create_national_media_config",
    "create_geo_media_config",
    "create_social_platform_configs",
    "create_simple_mff_config",
]

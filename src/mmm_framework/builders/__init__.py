"""
Builder classes for MMM configuration objects.

Provides fluent API for constructing complex configuration objects step-by-step.

This subpackage organizes builders by domain:
- base: Shared mixins and protocols for all builders
- prior: Prior distribution and transformation config builders
- variable: Variable configuration builders (media, control, KPI)
- model: Model-level configuration builders
- mff: MFF data format configuration builders

All public builders are re-exported from this module for convenience.
"""

from __future__ import annotations

# Base utilities and mixins
from .base import (
    BuilderProtocol,
    VariableConfigBuilderMixin,
)

# Prior and transformation builders
from .prior import (
    PriorConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
)

# Variable configuration builders
from .variable import (
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
)

# Model configuration builders
from .model import (
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ControlSelectionConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    DimensionAlignmentConfigBuilder,
)

# MFF configuration builders
from .mff import (
    MFFColumnConfigBuilder,
    MFFConfigBuilder,
)


__all__ = [
    # Base utilities
    "BuilderProtocol",
    "VariableConfigBuilderMixin",
    # Prior builders
    "PriorConfigBuilder",
    "AdstockConfigBuilder",
    "SaturationConfigBuilder",
    # Variable builders
    "MediaChannelConfigBuilder",
    "ControlVariableConfigBuilder",
    "KPIConfigBuilder",
    # Model builders
    "HierarchicalConfigBuilder",
    "SeasonalityConfigBuilder",
    "ControlSelectionConfigBuilder",
    "ModelConfigBuilder",
    "TrendConfigBuilder",
    "DimensionAlignmentConfigBuilder",
    # MFF builders
    "MFFColumnConfigBuilder",
    "MFFConfigBuilder",
]

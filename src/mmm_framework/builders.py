"""
Builder classes for MMM configuration objects.

Provides fluent API for constructing complex configuration objects step-by-step.

Note: This module is maintained for backwards compatibility.
The builder implementations have been moved to the `builders/` subpackage.

Method Naming Conventions
-------------------------
This module follows these builder method naming patterns:

1. **with_* pattern** (preferred for setting values):
   - ``with_display_name(name)`` - Set a display name
   - ``with_unit(unit)`` - Set a unit of measurement
   - ``with_dimensions(*dims)`` - Set dimensions
   - ``with_prior(prior)`` - Set a prior configuration
   - ``with_max_lag(lag)`` - Set maximum lag

2. **Convenience methods** (shorthand for common configurations):
   - ``national()`` - Set as national-level (Period only)
   - ``by_geo()`` - Set as geo-level (Period + Geography)
   - ``by_product()`` - Set as product-level (Period + Product)
   - ``by_geo_and_product()`` - Set as geo+product level
   - ``enabled()`` / ``disabled()`` - Toggle boolean settings
   - ``positive_only()`` - Constrain to positive values

3. **Action methods**:
   - ``build()`` - Construct the final configuration object
   - ``add_*()`` - Add items to collections
"""

from __future__ import annotations

# Re-export all builders from subpackage for backwards compatibility
from .builders import (
    # Base utilities
    BuilderProtocol,
    VariableConfigBuilderMixin,
    # Prior builders
    PriorConfigBuilder,
    AdstockConfigBuilder,
    SaturationConfigBuilder,
    # Variable builders
    MediaChannelConfigBuilder,
    ControlVariableConfigBuilder,
    KPIConfigBuilder,
    # Model builders
    HierarchicalConfigBuilder,
    SeasonalityConfigBuilder,
    ControlSelectionConfigBuilder,
    ModelConfigBuilder,
    TrendConfigBuilder,
    DimensionAlignmentConfigBuilder,
    # MFF builders
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

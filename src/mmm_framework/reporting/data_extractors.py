"""
Data extractors for MMM report generation.

This module is maintained for backwards compatibility.
The extractor implementations have been moved to the `extractors/` subpackage.

Provides adapters to extract report data from various MMM model types:
- BayesianMMM (core framework)
- NestedMMM, MultivariateMMM, CombinedMMM (extensions)
- PyMC-Marketing MMM class

Each extractor converts model-specific data structures into a unified
MMMDataBundle that the report generator can consume.
"""

from __future__ import annotations

# Re-export all extractors from subpackage for backwards compatibility
from .extractors import (
    # Data bundle
    MMMDataBundle,
    # Base classes and protocols
    HasTrace,
    HasModel,
    DataExtractor,
    # Mixins
    AggregationMixin,
    GeoExtractionMixin,
    ProductExtractionMixin,
    # Concrete extractors
    BayesianMMMExtractor,
    ExtendedMMMExtractor,
    PyMCMarketingExtractor,
    # Factory
    create_extractor,
)

__all__ = [
    # Data bundle
    "MMMDataBundle",
    # Base classes and protocols
    "HasTrace",
    "HasModel",
    "DataExtractor",
    # Mixins
    "AggregationMixin",
    "GeoExtractionMixin",
    "ProductExtractionMixin",
    # Concrete extractors
    "BayesianMMMExtractor",
    "ExtendedMMMExtractor",
    "PyMCMarketingExtractor",
    # Factory
    "create_extractor",
]

"""
Data extractors for MMM report generation.

Provides adapters to extract report data from various MMM model types:
- BayesianMMM (core framework)
- NestedMMM, MultivariateMMM, CombinedMMM (extensions)
- PyMC-Marketing MMM class

Each extractor converts model-specific data structures into a unified
MMMDataBundle that the report generator can consume.

This subpackage organizes extractors by domain:
- bundle: MMMDataBundle data container
- base: DataExtractor ABC and protocols
- mixins: Shared extraction utilities (aggregation, geo-level)
- bayesian: BayesianMMMExtractor for core framework
- extended: ExtendedMMMExtractor for extension models
- pymc_marketing: PyMCMarketingExtractor for compatibility
"""

from __future__ import annotations

from typing import Any

# Data bundle
from .bundle import MMMDataBundle

# Base classes and protocols
from .base import (
    HasTrace,
    HasModel,
    DataExtractor,
)

# Mixins
from .mixins import (
    AggregationMixin,
    GeoExtractionMixin,
    ProductExtractionMixin,
)

# Concrete extractors
from .bayesian import BayesianMMMExtractor
from .extended import ExtendedMMMExtractor
from .pymc_marketing import PyMCMarketingExtractor


def create_extractor(model: Any, **kwargs) -> DataExtractor:
    """
    Factory function to create appropriate extractor for model type.

    Parameters
    ----------
    model : Any
        MMM model instance
    **kwargs
        Additional arguments passed to extractor

    Returns
    -------
    DataExtractor
        Appropriate extractor for the model type
    """
    model_type = type(model).__name__

    if model_type == "BayesianMMM":
        return BayesianMMMExtractor(model, **kwargs)
    elif model_type in ("NestedMMM", "MultivariateMMM", "CombinedMMM"):
        return ExtendedMMMExtractor(model, **kwargs)
    elif model_type == "MMM":
        # pymc-marketing MMM
        return PyMCMarketingExtractor(model, **kwargs)
    else:
        # Try BayesianMMM extractor as default
        return BayesianMMMExtractor(model, **kwargs)


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

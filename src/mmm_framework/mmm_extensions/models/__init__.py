"""
Models subpackage for MMM Extensions.

This package provides extended MMM model implementations:
- BaseExtendedMMM: Base class with common functionality
- NestedMMM: Nested/mediated causal pathways
- MultivariateMMM: Multiple correlated outcomes with cross-effects
- CombinedMMM: Combined nested + multivariate model
"""

from .base import BaseExtendedMMM
from .nested import NestedMMM
from .multivariate import MultivariateMMM
from .combined import CombinedMMM

# Re-export result containers for backwards compatibility
from ..results import (
    MediationEffects,
    CrossEffectSummary,
    ModelResults,
    EffectDecomposition,
)

__all__ = [
    "BaseExtendedMMM",
    "NestedMMM",
    "MultivariateMMM",
    "CombinedMMM",
    # Result containers
    "MediationEffects",
    "CrossEffectSummary",
    "ModelResults",
    "EffectDecomposition",
]

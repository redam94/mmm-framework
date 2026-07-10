"""
Models subpackage for MMM Extensions.

This package provides extended MMM model implementations:
- BaseExtendedMMM: Base class with common functionality
- NestedMMM: Nested/mediated causal pathways
- MultivariateMMM: Multiple correlated outcomes with cross-effects
- CombinedMMM: Combined nested + multivariate model
- StructuralNestedMMM: DAG of mediator equations with per-mediator
  dynamics/measurement + shared latent factors
"""

from .base import BaseExtendedMMM
from .nested import NestedMMM
from .multivariate import MultivariateMMM
from .combined import CombinedMMM
from .structural import StructuralNestedMMM

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
    "StructuralNestedMMM",
    # Result containers
    "MediationEffects",
    "CrossEffectSummary",
    "ModelResults",
    "EffectDecomposition",
]

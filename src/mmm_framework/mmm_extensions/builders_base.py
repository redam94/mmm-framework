"""
Base classes and utilities for extension builders.

This module provides shared functionality for extension builders that
operate on extension-specific config classes (frozen dataclasses).

Note: The extension builders use different config classes than the base
builders (frozen dataclasses vs Pydantic models), so they cannot directly
inherit. This module provides common patterns and utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from typing import Self

T = TypeVar("T")


@runtime_checkable
class ExtensionBuilderProtocol(Protocol[T]):
    """Protocol for extension configuration builders.

    All extension builders should implement a build() method that returns
    the final configuration object.
    """

    def build(self) -> T:
        """Build and return the configuration object."""
        ...


class AdstockBuilderMixin:
    """
    Mixin providing common adstock configuration methods.

    This mixin provides shared convenience methods for configuring adstock
    parameters. It can be used by any builder that configures adstock.

    Attributes
    ----------
    _l_max : int
        Maximum lag for adstock transformation.
    _prior_alpha : float
        Alpha parameter for Beta prior on decay rate.
    _prior_beta : float
        Beta parameter for Beta prior on decay rate.
    _normalize : bool
        Whether to normalize adstock weights.
    """

    _l_max: int
    _prior_alpha: float
    _prior_beta: float
    _normalize: bool

    def _init_adstock_fields(
        self,
        l_max: int = 8,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        normalize: bool = True,
    ) -> None:
        """Initialize adstock fields with defaults."""
        self._l_max = l_max
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._normalize = normalize

    def with_max_lag(self, l_max: int) -> Self:
        """Set maximum lag length.

        Parameters
        ----------
        l_max : int
            Maximum number of periods for carryover effects.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._l_max = l_max
        return self

    def with_slow_decay(self) -> Self:
        """Configure for slow decay (long memory).

        Sets Beta(3, 1) prior which favors higher decay rates,
        meaning effects persist longer.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._prior_alpha = 3.0
        self._prior_beta = 1.0
        return self

    def with_fast_decay(self) -> Self:
        """Configure for fast decay (short memory).

        Sets Beta(1, 3) prior which favors lower decay rates,
        meaning effects dissipate quickly.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._prior_alpha = 1.0
        self._prior_beta = 3.0
        return self

    def without_normalization(self) -> Self:
        """Disable weight normalization.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._normalize = False
        return self


class SaturationBuilderMixin:
    """
    Mixin providing common saturation configuration methods.

    This mixin provides shared convenience methods for configuring saturation
    parameters. It can be used by any builder that configures saturation.

    Attributes
    ----------
    _lam_alpha : float
        Alpha parameter for lambda prior (logistic saturation).
    _lam_beta : float
        Beta parameter for lambda prior.
    _kappa_alpha : float
        Alpha parameter for kappa prior (Hill saturation).
    _kappa_beta : float
        Beta parameter for kappa prior.
    _slope_alpha : float
        Alpha parameter for slope prior (Hill saturation).
    _slope_beta : float
        Beta parameter for slope prior.
    """

    _lam_alpha: float
    _lam_beta: float
    _kappa_alpha: float
    _kappa_beta: float
    _slope_alpha: float
    _slope_beta: float

    def _init_saturation_fields(
        self,
        lam_alpha: float = 3.0,
        lam_beta: float = 1.0,
        kappa_alpha: float = 2.0,
        kappa_beta: float = 2.0,
        slope_alpha: float = 3.0,
        slope_beta: float = 1.0,
    ) -> None:
        """Initialize saturation fields with defaults."""
        self._lam_alpha = lam_alpha
        self._lam_beta = lam_beta
        self._kappa_alpha = kappa_alpha
        self._kappa_beta = kappa_beta
        self._slope_alpha = slope_alpha
        self._slope_beta = slope_beta

    def with_strong_saturation(self) -> Self:
        """Configure for strong diminishing returns.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._lam_alpha = 5.0
        self._lam_beta = 1.0
        return self

    def with_weak_saturation(self) -> Self:
        """Configure for weak diminishing returns.

        Returns
        -------
        Self
            Builder instance for method chaining.
        """
        self._lam_alpha = 1.0
        self._lam_beta = 2.0
        return self


class EffectPriorMixin:
    """
    Mixin for configuring effect prior parameters.

    This mixin provides shared methods for configuring effect priors
    (constraint type and scale).
    """

    def _tight_prior(self, sigma: float) -> float:
        """Return tightened prior scale."""
        return sigma * 0.5

    def _wide_prior(self, sigma: float) -> float:
        """Return widened prior scale."""
        return sigma * 2.0


__all__ = [
    "ExtensionBuilderProtocol",
    "AdstockBuilderMixin",
    "SaturationBuilderMixin",
    "EffectPriorMixin",
]

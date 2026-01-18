"""
Prior and transformation configuration builders.

Provides builders for PriorConfig, AdstockConfig, and SaturationConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import (
    AdstockConfig,
    AdstockType,
    PriorConfig,
    PriorType,
    SaturationConfig,
    SaturationType,
)

if TYPE_CHECKING:
    from typing import Self


class PriorConfigBuilder:
    """
    Builder for PriorConfig objects.

    Examples
    --------
    >>> prior = (PriorConfigBuilder()
    ...     .half_normal(sigma=2.0)
    ...     .with_dims("channel")
    ...     .build())

    >>> prior = (PriorConfigBuilder()
    ...     .gamma(alpha=2, beta=1)
    ...     .build())
    """

    def __init__(self) -> None:
        self._distribution: PriorType | None = None
        self._params: dict[str, float] = {}
        self._dims: str | list[str] | None = None

    def half_normal(self, sigma: float = 1.0) -> Self:
        """Set HalfNormal distribution."""
        self._distribution = PriorType.HALF_NORMAL
        self._params = {"sigma": sigma}
        return self

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Set Normal distribution."""
        self._distribution = PriorType.NORMAL
        self._params = {"mu": mu, "sigma": sigma}
        return self

    def log_normal(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Set LogNormal distribution."""
        self._distribution = PriorType.LOG_NORMAL
        self._params = {"mu": mu, "sigma": sigma}
        return self

    def gamma(self, alpha: float = 2.0, beta: float = 1.0) -> Self:
        """Set Gamma distribution."""
        self._distribution = PriorType.GAMMA
        self._params = {"alpha": alpha, "beta": beta}
        return self

    def beta(self, alpha: float = 2.0, beta: float = 2.0) -> Self:
        """Set Beta distribution."""
        self._distribution = PriorType.BETA
        self._params = {"alpha": alpha, "beta": beta}
        return self

    def truncated_normal(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        lower: float = 0.0,
        upper: float | None = None,
    ) -> Self:
        """Set TruncatedNormal distribution."""
        self._distribution = PriorType.TRUNCATED_NORMAL
        self._params = {"mu": mu, "sigma": sigma, "lower": lower}
        if upper is not None:
            self._params["upper"] = upper
        return self

    def half_student_t(self, nu: float = 3.0, sigma: float = 1.0) -> Self:
        """Set HalfStudentT distribution."""
        self._distribution = PriorType.HALF_STUDENT_T
        self._params = {"nu": nu, "sigma": sigma}
        return self

    def with_dims(self, dims: str | list[str]) -> Self:
        """Set dimension(s) for the prior."""
        self._dims = dims
        return self

    def with_params(self, **params: float) -> Self:
        """Set additional parameters."""
        self._params.update(params)
        return self

    def build(self) -> PriorConfig:
        """Build the PriorConfig object."""
        if self._distribution is None:
            raise ValueError("Distribution not set. Call a distribution method first.")

        return PriorConfig(
            distribution=self._distribution,
            params=self._params,
            dims=self._dims,
        )


class AdstockConfigBuilder:
    """
    Builder for AdstockConfig objects.

    Examples
    --------
    >>> adstock = (AdstockConfigBuilder()
    ...     .geometric()
    ...     .with_max_lag(8)
    ...     .with_alpha_prior(PriorConfigBuilder().beta(1, 3).build())
    ...     .build())
    """

    def __init__(self) -> None:
        self._type: AdstockType = AdstockType.GEOMETRIC
        self._l_max: int = 8
        self._normalize: bool = True
        self._alpha_prior: PriorConfig | None = None
        self._theta_prior: PriorConfig | None = None

    def geometric(self) -> Self:
        """Use geometric adstock transformation."""
        self._type = AdstockType.GEOMETRIC
        return self

    def weibull(self) -> Self:
        """Use Weibull adstock transformation."""
        self._type = AdstockType.WEIBULL
        return self

    def delayed(self) -> Self:
        """Use delayed adstock transformation."""
        self._type = AdstockType.DELAYED
        return self

    def none(self) -> Self:
        """Disable adstock transformation."""
        self._type = AdstockType.NONE
        self._l_max = 1
        return self

    def with_max_lag(self, l_max: int) -> Self:
        """Set maximum lag weeks (1-52)."""
        if not 1 <= l_max <= 52:
            raise ValueError(f"l_max must be between 1 and 52, got {l_max}")
        self._l_max = l_max
        return self

    def with_normalize(self, normalize: bool = True) -> Self:
        """Set whether to normalize adstock weights."""
        self._normalize = normalize
        return self

    def with_alpha_prior(self, prior: PriorConfig) -> Self:
        """Set prior for decay rate (geometric adstock)."""
        self._alpha_prior = prior
        return self

    def with_theta_prior(self, prior: PriorConfig) -> Self:
        """Set prior for peak delay (Weibull adstock)."""
        self._theta_prior = prior
        return self

    # Convenience methods for common configurations
    def with_slow_decay(self) -> Self:
        """Configure for slow decay (long memory)."""
        self._alpha_prior = PriorConfigBuilder().beta(alpha=3.0, beta=1.0).build()
        return self

    def with_fast_decay(self) -> Self:
        """Configure for fast decay (short memory)."""
        self._alpha_prior = PriorConfigBuilder().beta(alpha=1.0, beta=3.0).build()
        return self

    def build(self) -> AdstockConfig:
        """Build the AdstockConfig object."""
        return AdstockConfig(
            type=self._type,
            l_max=self._l_max,
            normalize=self._normalize,
            alpha_prior=self._alpha_prior,
            theta_prior=self._theta_prior,
        )


class SaturationConfigBuilder:
    """
    Builder for SaturationConfig objects.

    Examples
    --------
    >>> saturation = (SaturationConfigBuilder()
    ...     .hill()
    ...     .with_kappa_prior(PriorConfigBuilder().beta(2, 2).build())
    ...     .with_slope_prior(PriorConfigBuilder().half_normal(1.5).build())
    ...     .with_kappa_bounds(0.1, 0.9)
    ...     .build())
    """

    def __init__(self) -> None:
        self._type: SaturationType = SaturationType.HILL
        self._kappa_prior: PriorConfig | None = None
        self._slope_prior: PriorConfig | None = None
        self._beta_prior: PriorConfig | None = None
        self._kappa_bounds: tuple[float, float] = (0.1, 0.9)

    def hill(self) -> Self:
        """Use Hill saturation function."""
        self._type = SaturationType.HILL
        return self

    def logistic(self) -> Self:
        """Use logistic saturation function."""
        self._type = SaturationType.LOGISTIC
        return self

    def michaelis_menten(self) -> Self:
        """Use Michaelis-Menten saturation function."""
        self._type = SaturationType.MICHAELIS_MENTEN
        return self

    def tanh(self) -> Self:
        """Use tanh saturation function."""
        self._type = SaturationType.TANH
        return self

    def none(self) -> Self:
        """Disable saturation transformation."""
        self._type = SaturationType.NONE
        return self

    def with_kappa_prior(self, prior: PriorConfig) -> Self:
        """Set prior for half-saturation point (EC50)."""
        self._kappa_prior = prior
        return self

    def with_slope_prior(self, prior: PriorConfig) -> Self:
        """Set prior for curve steepness."""
        self._slope_prior = prior
        return self

    def with_beta_prior(self, prior: PriorConfig) -> Self:
        """Set prior for maximum effect scaling."""
        self._beta_prior = prior
        return self

    def with_kappa_bounds(self, lower: float, upper: float) -> Self:
        """Set percentile bounds for kappa prior (data-driven)."""
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                f"Bounds must be 0 <= lower < upper <= 1, got ({lower}, {upper})"
            )
        self._kappa_bounds = (lower, upper)
        return self

    # Convenience methods for common configurations
    def with_strong_saturation(self) -> Self:
        """Configure for strong diminishing returns."""
        self._slope_prior = PriorConfigBuilder().half_normal(sigma=2.5).build()
        return self

    def with_weak_saturation(self) -> Self:
        """Configure for weak diminishing returns."""
        self._slope_prior = PriorConfigBuilder().half_normal(sigma=0.5).build()
        return self

    def build(self) -> SaturationConfig:
        """Build the SaturationConfig object."""
        return SaturationConfig(
            type=self._type,
            kappa_prior=self._kappa_prior,
            slope_prior=self._slope_prior,
            beta_prior=self._beta_prior,
            kappa_bounds_percentiles=self._kappa_bounds,
        )


__all__ = [
    "PriorConfigBuilder",
    "AdstockConfigBuilder",
    "SaturationConfigBuilder",
]

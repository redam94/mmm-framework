"""
Trend component strategies for BayesianMMM.

This module implements the Strategy pattern for building different
types of trend components in the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    from ..trend_config import TrendConfig
    from ...data_preparation import PreparedData


@runtime_checkable
class TrendStrategy(Protocol):
    """Protocol for trend building strategies."""

    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """
        Build the trend component.

        Parameters
        ----------
        model : pm.Model
            The PyMC model context.
        data : PreparedData
            Prepared data containing trend features.
        config : TrendConfig
            Trend configuration parameters.

        Returns
        -------
        pt.TensorVariable
            Trend component tensor, shape (n_obs,).
        """
        ...


class BaseTrendStrategy(ABC):
    """Base class for trend strategies with common utilities."""

    @abstractmethod
    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build the trend component."""
        pass


class LinearTrendStrategy(BaseTrendStrategy):
    """Simple linear trend: trend = k * t."""

    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build linear trend component."""
        t = data.trend_features.get("t")
        if t is None:
            t = np.linspace(0, 1, data.n_obs)

        with model:
            # Slope parameter
            k = pm.Normal("trend_k", mu=0, sigma=1)

            # Linear trend
            trend = k * t

        return trend


class PiecewiseTrendStrategy(BaseTrendStrategy):
    """
    Prophet-style piecewise linear trend with changepoints.

    The trend is modeled as:
        trend(t) = (k + A @ delta) * t + (m + A @ (-s * delta))

    Where:
    - k is the base growth rate
    - delta are the changepoint adjustments
    - A is the changepoint indicator matrix
    - s are the changepoint locations
    - m is the offset
    """

    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build piecewise linear trend component."""
        t = data.trend_features["t"]
        A = data.trend_features["A"]
        s = data.trend_features["s"]

        n_changepoints = len(s)

        with model:
            # Base growth rate
            k = pm.Normal("trend_k", mu=0, sigma=1)

            # Changepoint adjustments (sparse prior)
            delta = pm.Laplace(
                "trend_delta",
                mu=0,
                b=config.changepoint_prior_scale,
                shape=n_changepoints,
            )

            # Offset
            m = pm.Normal("trend_m", mu=0, sigma=1)

            # Piecewise linear trend
            # Growth rate at each point
            growth = k + pt.dot(A, delta)

            # Offset adjustment for continuity
            gamma = -s * delta
            offset = m + pt.dot(A, gamma)

            trend = growth * t + offset

        return trend


class SplineTrendStrategy(BaseTrendStrategy):
    """
    B-spline trend with random walk prior on coefficients.

    This provides smooth, flexible trends without explicit changepoints.
    """

    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build spline trend component."""
        basis = data.trend_features["spline_basis"]
        n_basis = basis.shape[1]

        with model:
            # Random walk prior on spline coefficients
            # This encourages smoothness
            coef_init = pm.Normal("trend_coef_init", mu=0, sigma=1)

            coef_diffs = pm.Normal(
                "trend_coef_diffs",
                mu=0,
                sigma=config.spline_prior_scale,
                shape=n_basis - 1,
            )

            # Cumulative sum to get coefficients
            coefs = pt.concatenate(
                [
                    pt.atleast_1d(coef_init),
                    coef_init + pt.cumsum(coef_diffs),
                ]
            )

            # Spline trend
            trend = pt.dot(basis, coefs)

        return trend


class GPTrendStrategy(BaseTrendStrategy):
    """
    Gaussian Process trend using HSGP (Hilbert Space GP) approximation.

    This provides non-parametric, smooth trends with uncertainty
    quantification, while remaining computationally efficient.
    """

    def build(
        self,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build GP trend component using HSGP approximation."""
        t_centered = data.trend_features.get("t_centered")
        if t_centered is None:
            t = data.trend_features.get("t", np.linspace(0, 1, data.n_obs))
            t_centered = t - 0.5

        try:
            return self._build_hsgp(model, t_centered, config)
        except Exception:
            # Fallback to spectral approximation
            return self._build_spectral(model, t_centered, config)

    def _build_hsgp(
        self,
        model: pm.Model,
        t_centered: np.ndarray,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build HSGP approximation."""
        from pymc.gp.util import HSGP

        n_obs = len(t_centered)

        with model:
            # Lengthscale prior
            ls_mu, ls_sigma = config.gp_lengthscale_prior
            ls = pm.InverseGamma(
                "trend_gp_ls",
                alpha=ls_mu,
                beta=ls_sigma,
            )

            # Variance prior
            var_mu, var_sigma = config.gp_variance_prior
            eta = pm.HalfNormal("trend_gp_eta", sigma=var_sigma)

            # HSGP approximation
            # Domain bounds
            L = config.gp_boundary_factor * 0.5  # t_centered in [-0.5, 0.5]

            gp = HSGP(
                m=config.gp_n_basis,
                L=L,
                cov_func=eta**2 * pm.gp.cov.ExpQuad(1, ls=ls),
            )

            trend = gp.prior("trend_gp", X=t_centered.reshape(-1, 1))

        return trend.flatten()

    def _build_spectral(
        self,
        model: pm.Model,
        t_centered: np.ndarray,
        config: TrendConfig,
    ) -> pt.TensorVariable:
        """Build spectral (random Fourier features) approximation."""
        n_basis = config.gp_n_basis
        n_obs = len(t_centered)

        with model:
            # Lengthscale
            ls_mu, ls_sigma = config.gp_lengthscale_prior
            ls = pm.InverseGamma("trend_gp_ls", alpha=ls_mu, beta=ls_sigma)

            # Amplitude
            var_mu, var_sigma = config.gp_variance_prior
            eta = pm.HalfNormal("trend_gp_eta", sigma=var_sigma)

            # Generate spectral frequencies
            # For squared exponential kernel, frequencies follow normal distribution
            omega = np.random.randn(n_basis) / (ls_mu / n_obs)

            # Basis functions
            sin_features = np.sin(2 * np.pi * t_centered[:, None] * omega[None, :])
            cos_features = np.cos(2 * np.pi * t_centered[:, None] * omega[None, :])
            basis = np.concatenate([sin_features, cos_features], axis=1)

            # Coefficients
            coefs = pm.Normal(
                "trend_gp_coefs",
                mu=0,
                sigma=1,
                shape=2 * n_basis,
            )

            # GP approximation
            scale = eta / np.sqrt(n_basis)
            trend = scale * pt.dot(basis, coefs)

        return trend


class TrendBuilder:
    """
    Factory for building trend components.

    Selects the appropriate strategy based on TrendType.
    """

    STRATEGIES = {
        "none": None,
        "linear": LinearTrendStrategy,
        "piecewise": PiecewiseTrendStrategy,
        "spline": SplineTrendStrategy,
        "gp": GPTrendStrategy,
    }

    @classmethod
    def build(
        cls,
        model: pm.Model,
        data: PreparedData,
        config: TrendConfig,
    ) -> pt.TensorVariable | None:
        """
        Build trend component using appropriate strategy.

        Parameters
        ----------
        model : pm.Model
            PyMC model context.
        data : PreparedData
            Prepared data with trend features.
        config : TrendConfig
            Trend configuration.

        Returns
        -------
        pt.TensorVariable | None
            Trend component, or None if trend_type is NONE.
        """
        trend_type = config.trend_type.value

        strategy_class = cls.STRATEGIES.get(trend_type)
        if strategy_class is None:
            return None

        strategy = strategy_class()
        return strategy.build(model, data, config)


__all__ = [
    "TrendStrategy",
    "BaseTrendStrategy",
    "LinearTrendStrategy",
    "PiecewiseTrendStrategy",
    "SplineTrendStrategy",
    "GPTrendStrategy",
    "TrendBuilder",
]

"""
Bayesian MMM model class - Robust Implementation v5 with Prediction Support.

Key design principles for stability:
1. Standardize all data (y, X_media, X_controls)
2. Use simple, well-understood priors
3. Avoid complex transformations in the graph
4. Pre-compute adstock outside the model
5. Use logistic saturation (numerically stable)
6. Flexible trend modeling with GP and spline options
7. Support for prediction and counterfactual analysis via Data
8. Save/load functionality for model persistence
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .config import (
    ModelConfig,
    ModelSpecification,
    SeasonalityConfig,
    HierarchicalConfig,
)
from .data_loader import PanelDataset
from .utils import compute_hdi_bounds
from .transforms import (
    geometric_adstock,
    geometric_adstock_2d,
    logistic_saturation,
    create_fourier_features,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)

# Backward compatibility aliases (original names from this module)
geometric_adstock_np = geometric_adstock
logistic_saturation_np = logistic_saturation

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================


class TrendType(str, Enum):
    """Available trend specifications."""

    NONE = "none"
    LINEAR = "linear"
    PIECEWISE = "piecewise"
    SPLINE = "spline"
    GP = "gaussian_process"


@dataclass
class TrendConfig:
    """Configuration for trend component.

    Parameters
    ----------
    type : TrendType
        Type of trend to use.

    # Piecewise trend parameters (Prophet-style)
    n_changepoints : int
        Number of potential changepoints for piecewise trend.
    changepoint_range : float
        Proportion of time range to place changepoints (0-1).
    changepoint_prior_scale : float
        Prior scale for changepoint magnitudes.

    # Spline trend parameters
    n_knots : int
        Number of knots for spline trend.
    spline_degree : int
        Degree of B-spline (default 3 = cubic).
    spline_prior_sigma : float
        Prior sigma for spline coefficients.

    # Gaussian Process trend parameters
    gp_lengthscale_prior_mu : float
        Prior mean for GP lengthscale (in proportion of time range).
    gp_lengthscale_prior_sigma : float
        Prior sigma for GP lengthscale.
    gp_amplitude_prior_sigma : float
        Prior sigma for GP amplitude (HalfNormal).
    gp_n_basis : int
        Number of basis functions for HSGP approximation.
    gp_c : float
        Boundary factor for HSGP (typically 1.5-2.0).

    # Linear trend parameters
    growth_prior_mu : float
        Prior mean for linear growth rate.
    growth_prior_sigma : float
        Prior sigma for linear growth rate.
    """

    type: TrendType = TrendType.LINEAR

    # Piecewise trend parameters
    n_changepoints: int = 10
    changepoint_range: float = 0.8
    changepoint_prior_scale: float = 0.05

    # Spline trend parameters
    n_knots: int = 10
    spline_degree: int = 3
    spline_prior_sigma: float = 1.0

    # Gaussian Process trend parameters
    gp_lengthscale_prior_mu: float = 0.3
    gp_lengthscale_prior_sigma: float = 0.2
    gp_amplitude_prior_sigma: float = 0.5
    gp_n_basis: int = 20
    gp_c: float = 1.5

    # Linear trend parameters
    growth_prior_mu: float = 0.0
    growth_prior_sigma: float = 0.1

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_knots": self.n_knots,
            "spline_degree": self.spline_degree,
            "spline_prior_sigma": self.spline_prior_sigma,
            "gp_lengthscale_prior_mu": self.gp_lengthscale_prior_mu,
            "gp_lengthscale_prior_sigma": self.gp_lengthscale_prior_sigma,
            "gp_amplitude_prior_sigma": self.gp_amplitude_prior_sigma,
            "gp_n_basis": self.gp_n_basis,
            "gp_c": self.gp_c,
            "growth_prior_mu": self.growth_prior_mu,
            "growth_prior_sigma": self.growth_prior_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrendConfig:
        """Create from dictionary."""
        data = data.copy()
        data["type"] = TrendType(data["type"])
        return cls(**data)


# =============================================================================
# Helper functions
# =============================================================================
# NOTE: Transform functions (adstock, saturation, seasonality, trend) are now
# imported from mmm_framework.transforms module. The imports at the top of this
# file and the backward compatibility aliases ensure existing code continues
# to work. See src/mmm_framework/transforms/ for the implementations.


# =============================================================================
# Results container
# =============================================================================


@dataclass
class MMMResults:
    """Container for fitted model results."""

    trace: az.InferenceData
    model: pm.Model
    panel: PanelDataset
    channel_contributions: pd.DataFrame | None = None
    diagnostics: dict = field(default_factory=dict)
    y_mean: float = 0.0
    y_std: float = 1.0

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""
        return az.summary(self.trace, var_names=var_names)

    def plot_trace(self, var_names: list[str] | None = None, **kwargs):
        """Plot trace diagnostics."""
        return az.plot_trace(self.trace, var_names=var_names, **kwargs)

    def plot_posterior(self, var_names: list[str] | None = None, **kwargs):
        """Plot posterior distributions."""
        return az.plot_posterior(self.trace, var_names=var_names, **kwargs)


@dataclass
class PredictionResults:
    """Container for prediction results."""

    posterior_predictive: az.InferenceData
    y_pred_mean: np.ndarray
    y_pred_std: np.ndarray
    y_pred_hdi_low: np.ndarray
    y_pred_hdi_high: np.ndarray
    y_pred_samples: np.ndarray  # Shape: (n_samples, n_obs)

    @property
    def n_samples(self) -> int:
        return self.y_pred_samples.shape[0]

    @property
    def n_obs(self) -> int:
        return self.y_pred_samples.shape[1]


@dataclass
class ContributionResults:
    """Container for counterfactual contribution results."""

    # Per-channel contributions (original scale)
    channel_contributions: pd.DataFrame  # (n_obs, n_channels)

    # Total contributions over time or for specified period
    total_contributions: pd.Series  # Indexed by channel

    # Percentage of total effect
    contribution_pct: pd.Series

    # Baseline prediction (all channels present)
    baseline_prediction: np.ndarray

    # Counterfactual predictions (each channel zeroed)
    counterfactual_predictions: dict[str, np.ndarray]

    # Time period used for calculation (None = all)
    time_period: tuple[int, int] | None = None

    # Uncertainty (if computed with multiple samples)
    contribution_hdi_low: pd.Series | None = None
    contribution_hdi_high: pd.Series | None = None

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        data = {
            "Channel": self.total_contributions.index,
            "Total Contribution": self.total_contributions.values,
            "Contribution %": self.contribution_pct.values,
        }
        if self.contribution_hdi_low is not None:
            data["HDI 3%"] = self.contribution_hdi_low.values
            data["HDI 97%"] = self.contribution_hdi_high.values
        return pd.DataFrame(data)


@dataclass
class ComponentDecomposition:
    """Container for full component decomposition results."""

    # Component contributions (original scale, per observation)
    intercept: np.ndarray
    trend: np.ndarray
    seasonality: np.ndarray
    media_total: np.ndarray
    media_by_channel: pd.DataFrame
    controls_total: np.ndarray
    controls_by_var: pd.DataFrame | None
    geo_effects: np.ndarray | None
    product_effects: np.ndarray | None

    # Aggregated totals
    total_intercept: float
    total_trend: float
    total_seasonality: float
    total_media: float
    total_controls: float
    total_geo: float | None
    total_product: float | None

    # Scaling parameters for reference
    y_mean: float
    y_std: float

    def summary(self) -> pd.DataFrame:
        """Get summary of component contributions."""
        components = {
            "Base (Intercept)": self.total_intercept,
            "Trend": self.total_trend,
            "Seasonality": self.total_seasonality,
            "Media (Total)": self.total_media,
            "Controls (Total)": self.total_controls,
        }

        if self.total_geo is not None:
            components["Geo Effects"] = self.total_geo
        if self.total_product is not None:
            components["Product Effects"] = self.total_product

        total = sum(components.values())

        df = pd.DataFrame(
            {
                "Component": list(components.keys()),
                "Total Contribution": list(components.values()),
                "Contribution %": [
                    v / total * 100 if total != 0 else 0 for v in components.values()
                ],
            }
        )

        return df

    def media_summary(self) -> pd.DataFrame:
        """Get detailed media channel breakdown."""
        totals = self.media_by_channel.sum()
        total_media = totals.sum()

        return pd.DataFrame(
            {
                "Channel": totals.index,
                "Total Contribution": totals.values,
                "Share of Media %": (
                    (totals / total_media * 100).values
                    if total_media != 0
                    else [0] * len(totals)
                ),
            }
        )

    def controls_summary(self) -> pd.DataFrame | None:
        """Get detailed control variable breakdown."""
        if self.controls_by_var is None:
            return None

        totals = self.controls_by_var.sum()
        total_controls = totals.sum()

        return pd.DataFrame(
            {
                "Variable": totals.index,
                "Total Contribution": totals.values,
                "Share of Controls %": (
                    (totals / total_controls * 100).values
                    if total_controls != 0
                    else [0] * len(totals)
                ),
            }
        )


# =============================================================================
# Main Model Class
# =============================================================================


class BayesianMMM:
    """
    Bayesian Marketing Mix Model - Robust Implementation with Prediction Support.

    This implementation prioritizes numerical stability:
    - All data is standardized before modeling
    - Adstock is pre-computed at fixed alpha values
    - Logistic saturation is used (more stable than Hill)
    - Priors are carefully scaled for standardized data
    - Flexible trend modeling with GP, spline, and piecewise options
    - Support for prediction and counterfactual analysis
    - Save/load functionality for model persistence

    Parameters
    ----------
    panel : PanelDataset
        Panel data from MFFLoader.
    model_config : ModelConfig
        Model configuration.
    trend_config : TrendConfig, optional
        Trend specification.
    adstock_alphas : list[float], optional
        Fixed adstock decay values to pre-compute.
    """

    # Version for save/load compatibility
    _VERSION = "1.0.0"

    def __init__(
        self,
        panel: PanelDataset,
        model_config: ModelConfig,
        trend_config: TrendConfig | None = None,
        adstock_alphas: list[float] | None = None,
    ):
        self.panel = panel
        self.model_config = model_config
        self.trend_config = trend_config or TrendConfig()
        self.adstock_alphas = adstock_alphas or [0.0, 0.3, 0.5, 0.7, 0.9]

        self.mff_config = panel.config
        self.hierarchical_config = model_config.hierarchical
        self.seasonality_config = model_config.seasonality

        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None

        # Store scaling parameters for prediction
        self._scaling_params: dict[str, Any] = {}

        self._prepare_data()

    def _prepare_data(self):
        """Prepare and standardize all data."""
        # === Raw data ===
        self.y_raw = self.panel.y.values.astype(np.float64)
        self.X_media_raw = self.panel.X_media.values.astype(np.float64)

        if self.panel.X_controls is not None and self.panel.X_controls.shape[1] > 0:
            self.X_controls_raw = self.panel.X_controls.values.astype(np.float64)
        else:
            self.X_controls_raw = None

        # === Dimensions ===
        self.n_obs = len(self.y_raw)
        self.n_channels = self.X_media_raw.shape[1]
        self.n_controls = (
            self.X_controls_raw.shape[1] if self.X_controls_raw is not None else 0
        )

        self.channel_names = list(self.panel.coords.channels)
        self.control_names = (
            list(self.panel.coords.controls) if self.n_controls > 0 else []
        )

        # === Standardize target ===
        self.y_mean = float(self.y_raw.mean())
        self.y_std = float(self.y_raw.std()) + 1e-8
        self.y = (self.y_raw - self.y_mean) / self.y_std

        # Store scaling parameters
        self._scaling_params["y_mean"] = self.y_mean
        self._scaling_params["y_std"] = self.y_std

        # === Pre-compute adstocked media at fixed alphas ===
        # Store max values for each channel for normalization
        self._media_max = {}
        self.X_media_adstocked = {}

        for alpha in self.adstock_alphas:
            adstocked = geometric_adstock_2d(self.X_media_raw, alpha)
            # Store max values (use maximum across all alphas for consistent scaling)
            for c in range(self.n_channels):
                key = self.channel_names[c]
                current_max = adstocked[:, c].max()
                if key not in self._media_max:
                    self._media_max[key] = current_max
                else:
                    self._media_max[key] = max(self._media_max[key], current_max)

        # Normalize using consistent max values
        for alpha in self.adstock_alphas:
            adstocked = geometric_adstock_2d(self.X_media_raw, alpha)
            normalized = np.zeros_like(adstocked)
            for c, ch_name in enumerate(self.channel_names):
                normalized[:, c] = adstocked[:, c] / (self._media_max[ch_name] + 1e-8)
            self.X_media_adstocked[alpha] = normalized

        self._scaling_params["media_max"] = self._media_max.copy()

        # === Standardize controls ===
        if self.X_controls_raw is not None:
            self.control_mean = self.X_controls_raw.mean(axis=0)
            self.control_std = self.X_controls_raw.std(axis=0) + 1e-8
            self.X_controls = (
                self.X_controls_raw - self.control_mean
            ) / self.control_std
            self._scaling_params["control_mean"] = self.control_mean.copy()
            self._scaling_params["control_std"] = self.control_std.copy()
        else:
            self.X_controls = None

        # === Geo/product info ===
        self.has_geo = self.panel.coords.has_geo
        self.has_product = self.panel.coords.has_product
        self.n_geos = self.panel.coords.n_geos
        self.n_products = self.panel.coords.n_products

        if self.has_geo:
            self.geo_names = list(self.panel.coords.geographies)
            self.geo_idx = self._get_group_indices("geography")
        else:
            self.geo_idx = np.zeros(self.n_obs, dtype=np.int32)

        if self.has_product:
            self.product_names = list(self.panel.coords.products)
            self.product_idx = self._get_group_indices("product")
        else:
            self.product_idx = np.zeros(self.n_obs, dtype=np.int32)

        # === Time index ===
        self.n_periods = self.panel.coords.n_periods
        self.time_idx = self._get_time_index()
        self.t_scaled = np.linspace(0, 1, self.n_periods)  # Unique time points [0, 1]

        # === Seasonality features ===
        self._prepare_seasonality()

        # === Trend features ===
        self._prepare_trend()

        # === Media hierarchy ===
        self.media_groups = self.mff_config.get_hierarchical_media_groups()
        self.has_media_hierarchy = len(self.media_groups) > 0

    def _get_group_indices(self, level_name: str) -> np.ndarray:
        """Get group indices for a hierarchical level."""
        cols = self.mff_config.columns
        col_name = getattr(cols, level_name)

        if isinstance(self.panel.index, pd.MultiIndex):
            values = self.panel.index.get_level_values(col_name)
            if level_name == "geography":
                categories = self.geo_names
            else:
                categories = self.product_names
            return pd.Categorical(values, categories=categories).codes.astype(np.int32)
        return np.zeros(self.n_obs, dtype=np.int32)

    def _get_time_index(self) -> np.ndarray:
        """Get time index for each observation."""
        cols = self.mff_config.columns

        if isinstance(self.panel.index, pd.MultiIndex):
            period_values = self.panel.index.get_level_values(cols.period)
            periods_unique = list(self.panel.coords.periods)
            return pd.Categorical(
                period_values, categories=periods_unique
            ).codes.astype(np.int32)
        return np.arange(self.n_obs, dtype=np.int32)

    def _prepare_seasonality(self):
        """Prepare Fourier features for seasonality."""
        self.seasonality_features = {}
        t = np.arange(self.n_periods)

        if self.seasonality_config.yearly and self.seasonality_config.yearly > 0:
            period = 52  # Weekly data
            order = self.seasonality_config.yearly
            features = create_fourier_features(t, period, order)
            if features.shape[1] > 0:
                self.seasonality_features["yearly"] = features

    def _prepare_trend(self):
        """Prepare trend features based on configuration."""
        t_unique = np.linspace(0, 1, self.n_periods)

        self.trend_features = {}

        if self.trend_config.type == TrendType.SPLINE:
            # Create B-spline basis
            self.trend_features["spline_basis"] = create_bspline_basis(
                t_unique,
                n_knots=self.trend_config.n_knots,
                degree=self.trend_config.spline_degree,
            )
            self.trend_features["n_spline_coef"] = self.trend_features[
                "spline_basis"
            ].shape[1]

        elif self.trend_config.type == TrendType.PIECEWISE:
            # Create piecewise linear design matrix
            s, A = create_piecewise_trend_matrix(
                t_unique,
                n_changepoints=self.trend_config.n_changepoints,
                changepoint_range=self.trend_config.changepoint_range,
            )
            self.trend_features["changepoints"] = s
            self.trend_features["changepoint_matrix"] = A

        elif self.trend_config.type == TrendType.GP:
            # Store GP config for model building
            self.trend_features["gp_config"] = {
                "lengthscale_mu": self.trend_config.gp_lengthscale_prior_mu,
                "lengthscale_sigma": self.trend_config.gp_lengthscale_prior_sigma,
                "amplitude_sigma": self.trend_config.gp_amplitude_prior_sigma,
                "n_basis": self.trend_config.gp_n_basis,
                "c": self.trend_config.gp_c,
            }

    def _get_time_mask(
        self, time_period: tuple[int, int] | None
    ) -> NDArray[np.bool_]:
        """Create boolean mask for time period filtering.

        Parameters
        ----------
        time_period : tuple[int, int] | None
            (start_idx, end_idx) inclusive range, or None for all observations.
            Both start and end indices are inclusive.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask array of shape (n_obs,). True for observations
            within the specified time period.

        Examples
        --------
        >>> # Mask for time indices 10 through 20
        >>> mask = mmm._get_time_mask((10, 20))
        >>> filtered_y = mmm.y_raw[mask]

        >>> # No filtering (all observations)
        >>> mask = mmm._get_time_mask(None)
        >>> assert mask.all()
        """
        if time_period is not None:
            start_idx, end_idx = time_period
            return (self.time_idx >= start_idx) & (self.time_idx <= end_idx)
        return np.ones(self.n_obs, dtype=bool)

    def _build_coords(self) -> dict:
        """Build PyMC coordinate dictionary."""
        coords = {
            "obs": np.arange(self.n_obs),
            "channel": self.channel_names,
        }

        if self.has_geo:
            coords["geo"] = self.geo_names
        if self.has_product:
            coords["product"] = self.product_names
        if self.n_controls > 0:
            coords["control"] = self.control_names

        for name, features in self.seasonality_features.items():
            n_features = features.shape[1]
            coords[f"{name}_fourier"] = [f"{name}_{i}" for i in range(n_features)]

        for parent, children in self.media_groups.items():
            coords[f"{parent}_platform"] = list(children)

        # Add trend-specific coordinates
        if self.trend_config.type == TrendType.SPLINE:
            n_coef = self.trend_features.get("n_spline_coef", 0)
            coords["spline_idx"] = list(range(n_coef))

        elif self.trend_config.type == TrendType.PIECEWISE:
            n_cp = len(self.trend_features.get("changepoints", []))
            coords["changepoint"] = list(range(n_cp))

        return coords

    def _build_trend_component(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build the trend component based on configuration."""

        # FIX: Convert t_scaled to PyTensor tensor for indexing with PyTensor variables
        t_scaled_tensor = pt.as_tensor_variable(self.t_scaled)

        if self.trend_config.type == TrendType.NONE:
            return pt.zeros(time_idx.shape[0])

        elif self.trend_config.type == TrendType.LINEAR:
            trend_slope = pm.Normal(
                "trend_slope",
                mu=self.trend_config.growth_prior_mu,
                sigma=self.trend_config.growth_prior_sigma,
            )
            return trend_slope * t_scaled_tensor[time_idx]

        elif self.trend_config.type == TrendType.PIECEWISE:
            return self._build_piecewise_trend(model, time_idx)

        elif self.trend_config.type == TrendType.SPLINE:
            return self._build_spline_trend(model, time_idx)

        elif self.trend_config.type == TrendType.GP:
            return self._build_gp_trend(model, time_idx)

        else:
            warnings.warn(f"Unknown trend type: {self.trend_config.type}, using linear")
            trend_slope = pm.Normal("trend_slope", mu=0, sigma=0.5)
            return trend_slope * t_scaled_tensor[time_idx]

    def _build_piecewise_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build Prophet-style piecewise linear trend."""

        s = self.trend_features["changepoints"]
        A = self.trend_features["changepoint_matrix"]
        n_changepoints = len(s)

        # Base growth rate
        k = pm.Normal(
            "trend_k",
            mu=self.trend_config.growth_prior_mu,
            sigma=self.trend_config.growth_prior_sigma,
        )

        # Changepoint adjustments with Laplace prior (promotes sparsity)
        delta = pm.Laplace(
            "trend_delta",
            mu=0,
            b=self.trend_config.changepoint_prior_scale,
            shape=n_changepoints,
            dims="changepoint",
        )

        # Intercept adjustment to keep trend continuous
        m = pm.Normal("trend_m", mu=0, sigma=0.5)

        # Build trend at unique time points
        t_unique = np.linspace(0, 1, self.n_periods)

        # FIX: Convert numpy arrays to PyTensor tensors
        t_unique_tensor = pt.as_tensor_variable(t_unique)
        A_tensor = pt.as_tensor_variable(A)
        s_tensor = pt.as_tensor_variable(s)

        # gamma ensures continuity at changepoints
        gamma = -s_tensor * delta

        # Trend at unique times
        trend_unique = (
            k * t_unique_tensor + pt.dot(A_tensor, delta) + m + pt.dot(A_tensor, gamma)
        )

        # Map to observations
        return trend_unique[time_idx]

    def _build_spline_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build B-spline trend."""

        basis = self.trend_features["spline_basis"]
        n_coef = basis.shape[1]

        # Spline coefficients with smoothness prior
        # Using random walk prior for smooth trends
        spline_coef_raw = pm.Normal(
            "spline_coef_raw", mu=0, sigma=1, shape=n_coef, dims="spline_idx"
        )

        # Scale parameter for smoothness
        spline_scale = pm.HalfNormal(
            "spline_scale", sigma=self.trend_config.spline_prior_sigma
        )

        # Apply cumulative sum for random walk behavior (smoother trends)
        spline_coef = pm.Deterministic(
            "spline_coef", spline_scale * pt.cumsum(spline_coef_raw), dims="spline_idx"
        )

        # FIX: Convert basis to PyTensor tensor
        basis_tensor = pt.as_tensor_variable(basis)

        # Compute trend at unique time points
        trend_unique = pt.dot(basis_tensor, spline_coef)

        # Center the trend (remove mean)
        trend_unique = trend_unique - trend_unique.mean()

        # Map to observations
        return trend_unique[time_idx]

    def _build_gp_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build Gaussian Process trend using HSGP approximation."""

        gp_config = self.trend_features["gp_config"]

        # GP hyperparameters
        gp_lengthscale = pm.LogNormal(
            "gp_lengthscale",
            mu=np.log(gp_config["lengthscale_mu"]),
            sigma=gp_config["lengthscale_sigma"],
        )

        gp_amplitude = pm.HalfNormal("gp_amplitude", sigma=gp_config["amplitude_sigma"])

        # Use Hilbert Space GP approximation for efficiency
        # This is much faster than full GP for time series
        try:
            import pymc.gp as gp_module

            # Matern 3/2 covariance for smooth but flexible trends
            cov_func = gp_amplitude**2 * gp_module.cov.Matern32(
                input_dim=1, ls=gp_lengthscale
            )

            # HSGP approximation
            gp = gp_module.HSGP(
                m=[gp_config["n_basis"]], c=gp_config["c"], cov_func=cov_func
            )

            # Time points for unique periods, scaled to [-1, 1] for HSGP
            t_unique = np.linspace(-1, 1, self.n_periods).reshape(-1, 1)

            # Build GP prior
            trend_unique = gp.prior("trend_gp", X=t_unique)

            # Map to observations
            return trend_unique[time_idx]

        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"HSGP not available ({e}), falling back to basis function GP"
            )
            return self._build_gp_trend_basis(
                model, gp_lengthscale, gp_amplitude, time_idx
            )

    def _build_gp_trend_basis(
        self,
        model: pm.Model,
        lengthscale: pt.TensorVariable,
        amplitude: pt.TensorVariable,
        time_idx,
    ) -> pt.TensorVariable:
        """
        Build GP trend using explicit basis function approximation.

        This is a fallback when HSGP is not available.
        Uses a spectral (Fourier) approximation to the GP.
        """
        gp_config = self.trend_features["gp_config"]
        n_basis = gp_config["n_basis"]

        # Time points
        t_unique = np.linspace(0, 1, self.n_periods)

        # Create Fourier basis for GP approximation
        # This approximates a stationary GP with spectral methods
        frequencies = np.arange(1, n_basis + 1)

        # Basis functions
        basis_sin = np.sin(2 * np.pi * np.outer(t_unique, frequencies))
        basis_cos = np.cos(2 * np.pi * np.outer(t_unique, frequencies))
        basis = np.hstack([basis_sin, basis_cos])  # (n_periods, 2*n_basis)

        # FIX: Convert to PyTensor tensor
        basis_tensor = pt.as_tensor_variable(basis)

        # Spectral density weights (approximate Matern 3/2)
        # S(w) ∝ (1 + (w*l)^2)^(-2) for Matern 3/2
        omega = 2 * np.pi * frequencies
        omega_tensor = pt.as_tensor_variable(omega)

        # GP basis coefficients
        gp_coef = pm.Normal("gp_coef", mu=0, sigma=1, shape=2 * n_basis)

        # Compute spectral weights (need to be differentiable)
        # Use a simpler squared exponential approximation for stability
        spectral_weights_sin = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights_cos = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights = pt.concatenate([spectral_weights_sin, spectral_weights_cos])

        # Scale coefficients by spectral weights and amplitude
        scaled_coef = amplitude * gp_coef * pt.sqrt(spectral_weights / n_basis)

        # Compute trend
        trend_unique = pt.dot(basis_tensor, scaled_coef)

        # Center
        trend_unique = trend_unique - trend_unique.mean()

        # Map to observations
        return trend_unique[time_idx]

    def _prepare_media_data_for_model(
        self, X_media_raw: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare media data for model (compute adstock at low/high alpha).

        Parameters
        ----------
        X_media_raw : np.ndarray, optional
            Raw media data. If None, uses training data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (X_adstock_low, X_adstock_high) - normalized adstocked media
        """
        if X_media_raw is None:
            X_media_raw = self.X_media_raw

        alpha_low = self.adstock_alphas[0]
        alpha_high = self.adstock_alphas[-1]

        # Compute adstock
        adstock_low = geometric_adstock_2d(X_media_raw, alpha_low)
        adstock_high = geometric_adstock_2d(X_media_raw, alpha_high)

        # Normalize using training data max values
        for c, ch_name in enumerate(self.channel_names):
            max_val = self._media_max[ch_name] + 1e-8
            adstock_low[:, c] = adstock_low[:, c] / max_val
            adstock_high[:, c] = adstock_high[:, c] / max_val

        return adstock_low, adstock_high

    def _build_model(self) -> pm.Model:
        """Build the PyMC model with Data for prediction support."""
        coords = self._build_coords()

        # Prepare media data
        X_adstock_low, X_adstock_high = self._prepare_media_data_for_model()

        with pm.Model(coords=coords) as model:
            # =================================================================
            # MUTABLE DATA (for prediction)
            # =================================================================
            X_media_low_data = pm.Data(
                "X_media_low", X_adstock_low, dims=("obs", "channel")
            )
            X_media_high_data = pm.Data(
                "X_media_high", X_adstock_high, dims=("obs", "channel")
            )

            if self.X_controls is not None:
                X_controls_data = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )

            time_idx_data = pm.Data("time_idx", self.time_idx)
            geo_idx_data = pm.Data("geo_idx", self.geo_idx)
            product_idx_data = pm.Data("product_idx", self.product_idx)

            n_obs_data = X_media_low_data.shape[0]

            # =================================================================
            # INTERCEPT
            # =================================================================
            # Since y is standardized (mean=0, std=1), intercept should be ~0
            intercept = pm.Normal("intercept", mu=0, sigma=0.5)

            # =================================================================
            # TREND
            # =================================================================
            trend = self._build_trend_component(model, time_idx_data)

            # Store trend for diagnostics
            pm.Deterministic("trend_component", trend)

            # =================================================================
            # SEASONALITY
            # =================================================================
            n_periods = self.n_periods
            seasonality_at_periods = pt.zeros(n_periods)

            for name, features in self.seasonality_features.items():
                n_features = features.shape[1]
                season_coef = pm.Normal(
                    f"season_{name}",
                    mu=0,
                    sigma=0.3,
                    shape=n_features,
                    dims=f"{name}_fourier",
                )
                # Explicitly convert features to tensor for proper computation
                features_tensor = pt.as_tensor_variable(features)
                # Compute seasonal effect at each unique time period
                season_effect = pt.dot(features_tensor, season_coef)
                seasonality_at_periods = seasonality_at_periods + season_effect

            # Map from unique periods to observations using time index
            # Use subtensor for explicit advanced indexing
            seasonality = seasonality_at_periods[time_idx_data]

            # Store seasonality component
            pm.Deterministic("seasonality_component", seasonality)

            # Also store seasonality at unique periods for diagnostics
            pm.Deterministic("seasonality_by_period", seasonality_at_periods)
            # ###
            # seasonality = pt.zeros(n_obs_data)
            # for name, features in self.seasonality_features.items():
            #     n_features = features.shape[1]
            #     season_coef = pm.Normal(
            #         f"season_{name}",
            #         mu=0,
            #         sigma=0.3,
            #         shape=n_features,
            #     )
            #     # FIX: Convert features to PyTensor tensor
            #     features_tensor = pt.as_tensor_variable(features)
            #     season_effect = pt.dot(features_tensor, season_coef)
            #     seasonality = seasonality + season_effect[time_idx_data]

            # =================================================================
            # GEO EFFECTS (if applicable)
            # =================================================================
            if self.has_geo and self.hierarchical_config.pool_across_geo:
                geo_sigma = pm.HalfNormal("geo_sigma", sigma=0.3)
                geo_offset = pm.Normal("geo_offset", mu=0, sigma=1, shape=self.n_geos)
                geo_effect = geo_sigma * geo_offset
                geo_contribution = geo_effect[geo_idx_data]
            else:
                geo_contribution = pt.zeros(n_obs_data)

            # =================================================================
            # PRODUCT EFFECTS (if applicable)
            # =================================================================
            if self.has_product and self.hierarchical_config.pool_across_product:
                product_sigma = pm.HalfNormal("product_sigma", sigma=0.3)
                product_offset = pm.Normal(
                    "product_offset", mu=0, sigma=1, shape=self.n_products
                )
                product_effect = product_sigma * product_offset
                product_contribution = product_effect[product_idx_data]
            else:
                product_contribution = pt.zeros(n_obs_data)

            # =================================================================
            # MEDIA EFFECTS
            # =================================================================
            # Strategy: interpolate between pre-computed adstock levels,
            # apply logistic saturation, multiply by coefficient

            channel_contribs = []

            for c, channel_name in enumerate(self.channel_names):
                # Get pre-computed adstock at low and high alpha
                x_low = X_media_low_data[:, c]
                x_high = X_media_high_data[:, c]

                # Adstock mixing parameter
                adstock_mix = pm.Beta(f"adstock_{channel_name}", alpha=2, beta=2)

                # Interpolate
                x_adstocked = (1 - adstock_mix) * x_low + adstock_mix * x_high

                # Saturation parameter (logistic)
                # Use Exponential for numerical stability near 0
                sat_lam = pm.Exponential(f"sat_lam_{channel_name}", lam=0.5)

                # Apply saturation with numerical stability
                # Clip the exponent to prevent overflow
                exponent = pt.clip(-sat_lam * x_adstocked, -20, 0)
                x_saturated = 1 - pt.exp(exponent)

                # Channel coefficient (positive, scaled for standardized y)
                beta = pm.HalfNormal(f"beta_{channel_name}", sigma=0.5)

                channel_contrib = beta * x_saturated
                channel_contribs.append(channel_contrib)

            # Stack and sum
            media_matrix = pt.stack(channel_contribs, axis=1)
            media_contribution = media_matrix.sum(axis=1)

            # Store for diagnostics
            pm.Deterministic(
                "channel_contributions", media_matrix, dims=("obs", "channel")
            )
            pm.Deterministic("media_total", media_contribution)

            # =================================================================
            # CONTROL EFFECTS
            # =================================================================
            if self.n_controls > 0:
                # Normal priors for standardized controls
                beta_controls = pm.Normal(
                    "beta_controls",
                    mu=0,
                    sigma=0.5,
                    shape=self.n_controls,
                )
                control_contribution = pt.dot(X_controls_data, beta_controls)
            else:
                control_contribution = pt.zeros(n_obs_data)

            # =================================================================
            # COMBINE AND LIKELIHOOD
            # =================================================================
            mu = (
                intercept
                + trend
                + seasonality
                + geo_contribution
                + product_contribution
                + media_contribution
                + control_contribution
            )

            # Noise (should capture remaining variance after standardization)
            sigma = pm.HalfNormal("sigma", sigma=0.5)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic("y_obs_scaled", y_obs*self.y_std + self.y_mean, dims="obs")

        return model

    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model
    
    def get_prior(self,
                  samples: int = 500,
                  random_seed: int | None = None
                  ) -> az.InferenceData:
        """
        Sample from the prior distribution of the model.

        Returns
        -------
        az.InferenceData
            InferenceData object containing prior samples.
        """
        with self.model:
            prior_trace = pm.sample_prior_predictive(samples=samples, random_seed=random_seed)
        return prior_trace
    
    def fit(
        self,
        draws: int | None = None,
        tune: int | None = None,
        chains: int | None = None,
        target_accept: float | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> MMMResults:
        """
        Fit the model using MCMC.

        Parameters
        ----------
        draws : int, optional
            Number of posterior draws per chain. Default from config.
        tune : int, optional
            Number of tuning samples. Default from config.
        chains : int, optional
            Number of MCMC chains. Default from config.
        target_accept : float, optional
            Target acceptance rate for NUTS. Default 0.9.
        random_seed : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional arguments passed to pm.sample().

        Returns
        -------
        MMMResults
            Fitted model results with diagnostics.
        """
        draws = draws or self.model_config.n_draws
        tune = tune or self.model_config.n_tune
        chains = chains or self.model_config.n_chains
        target_accept = target_accept or 0.9
        random_seed = random_seed or self.model_config.optim_seed

        # Sampler
        nuts_sampler = "numpyro" if self.model_config.use_numpyro else "pymc"

        prior = self.get_prior(samples=1000, random_seed=random_seed)

        with self.model:

            trace: az.InferenceData = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                init="adapt_diag",
                **kwargs,
            )
        trace.extend(prior)

        self._trace = trace

        # Diagnostics
        try:
            div_count = int(trace.sample_stats.diverging.sum().values)
        except:
            div_count = 0

        diagnostics = {
            "divergences": div_count,
            "rhat_max": float(az.rhat(trace).max().to_array().max()),
            "ess_bulk_min": float(az.ess(trace, method="bulk").min().to_array().min()),
        }

        results = MMMResults(
            trace=trace,
            model=self.model,
            panel=self.panel,
            diagnostics=diagnostics,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

        # Compute contributions
        results = self._compute_contributions(results)

        return results

    def _compute_contributions(self, results: MMMResults) -> MMMResults:
        """Compute channel-level contributions in original scale."""
        try:
            contrib_posterior = results.trace.posterior["channel_contributions"]
            contrib_mean = contrib_posterior.mean(dim=["chain", "draw"]).values

            # Scale back to original units
            contrib_original = contrib_mean * self.y_std

            results.channel_contributions = pd.DataFrame(
                contrib_original,
                index=self.panel.index,
                columns=self.channel_names,
            )
        except Exception as e:
            warnings.warn(f"Could not compute contributions: {e}")

        return results

    def predict(
        self,
        X_media: np.ndarray | None = None,
        X_controls: np.ndarray | None = None,
        time_idx: np.ndarray | None = None,
        geo_idx: np.ndarray | None = None,
        product_idx: np.ndarray | None = None,
        return_original_scale: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> PredictionResults:
        """
        Generate predictions with optionally modified input data.

        This method allows running posterior predictive sampling with modified
        inputs, enabling counterfactual analysis and contribution calculation.

        Parameters
        ----------
        X_media : np.ndarray, optional
            Media data of shape (n_obs, n_channels). If None, uses training data.
            Should be in **original scale** (not adstocked or normalized).
        X_controls : np.ndarray, optional
            Control data of shape (n_obs, n_controls). If None, uses training data.
            Should be in **original scale** (not standardized).
        time_idx : np.ndarray, optional
            Time indices. If None, uses training data indices.
        geo_idx : np.ndarray, optional
            Geography indices. If None, uses training data indices.
        product_idx : np.ndarray, optional
            Product indices. If None, uses training data indices.
        return_original_scale : bool
            If True (default), predictions are returned in original scale.
            If False, predictions are in standardized scale.
        hdi_prob : float
            Probability mass for HDI calculation (default 0.94).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        PredictionResults
            Container with predictions, including mean, std, HDI, and samples.

        Examples
        --------
        >>> # Get baseline predictions (same as training)
        >>> pred = mmm.predict()

        >>> # Prediction with TV spend zeroed out
        >>> X_media_no_tv = panel.X_media.values.copy()
        >>> X_media_no_tv[:, 0] = 0  # Zero out first channel (TV)
        >>> pred_no_tv = mmm.predict(X_media=X_media_no_tv)

        >>> # TV contribution = baseline - counterfactual
        >>> tv_contrib = pred.y_pred_mean - pred_no_tv.y_pred_mean
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Use training data as defaults
        if X_media is None:
            X_media = self.X_media_raw
        if time_idx is None:
            time_idx = self.time_idx
        if geo_idx is None:
            geo_idx = self.geo_idx
        if product_idx is None:
            product_idx = self.product_idx

        # Prepare media data (compute adstock and normalize)
        X_adstock_low, X_adstock_high = self._prepare_media_data_for_model(X_media)

        # Prepare controls (standardize)
        if X_controls is not None:
            X_controls_scaled = (X_controls - self.control_mean) / self.control_std
        elif self.X_controls is not None:
            X_controls_scaled = self.X_controls
        else:
            X_controls_scaled = None

        # Update model data
        with self.model:
            pm.set_data(
                {
                    "X_media_low": X_adstock_low,
                    "X_media_high": X_adstock_high,
                    "time_idx": time_idx.astype(np.int32),
                    "geo_idx": geo_idx.astype(np.int32),
                    "product_idx": product_idx.astype(np.int32),
                }
            )

            if X_controls_scaled is not None and self.n_controls > 0:
                pm.set_data({"X_controls": X_controls_scaled})

            # Sample posterior predictive
            ppc = pm.sample_posterior_predictive(
                self._trace,
                var_names=["y_obs"],
                random_seed=random_seed,
            )

        # Reset model data to training values
        with self.model:
            X_adstock_low_train, X_adstock_high_train = (
                self._prepare_media_data_for_model()
            )
            pm.set_data(
                {
                    "X_media_low": X_adstock_low_train,
                    "X_media_high": X_adstock_high_train,
                    "time_idx": self.time_idx,
                    "geo_idx": self.geo_idx,
                    "product_idx": self.product_idx,
                }
            )
            if self.X_controls is not None:
                pm.set_data({"X_controls": self.X_controls})

        # Extract predictions
        y_pred_samples = ppc.posterior_predictive["y_obs"].values
        # Flatten chains and draws: (n_chains, n_draws, n_obs) -> (n_samples, n_obs)
        n_chains, n_draws, n_obs = y_pred_samples.shape
        y_pred_samples = y_pred_samples.reshape(n_chains * n_draws, n_obs)

        # Convert to original scale if requested
        if return_original_scale:
            y_pred_samples = y_pred_samples * self.y_std + self.y_mean

        # Compute statistics
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)

        y_pred_hdi_low, y_pred_hdi_high = compute_hdi_bounds(
            y_pred_samples, hdi_prob=hdi_prob, axis=0
        )

        return PredictionResults(
            posterior_predictive=ppc,
            y_pred_mean=y_pred_mean,
            y_pred_std=y_pred_std,
            y_pred_hdi_low=y_pred_hdi_low,
            y_pred_hdi_high=y_pred_hdi_high,
            y_pred_samples=y_pred_samples,
        )

    def compute_component_decomposition(
        self,
        hdi_prob: float = 0.94,
    ) -> ComponentDecomposition:
        """
        Compute full component decomposition of the model.

        Returns contribution from each component:
        - Intercept (base)
        - Trend
        - Seasonality
        - Media (total and by channel)
        - Controls (total and by variable)
        - Geo effects (if applicable)
        - Product effects (if applicable)

        Parameters
        ----------
        hdi_prob : float
            HDI probability (not currently used, for future extension).

        Returns
        -------
        ComponentDecomposition
            Full breakdown of model components.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        posterior = self._trace.posterior

        # Helper to get mean across chains and draws
        def get_mean(var_name: str) -> np.ndarray:
            if var_name in posterior:
                return posterior[var_name].mean(dim=["chain", "draw"]).values
            return np.zeros(self.n_obs)

        # Extract all components (standardized/scaled space)
        intercept_scaled = get_mean("intercept_component")
        trend_scaled = get_mean("trend_component")
        seasonality_scaled = get_mean("seasonality_component")
        media_total_scaled = get_mean("media_total")
        controls_total_scaled = get_mean("controls_total")
        geo_scaled = get_mean("geo_component") if self.has_geo else None
        product_scaled = get_mean("product_component") if self.has_product else None

        # Channel-level media contributions (scaled space)
        channel_contributions_scaled = get_mean("channel_contributions")

        # Control-level contributions (scaled space)
        if self.n_controls > 0 and "control_contributions" in posterior:
            control_contributions_scaled = get_mean("control_contributions")
        else:
            control_contributions_scaled = None

        # Convert to original scale
        intercept = intercept_scaled * self.y_std + self.y_mean
        trend = trend_scaled * self.y_std
        seasonality = seasonality_scaled * self.y_std
        media_total = media_total_scaled * self.y_std
        controls_total = controls_total_scaled * self.y_std

        # Channel contributions (original scale)
        media_by_channel = pd.DataFrame(
            channel_contributions_scaled * self.y_std,
            index=self.panel.index,
            columns=self.channel_names,
        )

        # Control contributions (original scale)
        if control_contributions_scaled is not None:
            controls_by_var = pd.DataFrame(
                control_contributions_scaled * self.y_std,
                index=self.panel.index,
                columns=self.control_names,
            )
        else:
            controls_by_var = None

        # Geo/product effects
        geo_effects = geo_scaled * self.y_std if geo_scaled is not None else None
        product_effects = product_scaled * self.y_std if product_scaled is not None else None

        # Compute totals
        total_intercept = float(intercept.sum())
        total_trend = float(trend.sum())
        total_seasonality = float(seasonality.sum())
        total_media = float(media_total.sum())
        total_controls = float(controls_total.sum())
        total_geo = float(geo_effects.sum()) if geo_effects is not None else None
        total_product = (
            float(product_effects.sum()) if product_effects is not None else None
        )

        return ComponentDecomposition(
            intercept=intercept,
            trend=trend,
            seasonality=seasonality,
            media_total=media_total,
            media_by_channel=media_by_channel,
            controls_total=controls_total,
            controls_by_var=controls_by_var,
            geo_effects=geo_effects,
            product_effects=product_effects,
            total_intercept=total_intercept,
            total_trend=total_trend,
            total_seasonality=total_seasonality,
            total_media=total_media,
            total_controls=total_controls,
            total_geo=total_geo,
            total_product=total_product,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

    def compute_counterfactual_contributions(
        self,
        time_period: tuple[int, int] | None = None,
        channels: list[str] | None = None,
        compute_uncertainty: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> ContributionResults:
        """
        Compute channel contributions using counterfactual analysis.

        For each channel, this method:
        1. Gets baseline prediction (all channels present)
        2. Gets counterfactual prediction (channel zeroed out)
        3. Computes contribution as: baseline - counterfactual

        This approach properly accounts for saturation and adstock effects.

        Parameters
        ----------
        time_period : tuple[int, int], optional
            Time period (start_idx, end_idx) for contribution calculation.
            If None, uses entire time series.
            Indices are inclusive on both ends.
        channels : list[str], optional
            List of channel names to compute contributions for.
            If None, computes for all channels.
        compute_uncertainty : bool
            If True, computes HDI for contributions using posterior samples.
        hdi_prob : float
            Probability mass for HDI calculation (default 0.94).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ContributionResults
            Container with per-observation and total contributions.

        Examples
        --------
        >>> # Total contributions over entire period
        >>> contrib = mmm.compute_counterfactual_contributions()
        >>> print(contrib.summary())

        >>> # Contributions for specific time period (weeks 50-100)
        >>> contrib_period = mmm.compute_counterfactual_contributions(
        ...     time_period=(50, 100)
        ... )

        >>> # Contributions for specific channels
        >>> contrib_tv = mmm.compute_counterfactual_contributions(
        ...     channels=["TV", "Digital"]
        ... )
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names

        # Validate channels
        invalid_channels = [c for c in channels if c not in self.channel_names]
        if invalid_channels:
            raise ValueError(f"Unknown channels: {invalid_channels}")

        # Determine observation mask for time period
        time_mask = self._get_time_mask(time_period)

        # Get baseline prediction
        baseline_pred = self.predict(
            return_original_scale=True,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
        )

        # Store counterfactual predictions
        counterfactual_preds = {}

        # Compute counterfactual for each channel
        for channel in channels:
            # Create media data with this channel zeroed out
            X_media_counterfactual = self.X_media_raw.copy()
            ch_idx = self.channel_names.index(channel)
            X_media_counterfactual[:, ch_idx] = 0.0

            # Get counterfactual prediction
            cf_pred = self.predict(
                X_media=X_media_counterfactual,
                return_original_scale=True,
                hdi_prob=hdi_prob,
                random_seed=random_seed,
            )

            counterfactual_preds[channel] = cf_pred

        # Compute contributions
        # Contribution = baseline - counterfactual
        contribution_data = {}
        contribution_samples = {}  # For uncertainty

        for channel in channels:
            cf_pred = counterfactual_preds[channel]

            # Per-observation contribution
            contrib = baseline_pred.y_pred_mean - cf_pred.y_pred_mean
            contribution_data[channel] = contrib

            if compute_uncertainty:
                # Contribution samples for uncertainty
                contrib_samples = baseline_pred.y_pred_samples - cf_pred.y_pred_samples
                contribution_samples[channel] = contrib_samples

        # Create DataFrame
        channel_contributions = pd.DataFrame(
            contribution_data,
            index=self.panel.index,
        )

        # Apply time mask for totals
        if time_period is not None:
            contrib_masked = channel_contributions.iloc[time_mask]
        else:
            contrib_masked = channel_contributions

        # Total contributions
        total_contributions = contrib_masked.sum()

        # Percentage of total
        total_effect = total_contributions.sum()
        contribution_pct = (
            (total_contributions / total_effect * 100)
            if total_effect != 0
            else total_contributions * 0
        )

        # HDI for totals
        contribution_hdi_low = None
        contribution_hdi_high = None

        if compute_uncertainty:
            hdi_low_values = {}
            hdi_high_values = {}

            for channel in channels:
                samples = contribution_samples[channel]
                if time_period is not None:
                    samples = samples[:, time_mask]

                # Sum over time for each sample
                total_samples = samples.sum(axis=1)

                low, high = compute_hdi_bounds(total_samples, hdi_prob=hdi_prob, axis=0)
                hdi_low_values[channel] = low
                hdi_high_values[channel] = high

            contribution_hdi_low = pd.Series(hdi_low_values)
            contribution_hdi_high = pd.Series(hdi_high_values)

        # Store counterfactual predictions as arrays
        cf_pred_arrays = {
            ch: pred.y_pred_mean for ch, pred in counterfactual_preds.items()
        }

        return ContributionResults(
            channel_contributions=channel_contributions,
            total_contributions=total_contributions,
            contribution_pct=contribution_pct,
            baseline_prediction=baseline_pred.y_pred_mean,
            counterfactual_predictions=cf_pred_arrays,
            time_period=time_period,
            contribution_hdi_low=contribution_hdi_low,
            contribution_hdi_high=contribution_hdi_high,
        )

    def compute_marginal_contributions(
        self,
        spend_increase_pct: float = 10.0,
        time_period: tuple[int, int] | None = None,
        channels: list[str] | None = None,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute marginal contributions for a given spend increase.

        This shows how much additional outcome we'd get from increasing
        spend by a given percentage, accounting for saturation.

        Parameters
        ----------
        spend_increase_pct : float
            Percentage increase in spend to simulate (default 10%).
        time_period : tuple[int, int], optional
            Time period for calculation. If None, uses entire series.
        channels : list[str], optional
            Channels to analyze. If None, uses all channels.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with marginal contribution analysis.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names
        multiplier = 1.0 + spend_increase_pct / 100.0

        # Determine time mask
        time_mask = self._get_time_mask(time_period)

        # Get baseline prediction
        baseline_pred = self.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()

        results = []

        for channel in channels:
            ch_idx = self.channel_names.index(channel)

            # Create media data with increased spend for this channel
            X_media_increased = self.X_media_raw.copy()
            X_media_increased[:, ch_idx] *= multiplier

            # Get prediction with increased spend
            increased_pred = self.predict(
                X_media=X_media_increased,
                random_seed=random_seed,
            )
            increased_total = increased_pred.y_pred_mean[time_mask].sum()

            # Compute marginal contribution
            marginal_contrib = increased_total - baseline_total

            # Current spend
            current_spend = self.X_media_raw[time_mask, ch_idx].sum()
            spend_increase = current_spend * (multiplier - 1)

            # Marginal ROAS
            marginal_roas = (
                marginal_contrib / spend_increase if spend_increase > 0 else 0
            )

            results.append(
                {
                    "Channel": channel,
                    "Current Spend": current_spend,
                    f"Spend Increase ({spend_increase_pct}%)": spend_increase,
                    "Marginal Contribution": marginal_contrib,
                    "Marginal ROAS": marginal_roas,
                }
            )

        return pd.DataFrame(results)

    def what_if_scenario(
        self,
        spend_changes: dict[str, float],
        time_period: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> dict:
        """
        Run a what-if scenario with custom spend changes.

        Parameters
        ----------
        spend_changes : dict[str, float]
            Dictionary mapping channel names to spend multipliers.
            E.g., {"TV": 1.2, "Digital": 0.8} means +20% TV, -20% Digital.
        time_period : tuple[int, int], optional
            Time period for calculation.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with scenario analysis results.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Determine time mask
        time_mask = self._get_time_mask(time_period)

        # Get baseline
        baseline_pred = self.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()

        # Create scenario media data
        X_media_scenario = self.X_media_raw.copy()

        spend_summary = {}
        for channel, multiplier in spend_changes.items():
            if channel not in self.channel_names:
                raise ValueError(f"Unknown channel: {channel}")

            ch_idx = self.channel_names.index(channel)
            original_spend = X_media_scenario[time_mask, ch_idx].sum()
            X_media_scenario[:, ch_idx] *= multiplier
            new_spend = X_media_scenario[time_mask, ch_idx].sum()

            spend_summary[channel] = {
                "original": original_spend,
                "scenario": new_spend,
                "change": new_spend - original_spend,
                "change_pct": (multiplier - 1) * 100,
            }

        # Get scenario prediction
        scenario_pred = self.predict(
            X_media=X_media_scenario,
            random_seed=random_seed,
        )
        scenario_total = scenario_pred.y_pred_mean[time_mask].sum()

        # Compute impact
        outcome_change = scenario_total - baseline_total
        outcome_change_pct = (
            (outcome_change / baseline_total * 100) if baseline_total != 0 else 0
        )

        return {
            "baseline_outcome": baseline_total,
            "scenario_outcome": scenario_total,
            "outcome_change": outcome_change,
            "outcome_change_pct": outcome_change_pct,
            "spend_changes": spend_summary,
            "baseline_prediction": baseline_pred.y_pred_mean,
            "scenario_prediction": scenario_pred.y_pred_mean,
        }

    def sample_prior_predictive(self, samples: int = 500) -> az.InferenceData:
        """Sample from prior predictive distribution."""
        with self.model:
            return pm.sample_prior_predictive(samples=samples)

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return az.summary(self._trace, var_names=var_names)

    # =========================================================================
    # Save and Load Methods
    # =========================================================================

    def save(
        self,
        path: str | Path,
        save_trace: bool = True,
        compress: bool = True,
    ) -> None:
        """
        Save the fitted model to disk.

        This saves all necessary components to reconstruct and use the model:
        - Model configuration (MFFConfig, ModelConfig, TrendConfig)
        - Fitted trace (ArviZ InferenceData)
        - Scaling parameters for predictions
        - Adstock alphas

        The model can be loaded later with `BayesianMMM.load()`.

        Parameters
        ----------
        path : str or Path
            Directory path to save the model. Will be created if it doesn't exist.
        save_trace : bool, default True
            Whether to save the fitted trace. If False, only configs and
            scaling parameters are saved (useful for just saving the setup).
        compress : bool, default True
            Whether to compress the saved files (uses gzip for NetCDF).

        Examples
        --------
        >>> # Fit and save a model
        >>> mmm = BayesianMMM(panel, model_config, trend_config)
        >>> results = mmm.fit()
        >>> mmm.save("models/my_mmm_model")

        >>> # Load and use later
        >>> mmm_loaded = BayesianMMM.load("models/my_mmm_model", panel)
        >>> predictions = mmm_loaded.predict()

        Notes
        -----
        The saved model does NOT include the panel data. When loading,
        you must provide compatible panel data (same structure, channels, etc.).

        See Also
        --------
        MMMSerializer : The underlying serialization class.
        """
        from .serialization import MMMSerializer

        MMMSerializer.save(self, path, save_trace=save_trace, compress=compress)

    @classmethod
    def load(
        cls,
        path: str | Path,
        panel: PanelDataset,
        rebuild_model: bool = True,
    ) -> BayesianMMM:
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Directory path where the model was saved.
        panel : PanelDataset
            Panel data to use with the loaded model. Must be compatible
            with the original data (same channels, controls, dimensions).
        rebuild_model : bool, default True
            Whether to rebuild the PyMC model. Set to False if you only
            need access to the trace and don't need to make predictions.

        Returns
        -------
        BayesianMMM
            Loaded model instance with fitted trace (if available).

        Examples
        --------
        >>> # Load a saved model
        >>> panel = load_mff(data, config)
        >>> mmm = BayesianMMM.load("models/my_mmm_model", panel)

        >>> # Make predictions
        >>> predictions = mmm.predict()

        >>> # Access the trace
        >>> summary = mmm.summary()

        Raises
        ------
        ValueError
            If the panel data is incompatible with the saved model.
        FileNotFoundError
            If the model files are not found.

        See Also
        --------
        MMMSerializer : The underlying serialization class.
        """
        from .serialization import MMMSerializer

        return MMMSerializer.load(path, panel, rebuild_model=rebuild_model)

    def save_trace_only(self, path: str | Path) -> None:
        """
        Save only the fitted trace to a file.

        This is useful for quick saves when you don't need to save
        the full model configuration.

        Parameters
        ----------
        path : str or Path
            File path for the trace (should end in .nc or .nc.gz).
        """
        if self._trace is None:
            raise ValueError("No trace to save. Fit the model first.")

        from .serialization import MMMSerializer

        MMMSerializer.save_trace_only(self._trace, path)

    def load_trace_only(self, path: str | Path) -> None:
        """
        Load a trace from a file into the current model.

        Parameters
        ----------
        path : str or Path
            File path to the trace (.nc or .nc.gz).
        """
        from .serialization import MMMSerializer

        self._trace = MMMSerializer.load_trace_only(path)

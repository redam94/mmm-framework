"""Data preparation utilities for BayesianMMM.

This module provides helper classes for preparing and standardizing
data for use in Bayesian Marketing Mix Models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .transforms import (
    geometric_adstock_2d,
    create_fourier_features,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .data_loader import PanelDataset
    from .config import SeasonalityConfig, MFFConfig


@dataclass
class ScalingParameters:
    """Container for data scaling parameters.

    These parameters are needed to transform predictions back to the
    original scale and for consistent predictions on new data.

    Attributes
    ----------
    y_mean : float
        Mean of the target variable.
    y_std : float
        Standard deviation of the target variable.
    media_max : dict[str, float]
        Maximum adstocked value for each media channel.
    control_mean : NDArray | None
        Mean of control variables (None if no controls).
    control_std : NDArray | None
        Standard deviation of control variables.
    """

    y_mean: float
    y_std: float
    media_max: dict[str, float]
    control_mean: NDArray | None = None
    control_std: NDArray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "media_max": {k: float(v) for k, v in self.media_max.items()},
        }
        if self.control_mean is not None:
            result["control_mean"] = self.control_mean.tolist()
            result["control_std"] = self.control_std.tolist()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalingParameters:
        """Create from dictionary."""
        return cls(
            y_mean=data["y_mean"],
            y_std=data["y_std"],
            media_max=data["media_max"],
            control_mean=(
                np.array(data["control_mean"]) if "control_mean" in data else None
            ),
            control_std=(
                np.array(data["control_std"]) if "control_std" in data else None
            ),
        )


@dataclass
class PreparedData:
    """Container for prepared/transformed data.

    This holds all the preprocessed data needed for model building,
    including standardized values, indices, and feature matrices.

    Attributes
    ----------
    y : NDArray
        Standardized target variable.
    y_raw : NDArray
        Raw target variable.
    X_media_adstocked : dict[float, NDArray]
        Adstocked and normalized media data by alpha value.
    X_media_raw : NDArray
        Raw media data.
    X_controls : NDArray | None
        Standardized control variables (None if no controls).
    X_controls_raw : NDArray | None
        Raw control variables.
    scaling_params : ScalingParameters
        Parameters used for standardization.
    n_obs : int
        Number of observations.
    n_channels : int
        Number of media channels.
    n_controls : int
        Number of control variables.
    channel_names : list[str]
        Names of media channels.
    control_names : list[str]
        Names of control variables.
    time_idx : NDArray
        Time period index for each observation.
    geo_idx : NDArray
        Geography index for each observation.
    product_idx : NDArray
        Product index for each observation.
    n_periods : int
        Number of unique time periods.
    n_geos : int
        Number of unique geographies.
    n_products : int
        Number of unique products.
    has_geo : bool
        Whether data has geography dimension.
    has_product : bool
        Whether data has product dimension.
    t_scaled : NDArray
        Time values scaled to [0, 1].
    seasonality_features : dict[str, NDArray]
        Fourier features for seasonality.
    trend_features : dict[str, Any]
        Features for trend modeling.
    """

    # Target data
    y: NDArray
    y_raw: NDArray

    # Media data
    X_media_adstocked: dict[float, NDArray]
    X_media_raw: NDArray

    # Control data
    X_controls: NDArray | None
    X_controls_raw: NDArray | None

    # Scaling
    scaling_params: ScalingParameters

    # Dimensions
    n_obs: int
    n_channels: int
    n_controls: int
    channel_names: list[str]
    control_names: list[str]

    # Indices
    time_idx: NDArray
    geo_idx: NDArray
    product_idx: NDArray

    # Dimension sizes
    n_periods: int
    n_geos: int
    n_products: int
    has_geo: bool
    has_product: bool

    # Time
    t_scaled: NDArray

    # Features
    seasonality_features: dict[str, NDArray] = field(default_factory=dict)
    trend_features: dict[str, Any] = field(default_factory=dict)

    # Optional geo/product names
    geo_names: list[str] | None = None
    product_names: list[str] | None = None


class DataPreparator:
    """Prepares panel data for Bayesian MMM.

    This class handles all data preprocessing steps including:
    - Standardization of target and control variables
    - Adstock transformation and normalization of media data
    - Creation of seasonality features (Fourier terms)
    - Creation of trend features (spline/piecewise/GP)
    - Index creation for hierarchical dimensions

    Parameters
    ----------
    panel : PanelDataset
        The panel dataset to prepare.
    adstock_alphas : list[float]
        Alpha values for geometric adstock.
    seasonality_config : SeasonalityConfig | None
        Configuration for seasonality features.
    trend_config : Any | None
        Configuration for trend features.

    Examples
    --------
    >>> from mmm_framework.data_preparation import DataPreparator
    >>> preparator = DataPreparator(
    ...     panel=panel,
    ...     adstock_alphas=[0.0, 0.3, 0.5, 0.7, 0.9],
    ... )
    >>> prepared = preparator.prepare()
    >>> print(prepared.n_obs, prepared.n_channels)
    """

    def __init__(
        self,
        panel: PanelDataset,
        adstock_alphas: list[float],
        seasonality_config: Any | None = None,
        trend_config: Any | None = None,
    ):
        self.panel = panel
        self.adstock_alphas = adstock_alphas
        self.seasonality_config = seasonality_config
        self.trend_config = trend_config
        self._mff_config = panel.config

    def prepare(self) -> PreparedData:
        """Prepare all data for model building.

        Returns
        -------
        PreparedData
            Container with all prepared data.
        """
        # Extract raw data
        y_raw = self.panel.y.values.astype(np.float64)
        X_media_raw = self.panel.X_media.values.astype(np.float64)

        if self.panel.X_controls is not None and self.panel.X_controls.shape[1] > 0:
            X_controls_raw = self.panel.X_controls.values.astype(np.float64)
        else:
            X_controls_raw = None

        # Dimensions
        n_obs = len(y_raw)
        n_channels = X_media_raw.shape[1]
        n_controls = X_controls_raw.shape[1] if X_controls_raw is not None else 0

        channel_names = list(self.panel.coords.channels)
        control_names = list(self.panel.coords.controls) if n_controls > 0 else []

        # Standardize target
        y_mean = float(y_raw.mean())
        y_std = float(y_raw.std()) + 1e-8
        y = (y_raw - y_mean) / y_std

        # Compute adstocked media
        X_media_adstocked, media_max = self._compute_adstocked_media(
            X_media_raw, channel_names
        )

        # Standardize controls
        if X_controls_raw is not None:
            control_mean = X_controls_raw.mean(axis=0)
            control_std = X_controls_raw.std(axis=0) + 1e-8
            X_controls = (X_controls_raw - control_mean) / control_std
        else:
            X_controls = None
            control_mean = None
            control_std = None

        # Create scaling parameters
        scaling_params = ScalingParameters(
            y_mean=y_mean,
            y_std=y_std,
            media_max=media_max,
            control_mean=control_mean,
            control_std=control_std,
        )

        # Geo/product info
        has_geo = self.panel.coords.has_geo
        has_product = self.panel.coords.has_product
        n_geos = self.panel.coords.n_geos
        n_products = self.panel.coords.n_products

        geo_names = list(self.panel.coords.geographies) if has_geo else None
        product_names = list(self.panel.coords.products) if has_product else None

        # Indices
        geo_idx = (
            self._get_group_indices("geography", geo_names)
            if has_geo
            else np.zeros(n_obs, dtype=np.int32)
        )
        product_idx = (
            self._get_group_indices("product", product_names)
            if has_product
            else np.zeros(n_obs, dtype=np.int32)
        )
        time_idx = self._get_time_index()

        # Time info
        n_periods = self.panel.coords.n_periods
        t_scaled = np.linspace(0, 1, n_periods)

        # Seasonality features
        seasonality_features = self._prepare_seasonality(n_periods)

        # Trend features
        trend_features = self._prepare_trend(n_periods)

        return PreparedData(
            y=y,
            y_raw=y_raw,
            X_media_adstocked=X_media_adstocked,
            X_media_raw=X_media_raw,
            X_controls=X_controls,
            X_controls_raw=X_controls_raw,
            scaling_params=scaling_params,
            n_obs=n_obs,
            n_channels=n_channels,
            n_controls=n_controls,
            channel_names=channel_names,
            control_names=control_names,
            time_idx=time_idx,
            geo_idx=geo_idx,
            product_idx=product_idx,
            n_periods=n_periods,
            n_geos=n_geos,
            n_products=n_products,
            has_geo=has_geo,
            has_product=has_product,
            t_scaled=t_scaled,
            seasonality_features=seasonality_features,
            trend_features=trend_features,
            geo_names=geo_names,
            product_names=product_names,
        )

    def _compute_adstocked_media(
        self,
        X_media_raw: NDArray,
        channel_names: list[str],
    ) -> tuple[dict[float, NDArray], dict[str, float]]:
        """Compute adstocked and normalized media data.

        Returns
        -------
        tuple
            (adstocked_dict, media_max_dict)
        """
        media_max: dict[str, float] = {}
        adstocked_dict: dict[float, NDArray] = {}

        # First pass: compute max values
        for alpha in self.adstock_alphas:
            adstocked = geometric_adstock_2d(X_media_raw, alpha)
            for c in range(len(channel_names)):
                key = channel_names[c]
                current_max = float(adstocked[:, c].max())
                if key not in media_max:
                    media_max[key] = current_max
                else:
                    media_max[key] = max(media_max[key], current_max)

        # Second pass: normalize
        for alpha in self.adstock_alphas:
            adstocked = geometric_adstock_2d(X_media_raw, alpha)
            normalized = np.zeros_like(adstocked)
            for c, ch_name in enumerate(channel_names):
                normalized[:, c] = adstocked[:, c] / (media_max[ch_name] + 1e-8)
            adstocked_dict[alpha] = normalized

        return adstocked_dict, media_max

    def _get_group_indices(
        self,
        level_name: str,
        categories: list[str] | None,
    ) -> NDArray:
        """Get group indices for a hierarchical level."""
        cols = self._mff_config.columns
        col_name = getattr(cols, level_name)

        if isinstance(self.panel.index, pd.MultiIndex) and categories:
            values = self.panel.index.get_level_values(col_name)
            return pd.Categorical(values, categories=categories).codes.astype(np.int32)
        return np.zeros(len(self.panel.y), dtype=np.int32)

    def _get_time_index(self) -> NDArray:
        """Get time index for each observation."""
        cols = self._mff_config.columns

        if isinstance(self.panel.index, pd.MultiIndex):
            period_values = self.panel.index.get_level_values(cols.period)
            periods_unique = list(self.panel.coords.periods)
            return pd.Categorical(
                period_values, categories=periods_unique
            ).codes.astype(np.int32)
        return np.arange(len(self.panel.y), dtype=np.int32)

    def _prepare_seasonality(self, n_periods: int) -> dict[str, NDArray]:
        """Prepare Fourier features for seasonality."""
        features: dict[str, NDArray] = {}
        t = np.arange(n_periods)

        if self.seasonality_config is not None:
            if (
                hasattr(self.seasonality_config, "yearly")
                and self.seasonality_config.yearly
            ):
                if self.seasonality_config.yearly > 0:
                    period = 52  # Weekly data
                    order = self.seasonality_config.yearly
                    fourier_features = create_fourier_features(t, period, order)
                    if fourier_features.shape[1] > 0:
                        features["yearly"] = fourier_features

        return features

    def _prepare_trend(self, n_periods: int) -> dict[str, Any]:
        """Prepare trend features based on configuration."""
        t_unique = np.linspace(0, 1, n_periods)
        features: dict[str, Any] = {}

        if self.trend_config is None:
            return features

        # Import TrendType here to avoid circular imports
        from .model import TrendType

        if self.trend_config.type == TrendType.SPLINE:
            features["spline_basis"] = create_bspline_basis(
                t_unique,
                n_knots=self.trend_config.n_knots,
                degree=self.trend_config.spline_degree,
            )
            features["n_spline_coef"] = features["spline_basis"].shape[1]

        elif self.trend_config.type == TrendType.PIECEWISE:
            s, A = create_piecewise_trend_matrix(
                t_unique,
                n_changepoints=self.trend_config.n_changepoints,
                changepoint_range=self.trend_config.changepoint_range,
            )
            features["changepoints"] = s
            features["changepoint_matrix"] = A

        elif self.trend_config.type == TrendType.GP:
            features["gp_config"] = {
                "lengthscale_mu": self.trend_config.gp_lengthscale_prior_mu,
                "lengthscale_sigma": self.trend_config.gp_lengthscale_prior_sigma,
                "amplitude_sigma": self.trend_config.gp_amplitude_prior_sigma,
                "n_basis": self.trend_config.gp_n_basis,
                "c": self.trend_config.gp_c,
            }

        return features


def standardize_array(
    data: NDArray,
    epsilon: float = 1e-8,
) -> tuple[NDArray, float, float]:
    """Standardize an array to zero mean and unit variance.

    Parameters
    ----------
    data : NDArray
        Input data array.
    epsilon : float
        Small value added to std to prevent division by zero.

    Returns
    -------
    tuple
        (standardized_data, mean, std)
    """
    mean = float(data.mean())
    std = float(data.std()) + epsilon
    return (data - mean) / std, mean, std


def unstandardize_array(
    data: NDArray,
    mean: float,
    std: float,
) -> NDArray:
    """Reverse standardization.

    Parameters
    ----------
    data : NDArray
        Standardized data.
    mean : float
        Original mean.
    std : float
        Original standard deviation.

    Returns
    -------
    NDArray
        Data in original scale.
    """
    return data * std + mean

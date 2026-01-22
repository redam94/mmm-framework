"""
BayesianMMM - Main model class.

This module contains the BayesianMMM class which orchestrates
model building, fitting, and prediction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..config import ModelConfig
from ..data_loader import PanelDataset
from ..utils import compute_hdi_bounds
from ..transforms import (
    geometric_adstock_2d,
    create_fourier_features,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)

from .results import (
    MMMResults,
    PredictionResults,
    ContributionResults,
    ComponentDecomposition,
)
from .trend_config import TrendType, TrendConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        self._media_max = {}
        self.X_media_adstocked = {}

        for alpha in self.adstock_alphas:
            adstocked = geometric_adstock_2d(self.X_media_raw, alpha)
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
        self.t_scaled = np.linspace(0, 1, self.n_periods)

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
            self.trend_features["spline_basis"] = create_bspline_basis(
                t_unique,
                n_knots=self.trend_config.n_knots,
                degree=self.trend_config.spline_degree,
            )
            self.trend_features["n_spline_coef"] = self.trend_features[
                "spline_basis"
            ].shape[1]

        elif self.trend_config.type == TrendType.PIECEWISE:
            s, A = create_piecewise_trend_matrix(
                t_unique,
                n_changepoints=self.trend_config.n_changepoints,
                changepoint_range=self.trend_config.changepoint_range,
            )
            self.trend_features["changepoints"] = s
            self.trend_features["changepoint_matrix"] = A

        elif self.trend_config.type == TrendType.GP:
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
        """Create boolean mask for time period filtering."""
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

        if self.trend_config.type == TrendType.SPLINE:
            n_coef = self.trend_features.get("n_spline_coef", 0)
            coords["spline_idx"] = list(range(n_coef))

        elif self.trend_config.type == TrendType.PIECEWISE:
            n_cp = len(self.trend_features.get("changepoints", []))
            coords["changepoint"] = list(range(n_cp))

        return coords

    def _build_trend_component(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build the trend component based on configuration."""
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

        k = pm.Normal(
            "trend_k",
            mu=self.trend_config.growth_prior_mu,
            sigma=self.trend_config.growth_prior_sigma,
        )

        delta = pm.Laplace(
            "trend_delta",
            mu=0,
            b=self.trend_config.changepoint_prior_scale,
            shape=n_changepoints,
            dims="changepoint",
        )

        m = pm.Normal("trend_m", mu=0, sigma=0.5)

        t_unique = np.linspace(0, 1, self.n_periods)
        t_unique_tensor = pt.as_tensor_variable(t_unique)
        A_tensor = pt.as_tensor_variable(A)
        s_tensor = pt.as_tensor_variable(s)

        gamma = -s_tensor * delta
        trend_unique = (
            k * t_unique_tensor + pt.dot(A_tensor, delta) + m + pt.dot(A_tensor, gamma)
        )

        return trend_unique[time_idx]

    def _build_spline_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build B-spline trend."""
        basis = self.trend_features["spline_basis"]
        n_coef = basis.shape[1]

        spline_coef_raw = pm.Normal(
            "spline_coef_raw", mu=0, sigma=1, shape=n_coef, dims="spline_idx"
        )

        spline_scale = pm.HalfNormal(
            "spline_scale", sigma=self.trend_config.spline_prior_sigma
        )

        spline_coef = pm.Deterministic(
            "spline_coef", spline_scale * pt.cumsum(spline_coef_raw), dims="spline_idx"
        )

        basis_tensor = pt.as_tensor_variable(basis)
        trend_unique = pt.dot(basis_tensor, spline_coef)
        trend_unique = trend_unique - trend_unique.mean()

        return trend_unique[time_idx]

    def _build_gp_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build Gaussian Process trend using HSGP approximation."""
        gp_config = self.trend_features["gp_config"]

        gp_lengthscale = pm.LogNormal(
            "gp_lengthscale",
            mu=np.log(gp_config["lengthscale_mu"]),
            sigma=gp_config["lengthscale_sigma"],
        )

        gp_amplitude = pm.HalfNormal("gp_amplitude", sigma=gp_config["amplitude_sigma"])

        try:
            import pymc.gp as gp_module

            cov_func = gp_amplitude**2 * gp_module.cov.Matern32(
                input_dim=1, ls=gp_lengthscale
            )

            gp = gp_module.HSGP(
                m=[gp_config["n_basis"]], c=gp_config["c"], cov_func=cov_func
            )

            t_unique = np.linspace(-1, 1, self.n_periods).reshape(-1, 1)
            trend_unique = gp.prior("trend_gp", X=t_unique)

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
        """Build GP trend using explicit basis function approximation."""
        gp_config = self.trend_features["gp_config"]
        n_basis = gp_config["n_basis"]

        t_unique = np.linspace(0, 1, self.n_periods)
        frequencies = np.arange(1, n_basis + 1)

        basis_sin = np.sin(2 * np.pi * np.outer(t_unique, frequencies))
        basis_cos = np.cos(2 * np.pi * np.outer(t_unique, frequencies))
        basis = np.hstack([basis_sin, basis_cos])

        basis_tensor = pt.as_tensor_variable(basis)

        omega = 2 * np.pi * frequencies
        omega_tensor = pt.as_tensor_variable(omega)

        gp_coef = pm.Normal("gp_coef", mu=0, sigma=1, shape=2 * n_basis)

        spectral_weights_sin = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights_cos = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights = pt.concatenate([spectral_weights_sin, spectral_weights_cos])

        scaled_coef = amplitude * gp_coef * pt.sqrt(spectral_weights / n_basis)
        trend_unique = pt.dot(basis_tensor, scaled_coef)
        trend_unique = trend_unique - trend_unique.mean()

        return trend_unique[time_idx]

    def _prepare_media_data_for_model(
        self, X_media_raw: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare media data for model (compute adstock at low/high alpha)."""
        if X_media_raw is None:
            X_media_raw = self.X_media_raw

        alpha_low = self.adstock_alphas[0]
        alpha_high = self.adstock_alphas[-1]

        adstock_low = geometric_adstock_2d(X_media_raw, alpha_low)
        adstock_high = geometric_adstock_2d(X_media_raw, alpha_high)

        for c, ch_name in enumerate(self.channel_names):
            max_val = self._media_max[ch_name] + 1e-8
            adstock_low[:, c] = adstock_low[:, c] / max_val
            adstock_high[:, c] = adstock_high[:, c] / max_val

        return adstock_low, adstock_high

    def _build_model(self) -> pm.Model:
        """Build the PyMC model with Data for prediction support."""
        coords = self._build_coords()
        X_adstock_low, X_adstock_high = self._prepare_media_data_for_model()

        with pm.Model(coords=coords) as model:
            # MUTABLE DATA
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

            # INTERCEPT
            intercept = pm.Normal("intercept", mu=0, sigma=0.5)

            # TREND
            trend = self._build_trend_component(model, time_idx_data)
            pm.Deterministic("trend_component", trend)

            # SEASONALITY
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
                features_tensor = pt.as_tensor_variable(features)
                season_effect = pt.dot(features_tensor, season_coef)
                seasonality_at_periods = seasonality_at_periods + season_effect

            seasonality = seasonality_at_periods[time_idx_data]
            pm.Deterministic("seasonality_component", seasonality)
            pm.Deterministic("seasonality_by_period", seasonality_at_periods)

            # GEO EFFECTS
            if self.has_geo and self.hierarchical_config.pool_across_geo:
                geo_sigma = pm.HalfNormal("geo_sigma", sigma=0.3)
                geo_offset = pm.Normal("geo_offset", mu=0, sigma=1, shape=self.n_geos)
                geo_effect = geo_sigma * geo_offset
                geo_contribution = geo_effect[geo_idx_data]
            else:
                geo_contribution = pt.zeros(n_obs_data)

            # PRODUCT EFFECTS
            if self.has_product and self.hierarchical_config.pool_across_product:
                product_sigma = pm.HalfNormal("product_sigma", sigma=0.3)
                product_offset = pm.Normal(
                    "product_offset", mu=0, sigma=1, shape=self.n_products
                )
                product_effect = product_sigma * product_offset
                product_contribution = product_effect[product_idx_data]
            else:
                product_contribution = pt.zeros(n_obs_data)

            # MEDIA EFFECTS
            channel_contribs = []

            for c, channel_name in enumerate(self.channel_names):
                x_low = X_media_low_data[:, c]
                x_high = X_media_high_data[:, c]

                adstock_mix = pm.Beta(f"adstock_{channel_name}", alpha=2, beta=2)
                x_adstocked = (1 - adstock_mix) * x_low + adstock_mix * x_high

                sat_lam = pm.Exponential(f"sat_lam_{channel_name}", lam=0.5)
                exponent = pt.clip(-sat_lam * x_adstocked, -20, 0)
                x_saturated = 1 - pt.exp(exponent)

                beta = pm.HalfNormal(f"beta_{channel_name}", sigma=0.5)
                channel_contrib = beta * x_saturated
                channel_contribs.append(channel_contrib)

            media_matrix = pt.stack(channel_contribs, axis=1)
            media_contribution = media_matrix.sum(axis=1)

            pm.Deterministic(
                "channel_contributions", media_matrix, dims=("obs", "channel")
            )
            pm.Deterministic("media_total", media_contribution)

            # CONTROL EFFECTS
            if self.n_controls > 0:
                beta_controls = pm.Normal(
                    "beta_controls",
                    mu=0,
                    sigma=0.5,
                    shape=self.n_controls,
                )
                control_contribution = pt.dot(X_controls_data, beta_controls)
            else:
                control_contribution = pt.zeros(n_obs_data)

            # COMBINE AND LIKELIHOOD
            mu = (
                intercept
                + trend
                + seasonality
                + geo_contribution
                + product_contribution
                + media_contribution
                + control_contribution
            )

            sigma = pm.HalfNormal("sigma", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic("y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs")

        return model

    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def get_prior(
        self,
        samples: int = 500,
        random_seed: int | None = None,
    ) -> az.InferenceData:
        """Sample from the prior distribution of the model."""
        with self.model:
            prior_trace = pm.sample_prior_predictive(
                samples=samples, random_seed=random_seed
            )
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

        Args:
            draws: Number of posterior draws per chain. Default from config.
            tune: Number of tuning samples. Default from config.
            chains: Number of MCMC chains. Default from config.
            target_accept: Target acceptance rate for NUTS. Default 0.9.
            random_seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to pm.sample().

        Returns:
            Fitted model results with diagnostics.
        """
        draws = draws or self.model_config.n_draws
        tune = tune or self.model_config.n_tune
        chains = chains or self.model_config.n_chains
        target_accept = target_accept or 0.9
        random_seed = random_seed or self.model_config.optim_seed

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

        try:
            div_count = int(trace.sample_stats.diverging.sum().values)
        except Exception:
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

        return results

    def predict(
        self,
        X_media: np.ndarray | None = None,
        X_controls: np.ndarray | None = None,
        return_original_scale: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> PredictionResults:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        X_media : np.ndarray, optional
            New media data for counterfactual. If None, uses training data.
        X_controls : np.ndarray, optional
            New control data. If None, uses training data.
        return_original_scale : bool
            If True, returns predictions in original scale.
        hdi_prob : float
            HDI probability for uncertainty intervals.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        PredictionResults
            Prediction results with samples and uncertainty bounds.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_adstock_low, X_adstock_high = self._prepare_media_data_for_model(X_media)

        with self.model:
            pm.set_data({
                "X_media_low": X_adstock_low,
                "X_media_high": X_adstock_high,
            })

            if X_controls is not None and self.n_controls > 0:
                X_controls_std = (X_controls - self.control_mean) / self.control_std
                pm.set_data({"X_controls": X_controls_std})

            pp = pm.sample_posterior_predictive(
                self._trace,
                var_names=["y_obs"],
                random_seed=random_seed,
            )

        y_samples = pp.posterior_predictive["y_obs"].values
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])

        if return_original_scale:
            y_samples = y_samples * self.y_std + self.y_mean

        y_mean = y_samples.mean(axis=0)
        y_std = y_samples.std(axis=0)
        y_hdi_low, y_hdi_high = compute_hdi_bounds(y_samples, hdi_prob=hdi_prob, axis=0)

        return PredictionResults(
            posterior_predictive=pp,
            y_pred_mean=y_mean,
            y_pred_std=y_std,
            y_pred_hdi_low=y_hdi_low,
            y_pred_hdi_high=y_hdi_high,
            y_pred_samples=y_samples,
        )

    def compute_component_decomposition(self) -> ComponentDecomposition:
        """
        Decompose predictions into component contributions.

        Returns
        -------
        ComponentDecomposition
            Full component breakdown in original scale.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        posterior = self._trace.posterior

        def get_mean(var_name: str) -> np.ndarray:
            if var_name in posterior:
                return posterior[var_name].mean(dim=["chain", "draw"]).values
            return np.zeros(self.n_obs)

        intercept_scaled = get_mean("intercept_component")
        trend_scaled = get_mean("trend_component")
        seasonality_scaled = get_mean("seasonality_component")
        media_total_scaled = get_mean("media_total")
        controls_total_scaled = get_mean("controls_total")
        geo_scaled = get_mean("geo_component") if self.has_geo else None
        product_scaled = get_mean("product_component") if self.has_product else None

        channel_contributions_scaled = get_mean("channel_contributions")

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

        media_by_channel = pd.DataFrame(
            channel_contributions_scaled * self.y_std,
            index=self.panel.index,
            columns=self.channel_names,
        )

        if control_contributions_scaled is not None:
            controls_by_var = pd.DataFrame(
                control_contributions_scaled * self.y_std,
                index=self.panel.index,
                columns=self.control_names,
            )
        else:
            controls_by_var = None

        geo_effects = geo_scaled * self.y_std if geo_scaled is not None else None
        product_effects = product_scaled * self.y_std if product_scaled is not None else None

        total_intercept = float(intercept.sum())
        total_trend = float(trend.sum())
        total_seasonality = float(seasonality.sum())
        total_media = float(media_total.sum())
        total_controls = float(controls_total.sum())
        total_geo = float(geo_effects.sum()) if geo_effects is not None else None
        total_product = float(product_effects.sum()) if product_effects is not None else None

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

        Parameters
        ----------
        time_period : tuple[int, int], optional
            Time period (start_idx, end_idx) for contribution calculation.
        channels : list[str], optional
            List of channel names to compute contributions for.
        compute_uncertainty : bool
            If True, computes HDI for contributions.
        hdi_prob : float
            Probability mass for HDI calculation.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ContributionResults
            Container with per-observation and total contributions.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names

        invalid_channels = [c for c in channels if c not in self.channel_names]
        if invalid_channels:
            raise ValueError(f"Unknown channels: {invalid_channels}")

        time_mask = self._get_time_mask(time_period)

        baseline_pred = self.predict(
            return_original_scale=True,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
        )

        counterfactual_preds = {}

        for channel in channels:
            X_media_counterfactual = self.X_media_raw.copy()
            ch_idx = self.channel_names.index(channel)
            X_media_counterfactual[:, ch_idx] = 0.0

            cf_pred = self.predict(
                X_media=X_media_counterfactual,
                return_original_scale=True,
                hdi_prob=hdi_prob,
                random_seed=random_seed,
            )

            counterfactual_preds[channel] = cf_pred

        contribution_data = {}
        contribution_samples = {}

        for channel in channels:
            cf_pred = counterfactual_preds[channel]
            contrib = baseline_pred.y_pred_mean - cf_pred.y_pred_mean
            contribution_data[channel] = contrib

            if compute_uncertainty:
                contrib_samples = baseline_pred.y_pred_samples - cf_pred.y_pred_samples
                contribution_samples[channel] = contrib_samples

        channel_contributions = pd.DataFrame(
            contribution_data,
            index=self.panel.index,
        )

        if time_period is not None:
            contrib_masked = channel_contributions.iloc[time_mask]
        else:
            contrib_masked = channel_contributions

        total_contributions = contrib_masked.sum()

        total_effect = total_contributions.sum()
        contribution_pct = (
            (total_contributions / total_effect * 100)
            if total_effect != 0
            else total_contributions * 0
        )

        contribution_hdi_low = None
        contribution_hdi_high = None

        if compute_uncertainty:
            hdi_low_values = {}
            hdi_high_values = {}

            for channel in channels:
                samples = contribution_samples[channel]
                if time_period is not None:
                    samples = samples[:, time_mask]

                total_samples = samples.sum(axis=1)

                low, high = compute_hdi_bounds(total_samples, hdi_prob=hdi_prob, axis=0)
                hdi_low_values[channel] = low
                hdi_high_values[channel] = high

            contribution_hdi_low = pd.Series(hdi_low_values)
            contribution_hdi_high = pd.Series(hdi_high_values)

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
        """Compute marginal contributions for a given spend increase."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names
        multiplier = 1.0 + spend_increase_pct / 100.0

        time_mask = self._get_time_mask(time_period)

        baseline_pred = self.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()

        results = []

        for channel in channels:
            ch_idx = self.channel_names.index(channel)

            X_media_increased = self.X_media_raw.copy()
            X_media_increased[:, ch_idx] *= multiplier

            increased_pred = self.predict(
                X_media=X_media_increased,
                random_seed=random_seed,
            )
            increased_total = increased_pred.y_pred_mean[time_mask].sum()

            marginal_contrib = increased_total - baseline_total

            current_spend = self.X_media_raw[time_mask, ch_idx].sum()
            spend_increase = current_spend * (multiplier - 1)

            marginal_roas = (
                marginal_contrib / spend_increase if spend_increase > 0 else 0
            )

            results.append({
                "Channel": channel,
                "Current Spend": current_spend,
                f"Spend Increase ({spend_increase_pct}%)": spend_increase,
                "Marginal Contribution": marginal_contrib,
                "Marginal ROAS": marginal_roas,
            })

        return pd.DataFrame(results)

    def what_if_scenario(
        self,
        spend_changes: dict[str, float],
        time_period: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> dict:
        """Run a what-if scenario with custom spend changes."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        time_mask = self._get_time_mask(time_period)

        baseline_pred = self.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()

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

        scenario_pred = self.predict(
            X_media=X_media_scenario,
            random_seed=random_seed,
        )
        scenario_total = scenario_pred.y_pred_mean[time_mask].sum()

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

    def save(
        self,
        path: str | Path,
        save_trace: bool = True,
        compress: bool = True,
    ) -> None:
        """Save the fitted model to disk."""
        from ..serialization import MMMSerializer

        MMMSerializer.save(self, path, save_trace=save_trace, compress=compress)

    @classmethod
    def load(
        cls,
        path: str | Path,
        panel: PanelDataset,
        rebuild_model: bool = True,
    ) -> BayesianMMM:
        """Load a saved model from disk."""
        from ..serialization import MMMSerializer

        return MMMSerializer.load(path, panel, rebuild_model=rebuild_model)

    def save_trace_only(self, path: str | Path) -> None:
        """Save only the fitted trace to a file."""
        if self._trace is None:
            raise ValueError("No trace to save. Fit the model first.")

        from ..serialization import MMMSerializer

        MMMSerializer.save_trace_only(self._trace, path)

    def load_trace_only(self, path: str | Path) -> None:
        """Load a trace from a file into the current model."""
        from ..serialization import MMMSerializer

        self._trace = MMMSerializer.load_trace_only(path)


__all__ = ["BayesianMMM"]

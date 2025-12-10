"""
Bayesian MMM model class - Robust Implementation v3.

Key design principles for stability:
1. Standardize all data (y, X_media, X_controls)
2. Use simple, well-understood priors
3. Avoid complex transformations in the graph
4. Pre-compute adstock outside the model
5. Use logistic saturation (numerically stable)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

class TrendType(str, Enum):
    """Available trend specifications."""
    NONE = "none"
    LINEAR = "linear"


@dataclass
class TrendConfig:
    """Configuration for trend component."""
    type: TrendType = TrendType.LINEAR


# =============================================================================
# Helper functions
# =============================================================================

def create_fourier_features(t: np.ndarray, period: float, order: int) -> np.ndarray:
    """Create Fourier features for seasonality."""
    features = []
    for i in range(1, order + 1):
        features.append(np.sin(2 * np.pi * i * t / period))
        features.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(features) if features else np.zeros((len(t), 0))


def geometric_adstock_np(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply geometric adstock transformation.
    
    y[t] = x[t] + alpha * y[t-1]
    
    Normalized so that sum of weights = 1/(1-alpha)
    """
    n = len(x)
    result = np.zeros(n)
    result[0] = x[0]
    for t in range(1, n):
        result[t] = x[t] + alpha * result[t - 1]
    return result


def logistic_saturation_np(x: np.ndarray, lam: float) -> np.ndarray:
    """Logistic saturation: 1 - exp(-lam * x)"""
    return 1.0 - np.exp(-lam * np.clip(x, 0, None))


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


# =============================================================================
# Main Model Class
# =============================================================================

class BayesianMMM:
    """
    Bayesian Marketing Mix Model - Robust Implementation.
    
    This implementation prioritizes numerical stability:
    - All data is standardized before modeling
    - Adstock is pre-computed at fixed alpha values
    - Logistic saturation is used (more stable than Hill)
    - Priors are carefully scaled for standardized data
    
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
        self.n_controls = self.X_controls_raw.shape[1] if self.X_controls_raw is not None else 0
        
        self.channel_names = list(self.panel.coords.channels)
        self.control_names = list(self.panel.coords.controls) if self.n_controls > 0 else []
        
        # === Standardize target ===
        self.y_mean = float(self.y_raw.mean())
        self.y_std = float(self.y_raw.std()) + 1e-8
        self.y = (self.y_raw - self.y_mean) / self.y_std
        
        # === Pre-compute adstocked media at fixed alphas ===
        # This avoids putting adstock in the PyMC graph
        self.X_media_adstocked = {}
        for alpha in self.adstock_alphas:
            adstocked = np.zeros_like(self.X_media_raw)
            for c in range(self.n_channels):
                adstocked[:, c] = geometric_adstock_np(self.X_media_raw[:, c], alpha)
            # Scale to [0, 1] range
            adstocked_max = adstocked.max(axis=0, keepdims=True) + 1e-8
            self.X_media_adstocked[alpha] = adstocked / adstocked_max
        
        # === Standardize controls ===
        if self.X_controls_raw is not None:
            self.control_mean = self.X_controls_raw.mean(axis=0)
            self.control_std = self.X_controls_raw.std(axis=0) + 1e-8
            self.X_controls = (self.X_controls_raw - self.control_mean) / self.control_std
        else:
            self.X_controls = None
        
        # === Geo/product info ===
        self.has_geo = self.panel.coords.has_geo
        self.has_product = self.panel.coords.has_product
        self.n_geos = self.panel.coords.n_geos
        self.n_products = self.panel.coords.n_products
        
        if self.has_geo:
            self.geo_names = list(self.panel.coords.geographies)
            self.geo_idx = self._get_group_indices('geography')
        
        if self.has_product:
            self.product_names = list(self.panel.coords.products)
            self.product_idx = self._get_group_indices('product')
        
        # === Time index ===
        self.n_periods = self.panel.coords.n_periods
        self.time_idx = self._get_time_index()
        self.t_scaled = self.time_idx / max(self.n_periods - 1, 1)  # [0, 1]
        
        # === Seasonality features ===
        self._prepare_seasonality()
        
        # === Media hierarchy ===
        self.media_groups = self.mff_config.get_hierarchical_media_groups()
        self.has_media_hierarchy = len(self.media_groups) > 0
    
    def _get_group_indices(self, level_name: str) -> np.ndarray:
        """Get group indices for a hierarchical level."""
        cols = self.mff_config.columns
        col_name = getattr(cols, level_name)
        
        if isinstance(self.panel.index, pd.MultiIndex):
            values = self.panel.index.get_level_values(col_name)
            if level_name == 'geography':
                categories = self.geo_names
            else:
                categories = self.product_names
            return pd.Categorical(values, categories=categories).codes
        return np.zeros(self.n_obs, dtype=int)
    
    def _get_time_index(self) -> np.ndarray:
        """Get time index for each observation."""
        cols = self.mff_config.columns
        
        if isinstance(self.panel.index, pd.MultiIndex):
            period_values = self.panel.index.get_level_values(cols.period)
            periods_unique = list(self.panel.coords.periods)
            return pd.Categorical(period_values, categories=periods_unique).codes
        return np.arange(self.n_obs)
    
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
        
        return coords
    
    def _build_model(self) -> pm.Model:
        """Build the PyMC model."""
        coords = self._build_coords()
        
        with pm.Model(coords=coords) as model:
            # =================================================================
            # INTERCEPT
            # =================================================================
            # Since y is standardized (mean=0, std=1), intercept should be ~0
            intercept = pm.Normal("intercept", mu=0, sigma=0.5)
            
            # =================================================================
            # TREND
            # =================================================================
            if self.trend_config.type == TrendType.LINEAR:
                trend_slope = pm.Normal("trend_slope", mu=0, sigma=0.5)
                trend = trend_slope * self.t_scaled[self.time_idx]
            else:
                trend = np.zeros(self.n_obs)
            
            # =================================================================
            # SEASONALITY
            # =================================================================
            seasonality = np.zeros(self.n_obs)
            for name, features in self.seasonality_features.items():
                n_features = features.shape[1]
                season_coef = pm.Normal(
                    f"season_{name}",
                    mu=0,
                    sigma=0.3,
                    shape=n_features,
                )
                season_effect = pt.dot(features, season_coef)
                seasonality = seasonality + season_effect[self.time_idx]
            
            # =================================================================
            # GEO EFFECTS (if applicable)
            # =================================================================
            if self.has_geo and self.hierarchical_config.pool_across_geo:
                geo_sigma = pm.HalfNormal("geo_sigma", sigma=0.3)
                geo_offset = pm.Normal("geo_offset", mu=0, sigma=1, shape=self.n_geos)
                geo_effect = geo_sigma * geo_offset
                geo_contribution = geo_effect[self.geo_idx]
            else:
                geo_contribution = np.zeros(self.n_obs)
            
            # =================================================================
            # PRODUCT EFFECTS (if applicable)
            # =================================================================
            if self.has_product and self.hierarchical_config.pool_across_product:
                product_sigma = pm.HalfNormal("product_sigma", sigma=0.3)
                product_offset = pm.Normal("product_offset", mu=0, sigma=1, shape=self.n_products)
                product_effect = product_sigma * product_offset
                product_contribution = product_effect[self.product_idx]
            else:
                product_contribution = np.zeros(self.n_obs)
            
            # =================================================================
            # MEDIA EFFECTS
            # =================================================================
            # Strategy: interpolate between pre-computed adstock levels,
            # apply logistic saturation, multiply by coefficient
            
            channel_contribs = []
            
            for c, channel_name in enumerate(self.channel_names):
                # Get pre-computed adstock at low and high alpha
                x_low = self.X_media_adstocked[self.adstock_alphas[0]][:, c]
                x_high = self.X_media_adstocked[self.adstock_alphas[-1]][:, c]
                
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
            pm.Deterministic("channel_contributions", media_matrix, dims=("obs", "channel"))
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
                control_contribution = pt.dot(self.X_controls, beta_controls)
            else:
                control_contribution = np.zeros(self.n_obs)
            
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
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
        
        return model
    
    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model
    
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
        
        with self.model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                init="adapt_diag",
                **kwargs,
            )
        
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
    
    def sample_prior_predictive(self, samples: int = 500) -> az.InferenceData:
        """Sample from prior predictive distribution."""
        with self.model:
            return pm.sample_prior_predictive(samples=samples)
    
    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return az.summary(self._trace, var_names=var_names)
"""
Data extractors for MMM report generation.

Provides adapters to extract report data from various MMM model types:
- BayesianMMM (core framework)
- NestedMMM, MultivariateMMM, CombinedMMM (extensions)
- PyMC-Marketing MMM class

Each extractor converts model-specific data structures into a unified
MMMDataBundle that the report generator can consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd
from loguru import logger
try:
    import arviz as az
except ImportError:
    az = None
from .helpers import _safe_get_column, _safe_to_numpy

@dataclass
class MMMDataBundle:
    """
    Unified data container for MMM report generation.
    
    All fields are optional - sections will gracefully skip if data is missing.
    """
    
    # Time index
    dates: np.ndarray | pd.DatetimeIndex | list | None = None

    
    # Metadata
    geo_names: list[str] | None = None
    
    # Geo-level observed values: {geo_name: ndarray of shape (n_periods,)}
    actual_by_geo: dict[str, np.ndarray] | None = None
    
    # Geo-level predictions: {geo_name: {"mean": ndarray, "lower": ndarray, "upper": ndarray}}
    predicted_by_geo: dict[str, dict[str, np.ndarray]] | None = None
    
    # Geo-level fit statistics: {geo_name: {"r2": float, "rmse": float, "mape": float}}
    fit_statistics_by_geo: dict[str, dict[str, float]] | None = None

    # Actual vs predicted
    actual: np.ndarray | None = None
    predicted: dict[str, np.ndarray] | None = None  # {"mean", "lower", "upper"}
    
    # Fit statistics
    fit_statistics: dict[str, float] | None = None  # {"r2", "rmse", "mae", "mape"}
    
    # Summary metrics
    total_revenue: float | None = None
    marketing_attributed_revenue: dict[str, float] | None = None  # {"mean", "lower", "upper"}
    blended_roi: dict[str, float] | None = None  # {"mean", "lower", "upper"}
    marketing_contribution_pct: dict[str, float] | None = None  # {"mean", "lower", "upper"}
    
    # Channel-level ROI
    channel_roi: dict[str, dict[str, float]] | None = None  # {channel: {"mean", "lower", "upper"}}
    channel_spend: dict[str, float] | None = None
    channel_contribution: dict[str, dict[str, float]] | None = None
    
    # Decomposition
    component_totals: dict[str, float] | None = None  # {component: total_contribution}
    component_time_series: dict[str, np.ndarray] | None = None  # {component: time_series}
    
    # Saturation and adstock
    saturation_curves: dict[str, dict[str, np.ndarray]] | None = None  # {channel: {"spend", "response"}}
    adstock_curves: dict[str, np.ndarray] | None = None  # {channel: lag_weights}
    current_spend: dict[str, float] | None = None
    
    # Sensitivity analysis
    sensitivity_results: dict[str, Any] | None = None
    
    # Model specification info
    model_specification: dict[str, Any] | None = None
    
    # MCMC diagnostics
    diagnostics: dict[str, Any] | None = None  # {"divergences", "rhat_max", "ess_bulk_min"}
    trace_data: dict[str, np.ndarray] | None = None
    trace_parameters: list[str] | None = None
    
    # Prior/posterior comparison
    prior_samples: dict[str, np.ndarray] | None = None
    posterior_samples: dict[str, np.ndarray] | None = None
    
    # Channel names
    channel_names: list[str] | None = None
    
    # Extended model data (nested, multivariate)
    mediator_effects: dict[str, Any] | None = None
    cross_effects: dict[str, Any] | None = None
    outcome_correlations: np.ndarray | None = None
    
    # Geographic data
    geo_names: list[str] | None = None
    geo_performance: dict[str, dict[str, Any]] | None = None  # {geo: {metric: value}}
    geo_roi: dict[str, dict[str, dict[str, float]]] | None = None  # {geo: {channel: {"mean", "lower", "upper"}}}
    geo_contribution: dict[str, dict[str, float]] | None = None  # {geo: {component: contribution}}
    
    # Mediator pathway data (nested models)
    mediator_names: list[str] | None = None
    mediator_pathways: dict[str, dict[str, Any]] | None = None  # {channel: {mediator: {"direct", "indirect", "total"}}}
    mediator_time_series: dict[str, np.ndarray] | None = None  # {mediator: values}
    total_indirect_effect: dict[str, float] | None = None  # {"mean", "lower", "upper"}
    
    # Cannibalization / cross-product effects
    product_names: list[str] | None = None
    cannibalization_matrix: dict[str, dict[str, dict[str, float]]] | None = None  # {source: {target: {"mean", "lower", "upper"}}}
    net_product_effects: dict[str, dict[str, float]] | None = None  # {product: {"direct", "cannibalization", "net"}}
    component_time_series_by_geo: dict[str, dict[str, np.ndarray]] | None = None

    # Geo-level component totals: {geo_name: {component_name: float}}
    component_totals_by_geo: dict[str, dict[str, float]] | None = None

    # Product-level observed values: {product_name: ndarray of shape (n_periods,)}
    actual_by_product: dict[str, np.ndarray] | None = None

    # Product-level predictions: {product_name: {"mean": ndarray, "lower": ndarray, "upper": ndarray}}
    predicted_by_product: dict[str, dict[str, np.ndarray]] | None = None

    # Product-level fit statistics: {product_name: {"r2": float, "rmse": float, "mape": float}}
    fit_statistics_by_product: dict[str, dict[str, float]] | None = None

    # Product-level component time series: {product_name: {component_name: ndarray}}
    component_time_series_by_product: dict[str, dict[str, np.ndarray]] | None = None

    # Product-level component totals: {product_name: {component_name: float}}
    component_totals_by_product: dict[str, dict[str, float]] | None = None

    @property
    def has_geo_data(self) -> bool:
        """Check if geo-level data is available."""
        return (
            self.geo_names is not None 
            and len(self.geo_names) > 1
            and self.actual_by_geo is not None
        )
    
    @property
    def has_geo_decomposition(self) -> bool:
        """Check if geo-level decomposition is available."""
        return (
            self.geo_names is not None
            and len(self.geo_names) > 1
            and self.component_time_series_by_geo is not None
        )

    @property
    def has_product_data(self) -> bool:
        """Check if product-level data is available."""
        return (
            self.product_names is not None
            and len(self.product_names) > 1
            and self.actual_by_product is not None
        )

    @property
    def has_product_decomposition(self) -> bool:
        """Check if product-level decomposition is available."""
        return (
            self.product_names is not None
            and len(self.product_names) > 1
            and self.component_time_series_by_product is not None
        )

@runtime_checkable
class HasTrace(Protocol):
    """Protocol for objects with ArviZ InferenceData trace."""
    @property
    def trace(self) -> Any: ...


@runtime_checkable
class HasModel(Protocol):
    """Protocol for objects with PyMC model."""
    @property
    def model(self) -> Any: ...


class DataExtractor(ABC):
    """
    Base class for model data extractors.

    All concrete extractors should inherit from this class and implement
    the `extract()` method. Shared utilities for HDI computation, diagnostics
    extraction, and fit statistics are provided.

    Attributes
    ----------
    ci_prob : float
        Credible interval probability (default 0.8).

    Examples
    --------
    >>> class MyExtractor(DataExtractor):
    ...     def __init__(self, model, ci_prob=0.8):
    ...         self.model = model
    ...         self._ci_prob = ci_prob
    ...
    ...     @property
    ...     def ci_prob(self):
    ...         return self._ci_prob
    ...
    ...     def extract(self):
    ...         bundle = MMMDataBundle()
    ...         # ... extract data
    ...         return bundle
    """

    @property
    def ci_prob(self) -> float:
        """Credible interval probability. Override in subclass."""
        return getattr(self, '_ci_prob', 0.8)

    @abstractmethod
    def extract(self) -> MMMDataBundle:
        """Extract data from model into unified bundle."""
        pass

    def _compute_hdi(
        self,
        samples: np.ndarray,
        prob: float | None = None,
    ) -> tuple[float, float]:
        """
        Compute highest density interval from samples.

        Parameters
        ----------
        samples : np.ndarray
            MCMC samples.
        prob : float, optional
            HDI probability. If None, uses self.ci_prob.

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        if prob is None:
            prob = self.ci_prob

        if az is not None:
            hdi = az.hdi(samples, hdi_prob=prob)
            return float(hdi[0]), float(hdi[1])
        else:
            # Fallback to percentile-based interval
            alpha = (1 - prob) / 2
            return float(np.percentile(samples, alpha * 100)), float(np.percentile(samples, (1 - alpha) * 100))

    def _compute_percentile_bounds(
        self,
        samples: np.ndarray,
        prob: float | None = None,
        axis: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute percentile-based credible interval bounds.

        Parameters
        ----------
        samples : np.ndarray
            Sample array.
        prob : float, optional
            Credible interval probability. If None, uses self.ci_prob.
        axis : int
            Axis along which to compute percentiles.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower, upper) bound arrays.
        """
        if prob is None:
            prob = self.ci_prob

        alpha = (1 - prob) / 2
        lower = np.percentile(samples, alpha * 100, axis=axis)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=axis)
        return lower, upper

    def _compute_fit_statistics(
        self,
        actual: np.ndarray | None,
        predicted: dict[str, np.ndarray] | None,
    ) -> dict[str, float] | None:
        """
        Compute model fit statistics: R², RMSE, MAE, MAPE.

        Parameters
        ----------
        actual : np.ndarray or None
            Actual observed values.
        predicted : dict or None
            Predictions dict with "mean" key.

        Returns
        -------
        dict[str, float] or None
            Dictionary with "r2", "rmse", "mae", "mape" keys.
        """
        if actual is None or predicted is None:
            return None

        y_true = actual
        y_pred = predicted.get("mean")
        if y_pred is None:
            return None

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        else:
            mape = np.nan

        return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae), "mape": float(mape)}

    def _extract_diagnostics(self, trace: Any) -> dict[str, Any]:
        """Extract MCMC diagnostics from ArviZ trace."""
        if az is None or trace is None:
            return {}

        try:
            # Get summary stats
            summary = az.summary(trace)

            diagnostics = {
                "rhat_max": float(summary["r_hat"].max()),
                "ess_bulk_min": float(summary["ess_bulk"].min()),
                "ess_tail_min": float(summary["ess_tail"].min()),
            }

            # Check for divergences in sample stats
            if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
                divergences = trace.sample_stats["diverging"].values.sum()
                diagnostics["divergences"] = int(divergences)
            else:
                diagnostics["divergences"] = 0

            return diagnostics
        except Exception:
            return {}


class AggregationMixin:
    """
    Mixin providing data aggregation utilities for extractors.

    Provides methods for aggregating data by period, geography, and product
    while properly propagating uncertainty through sample aggregation.
    """

    def _aggregate_by_period_simple(
        self,
        values: np.ndarray,
        periods: list,
        unique_periods: list,
    ) -> np.ndarray:
        """
        Aggregate values by period using simple summation.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate, shape (n_obs,).
        periods : list
            Period label for each observation.
        unique_periods : list
            Unique periods in order.

        Returns
        -------
        np.ndarray
            Aggregated values, shape (n_periods,).
        """
        period_to_idx = {p: i for i, p in enumerate(unique_periods)}
        n_periods = len(unique_periods)

        result = np.zeros(n_periods)
        for i, (val, period) in enumerate(zip(values, periods)):
            if period in period_to_idx:
                result[period_to_idx[period]] += val

        return result

    def _aggregate_samples_by_period(
        self,
        samples: np.ndarray,
        periods: list,
        unique_periods: list,
        ci_prob: float = 0.8,
    ) -> dict[str, np.ndarray] | None:
        """
        Aggregate samples by period while preserving uncertainty.

        This method properly propagates uncertainty by aggregating
        samples first, then computing statistics on aggregated values.

        Parameters
        ----------
        samples : np.ndarray
            Sample array of shape (n_samples, n_obs).
        periods : list
            Period label for each observation.
        unique_periods : list
            Unique periods in order.
        ci_prob : float
            Credible interval probability.

        Returns
        -------
        dict[str, np.ndarray] or None
            Dictionary with "mean", "lower", "upper" arrays.
        """
        if samples is None or len(periods) == 0:
            return None

        try:
            period_to_idx = {p: i for i, p in enumerate(unique_periods)}
            obs_period_idx = np.array([period_to_idx.get(p, -1) for p in periods])

            n_samples = samples.shape[0]
            n_periods = len(unique_periods)

            # Aggregate samples by period
            samples_agg = np.zeros((n_samples, n_periods))

            for t in range(n_periods):
                mask = (obs_period_idx == t)
                if mask.any():
                    samples_agg[:, t] = samples[:, mask].sum(axis=1)

            # Compute statistics
            alpha = (1 - ci_prob) / 2

            return {
                "mean": samples_agg.mean(axis=0),
                "lower": np.percentile(samples_agg, alpha * 100, axis=0),
                "upper": np.percentile(samples_agg, (1 - alpha) * 100, axis=0),
            }

        except Exception:
            return None

    def _aggregate_by_group(
        self,
        values: np.ndarray,
        group_idx: np.ndarray,
        n_groups: int,
    ) -> np.ndarray:
        """
        Aggregate values by group index.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate.
        group_idx : np.ndarray
            Group index for each value.
        n_groups : int
            Number of groups.

        Returns
        -------
        np.ndarray
            Aggregated values per group.
        """
        result = np.zeros(n_groups)
        for i, val in enumerate(values):
            if 0 <= group_idx[i] < n_groups:
                result[group_idx[i]] += val
        return result

class BayesianMMMExtractorGeoMixin:
    """
    Mixin class with geo-level extraction methods.
    
    Add these methods to the existing BayesianMMMExtractor class.
    """
    
    def _extract_geo_level_fit_data(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract geo-level model fit data.
        
        Populates:
        - bundle.actual_by_geo
        - bundle.predicted_by_geo
        - bundle.fit_statistics_by_geo
        """
        # Check if we have geo-level data
        if not hasattr(self, 'panel') or self.panel is None:
            return bundle
        
        panel = self.panel
        mmm = self.mmm
        
        # Get geo info
        geo_names = self._get_geo_names()
        if geo_names is None or len(geo_names) <= 1:
            return bundle
        
        bundle.geo_names = geo_names
        
        try:
            # Get indices
            geo_idx = self._get_geo_indices()
            if geo_idx is None:
                return bundle
            
            # Get observed values (original scale)
            y_obs = self._get_actual_original_scale()
            if y_obs is None:
                return bundle
            
            # Get predictions
            y_pred_mean, y_pred_lower, y_pred_upper = self._get_predictions_original_scale()
            if y_pred_mean is None:
                return bundle
            
            # Get period info
            n_periods = mmm.n_periods if hasattr(mmm, 'n_periods') else len(bundle.dates)
            n_geos = len(geo_names)
            
            # Initialize geo-level storage
            bundle.actual_by_geo = {}
            bundle.predicted_by_geo = {}
            bundle.fit_statistics_by_geo = {}
            
            # Aggregate by geo (sum over products if applicable)
            for g_idx, geo in enumerate(geo_names):
                # Get mask for this geo
                geo_mask = (geo_idx == g_idx)
                
                # Aggregate observed values for this geo over time
                y_obs_geo = self._aggregate_by_period(y_obs[geo_mask], n_periods, geo_mask)
                y_pred_mean_geo = self._aggregate_by_period(y_pred_mean[geo_mask], n_periods, geo_mask)
                y_pred_lower_geo = self._aggregate_by_period(y_pred_lower[geo_mask], n_periods, geo_mask)
                y_pred_upper_geo = self._aggregate_by_period(y_pred_upper[geo_mask], n_periods, geo_mask)
                
                bundle.actual_by_geo[geo] = y_obs_geo
                bundle.predicted_by_geo[geo] = {
                    "mean": y_pred_mean_geo,
                    "lower": y_pred_lower_geo,
                    "upper": y_pred_upper_geo,
                }
                
                # Compute fit statistics for this geo
                bundle.fit_statistics_by_geo[geo] = self._compute_fit_statistics(
                    y_obs_geo, 
                    {"mean": y_pred_mean_geo, "lower": y_pred_lower_geo, "upper": y_pred_upper_geo}
                )
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract geo-level fit data: {e}")
        
        return bundle
    
    def _extract_geo_level_decomposition(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract geo-level decomposition data.
        
        Populates:
        - bundle.component_time_series_by_geo
        - bundle.component_totals_by_geo
        """
        if bundle.geo_names is None or len(bundle.geo_names) <= 1:
            return bundle
        
        try:
            mmm = self.mmm
            panel = self.panel
            
            geo_names = bundle.geo_names
            geo_idx = self._get_geo_indices()
            n_periods = mmm.n_periods if hasattr(mmm, 'n_periods') else len(bundle.dates)
            
            # Get decomposition components at observation level
            components = self._get_decomposition_components_obs_level()
            if components is None:
                return bundle
            
            bundle.component_time_series_by_geo = {}
            bundle.component_totals_by_geo = {}
            
            for g_idx, geo in enumerate(geo_names):
                geo_mask = (geo_idx == g_idx)
                
                bundle.component_time_series_by_geo[geo] = {}
                bundle.component_totals_by_geo[geo] = {}
                
                for comp_name, comp_values in components.items():
                    # Aggregate this component for this geo over time
                    comp_geo = self._aggregate_by_period(
                        comp_values[geo_mask], n_periods, geo_mask
                    )
                    bundle.component_time_series_by_geo[geo][comp_name] = comp_geo
                    bundle.component_totals_by_geo[geo][comp_name] = float(comp_geo.sum())
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract geo-level decomposition: {e}")
        
        return bundle
    
    def _get_geo_names(self) -> list[str] | None:
        """Get geography names from panel or model."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'geographies'):
            geographies = self.panel.coords.geographies
            if geographies is not None:
                return list(geographies)
        if hasattr(self.mmm, 'geo_names'):
            geo_names = self.mmm.geo_names
            if geo_names is not None:
                return list(geo_names)
        return None
    
    def _get_geo_indices(self) -> np.ndarray | None:
        """Get geo index for each observation."""
        if hasattr(self.mmm, 'geo_idx') and self.mmm.geo_idx is not None:
            return np.array(self.mmm.geo_idx)
        if hasattr(self.panel, 'geo_idx') and self.panel.geo_idx is not None:
            return np.array(self.panel.geo_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            geo_names = self._get_geo_names()
            if geo_names is not None and len(geo_names) > 1:
                # Try to get from y.index if it's a MultiIndex
                if hasattr(self.panel, 'y') and hasattr(self.panel.y, 'index'):
                    idx = self.panel.y.index
                    if hasattr(idx, 'get_level_values'):
                        # Try both cases for geography level name
                        for level_name in ['geography', 'Geography']:
                            try:
                                geo_values = idx.get_level_values(level_name)
                                # Convert geo names to indices
                                geo_to_idx = {g: i for i, g in enumerate(geo_names)}
                                return np.array([geo_to_idx.get(str(g), 0) for g in geo_values])
                            except KeyError:
                                continue
        return None

    def _get_product_names(self) -> list[str] | None:
        """Get product names from panel or model."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'products'):
            products = self.panel.coords.products
            if products is not None:
                return list(products)
        if hasattr(self.mmm, 'product_names'):
            return list(self.mmm.product_names)
        return None

    def _get_product_indices(self) -> np.ndarray | None:
        """Get product index for each observation."""
        if hasattr(self.mmm, 'product_idx'):
            return np.array(self.mmm.product_idx)
        if hasattr(self.panel, 'product_idx'):
            return np.array(self.panel.product_idx)
        return None

    def _get_time_indices(self) -> np.ndarray | None:
        """Get time index for each observation."""
        if hasattr(self.mmm, 'time_idx') and self.mmm.time_idx is not None:
            return np.array(self.mmm.time_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            unique_periods = self._get_unique_periods()
            if unique_periods is not None and len(unique_periods) > 0:
                if hasattr(self.panel, 'y') and hasattr(self.panel.y, 'index'):
                    idx = self.panel.y.index
                    if hasattr(idx, 'get_level_values'):
                        # Try both cases for period level name
                        for level_name in ['period', 'Period']:
                            try:
                                period_values = idx.get_level_values(level_name)
                                # Convert periods to indices
                                period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                                return np.array([period_to_idx.get(str(p), 0) for p in period_values])
                            except KeyError:
                                continue
                        # Try first level if period not found by name
                        try:
                            period_values = idx.get_level_values(0)
                            period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                            return np.array([period_to_idx.get(str(p), 0) for p in period_values])
                        except Exception:
                            pass
                    else:
                        # Simple index - assume it's period only
                        period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                        return np.array([period_to_idx.get(str(p), 0) for p in idx])

        # Fallback: construct from panel coords
        if self.panel is not None and hasattr(self.panel, 'coords'):
            n_obs = len(self.panel.y)
            n_periods = self.panel.coords.n_periods
            n_geos = self.panel.coords.n_geos
            n_products = self.panel.coords.n_products
            # Assumes data ordered as periods innermost
            if n_obs == n_periods * n_geos * n_products:
                return np.tile(np.arange(n_periods), n_geos * n_products)
        return None

    def _get_actual_original_scale(self) -> np.ndarray | None:
        """Get observed values in original scale."""
        if self.panel is None:
            return None
        
        y_standardized = self.panel.y.values.flatten()
        y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
        y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
        
        return y_standardized * y_std + y_mean
    
    def _get_predictions_original_scale(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Get predictions in original scale with uncertainty."""
        trace = getattr(self.mmm, '_trace', None)
        if trace is None:
            return None, None, None
        
        try:
            if hasattr(trace, 'posterior_predictive') and 'y' in trace.posterior_predictive:
                y_samples = trace.posterior_predictive['y'].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            elif hasattr(trace, 'posterior') and 'mu' in trace.posterior:
                y_samples = trace.posterior['mu'].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            else:
                return None, None, None
            
            # Transform to original scale
            y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
            y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
            
            y_samples_orig = y_samples * y_std + y_mean
            
            y_pred_mean = y_samples_orig.mean(axis=0)
            alpha = (1 - self.ci_prob) / 2
            y_pred_lower = np.percentile(y_samples_orig, alpha * 100, axis=0)
            y_pred_upper = np.percentile(y_samples_orig, (1 - alpha) * 100, axis=0)
            
            return y_pred_mean, y_pred_lower, y_pred_upper
            
        except Exception:
            return None, None, None
    
    def _aggregate_by_period(
        self,
        values: np.ndarray,
        n_periods: int,
        geo_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate values by period for a specific geo.
        
        If there are multiple products, this sums over products within each period.
        """
        # Simple case: one observation per period per geo
        n_obs = len(values)
        if n_obs == n_periods:
            return values
        
        # Multiple products case: need to reshape and sum
        # Assumes data is ordered: periods are innermost, then geos, then products
        n_products = n_obs // n_periods
        if n_products * n_periods == n_obs:
            return values.reshape(n_products, n_periods).sum(axis=0)
        
        # Fallback: return as-is (may need custom handling)
        return values

    def _aggregate_by_period_with_indices(
        self,
        values: np.ndarray,
        time_idx: np.ndarray,
        dim_mask: np.ndarray,
        n_periods: int,
    ) -> np.ndarray:
        """
        Aggregate values by period using explicit time indices.

        This is a more robust aggregation method that uses the time_idx array
        to properly group observations by period, regardless of data ordering.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate, shape (n_obs,)
        time_idx : np.ndarray
            Time period index for each observation, shape (n_obs,)
        dim_mask : np.ndarray
            Boolean mask for filtering (e.g., specific geo or product), shape (n_obs,)
        n_periods : int
            Number of unique time periods

        Returns
        -------
        np.ndarray
            Aggregated values by period, shape (n_periods,)
        """
        result = np.zeros(n_periods)
        filtered_values = values[dim_mask]
        filtered_time_idx = time_idx[dim_mask]

        for t in range(n_periods):
            time_mask = (filtered_time_idx == t)
            if time_mask.any():
                result[t] = filtered_values[time_mask].sum()

        return result

    def _get_decomposition_components_obs_level(self) -> dict[str, np.ndarray] | None:
        """
        Get decomposition components at observation level (not yet aggregated).
        
        Returns dict mapping component name to array of shape (n_obs,).
        """
        trace = getattr(self.mmm, '_trace', None)
        if trace is None:
            return None
        
        components = {}
        posterior = trace.posterior
        
        y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
        y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
        n_obs = len(self.panel.y.values.flatten())
        
        try:
            # Intercept/Baseline
            if 'intercept' in posterior:
                intercept = float(posterior['intercept'].values.mean())
                # Baseline in original scale includes both the intercept contribution
                # and the mean offset (y_mean) to ensure decomposition sums to predictions
                # Per-observation baseline = intercept * y_std + y_mean
                baseline_per_obs = intercept * y_std + y_mean
                components['Baseline'] = np.full(n_obs, baseline_per_obs)
            
            # Trend
            if 'trend' in posterior:
                trend_samples = posterior['trend'].values
                trend_mean = trend_samples.mean(axis=(0, 1))
                if len(trend_mean) == n_obs:
                    components['Trend'] = trend_mean * y_std
            
            # Seasonality
            if 'seasonality' in posterior:
                seas_samples = posterior['seasonality'].values
                seas_mean = seas_samples.mean(axis=(0, 1))
                if len(seas_mean) == n_obs:
                    components['Seasonality'] = seas_mean * y_std
            
            # Media channels
            channel_names = self._get_channel_names()

            # First try the channel_contributions array (most common for geo models)
            if 'channel_contributions' in posterior:
                contrib_da = posterior['channel_contributions']
                contrib_vals = contrib_da.values  # (chains, draws, obs, channels)
                contrib_mean = contrib_vals.mean(axis=(0, 1))  # (obs, channels)

                # Check shape and extract per-channel
                if contrib_mean.ndim == 2:
                    if contrib_mean.shape[0] == n_obs and contrib_mean.shape[1] == len(channel_names):
                        # Shape is (obs, channels)
                        for i, ch in enumerate(channel_names):
                            components[ch] = contrib_mean[:, i] * y_std
                    elif contrib_mean.shape[1] == n_obs and contrib_mean.shape[0] == len(channel_names):
                        # Shape is (channels, obs)
                        for i, ch in enumerate(channel_names):
                            components[ch] = contrib_mean[i, :] * y_std
            else:
                # Fall back to individual channel contribution variables
                for ch in channel_names:
                    contrib_key = f'channel_contribution_{ch}'
                    if contrib_key in posterior:
                        contrib = posterior[contrib_key].values.mean(axis=(0, 1))
                        if len(contrib) == n_obs:
                            components[ch] = contrib * y_std

            # Controls
            if hasattr(self.mmm, 'control_names'):
                for ctrl in self.mmm.control_names:
                    ctrl_key = f'control_contribution_{ctrl}'
                    if ctrl_key in posterior:
                        ctrl_contrib = posterior[ctrl_key].values.mean(axis=(0, 1))
                        if len(ctrl_contrib) == n_obs:
                            components[f'Control: {ctrl}'] = ctrl_contrib * y_std
            
            return components if components else None
            
        except Exception:
            return None

class BayesianMMMExtractor(DataExtractor, AggregationMixin):
    """
    Extract data from mmm-framework's BayesianMMM class.

    Inherits shared utilities from DataExtractor and AggregationMixin
    for HDI computation, fit statistics, and data aggregation.

    Parameters
    ----------
    mmm : BayesianMMM
        Fitted BayesianMMM instance
    panel : PanelDataset
        Panel data used for fitting
    results : MMMResults, optional
        Fit results if available
    ci_prob : float
        Credible interval probability (default 0.8)
    """

    def __init__(
        self,
        mmm: Any,
        panel: Any | None = None,
        results: Any | None = None,
        ci_prob: float = 0.8,
    ):
        logger.debug("Initializing BayesianMMMExtractor")
        self.mmm = mmm
        self.panel = panel or getattr(mmm, "panel", None)
        self.results = results or getattr(mmm, "_results", None)
        self._ci_prob = ci_prob

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob
    
    def extract(self) -> MMMDataBundle:
        """Extract all available data from BayesianMMM."""
        bundle = MMMDataBundle()
        logger.debug("Extracting data from BayesianMMM model")
        # Extract basic info
        bundle.channel_names = self._get_channel_names()
        bundle.dates = self._get_dates()
        
        # Actual values
        bundle.actual = self._get_actual()
        
        # Extract predictions if model is fitted
        if self.results is not None or getattr(self.mmm, "_trace", None) is not None:
            logger.debug("Model appears to be fitted, extracting predictions and diagnostics")
            bundle.predicted = self._get_predictions()
            bundle.fit_statistics = self._compute_fit_statistics(bundle.actual, bundle.predicted)
            bundle.diagnostics = self._extract_diagnostics(getattr(self.mmm, "_trace", None))
            
            # ROI and contributions
            bundle.channel_roi = self._compute_channel_roi()
            bundle.component_totals = self._get_component_totals()
            bundle.component_time_series = self._get_component_time_series()
            
            # Summary metrics
            bundle.total_revenue = float(bundle.actual.sum()) if bundle.actual is not None else None
            bundle.marketing_attributed_revenue = self._compute_marketing_attribution()
            bundle.blended_roi = self._compute_blended_roi()
            bundle.marketing_contribution_pct = self._compute_marketing_contribution_pct(bundle.total_revenue)
            
            # Saturation and adstock
            bundle.saturation_curves = self._get_saturation_curves()
            bundle.adstock_curves = self._get_adstock_curves()
            bundle.current_spend = self._get_current_spend()
            
            # Trace data for diagnostics
            bundle.trace_data, bundle.trace_parameters = self._get_trace_data()
            
            # Prior/posterior
            bundle.prior_samples, bundle.posterior_samples = self._get_prior_posterior()
            bundle = self._extract_aggregated_fit_data(bundle)
            bundle = self._extract_aggregated_decomposition(bundle)

            bundle = self._extract_geo_level_fit_data(bundle)
            bundle = self._extract_geo_level_decomposition(bundle)

            # Product-level data
            bundle = self._extract_product_level_fit_data(bundle)
            bundle = self._extract_product_level_decomposition(bundle)

        # Model specification
        bundle.model_specification = self._get_model_specification()
        
        return bundle
    
    def _get_unique_periods(self) -> list | None:
        """Get unique period labels in order."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'periods'):
            periods = self.panel.coords.periods
            if hasattr(periods[0], 'strftime'):
                return [p.strftime('%Y-%m-%d') for p in periods]
            return list(periods)
        if hasattr(self, '_dates') and self._dates is not None:
            return list(self._dates)
        return None
    
    def _aggregate_predictions_with_uncertainty(
        self,
        periods: list,
        unique_periods: list,
    ) -> dict[str, np.ndarray] | None:
        """
        Aggregate predictions while properly propagating uncertainty.

        The key insight: we must sum the posterior samples first,
        THEN compute percentiles. Summing percentiles directly gives
        invalid (too wide) bounds.
        """
        import pandas as pd

        try:
            import logging

            # Method 1: Use model's predict() method (most reliable)
            y_samples_orig = None
            if hasattr(self.mmm, 'predict'):
                try:
                    logging.info("Attempting to get predictions via model.predict()")
                    pred_results = self.mmm.predict(return_original_scale=True, hdi_prob=self.ci_prob)
                    y_samples_orig = pred_results.y_pred_samples  # (n_samples, n_obs)
                    logging.info(f"predict() successful, samples shape: {y_samples_orig.shape}, mean: {y_samples_orig.mean():.2f}")
                except Exception as e:
                    logging.warning(f"predict() failed: {e}")
                    y_samples_orig = None

            # Method 2: Try to reconstruct from trace components
            if y_samples_orig is None:
                logging.info("Attempting to reconstruct predictions from trace")
                y_samples_orig = self._reconstruct_predictions_from_trace()
                if y_samples_orig is not None:
                    logging.info(f"Reconstruction successful, samples shape: {y_samples_orig.shape}, mean: {y_samples_orig.mean():.2f}")

            if y_samples_orig is None:
                logging.warning("Could not obtain prediction samples")
                return None

            n_samples, n_obs = y_samples_orig.shape

            # Create period index for aggregation
            period_to_idx = {p: i for i, p in enumerate(unique_periods)}
            obs_period_idx = np.array([period_to_idx[p] for p in periods])

            n_periods = len(unique_periods)

            # Aggregate samples by period (sum over geo/product for each sample)
            # Result: (n_samples, n_periods)
            y_samples_agg = np.zeros((n_samples, n_periods))

            for t in range(n_periods):
                mask = (obs_period_idx == t)
                if mask.any():
                    # Sum across all observations in this period (across geos/products)
                    y_samples_agg[:, t] = y_samples_orig[:, mask].sum(axis=1)

            # Now compute statistics on the aggregated samples
            y_pred_mean = y_samples_agg.mean(axis=0)
            alpha = (1 - self.ci_prob) / 2
            y_pred_lower = np.percentile(y_samples_agg, alpha * 100, axis=0)
            y_pred_upper = np.percentile(y_samples_agg, (1 - alpha) * 100, axis=0)

            return {
                "mean": y_pred_mean,
                "lower": y_pred_lower,
                "upper": y_pred_upper,
            }
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to aggregate predictions with uncertainty: {e}")
            import traceback
            logging.warning(traceback.format_exc())
            return None

    def _reconstruct_predictions_from_trace(self) -> np.ndarray | None:
        """
        Reconstruct prediction samples by summing component samples.

        This is a fallback when predict() isn't available.
        Returns shape (n_samples, n_obs) in original scale.
        """
        import logging

        trace = getattr(self.mmm, '_trace', None)
        if trace is None or not hasattr(trace, 'posterior'):
            logging.warning("No trace or posterior found")
            return None

        try:
            posterior = trace.posterior
            y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
            y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0

            logging.info(f"Reconstruction: y_std={y_std:.4f}, y_mean={y_mean:.4f}")

            # Method 1: Try to get y_obs from posterior_predictive and transform
            # (This is only populated if sample_posterior_predictive was called)
            if hasattr(trace, 'posterior_predictive') and 'y_obs' in trace.posterior_predictive:
                y_ppc = trace.posterior_predictive['y_obs'].values  # (chains, draws, obs)
                n_chains, n_draws, n_obs = y_ppc.shape
                y_samples = y_ppc.reshape(n_chains * n_draws, n_obs)
                y_samples_orig = y_samples * y_std + y_mean
                logging.info(f"Using y_obs from posterior_predictive: shape {y_samples_orig.shape}, mean={y_samples_orig.mean():.2f}")
                return y_samples_orig

            # Method 2: Reconstruct mu from component samples
            # Note: y_obs_scaled in posterior contains observed data, not predictions
            # Get shapes from a known variable
            if 'intercept' in posterior:
                intercept_samples = posterior['intercept'].values
                n_chains, n_draws = intercept_samples.shape[:2]
                logging.info(f"Intercept samples shape: {intercept_samples.shape}, mean: {intercept_samples.mean():.4f}")
            else:
                logging.warning("No intercept found in posterior")
                return None

            n_obs = len(self.panel.y) if self.panel is not None else None
            if n_obs is None:
                return None

            n_samples = n_chains * n_draws

            # Start with intercept (broadcast to all observations)
            intercept_flat = intercept_samples.reshape(n_samples, -1)
            if intercept_flat.shape[1] == 1:
                mu_samples = np.broadcast_to(intercept_flat, (n_samples, n_obs)).copy()
            else:
                mu_samples = intercept_flat

            logging.info(f"After intercept: mu mean={mu_samples.mean():.4f}")

            # Add trend if present
            if 'trend_component' in posterior:
                trend = posterior['trend_component'].values.reshape(n_samples, -1)
                if trend.shape[1] == n_obs:
                    mu_samples = mu_samples + trend
                    logging.info(f"After trend: mu mean={mu_samples.mean():.4f}")

            # Add seasonality if present
            if 'seasonality_component' in posterior:
                seas = posterior['seasonality_component'].values.reshape(n_samples, -1)
                if seas.shape[1] == n_obs:
                    mu_samples = mu_samples + seas
                    logging.info(f"After seasonality: mu mean={mu_samples.mean():.4f}")

            # Add geo effect if present (hierarchical geo model)
            if 'geo_offset' in posterior and 'geo_sigma' in posterior:
                geo_idx = self._get_geo_indices()
                if geo_idx is not None:
                    geo_sigma = posterior['geo_sigma'].values.reshape(n_samples, 1)
                    geo_offset = posterior['geo_offset'].values.reshape(n_samples, -1)
                    geo_effect = geo_sigma * geo_offset  # (n_samples, n_geos)
                    geo_contrib = geo_effect[:, geo_idx]  # (n_samples, n_obs)
                    mu_samples = mu_samples + geo_contrib
                    logging.info(f"After geo effect: mu mean={mu_samples.mean():.4f}")

            # Add product effect if present (hierarchical product model)
            if 'product_offset' in posterior and 'product_sigma' in posterior:
                product_idx = self._get_product_indices()
                if product_idx is not None:
                    product_sigma = posterior['product_sigma'].values.reshape(n_samples, 1)
                    product_offset = posterior['product_offset'].values.reshape(n_samples, -1)
                    product_effect = product_sigma * product_offset
                    product_contrib = product_effect[:, product_idx]
                    mu_samples = mu_samples + product_contrib
                    logging.info(f"After product effect: mu mean={mu_samples.mean():.4f}")

            # Add media contribution
            if 'media_total' in posterior:
                media = posterior['media_total'].values.reshape(n_samples, -1)
                if media.shape[1] == n_obs:
                    mu_samples = mu_samples + media
                    logging.info(f"After media: mu mean={mu_samples.mean():.4f}")

            # Add control contribution if present
            # First check for stored control_contributions
            if 'control_contributions' in posterior:
                ctrl = posterior['control_contributions'].values
                # Sum over control dimension if needed
                if ctrl.ndim > 3:
                    ctrl = ctrl.sum(axis=-1)  # Sum over controls
                ctrl = ctrl.reshape(n_samples, -1)
                if ctrl.shape[1] == n_obs:
                    mu_samples = mu_samples + ctrl
                    logging.info(f"After controls: mu mean={mu_samples.mean():.4f}")
            # Otherwise, compute from beta_controls and X_controls
            elif 'beta_controls' in posterior and hasattr(self.mmm, 'X_controls') and self.mmm.X_controls is not None:
                beta_samples = posterior['beta_controls'].values  # (chains, draws, n_controls)
                beta_samples = beta_samples.reshape(n_samples, -1)  # (n_samples, n_controls)
                X_controls = self.mmm.X_controls  # (n_obs, n_controls), already standardized
                # Control contribution = X_controls @ beta_controls for each sample
                ctrl = np.einsum('oc,sc->so', X_controls, beta_samples)  # (n_samples, n_obs)
                mu_samples = mu_samples + ctrl
                logging.info(f"After controls (computed from beta): mu mean={mu_samples.mean():.4f}")

            # Convert to original scale: y_original = mu_standardized * y_std + y_mean
            logging.info(f"Final mu (standardized) mean={mu_samples.mean():.4f}")
            y_samples_orig = mu_samples * y_std + y_mean
            logging.info(f"Final y (original scale) mean={y_samples_orig.mean():.4f}")

            return y_samples_orig

        except Exception as e:
            import logging
            import traceback
            logging.warning(f"Failed to reconstruct predictions: {e}")
            logging.warning(traceback.format_exc())
            return None

    def _extract_aggregated_fit_data(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract properly aggregated model fit data for multi-geo models.
        
        This ensures bundle.dates, bundle.actual, and bundle.predicted are
        aggregated to period-level (summed over geo and product).
        
        For single-geo models, this is a pass-through.
        For multi-geo models, this aggregates by period.
        
        IMPORTANT: Uncertainty bounds must be computed from aggregated samples,
        not by summing bounds directly.
        """
        import pandas as pd
        
        if not hasattr(self, 'panel') or self.panel is None:
            return bundle
        
        # Get unique periods
        unique_periods = self._get_unique_periods()
        if unique_periods is None:
            return bundle
        
        # Check if we have multi-dimensional data
        geo_idx = self._get_geo_indices()
        has_geo = geo_idx is not None and len(set(geo_idx)) > 1
        
        if not has_geo:
            # Single geo - just ensure dates are unique periods
            bundle.dates = unique_periods
            return bundle
        
        try:
            # Get period labels for each observation
            periods = self._get_period_labels_per_obs()
            if periods is None:
                return bundle
            
            # Get observed values (original scale)
            y_obs = self._get_actual_original_scale()
            if y_obs is None:
                return bundle
            
            # Aggregate observed values by period
            df_obs = pd.DataFrame({
                'period': periods,
                'actual': y_obs,
            })
            agg_obs = df_obs.groupby('period')['actual'].sum()
            agg_obs = agg_obs.reindex(unique_periods).fillna(0)
            
            # Update bundle with aggregated observed data
            bundle.dates = unique_periods
            bundle.actual = agg_obs.values
            
            # For predictions, we need to aggregate samples properly to get valid bounds
            bundle.predicted = self._aggregate_predictions_with_uncertainty(
                periods, unique_periods
            )
            
            if bundle.predicted is not None:
                # Recompute fit statistics on aggregated data
                bundle.fit_statistics = self._compute_fit_statistics(
                    bundle.actual, bundle.predicted
                )
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract aggregated fit data: {e}")
            import traceback
            logging.warning(traceback.format_exc())
        
        return bundle

    def _get_channel_names(self) -> list[str]:
        """Get channel names from model or panel."""
        if hasattr(self.mmm, "channel_names"):
            logger.debug("Retrieving channel names from model")
            return list(self.mmm.channel_names)
        if self.panel is not None and hasattr(self.panel, "channel_names"):
            logger.debug("Retrieving channel names from panel data")
            return list(self.panel.channel_names)
        logger.debug("Channel names not found")
        return []
    
    def _get_dates(self) -> np.ndarray | None:
        """Get date index."""
        if self.panel is not None:
            logger.debug("Retrieving dates from panel data")
            if hasattr(self.panel, "dates"):
                logger.debug("Using 'dates' attribute from panel")
                return np.array(self.panel.dates)
            if hasattr(self.panel, "index"):
                logger.debug("Using 'index' attribute from panel")
                return np.array(self.panel.index)
        logger.debug("Dates not found")
        return None
    
    def _get_unique_periods(self) -> list | None:
        """Get unique period labels in order."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'periods'):
            periods = self.panel.coords.periods
            if hasattr(periods[0], 'strftime'):
                return [p.strftime('%Y-%m-%d') for p in periods]
            return list(periods)
        if hasattr(self, '_dates') and self._dates is not None:
            return list(self._dates)
        return None
    
    def _get_period_labels_per_obs(self) -> list | None:
        """Get period label for each observation."""
        if hasattr(self.panel, 'time_idx') and self.panel.time_idx is not None:
            unique_periods = self._get_unique_periods()
            if unique_periods is None:
                return None
            time_idx = self.panel.time_idx
            return [unique_periods[int(t)] for t in time_idx]

        # Fallback: try to get from panel index
        if hasattr(self.panel, 'y') and hasattr(self.panel.y, 'index'):
            idx = self.panel.y.index
            if hasattr(idx, 'get_level_values'):
                # MultiIndex - get period level (try both cases)
                for level_name in ['period', 'Period']:
                    try:
                        periods = idx.get_level_values(level_name)
                        if hasattr(periods[0], 'strftime'):
                            return [p.strftime('%Y-%m-%d') for p in periods]
                        return list(periods)
                    except KeyError:
                        continue
                # Try first level if period not found by name
                try:
                    periods = idx.get_level_values(0)
                    if hasattr(periods[0], 'strftime'):
                        return [p.strftime('%Y-%m-%d') for p in periods]
                    return list(periods)
                except Exception:
                    pass
            else:
                # Simple index (period only)
                if hasattr(idx[0], 'strftime'):
                    return [p.strftime('%Y-%m-%d') for p in idx]
                return list(idx)

        return None
    
    def _get_actual(self) -> np.ndarray | None:
        """Get actual KPI values."""
        if hasattr(self.mmm, "y"):
            logger.debug("Retrieving actual values from model")
            y = self.mmm.y
            # Unstandardize if needed
            if hasattr(self.mmm, "y_mean") and hasattr(self.mmm, "y_std"):
                logger.debug("Unstandardizing actual values")
                return y * self.mmm.y_std + self.mmm.y_mean
            return np.array(y)
        if self.panel is not None and hasattr(self.panel, "y"):
            logger.debug("Retrieving actual values from panel data")
            return np.array(self.panel.y)
        logger.debug("Actual values not found")
        return None
    
    def _get_predictions(self) -> dict[str, np.ndarray] | None:
        """Get posterior predictive mean and CI."""
        try:
            if hasattr(self.mmm, "predict"):
                logger.debug("Generating predictions using model's predict method")
                pred = self.mmm.predict()
                return {
                    "mean": np.array(pred.y_pred_mean),
                    "lower": np.array(pred.y_pred_hdi_low),
                    "upper": np.array(pred.y_pred_hdi_high),
                }
            
            # Try to get from trace
            trace = getattr(self.mmm, "_trace", None)
            if trace is not None and hasattr(trace, "posterior_predictive"):
                logger.debug("Extracting predictions from model trace")
                pp = trace.posterior_predictive
                if "y_obs" in pp:
                    samples = pp["y_obs"].values.reshape(-1, pp["y_obs"].shape[-1])
                    mean = samples.mean(axis=0)
                    lower, upper = np.percentile(samples, [10, 90], axis=0)
                    return {"mean": mean, "lower": lower, "upper": upper}
        except Exception as e:
            logger.warning(f"Error extracting predictions: {e}")
            pass
        logger.debug("Predictions not found")
        return None
    
    # _compute_fit_statistics inherited from DataExtractor

    def _extract_geo_level_fit_data(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract geo-level model fit data.
        
        Aggregation strategy:
        - For each geo: sum over products (if any) for each period
        - Result is a time series of length n_periods for each geo
        
        IMPORTANT: Uncertainty must be computed from aggregated samples per geo.
        
        Populates:
        - bundle.actual_by_geo
        - bundle.predicted_by_geo
        - bundle.fit_statistics_by_geo
        """
        import pandas as pd
        
        # Check if we have geo-level data
        if not hasattr(self, 'panel') or self.panel is None:
            return bundle
        
        panel = self.panel
        mmm = self.mmm
        
        # Get geo info
        geo_names = self._get_geo_names()
        if geo_names is None or len(geo_names) <= 1:
            return bundle
        
        bundle.geo_names = geo_names
        
        try:
            # Get period labels (unique periods in order)
            unique_periods = self._get_unique_periods()
            if unique_periods is None:
                return bundle
            
            # Get period index for each observation
            periods = self._get_period_labels_per_obs()
            if periods is None:
                return bundle
            
            # Get geo index for each observation
            geo_idx = self._get_geo_indices()
            if geo_idx is None:
                return bundle
            
            # Map geo indices to names
            geo_labels = [geo_names[int(g)] for g in geo_idx]
            
            # Get observed values (original scale)
            y_obs = self._get_actual_original_scale()
            if y_obs is None:
                return bundle
            
            # Build DataFrame for observed data aggregation
            df_obs = pd.DataFrame({
                'period': periods,
                'geo': geo_labels,
                'actual': y_obs,
            })
            
            # Initialize geo-level storage
            bundle.actual_by_geo = {}
            bundle.predicted_by_geo = {}
            bundle.fit_statistics_by_geo = {}
            
            # Get posterior samples for proper uncertainty propagation
            y_samples_orig = self._get_posterior_samples_original_scale()
            
            # Aggregate observed and predicted by geo
            for geo in geo_names:
                geo_df = df_obs[df_obs['geo'] == geo]
                if len(geo_df) == 0:
                    continue
                
                # Aggregate observed values for this geo
                agg_obs = geo_df.groupby('period')['actual'].sum()
                agg_obs = agg_obs.reindex(unique_periods).fillna(0)
                
                y_obs_geo = agg_obs.values
                bundle.actual_by_geo[geo] = y_obs_geo
                
                # Aggregate predictions with proper uncertainty for this geo
                if y_samples_orig is not None:
                    geo_mask = np.array(geo_labels) == geo
                    pred_geo = self._aggregate_samples_by_period(
                        y_samples_orig[:, geo_mask],
                        [p for p, g in zip(periods, geo_labels) if g == geo],
                        unique_periods,
                    )
                    
                    if pred_geo is not None:
                        bundle.predicted_by_geo[geo] = pred_geo
                        bundle.fit_statistics_by_geo[geo] = self._compute_fit_statistics(
                            y_obs_geo, pred_geo
                        )
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract geo-level fit data: {e}")
            import traceback
            logging.warning(traceback.format_exc())
        
        return bundle
    
    def _aggregate_samples_by_period(
        self,
        samples: np.ndarray,  # (n_samples, n_obs_subset)
        periods: list,  # period label for each obs in subset
        unique_periods: list,
    ) -> dict[str, np.ndarray] | None:
        """
        Aggregate samples by period, properly propagating uncertainty.
        
        For each period, sums across observations (products) in that period,
        then computes percentiles on the summed samples.
        """
        if samples is None or len(periods) == 0:
            return None
        
        try:
            period_to_idx = {p: i for i, p in enumerate(unique_periods)}
            obs_period_idx = np.array([period_to_idx[p] for p in periods])
            
            n_samples = samples.shape[0]
            n_periods = len(unique_periods)
            
            # Aggregate samples by period
            samples_agg = np.zeros((n_samples, n_periods))
            
            for t in range(n_periods):
                mask = (obs_period_idx == t)
                if mask.any():
                    samples_agg[:, t] = samples[:, mask].sum(axis=1)
            
            # Compute statistics
            alpha = (1 - self.ci_prob) / 2
            
            return {
                "mean": samples_agg.mean(axis=0),
                "lower": np.percentile(samples_agg, alpha * 100, axis=0),
                "upper": np.percentile(samples_agg, (1 - alpha) * 100, axis=0),
            }
            
        except Exception:
            return None

    def _get_posterior_samples_original_scale(self) -> np.ndarray | None:
        """Get posterior samples in original scale, shape (n_samples, n_obs)."""
        try:
            # Method 1: Use model's predict() method (most reliable)
            if hasattr(self.mmm, 'predict'):
                try:
                    pred_results = self.mmm.predict(return_original_scale=True, hdi_prob=self.ci_prob)
                    return pred_results.y_pred_samples  # (n_samples, n_obs)
                except Exception:
                    pass

            # Method 2: Reconstruct from trace components
            y_samples_orig = self._reconstruct_predictions_from_trace()
            if y_samples_orig is not None:
                return y_samples_orig

            # Method 3: Try to get from trace directly (legacy)
            trace = getattr(self.mmm, '_trace', None)
            if trace is None:
                return None

            y_samples = None
            if hasattr(trace, 'posterior_predictive'):
                pp = trace.posterior_predictive
                for var_name in ['y', 'y_obs', 'likelihood']:
                    if var_name in pp:
                        y_samples = pp[var_name].values
                        break

            if y_samples is None:
                return None

            # Reshape to (n_samples, n_obs)
            n_chains, n_draws = y_samples.shape[:2]
            n_obs = y_samples.shape[-1]
            y_samples = y_samples.reshape(n_chains * n_draws, n_obs)

            # Transform to original scale
            y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
            y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1

            return y_samples * y_std + y_mean

        except Exception:
            return None

    def _extract_geo_level_decomposition(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract geo-level decomposition data.
        
        Aggregation strategy:
        - For each geo and component: sum over products (if any) for each period
        - Result is a time series of length n_periods for each geo-component pair
        
        Populates:
        - bundle.component_time_series_by_geo
        - bundle.component_totals_by_geo
        """
        import pandas as pd
        
        if bundle.geo_names is None or len(bundle.geo_names) <= 1:
            return bundle
        
        try:
            mmm = self.mmm
            panel = self.panel
            
            geo_names = bundle.geo_names
            
            # Get period and geo info for each observation
            unique_periods = self._get_unique_periods()
            periods = self._get_period_labels_per_obs()
            geo_idx = self._get_geo_indices()
            
            if periods is None or geo_idx is None or unique_periods is None:
                return bundle
            
            geo_labels = [geo_names[int(g)] for g in geo_idx]
            
            # Get decomposition components at observation level
            components = self._get_decomposition_components_obs_level()
            if components is None:
                return bundle
            
            bundle.component_time_series_by_geo = {}
            bundle.component_totals_by_geo = {}
            
            for geo in geo_names:
                bundle.component_time_series_by_geo[geo] = {}
                bundle.component_totals_by_geo[geo] = {}
                
                for comp_name, comp_values in components.items():
                    # Build DataFrame for this component
                    df = pd.DataFrame({
                        'period': periods,
                        'geo': geo_labels,
                        'value': comp_values,
                    })
                    
                    # Filter to this geo and aggregate by period
                    geo_df = df[df['geo'] == geo]
                    if len(geo_df) == 0:
                        continue
                    
                    agg = geo_df.groupby('period')['value'].sum()
                    agg = agg.reindex(unique_periods).fillna(0)
                    
                    comp_geo = agg.values
                    bundle.component_time_series_by_geo[geo][comp_name] = comp_geo
                    bundle.component_totals_by_geo[geo][comp_name] = float(comp_geo.sum())
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract geo-level decomposition: {e}")
            import traceback
            logging.warning(traceback.format_exc())
        
        return bundle

    def _extract_product_level_fit_data(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract product-level model fit data.

        Populates:
        - bundle.product_names
        - bundle.actual_by_product
        - bundle.predicted_by_product
        - bundle.fit_statistics_by_product
        """
        # Check if we have product-level data
        if not hasattr(self, 'panel') or self.panel is None:
            return bundle

        mmm = self.mmm

        # Get product info
        product_names = self._get_product_names()
        if product_names is None or len(product_names) <= 1:
            return bundle

        bundle.product_names = product_names

        try:
            # Get indices
            product_idx = self._get_product_indices()
            time_idx = self._get_time_indices()
            if product_idx is None or time_idx is None:
                return bundle

            # Get observed values (original scale)
            y_obs = self._get_actual_original_scale()
            if y_obs is None:
                return bundle

            # Get predictions
            y_pred_mean, y_pred_lower, y_pred_upper = self._get_predictions_original_scale()
            if y_pred_mean is None:
                return bundle

            # Get period info
            n_periods = mmm.n_periods if hasattr(mmm, 'n_periods') else len(bundle.dates)

            # Initialize product-level storage
            bundle.actual_by_product = {}
            bundle.predicted_by_product = {}
            bundle.fit_statistics_by_product = {}

            # Aggregate by product (sum over geos within each period)
            for p_idx, product in enumerate(product_names):
                # Get mask for this product
                product_mask = (product_idx == p_idx)

                # Aggregate observed values for this product over time
                y_obs_prod = self._aggregate_by_period_with_indices(
                    y_obs, time_idx, product_mask, n_periods
                )
                y_pred_mean_prod = self._aggregate_by_period_with_indices(
                    y_pred_mean, time_idx, product_mask, n_periods
                )
                y_pred_lower_prod = self._aggregate_by_period_with_indices(
                    y_pred_lower, time_idx, product_mask, n_periods
                )
                y_pred_upper_prod = self._aggregate_by_period_with_indices(
                    y_pred_upper, time_idx, product_mask, n_periods
                )

                bundle.actual_by_product[product] = y_obs_prod
                bundle.predicted_by_product[product] = {
                    "mean": y_pred_mean_prod,
                    "lower": y_pred_lower_prod,
                    "upper": y_pred_upper_prod,
                }

                # Compute fit statistics for this product
                bundle.fit_statistics_by_product[product] = self._compute_fit_statistics(
                    y_obs_prod,
                    {"mean": y_pred_mean_prod, "lower": y_pred_lower_prod, "upper": y_pred_upper_prod}
                )

        except Exception as e:
            import logging
            logging.warning(f"Failed to extract product-level fit data: {e}")

        return bundle

    def _extract_product_level_decomposition(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract product-level decomposition data.

        Populates:
        - bundle.component_time_series_by_product
        - bundle.component_totals_by_product
        """
        if bundle.product_names is None or len(bundle.product_names) <= 1:
            return bundle

        try:
            mmm = self.mmm

            product_names = bundle.product_names
            product_idx = self._get_product_indices()
            time_idx = self._get_time_indices()
            if product_idx is None or time_idx is None:
                return bundle

            n_periods = mmm.n_periods if hasattr(mmm, 'n_periods') else len(bundle.dates)

            # Get decomposition components at observation level
            components = self._get_decomposition_components_obs_level()
            if components is None:
                return bundle

            bundle.component_time_series_by_product = {}
            bundle.component_totals_by_product = {}

            for p_idx, product in enumerate(product_names):
                product_mask = (product_idx == p_idx)

                bundle.component_time_series_by_product[product] = {}
                bundle.component_totals_by_product[product] = {}

                for comp_name, comp_values in components.items():
                    # Aggregate this component for this product over time
                    comp_prod = self._aggregate_by_period_with_indices(
                        comp_values, time_idx, product_mask, n_periods
                    )
                    bundle.component_time_series_by_product[product][comp_name] = comp_prod
                    bundle.component_totals_by_product[product][comp_name] = float(comp_prod.sum())

        except Exception as e:
            import logging
            logging.warning(f"Failed to extract product-level decomposition: {e}")

        return bundle

    
    def _get_geo_names(self) -> list[str] | None:
        """Get geography names from panel or model."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'geographies'):
            geographies = self.panel.coords.geographies
            if geographies is not None:
                return list(geographies)
        if hasattr(self.mmm, 'geo_names'):
            geo_names = self.mmm.geo_names
            if geo_names is not None:
                return list(geo_names)
        return None
    
    def _get_geo_indices(self) -> np.ndarray | None:
        """Get geo index for each observation."""
        if hasattr(self.mmm, 'geo_idx') and self.mmm.geo_idx is not None:
            return np.array(self.mmm.geo_idx)
        if hasattr(self.panel, 'geo_idx') and self.panel.geo_idx is not None:
            return np.array(self.panel.geo_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            geo_names = self._get_geo_names()
            if geo_names is not None and len(geo_names) > 1:
                # Try to get from y.index if it's a MultiIndex
                if hasattr(self.panel, 'y') and hasattr(self.panel.y, 'index'):
                    idx = self.panel.y.index
                    if hasattr(idx, 'get_level_values'):
                        # Try both cases for geography level name
                        for level_name in ['geography', 'Geography']:
                            try:
                                geo_values = idx.get_level_values(level_name)
                                # Convert geo names to indices
                                geo_to_idx = {g: i for i, g in enumerate(geo_names)}
                                return np.array([geo_to_idx.get(str(g), 0) for g in geo_values])
                            except KeyError:
                                continue
        return None

    def _get_product_names(self) -> list[str] | None:
        """Get product names from panel or model."""
        if hasattr(self.panel, 'coords') and hasattr(self.panel.coords, 'products'):
            products = self.panel.coords.products
            if products is not None:
                return list(products)
        if hasattr(self.mmm, 'product_names'):
            return list(self.mmm.product_names)
        return None

    def _get_product_indices(self) -> np.ndarray | None:
        """Get product index for each observation."""
        if hasattr(self.mmm, 'product_idx'):
            return np.array(self.mmm.product_idx)
        if hasattr(self.panel, 'product_idx'):
            return np.array(self.panel.product_idx)
        return None

    def _get_time_indices(self) -> np.ndarray | None:
        """Get time index for each observation."""
        if hasattr(self.mmm, 'time_idx') and self.mmm.time_idx is not None:
            return np.array(self.mmm.time_idx)

        # Try to extract from panel's MultiIndex
        if self.panel is not None:
            unique_periods = self._get_unique_periods()
            if unique_periods is not None and len(unique_periods) > 0:
                if hasattr(self.panel, 'y') and hasattr(self.panel.y, 'index'):
                    idx = self.panel.y.index
                    if hasattr(idx, 'get_level_values'):
                        # Try both cases for period level name
                        for level_name in ['period', 'Period']:
                            try:
                                period_values = idx.get_level_values(level_name)
                                # Convert periods to indices
                                period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                                return np.array([period_to_idx.get(str(p), 0) for p in period_values])
                            except KeyError:
                                continue
                        # Try first level if period not found by name
                        try:
                            period_values = idx.get_level_values(0)
                            period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                            return np.array([period_to_idx.get(str(p), 0) for p in period_values])
                        except Exception:
                            pass
                    else:
                        # Simple index - assume it's period only
                        period_to_idx = {str(p): i for i, p in enumerate(unique_periods)}
                        return np.array([period_to_idx.get(str(p), 0) for p in idx])

        # Fallback: construct from panel coords
        if self.panel is not None and hasattr(self.panel, 'coords'):
            n_obs = len(self.panel.y)
            n_periods = self.panel.coords.n_periods
            n_geos = self.panel.coords.n_geos
            n_products = self.panel.coords.n_products
            # Assumes data ordered as periods innermost
            if n_obs == n_periods * n_geos * n_products:
                return np.tile(np.arange(n_periods), n_geos * n_products)
        return None

    def _get_actual_original_scale(self) -> np.ndarray | None:
        """Get observed values in original scale."""
        if self.panel is None:
            return None
        
        y_standardized = self.panel.y.values.flatten()
        y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
        y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
        
        return y_standardized * y_std + y_mean
    
    def _get_predictions_original_scale(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """Get predictions in original scale with uncertainty."""
        trace = getattr(self.mmm, '_trace', None)
        if trace is None:
            return None, None, None
        
        try:
            if hasattr(trace, 'posterior_predictive') and 'y' in trace.posterior_predictive:
                y_samples = trace.posterior_predictive['y'].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            elif hasattr(trace, 'posterior') and 'mu' in trace.posterior:
                y_samples = trace.posterior['mu'].values
                y_samples = y_samples.reshape(-1, y_samples.shape[-1])
            else:
                return None, None, None
            
            # Transform to original scale
            y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
            y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
            
            y_samples_orig = y_samples * y_std + y_mean
            
            y_pred_mean = y_samples_orig.mean(axis=0)
            alpha = (1 - self.ci_prob) / 2
            y_pred_lower = np.percentile(y_samples_orig, alpha * 100, axis=0)
            y_pred_upper = np.percentile(y_samples_orig, (1 - alpha) * 100, axis=0)
            
            return y_pred_mean, y_pred_lower, y_pred_upper
        except Exception:
            return None, None, None
    
    def _aggregate_by_period(
        self,
        values: np.ndarray,
        n_periods: int,
        geo_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate values by period for a specific geo.
        
        If there are multiple products, this sums over products within each period.
        """
        # Simple case: one observation per period per geo
        n_obs = len(values)
        if n_obs == n_periods:
            return values
        
        # Multiple products case: need to reshape and sum
        # Assumes data is ordered: periods are innermost, then geos, then products
        n_products = n_obs // n_periods
        if n_products * n_periods == n_obs:
            return values.reshape(n_products, n_periods).sum(axis=0)
        
        # Fallback: return as-is (may need custom handling)
        return values

    def _aggregate_by_period_with_indices(
        self,
        values: np.ndarray,
        time_idx: np.ndarray,
        dim_mask: np.ndarray,
        n_periods: int,
    ) -> np.ndarray:
        """
        Aggregate values by period using explicit time indices.

        This is a more robust aggregation method that uses the time_idx array
        to properly group observations by period, regardless of data ordering.

        Parameters
        ----------
        values : np.ndarray
            Values to aggregate, shape (n_obs,)
        time_idx : np.ndarray
            Time period index for each observation, shape (n_obs,)
        dim_mask : np.ndarray
            Boolean mask for filtering (e.g., specific geo or product), shape (n_obs,)
        n_periods : int
            Number of unique time periods

        Returns
        -------
        np.ndarray
            Aggregated values by period, shape (n_periods,)
        """
        result = np.zeros(n_periods)
        filtered_values = values[dim_mask]
        filtered_time_idx = time_idx[dim_mask]

        for t in range(n_periods):
            time_mask = (filtered_time_idx == t)
            if time_mask.any():
                result[t] = filtered_values[time_mask].sum()

        return result

    def _get_decomposition_components_obs_level(self) -> dict[str, np.ndarray] | None:
        """
        Get decomposition components at observation level (not yet aggregated).
        
        Returns dict mapping component name to array of shape (n_obs,).
        """
        trace = getattr(self.mmm, '_trace', None)
        if trace is None:
            return None
        
        components = {}
        posterior = trace.posterior
        
        y_mean = self.mmm.y_mean if hasattr(self.mmm, 'y_mean') else 0
        y_std = self.mmm.y_std if hasattr(self.mmm, 'y_std') else 1
        n_obs = len(self.panel.y.values.flatten())
        
        try:
            # Intercept/Baseline
            if 'intercept' in posterior:
                intercept = float(posterior['intercept'].values.mean())
                # Baseline in original scale includes both the intercept contribution
                # and the mean offset (y_mean) to ensure decomposition sums to predictions
                # Per-observation baseline = intercept * y_std + y_mean
                baseline_per_obs = intercept * y_std + y_mean
                components['Baseline'] = np.full(n_obs, baseline_per_obs)
            
            # Trend
            if 'trend' in posterior:
                trend_samples = posterior['trend'].values
                trend_mean = trend_samples.mean(axis=(0, 1))
                if len(trend_mean) == n_obs:
                    components['Trend'] = trend_mean * y_std
            
            # Seasonality
            if 'seasonality' in posterior:
                seas_samples = posterior['seasonality'].values
                seas_mean = seas_samples.mean(axis=(0, 1))
                if len(seas_mean) == n_obs:
                    components['Seasonality'] = seas_mean * y_std
            
            # Media channels
            channel_names = self._get_channel_names()

            # First try the channel_contributions array (most common for geo models)
            if 'channel_contributions' in posterior:
                contrib_da = posterior['channel_contributions']
                contrib_vals = contrib_da.values  # (chains, draws, obs, channels)
                contrib_mean = contrib_vals.mean(axis=(0, 1))  # (obs, channels)

                # Check shape and extract per-channel
                if contrib_mean.ndim == 2:
                    if contrib_mean.shape[0] == n_obs and contrib_mean.shape[1] == len(channel_names):
                        # Shape is (obs, channels)
                        for i, ch in enumerate(channel_names):
                            components[ch] = contrib_mean[:, i] * y_std
                    elif contrib_mean.shape[1] == n_obs and contrib_mean.shape[0] == len(channel_names):
                        # Shape is (channels, obs)
                        for i, ch in enumerate(channel_names):
                            components[ch] = contrib_mean[i, :] * y_std
            else:
                # Fall back to individual channel contribution variables
                for ch in channel_names:
                    contrib_key = f'channel_contribution_{ch}'
                    if contrib_key in posterior:
                        contrib = posterior[contrib_key].values.mean(axis=(0, 1))
                        if len(contrib) == n_obs:
                            components[ch] = contrib * y_std

            # Controls
            if hasattr(self.mmm, 'control_names'):
                for ctrl in self.mmm.control_names:
                    ctrl_key = f'control_contribution_{ctrl}'
                    if ctrl_key in posterior:
                        ctrl_contrib = posterior[ctrl_key].values.mean(axis=(0, 1))
                        if len(ctrl_contrib) == n_obs:
                            components[f'Control: {ctrl}'] = ctrl_contrib * y_std
            
            return components if components else None
            
        except Exception:
            return None
    
    def _compute_channel_roi(self) -> dict[str, dict[str, float]] | None:
        """Compute channel ROI with uncertainty - FIXED VERSION."""
        try:
            channels = self._get_channel_names()
            if not channels:
                return None
            
            current_spend = self._get_current_spend()
            if not current_spend:
                logger.debug("No spend data for ROI computation")
                return None
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                return None
            
            posterior = trace.posterior
            y_std = getattr(self.mmm, "y_std", 1.0)
            n_obs = getattr(self.mmm, "n_obs", 52)
            
            roi_results = {}
            
            for ch in channels:
                spend = current_spend.get(ch, 0)
                if spend <= 0:
                    continue
                
                # Try to get contribution samples
                contrib_samples = None
                
                # Method 1: Direct contribution variable
                contrib_names = [
                    f"contribution_{ch}",
                    f"channel_contribution_{ch}",
                ]
                for contrib_name in contrib_names:
                    if contrib_name in posterior:
                        vals = posterior[contrib_name].values
                        # Flatten chains and draws, sum over time if needed
                        if vals.ndim > 2:
                            contrib_samples = vals.reshape(-1, *vals.shape[2:]).sum(axis=-1) * y_std
                        else:
                            contrib_samples = vals.flatten() * y_std * n_obs
                        break
                
                # Method 2: From channel_contributions array
                if contrib_samples is None and "channel_contributions" in posterior:
                    try:
                        ch_idx = channels.index(ch)
                        contrib_da = posterior["channel_contributions"]
                        vals = contrib_da.values  # numpy first!
                        
                        # Flatten chains/draws, extract channel, sum over time
                        flat = vals.reshape(-1, *vals.shape[2:])
                        if flat.ndim == 2:  # (samples, time)
                            contrib_samples = flat.sum(axis=-1) * y_std
                        elif flat.ndim == 3:  # (samples, time, channels)
                            contrib_samples = flat[:, :, ch_idx].sum(axis=-1) * y_std
                    except Exception as e:
                        logger.debug(f"Could not extract from channel_contributions: {e}")
                
                # Method 3: Estimate from beta
                if contrib_samples is None:
                    for beta_name in [f"beta_{ch}", f"beta_media_{ch}"]:
                        if beta_name in posterior:
                            beta_vals = posterior[beta_name].values.flatten()
                            # Rough estimate
                            contrib_samples = beta_vals * y_std * n_obs * 0.5
                            break
                
                if contrib_samples is not None and len(contrib_samples) > 0:
                    roi_samples = contrib_samples / spend
                    
                    roi_results[ch] = {
                        "mean": float(np.mean(roi_samples)),
                        "lower": float(np.percentile(roi_samples, (1 - self.ci_prob) / 2 * 100)),
                        "upper": float(np.percentile(roi_samples, (1 + self.ci_prob) / 2 * 100)),
                    }
            
            return roi_results if roi_results else None
            
        except Exception as e:
            logger.warning(f"Error computing channel ROI: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _get_component_totals(self) -> dict[str, float] | None:
        """Get total contribution by component"""
        try:
            logger.debug("Computing component totals")
            
            # Try model's method first
            if hasattr(self.mmm, "compute_contributions"):
                contrib = self.mmm.compute_contributions()
                if hasattr(contrib, "component_totals"):
                    return dict(contrib.component_totals)
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                return None
            
            posterior = trace.posterior
            y_std = getattr(self.mmm, "y_std", 1.0)
            n_obs = getattr(self.mmm, "n_obs", 52)
            
            totals = {}
            
            # Baseline/Intercept
            if "intercept" in posterior:
                intercept = float(posterior["intercept"].values.mean()) * y_std * n_obs
                totals["Baseline"] = intercept
            
            # Media channels
            channels = self._get_channel_names()
            current_spend = self._get_current_spend()
            
            for ch in channels:
                # Try to get contribution from trace
                contrib_names = [
                    f"contribution_{ch}",
                    f"channel_contribution_{ch}",
                    f"media_contribution_{ch}",
                ]
                
                found = False
                for contrib_name in contrib_names:
                    if contrib_name in posterior:
                        # Sum over time
                        contrib_vals = posterior[contrib_name].values
                        total_contrib = float(contrib_vals.mean()) * y_std
                        if contrib_vals.ndim > 2:
                            # Has time dimension - sum it
                            total_contrib = float(contrib_vals.sum(axis=-1).mean()) * y_std
                        totals[ch] = total_contrib
                        found = True
                        break
                
                if not found:
                    # Estimate from beta
                    for beta_name in [f"beta_{ch}", f"beta_media_{ch}"]:
                        if beta_name in posterior:
                            beta_val = float(posterior[beta_name].values.mean())
                            
                            # Try to get media sum
                            if current_spend and ch in current_spend:
                                # Rough estimate
                                totals[ch] = beta_val * y_std * n_obs * 0.5  # Approximate
                            else:
                                totals[ch] = beta_val * y_std * n_obs
                            break
            
            return totals if totals else None
            
        except Exception as e:
            logger.warning(f"Error computing component totals: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _get_component_time_series(self) -> dict[str, np.ndarray] | None:
        """Get component time series for decomposition chart - FIXED VERSION."""
        try:
            logger.debug("Computing component time series")
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                return None
            
            posterior = trace.posterior
            y_std = getattr(self.mmm, "y_std", 1.0)
            n_obs = getattr(self.mmm, "n_obs", 52)
            
            components = {}
            
            # Baseline - constant
            if "intercept" in posterior:
                intercept = float(posterior["intercept"].values.mean()) * y_std
                components["Baseline"] = np.full(n_obs, intercept)
            
            # Trend
            trend_names = ["trend", "trend_contribution", "trend_component"]
            for trend_name in trend_names:
                if trend_name in posterior:
                    trend_vals = posterior[trend_name].values
                    # Average over chains and draws
                    trend_mean = trend_vals.mean(axis=(0, 1))
                    if len(trend_mean) == n_obs:
                        components["Trend"] = trend_mean * y_std
                    break
            
            # Seasonality
            seas_names = ["seasonality", "seasonality_component", "seasonal_effect"]
            for seas_name in seas_names:
                if seas_name in posterior:
                    seas_vals = posterior[seas_name].values
                    seas_mean = seas_vals.mean(axis=(0, 1))
                    if len(seas_mean) == n_obs:
                        components["Seasonality"] = seas_mean * y_std
                    break
            
            # Channel contributions
            channels = self._get_channel_names()
            
            # Try channel_contributions array
            if "channel_contributions" in posterior:
                contrib_da = posterior["channel_contributions"]
                contrib_vals = contrib_da.values  # Get numpy first!
                
                # Average over chains and draws
                if contrib_vals.ndim >= 3:
                    contrib_mean = contrib_vals.mean(axis=(0, 1))  # (time, channels) or (channels,)
                    
                    if contrib_mean.ndim == 2 and contrib_mean.shape[0] == n_obs:
                        # Shape is (time, channels)
                        for i, ch in enumerate(channels):
                            if i < contrib_mean.shape[1]:
                                components[ch] = contrib_mean[:, i] * y_std
                    elif contrib_mean.ndim == 2 and contrib_mean.shape[1] == n_obs:
                        # Shape is (channels, time)
                        for i, ch in enumerate(channels):
                            if i < contrib_mean.shape[0]:
                                components[ch] = contrib_mean[i, :] * y_std
            else:
                # Try individual contribution variables
                for ch in channels:
                    contrib_names = [
                        f"contribution_{ch}",
                        f"channel_contribution_{ch}",
                        f"media_contribution_{ch}",
                    ]
                    
                    for contrib_name in contrib_names:
                        if contrib_name in posterior:
                            contrib_vals = posterior[contrib_name].values
                            contrib_mean = contrib_vals.mean(axis=(0, 1))
                            if len(contrib_mean) == n_obs:
                                components[ch] = contrib_mean * y_std
                            break
            
            return components if components else None
            
        except Exception as e:
            logger.warning(f"Error computing component time series: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _compute_marketing_attribution(self) -> dict[str, float] | None:
        """Compute total marketing-attributed revenue with uncertainty."""
        try:
            # First, try using the model's compute_contributions method
            if hasattr(self.mmm, "compute_counterfactual_contributions"):
                contrib_results = self.mmm.compute_counterfactual_contributions(
                    compute_uncertainty=True,
                    hdi_prob=self.ci_prob,
                )
                total = float(contrib_results.total_contributions.sum())
                # Get uncertainty from HDI if available
                if contrib_results.contribution_hdi_low is not None:
                    lower = float(contrib_results.contribution_hdi_low.sum())
                    upper = float(contrib_results.contribution_hdi_high.sum())
                else:
                    # Rough estimate: ±15% of total
                    lower = total * 0.85
                    upper = total * 1.15
                return {"mean": total, "lower": lower, "upper": upper}
            
            
            logger.debug("Computing total marketing-attributed revenue")
            trace = getattr(self.mmm, "_trace", None)
            logger.debug("Accessing model trace for attribution computation")
            if trace is None:
                logger.debug("Model trace not found for attribution computation")
                return None
            
            channels = self._get_channel_names()
            
            # Sum contributions across channels
            total_samples = None
            for ch in channels:
                contrib_name = f"contribution_{ch}"
                if hasattr(trace, "posterior") and contrib_name in trace.posterior:
                    logger.debug(f"Adding contribution samples for channel: {ch}")
                    ch_samples = trace.posterior[contrib_name].values.sum(axis=-1).flatten()
                    if total_samples is None:
                        logger.debug("Initializing total samples for attribution")
                        total_samples = ch_samples
                    else:
                        logger.debug("Accumulating total samples for attribution")
                        total_samples = total_samples + ch_samples
            
            if total_samples is not None:
                logger.debug("Computing mean and HDI for total marketing attribution")
                mean = float(total_samples.mean())
                lower, upper = self._compute_hdi(total_samples, self.ci_prob)
                return {"mean": mean, "lower": lower, "upper": upper}
        except Exception as e:
            logger.warning(f"Error computing marketing attribution: {e}")
            pass
        
        return None
    
    def _compute_blended_roi(self) -> dict[str, float] | None:
        """Compute blended marketing ROI with uncertainty."""
        try:
            # Get total attribution and spend
            attribution = self._compute_marketing_attribution()
            spend = self._get_current_spend()
            
            if attribution is None or spend is None:
                return None
            
            total_spend = sum(spend.values())
            if total_spend == 0:
                return None
            
            # Simple division (would need full posterior for proper uncertainty)
            mean_roi = attribution["mean"] / total_spend
            lower_roi = attribution["lower"] / total_spend
            upper_roi = attribution["upper"] / total_spend
            
            return {"mean": mean_roi, "lower": lower_roi, "upper": upper_roi}
        except Exception:
            return None
    
    def _compute_marketing_contribution_pct(self, total_revenue: float | None) -> dict[str, float] | None:
        """Compute marketing contribution as percentage of total."""
        if total_revenue is None or total_revenue == 0:
            return None
        
        attribution = self._compute_marketing_attribution()
        if attribution is None:
            return None
        
        return {
            "mean": attribution["mean"] / total_revenue,
            "lower": attribution["lower"] / total_revenue,
            "upper": attribution["upper"] / total_revenue,
        }
    
    def _get_saturation_curves(self) -> dict[str, dict[str, np.ndarray]] | None:
        """Get saturation curve data for each channel - FIXED VERSION."""
        try:
            # If model has a method for this, use it
            if hasattr(self.mmm, "compute_saturation_curves"):
                result = self.mmm.compute_saturation_curves()
                if result:
                    return result
            
            channels = self._get_channel_names()
            if not channels:
                return None
            
            # Get current spend for setting spend range
            current_spend = self._get_current_spend()
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                logger.debug("No trace/posterior found for saturation curves")
                return None
            
            posterior = trace.posterior
            curves = {}
            
            for ch in channels:
                # Determine spend range
                max_spend = 1e6  # Default
                if current_spend and ch in current_spend:
                    max_spend = current_spend[ch] * 2  # 2x current spend
                
                spend_range = np.linspace(0, max_spend, 100)
                
                # Try Hill saturation parameters
                kappa_names = [f"kappa_{ch}", f"K_{ch}", f"sat_K_{ch}"]
                slope_names = [f"slope_{ch}", f"S_{ch}", f"sat_S_{ch}", f"n_{ch}"]
                
                kappa_val = None
                slope_val = None
                
                for k_name in kappa_names:
                    if k_name in posterior:
                        kappa_val = float(posterior[k_name].values.mean())
                        break
                
                for s_name in slope_names:
                    if s_name in posterior:
                        slope_val = float(posterior[s_name].values.mean())
                        break
                
                if kappa_val is not None and slope_val is not None:
                    # Hill function
                    response = spend_range ** slope_val / (kappa_val ** slope_val + spend_range ** slope_val)
                    
                    # Scale by beta if available
                    for beta_name in [f"beta_{ch}", f"beta_media_{ch}"]:
                        if beta_name in posterior:
                            beta_val = float(posterior[beta_name].values.mean())
                            response = response * beta_val
                            break
                    
                    curves[ch] = {"spend": spend_range, "response": response}
                    logger.debug(f"Generated Hill saturation curve for {ch}")
                    continue
                
                # Try exponential saturation (sat_lam)
                for lam_name in [f"sat_lam_{ch}", f"saturation_lam_{ch}", f"lam_{ch}"]:
                    if lam_name in posterior:
                        lam_val = float(posterior[lam_name].values.mean())
                        
                        # Exponential saturation: 1 - exp(-lam * x)
                        response = 1 - np.exp(-lam_val * spend_range / max(spend_range.max(), 1))
                        
                        # Scale by beta if available
                        for beta_name in [f"beta_{ch}", f"beta_media_{ch}"]:
                            if beta_name in posterior:
                                beta_val = float(posterior[beta_name].values.mean())
                                response = response * beta_val
                                break
                        
                        curves[ch] = {"spend": spend_range, "response": response}
                        logger.debug(f"Generated exponential saturation curve for {ch}")
                        break
            
            return curves if curves else None
            
        except Exception as e:
            logger.warning(f"Error getting saturation curves: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    
    def _get_adstock_curves(self) -> dict[str, np.ndarray] | None:
        """Get adstock decay weights for each channel"""
        try:
            if hasattr(self.mmm, "compute_adstock_curves"):
                result = self.mmm.compute_adstock_curves()
                if result:
                    return result
            
            channels = self._get_channel_names()
            if not channels:
                return None
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                return None
            
            posterior = trace.posterior
            curves = {}
            
            for ch in channels:
                # Try different alpha parameter names
                alpha_names = [f"alpha_{ch}", f"adstock_{ch}", f"decay_{ch}", f"adstock_alpha_{ch}"]
                
                alpha_val = None
                for alpha_name in alpha_names:
                    if alpha_name in posterior:
                        alpha_val = float(posterior[alpha_name].values.mean())
                        break
                
                if alpha_val is not None and 0 < alpha_val < 1:
                    # Get l_max from model config or default
                    l_max = getattr(self.mmm, "adstock_lmax", 8)
                    if hasattr(self.mmm, "model_config"):
                        l_max = getattr(self.mmm.model_config, "adstock_lmax", l_max)
                    
                    # Geometric decay weights
                    lags = np.arange(l_max)
                    weights = alpha_val ** lags
                    weights = weights / weights.sum()  # Normalize
                    
                    curves[ch] = weights
                    logger.debug(f"Generated adstock curve for {ch}: alpha={alpha_val:.3f}")
            
            return curves if curves else None
            
        except Exception as e:
            logger.warning(f"Error getting adstock curves: {e}")
            return None
    
    def _get_current_spend(self) -> dict[str, float] | None:
        """Get current spend levels by channel"""
        try:
            channels = self._get_channel_names()
            if not channels:
                logger.debug("No channel names found for spend extraction")
                return None
            
            # Try to get X_media from panel
            X_media = None
            if self.panel is not None and hasattr(self.panel, "X_media"):
                X_media = self.panel.X_media
            elif hasattr(self.mmm, "X_media_raw"):
                X_media = self.mmm.X_media_raw
            elif hasattr(self.mmm, "X_media"):
                X_media = self.mmm.X_media
            
            if X_media is None:
                logger.debug("No X_media found for spend extraction")
                return None
            
            spend = {}
            for i, ch in enumerate(channels):
                col_data = _safe_get_column(X_media, i, ch)
                if col_data is not None:
                    spend[ch] = float(col_data.sum())
                else:
                    logger.debug(f"Could not extract spend for channel {ch}")
            
            return spend if spend else None
            
        except Exception as e:
            logger.warning(f"Error getting current spend: {e}")
            return None
    
    def _get_trace_data(self) -> tuple[dict[str, np.ndarray] | None, list[str] | None]:
        """Get trace data for diagnostic plots."""
        try:
            trace = getattr(self.mmm, "_trace", None)
            if trace is None or not hasattr(trace, "posterior"):
                return None, None
            
            # Get key parameters
            params = []
            data = {}
            
            for var_name in trace.posterior.data_vars:
                if any(prefix in var_name for prefix in ["beta", "alpha", "sigma", "intercept"]):
                    values = trace.posterior[var_name].values
                    if values.ndim >= 2:
                        # Shape: (chain, draw, ...)
                        data[var_name] = values.reshape(values.shape[0], -1)[:, :values.shape[1]]
                    params.append(var_name)
            
            return data, params
        except Exception:
            return None, None
        
    def _extract_aggregated_decomposition(
        self,
        bundle: MMMDataBundle,
    ) -> MMMDataBundle:
        """
        Extract properly aggregated decomposition for multi-geo models.
        
        Ensures bundle.component_time_series and bundle.component_totals
        are aggregated to period-level.
        """
        import pandas as pd
        
        if not hasattr(self, 'panel') or self.panel is None:
            return bundle
        
        # Check if we have multi-dimensional data
        geo_idx = self._get_geo_indices()
        has_geo = geo_idx is not None and len(set(geo_idx)) > 1
        
        if not has_geo:
            return bundle
        
        try:
            unique_periods = self._get_unique_periods()
            periods = self._get_period_labels_per_obs()
            
            if unique_periods is None or periods is None:
                return bundle
            
            # Get decomposition components at observation level
            components = self._get_decomposition_components_obs_level()
            if components is None:
                return bundle
            
            bundle.component_time_series = {}
            bundle.component_totals = {}
            
            for comp_name, comp_values in components.items():
                df = pd.DataFrame({
                    'period': periods,
                    'value': comp_values,
                })
                
                agg = df.groupby('period')['value'].sum()
                agg = agg.reindex(unique_periods).fillna(0)
                
                bundle.component_time_series[comp_name] = agg.values
                bundle.component_totals[comp_name] = float(agg.sum())
            
        except Exception as e:
            import logging
            logging.warning(f"Failed to extract aggregated decomposition: {e}")
            import traceback
            logging.warning(traceback.format_exc())
        
        return bundle
    
    def _get_prior_posterior(self) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
        """Get prior and posterior samples for comparison."""
        try:
            trace = getattr(self.mmm, "_trace", None)
            if trace is None:
                logger.debug("No trace found for prior/posterior extraction")
                return None, None
            
            posterior_samples = {}
            prior_samples = {}
            
            # Get posterior samples
            if hasattr(trace, "posterior"):
                # Select key parameters to include
                key_prefixes = ["beta", "sigma", "intercept", "alpha", "adstock", 
                            "sat_lam", "kappa", "slope"]
                
                for var_name in trace.posterior.data_vars:
                    # Only include key parameters (not all deterministics)
                    if any(prefix in var_name for prefix in key_prefixes):
                        try:
                            samples = trace.posterior[var_name].values.flatten()
                            # Subsample if too many
                            if len(samples) > 2000:
                                idx = np.random.choice(len(samples), 2000, replace=False)
                                samples = samples[idx]
                            posterior_samples[var_name] = samples
                        except Exception as e:
                            logger.debug(f"Could not extract posterior for {var_name}: {e}")
            
            if not posterior_samples:
                logger.debug("No posterior samples extracted")
                return None, None
            
            # Try to get prior samples from trace (if sample_prior_predictive was called)
            if hasattr(trace, "prior"):
                for var_name in trace.prior.data_vars:
                    if var_name in posterior_samples:
                        try:
                            samples = trace.prior[var_name].values.flatten()
                            if len(samples) > 2000:
                                idx = np.random.choice(len(samples), 2000, replace=False)
                                samples = samples[idx]
                            prior_samples[var_name] = samples
                        except Exception as e:
                            logger.debug(f"Could not extract prior for {var_name}: {e}")
            
            # If no prior samples in trace, try to generate from model
            if not prior_samples:
                prior_samples = self._generate_prior_samples(posterior_samples.keys())
            
            logger.debug(f"Extracted {len(posterior_samples)} posterior params, {len(prior_samples)} prior params")
            
            return (
                prior_samples if prior_samples else None, 
                posterior_samples if posterior_samples else None
            )
            
        except Exception as e:
            logger.warning(f"Error in _get_prior_posterior: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None


    def _generate_prior_samples(self, param_names: list[str], n_samples: int = 1000) -> dict[str, np.ndarray]:
        """
        Generate prior samples from model specification.
        
        This is used when sample_prior_predictive wasn't called during fitting.
        """
        import pymc as pm
        
        prior_samples = {}
        
        try:
            model = getattr(self.mmm, "model", None) or getattr(self.mmm, "_model", None)
            if model is None:
                logger.debug("No PyMC model found for prior sampling")
                return {}
            
            # Sample from prior
            with model:
                prior_trace = pm.sample_prior_predictive(samples=n_samples, random_seed=42)
            
            if hasattr(prior_trace, "prior"):
                prior_data = prior_trace.prior
                for param in param_names:
                    if param in prior_data:
                        try:
                            samples = prior_data[param].values.flatten()
                            if len(samples) > n_samples:
                                idx = np.random.choice(len(samples), n_samples, replace=False)
                                samples = samples[idx]
                            prior_samples[param] = samples
                        except Exception:
                            pass
            
            logger.debug(f"Generated {len(prior_samples)} prior sample sets")
            
        except Exception as e:
            logger.debug(f"Could not generate prior samples: {e}")
            # Fall back to generating from known prior distributions
            prior_samples = self._generate_fallback_priors(param_names, n_samples)
        
        return prior_samples


    def _generate_fallback_priors(self, param_names: list[str], n_samples: int = 1000) -> dict[str, np.ndarray]:
        """
        Generate fallback prior samples based on common MMM prior conventions.
        
        Used when we can't sample from the actual model.
        """
        prior_samples = {}
        
        for param in param_names:
            try:
                if param.startswith("beta_") or param.startswith("beta_media_"):
                    # Beta coefficients: HalfNormal(sigma=2)
                    prior_samples[param] = np.abs(np.random.normal(0, 2, n_samples))
                
                elif param == "sigma":
                    # Noise scale: HalfNormal(sigma=1)
                    prior_samples[param] = np.abs(np.random.normal(0, 1, n_samples))
                
                elif param == "intercept":
                    # Intercept: Normal(0, 5)
                    prior_samples[param] = np.random.normal(0, 5, n_samples)
                
                elif param.startswith("alpha_") or param.startswith("adstock_"):
                    # Adstock decay: Beta(1, 3) - favors lower values
                    prior_samples[param] = np.random.beta(1, 3, n_samples)
                
                elif param.startswith("sat_lam_") or param.startswith("saturation_"):
                    # Saturation rate: Gamma(2, 1)
                    prior_samples[param] = np.random.gamma(2, 1, n_samples)
                
                elif param.startswith("kappa_") or param.startswith("K_"):
                    # Hill half-saturation: Gamma(2, 1)
                    prior_samples[param] = np.random.gamma(2, 1, n_samples)
                
                elif param.startswith("slope_") or param.startswith("S_"):
                    # Hill slope: Gamma(3, 1)
                    prior_samples[param] = np.random.gamma(3, 1, n_samples)
                
            except Exception:
                pass
        
        return prior_samples
    
    def _get_model_specification(self) -> dict[str, Any] | None:
        """Get model specification details."""
        spec = {}
        
        try:
            # Get from model config
            if hasattr(self.mmm, "model_config"):
                config = self.mmm.model_config
                spec["chains"] = getattr(config, "chains", 4)
                spec["draws"] = getattr(config, "draws", 2000)
                spec["tune"] = getattr(config, "tune", 1000)
            
            # Get trend type
            if hasattr(self.mmm, "trend_config"):
                trend = self.mmm.trend_config
                spec["baseline"] = f"{getattr(trend, 'type', 'Linear').value} trend + Fourier seasonality"
            
            spec["likelihood"] = "Normal with estimated scale"
            spec["media_effects"] = "Hill saturation × Geometric adstock"
            spec["priors"] = "Weakly informative"
            
            return spec
        except Exception:
            return None


class ExtendedMMMExtractor(DataExtractor):
    """
    Extract data from mmm-framework's extended MMM models.

    Inherits shared utilities from DataExtractor for HDI computation,
    fit statistics, and MCMC diagnostics.

    Supports NestedMMM, MultivariateMMM, and CombinedMMM.
    """

    def __init__(
        self,
        model: Any,
        ci_prob: float = 0.8,
    ):
        self.model = model
        self._ci_prob = ci_prob
        self._base_extractor = None

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob
    
    def extract(self) -> MMMDataBundle:
        """Extract data from extended MMM model."""
        # Start with base extraction if available
        bundle = MMMDataBundle()
        
        # Extract common data
        bundle.channel_names = self._get_channel_names()
        bundle.dates = self._get_dates()
        bundle.actual = self._get_actual()
        
        # Get trace for diagnostics
        trace = getattr(self.model, "_trace", None)
        if trace is not None:
            bundle.diagnostics = self._extract_diagnostics(trace)
        
        # Model-specific extraction
        model_type = type(self.model).__name__
        
        if "Nested" in model_type:
            bundle = self._extract_nested_data(bundle)
        elif "Multivariate" in model_type:
            bundle = self._extract_multivariate_data(bundle)
        elif "Combined" in model_type:
            bundle = self._extract_combined_data(bundle)
        
        # Model specification
        bundle.model_specification = self._get_model_specification()
        
        return bundle
    
    def _get_channel_names(self) -> list[str]:
        if hasattr(self.model, "channel_names"):
            return list(self.model.channel_names)
        return []
    
    def _get_dates(self) -> np.ndarray | None:
        if hasattr(self.model, "index"):
            return np.array(self.model.index)
        return None
    
    def _get_actual(self) -> np.ndarray | None:
        if hasattr(self.model, "y"):
            return np.array(self.model.y)
        return None
    
    def _extract_nested_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data specific to NestedMMM (mediated effects)."""
        try:
            if hasattr(self.model, "mediator_names"):
                mediator_effects = {}
                
                trace = getattr(self.model, "_trace", None)
                if trace is not None and hasattr(trace, "posterior"):
                    for med in self.model.mediator_names:
                        # Extract media -> mediator effects
                        for ch in bundle.channel_names or []:
                            effect_name = f"gamma_{ch}_{med}"
                            if effect_name in trace.posterior:
                                samples = trace.posterior[effect_name].values.flatten()
                                mediator_effects[f"{ch} → {med}"] = {
                                    "mean": float(samples.mean()),
                                    "lower": float(np.percentile(samples, 10)),
                                    "upper": float(np.percentile(samples, 90)),
                                }
                
                bundle.mediator_effects = mediator_effects
        except Exception:
            pass
        
        return bundle
    
    def _extract_multivariate_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data specific to MultivariateMMM (multiple outcomes)."""
        try:
            trace = getattr(self.model, "_trace", None)
            if trace is not None and hasattr(trace, "posterior"):
                # Extract correlation matrix if present
                if "corr_chol" in trace.posterior:
                    chol = trace.posterior["corr_chol"].values.mean(axis=(0, 1))
                    bundle.outcome_correlations = chol @ chol.T
        except Exception:
            pass
        
        return bundle
    
    def _extract_combined_data(self, bundle: MMMDataBundle) -> MMMDataBundle:
        """Extract data from CombinedMMM (nested + multivariate)."""
        bundle = self._extract_nested_data(bundle)
        bundle = self._extract_multivariate_data(bundle)
        return bundle
    
    def _get_model_specification(self) -> dict[str, Any]:
        """Get model specification for extended models."""
        spec = {
            "model_type": type(self.model).__name__,
            "likelihood": "Normal",
            "media_effects": "Configured transformations",
        }
        
        if hasattr(self.model, "mediator_names"):
            spec["mediators"] = list(self.model.mediator_names)
        
        return spec


class PyMCMarketingExtractor(DataExtractor):
    """
    Extract data from pymc-marketing's MMM class.

    Inherits shared utilities from DataExtractor for HDI computation,
    fit statistics, and MCMC diagnostics.

    Provides compatibility with the standard pymc-marketing MMM.
    """

    def __init__(
        self,
        mmm: Any,
        ci_prob: float = 0.8,
    ):
        self.mmm = mmm
        self._ci_prob = ci_prob

    @property
    def ci_prob(self) -> float:
        """Credible interval probability."""
        return self._ci_prob
    
    def extract(self) -> MMMDataBundle:
        """Extract data from pymc-marketing MMM."""
        bundle = MMMDataBundle()
        
        # Get channel names
        if hasattr(self.mmm, "channel_columns"):
            bundle.channel_names = list(self.mmm.channel_columns)
        
        # Get dates
        if hasattr(self.mmm, "X") and hasattr(self.mmm, "date_column"):
            bundle.dates = self.mmm.X[self.mmm.date_column].values
        
        # Get actual values
        if hasattr(self.mmm, "y"):
            bundle.actual = np.array(self.mmm.y)
        
        # Get predictions if fitted
        if hasattr(self.mmm, "idata") and self.mmm.idata is not None:
            try:
                posterior_pred = self.mmm.sample_posterior_predictive()
                if "y" in posterior_pred.posterior_predictive:
                    samples = posterior_pred.posterior_predictive["y"].values
                    samples = samples.reshape(-1, samples.shape[-1])
                    bundle.predicted = {
                        "mean": samples.mean(axis=0),
                        "lower": np.percentile(samples, 10, axis=0),
                        "upper": np.percentile(samples, 90, axis=0),
                    }
            except Exception:
                pass
            
            # Diagnostics
            bundle.diagnostics = self._extract_diagnostics(self.mmm.idata)
            
            # Channel contributions
            try:
                contrib = self.mmm.compute_channel_contribution_original_scale()
                if contrib is not None:
                    bundle.component_time_series = {}
                    bundle.component_totals = {}
                    
                    for ch in bundle.channel_names or []:
                        if ch in contrib.columns:
                            bundle.component_time_series[ch] = contrib[ch].values
                            bundle.component_totals[ch] = float(contrib[ch].sum())
            except Exception:
                pass
        
        # Model specification
        bundle.model_specification = {
            "likelihood": "Normal",
            "adstock": type(self.mmm.adstock).__name__ if hasattr(self.mmm, "adstock") else "Geometric",
            "saturation": type(self.mmm.saturation).__name__ if hasattr(self.mmm, "saturation") else "Hill",
        }
        
        return bundle


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
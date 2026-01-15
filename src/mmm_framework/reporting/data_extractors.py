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
    """Base class for model data extractors."""
    
    @abstractmethod
    def extract(self) -> MMMDataBundle:
        """Extract data from model into unified bundle."""
        pass
    
    def _compute_hdi(
        self,
        samples: np.ndarray,
        prob: float = 0.8,
    ) -> tuple[float, float]:
        """Compute highest density interval from samples."""
        if az is not None:
            hdi = az.hdi(samples, hdi_prob=prob)
            return float(hdi[0]), float(hdi[1])
        else:
            # Fallback to percentile-based interval
            alpha = (1 - prob) / 2
            return float(np.percentile(samples, alpha * 100)), float(np.percentile(samples, (1 - alpha) * 100))
    
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


class BayesianMMMExtractor(DataExtractor):
    """
    Extract data from mmm-framework's BayesianMMM class.
    
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
        self.ci_prob = ci_prob
    
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
        
        # Model specification
        bundle.model_specification = self._get_model_specification()
        
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
    
    def _compute_fit_statistics(
        self,
        actual: np.ndarray | None,
        predicted: dict[str, np.ndarray] | None,
    ) -> dict[str, float] | None:
        """Compute R², RMSE, MAE, MAPE."""
        logger.debug("Computing fit statistics")
        if actual is None or predicted is None:
            return None
        
        y_true = actual
        y_pred = predicted["mean"]
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (handle zeros)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        
        return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape}
    
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
    
    Supports NestedMMM, MultivariateMMM, and CombinedMMM.
    """
    
    def __init__(
        self,
        model: Any,
        ci_prob: float = 0.8,
    ):
        self.model = model
        self.ci_prob = ci_prob
        self._base_extractor = None
    
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
    
    Provides compatibility with the standard pymc-marketing MMM.
    """
    
    def __init__(
        self,
        mmm: Any,
        ci_prob: float = 0.8,
    ):
        self.mmm = mmm
        self.ci_prob = ci_prob
    
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
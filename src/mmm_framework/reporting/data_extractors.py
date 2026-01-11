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
        """Compute ROI with uncertainty for each channel."""
        try:
            logger.debug("Computing channel ROI")
            if hasattr(self.mmm, "compute_roi"):
                logger.debug("Using model's compute_roi method")
                roi_results = self.mmm.compute_roi()
                return roi_results
            logger.debug("Computing channel ROI manually from trace")
            # Manual computation from trace
            trace = getattr(self.mmm, "_trace", None)
            if trace is None:
                logger.debug("Model trace not found for ROI computation")
                return None
            
            channels = self._get_channel_names()
            channel_roi = {}
            
            for ch in channels:
                # Look for beta coefficients
                beta_name = f"beta_{ch}"
                if hasattr(trace, "posterior") and beta_name in trace.posterior:
                    samples = trace.posterior[beta_name].values.flatten()
                    # This is simplified - actual ROI needs contribution / spend
                    mean = float(samples.mean())
                    lower, upper = self._compute_hdi(samples, self.ci_prob)
                    channel_roi[ch] = {"mean": mean, "lower": lower, "upper": upper}
            
            return channel_roi if channel_roi else None
        except Exception as e:
            logger.warning(f"Error computing channel ROI: {e}")
            return None
    
    def _get_component_totals(self) -> dict[str, float] | None:
        """Get total contribution by component."""
        try:
            logger.debug("Retrieving component totals for decomposition")
            if hasattr(self.mmm, "compute_contributions"):
                logger.debug("Using model's compute_contributions method")
                contrib = self.mmm.compute_contributions()
                if hasattr(contrib, "component_totals"):
                    logger.debug("Component totals found")
                    return dict(contrib.component_totals)
        except Exception as e:
            logger.warning(f"Error retrieving component totals: {e}")
            pass
        return None
    
    def _get_component_time_series(self) -> dict[str, np.ndarray] | None:
        """Get component time series for decomposition chart."""
        try:
            logger.debug("Retrieving component time series for decomposition")
            if hasattr(self.mmm, "compute_contributions"):
                logger.debug("Using model's compute_contributions method")
                contrib = self.mmm.compute_contributions()
                if hasattr(contrib, "component_time_series"):
                    return {k: np.array(v) for k, v in contrib.component_time_series.items()}
        except Exception as e:
            logger.warning(f"Error retrieving component time series: {e}")
            pass
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
        """Get saturation curve data for each channel."""
        try:
            if hasattr(self.mmm, "compute_saturation_curves"):
                return self.mmm.compute_saturation_curves()
            
            # Generate from parameters
            channels = self._get_channel_names()
            curves = {}
            
            for ch in channels:
                # Get current spend range
                spend_range = np.linspace(0, 1e6, 100)  # Placeholder
                
                # Try to get saturation parameters
                trace = getattr(self.mmm, "_trace", None)
                if trace is not None and hasattr(trace, "posterior"):
                    # Look for Hill parameters
                    k_name = f"kappa_{ch}"
                    s_name = f"slope_{ch}"
                    
                    if k_name in trace.posterior and s_name in trace.posterior:
                        k = float(trace.posterior[k_name].values.mean())
                        s = float(trace.posterior[s_name].values.mean())
                        
                        # Hill function
                        response = spend_range ** s / (k ** s + spend_range ** s)
                        curves[ch] = {"spend": spend_range, "response": response}
            
            return curves if curves else None
        except Exception:
            return None
    
    def _get_adstock_curves(self) -> dict[str, np.ndarray] | None:
        """Get adstock decay weights for each channel."""
        try:
            if hasattr(self.mmm, "compute_adstock_curves"):
                return self.mmm.compute_adstock_curves()
            
            channels = self._get_channel_names()
            curves = {}
            
            trace = getattr(self.mmm, "_trace", None)
            if trace is not None and hasattr(trace, "posterior"):
                for ch in channels:
                    alpha_name = f"alpha_{ch}"
                    if alpha_name in trace.posterior:
                        alpha = float(trace.posterior[alpha_name].values.mean())
                        l_max = getattr(self.mmm, "adstock_lmax", 8)
                        
                        # Geometric decay
                        lags = np.arange(l_max)
                        weights = alpha ** lags
                        weights = weights / weights.sum()
                        curves[ch] = weights
            
            return curves if curves else None
        except Exception:
            return None
    
    def _get_current_spend(self) -> dict[str, float] | None:
        """Get current spend levels by channel."""
        try:
            if self.panel is not None and hasattr(self.panel, "X_media"):
                channels = self._get_channel_names()
                X_media = self.panel.X_media
                
                if X_media is not None:
                    return {ch: float(X_media[:, i].sum()) for i, ch in enumerate(channels)}
        except Exception:
            pass
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
                return None, None
            
            posterior_samples = {}
            prior_samples = {}
            
            # Get posterior
            if hasattr(trace, "posterior"):
                for var_name in trace.posterior.data_vars:
                    if any(prefix in var_name for prefix in ["beta", "sigma"]):
                        posterior_samples[var_name] = trace.posterior[var_name].values.flatten()
            
            # Get prior if available
            if hasattr(trace, "prior"):
                for var_name in trace.prior.data_vars:
                    if var_name in posterior_samples:
                        prior_samples[var_name] = trace.prior[var_name].values.flatten()
            
            return prior_samples if prior_samples else None, posterior_samples if posterior_samples else None
        except Exception:
            return None, None
    
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
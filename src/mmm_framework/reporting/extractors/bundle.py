"""
MMMDataBundle - Unified data container for MMM report generation.

This module defines the core data structure that all extractors populate
and all report sections consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MMMDataBundle:
    """
    Unified data container for MMM report generation.

    All fields are optional - sections will gracefully skip if data is missing.

    Attributes
    ----------
    dates : array-like, optional
        Time index for the model.
    actual : ndarray, optional
        Observed KPI values (aggregated to period level).
    predicted : dict, optional
        Predictions with "mean", "lower", "upper" keys.
    fit_statistics : dict, optional
        Model fit metrics like "r2", "rmse", "mae", "mape".
    channel_names : list[str], optional
        Names of media channels.
    channel_roi : dict, optional
        Channel-level ROI with uncertainty.
    channel_spend : dict, optional
        Total spend per channel.
    channel_contribution : dict, optional
        Channel contribution with uncertainty.
    component_totals : dict, optional
        Total contribution per component.
    component_time_series : dict, optional
        Time series per component.
    saturation_curves : dict, optional
        Saturation curve data per channel.
    adstock_curves : dict, optional
        Adstock decay weights per channel.
    current_spend : dict, optional
        Current spend level per channel.
    diagnostics : dict, optional
        MCMC diagnostics like divergences, rhat, ess.
    trace_data : dict, optional
        MCMC trace samples for diagnostic plots.
    trace_parameters : list[str], optional
        Parameter names for trace plots.
    prior_samples : dict, optional
        Prior samples for comparison.
    posterior_samples : dict, optional
        Posterior samples for comparison.
    model_specification : dict, optional
        Model configuration details.
    geo_names : list[str], optional
        Geography names for hierarchical models.
    geo_performance : dict, optional
        Geo-level performance metrics.
    geo_roi : dict, optional
        Geo-level ROI by channel.
    geo_contribution : dict, optional
        Geo-level contribution by component.
    actual_by_geo : dict, optional
        Observed values per geo.
    predicted_by_geo : dict, optional
        Predictions per geo with uncertainty.
    fit_statistics_by_geo : dict, optional
        Fit statistics per geo.
    component_time_series_by_geo : dict, optional
        Component time series per geo.
    component_totals_by_geo : dict, optional
        Component totals per geo.
    product_names : list[str], optional
        Product names for multi-product models.
    actual_by_product : dict, optional
        Observed values per product.
    predicted_by_product : dict, optional
        Predictions per product with uncertainty.
    fit_statistics_by_product : dict, optional
        Fit statistics per product.
    component_time_series_by_product : dict, optional
        Component time series per product.
    component_totals_by_product : dict, optional
        Component totals per product.
    mediator_names : list[str], optional
        Mediator names for nested models.
    mediator_pathways : dict, optional
        Effect pathways through mediators.
    mediator_time_series : dict, optional
        Mediator values over time.
    mediator_effects : dict, optional
        Mediator effect estimates.
    total_indirect_effect : dict, optional
        Total indirect effect with uncertainty.
    cannibalization_matrix : dict, optional
        Cross-product cannibalization effects.
    cross_effects : dict, optional
        Cross-outcome effects.
    outcome_correlations : ndarray, optional
        Correlation matrix between outcomes.
    net_product_effects : dict, optional
        Net effects per product after cannibalization.
    total_revenue : float, optional
        Total observed revenue/KPI.
    marketing_attributed_revenue : dict, optional
        Revenue attributed to marketing.
    blended_roi : dict, optional
        Overall blended ROI with uncertainty.
    marketing_contribution_pct : dict, optional
        Marketing contribution as percentage.
    sensitivity_results : dict, optional
        Sensitivity analysis results.
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

    @property
    def has_mediator_data(self) -> bool:
        """Check if mediator pathway data is available."""
        return (
            self.mediator_names is not None
            and len(self.mediator_names) > 0
            and self.mediator_pathways is not None
        )

    @property
    def has_cannibalization_data(self) -> bool:
        """Check if cannibalization data is available."""
        return (
            self.product_names is not None
            and len(self.product_names) > 1
            and self.cannibalization_matrix is not None
        )


__all__ = ["MMMDataBundle"]

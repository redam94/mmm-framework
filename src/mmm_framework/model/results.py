"""
Result containers for BayesianMMM.

This module contains the dataclasses used to return results from
model fitting, prediction, and contribution analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import arviz as az
    import pymc as pm

    from ..data_loader import PanelDataset


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
        import arviz as az
        return az.summary(self.trace, var_names=var_names)

    def plot_trace(self, var_names: list[str] | None = None, **kwargs):
        """Plot trace diagnostics."""
        import arviz as az
        return az.plot_trace(self.trace, var_names=var_names, **kwargs)

    def plot_posterior(self, var_names: list[str] | None = None, **kwargs):
        """Plot posterior distributions."""
        import arviz as az
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
    """
    Container for channel contribution results.

    These results are computed via counterfactual analysis:
    contribution = prediction(all channels) - prediction(channel zeroed out)

    Attributes
    ----------
    channel_contributions : pd.DataFrame
        Per-observation contributions, shape (n_obs, n_channels).
        Index matches the original panel index.
    total_contributions : pd.Series
        Total contribution by channel (summed over time).
    contribution_pct : pd.Series
        Contribution as percentage of total.
    baseline_prediction : np.ndarray
        Prediction with all channels present.
    counterfactual_predictions : dict[str, np.ndarray]
        Counterfactual predictions with each channel zeroed out.
    time_period : tuple[int, int] | None
        Time period used for calculation, if any.
    contribution_hdi_low : pd.Series | None
        Lower HDI for total contributions (if uncertainty computed).
    contribution_hdi_high : pd.Series | None
        Upper HDI for total contributions (if uncertainty computed).
    """

    channel_contributions: pd.DataFrame
    total_contributions: pd.Series
    contribution_pct: pd.Series
    baseline_prediction: np.ndarray
    counterfactual_predictions: dict[str, np.ndarray]
    time_period: tuple[int, int] | None = None
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


__all__ = [
    "MMMResults",
    "PredictionResults",
    "ContributionResults",
    "ComponentDecomposition",
]

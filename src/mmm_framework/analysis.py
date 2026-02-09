"""Analysis utilities for BayesianMMM.

This module provides helper classes for analyzing fitted Bayesian
Marketing Mix Models, including counterfactual analysis, marginal
contributions, and what-if scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .utils import compute_hdi_bounds

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .model import BayesianMMM, PredictionResults, ContributionResults


@dataclass
class MarginalAnalysisResult:
    """Result of marginal contribution analysis.

    Attributes
    ----------
    channel : str
        Channel name.
    current_spend : float
        Current total spend in the period.
    spend_increase : float
        Absolute spend increase.
    spend_increase_pct : float
        Percentage spend increase.
    marginal_contribution : float
        Additional outcome from the spend increase.
    marginal_roas : float
        Return on additional spend.
    """

    channel: str
    current_spend: float
    spend_increase: float
    spend_increase_pct: float
    marginal_contribution: float
    marginal_roas: float


@dataclass
class ScenarioResult:
    """Result of a what-if scenario analysis.

    Attributes
    ----------
    baseline_outcome : float
        Total outcome under baseline scenario.
    scenario_outcome : float
        Total outcome under modified scenario.
    outcome_change : float
        Absolute change in outcome.
    outcome_change_pct : float
        Percentage change in outcome.
    spend_changes : dict[str, dict]
        Spend change details by channel.
    baseline_prediction : NDArray
        Full baseline prediction array.
    scenario_prediction : NDArray
        Full scenario prediction array.
    """

    baseline_outcome: float
    scenario_outcome: float
    outcome_change: float
    outcome_change_pct: float
    spend_changes: dict[str, dict]
    baseline_prediction: NDArray
    scenario_prediction: NDArray


class MMMAnalyzer:
    """Analyzer for fitted BayesianMMM models.

    Provides methods for:
    - Counterfactual contribution analysis
    - Marginal contribution analysis
    - What-if scenario modeling

    Parameters
    ----------
    model : BayesianMMM
        A fitted BayesianMMM model.

    Examples
    --------
    >>> from mmm_framework.analysis import MMMAnalyzer
    >>> analyzer = MMMAnalyzer(fitted_model)
    >>> contributions = analyzer.compute_counterfactual_contributions()
    >>> print(contributions.summary())

    >>> marginal = analyzer.compute_marginal_contributions(spend_increase_pct=10)
    >>> print(marginal)
    """

    def __init__(self, model: BayesianMMM):
        self._model = model
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate that the model is fitted."""
        if self._model._trace is None:
            raise ValueError("Model not fitted. Call model.fit() first.")

    @property
    def channel_names(self) -> list[str]:
        """Get channel names from the model."""
        return self._model.channel_names

    @property
    def n_obs(self) -> int:
        """Get number of observations."""
        return self._model.n_obs

    def get_time_mask(
        self,
        time_period: tuple[int, int] | None,
    ) -> NDArray[np.bool_]:
        """Get time mask for filtering observations.

        Parameters
        ----------
        time_period : tuple[int, int] | None
            (start_idx, end_idx) inclusive, or None for all.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask array.
        """
        return self._model._get_time_mask(time_period)

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

        This is a convenience wrapper around the model's method.

        Parameters
        ----------
        time_period : tuple[int, int], optional
            Time period (start_idx, end_idx) for calculation.
        channels : list[str], optional
            Channels to analyze. If None, uses all.
        compute_uncertainty : bool
            Whether to compute HDI for contributions.
        hdi_prob : float
            HDI probability mass.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ContributionResults
            Contribution results container.
        """
        return self._model.compute_counterfactual_contributions(
            time_period=time_period,
            channels=channels,
            compute_uncertainty=compute_uncertainty,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
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

        This is a convenience wrapper around the model's method.

        Parameters
        ----------
        spend_increase_pct : float
            Percentage increase in spend to simulate.
        time_period : tuple[int, int], optional
            Time period for calculation.
        channels : list[str], optional
            Channels to analyze. If None, uses all.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Marginal contribution analysis.
        """
        return self._model.compute_marginal_contributions(
            spend_increase_pct=spend_increase_pct,
            time_period=time_period,
            channels=channels,
            random_seed=random_seed,
        )

    def what_if_scenario(
        self,
        spend_changes: dict[str, float],
        time_period: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> dict:
        """
        Run a what-if scenario with custom spend changes.

        This is a convenience wrapper around the model's method.

        Parameters
        ----------
        spend_changes : dict[str, float]
            Mapping of channel names to spend multipliers.
        time_period : tuple[int, int], optional
            Time period for calculation.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Scenario analysis results.
        """
        return self._model.what_if_scenario(
            spend_changes=spend_changes,
            time_period=time_period,
            random_seed=random_seed,
        )

    def compute_channel_roi(
        self,
        time_period: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute return on investment for each channel.

        ROI = Total Contribution / Total Spend

        Parameters
        ----------
        time_period : tuple[int, int], optional
            Time period for calculation.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            ROI analysis by channel.
        """
        # Get contributions
        contributions = self.compute_counterfactual_contributions(
            time_period=time_period,
            compute_uncertainty=True,
            random_seed=random_seed,
        )

        # Get time mask
        time_mask = self.get_time_mask(time_period)

        # Calculate spend and ROI
        results = []
        for channel in self.channel_names:
            ch_idx = self.channel_names.index(channel)
            spend = self._model.X_media_raw[time_mask, ch_idx].sum()
            contribution = contributions.total_contributions[channel]

            roi = contribution / spend if spend > 0 else 0

            result = {
                "Channel": channel,
                "Total Spend": spend,
                "Total Contribution": contribution,
                "Contribution %": contributions.contribution_pct[channel],
                "ROI": roi,
            }

            # Add HDI if available
            if contributions.contribution_hdi_low is not None:
                result["Contribution HDI Low"] = contributions.contribution_hdi_low[
                    channel
                ]
                result["Contribution HDI High"] = contributions.contribution_hdi_high[
                    channel
                ]

            results.append(result)

        return pd.DataFrame(results)

    def compute_saturation_curves(
        self,
        channel: str,
        spend_range: tuple[float, float] | None = None,
        n_points: int = 50,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute saturation curve for a channel.

        Shows how outcome changes across different spend levels.

        Parameters
        ----------
        channel : str
            Channel to analyze.
        spend_range : tuple[float, float], optional
            (min, max) spend range. If None, uses 0 to 2x current max.
        n_points : int
            Number of points on the curve.
        random_seed : int, optional
            Random seed.

        Returns
        -------
        pd.DataFrame
            Saturation curve data.
        """
        if channel not in self.channel_names:
            raise ValueError(f"Unknown channel: {channel}")

        ch_idx = self.channel_names.index(channel)
        current_max = self._model.X_media_raw[:, ch_idx].max()

        if spend_range is None:
            spend_range = (0.0, current_max * 2)

        spend_levels = np.linspace(spend_range[0], spend_range[1], n_points)

        results = []
        baseline_pred = self._model.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean.sum()

        for spend_level in spend_levels:
            # Create scenario with flat spend at this level
            X_media_scenario = self._model.X_media_raw.copy()
            X_media_scenario[:, ch_idx] = spend_level

            scenario_pred = self._model.predict(
                X_media=X_media_scenario,
                random_seed=random_seed,
            )
            scenario_total = scenario_pred.y_pred_mean.sum()

            # Contribution at this spend level
            contribution = (
                scenario_total
                - baseline_total
                + self._model.X_media_raw[:, ch_idx].sum() * 0
            )  # Placeholder

            results.append(
                {
                    "Spend Level": spend_level,
                    "Total Outcome": scenario_total,
                    "Relative to Baseline": scenario_total - baseline_total,
                }
            )

        return pd.DataFrame(results)


def compute_contribution_summary(
    contributions: ContributionResults,
) -> pd.DataFrame:
    """
    Create a summary DataFrame from contribution results.

    Parameters
    ----------
    contributions : ContributionResults
        Contribution analysis results.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    return contributions.summary()


def compute_period_contributions(
    contributions: ContributionResults,
    periods: list[tuple[int, int]],
    period_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute contributions for multiple time periods.

    Parameters
    ----------
    contributions : ContributionResults
        Base contribution results.
    periods : list[tuple[int, int]]
        List of (start, end) period tuples.
    period_names : list[str], optional
        Names for each period.

    Returns
    -------
    pd.DataFrame
        Contributions by period.
    """
    if period_names is None:
        period_names = [f"Period {i+1}" for i in range(len(periods))]

    results = []
    for name, (start, end) in zip(period_names, periods):
        # Filter to this period
        mask = (
            contributions.channel_contributions.index.get_level_values(0) >= start
        ) & (contributions.channel_contributions.index.get_level_values(0) <= end)

        period_totals = contributions.channel_contributions[mask].sum()

        for channel, total in period_totals.items():
            results.append(
                {
                    "Period": name,
                    "Channel": channel,
                    "Contribution": total,
                }
            )

    df = pd.DataFrame(results)
    return df.pivot(index="Channel", columns="Period", values="Contribution")

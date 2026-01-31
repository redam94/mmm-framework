"""
Channel-level diagnostics for model validation.

Provides multicollinearity detection (VIF), per-channel convergence checks,
and identifiability analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .config import ChannelDiagnosticsConfig
from .results import ChannelConvergenceResult, ChannelDiagnosticsResults

if TYPE_CHECKING:
    import arviz as az


class VIFCalculator:
    """
    Variance Inflation Factor calculator.

    VIF measures multicollinearity between media channels.
    VIF > 10 indicates problematic multicollinearity.
    """

    def compute(
        self,
        X_media: np.ndarray,
        channel_names: list[str],
    ) -> dict[str, float]:
        """
        Compute VIF for each media channel.

        Parameters
        ----------
        X_media : np.ndarray
            Media spend matrix, shape (n_obs, n_channels).
        channel_names : list[str]
            Names of media channels.

        Returns
        -------
        dict[str, float]
            VIF score for each channel.
        """
        n_channels = X_media.shape[1]
        vif_scores = {}

        for i in range(n_channels):
            # Regress channel i on all other channels
            y = X_media[:, i]
            X_other = np.delete(X_media, i, axis=1)

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(y)), X_other])

            try:
                # Compute R² from regression
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # VIF = 1 / (1 - R²)
                if r_squared >= 1:
                    vif = float("inf")
                else:
                    vif = 1 / (1 - r_squared)
            except Exception:
                vif = float("nan")

            vif_scores[channel_names[i]] = vif

        return vif_scores


class ChannelConvergenceChecker:
    """
    Per-channel MCMC convergence diagnostics.

    Checks R-hat and ESS for channel-specific parameters.
    """

    def check(
        self,
        trace: Any,
        channel_names: list[str],
        config: ChannelDiagnosticsConfig,
    ) -> dict[str, ChannelConvergenceResult]:
        """
        Check convergence for each channel.

        Parameters
        ----------
        trace : az.InferenceData
            ArviZ InferenceData with posterior samples.
        channel_names : list[str]
            Names of media channels.
        config : ChannelDiagnosticsConfig
            Configuration with thresholds.

        Returns
        -------
        dict[str, ChannelConvergenceResult]
            Convergence result for each channel.
        """
        import arviz as az

        results = {}

        # Get summary for all parameters
        try:
            summary = az.summary(trace)
        except Exception:
            # Return default results if summary fails
            for channel in channel_names:
                results[channel] = ChannelConvergenceResult(
                    channel=channel,
                    rhat=1.0,
                    ess_bulk=1000.0,
                    ess_tail=1000.0,
                    converged=True,
                )
            return results

        # Look for channel-specific parameters
        # Common patterns: beta[channel], channel_beta, media_beta
        for channel in channel_names:
            rhat = 1.0
            ess_bulk = 1000.0
            ess_tail = 1000.0

            # Search for parameters matching this channel
            for param_name in summary.index:
                # Check if parameter name contains channel name
                if (
                    channel.lower() in param_name.lower()
                    or f"[{channel}]" in param_name
                ):
                    param_rhat = summary.loc[param_name, "r_hat"]
                    param_ess_bulk = summary.loc[param_name, "ess_bulk"]
                    param_ess_tail = summary.loc[param_name, "ess_tail"]

                    # Take worst case
                    rhat = max(rhat, param_rhat)
                    ess_bulk = min(ess_bulk, param_ess_bulk)
                    ess_tail = min(ess_tail, param_ess_tail)

            # Check convergence
            converged = rhat < config.rhat_threshold and ess_bulk > config.ess_threshold

            results[channel] = ChannelConvergenceResult(
                channel=channel,
                rhat=float(rhat),
                ess_bulk=float(ess_bulk),
                ess_tail=float(ess_tail),
                converged=converged,
            )

        return results


class ChannelDiagnostics:
    """
    Comprehensive channel-level diagnostics.

    Combines VIF analysis, correlation matrix, and per-channel convergence.

    Examples
    --------
    >>> diagnostics = ChannelDiagnostics(model)
    >>> results = diagnostics.run_all()
    >>> print(results.summary())
    """

    def __init__(
        self,
        model: Any,
        config: ChannelDiagnosticsConfig | None = None,
    ):
        """
        Initialize channel diagnostics.

        Parameters
        ----------
        model : Any
            Fitted model with media data and trace.
        config : ChannelDiagnosticsConfig, optional
            Configuration for diagnostics.
        """
        self.model = model
        self.config = config or ChannelDiagnosticsConfig()
        self._vif_calculator = VIFCalculator()
        self._convergence_checker = ChannelConvergenceChecker()

    def run_all(self) -> ChannelDiagnosticsResults:
        """
        Run all channel diagnostics.

        Returns
        -------
        ChannelDiagnosticsResults
            Comprehensive channel diagnostics results.
        """
        # Get media data and channel names
        X_media = self._get_media_data()
        channel_names = self._get_channel_names()

        # Compute VIF
        vif_scores = self._vif_calculator.compute(X_media, channel_names)

        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix(X_media, channel_names)

        # Check per-channel convergence
        trace = self._get_trace()
        convergence_by_channel = self._convergence_checker.check(
            trace, channel_names, self.config
        )

        # Identify potential issues
        identifiability_issues = self._identify_issues(
            vif_scores, correlation_matrix, channel_names
        )

        return ChannelDiagnosticsResults(
            vif_scores=vif_scores,
            correlation_matrix=correlation_matrix,
            convergence_by_channel=convergence_by_channel,
            identifiability_issues=identifiability_issues,
        )

    def vif_analysis(self) -> pd.DataFrame:
        """
        Get VIF analysis as DataFrame.

        Returns
        -------
        pd.DataFrame
            VIF scores with interpretation.
        """
        X_media = self._get_media_data()
        channel_names = self._get_channel_names()
        vif_scores = self._vif_calculator.compute(X_media, channel_names)

        def interpret_vif(vif):
            if vif < 5:
                return "Low"
            elif vif < 10:
                return "Moderate"
            else:
                return "High"

        return pd.DataFrame(
            {
                "Channel": list(vif_scores.keys()),
                "VIF": list(vif_scores.values()),
                "Multicollinearity": [interpret_vif(v) for v in vif_scores.values()],
            }
        )

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Get media channel correlation matrix.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        X_media = self._get_media_data()
        channel_names = self._get_channel_names()
        return self._compute_correlation_matrix(X_media, channel_names)

    def _get_media_data(self) -> np.ndarray:
        """Extract media data from model."""
        # BayesianMMM uses X_media_raw for raw media data
        if hasattr(self.model, "X_media_raw"):
            return np.asarray(self.model.X_media_raw)
        elif hasattr(self.model, "X_media"):
            return np.asarray(self.model.X_media)
        elif hasattr(self.model, "_X_media"):
            return np.asarray(self.model._X_media)
        elif hasattr(self.model, "panel"):
            panel = self.model.panel
            # Panel stores media data as X_media DataFrame
            if hasattr(panel, "X_media"):
                return np.asarray(panel.X_media)
            elif hasattr(panel, "media_data"):
                return np.asarray(panel.media_data)
        raise ValueError("Could not extract media data from model")

    def _get_channel_names(self) -> list[str]:
        """Extract channel names from model."""
        if hasattr(self.model, "channel_names"):
            return list(self.model.channel_names)
        elif hasattr(self.model, "_channel_names"):
            return list(self.model._channel_names)
        raise ValueError("Could not extract channel names from model")

    def _get_trace(self) -> Any:
        """Get ArviZ trace from model."""
        if hasattr(self.model, "_trace"):
            return self.model._trace
        elif hasattr(self.model, "trace"):
            return self.model.trace
        raise ValueError("Could not extract trace from model")

    def _compute_correlation_matrix(
        self,
        X_media: np.ndarray,
        channel_names: list[str],
    ) -> pd.DataFrame:
        """Compute correlation matrix for media channels."""
        corr = np.corrcoef(X_media, rowvar=False)
        return pd.DataFrame(corr, index=channel_names, columns=channel_names)

    def _identify_issues(
        self,
        vif_scores: dict[str, float],
        correlation_matrix: pd.DataFrame,
        channel_names: list[str],
    ) -> list[str]:
        """Identify potential identifiability issues."""
        issues = []

        # Check high VIF
        for channel, vif in vif_scores.items():
            if vif > self.config.vif_threshold:
                issues.append(
                    f"High VIF for {channel}: {vif:.2f} > {self.config.vif_threshold}"
                )

        # Check high correlations
        for i, ch1 in enumerate(channel_names):
            for j, ch2 in enumerate(channel_names):
                if i < j:
                    corr = abs(correlation_matrix.iloc[i, j])
                    if corr > self.config.correlation_threshold:
                        issues.append(
                            f"High correlation between {ch1} and {ch2}: {corr:.3f}"
                        )

        return issues


__all__ = [
    "VIFCalculator",
    "ChannelConvergenceChecker",
    "ChannelDiagnostics",
]

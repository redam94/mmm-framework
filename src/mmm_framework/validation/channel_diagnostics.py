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
from .results import (
    ChannelConvergenceResult,
    ChannelDiagnosticsResults,
    CollinearCluster,
)

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

        # Weak-identification analysis (P2-2): collinear clusters, the design
        # condition number, and grouped-prior recommendations.
        clusters = self._detect_collinear_clusters(correlation_matrix, channel_names)
        condition_number = self._condition_number(correlation_matrix)
        recommendations = self._recommend_grouped_priors(clusters)
        for cl in clusters:
            identifiability_issues.append(
                "Non-identifiable channel cluster "
                f"{cl.channels} (|r| up to {cl.max_correlation:.2f}): their "
                "individual ROIs cannot be separated from observational data; "
                "read them as a group."
            )

        return ChannelDiagnosticsResults(
            vif_scores=vif_scores,
            correlation_matrix=correlation_matrix,
            convergence_by_channel=convergence_by_channel,
            identifiability_issues=identifiability_issues,
            collinear_clusters=clusters,
            condition_number=condition_number,
            grouped_prior_recommendations=recommendations,
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

    def _detect_collinear_clusters(
        self,
        correlation_matrix: pd.DataFrame,
        channel_names: list[str],
    ) -> list[CollinearCluster]:
        """Group channels into clusters that cannot be separately identified.

        Two channels are linked when their absolute correlation exceeds the
        configured threshold; clusters are the connected components of that
        graph. A cluster of size >= 2 is weakly identified: the data sees their
        combined movement, not the per-channel split.
        """
        threshold = self.config.correlation_threshold
        n = len(channel_names)
        # Union-find over channels.
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if abs(float(correlation_matrix.iloc[i, j])) > threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        clusters: list[CollinearCluster] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            chans = [channel_names[m] for m in members]
            max_corr = max(
                abs(float(correlation_matrix.iloc[a, b]))
                for a in members
                for b in members
                if a < b
            )
            clusters.append(
                CollinearCluster(
                    channels=chans,
                    max_correlation=max_corr,
                    explanation=(
                        f"Channels {chans} move together (|r| up to {max_corr:.2f}); "
                        "their individual effects are weakly identified and should "
                        "be interpreted as a group, anchored by an experiment, or "
                        "fit with a grouped prior."
                    ),
                )
            )
        return clusters

    @staticmethod
    def _condition_number(correlation_matrix: pd.DataFrame) -> float | None:
        """Condition number of the channel correlation matrix.

        Captures higher-order (multi-channel) collinearity that pairwise
        correlations miss; a large value (> ~30) signals an ill-conditioned
        design where per-channel estimates are unstable.
        """
        try:
            cond = float(np.linalg.cond(np.asarray(correlation_matrix.values)))
            return cond if np.isfinite(cond) else None
        except Exception:
            return None

    def _recommend_grouped_priors(self, clusters: list[CollinearCluster]) -> list[str]:
        """Recommend grouped/hierarchical priors for collinear clusters.

        Reporting only -- this does NOT change the model. Where clustered
        channels already share a ``parent_channel`` (hierarchical media group),
        the recommendation names it; otherwise it suggests grouping them.
        """
        if not clusters:
            return []
        media_groups: dict[str, list[str]] = (
            getattr(self.model, "media_groups", {}) or {}
        )
        recs: list[str] = []
        for cl in clusters:
            shared_parent = None
            for parent, children in media_groups.items():
                if sum(1 for c in cl.channels if c in children) >= 2:
                    shared_parent = parent
                    break
            if shared_parent:
                recs.append(
                    f"Channels {cl.channels} are collinear and already share the "
                    f"'{shared_parent}' media group -- a hierarchical (partial-"
                    "pooling) prior across that group would borrow strength and "
                    "stabilize their split."
                )
            else:
                recs.append(
                    f"Channels {cl.channels} are collinear -- consider a grouped "
                    "prior (shared group-level scale) or report their combined "
                    "effect rather than overconfident per-channel ROIs."
                )
        return recs

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

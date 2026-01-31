"""
Posterior predictive checks for model validation.

Provides comprehensive posterior predictive checking to assess model adequacy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .config import PPCConfig
from .results import PPCCheckResult, PPCResults

if TYPE_CHECKING:
    import arviz as az


@runtime_checkable
class PPCCheck(Protocol):
    """Protocol for individual PPC checks."""

    name: str
    description: str

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float,
    ) -> PPCCheckResult:
        """Compute the check result."""
        ...


class MeanCheck:
    """Check if posterior mean matches observed mean."""

    name = "Mean"
    description = "Compares observed mean to replicated data means"

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float = 0.05,
    ) -> PPCCheckResult:
        """
        Compute mean posterior predictive check.

        Parameters
        ----------
        y_obs : np.ndarray
            Observed data.
        y_rep : np.ndarray
            Replicated data, shape (n_samples, n_obs).
        significance_level : float
            Significance level for pass/fail.

        Returns
        -------
        PPCCheckResult
            Check result.
        """
        obs_mean = np.mean(y_obs)
        rep_means = np.mean(y_rep, axis=1)
        rep_mean = np.mean(rep_means)
        rep_std = np.std(rep_means)

        # Bayesian p-value: proportion of replicated means more extreme
        p_value = np.mean(rep_means >= obs_mean)
        p_value = min(p_value, 1 - p_value) * 2  # Two-tailed

        # Pass if p-value is not extreme (between 0.05 and 0.95)
        passed = significance_level < p_value < (1 - significance_level)

        return PPCCheckResult(
            check_name=self.name,
            observed_statistic=obs_mean,
            replicated_mean=rep_mean,
            replicated_std=rep_std,
            p_value=p_value,
            passed=passed,
            description=self.description,
        )


class VarianceCheck:
    """Check if posterior variance matches observed variance."""

    name = "Variance"
    description = "Compares observed variance to replicated data variances"

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float = 0.05,
    ) -> PPCCheckResult:
        """Compute variance posterior predictive check."""
        obs_var = np.var(y_obs)
        rep_vars = np.var(y_rep, axis=1)
        rep_mean = np.mean(rep_vars)
        rep_std = np.std(rep_vars)

        p_value = np.mean(rep_vars >= obs_var)
        p_value = min(p_value, 1 - p_value) * 2

        passed = significance_level < p_value < (1 - significance_level)

        return PPCCheckResult(
            check_name=self.name,
            observed_statistic=obs_var,
            replicated_mean=rep_mean,
            replicated_std=rep_std,
            p_value=p_value,
            passed=passed,
            description=self.description,
        )


class SkewnessCheck:
    """Check if posterior skewness matches observed skewness."""

    name = "Skewness"
    description = "Compares observed skewness to replicated data"

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float = 0.05,
    ) -> PPCCheckResult:
        """Compute skewness posterior predictive check."""
        from scipy import stats

        obs_skew = stats.skew(y_obs)
        rep_skews = np.array([stats.skew(y_rep[i, :]) for i in range(y_rep.shape[0])])
        rep_mean = np.mean(rep_skews)
        rep_std = np.std(rep_skews)

        p_value = np.mean(rep_skews >= obs_skew)
        p_value = min(p_value, 1 - p_value) * 2

        passed = significance_level < p_value < (1 - significance_level)

        return PPCCheckResult(
            check_name=self.name,
            observed_statistic=obs_skew,
            replicated_mean=rep_mean,
            replicated_std=rep_std,
            p_value=p_value,
            passed=passed,
            description=self.description,
        )


class AutocorrelationCheck:
    """Check if temporal autocorrelation structure is reproduced."""

    name = "Autocorrelation"
    description = "Compares lag-1 autocorrelation of observed vs replicated"

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float = 0.05,
    ) -> PPCCheckResult:
        """Compute autocorrelation posterior predictive check."""

        def lag1_autocorr(x):
            """Compute lag-1 autocorrelation."""
            if len(x) < 2:
                return 0.0
            return np.corrcoef(x[:-1], x[1:])[0, 1]

        obs_acf = lag1_autocorr(y_obs)
        rep_acfs = np.array([lag1_autocorr(y_rep[i, :]) for i in range(y_rep.shape[0])])

        # Handle NaN values
        rep_acfs = rep_acfs[~np.isnan(rep_acfs)]
        if len(rep_acfs) == 0:
            rep_mean = 0.0
            rep_std = 0.0
            p_value = 0.5
        else:
            rep_mean = np.mean(rep_acfs)
            rep_std = np.std(rep_acfs)
            p_value = np.mean(rep_acfs >= obs_acf)
            p_value = min(p_value, 1 - p_value) * 2

        passed = significance_level < p_value < (1 - significance_level)

        return PPCCheckResult(
            check_name=self.name,
            observed_statistic=obs_acf if not np.isnan(obs_acf) else 0.0,
            replicated_mean=rep_mean,
            replicated_std=rep_std,
            p_value=p_value,
            passed=passed,
            description=self.description,
        )


class ExtremesCheck:
    """Check if extreme values (min/max) are captured."""

    name = "Extremes"
    description = "Checks if min and max are within replicated range"

    def compute(
        self,
        y_obs: np.ndarray,
        y_rep: np.ndarray,
        significance_level: float = 0.05,
    ) -> PPCCheckResult:
        """Compute extremes posterior predictive check."""
        obs_max = np.max(y_obs)
        obs_min = np.min(y_obs)
        obs_range = obs_max - obs_min

        rep_maxs = np.max(y_rep, axis=1)
        rep_mins = np.min(y_rep, axis=1)
        rep_ranges = rep_maxs - rep_mins

        rep_mean = np.mean(rep_ranges)
        rep_std = np.std(rep_ranges)

        # P-value based on range comparison
        p_value = np.mean(rep_ranges >= obs_range)
        p_value = min(p_value, 1 - p_value) * 2

        # Also check if observed extremes fall within replicated extremes
        max_captured = np.mean(rep_maxs >= obs_max)
        min_captured = np.mean(rep_mins <= obs_min)

        # Pass if range is similar AND extremes are captured reasonably
        passed = (
            significance_level < p_value < (1 - significance_level)
            and max_captured > 0.01
            and min_captured > 0.01
        )

        return PPCCheckResult(
            check_name=self.name,
            observed_statistic=obs_range,
            replicated_mean=rep_mean,
            replicated_std=rep_std,
            p_value=p_value,
            passed=passed,
            description=self.description,
        )


# Registry of available checks
_CHECK_REGISTRY: dict[str, type] = {
    "mean": MeanCheck,
    "variance": VarianceCheck,
    "skewness": SkewnessCheck,
    "autocorrelation": AutocorrelationCheck,
    "extremes": ExtremesCheck,
}


class PPCValidator:
    """
    Posterior predictive check orchestrator.

    Runs a suite of posterior predictive checks to assess model adequacy.

    Examples
    --------
    >>> validator = PPCValidator(results)
    >>> ppc_results = validator.run()
    >>> print(ppc_results.summary())
    """

    def __init__(
        self,
        model: Any,
        config: PPCConfig | None = None,
    ):
        """
        Initialize PPC validator.

        Parameters
        ----------
        model : Any
            Fitted model with trace and predict method.
        config : PPCConfig, optional
            Configuration for PPC checks.
        """
        self.model = model
        self.config = config or PPCConfig()

        # Build list of checks
        self._checks: list[PPCCheck] = []
        for check_name in self.config.checks:
            if check_name.lower() in _CHECK_REGISTRY:
                self._checks.append(_CHECK_REGISTRY[check_name.lower()]())

    def run(
        self,
        y_obs: np.ndarray | None = None,
        y_rep: np.ndarray | None = None,
        random_seed: int | None = None,
    ) -> PPCResults:
        """
        Run posterior predictive checks.

        Parameters
        ----------
        y_obs : np.ndarray, optional
            Observed data. If None, extracted from model.
        y_rep : np.ndarray, optional
            Replicated data. If None, generated from posterior.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        PPCResults
            Results of all PPC checks.
        """
        # Get observed data
        if y_obs is None:
            y_obs = self._get_observed_data()

        # Generate replicated data if not provided
        if y_rep is None:
            y_rep = self._generate_replicated_data(random_seed)

        # Ensure y_rep is 2D (n_samples, n_obs)
        if y_rep.ndim == 1:
            y_rep = y_rep.reshape(1, -1)

        # Subsample if needed
        if y_rep.shape[0] > self.config.n_samples:
            rng = np.random.default_rng(random_seed)
            indices = rng.choice(
                y_rep.shape[0], size=self.config.n_samples, replace=False
            )
            y_rep = y_rep[indices]

        # Run all checks
        check_results = []
        for check in self._checks:
            result = check.compute(
                y_obs=y_obs,
                y_rep=y_rep,
                significance_level=self.config.significance_level,
            )
            check_results.append(result)

        return PPCResults(
            checks=check_results,
            y_obs=y_obs,
            y_rep=y_rep,
        )

    def _get_observed_data(self) -> np.ndarray:
        """Extract observed data from model on original scale."""
        from loguru import logger

        # Prefer raw (original scale) data for interpretability
        if hasattr(self.model, "y_raw"):
            data = np.asarray(self.model.y_raw).flatten()
            logger.debug(f"PPC: Using y_raw, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        # Fall back to panel data (original scale)
        elif hasattr(self.model, "panel"):
            panel = self.model.panel
            if hasattr(panel, "y"):
                data = np.asarray(panel.y).flatten()
                logger.debug(f"PPC: Using panel.y, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                return data
        # Last resort: standardized data (less interpretable)
        elif hasattr(self.model, "y"):
            data = np.asarray(self.model.y).flatten()
            logger.debug(f"PPC: Using y (standardized), shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        elif hasattr(self.model, "_y"):
            data = np.asarray(self.model._y).flatten()
            logger.debug(f"PPC: Using _y, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        raise ValueError("Could not extract observed data from model")

    def _generate_replicated_data(
        self,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Generate replicated data from posterior predictive on original scale.

        NOTE: For proper PPC, we need SAMPLES from the posterior predictive distribution,
        not the observed data. The model's posterior `y_obs_scaled` is just the observed
        data scaled back - it's NOT replicated samples.

        We prioritize:
        1. predict() method which generates new samples
        2. posterior_predictive group (from sample_posterior_predictive)
        3. Fallback to sample_posterior_predictive() call
        """
        from loguru import logger

        # Get scaling parameters for converting to original scale
        y_mean = getattr(self.model, "y_mean", 0.0)
        y_std = getattr(self.model, "y_std", 1.0)
        logger.debug(f"PPC replicated: y_mean={y_mean:.2f}, y_std={y_std:.2f}")

        # Try using predict method FIRST (generates new samples on original scale)
        if hasattr(self.model, "predict"):
            try:
                logger.debug("PPC replicated: Trying predict() method")
                pred_result = self.model.predict(return_original_scale=True)
                if hasattr(pred_result, "y_pred_samples") and pred_result.y_pred_samples is not None:
                    samples = np.asarray(pred_result.y_pred_samples)
                    if samples.size > 0:
                        logger.debug(f"PPC replicated: predict() returned shape={samples.shape}, range=[{samples.min():.2f}, {samples.max():.2f}]")
                        return samples
            except Exception as e:
                logger.debug(f"PPC replicated: predict() failed: {e}")

        # Check if posterior_predictive already exists in trace (from a previous PPC call)
        if hasattr(self.model, "_trace"):
            trace = self.model._trace
            if hasattr(trace, "posterior_predictive") and trace.posterior_predictive is not None:
                pp = trace.posterior_predictive
                logger.debug(f"PPC replicated: posterior_predictive has vars: {list(pp.data_vars)}")
                # Look for y_obs (standardized replicated samples)
                for var_name in ["y_obs", "y", "obs", "likelihood"]:
                    if var_name in pp:
                        data = pp[var_name].values
                        logger.debug(f"PPC replicated: Found {var_name} with shape={data.shape}")
                        # Flatten chains and draws
                        if data.ndim > 2:
                            data = data.reshape(-1, data.shape[-1])
                        # Convert to original scale
                        data = data * y_std + y_mean
                        logger.debug(f"PPC replicated: Using {var_name}, final shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                        return data

        # Fallback: explicitly sample from posterior predictive
        if hasattr(self.model, "_trace") and hasattr(self.model, "model"):
            import pymc as pm

            logger.debug("PPC replicated: Sampling from posterior predictive")
            with self.model.model:
                ppc = pm.sample_posterior_predictive(
                    self.model._trace,
                    random_seed=random_seed,
                    var_names=["y_obs"],
                )
                # Get y_obs from posterior predictive and convert to original scale
                if "y_obs" in ppc.posterior_predictive:
                    data = ppc.posterior_predictive["y_obs"].values
                    if data.ndim > 2:
                        data = data.reshape(-1, data.shape[-1])
                    # Convert to original scale
                    data = data * y_std + y_mean
                    logger.debug(f"PPC replicated: Sampled shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                    return data

        raise ValueError("Could not generate replicated data from model")


__all__ = [
    "PPCCheck",
    "MeanCheck",
    "VarianceCheck",
    "SkewnessCheck",
    "AutocorrelationCheck",
    "ExtremesCheck",
    "PPCValidator",
]

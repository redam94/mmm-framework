"""
Residual diagnostics for model validation.

Provides comprehensive residual analysis to assess model adequacy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .config import ResidualConfig
from .helpers.statistical_tests import (
    breusch_pagan_test,
    compute_acf,
    compute_pacf,
    durbin_watson_test,
    jarque_bera_test,
    ljung_box_test,
    shapiro_wilk_test,
)
from .results import ResidualDiagnosticsResults, TestResult

if TYPE_CHECKING:
    pass


class ResidualDiagnostics:
    """
    Comprehensive residual analysis.

    Performs statistical tests on model residuals to check for
    autocorrelation, heteroscedasticity, and non-normality.

    Examples
    --------
    >>> diagnostics = ResidualDiagnostics(model, results)
    >>> diag_results = diagnostics.run_all()
    >>> print(diag_results.summary())
    """

    def __init__(
        self,
        model: Any,
        config: ResidualConfig | None = None,
    ):
        """
        Initialize residual diagnostics.

        Parameters
        ----------
        model : Any
            Fitted model with trace.
        config : ResidualConfig, optional
            Configuration for residual tests.
        """
        self.model = model
        self.config = config or ResidualConfig()

        # Test registry
        self._test_registry = {
            "durbin_watson": self._run_durbin_watson,
            "ljung_box": self._run_ljung_box,
            "breusch_pagan": self._run_breusch_pagan,
            "shapiro_wilk": self._run_shapiro_wilk,
            "jarque_bera": self._run_jarque_bera,
        }

    def run_all(
        self,
        residuals: np.ndarray | None = None,
        fitted_values: np.ndarray | None = None,
    ) -> ResidualDiagnosticsResults:
        """
        Run all configured residual diagnostic tests.

        Parameters
        ----------
        residuals : np.ndarray, optional
            Model residuals. If None, computed from model.
        fitted_values : np.ndarray, optional
            Fitted values. If None, computed from model.

        Returns
        -------
        ResidualDiagnosticsResults
            Results of all diagnostic tests.
        """
        # Compute residuals and fitted values if not provided
        if residuals is None or fitted_values is None:
            computed_residuals, computed_fitted = self._compute_residuals()
            residuals = residuals if residuals is not None else computed_residuals
            fitted_values = (
                fitted_values if fitted_values is not None else computed_fitted
            )

        # Run all configured tests
        test_results: list[TestResult] = []
        for test_name in self.config.tests:
            if test_name.lower() in self._test_registry:
                result = self._test_registry[test_name.lower()](
                    residuals, fitted_values
                )
                test_results.append(result)

        # Compute ACF and PACF
        acf_values = compute_acf(residuals, max_lag=self.config.max_lag)
        pacf_values = compute_pacf(residuals, max_lag=self.config.max_lag)

        return ResidualDiagnosticsResults(
            test_results=test_results,
            residuals=residuals,
            fitted_values=fitted_values,
            acf_values=acf_values,
            pacf_values=pacf_values,
        )

    def _compute_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute residuals and fitted values from model on original scale."""
        # Get observed data (original scale)
        y_obs = self._get_observed_data()

        # Get predicted values (posterior mean, original scale)
        y_pred = self._get_predictions()

        # Compute residuals
        residuals = y_obs - y_pred

        return residuals, y_pred

    def _get_observed_data(self) -> np.ndarray:
        """Extract observed data from model on original scale."""
        from loguru import logger

        # Prefer raw (original scale) data for interpretability
        if hasattr(self.model, "y_raw"):
            data = np.asarray(self.model.y_raw).flatten()
            logger.debug(f"Residual: Using y_raw, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        # Fall back to panel data (original scale)
        elif hasattr(self.model, "panel"):
            panel = self.model.panel
            if hasattr(panel, "y"):
                data = np.asarray(panel.y).flatten()
                logger.debug(f"Residual: Using panel.y, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                return data
        # Last resort: standardized data
        elif hasattr(self.model, "y"):
            data = np.asarray(self.model.y).flatten()
            logger.debug(f"Residual: Using y (standardized), shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        elif hasattr(self.model, "_y"):
            data = np.asarray(self.model._y).flatten()
            logger.debug(f"Residual: Using _y, shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
            return data
        raise ValueError("Could not extract observed data from model")

    def _get_predictions(self) -> np.ndarray:
        """Get posterior mean predictions from model on original scale."""
        from loguru import logger

        # Get scaling parameters
        y_mean = getattr(self.model, "y_mean", 0.0)
        y_std = getattr(self.model, "y_std", 1.0)
        logger.debug(f"Residual predictions: y_mean={y_mean:.2f}, y_std={y_std:.2f}")

        # Try using predict method first (returns original scale)
        if hasattr(self.model, "predict"):
            try:
                pred_result = self.model.predict(return_original_scale=True)
                if hasattr(pred_result, "y_pred_mean"):
                    data = np.asarray(pred_result.y_pred_mean).flatten()
                    logger.debug(f"Residual predictions: Using predict(), shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")
                    return data
            except Exception as e:
                logger.debug(f"Residual predictions: predict() failed: {e}")

        # Try to get from trace
        if hasattr(self.model, "_trace"):
            trace = self.model._trace
            if hasattr(trace, "posterior"):
                posterior = trace.posterior
                logger.debug(f"Residual predictions: posterior has vars: {list(posterior.data_vars)[:10]}...")
                # First try scaled version (original scale) - this is the main variable in BayesianMMM
                if "y_obs_scaled" in posterior:
                    data = posterior["y_obs_scaled"].values
                    result = data.mean(axis=(0, 1))
                    logger.debug(f"Residual predictions: Using y_obs_scaled, shape={result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
                    return result
                # Try mu (model mean) and convert to original scale
                if "mu" in posterior:
                    data = posterior["mu"].values
                    # mu is on standardized scale, convert to original
                    result = data.mean(axis=(0, 1)) * y_std + y_mean
                    logger.debug(f"Residual predictions: Using mu (scaled), shape={result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
                    return result
                # Look for other prediction variables
                for var_name in ["y_hat", "y_pred"]:
                    if var_name in posterior:
                        data = posterior[var_name].values
                        result = data.mean(axis=(0, 1)) * y_std + y_mean
                        logger.debug(f"Residual predictions: Using {var_name}, shape={result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
                        return result

            # Also check posterior_predictive for y_obs (standardized)
            if hasattr(trace, "posterior_predictive") and trace.posterior_predictive is not None:
                pp = trace.posterior_predictive
                if "y_obs" in pp:
                    data = pp["y_obs"].values
                    # y_obs is on standardized scale, convert to original
                    result = data.mean(axis=(0, 1)) * y_std + y_mean
                    logger.debug(f"Residual predictions: Using posterior_predictive y_obs, shape={result.shape}, range=[{result.min():.2f}, {result.max():.2f}]")
                    return result

        # Fallback: try to find any suitable variable
        if hasattr(self.model, "_trace"):
            trace = self.model._trace
            if hasattr(trace, "posterior"):
                posterior = trace.posterior
                # Look for any variable that matches observation shape
                y_obs = self._get_observed_data()
                n_obs = len(y_obs)
                for var_name in posterior.data_vars:
                    data = posterior[var_name].values
                    if data.ndim >= 3 and data.shape[-1] == n_obs:
                        # Assume standardized, convert
                        result = data.mean(axis=(0, 1)) * y_std + y_mean
                        logger.debug(f"Residual predictions: Using fallback {var_name}, shape={result.shape}")
                        return result

        raise ValueError("Could not extract predictions from model")

    def _run_durbin_watson(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
    ) -> TestResult:
        """Run Durbin-Watson test."""
        return durbin_watson_test(
            residuals,
            significance_level=self.config.significance_level,
        )

    def _run_ljung_box(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
    ) -> TestResult:
        """Run Ljung-Box test."""
        return ljung_box_test(
            residuals,
            max_lag=self.config.max_lag,
            significance_level=self.config.significance_level,
        )

    def _run_breusch_pagan(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
    ) -> TestResult:
        """Run Breusch-Pagan test."""
        # Use fitted values as exogenous
        exog = np.column_stack([np.ones(len(fitted_values)), fitted_values])
        return breusch_pagan_test(
            residuals,
            exog=exog,
            significance_level=self.config.significance_level,
        )

    def _run_shapiro_wilk(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
    ) -> TestResult:
        """Run Shapiro-Wilk test."""
        return shapiro_wilk_test(
            residuals,
            significance_level=self.config.significance_level,
        )

    def _run_jarque_bera(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
    ) -> TestResult:
        """Run Jarque-Bera test."""
        return jarque_bera_test(
            residuals,
            significance_level=self.config.significance_level,
        )


__all__ = ["ResidualDiagnostics"]

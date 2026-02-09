"""
Fluent builders for validation configuration.

Provides a builder pattern for constructing ValidationConfig objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self

from .config import (
    CalibrationConfig,
    ChannelDiagnosticsConfig,
    CrossValidationConfig,
    ModelComparisonConfig,
    PPCConfig,
    ResidualConfig,
    SensitivityConfig,
    StabilityConfig,
    ValidationConfig,
    ValidationLevel,
)
from .results import LiftTestResult


class ValidationConfigBuilder:
    """
    Fluent builder for ValidationConfig.

    Examples
    --------
    >>> # Quick validation
    >>> config = ValidationConfigBuilder().quick().build()

    >>> # Standard with custom residual tests
    >>> config = (ValidationConfigBuilder()
    ...     .standard()
    ...     .with_residual_tests(("durbin_watson", "ljung_box"))
    ...     .build())

    >>> # Thorough with calibration
    >>> config = (ValidationConfigBuilder()
    ...     .thorough()
    ...     .with_calibration(lift_tests)
    ...     .build())
    """

    def __init__(self):
        """Initialize builder with default values."""
        self._level = ValidationLevel.STANDARD
        self._ppc_config = PPCConfig()
        self._residuals_config = ResidualConfig()
        self._channel_config = ChannelDiagnosticsConfig()
        self._cv_config = CrossValidationConfig()
        self._comparison_config = ModelComparisonConfig()
        self._sensitivity_config = SensitivityConfig()
        self._stability_config = StabilityConfig()
        self._calibration_config = CalibrationConfig()

        self._run_ppc = True
        self._run_residuals = True
        self._run_channel_diagnostics = True
        self._run_model_comparison = False
        self._run_cv = False
        self._run_sensitivity = False
        self._run_stability = False
        self._run_calibration = False

        self._lift_tests: list[LiftTestResult] | None = None
        self._generate_plots = True
        self._verbose = True

    def quick(self) -> Self:
        """Configure for quick validation."""
        self._level = ValidationLevel.QUICK
        self._run_ppc = True
        self._run_residuals = True
        self._run_channel_diagnostics = True
        self._run_model_comparison = False
        self._run_cv = False
        self._run_sensitivity = False
        self._run_stability = False
        return self

    def standard(self) -> Self:
        """Configure for standard validation."""
        self._level = ValidationLevel.STANDARD
        self._run_ppc = True
        self._run_residuals = True
        self._run_channel_diagnostics = True
        self._run_model_comparison = True
        self._run_cv = False
        self._run_sensitivity = False
        self._run_stability = False
        return self

    def thorough(self) -> Self:
        """Configure for thorough validation."""
        self._level = ValidationLevel.THOROUGH
        self._run_ppc = True
        self._run_residuals = True
        self._run_channel_diagnostics = True
        self._run_model_comparison = True
        self._run_cv = True
        self._run_sensitivity = True
        self._run_stability = True
        return self

    def with_ppc(
        self,
        n_samples: int = 500,
        checks: tuple[str, ...] | None = None,
        include_channel_checks: bool = True,
    ) -> Self:
        """
        Configure posterior predictive checks.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples to use.
        checks : tuple[str, ...], optional
            Which checks to run. Default: mean, variance, autocorrelation,
            skewness, extremes.
        include_channel_checks : bool
            Whether to include channel-specific checks.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_ppc = True
        self._ppc_config = PPCConfig(
            n_samples=n_samples,
            checks=checks or self._ppc_config.checks,
            include_channel_checks=include_channel_checks,
        )
        return self

    def with_residual_tests(
        self,
        tests: tuple[str, ...],
        max_lag: int = 20,
        significance_level: float = 0.05,
    ) -> Self:
        """
        Configure residual diagnostic tests.

        Parameters
        ----------
        tests : tuple[str, ...]
            Which tests to run. Options: durbin_watson, ljung_box,
            breusch_pagan, shapiro_wilk, jarque_bera.
        max_lag : int
            Maximum lag for autocorrelation tests.
        significance_level : float
            Significance level for hypothesis tests.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_residuals = True
        self._residuals_config = ResidualConfig(
            tests=tests,
            max_lag=max_lag,
            significance_level=significance_level,
        )
        return self

    def with_channel_diagnostics(
        self,
        vif_threshold: float = 10.0,
        correlation_threshold: float = 0.8,
        rhat_threshold: float = 1.01,
        ess_threshold: int = 400,
    ) -> Self:
        """
        Configure channel diagnostics.

        Parameters
        ----------
        vif_threshold : float
            VIF threshold for multicollinearity warning.
        correlation_threshold : float
            Correlation threshold for multicollinearity warning.
        rhat_threshold : float
            R-hat threshold for convergence.
        ess_threshold : int
            ESS threshold for convergence.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_channel_diagnostics = True
        self._channel_config = ChannelDiagnosticsConfig(
            vif_threshold=vif_threshold,
            correlation_threshold=correlation_threshold,
            rhat_threshold=rhat_threshold,
            ess_threshold=ess_threshold,
        )
        return self

    def with_cross_validation(
        self,
        n_folds: int = 5,
        strategy: str = "expanding",
        min_train_size: int = 52,
        gap: int = 0,
        test_size: int | None = None,
    ) -> Self:
        """
        Enable cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of CV folds.
        strategy : str
            CV strategy: expanding, rolling, or blocked.
        min_train_size : int
            Minimum training set size.
        gap : int
            Gap between train and test (for blocked CV).
        test_size : int, optional
            Fixed test size (for rolling CV).

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_cv = True
        self._cv_config = CrossValidationConfig(
            n_folds=n_folds,
            strategy=strategy,
            min_train_size=min_train_size,
            gap=gap,
            test_size=test_size,
        )
        return self

    def with_model_comparison(
        self,
        method: str = "loo",
        pointwise: bool = True,
    ) -> Self:
        """
        Enable model comparison.

        Parameters
        ----------
        method : str
            Comparison method: loo, waic, or both.
        pointwise : bool
            Whether to compute pointwise values.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_model_comparison = True
        self._comparison_config = ModelComparisonConfig(
            method=method,
            pointwise=pointwise,
        )
        return self

    def with_sensitivity_analysis(
        self,
        prior_multipliers: tuple[float, ...] = (0.5, 2.0),
        parameters_of_interest: tuple[str, ...] | None = None,
        include_specification_tests: bool = True,
    ) -> Self:
        """
        Enable sensitivity analysis.

        Parameters
        ----------
        prior_multipliers : tuple[float, ...]
            Multipliers for prior variance in sensitivity tests.
        parameters_of_interest : tuple[str, ...], optional
            Specific parameters to analyze.
        include_specification_tests : bool
            Whether to test specification variants.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_sensitivity = True
        self._sensitivity_config = SensitivityConfig(
            prior_multipliers=prior_multipliers,
            parameters_of_interest=parameters_of_interest,
            include_specification_tests=include_specification_tests,
        )
        return self

    def with_stability_analysis(
        self,
        n_bootstrap: int = 100,
        loo_subset_size: int | None = None,
        perturbation_level: float = 0.1,
        n_perturbations: int = 20,
    ) -> Self:
        """
        Enable stability analysis.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples.
        loo_subset_size : int, optional
            Subset size for LOO influence (None = all).
        perturbation_level : float
            Perturbation level for sensitivity.
        n_perturbations : int
            Number of perturbation runs.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_stability = True
        self._stability_config = StabilityConfig(
            n_bootstrap=n_bootstrap,
            loo_subset_size=loo_subset_size,
            perturbation_level=perturbation_level,
            n_perturbations=n_perturbations,
        )
        return self

    def with_calibration(
        self,
        lift_tests: list[LiftTestResult],
        ci_level: float = 0.94,
        tolerance_multiplier: float = 1.5,
    ) -> Self:
        """
        Enable calibration with external experiments.

        Parameters
        ----------
        lift_tests : list[LiftTestResult]
            List of lift test results for calibration.
        ci_level : float
            Credible interval level for comparison.
        tolerance_multiplier : float
            Tolerance multiplier for SE deviation.

        Returns
        -------
        Self
            Builder instance for chaining.
        """
        self._run_calibration = True
        self._lift_tests = lift_tests
        self._calibration_config = CalibrationConfig(
            ci_level=ci_level,
            tolerance_multiplier=tolerance_multiplier,
        )
        return self

    def without_ppc(self) -> Self:
        """Disable posterior predictive checks."""
        self._run_ppc = False
        return self

    def without_residuals(self) -> Self:
        """Disable residual diagnostics."""
        self._run_residuals = False
        return self

    def without_channel_diagnostics(self) -> Self:
        """Disable channel diagnostics."""
        self._run_channel_diagnostics = False
        return self

    def without_plots(self) -> Self:
        """Disable plot generation."""
        self._generate_plots = False
        return self

    def silent(self) -> Self:
        """Disable verbose output."""
        self._verbose = False
        return self

    def build(self) -> ValidationConfig:
        """
        Build and return the ValidationConfig.

        Returns
        -------
        ValidationConfig
            The configured validation settings.
        """
        return ValidationConfig(
            level=self._level,
            ppc=self._ppc_config,
            residuals=self._residuals_config,
            channel_diagnostics=self._channel_config,
            cross_validation=self._cv_config,
            model_comparison=self._comparison_config,
            sensitivity=self._sensitivity_config,
            stability=self._stability_config,
            calibration=self._calibration_config,
            run_ppc=self._run_ppc,
            run_residuals=self._run_residuals,
            run_channel_diagnostics=self._run_channel_diagnostics,
            run_model_comparison=self._run_model_comparison,
            run_cross_validation=self._run_cv,
            run_sensitivity=self._run_sensitivity,
            run_stability=self._run_stability,
            run_calibration=self._run_calibration,
            lift_tests=self._lift_tests,
            generate_plots=self._generate_plots,
            verbose=self._verbose,
        )


__all__ = ["ValidationConfigBuilder"]

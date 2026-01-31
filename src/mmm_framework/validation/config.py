"""
Configuration classes for model validation.

Provides dataclass-based configurations for all validation components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .results import LiftTestResult


class ValidationLevel(str, Enum):
    """Validation thoroughness level."""

    QUICK = "quick"  # ~30 seconds: PPC, residuals, channel diagnostics
    STANDARD = "standard"  # ~5 minutes: quick + LOO-CV, WAIC
    THOROUGH = "thorough"  # ~30+ minutes: standard + CV, sensitivity, stability


@dataclass(frozen=True)
class PPCConfig:
    """Configuration for posterior predictive checks."""

    n_samples: int = 500
    checks: tuple[str, ...] = (
        "mean",
        "variance",
        "autocorrelation",
        "skewness",
        "extremes",
    )
    include_channel_checks: bool = True
    significance_level: float = 0.05


@dataclass(frozen=True)
class ResidualConfig:
    """Configuration for residual diagnostics."""

    max_lag: int = 20
    significance_level: float = 0.05
    tests: tuple[str, ...] = (
        "durbin_watson",
        "ljung_box",
        "breusch_pagan",
        "shapiro_wilk",
        "jarque_bera",
    )


@dataclass(frozen=True)
class ChannelDiagnosticsConfig:
    """Configuration for channel diagnostics."""

    vif_threshold: float = 10.0
    correlation_threshold: float = 0.8
    rhat_threshold: float = 1.01
    ess_threshold: int = 400


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for cross-validation."""

    strategy: Literal["expanding", "rolling", "blocked"] = "expanding"
    n_folds: int = 5
    min_train_size: int = 52  # Minimum training observations
    gap: int = 0  # Gap between train and test (for blocked CV)
    test_size: int | None = None  # Fixed test size (for rolling)

    # Fitting options for CV runs
    draws_per_fold: int = 500
    tune_per_fold: int = 250
    chains_per_fold: int = 2


@dataclass(frozen=True)
class ModelComparisonConfig:
    """Configuration for model comparison."""

    method: Literal["loo", "waic", "both"] = "loo"
    pointwise: bool = True
    pareto_k_threshold: float = 0.7


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for sensitivity analysis."""

    prior_multipliers: tuple[float, ...] = (0.5, 2.0)
    parameters_of_interest: tuple[str, ...] | None = None
    include_specification_tests: bool = True

    # Fitting options for sensitivity runs
    draws_per_variant: int = 500
    tune_per_variant: int = 250
    chains_per_variant: int = 2


@dataclass(frozen=True)
class StabilityConfig:
    """Configuration for stability analysis."""

    n_bootstrap: int = 100
    loo_subset_size: int | None = None  # None = all observations
    perturbation_level: float = 0.1
    n_perturbations: int = 20


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for calibration checks."""

    ci_level: float = 0.94
    tolerance_multiplier: float = 1.5  # Allow 1.5x SE deviation


@dataclass
class ValidationConfig:
    """
    Complete validation configuration.

    Controls which validations to run and their parameters.

    Examples
    --------
    >>> # Quick validation
    >>> config = ValidationConfig.quick()

    >>> # Standard with custom residual tests
    >>> config = ValidationConfig.standard()

    >>> # Thorough with calibration data
    >>> config = ValidationConfig.thorough()
    """

    level: ValidationLevel = ValidationLevel.STANDARD

    # Individual component configs
    ppc: PPCConfig = field(default_factory=PPCConfig)
    residuals: ResidualConfig = field(default_factory=ResidualConfig)
    channel_diagnostics: ChannelDiagnosticsConfig = field(
        default_factory=ChannelDiagnosticsConfig
    )
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    model_comparison: ModelComparisonConfig = field(
        default_factory=ModelComparisonConfig
    )
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Which validations to run
    run_ppc: bool = True
    run_residuals: bool = True
    run_channel_diagnostics: bool = True
    run_model_comparison: bool = False  # Requires multiple models
    run_cross_validation: bool = False  # Expensive
    run_sensitivity: bool = False  # Expensive
    run_stability: bool = False  # Expensive
    run_calibration: bool = False  # Requires external data

    # Calibration data (set via builder)
    lift_tests: list[LiftTestResult] | None = None

    # Output options
    generate_plots: bool = True
    verbose: bool = True

    @classmethod
    def quick(cls) -> ValidationConfig:
        """
        Quick validation (convergence, PPC, residuals, channel diagnostics).

        Fast feedback on model quality, suitable for iterative development.
        """
        return cls(
            level=ValidationLevel.QUICK,
            run_ppc=True,
            run_residuals=True,
            run_channel_diagnostics=True,
            run_model_comparison=False,
            run_cross_validation=False,
            run_sensitivity=False,
            run_stability=False,
            run_calibration=False,
        )

    @classmethod
    def standard(cls) -> ValidationConfig:
        """
        Standard validation (quick + LOO-CV, WAIC).

        Good balance of thoroughness and compute time.
        """
        return cls(
            level=ValidationLevel.STANDARD,
            run_ppc=True,
            run_residuals=True,
            run_channel_diagnostics=True,
            run_model_comparison=True,
            run_cross_validation=False,
            run_sensitivity=False,
            run_stability=False,
            run_calibration=False,
        )

    @classmethod
    def thorough(cls) -> ValidationConfig:
        """
        Thorough validation (all checks).

        Comprehensive validation for production models.
        """
        return cls(
            level=ValidationLevel.THOROUGH,
            run_ppc=True,
            run_residuals=True,
            run_channel_diagnostics=True,
            run_model_comparison=True,
            run_cross_validation=True,
            run_sensitivity=True,
            run_stability=True,
            run_calibration=False,  # Still requires external data
        )


__all__ = [
    "ValidationLevel",
    "PPCConfig",
    "ResidualConfig",
    "ChannelDiagnosticsConfig",
    "CrossValidationConfig",
    "ModelComparisonConfig",
    "SensitivityConfig",
    "StabilityConfig",
    "CalibrationConfig",
    "ValidationConfig",
]

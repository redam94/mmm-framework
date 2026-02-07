"""
Result containers for model validation.

Provides dataclasses for all validation outputs following the existing
pattern in mmm_framework.model.results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


# =============================================================================
# Base Result Types
# =============================================================================


@dataclass
class TestResult:
    """Result of a single statistical test."""

    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "passed": self.passed,
            "threshold": self.threshold,
            "interpretation": self.interpretation,
        }


@dataclass
class ConvergenceSummary:
    """MCMC convergence summary."""

    divergences: int
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "divergences": self.divergences,
            "rhat_max": self.rhat_max,
            "ess_bulk_min": self.ess_bulk_min,
            "ess_tail_min": self.ess_tail_min,
            "converged": self.converged,
        }

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Metric": [
                    "Divergences",
                    "Max R-hat",
                    "Min ESS (bulk)",
                    "Min ESS (tail)",
                    "Converged",
                ],
                "Value": [
                    self.divergences,
                    f"{self.rhat_max:.4f}",
                    f"{self.ess_bulk_min:.0f}",
                    f"{self.ess_tail_min:.0f}",
                    "Yes" if self.converged else "No",
                ],
                "Status": [
                    "Pass" if self.divergences == 0 else "Fail",
                    "Pass" if self.rhat_max < 1.01 else "Fail",
                    "Pass" if self.ess_bulk_min > 400 else "Warning",
                    "Pass" if self.ess_tail_min > 400 else "Warning",
                    "Pass" if self.converged else "Fail",
                ],
            }
        )


# =============================================================================
# Posterior Predictive Check Results
# =============================================================================


@dataclass
class PPCCheckResult:
    """Result of a single posterior predictive check."""

    check_name: str
    observed_statistic: float
    replicated_mean: float
    replicated_std: float
    p_value: float  # Bayesian p-value
    passed: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "observed_statistic": self.observed_statistic,
            "replicated_mean": self.replicated_mean,
            "replicated_std": self.replicated_std,
            "p_value": self.p_value,
            "passed": self.passed,
            "description": self.description,
        }


@dataclass
class PPCResults:
    """Posterior predictive check results."""

    checks: list[PPCCheckResult]
    y_obs: np.ndarray
    y_rep: np.ndarray  # (n_samples, n_obs)
    overall_pass: bool = field(init=False)
    problematic_checks: list[str] = field(init=False)

    def __post_init__(self):
        self.problematic_checks = [c.check_name for c in self.checks if not c.passed]
        self.overall_pass = len(self.problematic_checks) == 0

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Check": [c.check_name for c in self.checks],
                "Observed": [f"{c.observed_statistic:.4f}" for c in self.checks],
                "Replicated Mean": [f"{c.replicated_mean:.4f}" for c in self.checks],
                "Replicated Std": [f"{c.replicated_std:.4f}" for c in self.checks],
                "Bayesian p-value": [f"{c.p_value:.4f}" for c in self.checks],
                "Status": ["Pass" if c.passed else "Fail" for c in self.checks],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checks": [c.to_dict() for c in self.checks],
            "overall_pass": self.overall_pass,
            "problematic_checks": self.problematic_checks,
        }


# =============================================================================
# Residual Diagnostics Results
# =============================================================================


@dataclass
class ResidualDiagnosticsResults:
    """Results from residual diagnostics."""

    test_results: list[TestResult]
    residuals: np.ndarray
    fitted_values: np.ndarray
    acf_values: np.ndarray
    pacf_values: np.ndarray
    overall_adequate: bool = field(init=False)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        failed_tests = [t for t in self.test_results if not t.passed]
        self.overall_adequate = len(failed_tests) == 0

        # Generate recommendations based on failed tests
        for test in failed_tests:
            if "autocorrelation" in test.test_name.lower():
                self.recommendations.append(
                    "Consider adding autoregressive terms or adjusting adstock"
                )
            elif "heteroscedasticity" in test.test_name.lower():
                self.recommendations.append(
                    "Consider using a different likelihood (e.g., Student-t)"
                )
            elif "normality" in test.test_name.lower():
                self.recommendations.append(
                    "Consider using a robust likelihood or checking for outliers"
                )

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Test": [t.test_name for t in self.test_results],
                "Statistic": [f"{t.statistic:.4f}" for t in self.test_results],
                "p-value": [f"{t.p_value:.4f}" for t in self.test_results],
                "Threshold": [f"{t.threshold:.4f}" for t in self.test_results],
                "Status": ["Pass" if t.passed else "Fail" for t in self.test_results],
                "Interpretation": [t.interpretation for t in self.test_results],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_results": [t.to_dict() for t in self.test_results],
            "overall_adequate": self.overall_adequate,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Channel Diagnostics Results
# =============================================================================


@dataclass
class ChannelConvergenceResult:
    """Convergence diagnostics for a single channel."""

    channel: str
    rhat: float
    ess_bulk: float
    ess_tail: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "rhat": self.rhat,
            "ess_bulk": self.ess_bulk,
            "ess_tail": self.ess_tail,
            "converged": self.converged,
        }


@dataclass
class ChannelDiagnosticsResults:
    """Results from channel diagnostics."""

    vif_scores: dict[str, float]
    correlation_matrix: pd.DataFrame
    convergence_by_channel: dict[str, ChannelConvergenceResult]
    identifiability_issues: list[str] = field(default_factory=list)
    multicollinearity_warning: bool = field(init=False)
    convergence_warning: bool = field(init=False)

    def __post_init__(self):
        # Check for high VIF
        self.multicollinearity_warning = any(v > 10.0 for v in self.vif_scores.values())

        # Check for convergence issues
        self.convergence_warning = any(
            not c.converged for c in self.convergence_by_channel.values()
        )

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        channels = list(self.vif_scores.keys())
        return pd.DataFrame(
            {
                "Channel": channels,
                "VIF": [f"{self.vif_scores[c]:.2f}" for c in channels],
                "R-hat": [
                    f"{self.convergence_by_channel[c].rhat:.4f}" for c in channels
                ],
                "ESS (bulk)": [
                    f"{self.convergence_by_channel[c].ess_bulk:.0f}" for c in channels
                ],
                "Converged": [
                    "Yes" if self.convergence_by_channel[c].converged else "No"
                    for c in channels
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vif_scores": self.vif_scores,
            "convergence_by_channel": {
                k: v.to_dict() for k, v in self.convergence_by_channel.items()
            },
            "identifiability_issues": self.identifiability_issues,
            "multicollinearity_warning": self.multicollinearity_warning,
            "convergence_warning": self.convergence_warning,
        }


# =============================================================================
# Model Comparison Results
# =============================================================================


@dataclass
class LOOResults:
    """Leave-one-out cross-validation results (PSIS-LOO)."""

    elpd_loo: float
    se_elpd_loo: float
    p_loo: float  # Effective number of parameters
    pareto_k: np.ndarray  # Per-observation Pareto k values
    n_bad_k: int  # Number of k > 0.7
    pointwise_elpd: np.ndarray | None = None

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Metric": [
                    "ELPD LOO",
                    "SE ELPD",
                    "p_loo",
                    "Bad Pareto k (>0.7)",
                ],
                "Value": [
                    f"{self.elpd_loo:.2f}",
                    f"{self.se_elpd_loo:.2f}",
                    f"{self.p_loo:.2f}",
                    str(self.n_bad_k),
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "elpd_loo": self.elpd_loo,
            "se_elpd_loo": self.se_elpd_loo,
            "p_loo": self.p_loo,
            "n_bad_k": self.n_bad_k,
        }


@dataclass
class WAICResults:
    """WAIC results."""

    waic: float
    se_waic: float
    p_waic: float  # Effective number of parameters
    pointwise: np.ndarray | None = None

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Metric": ["WAIC", "SE WAIC", "p_waic"],
                "Value": [
                    f"{self.waic:.2f}",
                    f"{self.se_waic:.2f}",
                    f"{self.p_waic:.2f}",
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "waic": self.waic,
            "se_waic": self.se_waic,
            "p_waic": self.p_waic,
        }


@dataclass
class ModelComparisonEntry:
    """Single model entry for comparison."""

    name: str
    loo: LOOResults | None = None
    waic: WAICResults | None = None


@dataclass
class ModelComparisonResults:
    """Results from comparing multiple models."""

    models: list[ModelComparisonEntry]
    loo_comparison: pd.DataFrame | None = None
    waic_comparison: pd.DataFrame | None = None
    stacking_weights: dict[str, float] | None = None
    best_model: str = field(init=False)

    def __post_init__(self):
        # Determine best model by ELPD LOO
        if self.models:
            best = max(
                self.models,
                key=lambda m: m.loo.elpd_loo if m.loo else float("-inf"),
            )
            self.best_model = best.name
        else:
            self.best_model = ""

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        rows = []
        for m in self.models:
            row = {"Model": m.name}
            if m.loo:
                row["ELPD LOO"] = f"{m.loo.elpd_loo:.2f}"
                row["SE"] = f"{m.loo.se_elpd_loo:.2f}"
                row["p_loo"] = f"{m.loo.p_loo:.2f}"
            if m.waic:
                row["WAIC"] = f"{m.waic.waic:.2f}"
            if self.stacking_weights:
                row["Stacking Weight"] = f"{self.stacking_weights.get(m.name, 0):.3f}"
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models": [
                {
                    "name": m.name,
                    "loo": m.loo.to_dict() if m.loo else None,
                    "waic": m.waic.to_dict() if m.waic else None,
                }
                for m in self.models
            ],
            "best_model": self.best_model,
            "stacking_weights": self.stacking_weights,
        }


# =============================================================================
# Cross-Validation Results
# =============================================================================


@dataclass
class CVFoldResult:
    """Result for a single cross-validation fold."""

    fold_idx: int
    train_size: int
    test_size: int
    rmse: float
    mae: float
    mape: float
    r2: float
    coverage: float  # % of observations within credible interval

    # Optional prediction data for visualization
    test_indices: np.ndarray | None = None  # Time indices of test observations
    y_true: np.ndarray | None = None  # Actual test values
    y_pred_mean: np.ndarray | None = None  # Predicted mean
    y_pred_ci_low: np.ndarray | None = None  # Lower CI bound (e.g., 3%)
    y_pred_ci_high: np.ndarray | None = None  # Upper CI bound (e.g., 97%)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "fold_idx": self.fold_idx,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "coverage": self.coverage,
        }
        # Include prediction data if available
        if self.test_indices is not None:
            result["test_indices"] = self.test_indices.tolist()
        if self.y_true is not None:
            result["y_true"] = self.y_true.tolist()
        if self.y_pred_mean is not None:
            result["y_pred_mean"] = self.y_pred_mean.tolist()
        if self.y_pred_ci_low is not None:
            result["y_pred_ci_low"] = self.y_pred_ci_low.tolist()
        if self.y_pred_ci_high is not None:
            result["y_pred_ci_high"] = self.y_pred_ci_high.tolist()
        return result


@dataclass
class CrossValidationResults:
    """Results from cross-validation."""

    strategy: str
    n_folds: int
    fold_results: list[CVFoldResult]

    # Aggregate metrics (computed in __post_init__)
    mean_rmse: float = field(init=False)
    std_rmse: float = field(init=False)
    mean_mae: float = field(init=False)
    mean_mape: float = field(init=False)
    mean_r2: float = field(init=False)
    mean_coverage: float = field(init=False)

    def __post_init__(self):
        if self.fold_results:
            self.mean_rmse = np.mean([f.rmse for f in self.fold_results])
            self.std_rmse = np.std([f.rmse for f in self.fold_results])
            self.mean_mae = np.mean([f.mae for f in self.fold_results])
            self.mean_mape = np.mean([f.mape for f in self.fold_results])
            self.mean_r2 = np.mean([f.r2 for f in self.fold_results])
            self.mean_coverage = np.mean([f.coverage for f in self.fold_results])
        else:
            self.mean_rmse = 0.0
            self.std_rmse = 0.0
            self.mean_mae = 0.0
            self.mean_mape = 0.0
            self.mean_r2 = 0.0
            self.mean_coverage = 0.0

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Metric": ["RMSE", "MAE", "MAPE", "R²", "Coverage"],
                "Mean": [
                    f"{self.mean_rmse:.4f}",
                    f"{self.mean_mae:.4f}",
                    f"{self.mean_mape:.4f}",
                    f"{self.mean_r2:.4f}",
                    f"{self.mean_coverage:.2%}",
                ],
                "Std": [
                    f"{self.std_rmse:.4f}",
                    f"{np.std([f.mae for f in self.fold_results]):.4f}",
                    f"{np.std([f.mape for f in self.fold_results]):.4f}",
                    f"{np.std([f.r2 for f in self.fold_results]):.4f}",
                    f"{np.std([f.coverage for f in self.fold_results]):.2%}",
                ],
            }
        )

    def fold_summary(self) -> pd.DataFrame:
        """Get per-fold summary DataFrame."""
        return pd.DataFrame(
            {
                "Fold": [f.fold_idx for f in self.fold_results],
                "Train Size": [f.train_size for f in self.fold_results],
                "Test Size": [f.test_size for f in self.fold_results],
                "RMSE": [f"{f.rmse:.4f}" for f in self.fold_results],
                "MAE": [f"{f.mae:.4f}" for f in self.fold_results],
                "R²": [f"{f.r2:.4f}" for f in self.fold_results],
                "Coverage": [f"{f.coverage:.2%}" for f in self.fold_results],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "n_folds": self.n_folds,
            "fold_results": [f.to_dict() for f in self.fold_results],
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_mae": self.mean_mae,
            "mean_mape": self.mean_mape,
            "mean_r2": self.mean_r2,
            "mean_coverage": self.mean_coverage,
        }


# =============================================================================
# Sensitivity Analysis Results
# =============================================================================


@dataclass
class SensitivityResults:
    """Results from sensitivity analysis."""

    base_estimates: dict[str, float]
    variant_estimates: dict[str, dict[str, float]]  # variant_name -> param -> estimate
    sensitivity_indices: dict[str, float]  # param -> sensitivity index
    robust_parameters: list[str] = field(default_factory=list)
    sensitive_parameters: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        params = list(self.base_estimates.keys())
        return pd.DataFrame(
            {
                "Parameter": params,
                "Base Estimate": [f"{self.base_estimates[p]:.4f}" for p in params],
                "Sensitivity Index": [
                    f"{self.sensitivity_indices.get(p, 0):.4f}" for p in params
                ],
                "Robust": [
                    "Yes" if p in self.robust_parameters else "No" for p in params
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_estimates": self.base_estimates,
            "variant_estimates": self.variant_estimates,
            "sensitivity_indices": self.sensitivity_indices,
            "robust_parameters": self.robust_parameters,
            "sensitive_parameters": self.sensitive_parameters,
        }


# =============================================================================
# Stability Analysis Results
# =============================================================================


@dataclass
class BootstrapResults:
    """Results from parametric bootstrap."""

    n_bootstrap: int
    parameter_means: dict[str, float]
    parameter_stds: dict[str, float]
    parameter_ci_low: dict[str, float]
    parameter_ci_high: dict[str, float]

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        params = list(self.parameter_means.keys())
        return pd.DataFrame(
            {
                "Parameter": params,
                "Mean": [f"{self.parameter_means[p]:.4f}" for p in params],
                "Std": [f"{self.parameter_stds[p]:.4f}" for p in params],
                "CI Low": [f"{self.parameter_ci_low[p]:.4f}" for p in params],
                "CI High": [f"{self.parameter_ci_high[p]:.4f}" for p in params],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_bootstrap": self.n_bootstrap,
            "parameter_means": self.parameter_means,
            "parameter_stds": self.parameter_stds,
            "parameter_ci_low": self.parameter_ci_low,
            "parameter_ci_high": self.parameter_ci_high,
        }


@dataclass
class InfluenceResults:
    """Results from leave-one-out influence analysis."""

    observation_influence: np.ndarray  # Influence score per observation
    influential_indices: list[int]  # Indices of influential observations
    influence_threshold: float

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame for influential observations."""
        return pd.DataFrame(
            {
                "Observation Index": self.influential_indices,
                "Influence Score": [
                    f"{self.observation_influence[i]:.4f}"
                    for i in self.influential_indices
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_influential": len(self.influential_indices),
            "influential_indices": self.influential_indices,
            "influence_threshold": self.influence_threshold,
        }


@dataclass
class StabilityResults:
    """Results from stability analysis."""

    bootstrap_results: BootstrapResults | None = None
    influence_results: InfluenceResults | None = None
    influential_observations: list[int] = field(default_factory=list)
    stability_score: float = 1.0  # 0-1 score (1 = very stable)

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        rows = [
            {"Metric": "Stability Score", "Value": f"{self.stability_score:.2f}"},
            {
                "Metric": "Influential Observations",
                "Value": str(len(self.influential_observations)),
            },
        ]
        if self.bootstrap_results:
            rows.append(
                {
                    "Metric": "Bootstrap Samples",
                    "Value": str(self.bootstrap_results.n_bootstrap),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bootstrap_results": (
                self.bootstrap_results.to_dict() if self.bootstrap_results else None
            ),
            "influence_results": (
                self.influence_results.to_dict() if self.influence_results else None
            ),
            "influential_observations": self.influential_observations,
            "stability_score": self.stability_score,
        }


# =============================================================================
# Calibration Results
# =============================================================================


@dataclass(frozen=True)
class LiftTestResult:
    """External lift test result for calibration."""

    channel: str
    test_period: tuple[str, str]  # (start, end)
    measured_lift: float
    lift_se: float  # Standard error
    holdout_regions: list[str] | None = None
    confidence_level: float = 0.95


@dataclass
class LiftTestComparison:
    """Comparison of model estimate to lift test."""

    channel: str
    model_estimate: float
    model_ci_low: float
    model_ci_high: float
    experimental_estimate: float
    experimental_se: float
    within_ci: bool
    relative_error: float  # (model - experiment) / experiment

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel": self.channel,
            "model_estimate": self.model_estimate,
            "model_ci_low": self.model_ci_low,
            "model_ci_high": self.model_ci_high,
            "experimental_estimate": self.experimental_estimate,
            "experimental_se": self.experimental_se,
            "within_ci": self.within_ci,
            "relative_error": self.relative_error,
        }


@dataclass
class CalibrationResults:
    """Results from calibration check."""

    lift_test_comparisons: list[LiftTestComparison]
    coverage_rate: float  # % of experiments within model CI
    mean_absolute_calibration_error: float
    calibrated: bool = field(init=False)

    def __post_init__(self):
        # Consider calibrated if coverage rate is reasonable (> 0.5)
        self.calibrated = self.coverage_rate >= 0.5

    def summary(self) -> pd.DataFrame:
        """Get summary DataFrame."""
        return pd.DataFrame(
            {
                "Channel": [c.channel for c in self.lift_test_comparisons],
                "Model Estimate": [
                    f"{c.model_estimate:.2f}" for c in self.lift_test_comparisons
                ],
                "Model CI": [
                    f"[{c.model_ci_low:.2f}, {c.model_ci_high:.2f}]"
                    for c in self.lift_test_comparisons
                ],
                "Experimental": [
                    f"{c.experimental_estimate:.2f} ± {c.experimental_se:.2f}"
                    for c in self.lift_test_comparisons
                ],
                "Within CI": [
                    "Yes" if c.within_ci else "No" for c in self.lift_test_comparisons
                ],
                "Relative Error": [
                    f"{c.relative_error:+.1%}" for c in self.lift_test_comparisons
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lift_test_comparisons": [c.to_dict() for c in self.lift_test_comparisons],
            "coverage_rate": self.coverage_rate,
            "mean_absolute_calibration_error": self.mean_absolute_calibration_error,
            "calibrated": self.calibrated,
        }


# =============================================================================
# Validation Summary
# =============================================================================


@dataclass
class ValidationSummary:
    """
    Comprehensive validation summary.

    Aggregates all validation results with overall assessment.
    """

    model_name: str
    validation_date: str = field(default_factory=lambda: datetime.now().isoformat())

    # Quick checks
    convergence: ConvergenceSummary | None = None
    ppc: PPCResults | None = None
    residuals: ResidualDiagnosticsResults | None = None
    channel_diagnostics: ChannelDiagnosticsResults | None = None

    # Model comparison
    model_comparison: ModelComparisonResults | None = None

    # Thorough checks
    cross_validation: CrossValidationResults | None = None
    sensitivity: SensitivityResults | None = None
    stability: StabilityResults | None = None
    calibration: CalibrationResults | None = None

    # Overall assessment
    overall_quality: Literal["excellent", "good", "acceptable", "poor"] = "acceptable"
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Full y_actual for CV time-series visualization (private)
    _full_y_actual: np.ndarray | None = None

    def summary(self) -> pd.DataFrame:
        """Get high-level summary DataFrame."""
        rows = [
            {"Component": "Model", "Value": self.model_name},
            {"Component": "Date", "Value": self.validation_date},
            {"Component": "Overall Quality", "Value": self.overall_quality.title()},
            {"Component": "Critical Issues", "Value": str(len(self.critical_issues))},
            {"Component": "Warnings", "Value": str(len(self.warnings))},
        ]

        if self.convergence:
            rows.append(
                {
                    "Component": "Convergence",
                    "Value": "Pass" if self.convergence.converged else "Fail",
                }
            )
        if self.ppc:
            rows.append(
                {
                    "Component": "PPC",
                    "Value": "Pass" if self.ppc.overall_pass else "Fail",
                }
            )
        if self.residuals:
            rows.append(
                {
                    "Component": "Residuals",
                    "Value": "Pass" if self.residuals.overall_adequate else "Fail",
                }
            )
        if self.cross_validation:
            rows.append(
                {
                    "Component": "CV Mean R²",
                    "Value": f"{self.cross_validation.mean_r2:.4f}",
                }
            )

        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "validation_date": self.validation_date,
            "overall_quality": self.overall_quality,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "convergence": self.convergence.to_dict() if self.convergence else None,
            "ppc": self.ppc.to_dict() if self.ppc else None,
            "residuals": self.residuals.to_dict() if self.residuals else None,
            "channel_diagnostics": (
                self.channel_diagnostics.to_dict() if self.channel_diagnostics else None
            ),
            "cross_validation": (
                self.cross_validation.to_dict() if self.cross_validation else None
            ),
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }

    def to_html_report(self, include_charts: bool = True) -> str:
        """
        Generate HTML validation report.

        Parameters
        ----------
        include_charts : bool
            Whether to include interactive Plotly charts.

        Returns
        -------
        str
            HTML report string.
        """
        # Import chart functions if needed
        if include_charts:
            try:
                from .charts import (
                    create_acf_chart,
                    create_cv_actual_vs_predicted_chart,
                    create_cv_coverage_chart,
                    create_cv_fold_metrics_chart,
                    create_pit_histogram,
                    create_ppc_density_plot,
                    create_ppc_statistics_plot,
                    create_ppc_time_series_plot,
                    create_qq_plot,
                    create_residual_time_series_plot,
                    create_residual_vs_fitted,
                    create_vif_chart,
                )

                charts_available = True
            except ImportError:
                charts_available = False
        else:
            charts_available = False

        # CSS styling matching the main MMM reports
        css = """
        :root {
            --color-primary: #8fa86a;
            --color-primary-dark: #6d8a4a;
            --color-accent: #6a8fa8;
            --color-accent-dark: #4a6d8a;
            --color-warning: #d4a86a;
            --color-danger: #c97067;
            --color-success: #6abf8a;
            --color-text: #2d3a2d;
            --color-text-muted: #5a6b5a;
            --color-bg: #fafbf9;
            --color-bg-alt: #f0f2ed;
            --color-surface: #ffffff;
            --color-border: #d4ddd4;
            --shadow-sm: 0 2px 8px rgba(45, 58, 45, 0.06);
            --shadow-md: 0 8px 24px rgba(45, 58, 45, 0.08);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--color-bg);
            color: var(--color-text);
            line-height: 1.7;
        }

        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .report-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-accent-dark) 100%);
            color: white;
            border-radius: 16px;
            margin-bottom: 2rem;
        }

        .report-header h1 {
            font-family: 'DM Serif Display', serif;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .report-header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .report-header .date {
            margin-top: 1rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .quality-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
        }

        .quality-badge.excellent { background: rgba(106, 191, 138, 0.3); }
        .quality-badge.good { background: rgba(106, 191, 138, 0.2); }
        .quality-badge.acceptable { background: rgba(212, 168, 106, 0.3); }
        .quality-badge.poor { background: rgba(201, 112, 103, 0.3); }

        .section {
            background: var(--color-surface);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--color-border);
        }

        .section h2 {
            font-family: 'DM Serif Display', serif;
            font-size: 1.6rem;
            color: var(--color-text);
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--color-primary);
        }

        .section h3 {
            font-size: 1.2rem;
            color: var(--color-text);
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        .section p {
            margin-bottom: 1rem;
            color: var(--color-text);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0;
        }

        .metric-card {
            background: var(--color-bg-alt);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--color-border);
        }

        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--color-primary-dark);
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-card .label {
            font-size: 0.85rem;
            color: var(--color-text-muted);
            margin-top: 0.25rem;
        }

        .metric-card.highlight { border-left: 4px solid var(--color-primary); }
        .metric-card.warning { border-left: 4px solid var(--color-warning); }
        .metric-card.danger { border-left: 4px solid var(--color-danger); }
        .metric-card.success { border-left: 4px solid var(--color-success); }

        .chart-container {
            width: 100%;
            min-height: 300px;
            margin: 1.5rem 0;
        }

        .callout {
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }

        .callout h4 {
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }

        .callout p, .callout ul {
            margin-bottom: 0;
            font-size: 0.95rem;
        }

        .callout ul { margin-left: 1.5rem; }
        .callout li { margin-bottom: 0.25rem; }

        .callout.insight {
            background: rgba(106, 143, 168, 0.1);
            border: 1px solid rgba(106, 143, 168, 0.3);
            border-left: 4px solid var(--color-accent);
        }
        .callout.insight h4 { color: var(--color-accent-dark); }

        .callout.warning {
            background: rgba(212, 168, 106, 0.1);
            border: 1px solid rgba(212, 168, 106, 0.3);
            border-left: 4px solid var(--color-warning);
        }
        .callout.warning h4 { color: #b8860b; }

        .callout.success {
            background: rgba(106, 191, 138, 0.1);
            border: 1px solid rgba(106, 191, 138, 0.3);
            border-left: 4px solid var(--color-success);
        }
        .callout.success h4 { color: #3d8b5a; }

        .callout.danger {
            background: rgba(201, 112, 103, 0.08);
            border: 1px solid rgba(201, 112, 103, 0.3);
            border-left: 4px solid var(--color-danger);
        }
        .callout.danger h4 { color: var(--color-danger); }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }

        .data-table th, .data-table td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--color-border);
        }

        .data-table th {
            background: var(--color-bg-alt);
            font-weight: 600;
            color: var(--color-text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .data-table tr:hover { background: var(--color-bg-alt); }

        .data-table .mono { font-family: 'JetBrains Mono', monospace; }
        .data-table .positive, .pass { color: var(--color-success); font-weight: 600; }
        .data-table .negative, .fail { color: var(--color-danger); font-weight: 600; }
        .data-table .uncertain, .warning-text { color: var(--color-warning); font-weight: 600; }

        .report-footer {
            text-align: center;
            padding: 2rem;
            color: var(--color-text-muted);
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .report-container { padding: 1rem; }
            .metrics-grid { grid-template-columns: 1fr 1fr; }
            .report-header h1 { font-size: 1.8rem; }
        }

        @media print {
            body { background: white; }
            .section { break-inside: avoid; box-shadow: none; border: 1px solid #ddd; }
            .chart-container { page-break-inside: avoid; }
        }
        """

        # Build HTML with professional layout
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>Model Validation Report</title>",
            "<link href='https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono&display=swap' rel='stylesheet'>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            f"<style>{css}</style>",
            "</head><body>",
            "<div class='report-container'>",
            # Header
            "<header class='report-header'>",
            f"<h1>Model Validation Report</h1>",
            f"<div class='subtitle'>{self.model_name}</div>",
            f"<div class='date'>Generated: {self.validation_date}</div>",
            f"<span class='quality-badge {self.overall_quality}'>{self.overall_quality}</span>",
            "</header>",
        ]

        # Critical issues
        if self.critical_issues:
            html_parts.append("<div class='callout danger'>")
            html_parts.append("<h4>Critical Issues</h4>")
            html_parts.append("<ul>")
            for issue in self.critical_issues:
                html_parts.append(f"<li>{issue}</li>")
            html_parts.append("</ul></div>")

        # Warnings
        if self.warnings:
            html_parts.append("<div class='callout warning'>")
            html_parts.append("<h4>Warnings</h4>")
            html_parts.append("<ul>")
            for warning in self.warnings:
                html_parts.append(f"<li>{warning}</li>")
            html_parts.append("</ul></div>")

        # Convergence
        if self.convergence:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Convergence Diagnostics</h2>")
            # Metrics grid for key convergence stats
            conv = self.convergence
            conv_class = "success" if conv.converged else "danger"
            html_parts.append("<div class='metrics-grid'>")
            html_parts.append(
                f"<div class='metric-card {conv_class}'><div class='value'>{conv.divergences}</div><div class='label'>Divergences</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{conv.rhat_max:.4f}</div><div class='label'>Max R-hat</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{conv.ess_bulk_min:.0f}</div><div class='label'>Min ESS (bulk)</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{conv.ess_tail_min:.0f}</div><div class='label'>Min ESS (tail)</div></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<h3>Detailed Summary</h3>")
            html_parts.append(
                self.convergence.summary().to_html(index=False, classes="data-table")
            )
            html_parts.append("</section>")

        # PPC with charts
        if self.ppc:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Posterior Predictive Checks</h2>")
            ppc_class = "success" if self.ppc.overall_pass else "warning"
            html_parts.append(f"<div class='callout {ppc_class}'>")
            html_parts.append(
                f"<h4>{'All checks passed' if self.ppc.overall_pass else 'Some checks require attention'}</h4>"
            )
            if self.ppc.problematic_checks:
                html_parts.append(
                    f"<p>Problematic checks: {', '.join(self.ppc.problematic_checks)}</p>"
                )
            html_parts.append("</div>")
            html_parts.append(
                self.ppc.summary().to_html(index=False, classes="data-table")
            )

            # Check if we have valid data for charts
            def _has_data(arr):
                """Check if array has data."""
                if arr is None:
                    return False
                try:
                    return np.asarray(arr).size > 0
                except Exception:
                    return False

            has_ppc_data = (
                charts_available
                and _has_data(self.ppc.y_obs)
                and _has_data(self.ppc.y_rep)
            )

            if has_ppc_data:
                # PPC Time Series Plot
                html_parts.append("<h3>Time Series Comparison</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_ppc_time_series_plot(self.ppc.y_obs, self.ppc.y_rep)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

                # PPC Density Plot
                html_parts.append("<h3>Distribution Comparison</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_ppc_density_plot(self.ppc.y_obs, self.ppc.y_rep)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

            # PPC Statistics Plot (always try if we have checks)
            if charts_available and self.ppc.checks:
                html_parts.append("<h3>Test Statistics Comparison</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_ppc_statistics_plot(self.ppc.checks)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

            # PIT Histogram (Probability Integral Transform for calibration)
            if has_ppc_data:
                html_parts.append(
                    "<h3>Probability Integral Transform (Calibration)</h3>"
                )
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_pit_histogram(self.ppc.y_obs, self.ppc.y_rep)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

            html_parts.append("</section>")  # Close PPC section

        # Residuals with charts
        if self.residuals:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Residual Diagnostics</h2>")
            html_parts.append(
                self.residuals.summary().to_html(index=False, classes="data-table")
            )

            # Check if we have valid residual data
            def _has_data(arr):
                """Check if array has data."""
                if arr is None:
                    return False
                try:
                    return np.asarray(arr).size > 0
                except Exception:
                    return False

            has_residuals = charts_available and _has_data(self.residuals.residuals)

            if has_residuals:
                # Residuals over time
                html_parts.append("<h3>Residuals Over Time</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_residual_time_series_plot(self.residuals.residuals)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

                # Residuals vs Fitted
                has_fitted = (
                    self.residuals.fitted_values is not None
                    and len(self.residuals.fitted_values) > 0
                )
                if has_fitted:
                    html_parts.append("<h3>Residuals vs Fitted Values</h3>")
                    html_parts.append("<div class='chart-container'>")
                    try:
                        fig = create_residual_vs_fitted(
                            self.residuals.residuals, self.residuals.fitted_values
                        )
                        html_parts.append(
                            fig.to_html(full_html=False, include_plotlyjs=False)
                        )
                    except Exception as e:
                        html_parts.append(
                            f"<p><em>Chart could not be generated: {e}</em></p>"
                        )
                    html_parts.append("</div>")

                # Q-Q Plot
                html_parts.append("<h3>Q-Q Plot (Normality)</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_qq_plot(self.residuals.residuals)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

                # ACF Plot
                has_acf = (
                    self.residuals.acf_values is not None
                    and len(self.residuals.acf_values) > 0
                )
                if has_acf:
                    html_parts.append("<h3>Autocorrelation Function (ACF)</h3>")
                    html_parts.append("<div class='chart-container'>")
                    try:
                        pacf = (
                            self.residuals.pacf_values
                            if hasattr(self.residuals, "pacf_values")
                            else None
                        )
                        n_obs = len(self.residuals.residuals)
                        fig = create_acf_chart(
                            self.residuals.acf_values, pacf, n_obs=n_obs
                        )
                        html_parts.append(
                            fig.to_html(full_html=False, include_plotlyjs=False)
                        )
                    except Exception as e:
                        html_parts.append(
                            f"<p><em>Chart could not be generated: {e}</em></p>"
                        )
                    html_parts.append("</div>")

            html_parts.append("</section>")  # Close Residuals section

        # Channel diagnostics with charts
        if self.channel_diagnostics:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Channel Diagnostics</h2>")
            html_parts.append(
                self.channel_diagnostics.summary().to_html(
                    index=False, classes="data-table"
                )
            )

            if charts_available and self.channel_diagnostics.vif_scores:
                html_parts.append("<h3>Variance Inflation Factors (VIF)</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_vif_chart(self.channel_diagnostics)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

            html_parts.append("</section>")  # Close Channel Diagnostics section

        # Model comparison
        if self.model_comparison and self.model_comparison.models:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Model Comparison</h2>")
            html_parts.append(
                self.model_comparison.summary().to_html(
                    index=False, classes="data-table"
                )
            )
            html_parts.append("</section>")

        # Cross-validation
        if self.cross_validation:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Cross-Validation Results</h2>")
            cv = self.cross_validation
            # Metrics grid for CV summary
            html_parts.append("<div class='metrics-grid'>")
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{cv.strategy}</div><div class='label'>Strategy</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{cv.n_folds}</div><div class='label'>Folds</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{cv.mean_r2:.4f}</div><div class='label'>Mean R²</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{cv.mean_coverage:.1%}</div><div class='label'>Mean Coverage</div></div>"
            )
            html_parts.append("</div>")
            html_parts.append("<h3>Per-Fold Metrics</h3>")
            html_parts.append(
                cv.fold_summary().to_html(index=False, classes="data-table")
            )
            html_parts.append("<h3>Aggregate Performance</h3>")
            html_parts.append("<table class='data-table'>")
            html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
            html_parts.append(
                f"<tr><td>Mean RMSE</td><td class='mono'>{cv.mean_rmse:.4f}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Mean MAE</td><td class='mono'>{cv.mean_mae:.4f}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Mean MAPE</td><td class='mono'>{cv.mean_mape:.2f}%</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Mean R²</td><td class='mono'>{cv.mean_r2:.4f}</td></tr>"
            )
            html_parts.append(
                f"<tr><td>Mean Coverage</td><td class='mono'>{cv.mean_coverage:.1%}</td></tr>"
            )
            html_parts.append("</table>")

            # CV Charts
            if charts_available and cv.fold_results:
                # CV Fold Metrics Chart
                html_parts.append("<h3>Performance by Fold</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_cv_fold_metrics_chart(
                        cv.fold_results, cv.mean_rmse, cv.mean_mae, cv.mean_r2
                    )
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

                # CV Coverage Chart
                html_parts.append("<h3>Credible Interval Coverage by Fold</h3>")
                html_parts.append("<div class='chart-container'>")
                try:
                    fig = create_cv_coverage_chart(cv.fold_results, cv.mean_coverage)
                    html_parts.append(
                        fig.to_html(full_html=False, include_plotlyjs=False)
                    )
                except Exception as e:
                    html_parts.append(
                        f"<p><em>Chart could not be generated: {e}</em></p>"
                    )
                html_parts.append("</div>")

                # CV Actual vs Predicted Time Series
                # Check if prediction data is available (first fold has y_pred_mean)
                if (
                    cv.fold_results[0].y_pred_mean is not None
                    and self._full_y_actual is not None
                ):
                    html_parts.append("<h3>Actual vs Predicted by Fold</h3>")
                    html_parts.append("<div class='chart-container'>")
                    try:
                        fig = create_cv_actual_vs_predicted_chart(
                            cv.fold_results,
                            self._full_y_actual,
                            title="Cross-Validation: Actual vs Predicted (Out-of-Sample)",
                        )
                        html_parts.append(
                            fig.to_html(full_html=False, include_plotlyjs=False)
                        )
                    except Exception as e:
                        html_parts.append(
                            f"<p><em>Chart could not be generated: {e}</em></p>"
                        )
                    html_parts.append("</div>")

            html_parts.append("</section>")  # Close CV section

        # Sensitivity Analysis
        if self.sensitivity:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Sensitivity Analysis</h2>")
            sens = self.sensitivity
            n_robust = len(sens.robust_parameters)
            n_sensitive = len(sens.sensitive_parameters)
            status_class = (
                "success"
                if n_sensitive == 0
                else "warning" if n_sensitive < 3 else "danger"
            )
            html_parts.append(
                f"<div class='callout {status_class}'>"
                f"<h4>{'All parameters are robust' if n_sensitive == 0 else 'Some parameters show sensitivity'}</h4>"
                f"<p><strong>Robust Parameters:</strong> {n_robust} | "
                f"<strong>Sensitive Parameters:</strong> {n_sensitive}</p>"
                f"</div>"
            )
            html_parts.append("<h3>Base Estimates</h3>")
            html_parts.append("<table class='data-table'>")
            html_parts.append(
                "<tr><th>Parameter</th><th>Estimate</th><th>Sensitivity Index</th><th>Status</th></tr>"
            )
            for param, value in sens.base_estimates.items():
                sens_idx = sens.sensitivity_indices.get(param, 0)
                is_robust = param in sens.robust_parameters
                status = (
                    "<span class='positive'>Robust</span>"
                    if is_robust
                    else "<span class='negative'>Sensitive</span>"
                )
                html_parts.append(
                    f"<tr><td>{param}</td><td class='mono'>{value:.4f}</td><td class='mono'>{sens_idx:.3f}</td><td>{status}</td></tr>"
                )
            html_parts.append("</table>")
            if sens.sensitive_parameters:
                html_parts.append("<h3>Sensitive Parameters (Require Attention)</h3>")
                html_parts.append("<ul>")
                for p in sens.sensitive_parameters:
                    html_parts.append(f"<li class='warning-text'>{p}</li>")
                html_parts.append("</ul>")
            html_parts.append("</section>")

        # Stability Analysis
        if self.stability:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Stability Analysis</h2>")
            stab = self.stability
            score = stab.stability_score
            score_class = (
                "success" if score >= 0.8 else "warning" if score >= 0.6 else "danger"
            )
            n_influential = len(stab.influential_observations)
            html_parts.append("<div class='metrics-grid'>")
            html_parts.append(
                f"<div class='metric-card {score_class}'><div class='value'>{score:.2f}</div><div class='label'>Stability Score</div></div>"
            )
            html_parts.append(
                f"<div class='metric-card'><div class='value'>{n_influential}</div><div class='label'>Influential Observations</div></div>"
            )
            html_parts.append("</div>")
            if stab.influence_results:
                inf = stab.influence_results
                html_parts.append("<h3>Influence Diagnostics (LOO Pareto-k)</h3>")
                html_parts.append("<table class='data-table'>")
                html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
                html_parts.append(
                    f"<tr><td>Threshold</td><td class='mono'>{inf.influence_threshold}</td></tr>"
                )
                html_parts.append(
                    f"<tr><td>Max Pareto-k</td><td class='mono'>{inf.observation_influence.max():.3f}</td></tr>"
                )
                html_parts.append(
                    f"<tr><td>Mean Pareto-k</td><td class='mono'>{inf.observation_influence.mean():.3f}</td></tr>"
                )
                html_parts.append(
                    f"<tr><td>Observations > Threshold</td><td class='mono'>{n_influential}</td></tr>"
                )
                html_parts.append("</table>")
            if stab.bootstrap_results:
                boot = stab.bootstrap_results
                html_parts.append("<h3>Bootstrap Results</h3>")
                html_parts.append(
                    f"<p><strong>Bootstrap Iterations:</strong> {boot.n_bootstrap}</p>"
                )
                html_parts.append("<table class='data-table'>")
                html_parts.append(
                    "<tr><th>Parameter</th><th>Mean</th><th>Std</th><th>95% CI</th></tr>"
                )
                for param in boot.parameter_means:
                    mean = boot.parameter_means[param]
                    std = boot.parameter_stds[param]
                    ci_low = boot.parameter_ci_low[param]
                    ci_high = boot.parameter_ci_high[param]
                    html_parts.append(
                        f"<tr><td>{param}</td><td class='mono'>{mean:.4f}</td><td class='mono'>{std:.4f}</td>"
                        f"<td class='mono'>[{ci_low:.4f}, {ci_high:.4f}]</td></tr>"
                    )
                html_parts.append("</table>")
            html_parts.append("</section>")

        # Calibration
        if self.calibration:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Calibration Results</h2>")
            calib = self.calibration
            status_class = "success" if calib.calibrated else "danger"
            status_text = "CALIBRATED" if calib.calibrated else "NOT CALIBRATED"
            html_parts.append(
                f"<div class='callout {status_class}'>"
                f"<h4>{status_text}</h4>"
                f"<p><strong>Coverage Rate:</strong> {calib.coverage_rate:.1%} | "
                f"<strong>Mean Abs. Error:</strong> {calib.mean_absolute_calibration_error:.1%}</p>"
                f"</div>"
            )
            html_parts.append("<h3>Lift Test Comparisons</h3>")
            html_parts.append("<table class='data-table'>")
            html_parts.append(
                "<tr><th>Channel</th><th>Model Estimate</th><th>Model 94% CI</th>"
                "<th>Experimental</th><th>Within CI</th><th>Relative Error</th></tr>"
            )
            for comp in calib.lift_test_comparisons:
                within_class = "positive" if comp.within_ci else "negative"
                within_text = "Yes" if comp.within_ci else "No"
                html_parts.append(
                    f"<tr><td>{comp.channel}</td>"
                    f"<td class='mono'>{comp.model_estimate:,.0f}</td>"
                    f"<td class='mono'>[{comp.model_ci_low:,.0f}, {comp.model_ci_high:,.0f}]</td>"
                    f"<td class='mono'>{comp.experimental_estimate:,.0f} ± {comp.experimental_se:,.0f}</td>"
                    f"<td class='{within_class}'>{within_text}</td>"
                    f"<td class='mono'>{comp.relative_error:+.1%}</td></tr>"
                )
            html_parts.append("</table>")
            html_parts.append("</section>")

        # Recommendations
        if self.recommendations:
            html_parts.append("<section class='section'>")
            html_parts.append("<h2>Recommendations</h2>")
            html_parts.append("<div class='callout insight'>")
            html_parts.append("<h4>Suggested Actions</h4>")
            html_parts.append("<ul>")
            for rec in self.recommendations:
                html_parts.append(f"<li>{rec}</li>")
            html_parts.append("</ul></div>")
            html_parts.append("</section>")

        # Footer
        html_parts.append("<footer class='report-footer'>")
        html_parts.append("<p>Generated by MMM Framework Validation Module</p>")
        html_parts.append("</footer>")
        html_parts.append("</div>")  # Close report-container
        html_parts.append("</body></html>")
        return "\n".join(html_parts)


__all__ = [
    # Base types
    "TestResult",
    "ConvergenceSummary",
    # PPC
    "PPCCheckResult",
    "PPCResults",
    # Residuals
    "ResidualDiagnosticsResults",
    # Channel diagnostics
    "ChannelConvergenceResult",
    "ChannelDiagnosticsResults",
    # Model comparison
    "LOOResults",
    "WAICResults",
    "ModelComparisonEntry",
    "ModelComparisonResults",
    # Cross-validation
    "CVFoldResult",
    "CrossValidationResults",
    # Sensitivity
    "SensitivityResults",
    # Stability
    "BootstrapResults",
    "InfluenceResults",
    "StabilityResults",
    # Calibration
    "LiftTestResult",
    "LiftTestComparison",
    "CalibrationResults",
    # Summary
    "ValidationSummary",
]

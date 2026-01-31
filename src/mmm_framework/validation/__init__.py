"""
Model Validation Package for MMM Framework.

Provides comprehensive tools for verifying the robustness of marketing mix models:

- Posterior Predictive Checks (PPC)
- Residual Diagnostics
- Channel Diagnostics (VIF, convergence)
- Model Comparison (LOO-CV, WAIC)
- Cross-Validation (time-series CV)
- Sensitivity Analysis
- Stability Analysis
- Experimental Calibration

Examples
--------
Quick validation:

>>> from mmm_framework.validation import ModelValidator
>>> validator = ModelValidator(model, results)
>>> summary = validator.quick_check()
>>> print(summary.overall_quality)
>>> print(summary.recommendations)

Standard validation with model comparison:

>>> from mmm_framework.validation import ValidationConfig
>>> config = ValidationConfig.standard()
>>> summary = validator.validate(config)

Thorough validation with custom settings:

>>> from mmm_framework.validation import ValidationConfigBuilder
>>> config = (ValidationConfigBuilder()
...     .thorough()
...     .with_cross_validation(n_folds=5, strategy="expanding")
...     .with_residual_tests(("durbin_watson", "ljung_box"))
...     .build())
>>> summary = validator.validate(config)

Generate HTML report:

>>> html = summary.to_html_report()
>>> with open("validation_report.html", "w") as f:
...     f.write(html)
"""

from .builders import ValidationConfigBuilder
from .channel_diagnostics import ChannelDiagnostics
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
from .posterior_predictive import PPCValidator
from .protocols import (
    HasControlData,
    HasMediaData,
    HasPanelData,
    HasScalingParams,
    Validatable,
)
from .residual_diagnostics import ResidualDiagnostics
from .results import (
    BootstrapResults,
    CalibrationResults,
    ChannelConvergenceResult,
    ChannelDiagnosticsResults,
    ConvergenceSummary,
    CrossValidationResults,
    CVFoldResult,
    InfluenceResults,
    LiftTestComparison,
    LiftTestResult,
    LOOResults,
    ModelComparisonEntry,
    ModelComparisonResults,
    PPCCheckResult,
    PPCResults,
    ResidualDiagnosticsResults,
    SensitivityResults,
    StabilityResults,
    TestResult,
    ValidationSummary,
    WAICResults,
)
from .validator import ModelValidator

__all__ = [
    # Main classes
    "ModelValidator",
    "ValidationConfigBuilder",
    # Validators
    "PPCValidator",
    "ResidualDiagnostics",
    "ChannelDiagnostics",
    # Config
    "ValidationConfig",
    "ValidationLevel",
    "PPCConfig",
    "ResidualConfig",
    "ChannelDiagnosticsConfig",
    "CrossValidationConfig",
    "ModelComparisonConfig",
    "SensitivityConfig",
    "StabilityConfig",
    "CalibrationConfig",
    # Results
    "ValidationSummary",
    "ConvergenceSummary",
    "TestResult",
    "PPCCheckResult",
    "PPCResults",
    "ResidualDiagnosticsResults",
    "ChannelConvergenceResult",
    "ChannelDiagnosticsResults",
    "LOOResults",
    "WAICResults",
    "ModelComparisonEntry",
    "ModelComparisonResults",
    "CVFoldResult",
    "CrossValidationResults",
    "SensitivityResults",
    "BootstrapResults",
    "InfluenceResults",
    "StabilityResults",
    "LiftTestResult",
    "LiftTestComparison",
    "CalibrationResults",
    # Protocols
    "Validatable",
    "HasMediaData",
    "HasPanelData",
    "HasScalingParams",
    "HasControlData",
]

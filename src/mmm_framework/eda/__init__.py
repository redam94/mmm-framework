"""
Pre-fit data quality for MMM: validation, outlier detection + remediation,
and exploratory data analysis (EDA) with plotly charts.

This package is the pre-fit sibling of :mod:`mmm_framework.validation`
(which diagnoses a *fitted* model). Everything here runs on the raw dataset
before ``BayesianMMM`` sees it.
"""

from .collinearity import collinearity_analysis
from .config import DataValidationConfig, EDAConfig, OutlierConfig
from .decomposition import decompose_series, decomposition_summary, stationarity_tests
from .loading import EDAPanel, load_eda_panel, seasonal_period_for_freq
from .outliers import OutlierDetector, detect_outliers
from .profiling import missingness_matrix, profile_panel, spend_share
from .remediation import apply_treatments, recommend_treatments
from .validators import DataValidator, validate_dataset
from .results import (
    DataValidationReport,
    DecompositionResult,
    EDAReport,
    OutlierFlag,
    OutlierReport,
    RemediationAction,
    ValidationIssue,
)

__all__ = [
    "OutlierConfig",
    "DataValidationConfig",
    "EDAConfig",
    "EDAPanel",
    "load_eda_panel",
    "seasonal_period_for_freq",
    "OutlierFlag",
    "RemediationAction",
    "OutlierReport",
    "ValidationIssue",
    "DataValidationReport",
    "DecompositionResult",
    "EDAReport",
    "OutlierDetector",
    "detect_outliers",
    "recommend_treatments",
    "apply_treatments",
    "decompose_series",
    "decomposition_summary",
    "stationarity_tests",
    "DataValidator",
    "validate_dataset",
    "profile_panel",
    "missingness_matrix",
    "spend_share",
    "collinearity_analysis",
]

"""
Tests for the validation module.

This module contains unit tests for:
- Cross-validation split generation
- CV metrics computation
- Sensitivity analysis utilities
- Stability analysis utilities
- Calibration utilities
- Result dataclasses
"""

import numpy as np
import pandas as pd
import pytest

from mmm_framework.validation import (
    ValidationConfig,
    ValidationConfigBuilder,
    ValidationLevel,
    CrossValidationConfig,
    SensitivityConfig,
    StabilityConfig,
    CalibrationConfig,
    PPCConfig,
    ResidualConfig,
    ChannelDiagnosticsConfig,
)
from mmm_framework.validation.results import (
    CVFoldResult,
    CrossValidationResults,
    SensitivityResults,
    StabilityResults,
    BootstrapResults,
    InfluenceResults,
    CalibrationResults,
    LiftTestResult,
    LiftTestComparison,
    TestResult,
    ConvergenceSummary,
    ValidationSummary,
)


class TestValidationConfig:
    """Tests for ValidationConfig and related configuration classes."""

    def test_quick_config(self):
        """Test quick validation configuration."""
        config = ValidationConfig.quick()

        assert config.level == ValidationLevel.QUICK
        assert config.run_ppc is True
        assert config.run_residuals is True
        assert config.run_channel_diagnostics is True
        assert config.run_model_comparison is False
        assert config.run_cross_validation is False
        assert config.run_sensitivity is False
        assert config.run_stability is False
        assert config.run_calibration is False

    def test_standard_config(self):
        """Test standard validation configuration."""
        config = ValidationConfig.standard()

        assert config.level == ValidationLevel.STANDARD
        assert config.run_model_comparison is True
        assert config.run_cross_validation is False

    def test_thorough_config(self):
        """Test thorough validation configuration."""
        config = ValidationConfig.thorough()

        assert config.level == ValidationLevel.THOROUGH
        assert config.run_cross_validation is True
        assert config.run_sensitivity is True
        assert config.run_stability is True


class TestValidationConfigBuilder:
    """Tests for ValidationConfigBuilder."""

    def test_builder_quick(self):
        """Test builder quick configuration."""
        config = ValidationConfigBuilder().quick().build()

        assert config.level == ValidationLevel.QUICK
        assert config.run_ppc is True

    def test_builder_with_cross_validation(self):
        """Test builder with cross-validation settings."""
        config = (
            ValidationConfigBuilder()
            .standard()
            .with_cross_validation(n_folds=5, strategy="expanding", min_train_size=26)
            .build()
        )

        assert config.run_cross_validation is True
        assert config.cross_validation.n_folds == 5
        assert config.cross_validation.strategy == "expanding"
        assert config.cross_validation.min_train_size == 26

    def test_builder_with_sensitivity(self):
        """Test builder with sensitivity settings."""
        config = (
            ValidationConfigBuilder()
            .with_sensitivity_analysis(
                prior_multipliers=(0.5, 1.5, 2.0),
                parameters_of_interest=("beta_tv", "beta_digital"),
            )
            .build()
        )

        assert config.run_sensitivity is True
        assert config.sensitivity.prior_multipliers == (0.5, 1.5, 2.0)
        assert config.sensitivity.parameters_of_interest == ("beta_tv", "beta_digital")

    def test_builder_with_stability(self):
        """Test builder with stability settings."""
        config = (
            ValidationConfigBuilder()
            .with_stability_analysis(n_bootstrap=50, perturbation_level=0.15)
            .build()
        )

        assert config.run_stability is True
        assert config.stability.n_bootstrap == 50
        assert config.stability.perturbation_level == 0.15

    def test_builder_with_calibration(self):
        """Test builder with calibration settings."""
        lift_tests = [
            LiftTestResult(
                channel="tv_spend",
                test_period=("2023-01-01", "2023-03-31"),
                measured_lift=10000,
                lift_se=2000,
            ),
        ]

        config = (
            ValidationConfigBuilder()
            .with_calibration(lift_tests=lift_tests, ci_level=0.90)
            .build()
        )

        assert config.run_calibration is True
        assert config.calibration.ci_level == 0.90
        assert len(config.lift_tests) == 1

    def test_builder_without_options(self):
        """Test builder disable options."""
        config = (
            ValidationConfigBuilder()
            .thorough()
            .without_ppc()
            .without_residuals()
            .without_plots()
            .silent()
            .build()
        )

        assert config.run_ppc is False
        assert config.run_residuals is False
        assert config.generate_plots is False
        assert config.verbose is False


class TestCVFoldResult:
    """Tests for CVFoldResult dataclass."""

    def test_fold_result_creation(self):
        """Test creating a CV fold result."""
        result = CVFoldResult(
            fold_idx=0,
            train_size=80,
            test_size=20,
            rmse=5.5,
            mae=4.2,
            mape=8.5,
            r2=0.85,
            coverage=0.92,
        )

        assert result.fold_idx == 0
        assert result.train_size == 80
        assert result.test_size == 20
        assert result.rmse == 5.5
        assert result.r2 == 0.85

    def test_fold_result_to_dict(self):
        """Test fold result serialization."""
        result = CVFoldResult(
            fold_idx=0,
            train_size=80,
            test_size=20,
            rmse=5.5,
            mae=4.2,
            mape=8.5,
            r2=0.85,
            coverage=0.92,
        )

        result_dict = result.to_dict()

        assert result_dict["fold_idx"] == 0
        assert result_dict["rmse"] == 5.5


class TestCrossValidationResults:
    """Tests for CrossValidationResults dataclass."""

    def test_cv_results_creation(self):
        """Test creating CV results with multiple folds."""
        folds = [
            CVFoldResult(0, 60, 20, 5.0, 4.0, 8.0, 0.80, 0.90),
            CVFoldResult(1, 80, 20, 5.5, 4.2, 8.5, 0.85, 0.92),
            CVFoldResult(2, 100, 20, 4.8, 3.8, 7.5, 0.88, 0.94),
        ]

        results = CrossValidationResults(
            strategy="expanding",
            n_folds=3,
            fold_results=folds,
        )

        assert results.strategy == "expanding"
        assert results.n_folds == 3
        assert len(results.fold_results) == 3

        # Check computed aggregates
        assert abs(results.mean_rmse - 5.1) < 0.1
        assert abs(results.mean_r2 - 0.843) < 0.01

    def test_cv_results_summary(self):
        """Test CV results summary DataFrame."""
        folds = [
            CVFoldResult(0, 60, 20, 5.0, 4.0, 8.0, 0.80, 0.90),
            CVFoldResult(1, 80, 20, 5.5, 4.2, 8.5, 0.85, 0.92),
        ]

        results = CrossValidationResults(
            strategy="expanding",
            n_folds=2,
            fold_results=folds,
        )

        summary = results.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "RMSE" in summary["Metric"].values


class TestSensitivityResults:
    """Tests for SensitivityResults dataclass."""

    def test_sensitivity_results_creation(self):
        """Test creating sensitivity results."""
        base_estimates = {"beta_tv": 0.5, "beta_digital": 0.3}
        variant_estimates = {
            "prior_x0.5": {"beta_tv": 0.45, "beta_digital": 0.28},
            "prior_x2.0": {"beta_tv": 0.55, "beta_digital": 0.32},
        }
        sensitivity_indices = {"beta_tv": 0.1, "beta_digital": 0.08}

        results = SensitivityResults(
            base_estimates=base_estimates,
            variant_estimates=variant_estimates,
            sensitivity_indices=sensitivity_indices,
            robust_parameters=["beta_tv", "beta_digital"],
            sensitive_parameters=[],
        )

        assert results.base_estimates["beta_tv"] == 0.5
        assert len(results.robust_parameters) == 2
        assert len(results.sensitive_parameters) == 0

    def test_sensitivity_results_summary(self):
        """Test sensitivity results summary DataFrame."""
        results = SensitivityResults(
            base_estimates={"beta_tv": 0.5},
            variant_estimates={"prior_x2.0": {"beta_tv": 0.55}},
            sensitivity_indices={"beta_tv": 0.1},
            robust_parameters=["beta_tv"],
            sensitive_parameters=[],
        )

        summary = results.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Parameter" in summary.columns


class TestStabilityResults:
    """Tests for StabilityResults dataclass."""

    def test_stability_results_creation(self):
        """Test creating stability results."""
        bootstrap = BootstrapResults(
            n_bootstrap=100,
            parameter_means={"beta_tv": 0.5},
            parameter_stds={"beta_tv": 0.05},
            parameter_ci_low={"beta_tv": 0.4},
            parameter_ci_high={"beta_tv": 0.6},
        )

        influence = InfluenceResults(
            observation_influence=np.array([0.3, 0.5, 0.8, 0.4]),
            influential_indices=[2],
            influence_threshold=0.7,
        )

        results = StabilityResults(
            bootstrap_results=bootstrap,
            influence_results=influence,
            influential_observations=[2],
            stability_score=0.85,
        )

        assert results.stability_score == 0.85
        assert len(results.influential_observations) == 1

    def test_stability_results_without_bootstrap(self):
        """Test stability results without bootstrap (LOO-only)."""
        results = StabilityResults(
            bootstrap_results=None,
            influence_results=None,
            influential_observations=[],
            stability_score=1.0,
        )

        assert results.stability_score == 1.0
        assert results.bootstrap_results is None


class TestCalibrationResults:
    """Tests for CalibrationResults dataclass."""

    def test_calibration_results_creation(self):
        """Test creating calibration results."""
        comparisons = [
            LiftTestComparison(
                channel="tv_spend",
                model_estimate=10500,
                model_ci_low=8000,
                model_ci_high=13000,
                experimental_estimate=10000,
                experimental_se=2000,
                within_ci=True,
                relative_error=0.05,
            ),
            LiftTestComparison(
                channel="digital_spend",
                model_estimate=6000,
                model_ci_low=4000,
                model_ci_high=8000,
                experimental_estimate=5500,
                experimental_se=1000,
                within_ci=True,
                relative_error=0.09,
            ),
        ]

        results = CalibrationResults(
            lift_test_comparisons=comparisons,
            coverage_rate=1.0,
            mean_absolute_calibration_error=0.07,
        )

        assert results.calibrated is True  # coverage >= 0.5
        assert results.coverage_rate == 1.0

    def test_calibration_results_summary(self):
        """Test calibration results summary DataFrame."""
        comparisons = [
            LiftTestComparison(
                channel="tv_spend",
                model_estimate=10500,
                model_ci_low=8000,
                model_ci_high=13000,
                experimental_estimate=10000,
                experimental_se=2000,
                within_ci=True,
                relative_error=0.05,
            ),
        ]

        results = CalibrationResults(
            lift_test_comparisons=comparisons,
            coverage_rate=1.0,
            mean_absolute_calibration_error=0.05,
        )

        summary = results.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Channel" in summary.columns
        assert "Model Estimate" in summary.columns


class TestLiftTestResult:
    """Tests for LiftTestResult dataclass."""

    def test_lift_test_result_creation(self):
        """Test creating a lift test result."""
        lift_test = LiftTestResult(
            channel="tv_spend",
            test_period=("2023-01-01", "2023-03-31"),
            measured_lift=15000,
            lift_se=3000,
            holdout_regions=["Northeast", "Southeast"],
            confidence_level=0.95,
        )

        assert lift_test.channel == "tv_spend"
        assert lift_test.measured_lift == 15000
        assert len(lift_test.holdout_regions) == 2


class TestConvergenceSummary:
    """Tests for ConvergenceSummary dataclass."""

    def test_convergence_summary_converged(self):
        """Test convergence summary for converged model."""
        summary = ConvergenceSummary(
            divergences=0,
            rhat_max=1.005,
            ess_bulk_min=500,
            ess_tail_min=450,
            converged=True,
        )

        assert summary.converged is True
        assert summary.divergences == 0

    def test_convergence_summary_not_converged(self):
        """Test convergence summary for non-converged model."""
        summary = ConvergenceSummary(
            divergences=50,
            rhat_max=1.15,
            ess_bulk_min=100,
            ess_tail_min=80,
            converged=False,
        )

        assert summary.converged is False
        assert summary.divergences == 50

    def test_convergence_summary_dataframe(self):
        """Test convergence summary DataFrame generation."""
        summary = ConvergenceSummary(
            divergences=0,
            rhat_max=1.005,
            ess_bulk_min=500,
            ess_tail_min=450,
            converged=True,
        )

        df = summary.summary()

        assert isinstance(df, pd.DataFrame)
        assert "Metric" in df.columns
        assert "Value" in df.columns
        assert "Status" in df.columns


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""

    def test_validation_summary_creation(self):
        """Test creating a validation summary."""
        convergence = ConvergenceSummary(
            divergences=0,
            rhat_max=1.005,
            ess_bulk_min=500,
            ess_tail_min=450,
            converged=True,
        )

        summary = ValidationSummary(
            model_name="BayesianMMM",
            convergence=convergence,
            overall_quality="good",
            critical_issues=[],
            warnings=["Minor autocorrelation detected"],
            recommendations=["Consider adding lag terms"],
        )

        assert summary.model_name == "BayesianMMM"
        assert summary.overall_quality == "good"
        assert len(summary.warnings) == 1

    def test_validation_summary_to_dict(self):
        """Test validation summary serialization."""
        summary = ValidationSummary(
            model_name="BayesianMMM",
            overall_quality="acceptable",
        )

        result_dict = summary.to_dict()

        assert result_dict["model_name"] == "BayesianMMM"
        assert result_dict["overall_quality"] == "acceptable"


class TestCVSplitGeneration:
    """Tests for CV split generation logic."""

    def test_expanding_window_splits(self):
        """Test expanding window CV split generation."""
        from mmm_framework.validation.validator import ModelValidator

        # Create a minimal mock for testing split generation
        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _create_cv_splits(self, n_obs, cv_config):
                return ModelValidator._create_cv_splits(self, n_obs, cv_config)

        validator = MockValidator()

        cv_config = CrossValidationConfig(
            strategy="expanding",
            n_folds=5,
            min_train_size=52,
            gap=0,
        )

        splits = validator._create_cv_splits(100, cv_config)

        assert len(splits) > 0

        # Check that train and test don't overlap
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0

        # Check that train comes before test
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()

    def test_rolling_window_splits(self):
        """Test rolling window CV split generation."""
        from mmm_framework.validation.validator import ModelValidator

        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _create_cv_splits(self, n_obs, cv_config):
                return ModelValidator._create_cv_splits(self, n_obs, cv_config)

        validator = MockValidator()

        cv_config = CrossValidationConfig(
            strategy="rolling",
            n_folds=5,
            min_train_size=40,
            gap=0,
            test_size=10,
        )

        splits = validator._create_cv_splits(100, cv_config)

        assert len(splits) > 0

        # Check fixed training window size
        for train_idx, test_idx in splits:
            assert len(train_idx) == 40


class TestCVMetricsComputation:
    """Tests for CV metrics computation."""

    def test_compute_cv_fold_metrics(self):
        """Test CV fold metrics computation."""
        from mmm_framework.validation.validator import ModelValidator

        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _compute_cv_fold_metrics(
                self, fold_idx, train_idx, test_idx, y_true, y_pred, y_samples, ci_level
            ):
                return ModelValidator._compute_cv_fold_metrics(
                    self,
                    fold_idx,
                    train_idx,
                    test_idx,
                    y_true,
                    y_pred,
                    y_samples,
                    ci_level,
                )

        validator = MockValidator()

        # Create test data
        np.random.seed(42)
        y_true = np.array([100, 105, 110, 108, 112])
        y_pred = np.array([102, 103, 108, 110, 115])
        y_samples = np.random.normal(loc=y_pred, scale=5, size=(100, 5))

        train_idx = np.arange(80)
        test_idx = np.arange(80, 85)

        result = validator._compute_cv_fold_metrics(
            fold_idx=0,
            train_idx=train_idx,
            test_idx=test_idx,
            y_true=y_true,
            y_pred=y_pred,
            y_samples=y_samples,
            ci_level=0.94,
        )

        assert isinstance(result, CVFoldResult)
        assert result.fold_idx == 0
        assert result.train_size == 80
        assert result.test_size == 5
        assert result.rmse > 0
        assert 0 <= result.coverage <= 1


class TestSensitivityIndexComputation:
    """Tests for sensitivity index computation."""

    def test_compute_sensitivity_indices(self):
        """Test sensitivity index computation."""
        from mmm_framework.validation.validator import ModelValidator

        class MockValidator:
            def _compute_sensitivity_indices(self, base_estimates, variant_estimates):
                return ModelValidator._compute_sensitivity_indices(
                    self, base_estimates, variant_estimates
                )

        validator = MockValidator()

        base_estimates = {"beta_tv": 0.5, "beta_digital": 0.3}
        variant_estimates = {
            "prior_x0.5": {"beta_tv": 0.45, "beta_digital": 0.28},
            "prior_x2.0": {"beta_tv": 0.55, "beta_digital": 0.32},
        }

        indices = validator._compute_sensitivity_indices(
            base_estimates, variant_estimates
        )

        assert "beta_tv" in indices
        assert "beta_digital" in indices
        assert all(v >= 0 for v in indices.values())

    def test_sensitivity_index_zero_mean(self):
        """Test sensitivity index with near-zero mean."""
        from mmm_framework.validation.validator import ModelValidator

        class MockValidator:
            def _compute_sensitivity_indices(self, base_estimates, variant_estimates):
                return ModelValidator._compute_sensitivity_indices(
                    self, base_estimates, variant_estimates
                )

        validator = MockValidator()

        base_estimates = {"param": 0.0}
        variant_estimates = {"var1": {"param": 0.01}, "var2": {"param": -0.01}}

        indices = validator._compute_sensitivity_indices(
            base_estimates, variant_estimates
        )

        assert "param" in indices
        assert np.isfinite(indices["param"])


class TestHTMLReportGeneration:
    """Tests for HTML report generation with new validation sections."""

    def test_html_report_with_cross_validation(self):
        """Test HTML report includes cross-validation section."""
        folds = [
            CVFoldResult(0, 60, 20, 5.0, 4.0, 8.0, 0.80, 0.90),
            CVFoldResult(1, 80, 20, 5.5, 4.2, 8.5, 0.85, 0.92),
        ]

        cv_results = CrossValidationResults(
            strategy="expanding",
            n_folds=2,
            fold_results=folds,
        )

        summary = ValidationSummary(
            model_name="TestModel",
            overall_quality="good",
            cross_validation=cv_results,
        )

        html = summary.to_html_report(include_charts=False)

        assert "Cross-Validation Results" in html
        assert "expanding" in html
        assert "Mean RMSE" in html
        assert "Mean R²" in html
        assert "Per-Fold Metrics" in html

    def test_html_report_with_sensitivity(self):
        """Test HTML report includes sensitivity analysis section."""
        sens_results = SensitivityResults(
            base_estimates={"beta_tv": 0.5, "beta_digital": 0.3},
            variant_estimates={"prior_x2.0": {"beta_tv": 0.55, "beta_digital": 0.32}},
            sensitivity_indices={"beta_tv": 0.1, "beta_digital": 0.08},
            robust_parameters=["beta_tv", "beta_digital"],
            sensitive_parameters=[],
        )

        summary = ValidationSummary(
            model_name="TestModel",
            overall_quality="good",
            sensitivity=sens_results,
        )

        html = summary.to_html_report(include_charts=False)

        assert "Sensitivity Analysis" in html
        assert "Robust Parameters" in html
        assert "Sensitivity Index" in html
        assert "beta_tv" in html

    def test_html_report_with_stability(self):
        """Test HTML report includes stability analysis section."""
        influence = InfluenceResults(
            observation_influence=np.array([0.3, 0.5, 0.8, 0.4]),
            influential_indices=[2],
            influence_threshold=0.7,
        )

        stab_results = StabilityResults(
            bootstrap_results=None,
            influence_results=influence,
            influential_observations=[2],
            stability_score=0.85,
        )

        summary = ValidationSummary(
            model_name="TestModel",
            overall_quality="good",
            stability=stab_results,
        )

        html = summary.to_html_report(include_charts=False)

        assert "Stability Analysis" in html
        assert "Stability Score" in html
        assert "0.85" in html
        assert "Influence Diagnostics" in html
        assert "Pareto-k" in html

    def test_html_report_with_calibration(self):
        """Test HTML report includes calibration section."""
        comparisons = [
            LiftTestComparison(
                channel="tv_spend",
                model_estimate=10500,
                model_ci_low=8000,
                model_ci_high=13000,
                experimental_estimate=10000,
                experimental_se=2000,
                within_ci=True,
                relative_error=0.05,
            ),
        ]

        calib_results = CalibrationResults(
            lift_test_comparisons=comparisons,
            coverage_rate=1.0,
            mean_absolute_calibration_error=0.05,
        )

        summary = ValidationSummary(
            model_name="TestModel",
            overall_quality="good",
            calibration=calib_results,
        )

        html = summary.to_html_report(include_charts=False)

        assert "Calibration Results" in html
        assert "CALIBRATED" in html
        assert "Coverage Rate" in html
        assert "Lift Test Comparisons" in html
        assert "tv_spend" in html

    def test_html_report_with_all_sections(self):
        """Test HTML report with all validation sections."""
        convergence = ConvergenceSummary(
            divergences=0,
            rhat_max=1.005,
            ess_bulk_min=500,
            ess_tail_min=450,
            converged=True,
        )

        cv_results = CrossValidationResults(
            strategy="expanding",
            n_folds=2,
            fold_results=[CVFoldResult(0, 60, 20, 5.0, 4.0, 8.0, 0.80, 0.90)],
        )

        sens_results = SensitivityResults(
            base_estimates={"beta_tv": 0.5},
            variant_estimates={},
            sensitivity_indices={"beta_tv": 0.1},
            robust_parameters=["beta_tv"],
            sensitive_parameters=[],
        )

        stab_results = StabilityResults(
            bootstrap_results=None,
            influence_results=None,
            influential_observations=[],
            stability_score=0.95,
        )

        calib_results = CalibrationResults(
            lift_test_comparisons=[
                LiftTestComparison(
                    channel="tv",
                    model_estimate=1000,
                    model_ci_low=800,
                    model_ci_high=1200,
                    experimental_estimate=950,
                    experimental_se=100,
                    within_ci=True,
                    relative_error=0.05,
                ),
            ],
            coverage_rate=1.0,
            mean_absolute_calibration_error=0.05,
        )

        summary = ValidationSummary(
            model_name="FullTestModel",
            overall_quality="good",
            convergence=convergence,
            cross_validation=cv_results,
            sensitivity=sens_results,
            stability=stab_results,
            calibration=calib_results,
        )

        html = summary.to_html_report(include_charts=False)

        # Check all sections present
        assert "Convergence Diagnostics" in html
        assert "Cross-Validation Results" in html
        assert "Sensitivity Analysis" in html
        assert "Stability Analysis" in html
        assert "Calibration Results" in html


class TestStabilityScoreComputation:
    """Tests for stability score computation."""

    def test_compute_stability_score_no_influential(self):
        """Test stability score with no influential observations."""
        from mmm_framework.validation.validator import ModelValidator

        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _compute_stability_score(
                self, n_influential, n_total, bootstrap_results=None
            ):
                return ModelValidator._compute_stability_score(
                    self, n_influential, n_total, bootstrap_results
                )

        validator = MockValidator()

        score = validator._compute_stability_score(n_influential=0, n_total=100)

        assert score == 1.0

    def test_compute_stability_score_with_influential(self):
        """Test stability score with influential observations."""
        from mmm_framework.validation.validator import ModelValidator

        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _compute_stability_score(
                self, n_influential, n_total, bootstrap_results=None
            ):
                return ModelValidator._compute_stability_score(
                    self, n_influential, n_total, bootstrap_results
                )

        validator = MockValidator()

        score = validator._compute_stability_score(n_influential=10, n_total=100)

        assert 0 < score < 1.0

    def test_compute_stability_score_with_bootstrap(self):
        """Test stability score with bootstrap results."""
        from mmm_framework.validation.validator import ModelValidator

        class MockModel:
            n_obs = 100

        class MockValidator:
            model = MockModel()

            def _compute_stability_score(
                self, n_influential, n_total, bootstrap_results=None
            ):
                return ModelValidator._compute_stability_score(
                    self, n_influential, n_total, bootstrap_results
                )

        validator = MockValidator()

        bootstrap = BootstrapResults(
            n_bootstrap=100,
            parameter_means={"beta_tv": 0.5, "beta_digital": 0.3},
            parameter_stds={"beta_tv": 0.05, "beta_digital": 0.03},
            parameter_ci_low={"beta_tv": 0.4, "beta_digital": 0.24},
            parameter_ci_high={"beta_tv": 0.6, "beta_digital": 0.36},
        )

        score = validator._compute_stability_score(
            n_influential=5,
            n_total=100,
            bootstrap_results=bootstrap,
        )

        assert 0 <= score <= 1.0

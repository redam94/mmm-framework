"""
Tests for analysis module.

These tests ensure that the MMMAnalyzer class and analysis utilities
work correctly for analyzing fitted BayesianMMM models.
"""

import numpy as np
import pandas as pd
import pytest


class TestMarginalAnalysisResult:
    """Tests for MarginalAnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation of MarginalAnalysisResult."""
        from mmm_framework.analysis import MarginalAnalysisResult

        result = MarginalAnalysisResult(
            channel="TV",
            current_spend=10000.0,
            spend_increase=1000.0,
            spend_increase_pct=10.0,
            marginal_contribution=500.0,
            marginal_roas=0.5,
        )

        assert result.channel == "TV"
        assert result.current_spend == 10000.0
        assert result.spend_increase == 1000.0
        assert result.spend_increase_pct == 10.0
        assert result.marginal_contribution == 500.0
        assert result.marginal_roas == 0.5


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation of ScenarioResult."""
        from mmm_framework.analysis import ScenarioResult

        result = ScenarioResult(
            baseline_outcome=1000.0,
            scenario_outcome=1100.0,
            outcome_change=100.0,
            outcome_change_pct=10.0,
            spend_changes={"TV": {"original": 500, "scenario": 600, "change": 100}},
            baseline_prediction=np.zeros(50),
            scenario_prediction=np.ones(50),
        )

        assert result.baseline_outcome == 1000.0
        assert result.scenario_outcome == 1100.0
        assert result.outcome_change == 100.0
        assert result.outcome_change_pct == 10.0
        assert "TV" in result.spend_changes


class TestMMMAnalyzerImports:
    """Tests for MMMAnalyzer imports."""

    def test_can_import_analyzer(self):
        """Test that MMMAnalyzer can be imported."""
        from mmm_framework.analysis import MMMAnalyzer

        assert MMMAnalyzer is not None

    def test_can_import_result_classes(self):
        """Test that result classes can be imported."""
        from mmm_framework.analysis import (
            MarginalAnalysisResult,
            ScenarioResult,
        )

        assert MarginalAnalysisResult is not None
        assert ScenarioResult is not None

    def test_can_import_helper_functions(self):
        """Test that helper functions can be imported."""
        from mmm_framework.analysis import (
            compute_contribution_summary,
            compute_period_contributions,
        )

        assert callable(compute_contribution_summary)
        assert callable(compute_period_contributions)


class TestMMMAnalyzerValidation:
    """Tests for MMMAnalyzer validation."""

    def test_raises_on_unfitted_model(self):
        """Test that analyzer raises on unfitted model."""
        from mmm_framework.analysis import MMMAnalyzer

        # Create mock unfitted model
        class MockUnfittedModel:
            _trace = None

        with pytest.raises(ValueError, match="not fitted"):
            MMMAnalyzer(MockUnfittedModel())


class TestMMMAnalyzerProperties:
    """Tests for MMMAnalyzer properties."""

    def test_channel_names_property(self):
        """Test channel_names property."""
        from mmm_framework.analysis import MMMAnalyzer

        # Create mock fitted model
        class MockFittedModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Radio", "Digital"]

        analyzer = MMMAnalyzer(MockFittedModel())
        assert analyzer.channel_names == ["TV", "Radio", "Digital"]

    def test_n_obs_property(self):
        """Test n_obs property."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockFittedModel:
            _trace = "mock_trace"
            n_obs = 100

        analyzer = MMMAnalyzer(MockFittedModel())
        assert analyzer.n_obs == 100


class TestMMMAnalyzerTimeMask:
    """Tests for MMMAnalyzer time mask."""

    def test_get_time_mask_delegates(self):
        """Test that get_time_mask delegates to model."""
        from mmm_framework.analysis import MMMAnalyzer

        expected_mask = np.array([True, True, False, False, True])

        class MockModel:
            _trace = "mock_trace"

            def _get_time_mask(self, time_period):
                return expected_mask

        analyzer = MMMAnalyzer(MockModel())
        mask = analyzer.get_time_mask((0, 10))

        np.testing.assert_array_equal(mask, expected_mask)


class TestContributionSummary:
    """Tests for compute_contribution_summary function."""

    def test_basic_summary(self):
        """Test basic contribution summary."""
        from mmm_framework.analysis import compute_contribution_summary

        # Create mock contribution results
        class MockContributionResults:
            def summary(self):
                return pd.DataFrame({
                    "Channel": ["TV", "Radio"],
                    "Contribution": [100, 50],
                })

        results = MockContributionResults()
        summary = compute_contribution_summary(results)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_available(self):
        """Test that main classes are importable."""
        from mmm_framework import analysis

        # Check main classes
        assert hasattr(analysis, "MMMAnalyzer")
        assert hasattr(analysis, "MarginalAnalysisResult")
        assert hasattr(analysis, "ScenarioResult")

        # Check functions
        assert hasattr(analysis, "compute_contribution_summary")
        assert hasattr(analysis, "compute_period_contributions")

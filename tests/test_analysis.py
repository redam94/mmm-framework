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


class TestMarginalAnalysisResultMethods:
    """Additional tests for MarginalAnalysisResult."""

    def test_result_with_negative_roas(self):
        """Test result with negative ROAS."""
        from mmm_framework.analysis import MarginalAnalysisResult

        result = MarginalAnalysisResult(
            channel="TV",
            current_spend=10000.0,
            spend_increase=1000.0,
            spend_increase_pct=10.0,
            marginal_contribution=-200.0,
            marginal_roas=-0.2,
        )

        assert result.marginal_roas < 0
        assert result.marginal_contribution < 0

    def test_result_with_zero_spend(self):
        """Test result with zero current spend."""
        from mmm_framework.analysis import MarginalAnalysisResult

        result = MarginalAnalysisResult(
            channel="New Channel",
            current_spend=0.0,
            spend_increase=1000.0,
            spend_increase_pct=float('inf'),
            marginal_contribution=500.0,
            marginal_roas=0.5,
        )

        assert result.current_spend == 0.0
        assert result.spend_increase == 1000.0


class TestScenarioResultMethods:
    """Additional tests for ScenarioResult."""

    def test_result_with_multiple_channels(self):
        """Test scenario with multiple channel changes."""
        from mmm_framework.analysis import ScenarioResult

        spend_changes = {
            "TV": {"original": 500, "scenario": 600, "change": 100},
            "Digital": {"original": 300, "scenario": 250, "change": -50},
            "Social": {"original": 200, "scenario": 200, "change": 0},
        }

        result = ScenarioResult(
            baseline_outcome=1000.0,
            scenario_outcome=1050.0,
            outcome_change=50.0,
            outcome_change_pct=5.0,
            spend_changes=spend_changes,
            baseline_prediction=np.zeros(50),
            scenario_prediction=np.ones(50) * 1.05,
        )

        assert len(result.spend_changes) == 3
        assert result.spend_changes["TV"]["change"] == 100
        assert result.spend_changes["Digital"]["change"] == -50

    def test_result_with_negative_outcome_change(self):
        """Test scenario with negative outcome change."""
        from mmm_framework.analysis import ScenarioResult

        result = ScenarioResult(
            baseline_outcome=1000.0,
            scenario_outcome=900.0,
            outcome_change=-100.0,
            outcome_change_pct=-10.0,
            spend_changes={"TV": {"original": 500, "scenario": 200, "change": -300}},
            baseline_prediction=np.ones(50),
            scenario_prediction=np.ones(50) * 0.9,
        )

        assert result.outcome_change < 0
        assert result.outcome_change_pct < 0


class TestMMMAnalyzerMethods:
    """Additional tests for MMMAnalyzer methods."""

    def test_compute_counterfactual_contributions_delegation(self):
        """Test that compute_counterfactual_contributions delegates to model."""
        from mmm_framework.analysis import MMMAnalyzer

        expected_result = "mock_contribution_result"

        class MockModel:
            _trace = "mock_trace"

            def compute_counterfactual_contributions(self, **kwargs):
                return expected_result

        analyzer = MMMAnalyzer(MockModel())
        result = analyzer.compute_counterfactual_contributions()

        assert result == expected_result

    def test_compute_marginal_contributions_delegation(self):
        """Test that compute_marginal_contributions delegates to model."""
        from mmm_framework.analysis import MMMAnalyzer

        expected_df = pd.DataFrame({"channel": ["TV"], "marginal": [100]})

        class MockModel:
            _trace = "mock_trace"

            def compute_marginal_contributions(self, **kwargs):
                return expected_df

        analyzer = MMMAnalyzer(MockModel())
        result = analyzer.compute_marginal_contributions(spend_increase_pct=10.0)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_compute_saturation_curves_validates_channel(self):
        """Test that compute_saturation_curves validates channel name."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Digital"]

        analyzer = MMMAnalyzer(MockModel())

        with pytest.raises(ValueError, match="Unknown channel"):
            analyzer.compute_saturation_curves("InvalidChannel")


class TestPeriodContributions:
    """Tests for compute_period_contributions function."""

    def test_with_custom_period_names(self):
        """Test period contributions with custom period names."""
        from mmm_framework.analysis import compute_period_contributions

        # Create mock contribution results
        class MockContributionResults:
            channel_contributions = pd.DataFrame(
                {"TV": [10, 20, 30, 40], "Digital": [5, 10, 15, 20]},
                index=pd.MultiIndex.from_tuples([(0, "2024-01"), (1, "2024-02"), (2, "2024-03"), (3, "2024-04")]),
            )

        results = MockContributionResults()
        periods = [(0, 1), (2, 3)]
        period_names = ["Q1", "Q2"]

        df = compute_period_contributions(results, periods, period_names)

        assert "Q1" in df.columns
        assert "Q2" in df.columns

    def test_with_default_period_names(self):
        """Test period contributions with default period names."""
        from mmm_framework.analysis import compute_period_contributions

        class MockContributionResults:
            channel_contributions = pd.DataFrame(
                {"TV": [10, 20, 30, 40]},
                index=pd.MultiIndex.from_tuples([(0, "2024-01"), (1, "2024-02"), (2, "2024-03"), (3, "2024-04")]),
            )

        results = MockContributionResults()
        periods = [(0, 1), (2, 3)]

        df = compute_period_contributions(results, periods)

        assert "Period 1" in df.columns
        assert "Period 2" in df.columns


class TestAnalyzerWithComplexMockModel:
    """Tests with more complex mock model setups."""

    def test_analyzer_with_multiple_channels(self):
        """Test analyzer with model having multiple channels."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Radio", "Digital", "Social", "Print"]
            n_obs = 52

        analyzer = MMMAnalyzer(MockModel())

        assert len(analyzer.channel_names) == 5
        assert "Print" in analyzer.channel_names
        assert analyzer.n_obs == 52

    def test_analyzer_time_mask_with_none_period(self):
        """Test time mask with None period returns all True."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockModel:
            _trace = "mock_trace"
            n_obs = 10

            def _get_time_mask(self, time_period):
                if time_period is None:
                    return np.ones(self.n_obs, dtype=bool)
                return np.zeros(self.n_obs, dtype=bool)

        analyzer = MMMAnalyzer(MockModel())
        mask = analyzer.get_time_mask(None)

        assert mask.all()
        assert len(mask) == 10

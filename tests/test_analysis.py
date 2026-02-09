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
                return pd.DataFrame(
                    {
                        "Channel": ["TV", "Radio"],
                        "Contribution": [100, 50],
                    }
                )

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
            spend_increase_pct=float("inf"),
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
                index=pd.MultiIndex.from_tuples(
                    [(0, "2024-01"), (1, "2024-02"), (2, "2024-03"), (3, "2024-04")]
                ),
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
                index=pd.MultiIndex.from_tuples(
                    [(0, "2024-01"), (1, "2024-02"), (2, "2024-03"), (3, "2024-04")]
                ),
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


class TestMMMAnalyzerWhatIfScenario:
    """Tests for MMMAnalyzer.what_if_scenario method."""

    def test_what_if_scenario_delegation(self):
        """Test that what_if_scenario delegates to model."""
        from mmm_framework.analysis import MMMAnalyzer

        expected_result = {
            "baseline_outcome": 1000.0,
            "scenario_outcome": 1100.0,
            "outcome_change": 100.0,
        }

        class MockModel:
            _trace = "mock_trace"

            def what_if_scenario(
                self, spend_changes, time_period=None, random_seed=None
            ):
                return expected_result

        analyzer = MMMAnalyzer(MockModel())
        result = analyzer.what_if_scenario(spend_changes={"TV": 1.2})

        assert result == expected_result


class TestMMMAnalyzerComputeChannelROI:
    """Tests for MMMAnalyzer.compute_channel_roi method."""

    def test_compute_channel_roi_basic(self):
        """Test basic compute_channel_roi."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockContributionResults:
            def __init__(self):
                self.total_contributions = pd.Series({"TV": 500.0, "Digital": 300.0})
                self.contribution_pct = pd.Series({"TV": 62.5, "Digital": 37.5})
                self.contribution_hdi_low = pd.Series({"TV": 400.0, "Digital": 200.0})
                self.contribution_hdi_high = pd.Series({"TV": 600.0, "Digital": 400.0})

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Digital"]
            X_media_raw = np.array(
                [
                    [100, 50],
                    [150, 60],
                    [120, 40],
                ]
            )

            def _get_time_mask(self, time_period):
                return np.ones(3, dtype=bool)

        analyzer = MMMAnalyzer(MockModel())

        # Mock the compute_counterfactual_contributions
        analyzer.compute_counterfactual_contributions = (
            lambda **kwargs: MockContributionResults()
        )

        result = analyzer.compute_channel_roi()

        assert isinstance(result, pd.DataFrame)
        assert "Channel" in result.columns
        assert "Total Spend" in result.columns
        assert "ROI" in result.columns
        assert len(result) == 2

    def test_compute_channel_roi_with_time_period(self):
        """Test compute_channel_roi with time period filter."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockContributionResults:
            def __init__(self):
                self.total_contributions = pd.Series({"TV": 200.0})
                self.contribution_pct = pd.Series({"TV": 100.0})
                self.contribution_hdi_low = None
                self.contribution_hdi_high = None

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV"]
            X_media_raw = np.array(
                [
                    [100],
                    [150],
                    [120],
                    [180],
                ]
            )

            def _get_time_mask(self, time_period):
                if time_period is None:
                    return np.ones(4, dtype=bool)
                start, end = time_period
                mask = np.zeros(4, dtype=bool)
                mask[start : end + 1] = True
                return mask

        analyzer = MMMAnalyzer(MockModel())
        analyzer.compute_counterfactual_contributions = (
            lambda **kwargs: MockContributionResults()
        )

        result = analyzer.compute_channel_roi(time_period=(0, 1))

        assert isinstance(result, pd.DataFrame)
        assert "ROI" in result.columns

    def test_compute_channel_roi_zero_spend(self):
        """Test compute_channel_roi handles zero spend."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockContributionResults:
            def __init__(self):
                self.total_contributions = pd.Series({"TV": 100.0})
                self.contribution_pct = pd.Series({"TV": 100.0})
                self.contribution_hdi_low = None
                self.contribution_hdi_high = None

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV"]
            X_media_raw = np.zeros((3, 1))  # All zeros

            def _get_time_mask(self, time_period):
                return np.ones(3, dtype=bool)

        analyzer = MMMAnalyzer(MockModel())
        analyzer.compute_counterfactual_contributions = (
            lambda **kwargs: MockContributionResults()
        )

        result = analyzer.compute_channel_roi()

        # ROI should be 0 when spend is 0
        assert result.loc[0, "ROI"] == 0


class TestMMMAnalyzerComputeSaturationCurves:
    """Tests for MMMAnalyzer.compute_saturation_curves method."""

    def test_compute_saturation_curves_returns_dataframe(self):
        """Test that compute_saturation_curves returns DataFrame."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockPredictionResults:
            def __init__(self, total):
                self.y_pred_mean = np.ones(52) * total

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Digital"]
            X_media_raw = np.random.rand(52, 2) * 100

            def predict(self, X_media=None, random_seed=None):
                if X_media is None:
                    return MockPredictionResults(1000)
                return MockPredictionResults(1100)

        analyzer = MMMAnalyzer(MockModel())

        result = analyzer.compute_saturation_curves("TV", n_points=5)

        assert isinstance(result, pd.DataFrame)
        assert "Spend Level" in result.columns
        assert "Total Outcome" in result.columns
        assert len(result) == 5

    def test_compute_saturation_curves_custom_range(self):
        """Test compute_saturation_curves with custom spend range."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockPredictionResults:
            def __init__(self):
                self.y_pred_mean = np.ones(10) * 500

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV"]
            X_media_raw = np.ones((10, 1)) * 50

            def predict(self, **kwargs):
                return MockPredictionResults()

        analyzer = MMMAnalyzer(MockModel())

        result = analyzer.compute_saturation_curves(
            "TV", spend_range=(0.0, 200.0), n_points=10
        )

        assert result["Spend Level"].min() == 0.0
        assert result["Spend Level"].max() == 200.0
        assert len(result) == 10

    def test_compute_saturation_curves_invalid_channel(self):
        """Test compute_saturation_curves raises on invalid channel."""
        from mmm_framework.analysis import MMMAnalyzer

        class MockModel:
            _trace = "mock_trace"
            channel_names = ["TV", "Digital"]

        analyzer = MMMAnalyzer(MockModel())

        with pytest.raises(ValueError, match="Unknown channel"):
            analyzer.compute_saturation_curves("Radio")


class TestContributionResultsSummaryEdgeCases:
    """Edge case tests for ContributionResults.summary."""

    def test_summary_with_hdi(self):
        """Test summary includes HDI when available."""
        from mmm_framework.model import ContributionResults

        results = ContributionResults(
            channel_contributions=pd.DataFrame(
                {
                    "TV": [100, 200, 300],
                    "Digital": [50, 100, 150],
                }
            ),
            total_contributions=pd.Series({"TV": 600, "Digital": 300}),
            contribution_pct=pd.Series({"TV": 66.7, "Digital": 33.3}),
            baseline_prediction=np.ones(3) * 1000,
            counterfactual_predictions={},
            contribution_hdi_low=pd.Series({"TV": 500, "Digital": 200}),
            contribution_hdi_high=pd.Series({"TV": 700, "Digital": 400}),
        )

        summary = results.summary()

        assert "HDI 3%" in summary.columns
        assert "HDI 97%" in summary.columns

    def test_summary_without_hdi(self):
        """Test summary without HDI."""
        from mmm_framework.model import ContributionResults

        results = ContributionResults(
            channel_contributions=pd.DataFrame(
                {
                    "TV": [100, 200],
                }
            ),
            total_contributions=pd.Series({"TV": 300}),
            contribution_pct=pd.Series({"TV": 100.0}),
            baseline_prediction=np.ones(2) * 500,
            counterfactual_predictions={},
        )

        summary = results.summary()

        assert "HDI 3%" not in summary.columns
        assert "HDI 97%" not in summary.columns


class TestComponentDecompositionSummaryEdgeCases:
    """Edge case tests for ComponentDecomposition.summary."""

    def test_summary_with_geo_effects(self):
        """Test summary includes geo effects when present."""
        from mmm_framework.model import ComponentDecomposition

        decomp = ComponentDecomposition(
            intercept=np.ones(10) * 100,
            trend=np.zeros(10),
            seasonality=np.zeros(10),
            media_total=np.ones(10) * 50,
            media_by_channel=pd.DataFrame({"TV": np.ones(10) * 50}),
            controls_total=np.zeros(10),
            controls_by_var=None,
            geo_effects=np.ones(10) * 20,
            product_effects=None,
            total_intercept=1000.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=500.0,
            total_controls=0.0,
            total_geo=200.0,
            total_product=None,
            y_mean=100.0,
            y_std=10.0,
        )

        summary = decomp.summary()

        assert "Geo Effects" in summary["Component"].values

    def test_summary_with_product_effects(self):
        """Test summary includes product effects when present."""
        from mmm_framework.model import ComponentDecomposition

        decomp = ComponentDecomposition(
            intercept=np.ones(10) * 100,
            trend=np.zeros(10),
            seasonality=np.zeros(10),
            media_total=np.ones(10) * 50,
            media_by_channel=pd.DataFrame({"TV": np.ones(10) * 50}),
            controls_total=np.zeros(10),
            controls_by_var=None,
            geo_effects=None,
            product_effects=np.ones(10) * 15,
            total_intercept=1000.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=500.0,
            total_controls=0.0,
            total_geo=None,
            total_product=150.0,
            y_mean=100.0,
            y_std=10.0,
        )

        summary = decomp.summary()

        assert "Product Effects" in summary["Component"].values

    def test_summary_zero_total(self):
        """Test summary handles zero total gracefully."""
        from mmm_framework.model import ComponentDecomposition

        decomp = ComponentDecomposition(
            intercept=np.zeros(5),
            trend=np.zeros(5),
            seasonality=np.zeros(5),
            media_total=np.zeros(5),
            media_by_channel=pd.DataFrame({"TV": np.zeros(5)}),
            controls_total=np.zeros(5),
            controls_by_var=None,
            geo_effects=None,
            product_effects=None,
            total_intercept=0.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=0.0,
            total_controls=0.0,
            total_geo=None,
            total_product=None,
            y_mean=0.0,
            y_std=1.0,
        )

        summary = decomp.summary()

        # All contribution percentages should be 0 when total is 0
        assert all(summary["Contribution %"] == 0)


class TestMediaSummaryEdgeCases:
    """Edge case tests for ComponentDecomposition.media_summary."""

    def test_media_summary_zero_media(self):
        """Test media_summary handles zero total media."""
        from mmm_framework.model import ComponentDecomposition

        decomp = ComponentDecomposition(
            intercept=np.ones(5) * 100,
            trend=np.zeros(5),
            seasonality=np.zeros(5),
            media_total=np.zeros(5),
            media_by_channel=pd.DataFrame(
                {
                    "TV": np.zeros(5),
                    "Digital": np.zeros(5),
                }
            ),
            controls_total=np.zeros(5),
            controls_by_var=None,
            geo_effects=None,
            product_effects=None,
            total_intercept=500.0,
            total_trend=0.0,
            total_seasonality=0.0,
            total_media=0.0,
            total_controls=0.0,
            total_geo=None,
            total_product=None,
            y_mean=100.0,
            y_std=10.0,
        )

        media_summary = decomp.media_summary()

        # Share of Media % should be 0 when total is 0
        assert all(media_summary["Share of Media %"] == 0)


class TestComputePeriodContributionsEdgeCases:
    """Edge case tests for compute_period_contributions."""

    def test_empty_period_range(self):
        """Test with period that has no matching data."""
        from mmm_framework.analysis import compute_period_contributions

        class MockContributionResults:
            channel_contributions = pd.DataFrame(
                {"TV": [10, 20, 30]},
                index=pd.MultiIndex.from_tuples(
                    [(0, "2024-01"), (1, "2024-02"), (2, "2024-03")]
                ),
            )

        results = MockContributionResults()
        # Period range outside available data
        periods = [(10, 15)]

        df = compute_period_contributions(results, periods)

        # Should return empty or zeros for that period
        assert isinstance(df, pd.DataFrame)


class TestScenarioResultEdgeCases:
    """Edge case tests for ScenarioResult."""

    def test_scenario_result_with_all_zero_changes(self):
        """Test ScenarioResult with no actual changes."""
        from mmm_framework.analysis import ScenarioResult

        result = ScenarioResult(
            baseline_outcome=1000.0,
            scenario_outcome=1000.0,
            outcome_change=0.0,
            outcome_change_pct=0.0,
            spend_changes={
                "TV": {"original": 500, "scenario": 500, "change": 0},
            },
            baseline_prediction=np.ones(10) * 100,
            scenario_prediction=np.ones(10) * 100,
        )

        assert result.outcome_change == 0.0
        assert result.outcome_change_pct == 0.0


class TestMarginalAnalysisResultEdgeCases:
    """Edge case tests for MarginalAnalysisResult."""

    def test_marginal_result_infinite_pct(self):
        """Test MarginalAnalysisResult with infinite percentage increase."""
        from mmm_framework.analysis import MarginalAnalysisResult

        result = MarginalAnalysisResult(
            channel="New Channel",
            current_spend=0.0,
            spend_increase=500.0,
            spend_increase_pct=float("inf"),
            marginal_contribution=250.0,
            marginal_roas=0.5,
        )

        assert result.spend_increase_pct == float("inf")
        assert result.marginal_roas == 0.5

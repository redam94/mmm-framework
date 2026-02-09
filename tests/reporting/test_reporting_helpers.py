"""
Test and Example Usage for MMM Reporting Helpers.

This file demonstrates how to use the reporting helper functions
with both BayesianMMM and extended model classes.

Run with:
    python test_reporting_helpers.py
"""

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import pytest

# Import the helpers module
from mmm_framework.reporting.helpers import (
    # Result containers
    ROIResult,
    PriorPosteriorComparison,
    SaturationCurveResult,
    AdstockResult,
    DecompositionResult,
    MediatedEffectResult,
    # ROI functions
    compute_roi_with_uncertainty,
    compute_marginal_roi,
    # Prior/posterior comparison
    get_prior_posterior_comparison,
    compute_shrinkage_summary,
    # Saturation
    compute_saturation_curves_with_uncertainty,
    # Adstock
    compute_adstock_weights,
    # Decomposition
    compute_component_decomposition,
    compute_decomposition_waterfall,
    # Extended models
    compute_mediated_effects,
    compute_cross_effects,
    # Summary
    generate_model_summary,
    # Utilities
    _compute_hdi,
    _flatten_samples,
)

# =============================================================================
# Fixtures / Mock Data
# =============================================================================


def create_mock_posterior():
    """Create a mock posterior object for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Simulate posterior samples for various parameters
    posterior_data = {
        "beta_TV": np.random.normal(0.5, 0.1, n_samples),
        "beta_Digital": np.random.normal(0.3, 0.08, n_samples),
        "beta_Social": np.random.normal(0.2, 0.12, n_samples),
        "adstock_TV": np.random.beta(3, 2, n_samples) * 0.5 + 0.3,  # ~0.3-0.8
        "adstock_Digital": np.random.beta(2, 3, n_samples) * 0.4 + 0.2,  # ~0.2-0.6
        "adstock_Social": np.random.beta(2, 2, n_samples) * 0.5 + 0.25,  # ~0.25-0.75
        "sat_lam_TV": np.random.gamma(2, 0.5, n_samples),
        "sat_lam_Digital": np.random.gamma(3, 0.4, n_samples),
        "sat_lam_Social": np.random.gamma(2.5, 0.45, n_samples),
        "intercept": np.random.normal(10, 0.5, n_samples),
        "sigma": np.abs(np.random.normal(0.3, 0.05, n_samples)),
    }

    # Add channel contributions
    n_obs = 52
    for ch in ["TV", "Digital", "Social"]:
        contrib = np.random.normal(100, 20, (n_samples, n_obs))
        posterior_data[f"contribution_{ch}"] = contrib

    # Create mock posterior object
    class MockArray:
        def __init__(self, values):
            self._values = np.asarray(values)

        @property
        def values(self):
            return self._values

        def mean(self):
            return MockScalar(np.mean(self._values))

        def std(self):
            return MockScalar(np.std(self._values))

    class MockScalar:
        def __init__(self, val):
            self._val = val

        @property
        def values(self):
            return self._val

        def __float__(self):
            return float(self._val)

    class MockPosterior:
        def __init__(self, data):
            self._data = data
            self.data_vars = list(data.keys())

        def __contains__(self, key):
            return key in self._data

        def __getitem__(self, key):
            return MockArray(self._data[key])

        def add_variable(self, name, values):
            """Helper to add new variables after creation."""
            self._data[name] = values
            if name not in self.data_vars:
                self.data_vars.append(name)

    return MockPosterior(posterior_data)


def create_mock_model(with_trace=True):
    """Create a mock BayesianMMM model for testing."""
    model = MagicMock()
    model.channel_names = ["TV", "Digital", "Social"]
    model.n_channels = 3
    model.n_obs = 52
    model.y_mean = 1000.0
    model.y_std = 100.0
    model.has_geo = False
    model.has_product = False

    # Create mock panel
    model.panel = MagicMock()
    model.panel.channel_names = ["TV", "Digital", "Social"]
    model.panel.X_media = np.random.uniform(1000, 50000, (52, 3))

    if with_trace:
        # Create mock trace
        model._trace = MagicMock()
        model._trace.posterior = create_mock_posterior()

        # Mock sample_stats for divergences
        model._trace.sample_stats = {"diverging": MagicMock(values=np.zeros((4, 500)))}
    else:
        model._trace = None

    return model


def create_mock_nested_model():
    """Create a mock NestedMMM model for testing."""
    model = create_mock_model()
    model.mediator_names = ["awareness"]
    model.outcome_names = ["sales"]

    # Add mediation-specific posterior variables
    posterior = model._trace.posterior
    n_samples = 1000
    np.random.seed(43)  # Different seed for different values

    # Add direct/indirect/total for each channel-outcome combination
    for ch in ["TV", "Digital", "Social"]:
        direct = np.random.normal(0.4 if ch == "TV" else 0.25, 0.08, n_samples)
        indirect = np.random.normal(0.1 if ch == "TV" else 0.05, 0.05, n_samples)
        total = direct + indirect

        posterior.add_variable(f"direct_{ch}_sales", direct)
        posterior.add_variable(f"indirect_{ch}_sales", indirect)
        posterior.add_variable(f"total_{ch}_sales", total)

    return model


def create_mock_multivariate_model():
    """Create a mock MultivariateMMM model for testing."""
    model = create_mock_model()
    model.outcome_names = ["product_a", "product_b"]

    # Add cross-effect matrix
    posterior = model._trace.posterior
    n_samples = 1000
    n_outcomes = 2
    np.random.seed(44)  # Different seed

    # Psi matrix: cross-effects between outcomes
    # Shape: (samples, n_outcomes, n_outcomes)
    psi_samples = np.zeros((n_samples, n_outcomes, n_outcomes))
    psi_samples[:, 0, 1] = np.random.normal(
        -0.1, 0.05, n_samples
    )  # A -> B (cannibalization)
    psi_samples[:, 1, 0] = np.random.normal(0.05, 0.03, n_samples)  # B -> A (halo)

    posterior.add_variable("psi", psi_samples)

    return model


# =============================================================================
# Unit Tests
# =============================================================================


class TestUtilities:
    """Test utility functions."""

    def test_compute_hdi_basic(self):
        """Test HDI computation."""
        samples = np.random.normal(0, 1, 10000)
        lower, upper = _compute_hdi(samples, 0.94)

        # 94% HDI for standard normal should be approximately (-1.88, 1.88)
        assert -2.1 < lower < -1.6
        assert 1.6 < upper < 2.1

    def test_compute_hdi_empty(self):
        """Test HDI with empty array."""
        lower, upper = _compute_hdi(np.array([]))
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_flatten_samples_1d(self):
        """Test flattening 1D array."""
        arr = np.array([1, 2, 3])
        result = _flatten_samples(arr)
        assert result.shape == (3,)

    def test_flatten_samples_2d(self):
        """Test flattening 2D array (chain x draw)."""
        arr = np.random.randn(4, 500)
        result = _flatten_samples(arr)
        assert result.shape == (2000,)

    def test_flatten_samples_3d(self):
        """Test flattening 3D array (chain x draw x dim)."""
        arr = np.random.randn(4, 500, 10)
        result = _flatten_samples(arr)
        assert result.shape == (2000, 10)


class TestROIResult:
    """Test ROIResult dataclass."""

    def test_to_dict(self):
        """Test ROIResult.to_dict()."""
        result = ROIResult(
            channel="TV",
            spend=100000,
            contribution_mean=150000,
            contribution_lower=120000,
            contribution_upper=180000,
            roi_mean=1.5,
            roi_lower=1.2,
            roi_upper=1.8,
            prob_positive=0.99,
            prob_profitable=0.85,
        )

        d = result.to_dict()
        assert d["channel"] == "TV"
        assert d["roi_mean"] == 1.5
        assert d["prob_profitable"] == 0.85


class TestComputeROI:
    """Test ROI computation functions."""

    def test_compute_roi_requires_fitted_model(self):
        """Test that unfitted model raises error."""
        model = create_mock_model(with_trace=False)

        with pytest.raises(ValueError, match="not fitted"):
            compute_roi_with_uncertainty(model)

    def test_compute_roi_basic(self):
        """Test basic ROI computation."""
        model = create_mock_model()

        spend_data = {"TV": 500000, "Digital": 300000, "Social": 200000}
        roi_df = compute_roi_with_uncertainty(model, spend_data=spend_data)

        assert isinstance(roi_df, pd.DataFrame)
        assert len(roi_df) == 3
        assert "roi_mean" in roi_df.columns
        assert "prob_profitable" in roi_df.columns

    def test_compute_roi_extracts_spend(self):
        """Test ROI extraction from panel data."""
        model = create_mock_model()

        roi_df = compute_roi_with_uncertainty(model)

        # Should extract spend from panel
        assert len(roi_df) == 3
        assert all(roi_df["spend"] > 0)


class TestPriorPosteriorComparison:
    """Test prior vs posterior comparison functions."""

    def test_get_prior_posterior_comparison(self):
        """Test basic prior-posterior comparison."""
        model = create_mock_model()

        # This will work for posterior but may not have prior samples
        comparisons = get_prior_posterior_comparison(
            model,
            parameters=["beta_TV", "intercept"],
            n_prior_samples=100,
        )

        assert len(comparisons) > 0
        assert all(isinstance(c, PriorPosteriorComparison) for c in comparisons)

    def test_compute_shrinkage_summary(self):
        """Test shrinkage summary computation."""
        comparisons = [
            PriorPosteriorComparison(
                parameter="beta_TV",
                prior_mean=0.0,
                prior_sd=1.0,
                posterior_mean=0.5,
                posterior_sd=0.1,
                posterior_hdi_low=0.3,
                posterior_hdi_high=0.7,
                shrinkage=0.9,
                prior_samples=None,
                posterior_samples=np.random.randn(100),
            ),
        ]

        summary = compute_shrinkage_summary(comparisons)

        assert isinstance(summary, pd.DataFrame)
        assert "shrinkage" in summary.columns
        assert summary.iloc[0]["shrinkage"] == 0.9


class TestSaturationCurves:
    """Test saturation curve computation."""

    def test_compute_saturation_curves(self):
        """Test saturation curve computation."""
        model = create_mock_model()

        curves = compute_saturation_curves_with_uncertainty(
            model,
            n_points=50,
            n_samples=100,
        )

        assert isinstance(curves, dict)
        # Should have curves for channels with saturation params
        for ch, curve in curves.items():
            assert isinstance(curve, SaturationCurveResult)
            assert len(curve.spend_grid) == 50
            assert 0 <= curve.saturation_level <= 1


class TestAdstockWeights:
    """Test adstock weight computation."""

    def test_compute_adstock_weights(self):
        """Test adstock weight computation."""
        model = create_mock_model()

        adstock = compute_adstock_weights(model)

        assert isinstance(adstock, dict)
        for ch, result in adstock.items():
            assert isinstance(result, AdstockResult)
            assert result.half_life > 0
            assert 0 <= result.total_carryover <= 1


class TestDecomposition:
    """Test component decomposition functions."""

    def test_compute_decomposition(self):
        """Test component decomposition."""
        model = create_mock_model()

        decomp = compute_component_decomposition(model, include_time_series=False)

        assert isinstance(decomp, list)
        assert all(isinstance(d, DecompositionResult) for d in decomp)

    def test_decomposition_waterfall(self):
        """Test waterfall chart formatting."""
        decomp = [
            DecompositionResult(
                component="Baseline",
                total_contribution=1000,
                contribution_lower=900,
                contribution_upper=1100,
                pct_of_total=0.5,
            ),
            DecompositionResult(
                component="TV",
                total_contribution=500,
                contribution_lower=400,
                contribution_upper=600,
                pct_of_total=0.25,
            ),
        ]

        waterfall_df = compute_decomposition_waterfall(decomp)

        assert isinstance(waterfall_df, pd.DataFrame)
        assert "start" in waterfall_df.columns
        assert "end" in waterfall_df.columns


class TestExtendedModels:
    """Test functions for extended models."""

    def test_compute_mediated_effects(self):
        """Test mediation effect computation."""
        model = create_mock_nested_model()

        effects = compute_mediated_effects(model)

        assert isinstance(effects, list)
        assert len(effects) > 0
        for e in effects:
            assert isinstance(e, MediatedEffectResult)
            assert 0 <= e.proportion_mediated <= 1

    def test_compute_cross_effects(self):
        """Test cross-effect computation."""
        model = create_mock_multivariate_model()

        cross_df = compute_cross_effects(model)

        assert isinstance(cross_df, pd.DataFrame)
        if len(cross_df) > 0:
            assert "source" in cross_df.columns
            assert "target" in cross_df.columns


class TestModelSummary:
    """Test model summary generation."""

    def test_generate_model_summary(self):
        """Test comprehensive model summary."""
        model = create_mock_model()

        summary = generate_model_summary(model)

        assert isinstance(summary, dict)
        assert "model_info" in summary
        assert "diagnostics" in summary
        assert summary["model_info"]["n_channels"] == 3


# =============================================================================
# Integration Examples
# =============================================================================


def example_roi_analysis():
    """Example: ROI analysis with uncertainty."""
    print("\n" + "=" * 60)
    print("Example: ROI Analysis with Uncertainty")
    print("=" * 60)

    model = create_mock_model()
    spend_data = {"TV": 500000, "Digital": 300000, "Social": 200000}

    roi_df = compute_roi_with_uncertainty(model, spend_data=spend_data)

    print("\nROI by Channel:")
    print(
        roi_df[
            ["channel", "roi_mean", "roi_hdi_low", "roi_hdi_high", "prob_profitable"]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    for _, row in roi_df.iterrows():
        prob = row["prob_profitable"] * 100
        print(
            f"  {row['channel']}: ROI = {row['roi_mean']:.2f} "
            f"[{row['roi_hdi_low']:.2f}, {row['roi_hdi_high']:.2f}], "
            f"P(ROI > 1) = {prob:.0f}%"
        )


def example_prior_posterior():
    """Example: Prior vs posterior comparison."""
    print("\n" + "=" * 60)
    print("Example: Prior vs Posterior Comparison")
    print("=" * 60)

    model = create_mock_model()

    comparisons = get_prior_posterior_comparison(
        model,
        parameters=["beta_TV", "beta_Digital", "sigma"],
    )

    print("\nPosterior Summary:")
    for c in comparisons:
        shrink_str = f"{c.shrinkage:.0%}" if c.shrinkage else "N/A"
        print(
            f"  {c.parameter}: mean = {c.posterior_mean:.3f}, "
            f"sd = {c.posterior_sd:.3f}, shrinkage = {shrink_str}"
        )


def example_saturation_analysis():
    """Example: Saturation curve analysis."""
    print("\n" + "=" * 60)
    print("Example: Saturation Analysis")
    print("=" * 60)

    model = create_mock_model()

    curves = compute_saturation_curves_with_uncertainty(model, n_points=50)

    print("\nSaturation Levels:")
    for ch, curve in curves.items():
        print(
            f"  {ch}: {curve.saturation_level:.0%} saturated, "
            f"marginal response = {curve.marginal_response_at_current:.4f}"
        )


def example_decomposition():
    """Example: Revenue decomposition."""
    print("\n" + "=" * 60)
    print("Example: Revenue Decomposition")
    print("=" * 60)

    model = create_mock_model()

    decomp = compute_component_decomposition(model, include_time_series=False)

    print("\nComponent Contributions:")
    for d in decomp:
        print(f"  {d.component}: ${d.total_contribution:,.0f} ({d.pct_of_total:.1%})")


def example_mediation_analysis():
    """Example: Mediation analysis for nested models."""
    print("\n" + "=" * 60)
    print("Example: Mediation Analysis")
    print("=" * 60)

    model = create_mock_nested_model()

    effects = compute_mediated_effects(model)

    print("\nDirect vs Indirect Effects:")
    for e in effects:
        print(f"  {e.channel} → {e.outcome}:")
        print(f"    Direct: {e.direct_mean:.3f}")
        print(f"    Indirect: {e.indirect_mean:.3f}")
        print(f"    % Mediated: {e.proportion_mediated:.0%}")


def example_full_summary():
    """Example: Full model summary."""
    print("\n" + "=" * 60)
    print("Example: Full Model Summary")
    print("=" * 60)

    model = create_mock_model()

    summary = generate_model_summary(model)

    print("\nModel Info:")
    for k, v in summary["model_info"].items():
        print(f"  {k}: {v}")

    print("\nDiagnostics:")
    for k, v in summary.get("diagnostics", {}).items():
        print(f"  {k}: {v}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("MMM Reporting Helpers - Test & Examples")
    print("=" * 60)

    # Run examples
    example_roi_analysis()
    example_prior_posterior()
    example_saturation_analysis()
    example_decomposition()
    example_mediation_analysis()
    example_full_summary()

    print("\n" + "=" * 60)
    print("Running unit tests...")
    print("=" * 60)

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])

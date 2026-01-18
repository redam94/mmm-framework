"""
Test suite for results module in mmm_framework.mmm_extensions.

Tests cover:
- MediationEffects dataclass
- CrossEffectSummary dataclass
- ModelResults dataclass
- EffectDecomposition dataclass

Note: Bypasses PyTensor compilation issues with special config.
"""

import pytensor
pytensor.config.exception_verbosity = 'high'
pytensor.config.cxx = ""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from mmm_framework.mmm_extensions.results import (
    MediationEffects,
    CrossEffectSummary,
    ModelResults,
    EffectDecomposition,
)


# =============================================================================
# MediationEffects Tests
# =============================================================================


class TestMediationEffects:
    """Tests for MediationEffects dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        effects = MediationEffects(
            channel="tv",
            direct_effect=0.5,
            direct_effect_sd=0.1,
            indirect_effects={"awareness": 0.3},
            total_indirect=0.3,
            total_effect=0.8,
            proportion_mediated=0.375,
        )

        assert effects.channel == "tv"
        assert effects.direct_effect == 0.5
        assert effects.direct_effect_sd == 0.1
        assert effects.indirect_effects == {"awareness": 0.3}
        assert effects.total_indirect == 0.3
        assert effects.total_effect == 0.8
        assert effects.proportion_mediated == pytest.approx(0.375)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        effects = MediationEffects(
            channel="digital",
            direct_effect=0.4,
            direct_effect_sd=0.08,
            indirect_effects={"awareness": 0.2, "consideration": 0.1},
            total_indirect=0.3,
            total_effect=0.7,
            proportion_mediated=0.43,
        )

        d = effects.to_dict()

        assert d["channel"] == "digital"
        assert d["direct_effect"] == 0.4
        assert d["direct_effect_sd"] == 0.08
        assert d["indirect_via_awareness"] == 0.2
        assert d["indirect_via_consideration"] == 0.1
        assert d["total_indirect"] == 0.3
        assert d["total_effect"] == 0.7
        assert d["proportion_mediated"] == 0.43

    def test_multiple_mediators(self):
        """Test with multiple indirect effects."""
        effects = MediationEffects(
            channel="social",
            direct_effect=0.2,
            direct_effect_sd=0.05,
            indirect_effects={
                "awareness": 0.15,
                "engagement": 0.10,
                "consideration": 0.05,
            },
            total_indirect=0.30,
            total_effect=0.50,
            proportion_mediated=0.60,
        )

        assert len(effects.indirect_effects) == 3
        assert effects.total_indirect == 0.30

        d = effects.to_dict()
        assert "indirect_via_awareness" in d
        assert "indirect_via_engagement" in d
        assert "indirect_via_consideration" in d

    def test_no_indirect_effects(self):
        """Test when there are no indirect effects."""
        effects = MediationEffects(
            channel="radio",
            direct_effect=0.5,
            direct_effect_sd=0.1,
            indirect_effects={},
            total_indirect=0.0,
            total_effect=0.5,
            proportion_mediated=0.0,
        )

        d = effects.to_dict()
        assert d["total_indirect"] == 0.0
        assert d["proportion_mediated"] == 0.0

    def test_nan_proportion_mediated(self):
        """Test when total effect is zero (NaN proportion)."""
        effects = MediationEffects(
            channel="print",
            direct_effect=0.0,
            direct_effect_sd=0.01,
            indirect_effects={},
            total_indirect=0.0,
            total_effect=0.0,
            proportion_mediated=float("nan"),
        )

        assert np.isnan(effects.proportion_mediated)

    def test_high_proportion_mediated(self):
        """Test with high proportion mediated."""
        effects = MediationEffects(
            channel="brand_tv",
            direct_effect=0.1,
            direct_effect_sd=0.02,
            indirect_effects={"awareness": 0.8, "consideration": 0.1},
            total_indirect=0.9,
            total_effect=1.0,
            proportion_mediated=0.9,
        )

        assert effects.proportion_mediated == 0.9
        assert effects.direct_effect < effects.total_indirect


# =============================================================================
# CrossEffectSummary Tests
# =============================================================================


class TestCrossEffectSummary:
    """Tests for CrossEffectSummary dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        summary = CrossEffectSummary(
            source="product_b",
            target="product_a",
            effect_type="cannibalization",
            mean=-0.15,
            sd=0.05,
            hdi_low=-0.24,
            hdi_high=-0.06,
        )

        assert summary.source == "product_b"
        assert summary.target == "product_a"
        assert summary.effect_type == "cannibalization"
        assert summary.mean == -0.15
        assert summary.sd == 0.05
        assert summary.hdi_low == -0.24
        assert summary.hdi_high == -0.06

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = CrossEffectSummary(
            source="product_b",
            target="product_a",
            effect_type="cannibalization",
            mean=-0.15,
            sd=0.05,
            hdi_low=-0.24,
            hdi_high=-0.06,
        )

        d = summary.to_dict()

        assert d["source"] == "product_b"
        assert d["target"] == "product_a"
        assert d["effect_type"] == "cannibalization"
        assert d["mean"] == -0.15
        assert d["sd"] == 0.05
        assert d["hdi_3%"] == -0.24
        assert d["hdi_97%"] == -0.06

    def test_halo_effect(self):
        """Test positive cross-effect (halo)."""
        summary = CrossEffectSummary(
            source="premium",
            target="budget",
            effect_type="halo",
            mean=0.10,
            sd=0.03,
            hdi_low=0.04,
            hdi_high=0.16,
        )

        assert summary.mean > 0  # Positive halo effect
        assert summary.hdi_low > 0  # Credible interval excludes zero

    def test_cannibalization_effect(self):
        """Test negative cross-effect (cannibalization)."""
        summary = CrossEffectSummary(
            source="multipack",
            target="single",
            effect_type="cannibalization",
            mean=-0.20,
            sd=0.04,
            hdi_low=-0.28,
            hdi_high=-0.12,
        )

        assert summary.mean < 0  # Negative cannibalization
        assert summary.hdi_high < 0  # Credible interval excludes zero

    def test_symmetric_effect(self):
        """Test symmetric cross-effect."""
        summary = CrossEffectSummary(
            source="a",
            target="b",
            effect_type="symmetric",
            mean=0.05,
            sd=0.08,
            hdi_low=-0.10,
            hdi_high=0.20,
        )

        assert summary.effect_type == "symmetric"
        # Interval includes zero - uncertain direction
        assert summary.hdi_low < 0 < summary.hdi_high


# =============================================================================
# ModelResults Tests
# =============================================================================


class TestModelResults:
    """Tests for ModelResults dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        mock_trace = MagicMock()
        mock_model = MagicMock()
        mock_config = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=mock_model,
            config=mock_config,
        )

        assert results.trace == mock_trace
        assert results.model == mock_model
        assert results.config == mock_config
        assert results.diagnostics == {}

    def test_with_diagnostics(self):
        """Test with diagnostics dictionary."""
        mock_trace = MagicMock()
        diagnostics = {
            "divergences": 0,
            "max_rhat": 1.01,
            "min_ess": 500,
        }

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
            diagnostics=diagnostics,
        )

        assert results.diagnostics["divergences"] == 0
        assert results.diagnostics["max_rhat"] == 1.01
        assert results.diagnostics["min_ess"] == 500

    def test_summary_calls_arviz(self):
        """Test that summary delegates to ArviZ."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.summary") as mock_summary:
            mock_summary.return_value = pd.DataFrame({"mean": [0.5], "sd": [0.1]})
            summary = results.summary(var_names=["alpha", "beta"])
            mock_summary.assert_called_once_with(mock_trace, var_names=["alpha", "beta"])

    def test_summary_without_var_names(self):
        """Test summary without specifying var_names."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.summary") as mock_summary:
            mock_summary.return_value = pd.DataFrame()
            _ = results.summary()
            mock_summary.assert_called_once_with(mock_trace, var_names=None)

    def test_plot_trace_calls_arviz(self):
        """Test that plot_trace delegates to ArviZ."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.plot_trace") as mock_plot:
            _ = results.plot_trace(var_names=["alpha"])
            mock_plot.assert_called_once_with(mock_trace, var_names=["alpha"])

    def test_plot_trace_with_kwargs(self):
        """Test plot_trace with additional kwargs."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.plot_trace") as mock_plot:
            _ = results.plot_trace(var_names=["alpha"], figsize=(12, 8))
            mock_plot.assert_called_once_with(
                mock_trace, var_names=["alpha"], figsize=(12, 8)
            )

    def test_plot_posterior_calls_arviz(self):
        """Test that plot_posterior delegates to ArviZ."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.plot_posterior") as mock_plot:
            _ = results.plot_posterior(var_names=["beta"])
            mock_plot.assert_called_once_with(mock_trace, var_names=["beta"])

    def test_plot_posterior_with_kwargs(self):
        """Test plot_posterior with additional kwargs."""
        mock_trace = MagicMock()

        results = ModelResults(
            trace=mock_trace,
            model=MagicMock(),
            config=MagicMock(),
        )

        with patch("arviz.plot_posterior") as mock_plot:
            _ = results.plot_posterior(var_names=["beta"], ref_val=0)
            mock_plot.assert_called_once_with(
                mock_trace, var_names=["beta"], ref_val=0
            )


# =============================================================================
# EffectDecomposition Tests
# =============================================================================


class TestEffectDecomposition:
    """Tests for EffectDecomposition dataclass."""

    def test_basic_creation(self):
        """Test basic initialization."""
        decomp = EffectDecomposition(
            channel="tv",
            outcome="sales",
            direct_mean=0.4,
            direct_sd=0.08,
            indirect_mean=0.3,
            indirect_sd=0.06,
            total_mean=0.7,
            total_sd=0.10,
        )

        assert decomp.channel == "tv"
        assert decomp.outcome == "sales"
        assert decomp.direct_mean == 0.4
        assert decomp.direct_sd == 0.08
        assert decomp.indirect_mean == 0.3
        assert decomp.indirect_sd == 0.06
        assert decomp.total_mean == 0.7
        assert decomp.total_sd == 0.10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        decomp = EffectDecomposition(
            channel="digital",
            outcome="product_a_sales",
            direct_mean=0.25,
            direct_sd=0.05,
            indirect_mean=0.15,
            indirect_sd=0.03,
            total_mean=0.40,
            total_sd=0.06,
        )

        d = decomp.to_dict()

        assert d["channel"] == "digital"
        assert d["outcome"] == "product_a_sales"
        assert d["direct_mean"] == 0.25
        assert d["direct_sd"] == 0.05
        assert d["indirect_mean"] == 0.15
        assert d["indirect_sd"] == 0.03
        assert d["total_mean"] == 0.40
        assert d["total_sd"] == 0.06

    def test_zero_indirect_effect(self):
        """Test with zero indirect effect (no mediation)."""
        decomp = EffectDecomposition(
            channel="radio",
            outcome="sales",
            direct_mean=0.5,
            direct_sd=0.10,
            indirect_mean=0.0,
            indirect_sd=0.0,
            total_mean=0.5,
            total_sd=0.10,
        )

        assert decomp.indirect_mean == 0.0
        assert decomp.total_mean == decomp.direct_mean

    def test_zero_direct_effect(self):
        """Test with zero direct effect (full mediation)."""
        decomp = EffectDecomposition(
            channel="brand_tv",
            outcome="sales",
            direct_mean=0.0,
            direct_sd=0.02,
            indirect_mean=0.6,
            indirect_sd=0.08,
            total_mean=0.6,
            total_sd=0.08,
        )

        assert decomp.direct_mean == 0.0
        assert decomp.total_mean == decomp.indirect_mean

    def test_effect_sum_consistency(self):
        """Test that total approximately equals direct + indirect."""
        decomp = EffectDecomposition(
            channel="tv",
            outcome="sales",
            direct_mean=0.4,
            direct_sd=0.08,
            indirect_mean=0.3,
            indirect_sd=0.06,
            total_mean=0.7,
            total_sd=0.10,
        )

        # Total should equal direct + indirect
        assert decomp.total_mean == pytest.approx(
            decomp.direct_mean + decomp.indirect_mean
        )

    def test_multiple_channel_decompositions(self):
        """Test creating decompositions for multiple channels."""
        decompositions = []

        for channel in ["tv", "digital", "social"]:
            decomp = EffectDecomposition(
                channel=channel,
                outcome="sales",
                direct_mean=0.3,
                direct_sd=0.05,
                indirect_mean=0.2,
                indirect_sd=0.04,
                total_mean=0.5,
                total_sd=0.07,
            )
            decompositions.append(decomp)

        assert len(decompositions) == 3
        assert all(d.outcome == "sales" for d in decompositions)

    def test_multiple_outcome_decompositions(self):
        """Test creating decompositions for multiple outcomes."""
        decompositions = []

        for outcome in ["product_a", "product_b", "product_c"]:
            decomp = EffectDecomposition(
                channel="tv",
                outcome=outcome,
                direct_mean=0.2,
                direct_sd=0.04,
                indirect_mean=0.1,
                indirect_sd=0.02,
                total_mean=0.3,
                total_sd=0.05,
            )
            decompositions.append(decomp)

        assert len(decompositions) == 3
        assert all(d.channel == "tv" for d in decompositions)


# =============================================================================
# Integration Tests
# =============================================================================


class TestResultsIntegration:
    """Integration tests for result containers."""

    def test_mediation_effects_to_dataframe(self):
        """Test converting multiple MediationEffects to DataFrame."""
        effects = [
            MediationEffects(
                channel="tv",
                direct_effect=0.5,
                direct_effect_sd=0.1,
                indirect_effects={"awareness": 0.3},
                total_indirect=0.3,
                total_effect=0.8,
                proportion_mediated=0.375,
            ),
            MediationEffects(
                channel="digital",
                direct_effect=0.4,
                direct_effect_sd=0.08,
                indirect_effects={"awareness": 0.2},
                total_indirect=0.2,
                total_effect=0.6,
                proportion_mediated=0.333,
            ),
        ]

        df = pd.DataFrame([e.to_dict() for e in effects])

        assert len(df) == 2
        assert "channel" in df.columns
        assert "direct_effect" in df.columns
        assert "proportion_mediated" in df.columns

    def test_cross_effect_summaries_to_dataframe(self):
        """Test converting multiple CrossEffectSummary to DataFrame."""
        summaries = [
            CrossEffectSummary(
                source="b",
                target="a",
                effect_type="cannibalization",
                mean=-0.15,
                sd=0.05,
                hdi_low=-0.24,
                hdi_high=-0.06,
            ),
            CrossEffectSummary(
                source="c",
                target="a",
                effect_type="halo",
                mean=0.10,
                sd=0.03,
                hdi_low=0.04,
                hdi_high=0.16,
            ),
        ]

        df = pd.DataFrame([s.to_dict() for s in summaries])

        assert len(df) == 2
        assert "source" in df.columns
        assert "target" in df.columns
        assert "hdi_3%" in df.columns

    def test_effect_decompositions_to_dataframe(self):
        """Test converting multiple EffectDecomposition to DataFrame."""
        decomps = [
            EffectDecomposition(
                channel="tv",
                outcome="product_a",
                direct_mean=0.4,
                direct_sd=0.08,
                indirect_mean=0.3,
                indirect_sd=0.06,
                total_mean=0.7,
                total_sd=0.10,
            ),
            EffectDecomposition(
                channel="tv",
                outcome="product_b",
                direct_mean=0.3,
                direct_sd=0.06,
                indirect_mean=0.2,
                indirect_sd=0.04,
                total_mean=0.5,
                total_sd=0.08,
            ),
        ]

        df = pd.DataFrame([d.to_dict() for d in decomps])

        assert len(df) == 2
        assert "channel" in df.columns
        assert "outcome" in df.columns
        assert "direct_mean" in df.columns
        assert "indirect_mean" in df.columns
        assert "total_mean" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

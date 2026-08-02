"""The extension seasonal period is 52.178571, not 52.0 (#275).

The core graph looks the frequency up in a table and gets exactly 52.0 weekly
observations per year. The extension graph divides 365.25 by the datetime
index's median spacing and gets 52.178571. The extension's docstring claimed
the two mirrored each other "so the component is comparable across models".

Measured here rather than asserted: max |Δ| of the yearly Fourier design over
104 weekly points is 0.04216 at order 1, **0.08174** at order 2 and 0.12376 at
order 3, on a basis whose amplitude is O(1) — small enough to look like noise
in a plot, large enough to move a decomposition.

Fixing it changes extension-model numbers, so it ships behind a flag whose
default reproduces today (R0.1 in technical-docs/deferred-causal-features.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions.components.temporal import (
    _component_periods,
    build_seasonality_contribution,
)
from mmm_framework.transforms.seasonality import (
    PERIODS_BY_FREQ,
    SeasonalityPeriodSource,
    create_fourier_features,
    frequency_from_median_days,
    periods_for_frequency,
)


class TestOneTable:
    def test_the_forecaster_reads_the_shared_table_not_a_copy(self):
        """It held a copy-pasted literal linked only by a comment."""
        from mmm_framework.validation import backtest

        assert backtest._PERIODS_BY_FREQ is PERIODS_BY_FREQ

    def test_the_core_graph_reads_the_shared_table(self):
        import inspect

        from mmm_framework.model.base import BayesianMMM

        src = inspect.getsource(BayesianMMM._prepare_seasonality)
        assert "periods_for_frequency" in src
        # The literal table must not survive anywhere in the builder.
        assert '"yearly": 52.0' not in src

    def test_the_table_itself(self):
        assert PERIODS_BY_FREQ["W"]["yearly"] == 52.0
        assert PERIODS_BY_FREQ["D"]["yearly"] == 365.25
        assert PERIODS_BY_FREQ["M"]["yearly"] == 12.0


class TestTheDivergence:
    """The number in the issue stays honest."""

    @pytest.mark.parametrize(
        "order,expected", [(1, 0.04216), (2, 0.08174), (3, 0.12376)]
    )
    def test_measured_divergence_over_104_weekly_points(self, order, expected):
        t = np.arange(104)
        table = create_fourier_features(t, PERIODS_BY_FREQ["W"]["yearly"], order)
        median = create_fourier_features(t, 365.25 / 7.0, order)
        assert np.abs(table - median).max() == pytest.approx(expected, abs=5e-5)

    def test_the_two_periods_are_not_equal(self):
        assert 365.25 / 7.0 == pytest.approx(52.178571428571431)
        assert PERIODS_BY_FREQ["W"]["yearly"] != 365.25 / 7.0


class TestComponentPeriods:
    def test_default_is_the_historical_median_rule(self):
        """Byte-identical: no existing extension fit moves."""
        got = _component_periods(7.0, None)
        want_yearly = 365.25 / 7.0
        assert got["yearly"] == want_yearly
        assert got["monthly"] == want_yearly / 12.0
        assert got["weekly"] == want_yearly / 52.0

    @pytest.mark.parametrize(
        "freq,days", [("W", 7.0), ("D", 1.0), ("M", 365.25 / 12.0)]
    )
    def test_frequency_table_matches_the_core_exactly(self, freq, days):
        """The gate that did not exist: the two sites agree when asked to."""
        got = _component_periods(days, SeasonalityPeriodSource.FREQUENCY_TABLE)
        assert got == periods_for_frequency(freq)
        assert got == PERIODS_BY_FREQ[freq]

    def test_a_real_month_resolves_to_the_monthly_row(self):
        # 28 is deliberately absent: a 28-day MEDIAN is a 4-weekly calendar, not
        # months (a monthly panel's median spacing is 30 or 31). See
        # TestToleranceIsTight.
        for days in (30.0, 31.0, 365.25 / 12.0):
            assert (
                _component_periods(days, SeasonalityPeriodSource.FREQUENCY_TABLE)
                == PERIODS_BY_FREQ["M"]
            )

    def test_an_untabulated_spacing_warns_and_falls_back(self):
        """Silence here would be the original defect in a new place."""
        with pytest.warns(UserWarning, match="matches no tabulated frequency"):
            got = _component_periods(3.0, SeasonalityPeriodSource.FREQUENCY_TABLE)
        assert got["yearly"] == 365.25 / 3.0

    def test_string_values_are_accepted(self):
        assert _component_periods(7.0, "frequency_table") == PERIODS_BY_FREQ["W"]


class TestFrequencyFromMedianDays:
    @pytest.mark.parametrize(
        "days,freq",
        [(7.0, "W"), (1.0, "D"), (30.0, "M"), (31.0, "M"), (6.9, "W"), (1.02, "D")],
    )
    def test_recognised_spacings(self, days, freq):
        assert frequency_from_median_days(days) == freq

    @pytest.mark.parametrize("days", [3.0, 0.0, -1.0, 100.0, 28.0, 35.0, 8.75])
    def test_unrecognised_spacings(self, days):
        assert frequency_from_median_days(days) is None


class TestItReachesTheGraph:
    @staticmethod
    def _design(period_source):
        """The seasonality design the builder actually uses, via the RV it makes."""
        import pymc as pm

        from mmm_framework.config import SeasonalityConfig

        idx = pd.date_range("2022-01-03", periods=104, freq="W-MON")
        cfg = SeasonalityConfig(yearly=2, period_source=period_source)
        with pm.Model():
            term = build_seasonality_contribution("", idx, 104, cfg)
            assert term is not None
            # term = design @ coefs; evaluating at the unit basis recovers design.
            coefs = pm.Model.get_context()["seasonality_coefs"]
            cols = [term.eval({coefs: np.eye(4)[j]}) for j in range(4)]
        return np.column_stack(cols)

    def test_default_graph_is_the_median_basis(self):
        got = self._design(None)
        want = create_fourier_features(np.arange(104), 365.25 / 7.0, 2)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_frequency_table_graph_is_the_core_basis(self):
        got = self._design(SeasonalityPeriodSource.FREQUENCY_TABLE)
        want = create_fourier_features(np.arange(104), 52.0, 2)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_the_two_graphs_differ_by_the_measured_amount(self):
        """Non-vacuity: the flag changes the graph, by exactly the reported Δ."""
        a = self._design(None)
        b = self._design(SeasonalityPeriodSource.FREQUENCY_TABLE)
        assert np.abs(a - b).max() == pytest.approx(0.08174, abs=5e-5)


class TestPartialTableRows:
    """The table's rows are deliberately partial; the consumer must cope.

    `PERIODS_BY_FREQ["W"]` tabulates no `weekly` period and `["M"]` only
    `yearly`, while the median rule always yields all three. Indexing with `[]`
    made the opt-in flag crash with a bare `KeyError` on configurations the
    default path — and the core model — accept and skip with a warning.
    """

    @pytest.mark.parametrize(
        "freq,missing", [("W", "weekly"), ("M", "monthly"), ("M", "weekly")]
    )
    def test_untabulated_component_warns_and_skips(self, freq, missing):
        import pymc as pm

        from mmm_framework.config import SeasonalityConfig

        days = {"W": 7.0, "M": 30.0}[freq]
        idx = pd.date_range("2022-01-03", periods=60, freq=f"{days:.0f}D")
        cfg = SeasonalityConfig(
            yearly=1,
            **{missing: 1},
            period_source=SeasonalityPeriodSource.FREQUENCY_TABLE,
        )
        with pm.Model():
            with pytest.warns(UserWarning, match="cannot be represented"):
                term = build_seasonality_contribution("", idx, 60, cfg)
        assert term is not None  # yearly survives

    def test_the_whole_graph_builds(self):
        """End to end: this raised KeyError('weekly') before the fix."""
        from mmm_framework.config import ModelConfig, SeasonalityConfig
        from mmm_framework.mmm_extensions.config import (
            MediatorConfig,
            MediatorType,
            NestedModelConfig,
        )
        from mmm_framework.mmm_extensions.models.nested import NestedMMM

        idx = pd.date_range("2022-01-03", periods=60, freq="W-MON")
        rng = np.random.default_rng(0)
        media = np.abs(rng.normal(100, 20, (60, 2)))
        y = 1000 + 4 * (40 + 0.3 * media[:, 0]) + 2 * media[:, 1]

        mc = ModelConfig()
        mc.seasonality = SeasonalityConfig(
            yearly=2, weekly=1, period_source=SeasonalityPeriodSource.FREQUENCY_TABLE
        )
        m = NestedMMM(
            media,
            y,
            ["TV", "Digital"],
            NestedModelConfig(
                mediators=(
                    MediatorConfig(name="A", mediator_type=MediatorType.FULLY_LATENT),
                )
            ),
            index=idx,
            model_config=mc,
        )
        with pytest.warns(UserWarning):
            assert m.model is not None


class TestToleranceIsTight:
    """A 4-weekly calendar is not monthly, and must not be silently called one."""

    def test_four_weekly_cadence_is_not_monthly(self):
        assert frequency_from_median_days(28.0) is None

    def test_the_mismatch_it_prevents_is_severe(self):
        """Anti-phase: 24x the divergence this module exists to remove."""
        t = np.arange(78)
        wrong = create_fourier_features(t, 12.0, 2)  # what "M" would have given
        right = create_fourier_features(t, 365.25 / 28.0, 2)
        assert np.abs(wrong - right).max() == pytest.approx(2.0, abs=1e-4)

    def test_real_month_ends_still_resolve(self):
        for days in (30.0, 31.0, 30.4375):
            assert frequency_from_median_days(days) == "M"

    def test_a_four_weekly_panel_warns_rather_than_inventing_a_period(self):
        with pytest.warns(UserWarning, match="matches no tabulated frequency"):
            got = _component_periods(28.0, SeasonalityPeriodSource.FREQUENCY_TABLE)
        assert got["yearly"] == pytest.approx(365.25 / 28.0)


class TestReachableFromASpec:
    """A remedy the product's own build path cannot apply is half a fix."""

    def test_spec_key_is_accepted_by_the_registry(self):
        from mmm_framework.agents.fitting import unconsumed_spec_path

        assert (
            unconsumed_spec_path(
                ["seasonality", "period_source"], "frequency_table", {}
            )
            is None
        )

    def test_spec_reaches_the_model_config(self):
        from mmm_framework.agents.fitting import _model_config_from_spec

        mc = _model_config_from_spec(
            {"seasonality": {"yearly": 2, "period_source": "frequency_table"}}
        )
        assert mc.seasonality.period_source is SeasonalityPeriodSource.FREQUENCY_TABLE

    def test_default_spec_leaves_it_unset(self):
        from mmm_framework.agents.fitting import _model_config_from_spec

        mc = _model_config_from_spec({"seasonality": {"yearly": 2}})
        assert mc.seasonality.period_source is None

    def test_a_typo_is_refused(self):
        from mmm_framework.agents.fitting import _model_config_from_spec

        with pytest.raises(ValueError, match="period_source"):
            _model_config_from_spec(
                {"seasonality": {"yearly": 2, "period_source": "freq_table"}}
            )


class TestCoreRefusesTheOtherSource:
    def test_datetime_median_is_refused_not_ignored(self):
        from mmm_framework.config import InferenceMethod, ModelConfig, SeasonalityConfig
        from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
        from mmm_framework.synth import dgp

        panel = dgp.build("clean", seed=3, n_weeks=60).panel()
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=4
        )
        cfg.seasonality = SeasonalityConfig(
            yearly=2, period_source=SeasonalityPeriodSource.DATETIME_MEDIAN
        )
        with pytest.raises(NotImplementedError, match="datetime_median"):
            BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))

    def test_frequency_table_is_accepted_and_changes_nothing(self):
        from mmm_framework.config import InferenceMethod, ModelConfig, SeasonalityConfig
        from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
        from mmm_framework.synth import dgp

        panel = dgp.build("clean", seed=3, n_weeks=60).panel()

        def build(source):
            cfg = ModelConfig(
                inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=4
            )
            cfg.seasonality = SeasonalityConfig(yearly=2, period_source=source)
            return BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))

        default = build(None)
        explicit = build(SeasonalityPeriodSource.FREQUENCY_TABLE)
        assert set(default.seasonality_features) == set(explicit.seasonality_features)
        for name, feats in default.seasonality_features.items():
            np.testing.assert_array_equal(feats, explicit.seasonality_features[name])

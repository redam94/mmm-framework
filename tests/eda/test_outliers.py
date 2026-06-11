"""Outlier detection + remediation graded against the synthetic DGP ground truth.

The scenarios in ``tests/synth/dgp.py`` record exactly where defects were
injected (``notes["spike_weeks"]``, ``notes["break_week"]``), so detector
recall/precision can be asserted, not eyeballed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth.dgp import (  # noqa: E402
    make_clean,
    make_heavy_tailed_noise,
    make_seasonality_misspec,
    make_spend_outliers,
    make_trend_break,
)

from mmm_framework.eda import (  # noqa: E402
    EDAPanel,
    OutlierConfig,
    apply_treatments,
    detect_outliers,
    recommend_treatments,
)
from mmm_framework.eda.remediation import STUDENT_T_ADVISORY  # noqa: E402

from .conftest import to_mff_long  # noqa: E402


def panel_from_scenario(sc) -> EDAPanel:
    """Build an EDAPanel directly from a Scenario's wide frames (no CSV)."""
    cols = {"Sales": sc.y.to_numpy()}
    cols.update({c: sc.spend[c].to_numpy() for c in sc.spend.columns})
    cols.update({c: sc.controls[c].to_numpy() for c in sc.controls.columns})
    wide = pd.DataFrame(cols, index=sc.weeks)
    long = to_mff_long(wide)
    return EDAPanel(
        df_wide=wide,
        df_long=long,
        kpi="Sales",
        media=list(sc.spend.columns),
        controls=list(sc.controls.columns),
        unassigned=[],
        dims=[],
        date_col="Period",
        freq="W-MON",
        roles_source="spec",
        source_path=None,
    )


class TestSpendOutliers:
    """The canonical failure mode: one ~15x data-entry spike per channel."""

    def test_recall_is_perfect_and_kind_is_isolated_spike(self):
        sc = make_spend_outliers()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)

        for channel, week in sc.notes["spike_weeks"].items():
            expected_period = str(sc.weeks[week].date())
            hits = [
                f
                for f in report.flags
                if f.variable == channel and f.period == expected_period
            ]
            assert hits, f"missed the injected spike in {channel} at {expected_period}"
            assert hits[0].kind == "isolated_spike"

    def test_precision_few_extra_flags_per_channel(self):
        sc = make_spend_outliers()
        report = detect_outliers(panel_from_scenario(sc))
        for channel in sc.spend.columns:
            extra = [
                f
                for f in report.flags
                if f.variable == channel
                and f.kind == "isolated_spike"
                and f.period != str(sc.weeks[sc.notes["spike_weeks"][channel]].date())
            ]
            assert len(extra) <= 2, f"{channel}: too many false spikes {extra}"

    def test_normalization_damage_reported(self):
        sc = make_spend_outliers()
        report = detect_outliers(panel_from_scenario(sc))
        for channel in sc.spend.columns:
            assert report.per_variable[channel]["normalization_damaged"] is True

    def test_winsorize_recommended_and_restores_normalization(self):
        sc = make_spend_outliers()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        actions = recommend_treatments(panel, report.flags, report.config)

        spike_flags = [f for f in report.flags if f.kind == "isolated_spike"]
        winsorize = [a for a in actions if a.strategy == "winsorize"]
        assert {a.flag_ids[0] for a in winsorize} >= {f.flag_id for f in spike_flags}

        treated = apply_treatments(panel.df_long, winsorize, report.flags, kpi="Sales")
        for channel in sc.spend.columns:
            vals = treated.loc[
                treated["VariableName"] == channel, "VariableValue"
            ].astype(float)
            p99 = np.percentile(vals, 99)
            assert vals.max() / p99 < 2.0, f"{channel} normalization still damaged"


class TestHeavyTailedNoise:
    def test_kpi_promo_spikes_flagged_as_heavy_tail(self):
        sc = make_heavy_tailed_noise()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)

        spike_periods = {str(sc.weeks[w].date()) for w in sc.notes["spike_weeks"]}
        kpi_flags = {f.period for f in report.flags if f.variable == "Sales"}
        found = spike_periods & kpi_flags
        assert len(found) >= 4, f"only found {len(found)}/5 promo spikes"

        assert report.per_variable["Sales"]["heavy_tailed"] is True
        assert any(f.kind == "heavy_tail_member" for f in report.flags)

    def test_dummy_recommended_with_student_t_advisory(self):
        sc = make_heavy_tailed_noise()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        actions = recommend_treatments(panel, report.flags, report.config)

        dummies = [a for a in actions if a.strategy == "dummy"]
        assert dummies, "expected dummy-control recommendations"
        assert all(a.spec_change and a.spec_change.get("add_control") for a in dummies)

        notes = [a for a in actions if a.action_id.startswith("note:heavy_tails")]
        assert notes and STUDENT_T_ADVISORY in notes[0].rationale


class TestTrendBreak:
    def test_break_detected_as_level_shift_not_winsorize(self):
        sc = make_trend_break()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)

        shifts = [
            f for f in report.flags if f.kind == "level_shift" and f.variable == "Sales"
        ]
        assert shifts, "missed the structural break"
        brk = sc.notes["break_week"]
        nearest = min(
            abs((pd.Timestamp(f.period) - sc.weeks[brk]).days) // 7 for f in shifts
        )
        assert nearest <= 6, f"level shift located {nearest} weeks from the break"

        actions = recommend_treatments(panel, report.flags, report.config)
        shift_actions = [a for a in actions if a.action_id.startswith("trend:")]
        assert shift_actions
        assert shift_actions[0].spec_change == {
            "setting_path": "trend.type",
            "value": "piecewise",
        }
        winsorized_ids = {
            fid for a in actions if a.strategy == "winsorize" for fid in a.flag_ids
        }
        assert not winsorized_ids & {f.flag_id for f in shifts}


class TestFalsePositives:
    def test_clean_world_fp_rate_below_2pct(self):
        sc = make_clean()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        n_points = len(sc.weeks) * (1 + len(sc.spend.columns))  # KPI + media
        point_flags = [f for f in report.flags if f.kind != "level_shift"]
        assert len(point_flags) / n_points < 0.02

    def test_seasonal_holiday_peaks_not_flagged_by_stl(self):
        sc = make_seasonality_misspec()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)

        n = len(sc.weeks)
        holiday_weeks = {
            str(sc.weeks[t].date()) for t in range(n) if (t % 52) in (47, 48, 50, 51)
        }
        stl_kpi_flags = {
            f.period
            for f in report.flags
            if f.variable == "Sales" and "stl_residual" in f.methods
        }
        flagged_holidays = stl_kpi_flags & holiday_weeks
        assert (
            len(flagged_holidays) <= len(holiday_weeks) // 3
        ), f"seasonal peaks misread as outliers: {sorted(flagged_holidays)}"


class TestApplyTreatments:
    def test_dummy_rows_have_kpi_grain_and_indicator_values(self):
        sc = make_heavy_tailed_noise()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        actions = recommend_treatments(panel, report.flags, report.config)
        dummies = [a for a in actions if a.strategy == "dummy"][:1]
        assert dummies

        treated = apply_treatments(panel.df_long, dummies, report.flags, kpi="Sales")
        name = dummies[0].params["dummy_name"]
        rows = treated[treated["VariableName"] == name]
        assert len(rows) == len(sc.weeks)  # full KPI grain
        assert set(rows["VariableValue"].unique()) == {0.0, 1.0}
        assert rows["VariableValue"].sum() == len(dummies[0].params["periods"])

    def test_exclude_periods_drops_rows(self):
        sc = make_clean()
        panel = panel_from_scenario(sc)
        from mmm_framework.eda.results import RemediationAction

        period = str(sc.weeks[0].date())
        action = RemediationAction(
            action_id="exclude:test",
            flag_ids=[],
            strategy="exclude_periods",
            params={"periods": [period]},
            rationale="test",
        )
        treated = apply_treatments(panel.df_long, [action], [])
        assert not (pd.to_datetime(treated["Period"]) == pd.Timestamp(period)).any()
        assert len(treated) < len(panel.df_long)

    def test_input_frame_not_mutated(self):
        sc = make_spend_outliers()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        actions = recommend_treatments(panel, report.flags, report.config)
        before = panel.df_long.copy()
        apply_treatments(panel.df_long, actions, report.flags, kpi="Sales")
        pd.testing.assert_frame_equal(panel.df_long, before)


class TestReportRoundTrip:
    def test_to_dict_from_dict(self):
        sc = make_spend_outliers()
        panel = panel_from_scenario(sc)
        report = detect_outliers(panel)
        report.actions = recommend_treatments(panel, report.flags, report.config)

        from mmm_framework.eda.results import OutlierReport

        d = report.to_dict()
        restored = OutlierReport.from_dict(d)
        assert len(restored.flags) == len(report.flags)
        assert len(restored.actions) == len(report.actions)
        assert {f.flag_id for f in restored.flags} == {f.flag_id for f in report.flags}


class TestSensitivity:
    def test_low_sensitivity_flags_no_more_than_default(self):
        sc = make_heavy_tailed_noise()
        panel = panel_from_scenario(sc)
        default = detect_outliers(panel, OutlierConfig.for_sensitivity("default"))
        low = detect_outliers(panel, OutlierConfig.for_sensitivity("low"))
        assert len(low.flags) <= len(default.flags)

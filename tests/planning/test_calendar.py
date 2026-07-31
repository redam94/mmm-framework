"""Plan periods join by label, and every date lands in exactly one (#216).

The defect this file guards is arithmetic, not statistics: plan and actual rows
were aligned POSITIONALLY while the source emitted them in LEXICOGRAPHIC label
order (``P1, P10, P11, P12, P13, P2, ...``). Per-channel totals survived because
sums are order-invariant, which is why it went unnoticed for so long.

Two traps this file is written around:

* **A flat plan hides the bug entirely.** If every period plans the same spend,
  positional and label joins agree. Every regression test here uses a RAMP.
* **A double-counted week is a double-counted dollar**, so the period map is
  checked exhaustive-and-disjoint over a full sweep rather than spot-checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.planning.calendar import (
    CalendarCoverageError,
    PlanningCalendar,
    label_sort_key,
)
from mmm_framework.planning.pacing import compute_pacing


# ---------------------------------------------------------------------------
# the shipped misalignment
# ---------------------------------------------------------------------------


class TestPlanActualJoin:
    def test_midflight_upload_joins_the_right_weeks(self):
        """A mid-flight upload for P5..P8 must compare against the plan's P5..P8.

        Pre-v1.4 this compared against P1..P4 and reported +296% over-pacing
        where the truth is +52%.
        """
        plan = [{"period": f"P{i}", "TV": i * 10.0} for i in range(1, 14)]
        actual = [{"period": f"P{i}", "TV": 99.0} for i in range(5, 9)]

        r = compute_pacing(plan, actual)
        row = r.channels[0]

        assert r.join == "label"
        assert r.periods == ["P5", "P6", "P7", "P8"]
        assert row.planned == pytest.approx(260.0)  # 50+60+70+80, NOT 10+20+30+40
        assert row.divergence_pct == pytest.approx((396.0 - 260.0) / 260.0, rel=1e-9)

    def test_the_ramp_is_what_makes_this_test_work(self):
        """Guard the guard: a FLAT plan cannot detect the misalignment.

        If someone 'simplifies' the fixture above to a constant spend, both the
        right and the wrong join return the same number and the regression goes
        silent.
        """
        flat_plan = [{"period": f"P{i}", "TV": 10.0} for i in range(1, 14)]
        actual = [{"period": f"P{i}", "TV": 99.0} for i in range(5, 9)]
        positional = sum(10.0 for _ in range(4))
        by_label = sum(10.0 for _ in range(5, 9))
        assert positional == by_label, "a flat plan is blind to the join order"

        r = compute_pacing(flat_plan, actual)
        assert r.channels[0].planned == pytest.approx(40.0)

    def test_row_order_does_not_change_the_answer(self):
        plan = [{"period": f"P{i}", "TV": i * 10.0} for i in range(1, 14)]
        actual = [{"period": f"P{i}", "TV": 99.0} for i in range(5, 9)]
        shuffled = [actual[i] for i in (2, 0, 3, 1)]
        assert (
            compute_pacing(plan, actual).channels[0].planned
            == compute_pacing(plan, shuffled).channels[0].planned
        )

    def test_a_delivery_period_outside_the_plan_is_excluded_from_both_sides(self):
        plan = [{"period": f"P{i}", "TV": i * 10.0} for i in range(1, 5)]
        actual = [{"period": p, "TV": 99.0} for p in ("P2", "P3", "P99")]
        r = compute_pacing(plan, actual)
        assert "P99" not in r.periods
        assert r.channels[0].planned == pytest.approx(50.0)  # P2+P3 = 20+30
        assert r.channels[0].actual == pytest.approx(198.0)  # P99 excluded too

    def test_unlabelled_inputs_fall_back_positionally(self):
        """Bare arrays have no vocabulary; the old behaviour must be preserved."""
        r = compute_pacing({"TV": [1.0, 2.0, 3.0, 4.0]}, {"TV": [1.0, 2.0]})
        assert r.join == "positional"
        assert r.channels[0].planned == pytest.approx(3.0)

    def test_scalar_totals_still_work(self):
        r = compute_pacing({"TV": 100.0}, {"TV": 80.0})
        assert r.join == "positional"
        assert r.channels[0].divergence_pct == pytest.approx(-0.2)


def test_label_sort_key_orders_p2_before_p10():
    labels = [f"P{i}" for i in range(1, 14)]
    assert sorted(labels) != labels, "lexicographic order is the bug"
    assert sorted(labels, key=label_sort_key) == labels


# ---------------------------------------------------------------------------
# the calendar
# ---------------------------------------------------------------------------


class TestPeriodMap:
    @pytest.mark.parametrize("n", [1, 2, 13, 52, 53, 104, 500])
    @pytest.mark.parametrize("fy", [1, 2, 4, 7, 12])
    def test_exhaustive_and_disjoint(self, n, fy):
        """Every date in the window maps to exactly one period.

        A date in two periods is a dollar counted twice; a date in none is a
        dollar dropped.
        """
        cal = PlanningCalendar(
            start="2024-01-01", n_periods=n, cadence="weekly", fy_start_month=fy
        )
        days = pd.date_range("2024-01-01", periods=n * 7, freq="D")
        labels = cal.period_of(days)
        assert all(lab is not None for lab in labels)
        counts: dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        assert len(counts) == n
        assert set(counts.values()) == {7}
        assert sum(counts.values()) == len(days)

    def test_dates_outside_the_window_are_none(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=4)
        got = cal.period_of(["2023-12-31", "2024-01-01", "2024-01-29"])
        assert got[0] is None and got[1] is not None and got[2] is None

    def test_labels_are_unique_or_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            PlanningCalendar(start="2024-01-01", n_periods=3, labels=("a", "b", "a"))

    def test_label_count_must_match(self):
        with pytest.raises(ValueError, match="entries"):
            PlanningCalendar(start="2024-01-01", n_periods=3, labels=("a", "b"))

    def test_bad_cadence_and_fy_month(self):
        with pytest.raises(ValueError, match="cadence"):
            PlanningCalendar(start="2024-01-01", n_periods=3, cadence="fortnightly")
        with pytest.raises(ValueError, match="fy_start_month"):
            PlanningCalendar(start="2024-01-01", n_periods=3, fy_start_month=13)


class TestFiscalGroups:
    def test_52_weeks_group_by_fiscal_year(self):
        cal = PlanningCalendar(
            start="2024-01-01", n_periods=52, cadence="weekly", fy_start_month=2
        )
        groups = cal.fiscal_groups()
        assert sum(len(v) for v in groups.values()) == 52
        # every label appears exactly once across groups
        flat = [lab for v in groups.values() for lab in v]
        assert len(set(flat)) == 52

    def test_december_can_belong_to_the_next_fiscal_year(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=4, fy_start_month=7)
        assert cal.fiscal_year_of("2024-12-15") == 2025
        assert cal.fiscal_year_of("2024-03-15") == 2024

    def test_calendar_year_when_fy_starts_in_january(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=4, fy_start_month=1)
        assert cal.fiscal_year_of("2024-12-15") == 2024


class TestCoverageRefusal:
    def test_partial_coverage_raises_naming_the_dates(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=4)
        with pytest.raises(CalendarCoverageError, match="outside the plan window"):
            cal.require_covers(["2024-01-02", "2024-06-01"])

    def test_full_coverage_is_silent(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=4)
        cal.require_covers(["2024-01-02", "2024-01-20"])

    def test_the_message_names_a_date(self):
        cal = PlanningCalendar(start="2024-01-01", n_periods=2)
        with pytest.raises(CalendarCoverageError, match="2024-06-01"):
            cal.require_covers(["2024-06-01"])


def test_period_bounds_are_contiguous_and_half_open():
    cal = PlanningCalendar(start="2024-01-01", n_periods=3, cadence="weekly")
    bounds = [cal.period_bounds(lab) for lab in cal.periods()]
    for (_, end), (nxt, _) in zip(bounds, bounds[1:]):
        assert end == nxt, "a gap or overlap between periods is a mis-counted week"


def test_to_dict_round_trips_the_vocabulary():
    cal = PlanningCalendar(start="2024-01-01", n_periods=5, fy_start_month=4)
    d = cal.to_dict()
    assert d["n_periods"] == 5 and d["fy_start_month"] == 4
    assert d["labels"] == cal.periods()


# ---------------------------------------------------------------------------
# the seasonality-constant drift hazard
# ---------------------------------------------------------------------------


def test_seasonality_period_constant_matches():
    """The seasonality period table has ONE definition, not copies that agree.

    `model/base.py` and `validation/backtest.py` used to hold separate literals
    of this table; they must agree or a forecast evaluates the Fourier basis at
    a different phase than the fit, and nothing pinned them equal before #216.
    That version of this test scraped the inlined literal out of
    `_prepare_seasonality` with a regex and compared the two dicts.

    Since #275 there is nothing to compare: both sites read
    `transforms.seasonality.PERIODS_BY_FREQ`. So this asserts the stronger
    property — same object, not equal copies — and fails loudly if either site
    goes back to inlining one. (A third implementation, the extension graphs'
    datetime-median rule, is a deliberate divergence behind
    `SeasonalityConfig.period_source`; see
    tests/mmm_extensions/test_seasonal_period_source.py.)
    """
    import inspect

    from mmm_framework.model import base
    from mmm_framework.transforms.seasonality import PERIODS_BY_FREQ
    from mmm_framework.validation import backtest

    assert backtest._PERIODS_BY_FREQ is PERIODS_BY_FREQ, (
        "validation/backtest.py no longer reads the shared seasonality period "
        "table; a copy here is how the two drifted in the first place"
    )

    src = inspect.getsource(base.BayesianMMM._prepare_seasonality)
    assert "periods_for_frequency" in src, (
        "model/base.py no longer resolves periods through the shared table"
    )
    assert '"yearly":' not in src, (
        "model/base.py has re-inlined a seasonality period literal:\n" + src
    )

    # The values themselves, so a change to the shared table is a deliberate act.
    assert PERIODS_BY_FREQ == {
        "W": {"yearly": 52.0, "monthly": 52.0 / 12.0},
        "D": {"yearly": 365.25, "monthly": 365.25 / 12.0, "weekly": 7.0},
        "M": {"yearly": 12.0},
    }


def test_the_fitted_seasonal_basis_is_the_shared_tables():
    """Behavioural, not textual: what the model BUILDS, at every frequency.

    The source-level assertions above catch a re-inlined literal. This catches
    the subtler drift they cannot — the indirection resolving to the wrong
    number — by grading the design matrix the model actually constructs against
    the shared table, for each tabulated frequency and each component it
    tabulates.
    """
    import numpy as np

    from mmm_framework.config import InferenceMethod, ModelConfig, SeasonalityConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.synth import dgp
    from mmm_framework.transforms.seasonality import (
        PERIODS_BY_FREQ,
        create_fourier_features,
    )

    panel = dgp.build("clean", seed=3, n_weeks=80).panel()

    for freq, table in PERIODS_BY_FREQ.items():
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC, n_chains=1, n_draws=4
        )
        cfg.seasonality = SeasonalityConfig(
            yearly=1 if "yearly" in table else None,
            monthly=1 if "monthly" in table else None,
            weekly=1 if "weekly" in table else None,
        )
        m = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
        m.mff_config.frequency = freq
        m._prepare_seasonality()

        t = np.arange(m.n_periods)
        for component, features in m.seasonality_features.items():
            order = features.shape[1] // 2
            want = create_fourier_features(t, table[component], order)
            np.testing.assert_allclose(
                features,
                want,
                rtol=0,
                atol=1e-12,
                err_msg=(
                    f"{freq}/{component} basis is not the shared table's "
                    f"period {table[component]}"
                ),
            )


def test_calendar_from_model_uses_the_panels_own_cadence():
    """Derived from the panel, never invented."""

    class _Panel:
        class coords:
            periods = pd.date_range("2021-01-04", periods=10, freq="W-MON")

    class _Model:
        panel = _Panel()

    cal = PlanningCalendar.from_model(_Model(), 13)
    assert cal.cadence == "weekly"
    assert cal.n_periods == 13
    # starts the period AFTER the model's last
    assert cal.start == pd.Timestamp("2021-01-04") + pd.Timedelta(weeks=10)
    assert len(set(cal.periods())) == 13
    assert np.all(np.diff(cal.starts().values).astype("timedelta64[D]").astype(int) == 7)


# ---------------------------------------------------------------------------
# flighting schedules carry real labels
# ---------------------------------------------------------------------------


class TestFlightingLabels:
    def test_a_calendar_gives_dated_labels_that_sort_correctly(self):
        from mmm_framework.planning.flighting import build_flighting_schedule

        cal = PlanningCalendar(start="2024-01-01", n_periods=13)
        sched = build_flighting_schedule({"TV": 1300.0}, 13, calendar=cal)
        labels = [r["period"] for r in sched["schedule"]]
        assert labels == cal.periods()
        assert sorted(labels) == labels, "dated labels must sort in plan order"

    def test_the_default_fallback_is_unchanged(self):
        from mmm_framework.planning.flighting import build_flighting_schedule

        sched = build_flighting_schedule({"TV": 1300.0}, 13)
        labels = [r["period"] for r in sched["schedule"]]
        assert labels == [f"P{i}" for i in range(1, 14)]
        # ...and this is exactly the vocabulary that sorts wrong, which is why
        # a calendar is preferred wherever one exists.
        assert sorted(labels) != labels

    def test_a_short_calendar_is_refused_not_padded(self):
        from mmm_framework.planning.flighting import build_flighting_schedule

        cal = PlanningCalendar(start="2024-01-01", n_periods=13)
        with pytest.raises(ValueError, match="refusing"):
            build_flighting_schedule({"TV": 1.0}, 20, calendar=cal)

    def test_explicit_period_labels_still_win(self):
        from mmm_framework.planning.flighting import build_flighting_schedule

        cal = PlanningCalendar(start="2024-01-01", n_periods=4)
        sched = build_flighting_schedule(
            {"TV": 4.0}, 4, calendar=cal, period_labels=["W1", "W2", "W3", "W4"]
        )
        assert [r["period"] for r in sched["schedule"]] == ["W1", "W2", "W3", "W4"]

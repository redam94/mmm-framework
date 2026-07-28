"""Tests for the Planner backend: geo/DMA allocation (B4), the forward flighting
calendar (B6), and the plan_budget / plan_scenario model-ops the Planner FE drives.

The allocator math is exercised against a fake fitted model whose per-(geo,
channel) response is a known concave (sqrt) curve, so optima are predictable
without a real MCMC fit. One slow end-to-end test runs the op against a real fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.agents import model_ops as M
from mmm_framework.planning import (
    build_flighting_schedule,
    combine_geo_curves,
    compute_response_curves_per_geo,
    optimize_budget_by_geo,
)
from mmm_framework.planning.flighting import _pattern_weights


class FakeGeoMMM:
    """Additive geo panel: contribution(obs, ch) = coef[geo, ch] * sqrt(spend)."""

    def __init__(self):
        self.channel_names = ["TV", "Search"]
        self.geo_names = ["North", "South"]
        self.has_geo = True
        self.n_geos = 2
        # 2 geos × 3 periods = 6 obs
        self.geo_idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        self.X_media_raw = np.full((6, 2), 100.0)
        # North favors TV, South favors Search (mirror image) → national is
        # symmetric, but per-geo reveals the heterogeneity.
        self._coef = np.array([[3.0, 1.0], [1.0, 3.0]])

    def sample_channel_contributions(
        self, X_media=None, max_draws=None, random_seed=None
    ):
        X = self.X_media_raw if X_media is None else X_media
        n_obs, C = X.shape
        out = np.zeros((4, n_obs, C))
        for i in range(n_obs):
            g = int(self.geo_idx[i])
            for c in range(C):
                out[:, i, c] = self._coef[g, c] * np.sqrt(max(X[i, c], 0.0))
        return out

    def what_if_scenario(
        self, spend_changes, time_period=None, max_draws=200, random_seed=None
    ):
        return {
            "baseline_outcome": 1000.0,
            "scenario_outcome": 1100.0,
            "outcome_change": 100.0,
            "outcome_change_pct": 10.0,
            "spend_changes": {
                ch: {
                    "original": 100.0,
                    "scenario": 100.0 * (1 + v),
                    "change": 100.0 * v,
                    "change_pct": 100.0 * v,
                }
                for ch, v in spend_changes.items()
            },
            "outcome_change_hdi": [50.0, 150.0],
            "prob_positive": 0.9,
            "n_draws": 50,
            "hdi_prob": 0.94,
        }


class TestFlighting:
    def test_even_conserves_and_is_flat(self):
        fl = build_flighting_schedule({"TV": 130.0}, 13, pattern="even")
        assert len(fl["schedule"]) == 13
        assert sum(r["total"] for r in fl["schedule"]) == pytest.approx(130.0)
        vals = [r["TV"] for r in fl["schedule"]]
        assert max(vals) - min(vals) < 1e-9

    def test_front_loaded_decreases(self):
        w = _pattern_weights("front_loaded", 5)
        assert w[0] > w[-1]
        assert w.sum() == pytest.approx(1.0)

    def test_back_loaded_increases(self):
        w = _pattern_weights("back_loaded", 5)
        assert w[0] < w[-1]

    def test_pulsed_has_off_periods(self):
        fl = build_flighting_schedule(
            {"TV": 100.0}, 6, pattern="pulsed", pulse_on=1, pulse_off=1
        )
        spend = [r["TV"] for r in fl["schedule"]]
        assert spend[0] > 0 and spend[1] == 0  # on, off, on, off...
        assert sum(spend) == pytest.approx(100.0)

    def test_conserves_per_channel_budget(self):
        fl = build_flighting_schedule(
            {"TV": 200.0, "Search": 50.0}, 8, pattern="front_loaded"
        )
        assert sum(fl["by_channel"]["TV"]) == pytest.approx(200.0)
        assert sum(fl["by_channel"]["Search"]) == pytest.approx(50.0)
        assert fl["total_budget"] == pytest.approx(250.0)

    def test_per_channel_pattern_override(self):
        fl = build_flighting_schedule(
            {"TV": 100.0, "Search": 100.0},
            4,
            pattern="even",
            per_channel_pattern={"Search": "pulsed"},
        )
        assert min(fl["by_channel"]["TV"]) == pytest.approx(max(fl["by_channel"]["TV"]))
        assert min(fl["by_channel"]["Search"]) == 0.0  # pulsed has off weeks

    def test_custom_pattern_validates_length(self):
        with pytest.raises(ValueError):
            _pattern_weights("custom", 4, custom=[1, 2])

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError):
            _pattern_weights("nope", 4)


class TestGeoAllocation:
    def test_per_geo_curves_recover_heterogeneity(self):
        mmm = FakeGeoMMM()
        curves = compute_response_curves_per_geo(mmm, max_draws=4)
        assert set(curves) == {"North", "South"}
        # each geo's base spend is its 3 periods × 100
        assert curves["North"].base_spend.tolist() == [300.0, 300.0]

    def test_combine_then_optimize_moves_money_within_geo(self):
        mmm = FakeGeoMMM()
        res = optimize_budget_by_geo(mmm, max_draws=4, n_steps=600)
        t = res.table
        assert {"geo", "channel"} <= set(t.columns)
        north = t[t["geo"] == "North"].set_index("channel")
        south = t[t["geo"] == "South"].set_index("channel")
        # North favors TV, South favors Search — money follows the per-geo curve
        assert north.loc["TV", "optimal_spend"] > north.loc["Search", "optimal_spend"]
        assert south.loc["Search", "optimal_spend"] > south.loc["TV", "optimal_spend"]
        # national budget is conserved
        assert t["optimal_spend"].sum() == pytest.approx(res.total_budget, rel=0.02)

    def test_combine_geo_curves_flattens_arms(self):
        mmm = FakeGeoMMM()
        curves = compute_response_curves_per_geo(mmm, max_draws=4)
        combined = combine_geo_curves(curves)
        assert len(combined.channel_names) == 4  # 2 geos × 2 channels
        assert combined.base_spend.shape == (4,)

    def test_per_geo_curves_carry_observed_range(self):
        """Each geo's own observed spend range flows onto its curves + arms so a
        geo-level recommendation can be flagged (issue #121)."""
        mmm = FakeGeoMMM()
        curves = compute_response_curves_per_geo(mmm, max_draws=4)
        # constant spend (100 each, 3 periods) → max==mean → multiplier 1.0
        north = curves["North"]
        assert north.n_obs == 3
        assert north.obs_max_spend.tolist() == [100.0, 100.0]
        assert np.allclose(north.max_obs_multiplier, [1.0, 1.0])
        # combined arms keep a per-arm n_obs (ragged-panel safe) + the multiplier
        combined = combine_geo_curves(curves)
        assert np.asarray(combined.n_obs).tolist() == [3, 3, 3, 3]
        assert np.allclose(combined.max_obs_multiplier, [1.0, 1.0, 1.0, 1.0])

    def test_geo_allocation_flags_extrapolation_beyond_geo_range(self):
        """A geo×channel arm scaled past that geo's observed spend range is
        flagged within_observed_range=False, exactly like the national path."""
        mmm = FakeGeoMMM()  # constant spend → any scale-up in a geo extrapolates
        res = optimize_budget_by_geo(mmm, max_draws=4, n_steps=600)
        t = res.table.set_index(["geo", "channel"])
        # North favors TV → TV scales up past 1.0× (flagged); Search scales down.
        assert bool(t.loc[("North", "TV"), "within_observed_range"]) is False
        assert bool(t.loc[("North", "Search"), "within_observed_range"]) is True
        assert t.loc[("North", "TV"), "max_obs_multiplier"] == pytest.approx(1.0)
        assert res.n_extrapolated >= 1

    def test_combined_arms_n_obs_is_per_geo_for_ragged_panels(self):
        """Geos with different period counts get the RIGHT per-arm multiplier — a
        single scalar n_obs would mis-scale the spiky geo."""

        class RaggedGeoMMM:
            channel_names = ["TV"]
            geo_names = ["A", "B"]
            has_geo = True
            n_geos = 2
            # geo A: 2 spiky periods (10, 90) → mean 50, max 90 → mult 1.8;
            # geo B: 3 flat periods (100) → mult 1.0.
            geo_idx = np.array([0, 0, 1, 1, 1], dtype=np.int32)
            X_media_raw = np.array([[10.0], [90.0], [100.0], [100.0], [100.0]])

            def sample_channel_contributions(
                self, X_media=None, max_draws=None, random_seed=None
            ):
                X = self.X_media_raw if X_media is None else X_media
                return np.sqrt(np.clip(X, 0, None))[None, :, :] * np.ones((3, 1, 1))

        curves = compute_response_curves_per_geo(RaggedGeoMMM(), max_draws=3)
        combined = combine_geo_curves(curves)
        assert np.asarray(combined.n_obs).tolist() == [2, 3]
        assert np.allclose(combined.max_obs_multiplier, [1.8, 1.0])


class TestPlanOps:
    def test_plan_budget_national(self):
        res = M.OPS["plan_budget"](FakeGeoMMM(), None, by_geo=False, max_draws=4)
        assert not res.get("error"), res.get("error")
        plan = res["dashboard"]["budget_plan"]
        assert plan["by_geo"] is False
        assert {a["channel"] for a in plan["allocation"]} == {"TV", "Search"}
        assert "geo_allocation" not in plan
        assert res["tables"]

    def test_plan_budget_geo_and_flighting(self):
        res = M.OPS["plan_budget"](
            FakeGeoMMM(),
            None,
            by_geo=True,
            flighting={"pattern": "front_loaded", "n_periods": 8},
            max_draws=4,
        )
        assert not res.get("error"), res.get("error")
        plan = res["dashboard"]["budget_plan"]
        assert plan["by_geo"] is True
        assert plan["geos"] == ["North", "South"]
        assert len(plan["geo_allocation"]) == 4
        # Issue #121: per-geo rows carry the extrapolation flag (dropped before).
        for r in plan["geo_allocation"]:
            assert "within_observed_range" in r
            assert "max_obs_multiplier" in r
        # constant-spend fixture → scaled-up arms are flagged as extrapolating
        assert plan.get("n_extrapolated", 0) >= 1
        fl = plan["flighting"]
        assert fl["pattern"] == "front_loaded" and fl["n_periods"] == 8
        # flighting spreads the rolled-up channel budgets
        total_sched = sum(r["total"] for r in fl["schedule"])
        total_alloc = sum(a["optimal_spend"] for a in plan["allocation"])
        assert total_sched == pytest.approx(total_alloc, rel=0.02)

    def test_plan_budget_rejects_bad_bounds(self):
        res = M.OPS["plan_budget"](FakeGeoMMM(), None, bounds={"Ghost": [0, 1]})
        assert res.get("error") and "Unknown channel" in res["error"]

    def test_plan_scenario_structured_output(self):
        res = M.OPS["plan_scenario"](
            FakeGeoMMM(), None, spend_changes={"TV": 0.2}, max_draws=4
        )
        assert not res.get("error"), res.get("error")
        sc = res["dashboard"]["budget_scenario"]
        assert sc["outcome_change"] == pytest.approx(100.0)
        assert sc["outcome_change_hdi"] == [50.0, 150.0]
        assert sc["prob_positive"] == pytest.approx(0.9)
        assert sc["channel_details"]["TV"]["change_pct"] == pytest.approx(20.0)

    def test_plan_scenario_rejects_unknown_channel(self):
        res = M.OPS["plan_scenario"](FakeGeoMMM(), None, spend_changes={"Ghost": 0.2})
        assert res.get("error") and "Unknown channel" in res["error"]


@pytest.mark.slow
def test_plan_budget_end_to_end_with_fitted_model(tmp_path):
    """Full path: fit a tiny national model, then run plan_budget with flighting."""
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    from mmm_framework.agents.fitting import build_model

    df = generate_synthetic_mff(n_weeks=60)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "inference": {"chains": 2, "draws": 100, "tune": 100},
    }
    mmm = build_model(spec, path)
    mmm.fit(random_seed=1)

    res = M.OPS["plan_budget"](
        mmm, None, flighting={"pattern": "even", "n_periods": 13}, max_draws=50
    )
    assert not res.get("error"), res.get("error")
    plan = res["dashboard"]["budget_plan"]
    assert {a["channel"] for a in plan["allocation"]} == {"TV", "Digital"}
    assert len(plan["flighting"]["schedule"]) == 13
    total_sched = sum(r["total"] for r in plan["flighting"]["schedule"])
    total_alloc = sum(a["optimal_spend"] for a in plan["allocation"])
    assert total_sched == pytest.approx(total_alloc, rel=0.02)


@pytest.mark.slow
def test_default_reallocation_and_augur_report_end_to_end(tmp_path):
    """Real fit → default_reallocation (±20%, within support) → the plan lands in
    the Augur "Media Performance Readout" as a rendered allocation section."""
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    from mmm_framework.agents.fitting import build_model
    from mmm_framework.planning import default_reallocation
    from mmm_framework.reporting.generator import ReportBuilder

    df = generate_synthetic_mff(n_weeks=60)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "inference": {"chains": 2, "draws": 100, "tune": 100},
    }
    mmm = build_model(spec, path)
    mmm.fit(random_seed=1)

    plan = default_reallocation(mmm, max_draws=50)
    # ±20% within support, no channel switched off, total preserved — on REAL curves.
    assert plan["deviation_cap"] == pytest.approx(0.20)
    cur = sum(r["current_spend"] for r in plan["allocation"])
    opt = sum(r["optimal_spend"] for r in plan["allocation"])
    assert opt == pytest.approx(cur, rel=0.02)
    for r in plan["allocation"]:
        assert 0.8 * r["current_spend"] - 1e-6 <= r["optimal_spend"]
        assert r["optimal_spend"] <= 1.2 * r["current_spend"] + 1e-6
        assert r["optimal_spend"] > 0

    html = (
        ReportBuilder()
        .with_model(mmm)
        .with_title("Media Performance Readout")
        .with_client("Acme")
        .augur_readout()
        .with_allocation(plan)
        .build()
        .render()
    )
    assert 'id="allocation"' in html
    assert "The optimized plan" in html
    assert "±20%" in html


class FakeNationalMMM:
    """Additive national model: contribution(obs, ch) = coef[ch] * sqrt(spend).

    TV has the higher marginal return, so a budget-neutral reallocation should
    push spend toward TV and trim Search — but only up to the deviation cap.
    """

    def __init__(self, coef=(3.0, 1.0), base=100.0, n_obs=4):
        self.channel_names = ["TV", "Search"]
        self.X_media_raw = np.full((n_obs, 2), float(base))
        self._coef = np.asarray(coef, dtype=float)

    def sample_channel_contributions(
        self, X_media=None, max_draws=None, random_seed=None
    ):
        X = self.X_media_raw if X_media is None else X_media
        n_obs, C = X.shape
        out = np.zeros((6, n_obs, C))
        for c in range(C):
            out[:, :, c] = self._coef[c] * np.sqrt(np.clip(X[:, c], 0.0, None))
        return out


class TestDefaultReallocation:
    """planning.default_reallocation: the ±20%, within-support report plan."""

    def test_bounds_no_channel_off_total_preserved(self):
        from mmm_framework.planning import default_reallocation

        plan = default_reallocation(FakeNationalMMM(), max_draws=20)
        assert plan["deviation_cap"] == pytest.approx(0.20)

        cur = sum(r["current_spend"] for r in plan["allocation"])
        opt = sum(r["optimal_spend"] for r in plan["allocation"])
        assert opt == pytest.approx(cur, rel=0.02)  # pure reallocation

        for r in plan["allocation"]:
            lo, hi = 0.8 * r["current_spend"], 1.2 * r["current_spend"]
            # every channel stays within ±20% (the sampled support) ...
            assert lo - 1e-6 <= r["optimal_spend"] <= hi + 1e-6
            # ... and none is switched off
            assert r["optimal_spend"] > 0
            assert -20.5 <= r["change_pct"] <= 20.5

    def test_moves_budget_toward_higher_marginal(self):
        from mmm_framework.planning import default_reallocation

        rows = {
            r["channel"]: r
            for r in default_reallocation(
                FakeNationalMMM(coef=(3.0, 1.0)), max_draws=20
            )["allocation"]
        }
        assert rows["TV"]["change_pct"] > 0  # winner scaled up
        assert rows["Search"]["change_pct"] < 0  # loser trimmed
        # the winner is pushed to (but not past) the +20% cap
        assert rows["TV"]["optimal_spend"] == pytest.approx(
            1.2 * rows["TV"]["current_spend"], rel=0.05
        )

    def test_custom_deviation_tightens_band(self):
        from mmm_framework.planning import default_reallocation

        plan = default_reallocation(FakeNationalMMM(), deviation=0.10, max_draws=20)
        assert plan["deviation_cap"] == pytest.approx(0.10)
        for r in plan["allocation"]:
            assert (
                0.9 * r["current_spend"] - 1e-6
                <= r["optimal_spend"]
                <= 1.1 * r["current_spend"] + 1e-6
            )

    def test_rejects_out_of_range_deviation(self):
        from mmm_framework.planning import default_reallocation

        with pytest.raises(ValueError):
            default_reallocation(FakeNationalMMM(), deviation=1.5)


# ---------------------------------------------------------------------------
# A saved plan carries its own dates (#216)
#
# `build_flighting_schedule` was called with no calendar, so every planner-built
# plan got P1..Pn labels. `compute_pacing` cannot join those to dated delivery,
# falls back to POSITIONAL truncation, and compares a mid-flight upload against
# the plan's FIRST periods. Measured on a ramped 4-week plan with delivery for
# weeks 3-4: +25.0% over-pacing reported where the truth is +87.5%.
# ---------------------------------------------------------------------------


class _DatedMMM(FakeGeoMMM):
    """FakeGeoMMM plus the dated panel index a real model always has."""

    def __init__(self, freq="W-MON", periods=6):
        super().__init__()

        class _P:
            pass

        self.panel = _P()
        self.panel.index = pd.date_range("2025-01-06", periods=periods, freq=freq)


class TestPlanCalendar:
    def test_the_plan_window_starts_after_the_training_data(self):
        from mmm_framework.agents.model_ops import _forward_calendar

        cal = _forward_calendar(_DatedMMM(), 4, {})
        assert cal is not None
        # training ends 2025-02-10; the plan starts the NEXT week
        assert str(cal.start.date()) == "2025-02-17"
        assert cal.cadence == "weekly" and cal.n_periods == 4

    def test_an_explicit_start_date_wins(self):
        from mmm_framework.agents.model_ops import _forward_calendar

        cal = _forward_calendar(_DatedMMM(), 4, {"start_date": "2030-01-07"})
        assert str(cal.start.date()) == "2030-01-07"

    def test_an_undated_panel_yields_no_calendar_rather_than_invented_dates(self):
        from mmm_framework.agents.model_ops import _forward_calendar

        assert _forward_calendar(FakeGeoMMM(), 4, {}) is None

    def test_the_schedule_and_the_saved_plan_carry_the_dates(self):
        res = M.OPS["plan_budget"](
            _DatedMMM(),
            None,
            by_geo=False,
            flighting={"pattern": "front_loaded", "n_periods": 4},
            max_draws=4,
        )
        plan = res["dashboard"]["budget_plan"]
        assert plan["flighting"]["periods"][0] == "2025-02-17"
        # persisted, so a plan read back does not depend on its reader
        # reconstructing a calendar
        assert plan["calendar"] == {
            "start": "2025-02-17",
            "n_periods": 4,
            "cadence": "weekly",
            "fy_start_month": 1,
        }

    def test_dated_labels_change_the_pacing_verdict_mid_flight(self):
        """The bug, end to end. Same plan, same delivery — only the labels
        differ, and the reported over-pacing is 3.5x off without them."""
        from mmm_framework.planning.calendar import PlanningCalendar
        from mmm_framework.planning.flighting import build_flighting_schedule
        from mmm_framework.platform.pacing import project_pacing

        # delivery for the LAST two weeks only, as a mid-flight upload is
        mid = [
            {"channel": "TV", "period": d, "spend": 150.0}
            for d in ("2025-01-20", "2025-01-27")
        ]
        cal = PlanningCalendar(start="2025-01-06", n_periods=4, cadence="weekly")

        undated = build_flighting_schedule({"TV": 400.0}, 4, pattern="front_loaded")
        dated = build_flighting_schedule(
            {"TV": 400.0}, 4, pattern="front_loaded", calendar=cal
        )
        assert undated["periods"] == ["P1", "P2", "P3", "P4"]

        a = project_pacing({"flighting": undated}, mid)
        b = project_pacing({"flighting": dated}, mid)
        assert a["join"] == "positional" and b["join"] == "label"

        # the plan ramps 130/110/90/70; weeks 3-4 are 160, not the first two's 240
        assert a["channels"][0]["planned"] == pytest.approx(240.0)
        assert b["channels"][0]["planned"] == pytest.approx(160.0)
        assert a["channels"][0]["divergence_pct"] == pytest.approx(0.25)
        assert b["channels"][0]["divergence_pct"] == pytest.approx(0.875)

    def test_a_flat_plan_cannot_detect_this(self):
        """Companion to the above, so a future 'simplification' to a flat
        fixture cannot silently disarm the guard (#216's own test-design note)."""
        from mmm_framework.planning.calendar import PlanningCalendar
        from mmm_framework.planning.flighting import build_flighting_schedule
        from mmm_framework.platform.pacing import project_pacing

        mid = [
            {"channel": "TV", "period": d, "spend": 150.0}
            for d in ("2025-01-20", "2025-01-27")
        ]
        cal = PlanningCalendar(start="2025-01-06", n_periods=4, cadence="weekly")
        flat_undated = build_flighting_schedule({"TV": 400.0}, 4, pattern="even")
        flat_dated = build_flighting_schedule(
            {"TV": 400.0}, 4, pattern="even", calendar=cal
        )
        a = project_pacing({"flighting": flat_undated}, mid)
        b = project_pacing({"flighting": flat_dated}, mid)
        assert a["channels"][0]["planned"] == b["channels"][0]["planned"]


# ---------------------------------------------------------------------------
# forecast_plan op (#223 wiring)
#
# `planning/forecast.py` shipped reachable only by a direct library call. This
# is the op behind the agent tool and the REST job.
# ---------------------------------------------------------------------------


class TestForecastOp:
    def _fitted(self, n_weeks=120):
        import contextlib
        import io
        import warnings

        from mmm_framework import BayesianMMM, ModelConfigBuilder, TrendConfig
        from mmm_framework.model import TrendType
        from mmm_framework.synth import dgp

        w = dgp.make_clean(seed=0, n_weeks=n_weeks)
        m = BayesianMMM(
            w.panel(),
            ModelConfigBuilder()
            .bayesian_numpyro()
            .with_chains(2)
            .with_draws(200)
            .with_tune(200)
            .build(),
            TrendConfig(type=TrendType.LINEAR),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                m.fit(random_seed=0)
        return m, w

    def _controls(self, w, n):
        return {c: [float(w.controls[c].mean())] * n for c in w.controls.columns}

    def test_an_unfitted_model_is_refused(self):
        res = M.OPS["forecast_plan"](FakeGeoMMM(), None, channel_budgets={"TV": 100.0})
        assert res["error"] and "no posterior" in res["error"]

    def test_neither_plan_shape_supplied_is_refused(self):
        m, _ = self._fitted()
        res = M.OPS["forecast_plan"](m, None)
        assert res["error"] and "future_media" in res["error"]

    @pytest.mark.slow
    def test_budgets_are_spread_and_the_payload_is_complete(self):
        m, w = self._fitted()
        res = M.OPS["forecast_plan"](
            m,
            None,
            channel_budgets={c: 800.0 for c in w.channels},
            n_periods=8,
            future_controls=self._controls(w, 8),
        )
        assert not res.get("error"), res.get("error")
        d = res["dashboard"]["forecast"]
        assert len(d["periods"]) == 8 and len(d["mean"]) == 8
        # dated, not P1..Pn — the plan window starts after training
        assert d["periods"][0].startswith("20")
        assert d["calendar"]["cadence"] == "weekly"
        # the DRAWS ride along: a window total cannot be rebuilt from bounds
        assert d["draws_b64"] and d["n_draws"] > 1
        assert set(d["by_channel"]) == set(w.channels)
        assert res["tables"]

    @pytest.mark.slow
    def test_the_markdown_leads_with_the_caveats(self):
        """A reader who stops after the headline has still been told how the
        interval is optimistic — so the caveats come first, not last."""
        m, w = self._fitted()
        res = M.OPS["forecast_plan"](
            m,
            None,
            # 5x the observed spend, so the extrapolation caveat definitely fires
            channel_budgets={c: 20000.0 for c in w.channels},
            n_periods=8,
            future_controls=self._controls(w, 8),
        )
        body = res["content"]
        assert "⚠️" in body
        first_caveat = body.index("⚠️")
        assert first_caveat < body.index("Total forecast KPI")
        assert "Planned above observed spend" in body

    @pytest.mark.slow
    def test_missing_controls_surface_as_an_op_error_not_a_crash(self):
        m, w = self._fitted()
        res = M.OPS["forecast_plan"](
            m, None, channel_budgets={c: 800.0 for c in w.channels}, n_periods=8
        )
        assert res["error"] and "planning assumption" in res["error"]

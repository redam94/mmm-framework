"""Tests for the experiment design engine (planning/design.py): matched-pair
geo assignment with randomization, DiD power math on the ROAS scale, placebo
calibration, and budget-neutral randomized flighting schedules."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning.design import (
    design_experiment,
    design_options,
    flighting_design,
    geo_lift_design,
    matched_pairs,
)


@pytest.fixture(scope="module")
def geo_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("geo_heterogeneous", seed=3, n_weeks=80)
    path = tmp_path_factory.mktemp("design") / "geo.csv"
    df.to_csv(path, index=False)
    return str(path), key


@pytest.fixture(scope="module")
def national_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("realistic", seed=7, n_weeks=80)
    path = tmp_path_factory.mktemp("design") / "nat.csv"
    df.to_csv(path, index=False)
    return str(path), key


class TestDesignOptions:
    def test_geo_panel_offers_geo_designs(self, geo_csv):
        path, key = geo_csv
        opts = design_options(path, "Sales", key["channels"][0])
        assert opts["recommended"] == "geo_lift"
        assert "matched_market_did" in opts["designs"]
        assert opts["n_geos"] >= 4

    def test_national_offers_flighting_only(self, national_csv):
        path, key = national_csv
        opts = design_options(path, "Sales", key["channels"][0])
        assert opts["designs"] == ["national_flighting"]
        assert opts["recommended"] == "national_flighting"


class TestGeoLift:
    def test_randomized_design_shape_and_power(self, geo_csv):
        path, key = geo_csv
        d = geo_lift_design(path, "Sales", key["channels"][0], duration=8, seed=11)
        assert d["design_key"] == "geo_lift" and d["randomized"]
        # complete partition into pairs; treatment/control disjoint
        assert set(d["treatment_geos"]).isdisjoint(d["control_geos"])
        assert len(d["assignment"]) == d["n_pairs"] >= 2
        for p in d["assignment"]:
            assert -1 <= p["correlation"] <= 1
        # power: MDE = 2.8 × SE, monotone decreasing in duration on the curve
        assert d["mde_roas"] == pytest.approx(2.8 * d["se_roas"])
        ses = [
            p["se_roas"] for p in sorted(d["power_curve"], key=lambda x: x["duration"])
        ]
        assert all(a >= b for a, b in zip(ses, ses[1:]))
        assert d["placebo"]["n_windows"] > 0 and d["placebo"]["sd"] > 0
        assert "difference-in-differences" in d["analysis_plan"]

    def test_randomization_is_seeded_and_varies(self, geo_csv):
        path, key = geo_csv
        ch = key["channels"][0]
        a = geo_lift_design(path, "Sales", ch, seed=1)
        b = geo_lift_design(path, "Sales", ch, seed=1)
        assert a["treatment_geos"] == b["treatment_geos"]  # reproducible
        flips = [
            geo_lift_design(path, "Sales", ch, seed=s)["treatment_geos"]
            for s in range(6)
        ]
        assert len({tuple(f) for f in flips}) > 1  # randomization actually flips

    def test_matched_market_did_is_observational_with_caveat(self, geo_csv):
        path, key = geo_csv
        d = geo_lift_design(path, "Sales", key["channels"][0], randomize=False)
        assert d["design_key"] == "matched_market_did" and not d["randomized"]
        assert "NOT randomized" in d["analysis_plan"]
        # deterministic: bigger geo treated, no seed dependence
        d2 = geo_lift_design(
            path, "Sales", key["channels"][0], randomize=False, seed=99
        )
        assert d["treatment_geos"] == d2["treatment_geos"]

    def test_holdout_uses_full_treated_spend(self, geo_csv):
        path, key = geo_csv
        ch = key["channels"][0]
        hold = geo_lift_design(path, "Sales", ch, design="holdout", seed=4)
        scale = geo_lift_design(
            path, "Sales", ch, design="scaling", intensity_pct=50, seed=4
        )
        assert hold["intensity_pct"] == -100.0
        # same assignment (same seed) -> holdout moves 2x the spend of a +50% cell
        assert hold["weekly_spend_delta"] == pytest.approx(
            2 * scale["weekly_spend_delta"], rel=1e-6
        )
        # more spend moved -> tighter ROAS measurement
        assert hold["se_roas"] < scale["se_roas"]

    def test_national_data_rejected(self, national_csv):
        path, key = national_csv
        with pytest.raises(ValueError, match="at least 4 geographies"):
            geo_lift_design(path, "Sales", key["channels"][0])


class TestMatching:
    def test_pairs_prefer_comovement_and_scale(self, geo_csv):
        path, _ = geo_csv
        from mmm_framework.planning.design import load_design_frame

        frame = load_design_frame(path, "Sales", "TV")
        pairs = matched_pairs(frame["kpi_wide"])
        # every geo used exactly once
        used = [g for p in pairs for g in (p["geo_a"], p["geo_b"])]
        assert sorted(used) == sorted(frame["geos"])
        # best pair first; both correlation flavors reported
        assert pairs[0]["distance"] <= pairs[-1]["distance"]
        assert all("residual_correlation" in p for p in pairs)

    def test_residual_matching_sees_through_shared_seasonality(self):
        """The raw-correlation trap: four geos share a HUGE seasonal cycle, so
        every raw pairwise correlation looks great. Only A and B share an
        idiosyncratic component. Model-structured matching must pair A-B (and
        hence C-D); raw-correlation matching cannot tell the pairs apart."""
        import pandas as pd

        rng = np.random.default_rng(0)
        n = 104
        t = np.arange(n)
        season = 100.0 * np.sin(2 * np.pi * t / 52.0)  # dominates everything
        shared_ab = np.cumsum(rng.normal(0, 1.0, n))  # idiosyncratic co-movement
        kpi_wide = pd.DataFrame(
            {
                "A": 500 + season + 5 * shared_ab + rng.normal(0, 2.0, n),
                "B": 510 + season + 5 * shared_ab + rng.normal(0, 2.0, n),
                "C": 505
                + season
                + 5 * np.cumsum(rng.normal(0, 1.0, n))
                + rng.normal(0, 2.0, n),
                "D": 495
                + season
                + 5 * np.cumsum(rng.normal(0, 1.0, n))
                + rng.normal(0, 2.0, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="W"),
        )
        # raw correlations are all high (seasonality dominates every pair)
        raw = kpi_wide.corr()
        off_diag = [raw.iloc[i, j] for i in range(4) for j in range(i + 1, 4)]
        assert min(off_diag) > 0.8

        pairs = matched_pairs(kpi_wide)
        pairing = {frozenset((p["geo_a"], p["geo_b"])) for p in pairs}
        assert frozenset(("A", "B")) in pairing
        ab = next(p for p in pairs if {p["geo_a"], p["geo_b"]} == {"A", "B"})
        others = [p for p in pairs if {p["geo_a"], p["geo_b"]} != {"A", "B"}]
        # the residual correlation exposes what raw correlation hides
        assert ab["residual_correlation"] > 0.8
        assert all(
            ab["residual_correlation"] > p["residual_correlation"] for p in others
        )

    def test_residualization_removes_structure(self, geo_csv):
        path, _ = geo_csv
        from mmm_framework.planning.design import (
            load_design_frame,
            residualize_geo_panel,
        )

        frame = load_design_frame(path, "Sales", "TV")
        rz = residualize_geo_panel(frame["kpi_wide"], frame["spend_wide"])
        # residuals are strictly quieter than the raw series
        for g in frame["geos"]:
            assert rz["residuals"][g].std() < frame["kpi_wide"][g].std()
        # features on the model's terms, one row per geo
        assert set(rz["features"].columns) == {
            "level",
            "trend_slope",
            "seasonal_amplitude",
            "residual_sd",
            "spend_share",
        }
        assert abs(rz["features"]["spend_share"].sum() - 1.0) < 1e-6


class TestRobustPower:
    def test_simulation_calibrated_power_and_balance(self, geo_csv):
        path, key = geo_csv
        d = geo_lift_design(path, "Sales", key["channels"][0], duration=8, seed=11)
        # 80-week pre-period gives plenty of placebo windows -> calibrated
        assert d["se_source"] == "placebo_calibrated"
        assert d["diagnostics"]["calibration_factor"] > 0
        assert d["diagnostics"]["matching"].startswith("model-structured")
        # covariate balance on the model's features
        feats = {r["feature"] for r in d["balance"]}
        assert "residual_sd" in feats and "spend_share" in feats
        assert all(np.isfinite(r["abs_std_diff"]) for r in d["balance"])
        assert d["diagnostics"]["max_balance_abs_std_diff"] == pytest.approx(
            max(r["abs_std_diff"] for r in d["balance"])
        )
        # the curve stays smooth/monotone after calibration
        ses = [
            p["se_roas"] for p in sorted(d["power_curve"], key=lambda x: x["duration"])
        ]
        assert all(a >= b for a, b in zip(ses, ses[1:]))


class TestFlighting:
    def test_budget_neutral_balanced_schedule(self, national_csv):
        path, key = national_csv
        d = flighting_design(
            path,
            "Sales",
            key["channels"][0],
            amplitude_pct=50,
            block_weeks=2,
            duration=12,
            seed=5,
        )
        mults = np.array([s["multiplier"] for s in d["schedule"]])
        assert mults.mean() == pytest.approx(1.0, abs=1e-3)  # budget neutral
        assert (mults > 1).sum() == (mults < 1).sum()  # balanced on/off
        # the schedule manufactures exogenous (randomized) spend variance
        assert 0 < d["identification"]["exogenous_share"] < 1
        small = flighting_design(
            path,
            "Sales",
            key["channels"][0],
            amplitude_pct=15,
            block_weeks=2,
            duration=12,
            seed=5,
        )
        assert (
            d["identification"]["exogenous_share"]
            > small["identification"]["exogenous_share"]
        )
        assert d["mde_roas"] == pytest.approx(2.8 * d["se_roas"])
        assert "intention-to-treat" in d["analysis_plan"]

    def test_block_order_randomized_but_seeded(self, national_csv):
        path, key = national_csv
        ch = key["channels"][0]
        a = flighting_design(path, "Sales", ch, seed=2)
        b = flighting_design(path, "Sales", ch, seed=2)
        assert a["schedule"] == b["schedule"]
        orders = {
            tuple(
                s["multiplier"]
                for s in flighting_design(path, "Sales", ch, seed=s)["schedule"]
            )
            for s in range(6)
        }
        assert len(orders) > 1

    def test_higher_amplitude_tightens_measurement(self, national_csv):
        path, key = national_csv
        ch = key["channels"][0]
        small = flighting_design(path, "Sales", ch, amplitude_pct=20, seed=3)
        big = flighting_design(path, "Sales", ch, amplitude_pct=60, seed=3)
        assert big["se_roas"] < small["se_roas"]

    def test_amplitude_validation(self, national_csv):
        path, key = national_csv
        with pytest.raises(ValueError, match="amplitude_pct"):
            flighting_design(path, "Sales", key["channels"][0], amplitude_pct=0)


class TestDispatcher:
    def test_auto_recommendation(self, geo_csv, national_csv):
        gpath, gkey = geo_csv
        npath, nkey = national_csv
        assert (
            design_experiment(gpath, "Sales", gkey["channels"][0])["design_key"]
            == "geo_lift"
        )
        assert (
            design_experiment(npath, "Sales", nkey["channels"][0])["design_key"]
            == "national_flighting"
        )

    def test_unknown_design_rejected(self, geo_csv):
        path, key = geo_csv
        with pytest.raises(ValueError, match="Unknown design"):
            design_experiment(path, "Sales", key["channels"][0], design_key="vibes")

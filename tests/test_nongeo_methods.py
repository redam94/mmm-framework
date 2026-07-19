"""Phase 2 non-geo experiment methods: ghost ads (user-level RCT power
calculator) + switchback (time-randomized design with carryover/autocorrelation-
honest power)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning import methods, simulation
from mmm_framework.planning.methods.ghost_ads import (
    GhostAdsDesign,
    ghost_ads_power,
    ghost_ads_power_at,
    ghost_ads_simulate,
    ghost_ads_users_for_mde,
)
from mmm_framework.planning.methods.switchback import switchback_power


@pytest.fixture(scope="module")
def national_csv(tmp_path_factory):
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("realistic", seed=3, n_weeks=120)
    path = tmp_path_factory.mktemp("nongeo") / "nat.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── ghost ads ────────────────────────────────────────────────────────────────


def _design(**kw) -> GhostAdsDesign:
    base = dict(users_reached=200_000, baseline_rate=0.02, treated_fraction=0.5)
    base.update(kw)
    return GhostAdsDesign(**base)


def test_ghost_ads_mde_matches_closed_form_reference():
    # Reference: two-proportion MDE with z=1.96, z_pow=0.8416 at p0=0.02,
    # n=100k/arm — the fixed point sits slightly above the null-SE seed.
    res = ghost_ads_power(_design())
    se_null = np.sqrt(2 * 0.02 * 0.98 / 100_000)
    seed_mde = (1.959963984540054 + 0.8416212335729143) * se_null
    assert res["mde_abs"] == pytest.approx(seed_mde, rel=0.05)
    assert res["mde_abs"] > seed_mde  # p1-variance makes the exact MDE larger
    assert res["mde_rel"] == pytest.approx(res["mde_abs"] / 0.02)


def test_ghost_ads_users_for_mde_inverts():
    d = _design()
    res = ghost_ads_power(d)
    n = ghost_ads_users_for_mde(d, res["mde_abs"])
    assert n == pytest.approx(d.users_reached, rel=0.01)


def test_ghost_ads_power_at_mde_hits_target():
    d = _design()
    res = ghost_ads_power(d)
    assert ghost_ads_power_at(d, res["mde_abs"]) == pytest.approx(0.80, abs=0.05)


def test_ghost_ads_simulation_validates_analytics():
    d = _design()
    res = ghost_ads_power(d)
    sim = ghost_ads_simulate(d, res["mde_abs"], n_sims=3000, seed=1)
    assert sim["empirical_power"] == pytest.approx(0.80, abs=0.06)
    assert sim["empirical_fpr"] == pytest.approx(0.05, abs=0.02)


def test_ghost_ads_rare_event_flagged_and_optimistic():
    # 3k users at 0.5% baseline → ~7 expected conversions/arm: rare-event.
    d = _design(users_reached=3_000, baseline_rate=0.005)
    res = ghost_ads_power(d)
    assert res["rare_event_regime"] is True


def test_ghost_ads_itt_tot_dilution():
    d = _design(exposure_rate=0.5)
    res = ghost_ads_power(d)
    assert res["tot_mde"] == pytest.approx(res["itt_mde"] / 0.5)


def test_ghost_ads_count_and_revenue_outcomes():
    c = ghost_ads_power(
        _design(outcome="count", baseline_mean=0.3, baseline_dispersion=1.5)
    )
    assert c["mde_abs"] > 0
    r = ghost_ads_power(_design(outcome="revenue", baseline_mean=5.0, value_sd=20.0))
    assert r["mde_abs"] > 0
    with pytest.raises(ValueError):
        _design(outcome="revenue", baseline_mean=5.0)  # missing value_sd


def test_ghost_ads_validation():
    with pytest.raises(ValueError):
        _design(users_reached=0)
    with pytest.raises(ValueError):
        _design(treated_fraction=1.0)
    with pytest.raises(ValueError):
        _design(exposure_rate=0.0)


# ── switchback ───────────────────────────────────────────────────────────────


def test_switchback_design_blocks_and_burnin(national_csv):
    from mmm_framework.planning.methods import switchback_design

    d = switchback_design(national_csv, "Sales", "TV", duration=12, carryover_weeks=3)
    assert d["method"] == "switchback"
    assert d["block_weeks"] == 3  # derived from carryover
    assert d["n_switches"] >= 1
    assert d["burn_in_weeks"] >= 1
    assert "carryover_warning" not in d


def test_switchback_short_blocks_warn(national_csv):
    from mmm_framework.planning.methods import switchback_design

    d = switchback_design(
        national_csv, "Sales", "TV", duration=12, block_weeks=1, carryover_weeks=4
    )
    assert "carryover_warning" in d


def test_switchback_design_effect_inflates_se(national_csv):
    from mmm_framework.planning.methods import switchback_design

    d = switchback_design(national_csv, "Sales", "TV", duration=12, carryover_weeks=3)
    sp = d["switchback_power"]
    assert d["ar1"]["rho"] > 0.2  # the synth world's KPI is autocorrelated
    assert sp["se_honest"] > sp["se_iid"]
    assert sp["mde_honest"] == pytest.approx(2.8 * sp["se_honest"])


def test_switchback_power_unit():
    design = {
        "schedule": [{"multiplier": m} for m in (1.5, 0.5, 1.5, 0.5, 1.5, 0.5)],
        "weekly_spend_delta": 100.0,
    }
    sp = switchback_power(design, sigma_y=50.0, design_effect=2.0)
    assert sp["se_honest"] == pytest.approx(sp["se_iid"] * np.sqrt(2.0))
    assert sp["mde_roas_honest"] == pytest.approx(sp["mde_honest"] / 100.0)


def test_switchback_via_design_experiment(national_csv):
    from mmm_framework.planning.design import design_experiment

    d = design_experiment(
        national_csv, "Sales", "TV", method="switchback", duration=12, block_weeks=2
    )
    assert d["method"] == "switchback"
    assert d["design_key"] == "national_flighting"
    assert "switchback_power" in d


# ── registry ─────────────────────────────────────────────────────────────────


def test_nongeo_methods_registered():
    assert methods.get_method("ghost_ads").requirement.family == "user"
    assert methods.get_method("switchback").requirement.family == "switchback"
    assert "switchback" in simulation._NATIONAL_ESTIMATORS


def test_methods_for_data_gates_user_family():
    rows = {r["key"]: r for r in methods.methods_for_data(n_geos=20, n_weeks=52)}
    assert rows["ghost_ads"]["supported"] is False  # no user counts
    rows = {
        r["key"]: r
        for r in methods.methods_for_data(n_geos=0, n_weeks=52, has_user_counts=True)
    }
    assert rows["ghost_ads"]["supported"] is True
    assert rows["switchback"]["supported"] is True

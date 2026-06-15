"""Tests for the experiment optimizer (planning/experiment_optimizer.py):
cool-down from adstock, Pareto non-dominated sorting + knee recommendation, and
an end-to-end suggest_experiment against a real (tiny) fit."""

from __future__ import annotations

import math

import pytest

from mmm_framework.planning import experiment_optimizer as EO


class _FakeAdstock:
    def __init__(self, alpha, half_life):
        self.alpha_mean = alpha
        self.half_life = half_life


def _patch_adstock(monkeypatch, result):
    import mmm_framework.reporting.helpers as H

    monkeypatch.setattr(H, "compute_adstock_weights", lambda mmm, channels=None: result)


def test_cooldown_from_adstock_decay(monkeypatch):
    _patch_adstock(monkeypatch, {"TV": _FakeAdstock(0.5, 1.0)})
    cd = EO.cooldown_weeks(object(), "TV")
    assert cd["basis"] == "adstock_decay"
    # ceil(log(0.05)/log(0.5)) = ceil(4.32) = 5
    assert cd["cooldown_weeks"] == 5


def test_cooldown_clips_long_memory(monkeypatch):
    _patch_adstock(monkeypatch, {"TV": _FakeAdstock(0.95, 13.5)})
    cd = EO.cooldown_weeks(object(), "TV", max_weeks=26)
    assert cd["cooldown_weeks"] == 26  # clipped


def test_cooldown_unknown_adstock_moderate_default(monkeypatch):
    _patch_adstock(monkeypatch, {})  # channel missing
    cd = EO.cooldown_weeks(object(), "TV", default_weeks=4)
    assert cd["basis"] == "default"
    assert cd["cooldown_weeks"] == 4


def test_cooldown_no_carryover(monkeypatch):
    _patch_adstock(monkeypatch, {"TV": _FakeAdstock(0.0, 0.0)})
    cd = EO.cooldown_weeks(object(), "TV")
    assert cd["basis"] == "no_carryover"
    assert cd["cooldown_weeks"] == 1


def test_cooldown_non_decaying(monkeypatch):
    _patch_adstock(
        monkeypatch, {"TV": _FakeAdstock(1.0, 0.0)}
    )  # explosive/non-decaying
    cd = EO.cooldown_weeks(object(), "TV", max_weeks=26)
    assert cd["basis"] == "non_decaying"
    assert cd["cooldown_weeks"] == 26


class _OC:
    def __init__(self, ocd, forgone, spend):
        self.opportunity_cost_dollar_median = ocd
        self.forgone_kpi_median = forgone
        self.spend_at_risk = spend


def test_tradeoff_spend_at_risk_fallback():
    """Review finding 1: a no-margin scaling-up test (forgone≈0) falls back to
    the budget at risk, so +100% no longer ties at 0 and dominates +50%."""
    # scaling-up, no margin: forgone 0 → spend_at_risk
    assert EO._tradeoff(_OC(None, 0.0, 5000.0), False) == (5000.0, "spend_at_risk")
    # holdout, no margin: forgone > 0 → forgone_kpi
    assert EO._tradeoff(_OC(None, 300.0, 2000.0), False) == (300.0, "forgone_kpi")
    # margin known → net-$ downside
    assert EO._tradeoff(_OC(123.0, 300.0, 2000.0), True) == (123.0, "net_dollar")


def test_mde_at_interpolation():
    pc = [
        {"duration": 4, "mde_roas": 0.40},
        {"duration": 8, "mde_roas": 0.30},
        {"duration": 12, "mde_roas": 0.25},
    ]
    assert EO._mde_at(pc, 8) == 0.30  # exact point
    assert EO._mde_at(pc, 6) == pytest.approx(0.35)  # interp 4↔8
    assert EO._mde_at(pc, 2) == 0.40  # clamp low
    assert EO._mde_at(pc, 20) == 0.25  # clamp high
    assert EO._mde_at([], 8) == float("inf")  # empty


def _cand(idx, mde, tradeoff, dur, *, powered=False, power_shortfall=0.0, power=None):
    return EO.CandidateEval(
        index=idx,
        design_key="geo_lift",
        mode="scaling",
        footprint="full",
        n_pairs=None,
        intensity_pct=50.0,
        duration=dur,
        mde_roas=mde,
        power_shortfall=power_shortfall,
        tradeoff=tradeoff,
        tradeoff_basis="forgone_kpi",
        power=power,
        power_target=0.8,
        forgone_kpi_median=tradeoff,
        opportunity_cost_dollar_median=None,
        net_profit_impact_median=None,
        spend_at_risk=0.0,
        pct_of_window_kpi=None,
        duration_effective=dur,
        powered=powered,
    )


def test_pareto_front_drops_dominated():
    a = _cand(0, 0.2, 100, 8)
    b = _cand(1, 0.3, 200, 12)  # dominated by a on all three objectives
    c = _cand(2, 0.1, 300, 4)  # better MDE + duration, worse tradeoff → non-dominated
    front = EO.pareto_front([a, b, c])
    assert front == [0, 2]
    assert not EO._dominates(a, c) and not EO._dominates(c, a)
    assert EO._dominates(a, b)


def test_pareto_skips_nonfinite_objectives():
    a = _cand(0, 0.2, 100, 8)
    bad = _cand(1, float("inf"), 50, 4)  # unpowered/insufficient MDE
    front = EO.pareto_front([a, bad])
    assert front == [0]


def test_recommend_prefers_powered_on_front():
    a = _cand(0, 0.2, 100, 8, powered=True)
    c = _cand(2, 0.1, 300, 4, powered=False)
    front = EO.pareto_front([a, c])
    rec = EO.recommend([a, c], front)
    assert rec == 0  # only powered front member


def test_recommend_knee_when_none_powered():
    a = _cand(0, 0.2, 100, 8)
    c = _cand(2, 0.1, 300, 4)
    front = EO.pareto_front([a, c])
    rec = EO.recommend([a, c], front)
    assert rec in front


def test_recommend_none_on_empty_front():
    assert EO.recommend([], []) is None


def test_power_for_consistent_with_mde():
    # at an effect equal to the MDE, power is ~80% by construction
    assert EO._power_for(0.28, [0.28] * 200) == pytest.approx(0.80, abs=0.02)
    # a large effect relative to the MDE → ~full power
    assert EO._power_for(0.10, [0.5] * 200) > 0.95
    # a tiny effect → power collapses toward alpha
    assert EO._power_for(0.5, [0.02] * 200) < 0.2


def test_power_shortfall_is_a_pareto_objective():
    # two designs identical on MDE/tradeoff/duration but different power: the
    # under-powered one (positive shortfall) is dominated by the powered one.
    powered = _cand(0, 0.2, 100, 8, power_shortfall=0.0, power=0.9, powered=True)
    under = _cand(1, 0.2, 100, 8, power_shortfall=0.2, power=0.6, powered=False)
    assert EO._dominates(powered, under)
    assert EO.pareto_front([powered, under]) == [0]


def test_recommend_prefers_powered_over_lower_mde():
    # an under-powered design with a better MDE is on the front, but recommend
    # picks the powered one (trying to keep power >= 80%).
    powered = _cand(0, 0.25, 100, 8, power_shortfall=0.0, power=0.85, powered=True)
    sharper_underpowered = _cand(
        1, 0.10, 90, 8, power_shortfall=0.15, power=0.65, powered=False
    )
    front = EO.pareto_front([powered, sharper_underpowered])
    assert set(front) == {0, 1}  # both non-dominated (power vs MDE trade)
    assert EO.recommend([powered, sharper_underpowered], front) == 0


def test_flighting_estimand_ses_levels():
    import numpy as np

    from mmm_framework.planning.design import flighting_estimand_ses

    # >=3 distinct levels → the tangent mROAS is identified (quadratic fit)
    es = flighting_estimand_ses(np.array([0.5, 1.0, 1.5] * 6), 100.0, 10.0)
    assert es["mroas_identified"] is True and es["n_distinct_levels"] == 3
    assert es["se_roas"] > 0 and es["se_mroas"] > 0
    # binary on/off → only a secant (curve NOT identified)
    es2 = flighting_estimand_ses(np.array([0.5, 1.5] * 8), 100.0, 10.0)
    assert es2["mroas_identified"] is False and es2["n_distinct_levels"] == 2
    # a single level can't identify anything → None
    assert flighting_estimand_ses(np.array([1.0] * 10), 100.0, 10.0) is None


def test_flighting_power_breakdown_roas_equals_contribution():
    import numpy as np

    es = {
        "se_roas": 0.1,
        "se_contribution": 5.0,
        "se_mroas": 0.2,
        "mroas_identified": True,
        "n_distinct_levels": 4,
    }
    # construct roas/contribution so the (rescaled) detection tests coincide:
    # roas/se_roas must equal contribution/se_contribution.
    contribution = np.full(200, 10.0)
    roas = np.full(200, 10.0 * (0.1 / 5.0))  # = 0.2
    mroas = np.full(200, 0.3)
    pb = EO._flighting_power_breakdown(
        es, {"roas": roas, "contribution": contribution, "mroas": mroas}, target=0.8
    )
    assert pb["roas"] == pytest.approx(pb["contribution"], abs=1e-9)
    assert pb["mroas_identified"] is True and pb["n_levels"] == 4
    assert pb["min"] == pytest.approx(min(pb["roas"], pb["mroas"]))
    # no posteriors → None (degrades gracefully)
    assert EO._flighting_power_breakdown(es, None, target=0.8) is None


def test_flighting_breakdown_gates_secant_mroas():
    """Review finding 1/10: a binary on/off's slope is a secant, not the curve —
    it must NOT earn or bind the mROAS objective; a degenerate (non-finite)
    required estimand makes the binding power unknown, not the survivor."""
    import numpy as np

    post = {
        "roas": np.full(200, 0.3),
        "contribution": np.full(200, 30.0),
        "mroas": np.full(200, 0.3),
    }
    # binary (not identified) → mroas not credited; binding = roas only
    es_bin = {
        "se_roas": 0.2,
        "se_contribution": 20.0,
        "se_mroas": 0.05,
        "mroas_identified": False,
        "n_distinct_levels": 2,
    }
    pb = EO._flighting_power_breakdown(es_bin, post, target=0.8)
    assert pb["mroas"] is None
    assert pb["min"] == pytest.approx(pb["roas"])
    # multi-level (identified) → binding = min(roas, mroas)
    es_multi = {
        "se_roas": 0.2,
        "se_contribution": 20.0,
        "se_mroas": 0.5,
        "mroas_identified": True,
        "n_distinct_levels": 4,
    }
    pb2 = EO._flighting_power_breakdown(es_multi, post, target=0.8)
    assert pb2["mroas"] is not None
    assert pb2["min"] == pytest.approx(min(pb2["roas"], pb2["mroas"]))
    # a required estimand non-finite → binding unknown (None), not the survivor
    es_bad = {
        "se_roas": float("nan"),
        "se_contribution": 20.0,
        "se_mroas": 0.1,
        "mroas_identified": True,
        "n_distinct_levels": 4,
    }
    assert EO._flighting_power_breakdown(es_bad, post, target=0.8)["min"] is None


def test_flighting_estimand_ses_rejects_near_collinear():
    import numpy as np

    from mmm_framework.planning.design import flighting_estimand_ses

    # levels within <2% of each other → near-singular → None (not garbage SEs)
    assert (
        flighting_estimand_ses(np.array([0.999, 1.0, 1.001] * 6), 100.0, 10.0) is None
    )
    # all-equal levels → None
    assert flighting_estimand_ses(np.array([1.0] * 12), 100.0, 10.0) is None
    # an x0 override is honored (operating point for the level/slope SEs)
    es = flighting_estimand_ses(np.array([0.5, 1.0, 1.5] * 6), 100.0, 10.0, x0=200.0)
    assert es is not None and es["se_roas"] > 0


@pytest.mark.slow
def test_suggest_experiment_national_flighting_power_breakdown(tmp_path):
    """National data: flighting candidates carry the 3-estimand power breakdown;
    a multi-level design identifies the marginal ROAS, a binary on/off does not;
    ROAS and contribution detection power coincide."""
    import logging

    logging.disable(logging.CRITICAL)
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import generate_mff

    df, key = generate_mff("realistic", seed=5, n_weeks=130)  # national (1 geo)
    path = str(tmp_path / "nat.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "media_channels": [{"name": n} for n in ["TV", "Search", "Social", "Display"]],
        "control_variables": [],
        "inference": {"draws": 50, "tune": 50, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    mmm, results, _ = build_and_fit(spec, path)
    assert not mmm.has_geo

    out = EO.suggest_experiment(
        mmm,
        path,
        "Sales",
        "TV",
        margin=0.5,
        duration_min=12,
        duration_max=20,
        intensity_min=-50,
        intensity_max=100,
        max_draws=50,
    )
    assert out["kind"] == "national"
    fl = [c for c in out["candidates"] if c["mode"] == "flighting"]
    assert fl and all(c["power_breakdown"] for c in fl)
    multi = [c for c in fl if c["power_breakdown"]["mroas_identified"]]
    onoff = [c for c in fl if not c["power_breakdown"]["mroas_identified"]]
    assert multi and onoff  # both a curve-identifying and an on/off design
    assert all(c["power_breakdown"]["n_levels"] >= 3 for c in multi)
    # on/off can't identify the curve → its mROAS power is gated to None (no
    # secant credit); a multi-level design reports a real mROAS power.
    assert all(c["power_breakdown"]["mroas"] is None for c in onoff)
    assert all(c["power_breakdown"]["mroas"] is not None for c in multi)
    for c in fl:
        pb = c["power_breakdown"]
        if pb["roas"] is not None and pb["contribution"] is not None:
            assert pb["roas"] == pytest.approx(pb["contribution"], abs=1e-6)


@pytest.mark.slow
def test_suggest_experiment_end_to_end(tmp_path):
    import logging

    logging.disable(logging.CRITICAL)
    from mmm_framework.agents.fitting import build_and_fit
    from mmm_framework.synth import dgp_geo
    from mmm_framework.synth.mff import geo_scenario_to_mff

    geos = ["North", "South", "East", "West", "G5", "G6", "G7", "G8"]
    sc = dgp_geo.build("geo_heterogeneous", seed=3, geos=geos, n_weeks=90)
    path = str(tmp_path / "geo.csv")
    geo_scenario_to_mff(sc).to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "kpi_level": "geo",
        "media_channels": [{"name": n} for n in ["TV", "Search", "Social", "Display"]],
        "control_variables": [],
        "inference": {"draws": 50, "tune": 50, "chains": 2, "random_seed": 0},
        "seasonality": {"yearly": 2},
        "trend": {"type": "linear"},
    }
    mmm, results, _ = build_and_fit(spec, path)

    # bound the search with custom ranges, incl. a go-dark (-100%) spend variation
    out = EO.suggest_experiment(
        mmm,
        path,
        "Sales",
        "TV",
        margin=0.5,
        duration_min=6,
        duration_max=16,
        intensity_min=-100,
        intensity_max=100,
        max_draws=50,
    )
    cands = out["candidates"]
    front = out["pareto_indices"]
    assert out["n_candidates"] > 0 and len(front) > 0
    # the explored design space reflects the requested ranges
    ds = out["design_space"]
    assert min(ds["durations"]) >= 6 and max(ds["durations"]) <= 16
    assert ds["intensity_min"] >= -100 and ds["intensity_max"] <= 100
    # a go-dark holdout candidate is present (the -100% end of the spend range)
    assert any(c["mode"] == "holdout" for c in cands)
    # and a scaling candidate too
    assert any(c["mode"] == "scaling" for c in cands)

    # the front must be non-dominated
    def objs(c):
        return (c["mde_roas"], c["tradeoff"], c["duration"])

    for i in front:
        for j in range(len(cands)):
            if j == i:
                continue
            oi, oj = objs(cands[i]), objs(cands[j])
            dominated = all(x <= y for x, y in zip(oj, oi)) and any(
                x < y for x, y in zip(oj, oi)
            )
            assert not dominated, f"front member {i} dominated by {j}"
    # recommended is on the front and carries a runnable setup + cool-down
    assert out["recommended_index"] in front
    rec = out["recommended"]
    assert rec["treatment_geos"] and rec["control_geos"]
    assert out["cooldown"]["cooldown_weeks"] >= 1
    # power is a 4th objective: every candidate carries a power in [0,1] and a
    # finite shortfall, and the target is exposed.
    assert out["power_target"] == pytest.approx(0.80)
    for c in cands:
        assert c["power"] is None or 0.0 <= c["power"] <= 1.0
        assert c["power_shortfall"] >= 0.0
    # JSON-safe across the boundary
    import json

    json.dumps(out, default=str)

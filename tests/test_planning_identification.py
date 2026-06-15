"""Tests for structural-parameter identification design (multi-level flighting).

Proves the locked-design properties: a flat schedule cannot identify adstock; >=3
spread in-support levels are required to claim the saturation curve; sharp pulses
beat one long block for adstock; the saturation clamp zeroes its Jacobian rows;
nuisance-collinear contrasts are killed by residualization; a diagonal prior
ridge keeps correlated ("banana ridge") draws well-conditioned; and the AR(1) /
cross-channel guards behave.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning import identification as ident


def _draws(rng, *, n=3000, beta=1.5, b_sd=0.3, alpha=0.5, a_sd=0.08, lam=1.0, l_sd=0.2):
    """Plausible flattened posterior draws for (beta, alpha, lam)."""
    return {
        "beta": np.clip(rng.normal(beta, b_sd, n), 1e-3, None),
        "alpha": np.clip(rng.normal(alpha, a_sd, n), 0.0, 0.95),
        "lam": np.clip(rng.normal(lam, l_sd, n), 1e-3, None),
    }


# Operating point chosen so lam*a sits in the sensitive (un-clamped) region.
OP = dict(op_spend=30.0, raw_max=120.0, y_std=1000.0)
NOISE = dict(sigma_lo=60.0, sigma_hi=110.0)


def _ident(mults, draws, *, in_support=True, **kw):
    return ident.structural_identification(
        np.asarray(mults, float),
        OP["op_spend"],
        OP["raw_max"],
        OP["y_std"],
        draws,
        in_support=in_support,
        **NOISE,
        **kw,
    )


# ── Jacobian / forward op ──────────────────────────────────────────────────────


def test_flat_schedule_identifies_nothing():
    """An all-BAU schedule has zero contrast → nothing is claimed and the
    headline binding is None (NOT the prior's resolve-from-0 probability)."""
    rng = np.random.default_rng(0)
    res = _ident([1.0] * 12, _draws(rng))
    assert res is not None
    # nothing claimed (beta included — the design moves no parameter)
    for p in ("beta", "alpha", "lam"):
        assert res["params"][p]["claimed"] is False
        assert res["params"][p]["contraction"] == pytest.approx(0.0, abs=1e-6)
    assert res["identifies_anything"] is False
    assert res["binding_power"] is None  # honesty rail: no false confidence
    assert res["binding_contraction"] is None


def test_near_flat_schedule_claims_nothing():
    """A near-flat (sub-separation) schedule provides negligible information →
    no parameter is claimed, despite a large relative alpha column."""
    rng = np.random.default_rng(12)
    res = _ident([1.0001, 0.9999] * 6, _draws(rng))
    assert res is not None
    assert res["identifies_anything"] is False
    assert res["binding_power"] is None
    assert res["params"]["alpha"]["claimed"] is False


def test_clamp_zeroes_saturated_jacobian_rows():
    """Where lam*a > 20 the in-graph gradient is 0, so those rows are zeroed."""
    jac = ident.structural_jacobian(
        np.array([1.0, 0.5, 1.5, 1.0]),
        OP["op_spend"],
        OP["raw_max"],
        OP["y_std"],
        {"beta": 1.5, "alpha": 0.5, "lam": 200.0},  # huge lam → fully saturated
    )
    assert jac is not None
    assert jac["n_clamped"] > 0
    assert np.allclose(jac["J_raw"][jac["clamp_mask"]], 0.0)


def test_normalized_basis_excites_saturation_curvature():
    """A spread of in-support levels produces a non-trivial saturation column."""
    jac = ident.structural_jacobian(
        np.array([0.5, 1.0, 1.5, 2.0] * 3),
        OP["op_spend"],
        OP["raw_max"],
        OP["y_std"],
        {"beta": 1.5, "alpha": 0.5, "lam": 1.0},
        prior_sd={"beta": 0.3, "alpha": 0.08, "lam": 0.2},
    )
    assert jac is not None
    i_lam = jac["names"].index("lam")
    assert np.linalg.norm(jac["J"][:, i_lam]) > 0.0


# ── Level count: curvature claim requires >=3 in-support levels ────────────────


def test_two_levels_do_not_claim_curvature():
    rng = np.random.default_rng(1)
    res = _ident([1.5, 0.5] * 6, _draws(rng))
    assert res is not None
    assert res["params"]["lam"]["claimed"] is False  # secant, not a curve


def test_three_levels_claim_curvature_and_more_levels_help():
    rng = np.random.default_rng(2)
    res3 = _ident([1.5, 1.0, 0.5] * 4, _draws(rng))
    res5 = _ident([1.6, 1.3, 1.0, 0.7, 0.4] * 3, _draws(rng))
    assert res3 is not None and res5 is not None
    assert res3["params"]["lam"]["claimed"] is True
    assert res5["params"]["lam"]["claimed"] is True
    # more, wider levels → at least as much curvature contraction
    assert (
        res5["params"]["lam"]["contraction"]
        >= res3["params"]["lam"]["contraction"] - 1e-6
    )


def test_in_support_gate_blocks_curvature_claim():
    """A top level beyond historical per-period support cannot claim the curve."""
    rng = np.random.default_rng(3)
    res = _ident([1.5, 1.0, 0.5] * 4, _draws(rng), in_support=False)
    assert res is not None
    assert res["params"]["lam"]["claimed"] is False


# ── Adstock: temporal contrast identifies alpha; sharp beats one long block ────


def test_sharp_pulses_beat_one_long_block_on_alpha():
    rng = np.random.default_rng(4)
    draws = _draws(rng)
    sharp = _ident([1.5, 0.5] * 6, draws)  # 12 transitions
    long_block = _ident([1.5] * 6 + [0.5] * 6, draws)  # 1 transition
    assert sharp is not None and long_block is not None
    assert sharp["params"]["alpha"]["claimed"] is True
    # sharper temporal contrast → better-determined carryover
    assert (
        sharp["params"]["alpha"]["contraction"]
        > long_block["params"]["alpha"]["contraction"]
    )


# ── Residualization profiles out the nuisance baseline ─────────────────────────


def test_trend_collinear_ramp_is_residualized_away():
    """A monotone spend ramp is collinear with the trend → little net info, vs a
    high-frequency design of comparable spend variance."""
    rng = np.random.default_rng(5)
    draws = _draws(rng)
    ramp = np.linspace(0.4, 1.6, 24)
    alt = np.array([1.6, 0.4] * 12)  # same amplitude, high frequency
    res_ramp = _ident(ramp, draws)
    res_alt = _ident(alt, draws)
    assert res_ramp is not None and res_alt is not None
    assert (
        res_ramp["params"]["beta"]["contraction"]
        < res_alt["params"]["beta"]["contraction"]
    )


# ── Diagonal prior ridge keeps a correlated ("banana") posterior stable ────────


def test_correlated_draws_do_not_blow_up():
    rng = np.random.default_rng(6)
    base = _draws(rng)
    # induce strong alpha<->lam correlation (the equifinality ridge)
    z = rng.normal(0, 1, base["alpha"].size)
    base["alpha"] = np.clip(0.5 + 0.08 * z, 0.0, 0.95)
    base["lam"] = np.clip(1.0 - 0.2 * z, 1e-3, None)  # anti-correlated
    res = _ident([0.5, 1.0, 1.5, 2.0] * 3, base)
    assert res is not None
    for p in ("beta", "alpha", "lam"):
        assert res["params"][p]["post_sd"] is not None
        assert np.isfinite(res["params"][p]["post_sd"])
    assert np.isfinite(res["condition"])


# ── A good multi-level in-support sharp schedule identifies all three ──────────


def test_good_design_claims_all_three_and_binds_on_worst():
    rng = np.random.default_rng(7)
    # 4 in-support levels, pulsed (sharp transitions) and balanced
    sched = [1.8, 0.6, 1.4, 0.2, 1.8, 0.6, 1.4, 0.2, 1.8, 0.6, 1.4, 0.2]
    res = _ident(sched, _draws(rng))
    assert res is not None
    assert res["identifies_anything"] is True
    for p in ("beta", "alpha", "lam"):
        assert res["params"][p]["claimed"] is True
        assert res["params"][p]["contraction"] > 0.0
    assert res["binding_power"] is not None
    # binding is the worst claimed power
    claimed_powers = [res["params"][p]["power"] for p in ("beta", "alpha", "lam")]
    assert res["binding_power"] == pytest.approx(min(claimed_powers), rel=1e-9)
    assert res["binding_contraction"] == pytest.approx(
        min(res["params"][p]["contraction"] for p in ("beta", "alpha", "lam")), rel=1e-9
    )
    assert res["upper_bound"] is True


def test_low_noise_strong_design_is_identified():
    """With low measurement noise a strong multi-level design contracts a
    parameter past the identified threshold (positive coverage of the gate)."""
    rng = np.random.default_rng(13)
    sched = [1.8, 0.4, 1.6, 0.2, 1.8, 0.4, 1.6, 0.2, 1.8, 0.4, 1.6, 0.2]
    res = ident.structural_identification(
        np.asarray(sched, float),
        OP["op_spend"],
        OP["raw_max"],
        OP["y_std"],
        _draws(rng),
        sigma_lo=8.0,  # low noise → real contraction
        sigma_hi=12.0,
    )
    assert res is not None
    assert res["identifies_anything"] is True
    assert any(res["params"][p]["identified"] for p in ("beta", "alpha", "lam"))


def test_power_brackets_conservative_below_optimistic():
    rng = np.random.default_rng(8)
    res = _ident([1.8, 0.6, 1.4, 0.2] * 3, _draws(rng))
    assert res is not None
    for p in ("beta", "alpha", "lam"):
        pw = res["params"][p]["power"]
        pw_opt = res["params"][p]["power_optimistic"]
        if pw is not None and pw_opt is not None:
            assert pw_opt >= pw - 1e-9  # smaller sigma → more power


# ── Degenerate inputs ──────────────────────────────────────────────────────────


def test_degenerate_inputs_return_none():
    rng = np.random.default_rng(9)
    draws = _draws(rng)
    # non-finite multiplier
    assert _ident([1.0, np.nan, 0.5], draws) is None
    # empty draws for a parameter
    bad = _draws(rng)
    bad["lam"] = np.array([])
    assert _ident([0.5, 1.0, 1.5], bad) is None
    # zero sigma
    assert (
        ident.structural_identification(
            np.array([0.5, 1.0, 1.5]),
            OP["op_spend"],
            OP["raw_max"],
            OP["y_std"],
            draws,
            sigma_lo=0.0,
            sigma_hi=0.0,
        )
        is None
    )


# ── AR(1) design effect + cross-channel VIF guards ─────────────────────────────


def test_ar1_design_effect_white_vs_correlated():
    rng = np.random.default_rng(10)
    white = rng.normal(0, 1, 200)
    d_white = ident.ar1_design_effect(white, 8)
    assert d_white["deff"] == pytest.approx(1.0, abs=0.4)

    # AR(1) with rho ~ 0.7
    n = 400
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    d_corr = ident.ar1_design_effect(x, 8)
    assert d_corr["rho"] > 0.4
    assert d_corr["deff"] > 1.5
    assert d_corr["deff"] <= 8.0  # capped at window length


def test_cross_channel_vif():
    rng = np.random.default_rng(11)
    target = rng.normal(0, 1, 40)
    other_collinear = (target + rng.normal(0, 0.1, 40))[:, None]
    other_indep = rng.normal(0, 1, (40, 2))
    assert ident.cross_channel_vif(target, other_collinear)["r2"] > 0.8
    assert ident.cross_channel_vif(target, other_indep)["r2"] < 0.4
    assert ident.cross_channel_vif(target, None)["vif"] == 1.0

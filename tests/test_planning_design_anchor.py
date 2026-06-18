"""Design-anchor tests (planning/design_anchor.py): the powered-to-detect
verdict, signed assurance (ANCHOR-3), and the realized sigma_exp clamp
(ANCHOR-4). Pure math — no model needed for these."""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning.design_anchor import (
    powered_to_detect,
    realized_sigma_exp_for_anchor,
)

_POWER_CURVE = [
    {"duration": d, "mde_roas": 0.28 * (8.0 / d) ** 0.5} for d in (4, 8, 12, 16)
]


def _effect(draws, median=None, hdi=None):
    draws = np.asarray(draws, float)
    median = float(np.median(draws)) if median is None else median
    hdi = hdi or [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]
    return {
        "incremental_roas_draws": list(draws),
        "incremental_roas_median": median,
        "incremental_roas_hdi": hdi,
    }


def test_powered_when_effect_well_above_mde():
    eff = _effect(np.full(100, 0.6))  # well above mde=0.28
    v = powered_to_detect(eff, _POWER_CURVE, duration=8, se_roas=0.1)
    assert v["verdict"] in ("powered", "overpowered")
    assert v["prob_detectable"] == pytest.approx(1.0)
    assert v["assurance"] > 0.8


def test_underpowered_when_effect_below_mde():
    eff = _effect(np.full(100, 0.1))  # below mde=0.28
    v = powered_to_detect(eff, _POWER_CURVE, duration=8, se_roas=0.1)
    assert v["verdict"] == "underpowered"
    assert v["prob_detectable"] < 0.5


def test_inconclusive_when_hdi_straddles_zero():
    rng = np.random.default_rng(0)
    eff = _effect(rng.normal(0.0, 0.5, 400))  # symmetric around 0
    v = powered_to_detect(eff, _POWER_CURVE, duration=8, se_roas=0.1)
    assert v["verdict"] == "inconclusive"


def test_assurance_on_tight_null_is_near_alpha():
    """Signed two-sided assurance: a posterior concentrated at 0 (a true null
    channel) scores ~alpha = 2*Phi(-z) ≈ 0.05 (ANCHOR-3) — NOT 0.025 as the
    one-sided |eff|-folded form would give at the null point."""
    rng = np.random.default_rng(1)
    eff = _effect(rng.normal(0.0, 0.01, 2000))  # tight null
    v = powered_to_detect(eff, _POWER_CURVE, duration=8, se_roas=0.1)
    assert v["assurance"] == pytest.approx(0.05, abs=0.02)


def test_recommended_duration_from_power_curve():
    eff = _effect(np.full(100, 0.20))  # median 0.20
    v = powered_to_detect(eff, _POWER_CURVE, duration=8, se_roas=0.1)
    # mde(d) <= 0.20 first occurs at d=16 (mde=0.198)
    assert v["recommended_duration"] == 16


def test_realized_sigma_exp_clamps_zero_se():
    draws = np.full(50, 0.5)
    sig, ret = realized_sigma_exp_for_anchor(draws, 0.0)
    assert sig > 0  # flat-placebo se -> clamped above 0
    assert np.array_equal(ret, draws)


def test_realized_sigma_exp_clamps_exploded_se():
    draws = np.full(50, 0.5)
    sig, _ = realized_sigma_exp_for_anchor(draws, 1e6)
    assert sig <= 50.0 * max(0.5, 0.05)  # clamped below rel_upper * scale


def test_realized_sigma_exp_passes_through_reasonable_se():
    draws = np.full(50, 0.5)
    sig, _ = realized_sigma_exp_for_anchor(draws, 0.12)
    assert sig == pytest.approx(0.12, abs=1e-6)
